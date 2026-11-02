#!/usr/bin/env python3
"""Train a behaviour-cloning regressor on expert trajectory data.

Data loading, the train/val split, normalisation and checkpoint saving live
here; the training loop itself is delegated to the regressor in regressors.py.
"""

import argparse
import glob
import os
import re
import time as _time

import numpy as np
import torch

from activations import ACTIVATIONS, DEFAULT_ACTIVATION
from regressors import build_mlp, get_regressor, available_regressors  # noqa: F401  (build_mlp re-exported)


def split_by_trajectory(inputs, labels, metadata, val_frac=0.2, seed=42):
    """Split into train/val by trajectory ID to avoid leakage."""
    traj_ids = np.unique(metadata[:, 0])
    rng = np.random.RandomState(seed)
    rng.shuffle(traj_ids)
    split = int(len(traj_ids) * (1 - val_frac))
    train_ids = set(traj_ids[:split].tolist())

    train_mask = np.array([m in train_ids for m in metadata[:, 0]])
    val_mask = ~train_mask

    return (inputs[train_mask], labels[train_mask],
            inputs[val_mask], labels[val_mask])


def main():
    parser = argparse.ArgumentParser(description="Train BC regressor")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--model", type=str, default="mlp",
                        choices=available_regressors(),
                        help=f"Model type (default: mlp, available: {available_regressors()})")
    # Defaults follow the selected settings of the 618-run hyperparameter study
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS),
                        help=f"Hidden-layer activation function (default: {DEFAULT_ACTIVATION})")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay (default: 0)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm, 0 to disable (default: 1.0)")
    parser.add_argument("--wheel_weight", type=float, default=0.5,
                        help="Loss weight of wheel outputs vs arm outputs in full mode (default: 0.5)")
    parser.add_argument("--out", type=str, default=None,
                        help="Checkpoint save path (default: auto-generated from params)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal (MPS) GPU if available")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes for parallel data loading (default: 0)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "base", "arm"],
                        help="Metric display mode: full (4+7 breakdown), base (wheel only), arm (torque only)")
    parser.add_argument("--early_stop", type=float, default=0.00001,
                        help="Stop if val loss improvement is below this for --patience epochs (default: 0)")
    parser.add_argument("--patience", type=int, default=40,
                        help="Number of epochs without improvement before early stopping (default: 20)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Randomly subsample to this many rows before splitting (default: use all)")
    args = parser.parse_args()

    # --- Load data ---
    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    feature_names = list(data["feature_names"]) if "feature_names" in data else None
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")
    if feature_names:
        print(f"Features: {feature_names}")

    if args.max_samples and inputs.shape[0] > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(inputs.shape[0], args.max_samples, replace=False)
        idx.sort()
        inputs, labels, metadata = inputs[idx], labels[idx], metadata[idx]
        print(f"Subsampled to {inputs.shape[0]} samples")

    ACTION_DIM = labels.shape[1]

    # --- Split by trajectory ---
    X_train, y_train, X_val, y_val = split_by_trajectory(inputs, labels, metadata)
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    # --- Normalisation (input + output) ---
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std

    label_mean = y_train.mean(axis=0)
    label_std = y_train.std(axis=0) + 1e-8
    y_train_norm = (y_train - label_mean) / label_std
    y_val_norm = (y_val - label_mean) / label_std
    print(f"Label stats: mean={np.round(label_mean, 4)}  std={np.round(label_std, 4)}")

    # --- Device ---
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = torch.device("cpu")

    y_val_orig = y_val  # already in original units (pre-normalization)

    # --- Label slices depend on mode ---
    mode = args.mode
    if mode == "full":
        wheel_cols = np.arange(4)
        n_torque = 7
        torque_offset = 4
    elif mode == "base":
        wheel_cols = np.arange(ACTION_DIM)
        n_torque = 0
        torque_offset = ACTION_DIM
    elif mode == "arm":
        wheel_cols = np.array([], dtype=int)
        n_torque = ACTION_DIM
        torque_offset = 0
    arm_names = [f"arm_{i}" for i in range(1, n_torque + 1)]

    # Per-output loss weights: wheel outputs scaled by --wheel_weight (only matters in full mode)
    loss_weights = np.ones(labels.shape[1])
    if mode == "full":
        loss_weights[wheel_cols] = args.wheel_weight

    # --- Checkpoint path (fixed once at start) ---
    if args.out:
        save_path = args.out
    else:
        # Auto-increment counter from existing .pt files in cwd
        existing = glob.glob("*.pt")
        counters = [int(m.group(1)) for f in existing
                    if (m := re.match(r"^(\d+)_", f))]
        next_id = max(counters, default=0) + 1
        save_path = (f"{next_id:03d}_bc_{mode}_{args.activation}_e{args.epochs}_h{args.hidden_dim}"
                     f"_l{args.num_layers}.pt")
    print(f"Checkpoint: {save_path}")

    # --- Per-epoch reporting callback (for models that support on_epoch) ---
    t_start = _time.time()

    def on_epoch(epoch, train_loss, val_loss, val_pred, improved, model):
        # Un-scale to original units
        all_pred_orig = val_pred * label_std + label_mean
        all_label_orig = y_val_orig
        n_val = len(all_label_orig)
        err_orig = all_pred_orig - all_label_orig  # signed error in original scale

        # --- Per-component MSE ---
        val_se_wheel = 0.0
        val_se_torque = np.zeros(max(n_torque, 1))
        if len(wheel_cols) > 0:
            val_se_wheel = (err_orig[:, wheel_cols] ** 2).sum()
        for j in range(n_torque):
            val_se_torque[j] = (err_orig[:, torque_offset + j] ** 2).sum()
        val_mse_wheel = val_se_wheel / n_val if len(wheel_cols) > 0 else 0.0
        val_mse_torque = val_se_torque[:n_torque] / n_val if n_torque > 0 else np.array([])
        val_mse_torque_avg = val_mse_torque.mean() if n_torque > 0 else 0.0

        # --- Additional metrics ---
        abs_err_a0 = np.abs(err_orig)

        mae_per_dim = abs_err_a0.mean(axis=0)
        max_err_per_dim = abs_err_a0.max(axis=0)
        p90_per_dim = np.percentile(abs_err_a0, 90, axis=0)

        # Sign accuracy: % of samples where pred and label have the same sign
        nonzero_mask = np.abs(all_label_orig) > 1e-6
        sign_match = np.sign(all_pred_orig) == np.sign(all_label_orig)
        sign_acc_per_dim = np.array([
            sign_match[nonzero_mask[:, d], d].mean() * 100 if nonzero_mask[:, d].any() else 100.0
            for d in range(ACTION_DIM)
        ])

        # Near-zero accuracy: when expert outputs ~0, does model also output ~0?
        near_zero_thresh = 0.02
        expert_near_zero = np.abs(all_label_orig) < near_zero_thresh
        pred_near_zero = np.abs(all_pred_orig) < near_zero_thresh
        zero_acc_per_dim = np.array([
            pred_near_zero[expert_near_zero[:, d], d].mean() * 100 if expert_near_zero[:, d].any() else 100.0
            for d in range(ACTION_DIM)
        ])

        tag = " *best*" if improved else ""

        # Summary line
        if mode == "full":
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_mae={train_loss:.6f}  val_mse={val_loss:.6f}  "
                  f"wheel={val_mse_wheel:.6f}  torque_avg={val_mse_torque_avg:.6f}{tag}")
            arm_str = "  ".join(f"{arm_names[j]}={val_mse_torque[j]:.6f}" for j in range(n_torque))
            print(f"         torque: {arm_str}")
        elif mode == "base":
            dim_names = ["v_cmd", "w_cmd"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_mae={train_loss:.6f}  val_mse={val_loss:.6f}  "
                  f"wheel={val_mse_wheel:.6f}{tag}")
        elif mode == "arm":
            dim_names = [f"arm_{i}" for i in range(1, ACTION_DIM + 1)]
            arm_str = "  ".join(f"{arm_names[j]}={val_mse_torque[j]:.6f}" for j in range(n_torque))
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_mae={train_loss:.6f}  val_mse={val_loss:.6f}  "
                  f"torque_avg={val_mse_torque_avg:.6f}{tag}")
            print(f"         torque: {arm_str}")

        # Detailed metrics (every 5 epochs or on save)
        if epoch % 5 == 0 or improved:
            if mode in ("base", "arm"):
                names = dim_names
            else:
                names = [f"w{i}" for i in range(1, 5)] + [f"a{i}" for i in range(1, 8)]
            mae_s = "  ".join(f"{names[d]}={mae_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            max_s = "  ".join(f"{names[d]}={max_err_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            p90_s = "  ".join(f"{names[d]}={p90_per_dim[d]:.4f}" for d in range(ACTION_DIM))
            sign_s = "  ".join(f"{names[d]}={sign_acc_per_dim[d]:.1f}%" for d in range(ACTION_DIM))
            zero_s = "  ".join(f"{names[d]}={zero_acc_per_dim[d]:.1f}%" for d in range(ACTION_DIM))
            print(f"         MAE:      {mae_s}")
            print(f"         max_err:  {max_s}")
            print(f"         p90:      {p90_s}")
            print(f"         sign_acc: {sign_s}")
            print(f"         zero_acc: {zero_s}")

    # --- Training (shared trainer in regressors.py) ---
    regressor = get_regressor(args.model)()
    metrics = regressor.fit(
        X_train_norm, y_train_norm, X_val_norm, y_val_norm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        loss_weights=loss_weights,
        early_stop=args.early_stop,
        patience=args.patience,
        num_workers=args.num_workers,
        device=device,
        on_epoch=on_epoch,
    )

    # Print param count from the actual trained model
    if regressor.model is not None and hasattr(regressor.model, 'parameters'):
        n_params = sum(p.numel() for p in regressor.model.parameters())
        print(f"Model: {args.model}  {n_params} params")

    # --- Save checkpoint via regressor interface (model-agnostic) ---
    regressor.save_checkpoint(save_path,
                              input_mean=mean,
                              input_std=std,
                              label_mean=label_mean,
                              label_std=label_std,
                              epochs=args.epochs,
                              batch_size=args.batch_size,
                              lr=args.lr,
                              action_dim=ACTION_DIM,
                              feature_names=feature_names)

    total_time = _time.time() - t_start
    total_min = int(total_time // 60)
    final_path = save_path.replace(".pt", f"_{total_min}min.pt")
    if os.path.exists(save_path):
        os.rename(save_path, final_path)
    print(f"\nBest val loss: {metrics['val_mse']:.6f} at epoch {metrics['best_epoch']}")
    print(f"Total time: {total_time/60:.1f}min  Checkpoint: {final_path}")


if __name__ == "__main__":
    main()
