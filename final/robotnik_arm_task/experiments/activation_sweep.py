#!/usr/bin/env python3
"""Compare activation functions for the base and arm BC policies.

One command does everything:
  1. Split the full expert dataset into base and arm subsets (same inputs and
     residual arm torques as split_dataset.py)
  2. Train the same MLP once per activation function (same data, same settings, same seed)
  3. Write a Word report comparing how each activation function performed

Example:
  (run from final/robotnik_arm_task)
  python experiments/activation_sweep.py --data data/dataset_20261027_143306.npz --epochs 30
  python experiments/activation_sweep.py --data data/dataset_20261027_143306.npz --activations relu gelu tanh

The report and plots are saved to
  experiments/charts/activation_functions_comparison_<date_time>/

Only the activation function changes between runs, so any difference in the
results comes from the activation function.
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # save plots to file, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from docx import Document
from docx.shared import Inches
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")
REPORT_NAME = "activation_functions_comparison"

# Ensure project dir is on sys.path so we can import project modules
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Feature layout comes straight from split_dataset.py, so the sweep always
# trains on exactly the same inputs as the real policies.
from split_dataset import (BASE_INPUT_INDICES, ARM_INPUT_INDICES,
                           WHEEL_RADIUS, WHEEL_BASE_WIDTH, WHEEL_THRESHOLD)


# ---------------------------------------------------------------------------
# Dataset split (same steps as split_dataset.py)
# ---------------------------------------------------------------------------

# Full 44-dim input layout:
#  0,1       base_pos_x, base_pos_y
#  2         base_bearing
#  3,4,5     base_lin_vel_x/y/z
#  6,7,8     base_ang_vel_x/y/z
#  9..12     q_base (wheel angles)
#  13..16    qd_base (wheel velocities)
#  17..23    q_arm
#  24..30    qd_arm
#  31,32,33  ee_pos_x/y/z
#  34,35,36  target_x/y/z
#  37..43    tau_gravity_1..7

OUTPUT_NAMES = {
    "base": ["v_cmd (m/s)", "w_cmd (rad/s)"],
    "arm": [f"residual_torque_{i} (Nm)" for i in range(1, 8)],
}


def split_dataset(inputs, labels, metadata):
    """Split the full dataset into base-phase and arm-phase subsets.

    Returns two dicts with keys inputs / labels / metadata.
    """
    # Phase detection: wheels moving = base phase, wheels stopped = arm phase
    wheel_norms = np.linalg.norm(labels[:, :4], axis=1)
    base_mask = wheel_norms > WHEEL_THRESHOLD
    arm_mask = wheel_norms < WHEEL_THRESHOLD
    print(f"Base phase: {base_mask.sum()} samples ({100*base_mask.mean():.1f}%)")
    print(f"Arm phase:  {arm_mask.sum()} samples ({100*arm_mask.mean():.1f}%)")

    # Base: raw features + distance and heading error -> 2 outputs (v_cmd, w_cmd)
    base_raw = inputs[base_mask][:, BASE_INPUT_INDICES]
    base_pos_xy = inputs[base_mask][:, :2]
    bearing = inputs[base_mask][:, 2]
    target_xy = inputs[base_mask][:, 34:36]
    delta = target_xy - base_pos_xy
    dist_to_target = np.linalg.norm(delta, axis=1, keepdims=True)
    heading_to_target = np.arctan2(delta[:, 1], delta[:, 0])
    heading_error = np.arctan2(np.sin(heading_to_target - bearing),
                               np.cos(heading_to_target - bearing))[:, None]
    base_inputs = np.hstack([base_raw, dist_to_target, heading_error])

    # Convert wheel cmds [rw, lw, lw, rw] -> [v_cmd, w_cmd]
    wheel_raw = labels[base_mask][:, :4]
    rw = wheel_raw[:, 0]
    lw = wheel_raw[:, 1]
    v_cmd = (rw + lw) / 2.0 * WHEEL_RADIUS
    w_cmd = (rw - lw) / WHEEL_BASE_WIDTH * WHEEL_RADIUS
    base_labels = np.column_stack([v_cmd, w_cmd])

    # Arm: joint state, end-effector and target -> 7 residual torques
    # (expert torque minus gravity compensation, which the sim adds back)
    arm_inputs = inputs[arm_mask][:, ARM_INPUT_INDICES]
    arm_torques = labels[arm_mask][:, 4:]
    tau_gravity = inputs[arm_mask][:, 37:44]
    arm_labels = arm_torques - tau_gravity

    base = {"inputs": base_inputs, "labels": base_labels, "metadata": metadata[base_mask]}
    arm = {"inputs": arm_inputs, "labels": arm_labels, "metadata": metadata[arm_mask]}
    return base, arm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Same activation list and MLP builder as the real training code
from activations import ACTIVATIONS  # noqa: E402
from regressors import build_mlp  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Training (same recipe as train_regressor.py)
# ---------------------------------------------------------------------------

def prepare_data(subset, args, device):
    """Split by trajectory (80/20, seed 42) and normalise inputs and labels."""
    inputs, labels, metadata = subset["inputs"], subset["labels"], subset["metadata"]

    # Same trajectory split as train_regressor.py
    traj_ids = np.unique(metadata[:, 0])
    rng = np.random.RandomState(42)
    rng.shuffle(traj_ids)
    n_train_traj = int(len(traj_ids) * 0.8)
    train_mask = np.isin(metadata[:, 0], traj_ids[:n_train_traj])

    X_train, y_train = inputs[train_mask], labels[train_mask]
    X_val, y_val = inputs[~train_mask], labels[~train_mask]

    # Normalise with training-set statistics
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-8

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    train_ds = TensorDataset(to_tensor((X_train - x_mean) / x_std), to_tensor((y_train - y_mean) / y_std))
    val_ds = TensorDataset(to_tensor((X_val - x_mean) / x_std), to_tensor((y_val - y_mean) / y_std))

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=(device.type != "cpu"), persistent_workers=(args.num_workers > 0))
    return {
        "train_loader": DataLoader(train_ds, shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(val_ds, **loader_kwargs),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "y_val": y_val,             # in real units, for the final metrics
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": inputs.shape[1],
        "output_dim": labels.shape[1],
    }


def predict(model, loader, device):
    """Run the model over a loader and return (predictions, mean MSE) in normalised units."""
    criterion = nn.MSELoss(reduction="sum")
    model.eval()
    preds, total_se, n = [], 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_se += criterion(pred, yb).item()
            n += yb.numel()
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds), total_se / n


def train_one(activation, data, args, device):
    """Train one model with the given activation and return its results."""
    torch.manual_seed(42)  # same starting weights recipe for every activation
    model = build_mlp(data["input_dim"], data["output_dim"],
                      args.hidden_dim, args.num_layers, activation).to(device)
    # Report recipe: AdamW (no weight decay), MAE training loss, grad clip 1.0. Validation stays MSE.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    criterion = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train": [], "val": []}
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in data["train_loader"]:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= data["n_train"]

        # Validate
        _, val_loss = predict(model, data["val_loader"], device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Keep the best weights
        tag = ""
        if val_loss < best_val - args.early_stop:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            tag = "  *best*"
        print(f"  [{activation}] epoch {epoch:3d}/{args.epochs}  train={train_loss:.6f}  val={val_loss:.6f}{tag}")

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"  [{activation}] early stop: no improvement > {args.early_stop} for {args.patience} epochs")
            break

    train_time = time.time() - t_start

    # Final metrics with the best weights, in real units
    model.load_state_dict(best_state)
    pred_norm, _ = predict(model, data["val_loader"], device)
    pred = pred_norm * data["y_std"] + data["y_mean"]
    true = data["y_val"]
    err = pred - true

    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    r2 = 1 - (err ** 2).sum(axis=0) / ((true - true.mean(axis=0)) ** 2).sum(axis=0)

    return {
        "activation": activation,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history["val"]),
        "train_time_min": train_time / 60,
        "params": sum(p.numel() for p in model.parameters()),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Word report
# ---------------------------------------------------------------------------

def add_table(doc, header, rows):
    """Add a simple table with a header row."""
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = str(text)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = str(text)
    doc.add_paragraph()


def plot_val_curves(results, kind, path):
    """Validation loss per epoch, one line per activation."""
    plt.figure(figsize=(7, 4))
    for r in results:
        plt.plot(range(1, r["epochs_run"] + 1), r["history"]["val"], marker="o", markersize=3,
                 label=r["activation"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE (normalised, log scale)")
    plt.title(f"{kind.capitalize()} model: validation loss by activation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(all_results, settings, out_dir):
    doc = Document()
    doc.add_heading("Activation Function Comparison", level=0)
    doc.add_paragraph(f"Generated {time.strftime('%Y-%m-%d %H:%M')}. "
                      "Every model uses the same data split, settings and random seed; "
                      "only the hidden-layer activation function changes.")

    # Settings / flags
    doc.add_heading("Settings", level=1)
    add_table(doc, ["Setting", "Value"], settings)

    # One section per model type
    for kind, results in all_results.items():
        doc.add_heading(f"{kind.capitalize()} model", level=1)
        names = OUTPUT_NAMES[kind]
        ranked = sorted(results, key=lambda r: r["best_val_loss"])

        doc.add_heading("Summary (sorted best to worst)", level=2)
        add_table(doc,
                  ["Rank", "Activation", "Best val MSE", "Best epoch", "Epochs run",
                   "Mean MAE", "Mean RMSE", "Mean R²", "Time (min)"],
                  [[i + 1, r["activation"], f"{r['best_val_loss']:.6f}", r["best_epoch"], r["epochs_run"],
                    f"{r['mae'].mean():.4f}", f"{r['rmse'].mean():.4f}", f"{r['r2'].mean():.4f}",
                    f"{r['train_time_min']:.1f}"]
                   for i, r in enumerate(ranked)])

        doc.add_heading("MAE per output (lower is better)", level=2)
        add_table(doc, ["Activation"] + names,
                  [[r["activation"]] + [f"{v:.4f}" for v in r["mae"]] for r in ranked])

        doc.add_heading("R² per output (closer to 1 is better)", level=2)
        add_table(doc, ["Activation"] + names,
                  [[r["activation"]] + [f"{v:.4f}" for v in r["r2"]] for r in ranked])

        plot_path = os.path.join(out_dir, f"{REPORT_NAME}_{kind}.png")
        plot_val_curves(results, kind, plot_path)
        doc.add_heading("Validation loss per epoch", level=2)
        doc.add_picture(plot_path, width=Inches(6))

        best = ranked[0]
        doc.add_paragraph(f"Best activation for the {kind} model: {best['activation']} "
                          f"(val MSE {best['best_val_loss']:.6f}, mean R² {best['r2'].mean():.4f}).")

    doc.add_heading("How to read this", level=1)
    doc.add_paragraph("Best val MSE: error on the held-out 20% of trajectories, in normalised units "
                      "(1.0 = no better than predicting the average). Lower is better.")
    doc.add_paragraph("MAE / RMSE: average and root-mean-square error in real units "
                      "(m/s, rad/s or Nm). RMSE punishes large mistakes more.")
    doc.add_paragraph("R²: fraction of the expert's behaviour the model explains (1 = perfect, 0 = useless).")

    path = os.path.join(out_dir, f"{REPORT_NAME}.docx")
    doc.save(path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Split the dataset and compare activation functions")
    parser.add_argument("--data", type=str, required=True, help="Path to full .npz dataset")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to save the report "
                             "(default: experiments/charts/activation_functions_comparison_<date_time>)")
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATIONS), choices=list(ACTIVATIONS),
                        help="Activation functions to test (default: all)")
    parser.add_argument("--models", nargs="+", default=["base", "arm"], choices=["base", "arm"],
                        help="Which models to train (default: base arm)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer size (default: 512)")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of hidden layers (default: 3)")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size (default: 2048)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal GPU")
    parser.add_argument("--early_stop", type=float, default=0.0,
                        help="Val loss improvement threshold (default: 0)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Epochs without improvement before early stopping (default: 20)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Randomly subsample to this many rows before splitting (default: use all)")
    args = parser.parse_args()

    if args.mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    out_dir = args.out_dir or os.path.join(CHARTS_DIR, f"{REPORT_NAME}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load and split
    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")
    total_rows = inputs.shape[0]

    if args.max_samples and inputs.shape[0] > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(inputs.shape[0], args.max_samples, replace=False)
        idx.sort()
        inputs, labels, metadata = inputs[idx], labels[idx], metadata[idx]
        print(f"Subsampled to {inputs.shape[0]} samples")

    subsets = dict(zip(["base", "arm"], split_dataset(inputs, labels, metadata)))
    del inputs, labels, metadata, data

    # 2. Train every activation on every selected model
    all_results = {}
    data_rows = []
    for kind in args.models:
        print(f"\n{'='*60}\nPreparing {kind} data\n{'='*60}")
        prepared = prepare_data(subsets[kind], args, device)
        data_rows.append([f"{kind} train / val rows", f"{prepared['n_train']} / {prepared['n_val']}"])
        data_rows.append([f"{kind} inputs -> outputs", f"{prepared['input_dim']} -> {prepared['output_dim']}"])

        all_results[kind] = []
        for activation in args.activations:
            print(f"\n--- {kind} model, activation = {activation} ---")
            all_results[kind].append(train_one(activation, prepared, args, device))
        del prepared

    # 3. Write the report
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    settings = [
        ["Command", "python " + " ".join(sys.argv)],
        ["Dataset", args.data],
        ["Rows in dataset", total_rows],
        ["--max_samples", args.max_samples or "all"],
        *data_rows,
        ["Activations tested", ", ".join(args.activations)],
        ["--epochs (max)", args.epochs],
        ["--hidden_dim", args.hidden_dim],
        ["--num_layers", args.num_layers],
        ["--batch_size", args.batch_size],
        ["--lr", args.lr],
        ["--early_stop", args.early_stop],
        ["--patience", args.patience],
        ["--num_workers", args.num_workers],
        ["Optimizer / loss", "AdamW (wd 0) / MAE train, MSE val, grad clip 1.0"],
        ["Train/val split", "80/20 by trajectory, seed 42"],
        ["Random seed (weights)", 42],
        ["Device", gpu],
    ]
    path = write_report(all_results, settings, out_dir)
    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
