#!/usr/bin/env python3
"""Train a target-selector MLP on expert shortest-path data.

The model takes a fixed-size scene observation (all remaining candidates packed
into MAX_TARGETS slots with a validity mask, 20-dim total) and outputs one
score per slot (MAX_TARGETS outputs).  Training uses masked cross-entropy:
softmax is computed only over valid (mask=1) slots, and the label is the
slot the expert chose.

Usage:
    python train_selector.py --data data/dataset_selector_*.npz --epochs 100
"""
import argparse
import glob
import os
import re
import time as _time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from activations import ACTIVATIONS, DEFAULT_ACTIVATION
from regressors import build_mlp
from robotnik_arm_task import MAX_TARGETS


def masked_cross_entropy(logits, labels, masks):
    """Cross-entropy over valid slots only.

    Args:
        logits: (B, MAX_TARGETS) raw scores
        labels: (B,) int index of expert choice within valid slots
        masks: (B, MAX_TARGETS) 1.0 for valid, 0.0 for padding
    Returns:
        loss: scalar (mean over batch)
        correct: int
        total: int
    """
    # Set invalid slots to -inf so they get zero probability
    logits = logits.clone()
    logits[masks == 0] = -1e9

    loss = F.cross_entropy(logits, labels, reduction="mean")
    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    total = len(labels)
    return loss, correct, total


def split_by_episode(metadata, val_frac=0.2, seed=42):
    """Return train/val boolean masks splitting by episode ID."""
    episode_ids = np.unique(metadata[:, 0])
    rng = np.random.RandomState(seed)
    rng.shuffle(episode_ids)
    split = int(len(episode_ids) * (1 - val_frac))
    train_eps = set(episode_ids[:split].tolist())
    train_mask = np.array([m in train_eps for m in metadata[:, 0]])
    return train_mask, ~train_mask


def main():
    parser = argparse.ArgumentParser(description="Train target-selector (scene-level)")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset_selector_*.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS))
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out", type=str, default=None, help="Checkpoint save path")
    parser.add_argument("--mps", action="store_true")
    args = parser.parse_args()

    # Load data
    data = np.load(args.data)
    inputs = data["inputs"]       # (N, SELECTOR_INPUT_DIM=20)
    labels = data["labels"]       # (N,) int — index in packed slot order
    metadata = data["metadata"]   # (N, 2)  episode_id, decision_id

    print(f"Loaded {inputs.shape[0]} decisions, input_dim={inputs.shape[1]}")

    # Extract masks from inputs (last MAX_TARGETS columns)
    masks = inputs[:, -MAX_TARGETS:]  # (N, 5)
    obs = inputs[:, :-MAX_TARGETS]    # (N, 15)

    # Split by episode
    train_mask, val_mask = split_by_episode(metadata)
    X_train, lbl_train, m_train = inputs[train_mask], labels[train_mask], masks[train_mask]
    X_val, lbl_val, m_val = inputs[val_mask], labels[val_mask], masks[val_mask]

    print(f"Train: {X_train.shape[0]} decisions")
    print(f"Val:   {X_val.shape[0]} decisions")

    # Normalize inputs (obs part only, not mask)
    obs_train = X_train[:, :-MAX_TARGETS]
    mean = obs_train.mean(axis=0)
    std = obs_train.std(axis=0) + 1e-8
    X_train_norm = np.hstack([(X_train[:, :-MAX_TARGETS] - mean) / std, m_train])
    X_val_norm = np.hstack([(X_val[:, :-MAX_TARGETS] - mean) / std, m_val])

    # Device
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Build model: input_dim auto-sized from data (20 for basic, 35 for multi-drop-zone)
    input_dim = inputs.shape[1]
    output_dim = MAX_TARGETS
    model = build_mlp(input_dim, output_dim, args.hidden_dim, args.num_layers, args.activation)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Tensors
    t_X_train = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    t_lbl_train = torch.tensor(lbl_train, dtype=torch.long).to(device)
    t_m_train = torch.tensor(m_train, dtype=torch.float32).to(device)

    t_X_val = torch.tensor(X_val_norm, dtype=torch.float32).to(device)
    t_lbl_val = torch.tensor(lbl_val, dtype=torch.long).to(device)
    t_m_val = torch.tensor(m_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    best_val_acc = 0.0
    epochs_no_improve = 0
    t_start = _time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            # --- Train ---
            model.train()
            optimizer.zero_grad()
            logits = model(t_X_train)  # (N, 5)
            train_loss, train_correct, train_total = masked_cross_entropy(
                logits, t_lbl_train, t_m_train)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_acc = 100.0 * train_correct / max(train_total, 1)

            # --- Validate ---
            model.eval()
            with torch.no_grad():
                val_logits = model(t_X_val)
                val_loss, val_correct, val_total = masked_cross_entropy(
                    val_logits, t_lbl_val, t_m_val)
            val_loss_f = val_loss.item()
            val_acc = 100.0 * val_correct / max(val_total, 1)

            improved = val_loss_f < best_val_loss
            if improved:
                best_val_loss = val_loss_f
                best_epoch = epoch
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            tag = " *best*" if improved else ""
            if epoch % max(1, args.epochs // 10) == 0 or epoch == 1 or improved:
                print(f"Epoch {epoch:4d}/{args.epochs}  "
                      f"train_loss={train_loss.item():.4f}  train_acc={train_acc:.1f}%  "
                      f"val_loss={val_loss_f:.4f}  val_acc={val_acc:.1f}%{tag}")

            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs")
                break
    except KeyboardInterrupt:
        print(f"\nInterrupted at epoch {epoch}. Saving best checkpoint (epoch {best_epoch})...")

    elapsed = _time.time() - t_start

    if best_state:
        model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}  val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.1f}%")

    # Save checkpoint
    if args.out:
        save_path = args.out
    else:
        existing = glob.glob("*.pt")
        counters = [int(m.group(1)) for f in existing if (m := re.match(r"^(\d+)_", f))]
        next_id = max(counters, default=0) + 1
        save_path = f"{next_id:03d}_selector_{args.activation}_e{args.epochs}_h{args.hidden_dim}_l{args.num_layers}.pt"

    # Save mean/std for obs part only (not mask)
    torch.save({
        "model_type": "selector_scene",
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "activation": args.activation,
        "input_mean": mean,  # (15,) obs part only
        "input_std": std,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
    }, save_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Saved {save_path}  ({n_params} params)")


if __name__ == "__main__":
    main()
