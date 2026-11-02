#!/usr/bin/env python3
"""Analyse which input features matter most (and least) for the trained BC policy.

Methods:
  1. Permutation importance — shuffle each feature, measure val MSE increase
  2. Gradient saliency     — mean |∂output/∂input| over a sample of the val set

Usage:
  python feature_importance.py --checkpoint bc_policy_100.pt --data data/dataset_20261025_174551.npz
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
from run_policy import load_policy, build_mlp
from train_regressor import split_by_trajectory

# Feature names for known dataset layouts
NAMES_44 = (
    ["base_pos_x", "base_pos_y", "base_bearing"]
    + ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    + ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    + [f"q_base_{i}" for i in range(1, 5)]
    + [f"qd_base_{i}" for i in range(1, 5)]
    + [f"q_arm_{i}" for i in range(1, 8)]
    + [f"qd_arm_{i}" for i in range(1, 8)]
    + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
    + ["target_x", "target_y", "target_z"]
    + [f"tau_gravity_{i}" for i in range(1, 8)]
)

GROUPS_44 = {
    "base_pos":     [0, 1],
    "base_bearing": [2],
    "base_lin_vel": [3, 4, 5],
    "base_ang_vel": [6, 7, 8],
    "q_base":       list(range(9, 13)),
    "qd_base":      list(range(13, 17)),
    "q_arm":        list(range(17, 24)),
    "qd_arm":       list(range(24, 31)),
    "ee_pos":       [31, 32, 33],
    "target":       [34, 35, 36],
    "tau_gravity":  list(range(37, 44)),
}

NAMES_14 = (
    ["base_pos_x", "base_pos_y", "base_bearing"]
    + ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"]
    + ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]
    + ["target_x", "target_y", "target_z"]
    + ["dist_to_target", "heading_error"]
)

GROUPS_14 = {
    "base_pos":     [0, 1],
    "base_bearing": [2],
    "base_lin_vel": [3, 4, 5],
    "base_ang_vel": [6, 7, 8],
    "target":       [9, 10, 11],
    "goal_rel":     [12, 13],
}

NAMES_27 = (
    [f"q_arm_{i}" for i in range(1, 8)]
    + [f"qd_arm_{i}" for i in range(1, 8)]
    + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
    + ["target_x", "target_y", "target_z"]
    + [f"tau_gravity_{i}" for i in range(1, 8)]
)

GROUPS_27 = {
    "q_arm":        list(range(0, 7)),
    "qd_arm":       list(range(7, 14)),
    "ee_pos":       [14, 15, 16],
    "target":       [17, 18, 19],
    "tau_gravity":  list(range(20, 27)),
}


def get_names_and_groups(n_features):
    """Return (names, groups) matching the dataset dimension."""
    if n_features == 44:
        return NAMES_44, GROUPS_44
    elif n_features == 14:
        return NAMES_14, GROUPS_14
    elif n_features == 27:
        return NAMES_27, GROUPS_27
    else:
        # Fallback: generic names, no groups
        return [f"feat_{i}" for i in range(n_features)], None


def _eval_single_feature(args_tuple):
    """Worker function: shuffle one feature n_repeats times and return mean MSE increase.

    Receives all data as numpy arrays so it can be pickled across processes.
    """
    (feat_idx, X_norm, y_val, checkpoint_path, n_repeats, seed) = args_tuple
    # Each worker loads its own model (avoids pickling torch model / sharing GPU state)
    model, _, _, _, _, _ = load_policy(checkpoint_path, device="cpu")
    model.eval()
    X_t_base = torch.tensor(X_norm, dtype=torch.float32)
    y_t = torch.tensor(y_val, dtype=torch.float32)

    with torch.no_grad():
        baseline_mse = nn.functional.mse_loss(model(X_t_base), y_t).item()

    rng = np.random.RandomState(seed + feat_idx)
    mse_increases = []
    for _ in range(n_repeats):
        X_shuffled = X_norm.copy()
        X_shuffled[:, feat_idx] = rng.permutation(X_shuffled[:, feat_idx])
        X_s = torch.tensor(X_shuffled, dtype=torch.float32)
        with torch.no_grad():
            shuffled_mse = nn.functional.mse_loss(model(X_s), y_t).item()
        mse_increases.append(shuffled_mse - baseline_mse)
    return feat_idx, np.mean(mse_increases), baseline_mse


def permutation_importance(model, X_val, y_val, mean, std, device,
                           n_repeats=5, seed=42, num_workers=0, checkpoint_path=None):
    """Compute permutation importance for each feature.

    Returns array of shape (n_features,) — increase in MSE when each feature is shuffled.
    """
    X_norm = (X_val - mean) / std
    n_features = X_val.shape[1]
    feat_names, _ = get_names_and_groups(n_features)

    if num_workers > 0 and checkpoint_path:
        # Parallel: each worker handles one feature
        work = [(fi, X_norm, y_val, checkpoint_path, n_repeats, seed)
                for fi in range(n_features)]
        importance = np.zeros(n_features)
        baseline_mse = None
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_eval_single_feature, w): w[0] for w in work}
            done = 0
            for future in as_completed(futures):
                feat_idx, imp, bm = future.result()
                importance[feat_idx] = imp
                if baseline_mse is None:
                    baseline_mse = bm
                done += 1
                print(f"  [{done}/{n_features}] {feat_names[feat_idx]}: {imp:.6f}")
        return importance, baseline_mse
    else:
        # Sequential fallback
        model.eval()
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
        y_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        with torch.no_grad():
            baseline_mse = nn.functional.mse_loss(model(X_t), y_t).item()

        rng = np.random.RandomState(seed)
        importance = np.zeros(n_features)
        for feat_idx in range(n_features):
            mse_increases = []
            for _ in range(n_repeats):
                X_shuffled = X_norm.copy()
                X_shuffled[:, feat_idx] = rng.permutation(X_shuffled[:, feat_idx])
                X_s = torch.tensor(X_shuffled, dtype=torch.float32).to(device)
                with torch.no_grad():
                    shuffled_mse = nn.functional.mse_loss(model(X_s), y_t).item()
                mse_increases.append(shuffled_mse - baseline_mse)
            importance[feat_idx] = np.mean(mse_increases)
            print(f"  [{feat_idx+1}/{n_features}] {feat_names[feat_idx]}: {importance[feat_idx]:.6f}")
        return importance, baseline_mse


def gradient_saliency(model, X_val, mean, std, device, n_samples=5000, seed=42):
    """Compute mean absolute gradient of outputs w.r.t. each input feature.

    Returns array of shape (37,) — average |∂output/∂input| across samples and outputs.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(X_val), size=min(n_samples, len(X_val)), replace=False)

    X_norm = (X_val[idxs] - mean) / std
    X_t = torch.tensor(X_norm, dtype=torch.float32).to(device).requires_grad_(True)

    pred = model(X_t)
    pred.sum().backward()

    saliency = X_t.grad.abs().mean(dim=0).cpu().numpy()
    return saliency


def print_ranked(title, names, scores, groups=None):
    """Print features ranked by score, highest first."""
    order = np.argsort(scores)[::-1]

    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"  {'Rank':<6}{'Feature':<20}{'Score':>12}{'% of top':>10}")
    print(f"  {'-' * 48}")
    top = scores[order[0]] if scores[order[0]] != 0 else 1.0
    for rank, idx in enumerate(order, 1):
        pct = scores[idx] / top * 100
        bar = "#" * int(pct / 5)
        print(f"  {rank:<6}{names[idx]:<20}{scores[idx]:>12.6f}{pct:>9.1f}%  {bar}")

    # Group summary
    if groups:
        print(f"\n  {'Group':<16}{'Mean Score':>12}")
        print(f"  {'-' * 28}")
        group_scores = {}
        for gname, gidxs in groups.items():
            group_scores[gname] = np.mean(scores[gidxs])
        for gname in sorted(group_scores, key=group_scores.get, reverse=True):
            print(f"  {gname:<16}{group_scores[gname]:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Feature importance analysis for BC policy")
    parser.add_argument("--checkpoint", "--policy", type=str, default="bc_policy_100.pt",
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of val samples to use (default: 10000)")
    parser.add_argument("--n_repeats", type=int, default=5,
                        help="Permutation repeats per feature (default: 5)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal (MPS) GPU")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Parallel workers for permutation importance (default: 0 = sequential)")
    args = parser.parse_args()

    # Device
    import torch as _torch
    if args.mps and _torch.backends.mps.is_available():
        device = "mps"
    elif _torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    model, mean, std, _, _, _ = load_policy(args.checkpoint, device=device)
    print(f"Loaded policy from {args.checkpoint}  device={device}")

    # Load data and get val split
    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    _, _, X_val, y_val = split_by_trajectory(inputs, labels, metadata)
    print(f"Val set: {X_val.shape[0]} samples")

    # Subsample for speed
    rng = np.random.RandomState(0)
    if len(X_val) > args.n_samples:
        idxs = rng.choice(len(X_val), size=args.n_samples, replace=False)
        X_val = X_val[idxs]
        y_val = y_val[idxs]
        print(f"Subsampled to {args.n_samples} samples for analysis")

    n_features = X_val.shape[1]
    input_names, feature_groups = get_names_and_groups(n_features)
    print(f"Features: {n_features}  ({', '.join(input_names[:3])}...)")

    # --- Permutation importance ---
    workers_str = f" ({args.num_workers} workers)" if args.num_workers > 0 else " (sequential)"
    print(f"\nRunning permutation importance...{workers_str}")
    perm_imp, baseline = permutation_importance(model, X_val, y_val, mean, std, device,
                                                 n_repeats=args.n_repeats,
                                                 num_workers=args.num_workers,
                                                 checkpoint_path=args.checkpoint)
    print(f"Baseline val MSE: {baseline:.6f}")
    print_ranked("PERMUTATION IMPORTANCE (MSE increase when feature is shuffled)",
                 input_names, perm_imp, feature_groups)

    # --- Gradient saliency ---
    print("\nRunning gradient saliency...")
    saliency = gradient_saliency(model, X_val, mean, std, device, n_samples=args.n_samples)
    print_ranked("GRADIENT SALIENCY (mean |∂output/∂input|)",
                 input_names, saliency, feature_groups)

    # --- Features that DON'T matter ---
    perm_threshold = baseline * 0.001  # less than 0.1% MSE increase
    useless = [input_names[i] for i in range(n_features) if perm_imp[i] < perm_threshold]
    print(f"\n{'=' * 65}")
    print(f"  FEATURES WITH NEGLIGIBLE IMPACT (<0.1% MSE increase):")
    if useless:
        for name in useless:
            print(f"    - {name}")
    else:
        print(f"    All features contribute meaningfully.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()


