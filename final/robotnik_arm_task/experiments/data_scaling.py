#!/usr/bin/env python3
"""Data scaling experiment: train models at multiple dataset sizes and plot
learning curves to find the minimum data required for good performance.

Usage:
  python experiments/data_scaling.py \
    --data data/dataset_20261027_143306.npz --mode base \
    --models mlp linear random_forest \
    --sample_sizes 10000 50000 100000 500000 \
    --epochs 10 --mps --skip_eval
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SPLIT_SCRIPT = os.path.join(PROJECT_DIR, "split_dataset.py")
EVAL_SCRIPT = os.path.join(PROJECT_DIR, "evaluate.py")

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from regressors import get_regressor, available_regressors
from activations import ACTIVATIONS, DEFAULT_ACTIVATION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_eval_output(stdout):
    """Extract metrics from evaluate.py stdout."""
    metrics = {}
    m = re.search(r"Best EE error\s*:\s*mean=([\d.]+)\s+median=([\d.]+)", stdout)
    if m:
        metrics["ee_mean"] = float(m.group(1))
        metrics["ee_median"] = float(m.group(2))
    for threshold in ("0.05", "0.1", "0.2", "0.5", "1.0"):
        pattern = rf"best < {re.escape(threshold)}m:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)"
        m = re.search(pattern, stdout)
        if m:
            metrics[f"pct_within_{threshold}m"] = float(m.group(3))
    m = re.search(r"Crashed:\s*(\d+)/(\d+)", stdout)
    if m:
        metrics["crashes"] = int(m.group(1))
        metrics["total"] = int(m.group(2))
    return metrics


def get_device(use_mps):
    import torch
    if use_mps and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def random_train_val_split(inputs, labels, n_samples, val_frac=0.2, seed=42):
    """Randomly subsample n_samples rows then split into train/val."""
    rng = np.random.RandomState(seed)
    n_total = inputs.shape[0]

    if n_samples >= n_total:
        idx = np.arange(n_total)
    else:
        idx = rng.choice(n_total, n_samples, replace=False)
        idx.sort()

    X, y = inputs[idx], labels[idx]

    # Shuffle and split
    perm = rng.permutation(len(X))
    split = int(len(X) * (1 - val_frac))
    train_idx, val_idx = perm[:split], perm[split:]

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def plot_learning_curves(all_results, mode, out_dir):
    """Generate learning curve chart: metric vs sample size, one line per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping chart generation.")
        return

    # Organize by model
    models = sorted(set(r["model"] for r in all_results))
    sample_sizes = sorted(set(r["sample_size"] for r in all_results))

    metrics_to_plot = [
        ("val_mse", "Validation MSE"),
        ("mae", "Mean Absolute Error"),
        ("p90", "90th Percentile Error"),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(models), 3)))

    for ax, (key, label) in zip(axes, metrics_to_plot):
        for i, model_name in enumerate(models):
            model_results = [r for r in all_results if r["model"] == model_name]
            model_results.sort(key=lambda r: r["sample_size"])
            xs = [r["sample_size"] for r in model_results]
            ys = [r.get(key, float("nan")) for r in model_results]
            ax.plot(xs, ys, marker="o", label=model_name, color=colors[i], linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Data Scaling — {mode} policy", fontsize=14, fontweight="bold")
    fig.tight_layout()

    chart_path = os.path.join(out_dir, "data_scaling_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Data scaling experiment: train at multiple dataset sizes")
    parser.add_argument("--data", type=str, required=True, help="Path to full .npz dataset")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "arm"],
                        help="Policy mode")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["mlp", "linear", "random_forest"],
                        help=f"Models to train (available: {available_regressors()})")
    parser.add_argument("--sample_sizes", type=int, nargs="+",
                        default=[10000, 50000, 100000, 500000],
                        help="Dataset sizes to evaluate")
    parser.add_argument("--num_targets", type=int, default=20,
                        help="Evaluation targets for sim eval")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip headless sim evaluation (val metrics only)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: experiments/runs/data_scaling)")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS),
                        help=f"Hidden-layer activation for mlp/transformer (default: {DEFAULT_ACTIVATION})")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--early_stop", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--drop_base", type=str, nargs="*", default=[])
    parser.add_argument("--drop_arm", type=str, nargs="*", default=[])
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal GPU")

    args = parser.parse_args()

    runs_dir = args.out_dir or os.path.join(SCRIPT_DIR, "runs", "data_scaling")
    os.makedirs(runs_dir, exist_ok=True)

    sample_sizes = sorted(args.sample_sizes)
    max_size = sample_sizes[-1]

    # ------------------------------------------------------------------
    # Step 1: Split dataset at the largest requested size
    # ------------------------------------------------------------------
    print("=" * 64)
    print(f"DATA SCALING: {args.mode} policy")
    print(f"Models: {args.models}")
    print(f"Sample sizes: {sample_sizes}")
    print("=" * 64)

    split_cmd = [
        sys.executable, SPLIT_SCRIPT,
        "--data", args.data,
        "--out_dir", runs_dir,
        "--max_samples", str(max_size),
    ]
    drop_flag = "--drop_base" if args.mode == "base" else "--drop_arm"
    drop_list = args.drop_base if args.mode == "base" else args.drop_arm
    if drop_list:
        split_cmd.extend([drop_flag] + drop_list)

    print(f"\n[1/4] Splitting dataset (max_samples={max_size})...")
    split_proc = subprocess.run(split_cmd, capture_output=True, text=True)
    if split_proc.returncode != 0:
        print("ERROR: split_dataset.py failed")
        print(split_proc.stderr)
        sys.exit(1)
    print(split_proc.stdout.strip())

    # ------------------------------------------------------------------
    # Step 2: Load split data
    # ------------------------------------------------------------------
    dataset_path = os.path.join(runs_dir, f"dataset_{args.mode}.npz")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Expected {dataset_path} not found after split")
        sys.exit(1)

    data = np.load(dataset_path)
    full_inputs, full_labels, full_metadata = data["inputs"], data["labels"], data["metadata"]
    feature_names = list(data["feature_names"]) if "feature_names" in data else None
    print(f"\nLoaded {full_inputs.shape[0]} samples  inputs={full_inputs.shape}  labels={full_labels.shape}")

    device = get_device(args.mps)
    ACTION_DIM = full_labels.shape[1]

    all_results = []
    n_sizes = len(sample_sizes)
    n_models = len(args.models)

    # ------------------------------------------------------------------
    # Step 3: For each sample size, subsample + train each model
    # ------------------------------------------------------------------
    for si, n_samples in enumerate(sample_sizes, 1):
        print(f"\n{'='*64}")
        print(f"[2/4] Sample size {n_samples}  ({si}/{n_sizes})")
        print(f"{'='*64}")

        # Random subsample + train/val split
        actual_size = min(n_samples, full_inputs.shape[0])
        X_train, y_train, X_val, y_val = random_train_val_split(
            full_inputs, full_labels, n_samples)
        print(f"Samples: {actual_size}  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

        # Normalize
        input_mean = X_train.mean(axis=0)
        input_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - input_mean) / input_std
        X_val_norm = (X_val - input_mean) / input_std

        label_mean = y_train.mean(axis=0)
        label_std = y_train.std(axis=0) + 1e-8
        y_train_norm = (y_train - label_mean) / label_std
        y_val_norm = (y_val - label_mean) / label_std

        fit_kwargs = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            activation=args.activation,
            early_stop=args.early_stop,
            patience=args.patience,
            num_workers=args.num_workers,
            device=device,
        )

        norm_meta = dict(
            input_mean=input_mean,
            input_std=input_std,
            label_mean=label_mean,
            label_std=label_std,
            input_dim=full_inputs.shape[1],
            output_dim=full_labels.shape[1],
            feature_names=feature_names,
        )

        for mi, model_name in enumerate(args.models, 1):
            print(f"\n{'─'*50}")
            print(f"Training: {model_name}  ({mi}/{n_models})  samples={actual_size}")
            print(f"{'─'*50}")

            regressor_cls = get_regressor(model_name)
            regressor = regressor_cls()

            metrics = regressor.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, **fit_kwargs)

            # Save checkpoint
            ckpt_path = os.path.join(runs_dir, f"scaling_{args.mode}_{model_name}_{n_samples}.pt")
            regressor.save_checkpoint(ckpt_path, **norm_meta)

            result = {
                "sample_size": n_samples,
                "actual_size": actual_size,
                "model": model_name,
                **metrics,
                "checkpoint": ckpt_path,
            }
            all_results.append(result)
            print(f"  val_mse={metrics['val_mse']:.6f}  mae={metrics['mae']:.4f}  p90={metrics['p90']:.4f}")

    # ------------------------------------------------------------------
    # Step 3b: Optional sim evaluation
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print(f"\n[3/4] Running sim evaluation ({args.num_targets} targets per checkpoint)...")
        for result in all_results:
            model_name = result["model"]
            ckpt_path = result["checkpoint"]
            n_samples = result["sample_size"]
            print(f"\n  Evaluating: {model_name} @ {n_samples} samples...")

            if args.mode == "base":
                ckpt_flag = "--base_checkpoint"
                mode_flag = "--base_only"
            else:
                ckpt_flag = "--arm_checkpoint"
                mode_flag = "--arm_only"

            eval_cmd = [
                sys.executable, EVAL_SCRIPT,
                ckpt_flag, ckpt_path,
                mode_flag,
                "--headless",
                "--num_targets", str(args.num_targets),
            ]
            if args.mps:
                eval_cmd.append("--mps")

            eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
            if eval_proc.returncode != 0:
                print(f"    ERROR: eval failed for {model_name} @ {n_samples}")
                stderr_lines = (eval_proc.stderr or "").strip().splitlines()
                for line in stderr_lines[-5:]:
                    print(f"      {line}")
                result["eval_error"] = True
                continue

            eval_metrics = parse_eval_output(eval_proc.stdout)
            result.update(eval_metrics)
            ee = eval_metrics.get("ee_mean", "?")
            pct = eval_metrics.get("pct_within_0.5m", "?")
            print(f"    ee_mean={ee}  <0.5m={pct}%")
    else:
        print("\n[3/4] Skipping sim evaluation (--skip_eval)")

    # ------------------------------------------------------------------
    # Step 4: Print results table
    # ------------------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"DATA SCALING: {args.mode} policy")
    print(f"{'='*64}")

    header = f"{'Samples':>8s} | {'Model':<18s}| {'val_mse':>8s} | {'mae':>8s} | {'p90':>8s}"
    print(header)
    print("-" * len(header))

    for r in sorted(all_results, key=lambda x: (x["sample_size"], x["model"])):
        val_mse_s = f"{r['val_mse']:.6f}" if "val_mse" in r else "       -"
        mae_s = f"{r['mae']:.4f}" if "mae" in r else "       -"
        p90_s = f"{r['p90']:.4f}" if "p90" in r else "       -"
        print(f"{r['sample_size']:>8d} | {r['model']:<18s}| {val_mse_s:>8s} | {mae_s:>8s} | {p90_s:>8s}")

    print("=" * len(header))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    # Make results JSON-serializable
    for r in all_results:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)

    json_path = os.path.join(runs_dir, "data_scaling_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "models": args.models,
            "sample_sizes": sample_sizes,
            "epochs": args.epochs,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ------------------------------------------------------------------
    # Generate learning curve chart
    # ------------------------------------------------------------------
    print("\n[4/4] Generating learning curve chart...")
    plot_learning_curves(all_results, args.mode, runs_dir)


if __name__ == "__main__":
    main()
