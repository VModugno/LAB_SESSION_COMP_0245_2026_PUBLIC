#!/usr/bin/env python3
"""Multi-model comparison: train Linear, RF, MLP (and more) on the same data,
collect val metrics + headless sim evaluation, print comparison table.

Usage:
  python experiments/model_comparison.py \
    --data data/full.npz --mode base \
    --models mlp linear random_forest \
    --epochs 50 --num_targets 10 --max_samples 1000000 --mps
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

# Ensure project dir is on sys.path so we can import project modules
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from train_regressor import split_by_trajectory
from regressors import get_regressor, available_regressors
from activations import ACTIVATIONS, DEFAULT_ACTIVATION


# ---------------------------------------------------------------------------
# Helpers (reused from ablation_study.py patterns)
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


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_comparison(results, mode, out_dir, has_eval):
    """Generate bar charts comparing models across metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping chart generation.")
        return

    model_names = [r["model"] for r in results]

    # Collect metric groups: val metrics always, sim metrics if available
    val_metrics = [
        ("val_mse", "Validation MSE"),
        ("mae", "Mean Absolute Error"),
        ("p90", "90th Percentile Error"),
    ]
    sim_metrics = [
        ("ee_mean", "Mean EE Error (sim)"),
        ("pct_within_0.5m", "% Within 0.5m (sim)"),
    ]

    metrics_to_plot = list(val_metrics)
    if has_eval:
        metrics_to_plot.extend(sim_metrics)

    # Filter to metrics that at least one result has
    metrics_to_plot = [(k, label) for k, label in metrics_to_plot
                       if any(k in r for r in results)]

    if not metrics_to_plot:
        print("\nNo metrics available for plotting.")
        return

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax, (key, label) in zip(axes, metrics_to_plot):
        values = [r.get(key, 0) for r in results]
        bars = ax.bar(model_names, values, color=colors[:len(model_names)])
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val != 0:
                fmt = f"{val:.4f}" if val < 10 else f"{val:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Model Comparison — {mode} policy", fontsize=14, fontweight="bold")
    fig.tight_layout()

    chart_path = os.path.join(out_dir, "comparison_chart.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple regressor types on the same dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to full .npz dataset")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "arm"],
                        help="Policy mode")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["mlp", "linear", "random_forest"],
                        help=f"Models to compare (available: {available_regressors()})")
    parser.add_argument("--num_targets", type=int, default=20,
                        help="Evaluation targets for sim eval")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip headless sim evaluation (val metrics only)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: experiments/runs/comparison)")

    # Pass-through training hyperparams
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
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample dataset to N rows before splitting")
    parser.add_argument("--drop_base", type=str, nargs="*", default=[])
    parser.add_argument("--drop_arm", type=str, nargs="*", default=[])
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal GPU")

    args = parser.parse_args()

    runs_dir = args.out_dir or os.path.join(SCRIPT_DIR, "runs", "comparison")
    os.makedirs(runs_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Split dataset (subprocess, same as ablation)
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"MODEL COMPARISON: {args.mode} policy")
    print(f"Models: {args.models}")
    print("=" * 60)

    split_cmd = [
        sys.executable, SPLIT_SCRIPT,
        "--data", args.data,
        "--out_dir", runs_dir,
    ]
    if args.max_samples:
        split_cmd.extend(["--max_samples", str(args.max_samples)])
    drop_flag = "--drop_base" if args.mode == "base" else "--drop_arm"
    drop_list = args.drop_base if args.mode == "base" else args.drop_arm
    if drop_list:
        split_cmd.extend([drop_flag] + drop_list)

    print(f"\n[1/3] Splitting dataset...")
    split_proc = subprocess.run(split_cmd, capture_output=True, text=True)
    if split_proc.returncode != 0:
        print("ERROR: split_dataset.py failed")
        print(split_proc.stderr)
        sys.exit(1)
    print(split_proc.stdout.strip())

    # ------------------------------------------------------------------
    # Step 2: Load split data and train each model via regressors API
    # ------------------------------------------------------------------
    dataset_path = os.path.join(runs_dir, f"dataset_{args.mode}.npz")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Expected {dataset_path} not found after split")
        sys.exit(1)

    data = np.load(dataset_path)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    feature_names = list(data["feature_names"]) if "feature_names" in data else None
    print(f"\nLoaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")

    # Split by trajectory
    X_train, y_train, X_val, y_val = split_by_trajectory(inputs, labels, metadata)
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    # Normalize inputs
    input_mean = X_train.mean(axis=0)
    input_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - input_mean) / input_std
    X_val_norm = (X_val - input_mean) / input_std

    # Normalize labels
    ACTION_DIM = labels.shape[1]
    label_mean = y_train.mean(axis=0)
    label_std = y_train.std(axis=0) + 1e-8
    y_train_norm = (y_train - label_mean) / label_std
    y_val_norm = (y_val - label_mean) / label_std

    device = get_device(args.mps)

    # Common kwargs for regressors
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

    # Normalization metadata saved in every checkpoint
    norm_meta = dict(
        input_mean=input_mean,
        input_std=input_std,
        label_mean=label_mean,
        label_std=label_std,
        input_dim=inputs.shape[1],
        output_dim=labels.shape[1],
        feature_names=feature_names,
    )

    print(f"\n[2/3] Training models...")
    results = []
    checkpoints = {}

    for model_name in args.models:
        print(f"\n{'─'*50}")
        print(f"Training: {model_name}")
        print(f"{'─'*50}")

        regressor_cls = get_regressor(model_name)
        regressor = regressor_cls()

        # MLP trains on normalized data; sklearn models also get normalized data
        # for consistency (normalization is baked into the checkpoint for inference)
        metrics = regressor.fit(X_train_norm, y_train_norm, X_val_norm, y_val_norm, **fit_kwargs)

        # Save checkpoint
        ckpt_path = os.path.join(runs_dir, f"comparison_{args.mode}_{model_name}.pt")
        regressor.save_checkpoint(ckpt_path, **norm_meta)
        checkpoints[model_name] = ckpt_path

        result = {"model": model_name, **metrics, "checkpoint": ckpt_path}
        results.append(result)
        print(f"  Saved: {ckpt_path}")
        print(f"  val_mse={metrics['val_mse']:.6f}  mae={metrics['mae']:.4f}  p90={metrics['p90']:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Sim evaluation via evaluate.py --headless
    # ------------------------------------------------------------------
    if not args.skip_eval:
        print(f"\n[3/3] Running sim evaluation ({args.num_targets} targets per model)...")

        for result in results:
            model_name = result["model"]
            ckpt_path = result["checkpoint"]
            print(f"\n  Evaluating: {model_name} ...")

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
                print(f"    ERROR: eval failed for {model_name}")
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
        print("\n[3/3] Skipping sim evaluation (--skip_eval)")

    # ------------------------------------------------------------------
    # Print comparison table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON: {args.mode} policy")
    print(f"{'='*70}")

    has_eval = not args.skip_eval
    if has_eval:
        header = (f"{'Model':<18s}| {'val_mse':>8s} | {'mae':>8s} | {'p90':>8s} | "
                  f"{'ee_mean':>7s} | {'<0.5m':>5s}")
    else:
        header = f"{'Model':<18s}| {'val_mse':>8s} | {'mae':>8s} | {'p90':>8s}"
    print(header)
    print("-" * len(header))

    for r in results:
        val_mse_s = f"{r['val_mse']:.6f}" if "val_mse" in r else "       -"
        mae_s = f"{r['mae']:.4f}" if "mae" in r else "       -"
        p90_s = f"{r['p90']:.4f}" if "p90" in r else "       -"

        if has_eval:
            ee_s = f"{r['ee_mean']:.4f}" if "ee_mean" in r else "      -"
            pct_s = f"{r.get('pct_within_0.5m', 0):4.0f}%" if "pct_within_0.5m" in r else "    -"
            print(f"{r['model']:<18s}| {val_mse_s:>8s} | {mae_s:>8s} | {p90_s:>8s} | "
                  f"{ee_s:>7s} | {pct_s:>5s}")
        else:
            print(f"{r['model']:<18s}| {val_mse_s:>8s} | {mae_s:>8s} | {p90_s:>8s}")

    print("=" * len(header))

    # Save JSON results
    json_path = os.path.join(runs_dir, "comparison_results.json")
    # Make results JSON-serializable (convert numpy types)
    for r in results:
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                r[k] = float(v) if isinstance(v, np.floating) else int(v)
    with open(json_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "models": args.models,
            "epochs": args.epochs,
            "num_targets": args.num_targets,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ------------------------------------------------------------------
    # Generate comparison charts
    # ------------------------------------------------------------------
    plot_comparison(results, args.mode, runs_dir, has_eval)


if __name__ == "__main__":
    main()
