#!/usr/bin/env python3
"""Feature ablation study: systematically drop feature combinations, retrain, evaluate.

Generates all 2^N combinations of dropping the specified features, trains a policy
for each combination, evaluates via evaluate.py --headless, and prints a condensed
comparison table.

Usage:
  python experiments/ablation_study.py \
    --data data/dataset_full.npz \
    --mode base \
    --features heading_error dist_to_target \
    --epochs 25 --num_targets 10 --mps
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SPLIT_SCRIPT = os.path.join(PROJECT_DIR, "split_dataset.py")
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "train_regressor.py")
EVAL_SCRIPT = os.path.join(PROJECT_DIR, "evaluate.py")


def generate_combos(features, max_combo_size=None):
    """Generate all subsets of features to drop, from empty (baseline) to all."""
    limit = max_combo_size if max_combo_size is not None else len(features)
    limit = min(limit, len(features))
    combos = [()]  # baseline -- drop nothing
    for r in range(1, limit + 1):
        combos.extend(combinations(features, r))
    return combos


def combo_label(combo):
    if not combo:
        return "(none -- baseline)"
    return "+".join(combo)


def parse_train_output(stdout):
    """Extract metrics from train_regressor.py stdout."""
    metrics = {}

    # Checkpoint path — the last "Checkpoint: <path>" line is the final renamed one
    for m in re.finditer(r"Checkpoint:\s+(\S+\.pt)", stdout):
        metrics["checkpoint"] = m.group(1)

    # Best val loss: 0.001234 at epoch 18
    m = re.search(r"Best val loss:\s+([\d.]+)\s+at epoch\s+(\d+)", stdout)
    if m:
        metrics["best_val_loss"] = float(m.group(1))
        metrics["best_epoch"] = int(m.group(2))

    # Total time: 4.2min
    m = re.search(r"Total time:\s+([\d.]+)min", stdout)
    if m:
        metrics["train_time_min"] = float(m.group(1))

    return metrics


def parse_eval_output(stdout):
    """Extract metrics from evaluate.py stdout."""
    metrics = {}

    # Best EE error: mean=0.3120  median=0.2850  ...
    m = re.search(r"Best EE error\s*:\s*mean=([\d.]+)\s+median=([\d.]+)", stdout)
    if m:
        metrics["ee_mean"] = float(m.group(1))
        metrics["ee_median"] = float(m.group(2))

    # best < 0.2m: 8/20 (40.0%)
    for threshold in ("0.05", "0.1", "0.2", "0.5", "1.0"):
        pattern = rf"best < {re.escape(threshold)}m:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)"
        m = re.search(pattern, stdout)
        if m:
            metrics[f"pct_within_{threshold}m"] = float(m.group(3))

    # Crashed: 2/20
    m = re.search(r"Crashed:\s*(\d+)/(\d+)", stdout)
    if m:
        metrics["crashes"] = int(m.group(1))
        metrics["total"] = int(m.group(2))

    return metrics


def run_combo(combo, cfg, run_dir, train_passthrough):
    """Run split, train, then evaluate for one feature combination.

    Args:
        combo: tuple of feature names to drop
        cfg: plain dict with keys: mode, data, epochs, num_targets, mps
        run_dir: directory for this run's artifacts
        train_passthrough: list of extra CLI args for train_regressor.py
    """
    os.makedirs(run_dir, exist_ok=True)

    result = {"dropped": list(combo), "label": combo_label(combo)}

    # --- Step 1: Split dataset (no training) ---
    drop_flag = "--drop_base" if cfg["mode"] == "base" else "--drop_arm"
    split_cmd = [
        sys.executable, SPLIT_SCRIPT,
        "--data", cfg["data"],
        "--out_dir", run_dir,
    ]
    if cfg.get("max_samples"):
        split_cmd.extend(["--max_samples", str(cfg["max_samples"])])
    if combo:
        split_cmd.extend([drop_flag] + list(combo))

    split_proc = subprocess.run(split_cmd, capture_output=True, text=True)

    if split_proc.returncode != 0:
        result["error"] = "split_failed"
        result["stderr"] = (split_proc.stderr or "").strip().splitlines()[-5:]
        return result

    # --- Step 2: Train only the target policy ---
    dataset_file = f"dataset_{cfg['mode']}.npz"
    dataset_path = os.path.join(run_dir, dataset_file)

    if not os.path.exists(dataset_path):
        result["error"] = "no_dataset"
        return result

    train_cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--data", dataset_path,
        "--mode", cfg["mode"],
        "--epochs", str(cfg["epochs"]),
    ] + train_passthrough

    # Run from run_dir so checkpoints are saved there
    train_proc = subprocess.run(train_cmd, capture_output=True, text=True, cwd=run_dir)

    if train_proc.returncode != 0:
        result["error"] = "train_failed"
        result["stderr"] = (train_proc.stderr or "").strip().splitlines()[-5:]
        return result

    train_metrics = parse_train_output(train_proc.stdout)
    result.update(train_metrics)

    checkpoint = train_metrics.get("checkpoint")
    if not checkpoint:
        result["error"] = "no_checkpoint"
        return result

    # Resolve checkpoint path (relative to run_dir since we ran from there)
    if not os.path.isabs(checkpoint):
        checkpoint = os.path.join(run_dir, checkpoint)
    result["checkpoint"] = checkpoint

    # --- Step 3: Evaluate ---
    if cfg["mode"] == "base":
        ckpt_flag = "--base_checkpoint"
        mode_flag = "--base_only"
    else:
        ckpt_flag = "--arm_checkpoint"
        mode_flag = "--arm_only"

    eval_cmd = [
        sys.executable, EVAL_SCRIPT,
        ckpt_flag, checkpoint,
        mode_flag,
        "--headless",
        "--num_targets", str(cfg["num_targets"]),
    ]
    if cfg["mps"]:
        eval_cmd.append("--mps")

    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=PROJECT_DIR)

    if eval_proc.returncode != 0:
        result["error"] = "eval_failed"
        result["stderr"] = (eval_proc.stderr or "").strip().splitlines()[-5:]
        return result

    eval_metrics = parse_eval_output(eval_proc.stdout)
    result.update(eval_metrics)

    return result


def print_report(results, mode, features):
    """Print condensed comparison table."""
    n_features = len(features)
    n_combos = len(results)

    print(f"\n{'='*90}")
    print(f"ABLATION STUDY: {mode} policy  ({n_features} features, {n_combos} combinations)")
    print(f"Features tested: {', '.join(features)}")
    print(f"{'='*90}")

    header = (f"{'Dropped':<30s} | {'val_loss':>8s} | {'epoch':>5s} | "
              f"{'ee_mean':>7s} | {'ee_med':>6s} | {'<0.2m':>5s} | {'<0.5m':>5s} | {'crash':>5s}")
    print(header)
    print("-" * len(header))

    best_result = None
    best_score = float("inf")

    for r in results:
        if "error" in r:
            print(f"{r['label']:<30s} | {'ERROR':>8s} | {r.get('error', ''):>5s} |")
            continue

        val_loss = r.get("best_val_loss")
        epoch = r.get("best_epoch")
        ee_mean = r.get("ee_mean")
        ee_med = r.get("ee_median")
        pct_02 = r.get("pct_within_0.2m")
        pct_05 = r.get("pct_within_0.5m")
        crashes = r.get("crashes")

        val_s = f"{val_loss:.6f}" if val_loss is not None else "       -"
        epoch_s = f"{epoch:5d}" if epoch is not None else "    -"
        ee_mean_s = f"{ee_mean:.4f}" if ee_mean is not None else "      -"
        ee_med_s = f"{ee_med:.4f}" if ee_med is not None else "     -"
        pct_02_s = f"{pct_02:4.0f}%" if pct_02 is not None else "    -"
        pct_05_s = f"{pct_05:4.0f}%" if pct_05 is not None else "    -"
        crash_s = f"{crashes:5d}" if crashes is not None else "    -"

        print(f"{r['label']:<30s} | {val_s:>8s} | {epoch_s:>5s} | "
              f"{ee_mean_s:>7s} | {ee_med_s:>6s} | {pct_02_s:>5s} | {pct_05_s:>5s} | {crash_s:>5s}")

        # Track best by ee_mean (lower is better), fall back to val_loss
        score = ee_mean if ee_mean is not None else (val_loss if val_loss is not None else float("inf"))
        if score < best_score:
            best_score = score
            best_result = r

    print("=" * len(header))

    if best_result:
        print(f"\nBest combo: {best_result['label']}  "
              f"(val_loss={best_result.get('best_val_loss', '?')}, "
              f"ee_mean={best_result.get('ee_mean', '?')})")


def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation study: drop feature combos, retrain, evaluate")
    parser.add_argument("--data", type=str, required=True, help="Path to full .npz dataset")
    parser.add_argument("--mode", type=str, required=True, choices=["base", "arm"],
                        help="Policy mode to ablate")
    parser.add_argument("--features", type=str, nargs="+", required=True,
                        help="Feature names to ablate (all combinations will be tested)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per run")
    parser.add_argument("--num_targets", type=int, default=20,
                        help="Evaluation targets per run")
    parser.add_argument("--max_combo_size", type=int, default=None,
                        help="Max number of features to drop at once (caps 2^N growth)")
    parser.add_argument("--mps", action="store_true", help="Use Apple Metal GPU")
    parser.add_argument("--headless", action="store_true",
                        help="(always headless, kept for CLI compatibility)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: experiments/runs/)")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Max parallel workers (default: number of combos)")

    # Pass-through training hyperparams
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Subsample dataset to N rows before splitting (passed to split_dataset.py)")

    args = parser.parse_args()

    # Build pass-through args for train_regressor.py
    train_passthrough = [
        "--hidden_dim", str(args.hidden_dim),
        "--num_layers", str(args.num_layers),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--early_stop", str(args.early_stop),
        "--patience", str(args.patience),
    ]
    if args.mps:
        train_passthrough.append("--mps")

    # Generate combinations
    combos = generate_combos(args.features, args.max_combo_size)
    parallel = args.parallel if args.parallel is not None else len(combos)
    print(f"Ablation study: {args.mode} policy")
    print(f"Features: {args.features}")
    print(f"Combinations: {len(combos)} (2^{len(args.features)}"
          f"{f' capped at size {args.max_combo_size}' if args.max_combo_size else ''})")
    print(f"Epochs: {args.epochs}  Eval targets: {args.num_targets}")
    print(f"Parallel workers: {parallel}")

    # Plain-dict config (picklable, unlike argparse Namespace)
    cfg = {
        "mode": args.mode,
        "data": args.data,
        "epochs": args.epochs,
        "num_targets": args.num_targets,
        "mps": args.mps,
        "max_samples": args.max_samples,
    }

    # Run directory
    runs_dir = args.out_dir or os.path.join(SCRIPT_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Build per-combo arguments
    combo_args = []
    for combo in combos:
        run_id = f"{args.mode}_drop_{'_'.join(combo) if combo else 'none'}"
        run_dir = os.path.join(runs_dir, run_id)
        combo_args.append((combo, cfg, run_dir, train_passthrough, run_id))

    # Run all combinations in parallel
    results = [None] * len(combos)
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        future_to_idx = {}
        for idx, (combo, cfg_dict, run_dir, tp, run_id) in enumerate(combo_args):
            future = executor.submit(run_combo, combo, cfg_dict, run_dir, tp)
            future_to_idx[future] = (idx, run_id)

        for future in as_completed(future_to_idx):
            idx, run_id = future_to_idx[future]
            label = combo_label(combos[idx])
            try:
                result = future.result()
                result["run_id"] = run_id
                results[idx] = result
                # Progress summary
                err = result.get("error")
                if err:
                    stderr_lines = result.get("stderr", [])
                    print(f"  [{idx+1}/{len(combos)}] {label}  -> ERROR: {err}")
                    for line in stderr_lines:
                        print(f"    {line}")
                else:
                    print(f"  [{idx+1}/{len(combos)}] {label}  -> "
                          f"val_loss={result.get('best_val_loss', '?')}  "
                          f"ee_mean={result.get('ee_mean', '?')}  "
                          f"<0.2m={result.get('pct_within_0.2m', '?')}%")
            except Exception as exc:
                print(f"  [{idx+1}/{len(combos)}] {label}  -> EXCEPTION: {exc}")
                results[idx] = {
                    "dropped": list(combos[idx]),
                    "label": label,
                    "run_id": run_id,
                    "error": str(exc),
                }

    # Print final report
    print_report(results, args.mode, args.features)

    # Save JSON results
    json_path = os.path.join(SCRIPT_DIR, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "mode": args.mode,
            "features": args.features,
            "epochs": args.epochs,
            "num_targets": args.num_targets,
            "parallel": parallel,
            "results": results,
        }, f, indent=2)
    print(f"\nFull results saved to: {json_path}")


if __name__ == "__main__":
    main()
