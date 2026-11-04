#!/usr/bin/env python3
"""Sweep threshold_scale values for evaluate.py and plot comparison charts.

Runs all scales in parallel (one subprocess per scale).

Usage:
  python sweep_threshold.py --base_checkpoint <path> --arm_checkpoint <path> \
      --scales 0.25 0.5 0.75 1.0 --num_targets 10
"""
import argparse
import json
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_results(results):
    scales = [r["threshold_scale"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Threshold Scale Sweep", fontsize=14, fontweight="bold")

    # 1. Best EE error
    ax = axes[0, 0]
    ax.bar(range(len(scales)), [r["best_ee_mean"] for r in results],
           yerr=[r["best_ee_std"] or 0 for r in results],
           tick_label=[str(s) for s in scales], capsize=4, color="steelblue")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("Best EE Error (m)")
    ax.set_title("Best EE Error (mean ± std)")

    # 2. Final EE error
    ax = axes[0, 1]
    ax.bar(range(len(scales)), [r["final_ee_mean"] for r in results],
           tick_label=[str(s) for s in scales], color="coral")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("Final EE Error (m)")
    ax.set_title("Final EE Error (mean)")

    # 3. Base XY error to standoff
    ax = axes[0, 2]
    vals = [r["base_xy_err_mean"] or 0 for r in results]
    ax.bar(range(len(scales)), vals,
           tick_label=[str(s) for s in scales], color="seagreen")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("XY Error to Standoff (m)")
    ax.set_title("Base XY Distance to Standoff (mean)")

    # 4. Bearing error
    ax = axes[1, 0]
    vals = [r["bearing_err_deg_mean"] or 0 for r in results]
    ax.bar(range(len(scales)), vals,
           tick_label=[str(s) for s in scales], color="orchid")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("Bearing Error (deg)")
    ax.set_title("Base Bearing Error (mean)")

    # 5. Arm best EE
    ax = axes[1, 1]
    vals = [r["arm_best_mean"] or 0 for r in results]
    ax.bar(range(len(scales)), vals,
           tick_label=[str(s) for s in scales], color="goldenrod")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("Arm Best EE Error (m)")
    ax.set_title("Arm Phase Best EE (mean)")

    # 6. Switch time
    ax = axes[1, 2]
    vals = [r["switch_time_mean"] or 0 for r in results]
    ax.bar(range(len(scales)), vals,
           tick_label=[str(s) for s in scales], color="slategray")
    ax.set_xlabel("Threshold Scale")
    ax.set_ylabel("Time (s)")
    ax.set_title("Base Phase Duration (mean)")

    plt.tight_layout()
    out_path = "sweep_results/threshold_sweep_5-1.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Sweep threshold_scale and plot results")
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--arm_checkpoint", type=str, required=True)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0],
                        help="threshold_scale values to sweep (default: 0.25 0.5 0.75 1.0)")
    parser.add_argument("--num_targets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: number of scales)")
    args, extra = parser.parse_known_args()

    os.makedirs("sweep_results", exist_ok=True)

    max_workers = args.workers or len(args.scales)
    print(f"Launching {len(args.scales)} evaluations in parallel "
          f"(max {max_workers} workers)...")
    print(f"Scales: {args.scales}")
    print(f"Logs in sweep_results/scale_<value>.log\n")

    # Launch all in parallel
    import time as _time
    procs = {}
    log_files = {}
    for scale in args.scales:
        out_file = f"sweep_results/scale_{scale:.2f}.json"
        log_file = f"sweep_results/scale_{scale:.2f}.log"
        cmd = [
            sys.executable, "evaluate.py",
            "--base_checkpoint", args.base_checkpoint,
            "--arm_checkpoint", args.arm_checkpoint,
            "--threshold_scale", str(scale),
            "--num_targets", str(args.num_targets),
            "--seed", str(args.seed),
            "--json_out", out_file,
            "--headless",
        ] + extra
        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs[scale] = p
        log_files[scale] = lf
        print(f"  Started scale={scale:.2f}  pid={p.pid}")

    # Poll until all done, showing progress
    results_map = {}
    done = set()
    while len(done) < len(args.scales):
        _time.sleep(2)
        for scale, p in procs.items():
            if scale in done:
                continue
            ret = p.poll()
            if ret is not None:
                done.add(scale)
                log_files[scale].close()
                log_path = f"sweep_results/scale_{scale:.2f}.log"
                json_path = f"sweep_results/scale_{scale:.2f}.json"
                if ret == 0 and os.path.exists(json_path):
                    with open(json_path) as f:
                        r = json.load(f)
                    results_map[scale] = r
                    print(f"  [DONE] scale={scale:.2f}  "
                          f"best_ee={r['best_ee_mean']:.4f}  "
                          f"arm_best={r['arm_best_mean']:.4f}")
                else:
                    print(f"  [FAIL] scale={scale:.2f}  exit={ret}  (see {log_path})")
        running = len(args.scales) - len(done)
        if running > 0:
            print(f"  ... {running} still running ...", flush=True)

    # Order results by scale
    results = [results_map[s] for s in sorted(results_map.keys())]

    if not results:
        print("All runs failed.")
        return

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Scale':>8s} | {'Best EE':>10s} | {'Final EE':>10s} | {'Base XY':>10s} | "
          f"{'Bearing':>10s} | {'Arm Best':>10s} | {'Switch t':>10s}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['threshold_scale']:8.2f} | "
              f"{r['best_ee_mean']:10.4f} | "
              f"{r['final_ee_mean']:10.4f} | "
              f"{(r['base_xy_err_mean'] or 0):10.4f} | "
              f"{(r['bearing_err_deg_mean'] or 0):10.1f}° | "
              f"{(r['arm_best_mean'] or 0):10.4f} | "
              f"{(r['switch_time_mean'] or 0):10.1f}s")
    print(f"{'='*90}")

    plot_results(results)


if __name__ == "__main__":
    main()
