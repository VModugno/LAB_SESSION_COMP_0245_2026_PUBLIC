#!/usr/bin/env python3
"""Split the full expert dataset into base-navigation and arm-reaching subsets.

Phase detection uses the wheel command norm:
  - Base phase: norm(wheel_cmds) > 0.05  (wheels active)
  - Arm phase:  norm(wheel_cmds) < 0.05  (wheels stopped)

Base dataset (12 inputs -> 2 outputs):
  Inputs:  base_pos_xy(2), bearing(1), lin_vel_xy(2), ang_vel(3), target_xy(2),
           dist_to_target(1), heading_error(1)
  Labels:  v_cmd(1), w_cmd(1)  — linear/angular velocity from wheel inverse kinematics

Arm dataset (27 inputs -> 7 outputs):
  Inputs:  q_arm(7), qd_arm(7), ee_pos(3), target_ee_pos(3), tau_gravity(7)
  Labels:  torque_arm(7)
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from activations import ACTIVATIONS, DEFAULT_ACTIVATION


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

BASE_INPUT_INDICES = [0, 1, 2, 3, 4, 6, 7, 8, 34, 35]             # 10 raw dims + 2 computed

BASE_FEATURE_NAMES = [
    "base_pos_x", "base_pos_y", "base_bearing",
    "base_lin_vel_x", "base_lin_vel_y",
    "base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z",
    "target_x", "target_y",
    "dist_to_target", "heading_error",
]

ARM_FEATURE_NAMES = (
    [f"q_arm_{i}" for i in range(1, 8)]
    + [f"qd_arm_{i}" for i in range(1, 8)]
    + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
    + ["target_x", "target_y", "target_z"]
)
ARM_INPUT_INDICES = list(range(17, 24)) + list(range(24, 31)) + \
                    [31, 32, 33, 34, 35, 36]                        # 20 dims

# Full 11-dim label layout:  wheel_cmds(0:4), torque_arm(4:11)
# Wheel kinematic constants (from BaseControllerParams)
WHEEL_RADIUS = 0.11
WHEEL_BASE_WIDTH = 0.66
WHEEL_THRESHOLD = 0.05


def main():
    parser = argparse.ArgumentParser(description="Split expert dataset into base and arm subsets")
    parser.add_argument("--data", type=str, required=True, help="Path to full .npz dataset")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--train", action="store_true",
                        help="Train base and arm policies after splitting")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (used with --train, default: 100)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden layer size (used with --train, default: 512)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of hidden layers (used with --train, default: 3)")
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION, choices=list(ACTIVATIONS),
                        help=f"Hidden-layer activation (used with --train, default: {DEFAULT_ACTIVATION})")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size (used with --train, default: 2048)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (used with --train, default: 1e-3)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (used with --train, default: 0)")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple Metal GPU (used with --train)")
    parser.add_argument("--drop_base", type=str, nargs="*", default=[],
                        help="Base features to drop (e.g. --drop_base dist_to_target heading_error)")
    parser.add_argument("--drop_arm", type=str, nargs="*", default=[],
                        help="Arm features to drop (e.g. --drop_arm tau_gravity_1 tau_gravity_2)")
    parser.add_argument("--early_stop", type=float, default=0.0,
                        help="Val loss improvement threshold (used with --train, default: 0)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Epochs without improvement before early stopping (used with --train, default: 20)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Randomly subsample to this many rows before splitting (default: use all)")
    parser.add_argument("--arm_full_torque", action="store_true",
                        help="Use full expert torques as arm labels (includes gravity) instead of residual")
    parser.add_argument("--heading_threshold", type=float, default=0.15,
                        help="Heading error (rad) below which base sub-phase is 'drive' "
                             "(above = 'rotate'). Set to 0 to disable phase feature. (default: 0.15)")
    parser.add_argument("--phase_by_vcmd", type=float, default=None,
                        help="Label phase by expert v_cmd magnitude instead of heading error. "
                             "Samples with |v_cmd| >= this threshold are 'drive', rest are 'rotate'. "
                             "(e.g. 0.02)")
    args = parser.parse_args()

    data = np.load(args.data)
    inputs, labels, metadata = data["inputs"], data["labels"], data["metadata"]
    print(f"Loaded {inputs.shape[0]} samples  inputs={inputs.shape}  labels={labels.shape}")

    if args.max_samples and inputs.shape[0] > args.max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(inputs.shape[0], args.max_samples, replace=False)
        idx.sort()
        inputs, labels, metadata = inputs[idx], labels[idx], metadata[idx]
        print(f"Subsampled to {inputs.shape[0]} samples")

    # Phase detection
    wheel_norms = np.linalg.norm(labels[:, :4], axis=1)
    base_mask = wheel_norms > WHEEL_THRESHOLD
    arm_mask = wheel_norms < WHEEL_THRESHOLD
    print(f"Base phase: {base_mask.sum()} samples ({100*base_mask.mean():.1f}%)")
    print(f"Arm phase:  {arm_mask.sum()} samples ({100*arm_mask.mean():.1f}%)")

    out_dir = args.out_dir or os.path.dirname(args.data)

    # Base dataset: 13 inputs -> 2 outputs (v_cmd, w_cmd)
    # Raw features from full observation
    base_raw = inputs[base_mask][:, BASE_INPUT_INDICES]   # (N, 11)
    # Computed goal-relative features
    base_pos_xy = inputs[base_mask][:, :2]                # cols 0,1
    bearing = inputs[base_mask][:, 2]                     # col 2
    target_xy = inputs[base_mask][:, 34:36]               # cols 34,35
    delta = target_xy - base_pos_xy
    dist_to_target = np.linalg.norm(delta, axis=1, keepdims=True)           # (N, 1)
    heading_to_target = np.arctan2(delta[:, 1], delta[:, 0])                # (N,)
    heading_error = np.arctan2(np.sin(heading_to_target - bearing),
                               np.cos(heading_to_target - bearing))[:, None]  # (N, 1) wrapped
    base_inputs = np.hstack([base_raw, dist_to_target, heading_error])      # (N, 14)
    base_names = list(BASE_FEATURE_NAMES)

    # Drop requested base features
    if args.drop_base:
        drop_idxs = []
        for name in args.drop_base:
            if name in base_names:
                drop_idxs.append(base_names.index(name))
            else:
                print(f"WARNING: unknown base feature '{name}', skipping. "
                      f"Available: {base_names}")
        if drop_idxs:
            keep = [i for i in range(base_inputs.shape[1]) if i not in drop_idxs]
            dropped = [base_names[i] for i in drop_idxs]
            base_inputs = base_inputs[:, keep]
            base_names = [base_names[i] for i in keep]
            print(f"Dropped base features: {dropped}  → {base_inputs.shape[1]} dims")

    # Convert wheel cmds [rw, lw, lw, rw] -> [v_cmd, w_cmd]
    wheel_raw = labels[base_mask][:, :4]                  # (N, 4)
    rw = wheel_raw[:, 0]     # front_right == back_right
    lw = wheel_raw[:, 1]     # front_left  == back_left
    v_cmd = (rw + lw) / 2.0 * WHEEL_RADIUS               # m/s

    # Compute phase feature (rotate=0, drive=1) — appended as input feature
    phase_col = None
    if args.phase_by_vcmd is not None:
        is_drive = np.abs(v_cmd) >= args.phase_by_vcmd
        phase_col = is_drive.astype(np.float64)
        n_drive = is_drive.sum()
        n_rotate = len(phase_col) - n_drive
        print(f"Phase feature (by v_cmd): rotate={n_rotate} ({100*n_rotate/len(phase_col):.1f}%)  "
              f"drive={n_drive} ({100*n_drive/len(phase_col):.1f}%)  "
              f"|v_cmd| threshold={args.phase_by_vcmd}")
    elif args.heading_threshold > 0:
        is_rotate = np.abs(heading_error.flatten()) >= args.heading_threshold
        phase_col = (~is_rotate).astype(np.float64)
        n_rotate = is_rotate.sum()
        n_drive = len(phase_col) - n_rotate
        print(f"Phase feature (by heading): rotate={n_rotate} ({100*n_rotate/len(phase_col):.1f}%)  "
              f"drive={n_drive} ({100*n_drive/len(phase_col):.1f}%)  "
              f"threshold={args.heading_threshold:.3f} rad")
    if phase_col is not None:
        base_inputs = np.column_stack([base_inputs, phase_col])
        base_names.append("phase")
    w_cmd = (rw - lw) / WHEEL_BASE_WIDTH * WHEEL_RADIUS   # rad/s
    base_labels = np.column_stack([v_cmd, w_cmd])             # (N, 2)
    base_metadata = metadata[base_mask]
    base_path = os.path.join(out_dir, "dataset_base.npz")
    np.savez(base_path, inputs=base_inputs, labels=base_labels, metadata=base_metadata,
             feature_names=np.array(base_names))
    print(f"Saved base dataset: {base_path}  inputs={base_inputs.shape}  labels={base_labels.shape}")
    print(f"  Features: {base_names}")

    # Arm dataset: 20 inputs -> 7 outputs (torque arm only)
    # Extract raw columns then transform ee_pos and target to base-relative frame
    arm_raw = inputs[arm_mask][:, ARM_INPUT_INDICES]  # (N, 20)
    # q_arm(0:7), qd_arm(7:14) stay as-is; ee_pos(14:17), target(17:20) need transform
    arm_base_pos = inputs[arm_mask][:, :3].copy()       # base x, y, bearing→z=0
    arm_base_pos[:, 2] = 0.0  # base z position not stored; use 0 (ground plane)
    arm_bearing = inputs[arm_mask][:, 2]                 # base_bearing
    # Build per-sample rotation: R @ (world - base_pos)
    c = np.cos(arm_bearing)
    s = np.sin(arm_bearing)
    ee_world = arm_raw[:, 14:17].copy()
    tgt_world = arm_raw[:, 17:20].copy()
    # Translate to base origin (use actual base z from sim — col 0,1 are xy, z≈0.15 not stored,
    # but the height difference is what matters for the arm, so only rotate xy and keep z relative)
    # For simplicity: rotate xy, keep z as world (arm z is mostly base-independent)
    base_xy = inputs[arm_mask][:, :2]
    ee_dx = ee_world[:, 0] - base_xy[:, 0]
    ee_dy = ee_world[:, 1] - base_xy[:, 1]
    tgt_dx = tgt_world[:, 0] - base_xy[:, 0]
    tgt_dy = tgt_world[:, 1] - base_xy[:, 1]
    arm_raw[:, 14] = c * ee_dx + s * ee_dy      # ee_x in base frame
    arm_raw[:, 15] = -s * ee_dx + c * ee_dy      # ee_y in base frame
    arm_raw[:, 16] = ee_world[:, 2]               # ee_z (keep world)
    arm_raw[:, 17] = c * tgt_dx + s * tgt_dy      # target_x in base frame
    arm_raw[:, 18] = -s * tgt_dx + c * tgt_dy      # target_y in base frame
    arm_raw[:, 19] = tgt_world[:, 2]               # target_z (keep world)
    arm_inputs = arm_raw
    arm_names = list(ARM_FEATURE_NAMES)

    # Drop requested arm features
    if args.drop_arm:
        drop_idxs = []
        for name in args.drop_arm:
            if name in arm_names:
                drop_idxs.append(arm_names.index(name))
            else:
                print(f"WARNING: unknown arm feature '{name}', skipping. "
                      f"Available: {arm_names}")
        if drop_idxs:
            keep = [i for i in range(arm_inputs.shape[1]) if i not in drop_idxs]
            dropped = [arm_names[i] for i in drop_idxs]
            arm_inputs = arm_inputs[:, keep]
            arm_names = [arm_names[i] for i in keep]
            print(f"Dropped arm features: {dropped}  → {arm_inputs.shape[1]} dims")

    arm_torques = labels[arm_mask][:, 4:]               # raw expert torques (7)
    if args.arm_full_torque:
        arm_labels = arm_torques                          # full torques (includes gravity)
        label_desc = "full expert torques (includes gravity)"
    else:
        tau_gravity = inputs[arm_mask][:, 37:44]          # gravity comp from expert (7)
        arm_labels = arm_torques - tau_gravity             # residual torques
        label_desc = "residual torques (expert - gravity comp)"
    arm_metadata = metadata[arm_mask]
    arm_path = os.path.join(out_dir, "dataset_arm.npz")
    np.savez(arm_path, inputs=arm_inputs, labels=arm_labels, metadata=arm_metadata,
             feature_names=np.array(arm_names))
    print(f"Saved arm dataset:  {arm_path}  inputs={arm_inputs.shape}  labels={arm_labels.shape}")
    print(f"  Labels: {label_desc}")
    print(f"  Features: {arm_names}")

    if not args.train:
        return

    # --- Train both policies ---
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_regressor.py")
    common_args = [
        "--epochs", str(args.epochs),
        "--hidden_dim", str(args.hidden_dim),
        "--num_layers", str(args.num_layers),
        "--activation", args.activation,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_workers", str(args.num_workers),
        "--early_stop", str(args.early_stop),
        "--patience", str(args.patience),
    ]
    if args.mps:
        common_args.append("--mps")

    for name, path, mode in [("base", base_path, "base"), ("arm", arm_path, "arm")]:
        print(f"\n{'='*60}")
        print(f"Training {name} policy: {path}")
        print(f"{'='*60}")
        cmd = [sys.executable, train_script, "--data", path, "--mode", mode] + common_args
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"ERROR: {name} training failed (exit code {ret.returncode})")
            sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
