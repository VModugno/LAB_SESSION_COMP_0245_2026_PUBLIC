#!/usr/bin/env python3
"""DAgger data collection: roll out learned policies while recording expert labels.

Collects DAgger data for both the base-navigation and arm-reaching phases.
At each timestep the script records (observation, expert_action) — what the
expert controller would command at the state the learned policy actually visits.

Output: two files, ``{stem}_base.npz`` and ``{stem}_arm.npz``, whose format
matches the split datasets produced by ``split_dataset.py``.

Usage:
  python dagger_collect.py \
    --arm_checkpoint arm.pt --base_checkpoint base.pt \
    --num_targets 20 --out dagger_round1.npz --headless \
    --beta 1.0 --arm_full_torque

  # Merge with original dataset (stem-based: looks for foo_base.npz / foo_arm.npz)
  python dagger_collect.py \
    --arm_checkpoint arm.pt --base_checkpoint base.pt \
    --num_targets 20 --out merged.npz --headless \
    --append_to data/dataset.npz --arm_full_torque
"""

import argparse
import os
import sys

if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np
import pinocchio as pin
from simulation_and_control import MotorCommands

from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, ARM_REACH_LIMIT,
    compute_base_goal, BASE_NAV_TIMEOUT,
)
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState,
    arm_controller, base_controller, configure_gripper_fingers, quaternion2bearing,
)
from run_policy import load_policy, predict
from run_policy_sim import build_base_observation, build_arm_observation, WHEEL_RADIUS, WHEEL_BASE_WIDTH
from evaluate import generate_targets
from split_dataset import ARM_FEATURE_NAMES, BASE_FEATURE_NAMES


def _out_paths(out_arg):
    """Derive ``_base.npz`` and ``_arm.npz`` paths from the ``--out`` stem."""
    stem, _ = os.path.splitext(out_arg)
    return f"{stem}_base.npz", f"{stem}_arm.npz"


def collect_dagger_episode(args, target_ee_pos, target_id,
                           base_model, base_mean, base_std,
                           base_label_mean, base_label_std, base_feature_names,
                           arm_model, arm_mean, arm_std,
                           arm_label_mean, arm_label_std, arm_feature_names,
                           device, has_base_phases=False):
    """Run one episode (base nav -> arm reach).

    Returns ``{"base": (inputs, labels, meta) | None, "arm": (inputs, labels, meta) | None}``.
    """

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, target_ee_pos)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    # Arm expert controller state
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])

    # Base expert controller state (separate from the policy — updated by base_controller)
    expert_base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
    base_goal_pos, base_goal_bearing = compute_base_goal(target_ee_pos)

    # Phase tracking (same logic as run_policy_sim.py)
    phase = "base"
    wheel_lock_pos = None

    # Settle detection for phase switch
    best_ee_in_base = np.inf
    steps_since_ee_improved = 0
    SETTLE_PATIENCE = 500
    SETTLE_SPEED_THRESH = 0.05

    # Rotate/drive sub-phase state (persisted across steps for hysteresis)
    base_sub_phase = "rotate"

    # DAgger data buffers — separate for base and arm
    base_obs_list = []
    base_label_list = []
    base_meta_list = []
    arm_obs_list = []
    arm_label_list = []
    arm_meta_list = []

    step = 0
    base_step = 0
    arm_step = 0
    rng = np.random.RandomState()

    while step < args.max_steps:
        # ---- Sense ----
        try:
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            base_lin_vel = np.array(sim.GetBaseLinVelocity())
            base_ang_vel = np.array(sim.GetBaseAngVelocity())
        except Exception as e:
            print(f"  Sim crashed reading sensors at step={step}: {e}")
            break

        # Forward kinematics
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        # ---- Phase transition (base -> arm) ----
        if phase == "base":
            base_speed = np.linalg.norm(base_lin_vel[:2])
            switch = False
            switch_reason = ""

            if args.base_threshold is not None:
                base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
                if base_xy_dist < args.base_threshold:
                    switch = True
                    switch_reason = f"xy_dist={base_xy_dist:.4f}m"
            else:
                if ee_dist_error < best_ee_in_base - 0.01:
                    best_ee_in_base = ee_dist_error
                    steps_since_ee_improved = 0
                else:
                    steps_since_ee_improved += 1
                settled = (steps_since_ee_improved >= SETTLE_PATIENCE
                           and base_speed < SETTLE_SPEED_THRESH
                           and current_time > 2.0)
                if settled:
                    switch = True
                    switch_reason = "settle"

            # Force switch on timeout so we always collect arm data
            if not switch and current_time >= args.base_timeout:
                switch = True
                switch_reason = f"timeout ({args.base_timeout:.0f}s)"

            if switch:
                wheel_lock_pos = q_mes[0:4].copy()
                # Lock base with constraint (same as run_policy_sim.py)
                _p = sim.GetPyBulletClient()
                _base_pos_ori = _p.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
                _p.createConstraint(
                    sim.bot[0].bot_pybullet, -1, -1, -1, _p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0], list(_base_pos_ori[0]),
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=list(_base_pos_ori[1]))
                print(f"  Phase switch at t={current_time:.2f}s  ee_err={ee_dist_error:.4f}m  reason={switch_reason}")
                phase = "arm"

        # ---- Compute action ----
        if phase == "base":
            # Build base observation (without phase for both legacy and phase-output models)
            obs_feature_names = [n for n in base_feature_names if n != "phase"] \
                if has_base_phases else base_feature_names
            obs_base = build_base_observation(base_pos, base_bearing, base_lin_vel,
                                              base_ang_vel, target_ee_pos,
                                              feature_names=obs_feature_names)

            # Query expert base controller at the policy-visited state
            expert_wheel_cmds = base_controller(
                expert_base_state, base_pos, base_bearing,
                base_goal_pos, base_goal_bearing, current_time, base_params,
            )
            # Convert expert wheel cmds [rw, lw, lw, rw] -> [v_cmd, w_cmd]
            rw_expert = expert_wheel_cmds[0]
            lw_expert = expert_wheel_cmds[1]
            v_cmd_expert = (rw_expert + lw_expert) / 2.0 * WHEEL_RADIUS
            w_cmd_expert = (rw_expert - lw_expert) / WHEEL_BASE_WIDTH * WHEEL_RADIUS

            if has_base_phases:
                # Phase as input feature with hysteresis
                delta_base = target_ee_pos[:2] - base_pos[:2]
                heading_to_target = np.arctan2(delta_base[1], delta_base[0])
                heading_err = np.arctan2(np.sin(heading_to_target - base_bearing),
                                         np.cos(heading_to_target - base_bearing))
                abs_herr = abs(heading_err)
                if base_sub_phase == "rotate" and abs_herr < args.heading_threshold:
                    base_sub_phase = "drive"
                elif base_sub_phase == "drive" and abs_herr >= args.heading_threshold * 2:
                    base_sub_phase = "rotate"
                phase_val = 0.0 if base_sub_phase == "rotate" else 1.0
                obs_base = np.append(obs_base, phase_val)

            if args.expert_base:
                # Expert base mode: skip policy, use expert directly
                policy_wheel_cmds = expert_wheel_cmds
            else:
                vw = predict(base_model, base_mean, base_std, obs_base,
                             device=device,
                             label_mean=base_label_mean, label_std=base_label_std)

                # Hard output masking at inference
                if has_base_phases:
                    if base_sub_phase == "rotate":
                        v_cmd_policy = 0.0
                        w_cmd_policy = vw[1]
                    else:  # drive
                        v_cmd_policy = max(vw[0], 0.0)  # forward only
                        w_cmd_policy = 0.0
                else:
                    v_cmd_policy = vw[0] if abs(vw[0]) > 0.02 else 0.0
                    w_cmd_policy = vw[1] if abs(vw[1]) > 0.05 else 0.0

                v_cmd_policy = np.clip(v_cmd_policy, -0.25, 0.25)
                w_cmd_policy = np.clip(w_cmd_policy, -1.0, 1.0)
                lw_policy = (v_cmd_policy - w_cmd_policy * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
                rw_policy = (v_cmd_policy + w_cmd_policy * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
                policy_wheel_cmds = np.array([rw_policy, lw_policy, lw_policy, rw_policy])

            # Record base DAgger data (obs WITHOUT phase feature)
            base_obs_list.append(obs_base if not has_base_phases else obs_base[:-1])
            base_label_list.append([v_cmd_expert, w_cmd_expert])
            base_meta_list.append([target_id, base_step])
            base_step += 1

            # Beta selection: execute policy or expert
            if rng.random() < args.beta:
                wheel_cmds = policy_wheel_cmds
            else:
                wheel_cmds = expert_wheel_cmds

            # Expert arm hold during base phase
            torque_arm, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11],
                q_des_arm, q_init_arm, target_ee_pos, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper,
                time_step, arm_params, reach_active=False,
            )

        elif phase == "arm":
            # Build arm observation
            obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                                            base_pos=base_pos, base_bearing=base_bearing,
                                            feature_names=arm_feature_names)

            # Query learned policy
            policy_torque = predict(arm_model, arm_mean, arm_std, obs_arm,
                                    device=device,
                                    label_mean=arm_label_mean, label_std=arm_label_std)

            # Query expert at the policy-visited state (the DAgger insight)
            expert_torque, q_des_arm, qd_des_arm = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11],
                q_des_arm, q_init_arm, target_ee_pos, ee_pos,
                ee_frame_id, shoulder_frame_id, arm_lower, arm_upper,
                time_step, arm_params, reach_active=True,
            )

            # Record data
            if args.arm_full_torque:
                label = expert_torque.copy()
            else:
                tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
                label = expert_torque - tau_g[6:13]

            arm_obs_list.append(obs_arm)
            arm_label_list.append(label)
            arm_meta_list.append([target_id, arm_step])
            arm_step += 1

            # Choose action to execute (beta schedule)
            if rng.random() < args.beta:
                raw = policy_torque
                if args.arm_damping > 0:
                    raw = raw - args.arm_damping * qd_mes[4:11]
                torque_arm = np.clip(raw, -30.0, 30.0)
            else:
                torque_arm = expert_torque

            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        try:
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"  Sim crashed at step={step}: {e}")
            break

        current_time += time_step
        step += 1

        if step % 2000 == 0:
            print(f"  t={current_time:.2f}s [{phase}] ee_err={ee_dist_error:.4f}m  "
                  f"base_samples={len(base_obs_list)}  arm_samples={len(arm_obs_list)}")

    result = {}
    if base_obs_list:
        result["base"] = (np.array(base_obs_list), np.array(base_label_list), np.array(base_meta_list))
    else:
        result["base"] = None
    if arm_obs_list:
        result["arm"] = (np.array(arm_obs_list), np.array(arm_label_list), np.array(arm_meta_list))
    else:
        result["arm"] = None
    return result


def _merge_and_save(new_inputs, new_labels, new_meta, feature_names, out_path, append_path):
    """Concatenate with existing data if append_path exists, then save."""
    if append_path and os.path.exists(append_path):
        existing = np.load(append_path)
        orig_inputs = existing["inputs"]
        orig_labels = existing["labels"]
        orig_meta = existing["metadata"]
        print(f"  Merging with {append_path}: {orig_inputs.shape[0]} existing samples")

        if orig_inputs.shape[1] != new_inputs.shape[1]:
            print(f"  ERROR: input dim mismatch: existing={orig_inputs.shape[1]} vs new={new_inputs.shape[1]}")
            sys.exit(1)
        if orig_labels.shape[1] != new_labels.shape[1]:
            print(f"  ERROR: label dim mismatch: existing={orig_labels.shape[1]} vs new={new_labels.shape[1]}")
            sys.exit(1)

        new_inputs = np.concatenate([orig_inputs, new_inputs])
        new_labels = np.concatenate([orig_labels, new_labels])
        new_meta = np.concatenate([orig_meta, new_meta])
        print(f"  Merged total: {new_inputs.shape[0]} samples")

    np.savez(out_path, inputs=new_inputs, labels=new_labels, metadata=new_meta,
             feature_names=np.array(feature_names))
    print(f"  Saved: {out_path}  inputs={new_inputs.shape}  labels={new_labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="DAgger data collection")
    parser.add_argument("--arm_checkpoint", type=str, required=True,
                        help="Path to trained arm policy .pt checkpoint")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to trained base policy .pt checkpoint (not needed with --expert_base)")
    parser.add_argument("--num_targets", type=int, default=20,
                        help="Number of random targets to collect (default: 20)")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Max sim steps per episode (default: 100000)")
    parser.add_argument("--out", type=str, default="dagger_data.npz",
                        help="Output stem — saves {stem}_base.npz and {stem}_arm.npz (default: dagger_data.npz)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no GUI)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Probability of executing policy action vs expert "
                             "(1.0=pure policy rollout, 0.0=pure expert, default: 1.0)")
    parser.add_argument("--base_threshold", type=float, default=0.75,
                        help="XY distance for base->arm phase switch (default: 0.75)")
    parser.add_argument("--base_timeout", type=float, default=60.0,
                        help="Force phase switch after this many seconds if base hasn't "
                             "reached threshold (default: 60.0)")
    parser.add_argument("--arm_full_torque", action="store_true", default=True,
                        help="Record full expert torques including gravity (default: True)")
    parser.add_argument("--no_arm_full_torque", action="store_false", dest="arm_full_torque",
                        help="Record residual torques (subtract gravity)")
    parser.add_argument("--append_to", type=str, default=None,
                        help="Stem of existing split .npz files to merge with "
                             "(looks for {stem}_base.npz and {stem}_arm.npz; originals not modified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for target generation (default: 42)")
    parser.add_argument("--arm_damping", type=float, default=0.0,
                        help="Velocity damping coefficient for arm joints applied to policy "
                             "torques (default: 0, match run_policy_sim e.g. 30.0)")
    parser.add_argument("--expert_base", action="store_true",
                        help="Use expert base controller instead of learned policy "
                             "(perfect base nav, useful for arm-only DAgger)")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple Metal (MPS) GPU for inference")
    parser.add_argument("--heading_threshold", type=float, default=0.15,
                        help="Heading error (rad) for rotate/drive sub-phase transition "
                             "(only used when base checkpoint has 'phase' feature, default: 0.15)")
    args = parser.parse_args()

    if not args.expert_base and args.base_checkpoint is None:
        parser.error("--base_checkpoint is required unless --expert_base is set")

    # Device
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load policies
    if args.base_checkpoint is not None:
        base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
            load_policy(args.base_checkpoint, device=device)
    else:
        base_model = base_mean = base_std = base_lm = base_ls = base_fnames = None
    arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
        load_policy(args.arm_checkpoint, device=device)

    # Detect phase-as-input model (has 'phase' in feature names)
    has_base_phases = base_fnames is not None and "phase" in list(base_fnames)

    phase_info = f"input (thr={args.heading_threshold})" if has_base_phases else "no"
    print(f"Base: {'EXPERT' if args.expert_base else args.base_checkpoint}"
          f"  phases={phase_info}")
    print(f"Arm policy:  {args.arm_checkpoint}")
    print(f"Device: {device}  Beta: {args.beta}  Targets: {args.num_targets}")
    print(f"Full torque labels: {args.arm_full_torque}")

    # Generate targets
    targets = generate_targets(args.num_targets, seed=args.seed)

    all_base_inputs, all_base_labels, all_base_meta = [], [], []
    all_arm_inputs, all_arm_labels, all_arm_meta = [], [], []

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{args.num_targets}] Target: {target}")
        result = collect_dagger_episode(
            args, target, target_id=i,
            base_model=base_model, base_mean=base_mean, base_std=base_std,
            base_label_mean=base_lm, base_label_std=base_ls,
            base_feature_names=base_fnames,
            arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
            arm_label_mean=arm_lm, arm_label_std=arm_ls,
            arm_feature_names=arm_fnames,
            device=device,
            has_base_phases=has_base_phases,
        )
        base_data = result["base"]
        arm_data = result["arm"]
        n_base = len(base_data[0]) if base_data is not None else 0
        n_arm = len(arm_data[0]) if arm_data is not None else 0
        print(f"  Collected {n_base} base + {n_arm} arm samples")
        if base_data is not None:
            all_base_inputs.append(base_data[0])
            all_base_labels.append(base_data[1])
            all_base_meta.append(base_data[2])
        if arm_data is not None:
            all_arm_inputs.append(arm_data[0])
            all_arm_labels.append(arm_data[1])
            all_arm_meta.append(arm_data[2])

    if not all_base_inputs and not all_arm_inputs:
        print("\nNo data collected. Exiting.")
        sys.exit(1)

    # Derive output paths
    out_base_path, out_arm_path = _out_paths(args.out)

    # Derive append paths (stem-based, same convention as --out)
    append_base_path, append_arm_path = (None, None)
    if args.append_to:
        append_base_path, append_arm_path = _out_paths(args.append_to)

    # Resolve feature names (exclude "phase" — it's a label, not input, for phase-output models)
    base_fnames_out = list(base_fnames) if base_fnames is not None else list(BASE_FEATURE_NAMES)
    base_fnames_out = [n for n in base_fnames_out if n != "phase"]
    arm_fnames_out = list(arm_fnames) if arm_fnames is not None else list(ARM_FEATURE_NAMES)

    # Save base data
    if all_base_inputs:
        base_inputs = np.concatenate(all_base_inputs)
        base_labels = np.concatenate(all_base_labels)
        base_meta = np.concatenate(all_base_meta)
        print(f"\nBase DAgger samples: {base_inputs.shape[0]}  "
              f"inputs={base_inputs.shape}  labels={base_labels.shape}")
        _merge_and_save(base_inputs, base_labels, base_meta, base_fnames_out,
                        out_base_path, append_base_path)
    else:
        print("\nNo base-phase data collected.")

    # Save arm data
    if all_arm_inputs:
        arm_inputs = np.concatenate(all_arm_inputs)
        arm_labels = np.concatenate(all_arm_labels)
        arm_meta = np.concatenate(all_arm_meta)
        print(f"\nArm DAgger samples: {arm_inputs.shape[0]}  "
              f"inputs={arm_inputs.shape}  labels={arm_labels.shape}")
        _merge_and_save(arm_inputs, arm_labels, arm_meta, arm_fnames_out,
                        out_arm_path, append_arm_path)
    else:
        print("\nNo arm-phase data collected.")


if __name__ == "__main__":
    main()
