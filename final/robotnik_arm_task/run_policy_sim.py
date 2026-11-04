#!/usr/bin/env python3
"""Run a trained BC policy in PyBullet simulation to reach a target endpoint.

Supports two modes:
  1. Single-model mode (--checkpoint): one policy predicts all 11 outputs.
  2. Split-model mode (--base_checkpoint + --arm_checkpoint): base policy navigates,
     then arm policy reaches, with expert PD arm-hold during base navigation.
"""

import argparse
import os
import sys
import time
from cmath import phase

# Parse --headless early so the env var is set before module-level reads in robotnik_arm_task
if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np
import torch
import pinocchio as pin
from simulation_and_control import pb, MotorCommands
from robotnik_arm_task import (
    setup_simulation, setup_kinematics, add_target_visual,
    FIXED_FINGER_ANGLES, REACH_TOLERANCE, ARM_REACH_LIMIT,
    GRIP_TOLERANCE, DROP_ZONE_DEFAULT, GripState,
    NUM_DROP_ZONES, DROP_ZONES_DEFAULT, DROP_ZONE_COLORS,
    spawn_ball, add_drop_zone_visual, find_ee_link_id,
    auto_grip, auto_release,
    shortest_path_expert_sequential,
    shortest_path_expert_multi_drop,
    build_selector_scene_multi_drop,
    generate_random_drop_zones,
)
from robotnik_arm_control import (
    configure_gripper_fingers, quaternion2bearing,
)
from run_policy import load_policy, predict
from regressors import build_mlp
from activations import LEGACY_ACTIVATION


def load_gravity_comp(path, device="cpu"):
    """Load gravity compensation network from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hidden_dim = ckpt["hidden_dim"]
    num_layers = ckpt["num_layers"]
    activation = ckpt.get("activation", LEGACY_ACTIVATION)
    model = build_mlp(7, 7, hidden_dim, num_layers, activation)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["q_mean"], ckpt["q_std"], ckpt["tau_mean"], ckpt["tau_std"]


def predict_gravity(grav_model, q_mean, q_std, tau_mean, tau_std, q_arm, device="cpu"):
    """Predict gravity compensation torques from joint angles."""
    q_norm = (q_arm - q_mean) / q_std
    x = torch.tensor(q_norm, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = grav_model(x).cpu().numpy()[0]
    return pred * tau_std + tau_mean

# Wheel kinematic constants (from BaseControllerParams)
WHEEL_RADIUS = 0.11
WHEEL_BASE_WIDTH = 0.66
# Approximate height of arm shoulder joint above ground (from robot URDF)
SHOULDER_HEIGHT = 0.75


def build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                      q_mes, qd_mes, ee_pos, target_ee_pos, tau_gravity):
    """Assemble the 44-dim observation vector matching the training data format."""
    return np.concatenate([
        base_pos[:2],               # 2: base_pos_x, base_pos_y
        [base_bearing],             # 1: base_bearing
        base_lin_vel,               # 3: base_lin_vel_x/y/z
        base_ang_vel,               # 3: base_ang_vel_x/y/z
        q_mes[0:4],                 # 4: q_base (wheel angles)
        qd_mes[0:4],               # 4: qd_base (wheel velocities)
        q_mes[4:11],               # 7: q_arm
        qd_mes[4:11],              # 7: qd_arm
        ee_pos,                     # 3: ee_pos_x/y/z
        target_ee_pos,              # 3: target_x/y/z
        tau_gravity,                # 7: gravity compensation torques
    ])  # total: 44


def build_base_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                           target_ee_pos, feature_names=None):
    """Assemble base policy observation, optionally filtered by feature_names.

    Linear velocities are rotated into the base frame so that 'forward' and
    'lateral' are always relative to the robot's heading.
    """
    delta = target_ee_pos[:2] - base_pos[:2]
    dist_to_target = np.linalg.norm(delta)
    heading_to_target = np.arctan2(delta[1], delta[0])
    heading_error = np.arctan2(np.sin(heading_to_target - base_bearing),
                               np.cos(heading_to_target - base_bearing))

    # Rotate world-frame linear velocity into base frame
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    fwd_vel = c * base_lin_vel[0] + s * base_lin_vel[1]   # forward
    lat_vel = -s * base_lin_vel[0] + c * base_lin_vel[1]   # lateral (left)

    # Target in base frame (Cartesian, position-agnostic)
    dx = target_ee_pos[0] - base_pos[0]
    dy = target_ee_pos[1] - base_pos[1]
    target_in_base_x = c * dx + s * dy    # forward distance
    target_in_base_y = -s * dx + c * dy   # lateral distance

    # All possible base features
    ALL_BASE = {
        "base_pos_x": base_pos[0],
        "base_pos_y": base_pos[1],
        "base_bearing": base_bearing,
        "base_lin_vel_x": base_lin_vel[0],
        "base_lin_vel_y": base_lin_vel[1],
        "base_lin_vel_z": base_lin_vel[2],
        "base_ang_vel_x": base_ang_vel[0],
        "base_ang_vel_y": base_ang_vel[1],
        "base_ang_vel_z": base_ang_vel[2],
        "target_x": target_ee_pos[0],
        "target_y": target_ee_pos[1],
        "target_z": target_ee_pos[2],
        "dist_to_target": dist_to_target,
        "heading_error": heading_error,
        "fwd_vel": fwd_vel,
        "lat_vel": lat_vel,
        "target_in_base_x": target_in_base_x,
        "target_in_base_y": target_in_base_y,
    }

    if feature_names is None:
        # Default: base-frame velocities + Cartesian target in base frame (5-dim, fully generalizable)
        feature_names = [
            "fwd_vel", "lat_vel",
            "base_ang_vel_z",
            "target_in_base_x", "target_in_base_y",
        ]

    return np.array([ALL_BASE[name] for name in feature_names])


def build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                          base_pos=None, base_bearing=None,
                          feature_names=None):
    """Assemble arm policy observation, optionally filtered by feature_names.

    If base_pos and base_bearing are provided, ee_pos and target_ee_pos are
    expressed in the base frame (translation + yaw rotation), making the
    observation independent of the base's world-frame pose.
    """
    if base_pos is not None and base_bearing is not None:
        c, s = np.cos(base_bearing), np.sin(base_bearing)
        # Rotate xy into base frame, keep z as world (base z not in training data)
        base_xy = base_pos[:2]
        ee_dx, ee_dy = ee_pos[0] - base_xy[0], ee_pos[1] - base_xy[1]
        tgt_dx, tgt_dy = target_ee_pos[0] - base_xy[0], target_ee_pos[1] - base_xy[1]
        ee_rel = np.array([c * ee_dx + s * ee_dy, -s * ee_dx + c * ee_dy, ee_pos[2]])
        tgt_rel = np.array([c * tgt_dx + s * tgt_dy, -s * tgt_dx + c * tgt_dy, target_ee_pos[2]])
    else:
        ee_rel = ee_pos
        tgt_rel = target_ee_pos

    ALL_ARM = {}
    for i in range(7):
        ALL_ARM[f"q_arm_{i+1}"] = q_mes[4 + i]
    for i in range(7):
        ALL_ARM[f"qd_arm_{i+1}"] = qd_mes[4 + i]
    ALL_ARM["ee_pos_x"] = ee_rel[0]
    ALL_ARM["ee_pos_y"] = ee_rel[1]
    ALL_ARM["ee_pos_z"] = ee_rel[2]
    ALL_ARM["target_x"] = tgt_rel[0]
    ALL_ARM["target_y"] = tgt_rel[1]
    ALL_ARM["target_z"] = tgt_rel[2]

    if feature_names is None:
        feature_names = (
            [f"q_arm_{i}" for i in range(1, 8)]
            + [f"qd_arm_{i}" for i in range(1, 8)]
            + ["ee_pos_x", "ee_pos_y", "ee_pos_z"]
            + ["target_x", "target_y", "target_z"]
        )

    return np.array([ALL_ARM[name] for name in feature_names])


def run_sim(args, target_ee_pos, device, split_mode, single_mode,
            # single-mode policy
            model=None, mean=None, std=None,
            label_mean=None, label_std=None,
            # split-mode policies
            base_model=None, base_mean=None, base_std=None,
            base_label_mean=None, base_label_std=None, base_feature_names=None,
            has_base_phases=False,
            arm_model=None, arm_mean=None, arm_std=None,
            arm_label_mean=None, arm_label_std=None, arm_feature_names=None,
            # gravity compensation
            grav_model=None, grav_q_mean=None, grav_q_std=None,
            grav_tau_mean=None, grav_tau_std=None):
    """Run one simulation attempt. Returns True if finished normally, False if crashed."""

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, target_ee_pos)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    print("=" * 60)
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Initial EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Error (m)  : {np.linalg.norm(target_ee_pos - init_ee_pos):.4f}")
    if split_mode:
        if args.base_threshold is not None:
            print(f"Mode               : SPLIT (base -> arm at XY dist < {args.base_threshold}m)")
        else:
            print(f"Mode               : SPLIT (base -> arm on settle)")
    else:
        print(f"Mode               : SINGLE")
    print("=" * 60)

    best_ee_error = np.inf
    step = 0
    phase = "base" if split_mode else "single"
    phase_switched = False

    # Only stop once the EE has stayed inside the goal for HOLD_TIME seconds in a row;
    # drifting out resets the timer and the simulation keeps running
    HOLD_TIME = 1.0
    first_reach_time = None
    in_goal_since = None
    times_drifted_out = 0
    held = False

    # Phase switch tracking: settle detection

    # Per-phase tracking
    base_best_ee = np.inf
    base_final_ee = np.inf
    base_xy_at_switch = None
    base_xy_at_best = None
    base_bearing_at_best = None
    arm_best_ee = np.inf
    arm_final_ee = np.inf
    switch_time = None

    # Rotate/drive sub-phase state (persisted across steps for hysteresis)
    base_sub_phase = "rotate"  # start in rotate

    # Track last known good state for the final report on crash
    ee_pos = init_ee_pos.copy()
    ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)
    base_pos = np.array(sim.GetBasePosition())

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
            print(f"\nSim crashed reading sensors at t={current_time:.2f}s  step={step}: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "target": target_ee_pos.tolist(), "phase_switched": phase_switched,
                    "switch_time": switch_time, "base_best_ee": base_best_ee,
                    "base_final_ee": base_final_ee,
                    "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee}

        # Forward kinematics for EE position
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        tau_gravity_arm = np.zeros(7)  # not available at inference

        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error

        # ---- Phase transition check (split mode only) ----
        if phase == "base":
            base_speed = np.linalg.norm(base_lin_vel[:2])
            switch = False
            switch_reason = ""

            base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
            if base_xy_dist < args.base_threshold:
                switch = True
                switch_reason = f"xy_dist={base_xy_dist:.4f}m"

            # Track base phase metrics
            if ee_dist_error < base_best_ee:
                base_best_ee = ee_dist_error
                base_xy_at_best = base_pos[:2].copy()
                base_bearing_at_best = base_bearing
            base_final_ee = ee_dist_error

            if switch:
                phase_switched = True
                switch_time = current_time
                base_xy_at_switch = base_pos[:2].copy()
                wheel_lock_pos = q_mes[0:4].copy()
                # TODO: Remove base lock constraint — no controller code allowed in final version.
                #       For now, used to test arm policy without base drift.
                _p = sim.GetPyBulletClient()
                _base_pos_ori = _p.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
                base_lock_cid = _p.createConstraint(
                    sim.bot[0].bot_pybullet, -1, -1, -1, _p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0], list(_base_pos_ori[0]),
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=list(_base_pos_ori[1]))
                print(f">>> Phase switch: base -> arm at t={current_time:.2f}s  {switch_reason}")
                if arm_model is None:
                    # No arm policy (base_only mode) — stop here
                    break
                phase = "arm"

        # ---- Compute action based on mode/phase ----
        if phase == "single":
            obs = build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                                    q_mes, qd_mes, ee_pos, target_ee_pos, tau_gravity_arm)
            action = predict(model, mean, std, obs, device=device,
                             label_mean=label_mean, label_std=label_std)
            wheel_cmds = np.clip(action[:4], -0.25 / WHEEL_RADIUS, 0.25 / WHEEL_RADIUS)
            torque_arm = np.clip(action[4:], -30.0, 30.0)

        elif phase == "base":
            # Build base observation (without phase input)
            obs_feature_names = [n for n in base_feature_names if n != "phase"] \
                if has_base_phases else base_feature_names
            obs_base = build_base_observation(base_pos, base_bearing, base_lin_vel,
                                              base_ang_vel, target_ee_pos,
                                              feature_names=obs_feature_names)

            # Heading-based phase switching
            if has_base_phases:
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

            vw = predict(base_model, base_mean, base_std, obs_base,
                         device=device,
                         label_mean=base_label_mean, label_std=base_label_std)

            # Hard output masking at inference
            if has_base_phases:
                if base_sub_phase == "rotate":
                    v_cmd = 0.0
                    w_cmd = vw[1]
                else:  # drive
                    v_cmd = max(vw[0], 0.0)  # forward only
                    w_cmd = 0.0
            else:
                v_cmd = vw[0] if abs(vw[0]) > 0.02 else 0.0
                w_cmd = vw[1] if abs(vw[1]) > 0.05 else 0.0

            v_cmd = np.clip(v_cmd, -0.25, 0.25)
            w_cmd = np.clip(w_cmd, -1.0, 1.0)
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw])

            # Arm hold: run arm policy with target=current EE so it only outputs gravity torques
            if arm_model is not None:
                obs_arm_hold = build_arm_observation(q_mes, qd_mes, ee_pos, ee_pos,
                                                     base_pos=base_pos, base_bearing=base_bearing,
                                                     feature_names=arm_feature_names)
                torque_arm = predict(arm_model, arm_mean, arm_std, obs_arm_hold,
                                     device=device,
                                     label_mean=arm_label_mean, label_std=arm_label_std)
                torque_arm = np.clip(torque_arm, -20.0, 20.0)
            else:
                torque_arm = np.zeros(7)

        elif phase == "arm":
            # Arm policy
            obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                                            base_pos=base_pos, base_bearing=base_bearing,
                                            feature_names=arm_feature_names)
            raw_torque = predict(arm_model, arm_mean, arm_std, obs_arm,
                                 device=device,
                                 label_mean=arm_label_mean, label_std=arm_label_std)
            # Add gravity compensation if using learned gravity network
            if grav_model is not None:
                tau_grav = predict_gravity(grav_model, grav_q_mean, grav_q_std,
                                           grav_tau_mean, grav_tau_std,
                                           q_mes[4:11], device=device)
                raw_torque = raw_torque + tau_grav
            if args.arm_damping > 0:
                raw_torque = raw_torque - args.arm_damping * qd_mes[4:11]
            torque_arm = np.clip(raw_torque, -20.0, 20.0)

            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)
            # Track arm phase metrics
            if ee_dist_error < arm_best_ee:
                arm_best_ee = ee_dist_error
            arm_final_ee = ee_dist_error

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        try:
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"\nSim crashed at t={current_time:.2f}s  step={step}  phase={phase}")
            print(f"  Last base_pos: {np.round(base_pos, 3)}  wheel_cmds: {np.round(wheel_cmds, 3)}")
            print(f"  Error: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time,
                    "target": target_ee_pos.tolist(), "phase_switched": phase_switched,
                    "switch_time": switch_time, "base_best_ee": base_best_ee,
                    "base_final_ee": base_final_ee,
                    "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
                    "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
                    "base_bearing_at_best": base_bearing_at_best,
                    "arm_best_ee": arm_best_ee, "arm_final_ee": arm_final_ee}

        if not args.headless:
            if dyn_model.visualizer:
                for index in range(len(sim.bot)):
                    q = sim.GetMotorAngles(index)
                    dyn_model.DisplayModel(q)

            keys = sim.GetPyBulletClient().getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
                print("User quit.")
                break

        # ---- Check if target reached ----
        in_goal = ee_dist_error < REACH_TOLERANCE and current_time > 8.0
        if in_goal and in_goal_since is None:
            in_goal_since = current_time
            if first_reach_time is None:
                first_reach_time = current_time
            print(f"Entered goal ({REACH_TOLERANCE*1000:.0f}mm) at t={current_time:.2f}s, "
                  f"needs to stay {HOLD_TIME:.1f}s...")
        elif not in_goal and in_goal_since is not None:
            times_drifted_out += 1
            print(f"Drifted out of goal at t={current_time:.2f}s after "
                  f"{current_time - in_goal_since:.2f}s  EE err: {ee_dist_error:.4f}m, continuing")
            in_goal_since = None
        if in_goal_since is not None and current_time - in_goal_since >= HOLD_TIME:
            held = True
            print(f"Target held within {REACH_TOLERANCE*1000:.0f}mm for {HOLD_TIME:.1f}s "
                  f"at t={current_time:.2f}s!")
            break

        # ---- Logging ----
        if step % 500 == 0:
            phase_str = f" [{phase}]" if split_mode else ""
            if phase == "base" and has_base_phases:
                phase_str = f" [base/{base_sub_phase}]"
            print(f"t={current_time:.2f}s{phase_str} | Base: {np.round(base_pos[:2], 3)} | "
                  f"EE: {np.round(ee_pos, 3)} | EE err: {ee_dist_error:.4f}m | "
                  f"best: {best_ee_error:.4f}m")

        if not args.headless:
            time.sleep(0.001)
        current_time += time_step
        step += 1

    # ---- Final report ----
    reached = held
    print("=" * 60)
    print(f"Final EE Position  : {np.round(ee_pos, 4)}")
    print(f"Target EE Position : {target_ee_pos}")
    print(f"Final Error (m)    : {ee_dist_error:.4f}")
    print(f"Best Error (m)     : {best_ee_error:.4f}")
    if first_reach_time is not None:
        print(f"First reached at   : t={first_reach_time:.2f}s  "
              f"(drifted out {times_drifted_out}x, held {HOLD_TIME:.1f}s: {'Yes' if held else 'No'})")
    print(f"Steps: {step}  Time: {current_time:.2f}s")
    if split_mode:
        print(f"Phase switched     : {'Yes' if phase_switched else 'No (stayed in base phase)'}")
    print("=" * 60)
    return {
        "ok": True,
        "final_err": ee_dist_error,
        "best_err": best_ee_error,
        "reached": reached,
        "first_reach_time": first_reach_time,
        "times_drifted_out": times_drifted_out,
        "steps": step,
        "time": current_time,
        "target": target_ee_pos.tolist(),
        "phase_switched": phase_switched,
        "switch_time": switch_time,
        "base_best_ee": base_best_ee,
        "base_final_ee": base_final_ee,
        "base_xy_at_switch": base_xy_at_switch.tolist() if base_xy_at_switch is not None else None,
        "base_xy_at_best": base_xy_at_best.tolist() if base_xy_at_best is not None else None,
        "base_bearing_at_best": base_bearing_at_best,
        "arm_best_ee": arm_best_ee,
        "arm_final_ee": arm_final_ee,
    }


def load_selector(checkpoint_path, device="cpu"):
    """Load a trained target-selector scorer from checkpoint.

    Returns (model, mean, std).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_mlp(ckpt["input_dim"], ckpt["output_dim"],
                      ckpt["hidden_dim"], ckpt["num_layers"],
                      ckpt.get("activation", "gelu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["input_mean"], ckpt["input_std"]


def predict_selector(sel_model, sel_mean, sel_std, scene_obs, slot_order, device="cpu"):
    """Pick the best target using the scene-level selector.

    Args:
        scene_obs: (SELECTOR_INPUT_DIM,) full scene observation
        slot_order: list of slot indices matching packed order
    Returns:
        int, chosen target slot index
    """
    from robotnik_arm_task import MAX_TARGETS
    obs_part = scene_obs[:-MAX_TARGETS]
    mask = scene_obs[-MAX_TARGETS:]

    obs_norm = (obs_part - sel_mean) / sel_std
    x = np.concatenate([obs_norm, mask])
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sel_model(x_t).squeeze(0).cpu().numpy()  # (MAX_TARGETS,)

    # Mask invalid slots
    logits[mask == 0] = -np.inf
    best_packed_idx = int(np.argmax(logits))
    return slot_order[best_packed_idx]


def compute_base_threshold_for_target(target_ee_pos):
    """Compute the base XY distance threshold for phase switching given a target."""
    dz = abs(target_ee_pos[2] - SHOULDER_HEIGHT)
    if dz < ARM_REACH_LIMIT:
        return 0.75 * np.sqrt(ARM_REACH_LIMIT**2 - dz**2)
    return 0.3


def run_multi_target_sim(args, target_ee_positions, device,
                         split_mode, single_mode,
                         sel_model=None, sel_mean=None, sel_std=None,
                         use_expert_selector=False,
                         **policy_kwargs):
    """Run a multi-target episode using learned (or expert) selector + base/arm policies.

    Args:
        target_ee_positions: list of target positions (any length)
        sel_model, sel_mean, sel_std: selector policy (None if use_expert_selector)
        use_expert_selector: if True, use nearest-neighbor expert instead of learned selector
        **policy_kwargs: forwarded to _run_single_target_inline()

    Returns:
        dict with per-target results and episode-level metrics
    """
    from robotnik_arm_task import (
        shortest_path_expert,
        setup_simulation, setup_kinematics, add_target_visual,
    )
    from robotnik_arm_control import quaternion2bearing, configure_gripper_fingers

    n_targets = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)  # (N, 3)

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid,
                                                      np.array([0.1]*6))

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    for t in target_ee_positions:
        add_target_visual(client, t)

    remaining = set(range(n_targets))
    per_target_results = []
    visit_order = []
    total_steps = 0
    total_time = 0.0

    import pinocchio as pin
    from simulation_and_control import MotorCommands

    print(f"\n{'='*60}")
    print(f"Multi-target episode: {n_targets} targets")
    for i, t in enumerate(target_ee_positions):
        print(f"  Slot {i}: {np.round(t, 3)}")
    print(f"{'='*60}")
    episode_ok = True

    while remaining:
        # --- Decision point: pick next target ---
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        if use_expert_selector:
            chosen_idx, full_order = shortest_path_expert(base_pos, all_targets, remaining)
            print(f"    Optimal order: {full_order} (total dist: shortest path)")
        else:
            from robotnik_arm_task import build_selector_scene
            scene_obs, slot_order = build_selector_scene(base_pos, base_bearing, all_targets, remaining)
            chosen_idx = predict_selector(sel_model, sel_mean, sel_std, scene_obs, slot_order, device)

        current_target = all_targets[chosen_idx]
        remaining.remove(chosen_idx)
        visit_order.append(chosen_idx)

        print(f"\n>>> Selector chose slot {chosen_idx} ({np.round(current_target, 3)}) | "
              f"{len(remaining)} remaining")

        # --- Compute base threshold for this target ---
        args.base_threshold = compute_base_threshold_for_target(current_target)

        # --- Run single-target control loop ---
        result = _run_single_target_inline(
            args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, target_finger_angles,
            lower_limits, upper_limits,
            current_target, device, split_mode, single_mode,
            **policy_kwargs,
        )

        total_steps += result.get("steps", 0)
        total_time += result.get("time", 0.0)

        per_target_results.append({
            "slot": chosen_idx,
            "target": current_target.tolist(),
            **result,
        })

        reached = result.get("reached", False)
        print(f"    Slot {chosen_idx}: final_err={result.get('final_err', float('inf')):.4f}m  "
              f"reached={reached}")

        if not result.get("ok", True):
            episode_ok = False
            break

    # --- Episode summary ---
    reached_count = sum(1 for r in per_target_results if r.get("reached", False))
    print(f"\n{'='*60}")
    print(f"Episode complete: {reached_count}/{n_targets} targets reached")
    print(f"Visit order: {visit_order}")
    print(f"Total steps: {total_steps}  Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    return {
        "ok": episode_ok,
        "targets_reached": reached_count,
        "n_targets": n_targets,
        "visit_order": visit_order,
        "total_steps": total_steps,
        "total_time": total_time,
        "per_target": per_target_results,
    }


def _run_single_target_inline(args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
                               init_joint_angles, target_finger_angles,
                               lower_limits, upper_limits,
                               target_ee_pos, device, split_mode, single_mode,
                               **policy_kwargs):
    """Run one target within a multi-target episode (no sim reset).

    This is a simplified version of run_sim() that doesn't create a new sim.
    """
    import pinocchio as pin
    from simulation_and_control import MotorCommands
    from robotnik_arm_control import quaternion2bearing

    time_step = sim.GetTimeStep()
    current_time = 0.0
    cmd = MotorCommands()

    best_ee_error = np.inf
    step = 0
    phase = "base" if split_mode else "single"
    phase_switched = False
    wheel_lock_pos = None
    base_lock_cid = None

    HOLD_TIME = 1.0
    in_goal_since = None
    held = False

    base_sub_phase = "rotate"

    ee_pos = np.zeros(3)
    ee_dist_error = np.inf
    base_pos = np.array(sim.GetBasePosition())

    # Extract policy components from kwargs
    model = policy_kwargs.get("model")
    mean = policy_kwargs.get("mean")
    std = policy_kwargs.get("std")
    label_mean = policy_kwargs.get("label_mean")
    label_std = policy_kwargs.get("label_std")

    base_model = policy_kwargs.get("base_model")
    base_mean = policy_kwargs.get("base_mean")
    base_std = policy_kwargs.get("base_std")
    base_label_mean = policy_kwargs.get("base_label_mean")
    base_label_std = policy_kwargs.get("base_label_std")
    base_feature_names = policy_kwargs.get("base_feature_names")
    has_base_phases = policy_kwargs.get("has_base_phases", True)

    arm_model = policy_kwargs.get("arm_model")
    arm_mean = policy_kwargs.get("arm_mean")
    arm_std = policy_kwargs.get("arm_std")
    arm_label_mean = policy_kwargs.get("arm_label_mean")
    arm_label_std = policy_kwargs.get("arm_label_std")
    arm_feature_names = policy_kwargs.get("arm_feature_names")

    grav_model = policy_kwargs.get("grav_model")
    grav_q_mean = policy_kwargs.get("grav_q_mean")
    grav_q_std = policy_kwargs.get("grav_q_std")
    grav_tau_mean = policy_kwargs.get("grav_tau_mean")
    grav_tau_std = policy_kwargs.get("grav_tau_std")

    per_target_max_steps = getattr(args, "per_target_max_steps", args.max_steps)

    while step < per_target_max_steps:
        try:
            base_pos = np.array(sim.GetBasePosition())
            base_ori = np.array(sim.GetBaseOrientation())
            base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
            q_mes = np.array(sim.GetMotorAngles(0))
            qd_mes = np.array(sim.GetMotorVelocities(0))
            base_lin_vel = np.array(sim.GetBaseLinVelocity())
            base_ang_vel = np.array(sim.GetBaseAngVelocity())
        except Exception as e:
            print(f"  Sim crashed at t={current_time:.2f}s: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time}

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target_ee_pos - ee_pos)

        tau_gravity_arm = np.zeros(7)

        if ee_dist_error < best_ee_error:
            best_ee_error = ee_dist_error

        # Phase transition (split mode)
        if phase == "base":
            base_xy_dist = np.linalg.norm(base_pos[:2] - target_ee_pos[:2])
            if base_xy_dist < args.base_threshold:
                phase_switched = True
                wheel_lock_pos = q_mes[0:4].copy()
                _p = sim.GetPyBulletClient()
                _base_pos_ori = _p.getBasePositionAndOrientation(sim.bot[0].bot_pybullet)
                base_lock_cid = _p.createConstraint(
                    sim.bot[0].bot_pybullet, -1, -1, -1, _p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0], list(_base_pos_ori[0]),
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=list(_base_pos_ori[1]))
                print(f"    Phase switch: base -> arm at t={current_time:.2f}s  xy_dist={base_xy_dist:.4f}")
                if arm_model is None:
                    break
                phase = "arm"

        # Compute action
        if phase == "single":
            obs = build_observation(base_pos, base_bearing, base_lin_vel, base_ang_vel,
                                    q_mes, qd_mes, ee_pos, target_ee_pos, tau_gravity_arm)
            action = predict(model, mean, std, obs, device=device,
                             label_mean=label_mean, label_std=label_std)
            wheel_cmds = np.clip(action[:4], -0.25 / WHEEL_RADIUS, 0.25 / WHEEL_RADIUS)
            torque_arm = np.clip(action[4:], -30.0, 30.0)

        elif phase == "base":
            obs_feature_names = [n for n in base_feature_names if n != "phase"] \
                if has_base_phases else base_feature_names
            obs_base = build_base_observation(base_pos, base_bearing, base_lin_vel,
                                              base_ang_vel, target_ee_pos,
                                              feature_names=obs_feature_names)
            if has_base_phases:
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

            vw = predict(base_model, base_mean, base_std, obs_base,
                         device=device,
                         label_mean=base_label_mean, label_std=base_label_std)

            if has_base_phases:
                if base_sub_phase == "rotate":
                    v_cmd, w_cmd = 0.0, vw[1]
                else:
                    v_cmd, w_cmd = max(vw[0], 0.0), 0.0
            else:
                v_cmd = vw[0] if abs(vw[0]) > 0.02 else 0.0
                w_cmd = vw[1] if abs(vw[1]) > 0.05 else 0.0

            v_cmd = np.clip(v_cmd, -0.25, 0.25)
            w_cmd = np.clip(w_cmd, -1.0, 1.0)
            lw = (v_cmd - w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            rw = (v_cmd + w_cmd * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
            wheel_cmds = np.array([rw, lw, lw, rw])

            if arm_model is not None:
                obs_arm_hold = build_arm_observation(q_mes, qd_mes, ee_pos, ee_pos,
                                                     base_pos=base_pos, base_bearing=base_bearing,
                                                     feature_names=arm_feature_names)
                torque_arm = predict(arm_model, arm_mean, arm_std, obs_arm_hold,
                                     device=device,
                                     label_mean=arm_label_mean, label_std=arm_label_std)
                torque_arm = np.clip(torque_arm, -20.0, 20.0)
            else:
                torque_arm = np.zeros(7)

        elif phase == "arm":
            obs_arm = build_arm_observation(q_mes, qd_mes, ee_pos, target_ee_pos,
                                            base_pos=base_pos, base_bearing=base_bearing,
                                            feature_names=arm_feature_names)
            raw_torque = predict(arm_model, arm_mean, arm_std, obs_arm,
                                 device=device,
                                 label_mean=arm_label_mean, label_std=arm_label_std)
            if grav_model is not None:
                tau_grav = predict_gravity(grav_model, grav_q_mean, grav_q_std,
                                           grav_tau_mean, grav_tau_std,
                                           q_mes[4:11], device=device)
                raw_torque = raw_torque + tau_grav
            if args.arm_damping > 0:
                raw_torque = raw_torque - args.arm_damping * qd_mes[4:11]
            torque_arm = np.clip(raw_torque, -20.0, 20.0)
            wheel_cmds = np.clip(-50.0 * (q_mes[0:4] - wheel_lock_pos), -5.0, 5.0)

        # Send commands
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        try:
            sim.Step(cmd, "torque")
        except Exception as e:
            print(f"  Sim crashed at t={current_time:.2f}s: {e}")
            return {"ok": False, "final_err": float("inf"), "best_err": best_ee_error,
                    "reached": False, "steps": step, "time": current_time}

        # Check target reached with hold
        in_goal = ee_dist_error < REACH_TOLERANCE and current_time > 3.0
        if in_goal and in_goal_since is None:
            in_goal_since = current_time
        elif not in_goal and in_goal_since is not None:
            in_goal_since = None
        if in_goal_since is not None and current_time - in_goal_since >= HOLD_TIME:
            held = True
            print(f"    Target held at t={current_time:.2f}s")
            break

        if step % 1000 == 0:
            print(f"    t={current_time:.2f}s [{phase}] EE err: {ee_dist_error:.4f}m  best: {best_ee_error:.4f}m")

        if not args.headless:
            import time
            time.sleep(0.001)
        current_time += time_step
        step += 1

    # Release base lock if we created one
    if base_lock_cid is not None:
        try:
            sim.GetPyBulletClient().removeConstraint(base_lock_cid)
        except Exception:
            pass

    return {
        "ok": True,
        "final_err": ee_dist_error,
        "best_err": best_ee_error,
        "reached": held,
        "steps": step,
        "time": current_time,
        "phase_switched": phase_switched,
    }


def _run_pick_and_place_target_inline(args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
                                      init_joint_angles, target_finger_angles,
                                      lower_limits, upper_limits,
                                      ball_pos, drop_zone, grip_state,
                                      device, split_mode, single_mode,
                                      **policy_kwargs):
    """Run a single-ball pick-and-place cycle: nav_to_ball -> reach_ball -> nav_to_drop -> reach_drop.

    Uses learned policies for navigation/reaching, auto-grip/release for manipulation.
    Returns dict with per-ball results.
    """
    client = sim.GetPyBulletClient()
    rid = sim.bot[0].bot_pybullet
    ee_link_id = find_ee_link_id(client, rid)

    total_steps = 0
    total_time = 0.0

    # --- Phase 1+2: Navigate to ball and reach it ---
    args.base_threshold = compute_base_threshold_for_target(ball_pos)
    result_ball = _run_single_target_inline(
        args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, target_finger_angles,
        lower_limits, upper_limits,
        ball_pos, device, split_mode, single_mode,
        **policy_kwargs,
    )
    total_steps += result_ball.get("steps", 0)
    total_time += result_ball.get("time", 0.0)

    if not result_ball.get("ok", True):
        return {"ok": False, "gripped": False, "delivered": False,
                "steps": total_steps, "time": total_time}

    # --- Auto-grip check ---
    import pinocchio as pin
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    q_mes = np.array(sim.GetMotorAngles(0))
    q_all = np.concatenate((base_pos, base_ori, q_mes))
    q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    ball_world_pos, _ = client.getBasePositionAndOrientation(grip_state.ball_id)
    grip_dist = np.linalg.norm(ee_pos - np.array(ball_world_pos))
    gripped = grip_dist < GRIP_TOLERANCE
    if gripped:
        auto_grip(client, rid, ee_link_id, grip_state)
        print(f"    Auto-gripped (dist={grip_dist:.4f}m)")
    else:
        print(f"    Grip failed (dist={grip_dist:.4f}m)")
        return {"ok": True, "gripped": False, "delivered": False,
                "grip_dist": grip_dist,
                "steps": total_steps, "time": total_time}

    # --- Phase 3+4: Navigate to drop zone and release ---
    args.base_threshold = compute_base_threshold_for_target(drop_zone)
    result_drop = _run_single_target_inline(
        args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, target_finger_angles,
        lower_limits, upper_limits,
        drop_zone, device, split_mode, single_mode,
        **policy_kwargs,
    )
    total_steps += result_drop.get("steps", 0)
    total_time += result_drop.get("time", 0.0)

    # --- Auto-release check ---
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    q_mes = np.array(sim.GetMotorAngles(0))
    q_all = np.concatenate((base_pos, base_ori, q_mes))
    q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    ee_pos_drop = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    drop_dist = np.linalg.norm(ee_pos_drop - drop_zone)
    auto_release(client, grip_state)  # always release
    delivered = drop_dist < GRIP_TOLERANCE
    print(f"    Released at drop zone (dist={drop_dist:.4f}m, delivered={delivered})")

    return {
        "ok": result_drop.get("ok", True),
        "gripped": True,
        "delivered": delivered,
        "grip_dist": grip_dist,
        "drop_dist": drop_dist,
        "steps": total_steps,
        "time": total_time,
    }


def run_multi_target_pick_and_place_sim(args, target_ee_positions, device,
                                         split_mode, single_mode,
                                         drop_zone=None,
                                         sel_model=None, sel_mean=None, sel_std=None,
                                         use_expert_selector=False,
                                         **policy_kwargs):
    """Run a multi-ball pick-and-place episode using learned (or expert) selector.

    Spawns balls at each target position. For each ball: navigate, grip, carry to
    drop zone, release. Selector picks which ball to visit next.
    """
    from robotnik_arm_task import (
        shortest_path_expert_pnp, build_selector_scene,
        setup_simulation, setup_kinematics, add_target_visual,
    )
    from robotnik_arm_control import quaternion2bearing, configure_gripper_fingers

    if drop_zone is None:
        drop_zone = DROP_ZONE_DEFAULT.copy()
    else:
        drop_zone = np.array(drop_zone)

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, np.array([0.1]*6))

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    # Spawn balls and drop zone
    ee_link_id = find_ee_link_id(client, rid)
    grip_states = []
    for i, t in enumerate(target_ee_positions):
        ball_id = spawn_ball(client, t)
        grip_states.append(GripState(ball_id=ball_id))
        add_target_visual(client, t)
    add_drop_zone_visual(client, drop_zone)

    remaining = set(range(n_balls))
    per_ball_results = []
    visit_order = []
    total_steps = 0
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"Pick-and-place episode: {n_balls} balls, drop zone={np.round(drop_zone, 3)}")
    for i, t in enumerate(target_ee_positions):
        print(f"  Ball {i}: {np.round(t, 3)}")
    print(f"{'='*60}")

    while remaining:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        if use_expert_selector:
            chosen_idx, full_order = shortest_path_expert_pnp(base_pos, all_targets, remaining, drop_zone)
        else:
            scene_obs, slot_order = build_selector_scene(base_pos, base_bearing, all_targets, remaining)
            chosen_idx = predict_selector(sel_model, sel_mean, sel_std, scene_obs, slot_order, device)

        ball_pos = all_targets[chosen_idx]
        remaining.remove(chosen_idx)
        visit_order.append(chosen_idx)

        print(f"\n>>> Selector chose ball {chosen_idx} ({np.round(ball_pos, 3)}) | "
              f"{len(remaining)} remaining")

        result = _run_pick_and_place_target_inline(
            args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, target_finger_angles,
            lower_limits, upper_limits,
            ball_pos, drop_zone, grip_states[chosen_idx],
            device, split_mode, single_mode,
            **policy_kwargs,
        )

        total_steps += result.get("steps", 0)
        total_time += result.get("time", 0.0)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            **result,
        })

        if not result.get("ok", True):
            break

    # Summary
    n_gripped = sum(1 for r in per_ball_results if r.get("gripped", False))
    n_delivered = sum(1 for r in per_ball_results if r.get("delivered", False))
    print(f"\n{'='*60}")
    print(f"Pick-and-place complete: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered")
    print(f"Visit order: {visit_order}")
    print(f"Total steps: {total_steps}  Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    return {
        "ok": all(r.get("ok", True) for r in per_ball_results),
        "balls_gripped": n_gripped,
        "balls_delivered": n_delivered,
        "n_balls": n_balls,
        "visit_order": visit_order,
        "total_steps": total_steps,
        "total_time": total_time,
        "per_ball": per_ball_results,
    }


def run_sequential_pick_and_place_sim(args, target_ee_positions, device,
                                       split_mode, single_mode,
                                       drop_zone=None,
                                       sel_model=None, sel_mean=None, sel_std=None,
                                       use_expert_selector=False,
                                       **policy_kwargs):
    """Run sequential pick-and-place: pick ball_i, carry to ball_{i+1}, last to drop zone.

    Uses learned (or expert) selector for visit ordering, then executes the
    chain using learned base/arm policies.
    """
    from robotnik_arm_task import (
        build_selector_scene, setup_simulation, setup_kinematics, add_target_visual,
    )
    from robotnik_arm_control import quaternion2bearing, configure_gripper_fingers

    if drop_zone is None:
        drop_zone = DROP_ZONE_DEFAULT.copy()
    else:
        drop_zone = np.array(drop_zone)

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, np.array([0.1]*6))

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    ee_link_id = find_ee_link_id(client, rid)
    grip_states = []
    for i, t in enumerate(target_ee_positions):
        ball_id = spawn_ball(client, t)
        grip_states.append(GripState(ball_id=ball_id))
        add_target_visual(client, t)
    add_drop_zone_visual(client, drop_zone)

    # Determine visit order
    base_pos = np.array(sim.GetBasePosition())
    base_ori = np.array(sim.GetBaseOrientation())
    base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

    if use_expert_selector:
        _, full_order = shortest_path_expert_sequential(base_pos, all_targets,
                                                         set(range(n_balls)), drop_zone)
    else:
        # Use learned selector to pick first, then re-query after each delivery
        full_order = []
        remaining = set(range(n_balls))
        temp_pos = base_pos.copy()
        temp_bearing = base_bearing
        while remaining:
            scene_obs, slot_order = build_selector_scene(temp_pos, temp_bearing,
                                                          all_targets, remaining)
            chosen = predict_selector(sel_model, sel_mean, sel_std,
                                       scene_obs, slot_order, device)
            full_order.append(chosen)
            remaining.remove(chosen)
            # After picking ball, robot will be near next ball's position
            if full_order and len(remaining) > 0:
                temp_pos = all_targets[chosen]

    per_ball_results = []
    visit_order = full_order[:]
    total_steps = 0
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"Sequential pick-and-place: {n_balls} balls, drop zone={np.round(drop_zone, 3)}")
    print(f"Visit order: {full_order}")
    for i, t in enumerate(target_ee_positions):
        print(f"  Ball {i}: {np.round(t, 3)}")
    print(f"{'='*60}")

    for order_idx, chosen_idx in enumerate(full_order):
        ball_pos = all_targets[chosen_idx]
        gs = grip_states[chosen_idx]

        if order_idx < n_balls - 1:
            dest_pos = all_targets[full_order[order_idx + 1]]
            dest_label = f"ball {full_order[order_idx + 1]}"
        else:
            dest_pos = drop_zone
            dest_label = "drop zone"

        print(f"\n>>> Sequential: pick ball {chosen_idx}, carry to {dest_label} | "
              f"{n_balls - order_idx - 1} remaining")

        result = _run_pick_and_place_target_inline(
            args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, target_finger_angles,
            lower_limits, upper_limits,
            ball_pos, dest_pos, gs,
            device, split_mode, single_mode,
            **policy_kwargs,
        )

        total_steps += result.get("steps", 0)
        total_time += result.get("time", 0.0)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "dest_pos": dest_pos.tolist(),
            "dest_label": dest_label,
            **result,
        })

        if not result.get("ok", True):
            break

    n_gripped = sum(1 for r in per_ball_results if r.get("gripped", False))
    n_delivered = sum(1 for r in per_ball_results if r.get("delivered", False))
    print(f"\n{'='*60}")
    print(f"Sequential pick-and-place complete: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered")
    print(f"Visit order: {visit_order}")
    print(f"Total steps: {total_steps}  Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    return {
        "ok": all(r.get("ok", True) for r in per_ball_results),
        "balls_gripped": n_gripped,
        "balls_delivered": n_delivered,
        "n_balls": n_balls,
        "visit_order": visit_order,
        "total_steps": total_steps,
        "total_time": total_time,
        "per_ball": per_ball_results,
    }


def run_multi_drop_pick_and_place_sim(args, target_ee_positions, device,
                                       split_mode, single_mode,
                                       drop_zones=None,
                                       ball_zone_assignments=None,
                                       sel_model=None, sel_mean=None, sel_std=None,
                                       use_expert_selector=False,
                                       **policy_kwargs):
    """Run a multi-drop-zone pick-and-place episode.

    Each ball is assigned to one of the drop zones. The selector picks which ball
    to visit next. After delivering a ball to its assigned zone, the robot starts
    the next pick from that zone.
    """
    from robotnik_arm_task import (
        setup_simulation, setup_kinematics, add_target_visual,
        MAX_TARGETS,
    )
    from robotnik_arm_control import quaternion2bearing, configure_gripper_fingers
    import pybullet as p

    if drop_zones is None:
        drop_zones = [np.array(dz) for dz in DROP_ZONES_DEFAULT]
    else:
        drop_zones = [np.array(dz) for dz in drop_zones]

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    if ball_zone_assignments is None:
        rng = np.random.default_rng(args.multi_target_seed)
        ball_zone_assignments = [int(rng.integers(0, len(drop_zones))) for _ in range(n_balls)]

    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, np.array([0.1]*6))

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    # Spawn balls color-coded by zone
    ee_link_id = find_ee_link_id(client, rid)
    grip_states = []
    for i, t in enumerate(target_ee_positions):
        color = DROP_ZONE_COLORS[ball_zone_assignments[i]]
        ball_id = spawn_ball(client, t, color=color)
        grip_states.append(GripState(ball_id=ball_id, drop_zone_idx=ball_zone_assignments[i]))
        add_target_visual(client, t)

    # Spawn drop zone visuals
    for zi, dz in enumerate(drop_zones):
        color = DROP_ZONE_COLORS[zi]
        vis = client.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                        rgbaColor=[color[0], color[1], color[2], 0.5])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=dz.tolist())

    remaining = set(range(n_balls))
    per_ball_results = []
    visit_order = []
    total_steps = 0
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"Multi-drop-zone PnP: {n_balls} balls, {len(drop_zones)} zones")
    for i, t in enumerate(target_ee_positions):
        print(f"  Ball {i}: {np.round(t, 3)} -> zone {ball_zone_assignments[i]}")
    for zi, dz in enumerate(drop_zones):
        print(f"  Zone {zi}: {np.round(dz, 3)}")
    print(f"{'='*60}")

    while remaining:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        if use_expert_selector:
            chosen_idx, full_order = shortest_path_expert_multi_drop(
                base_pos, all_targets, remaining, drop_zones, ball_zone_assignments)
        else:
            scene_obs, slot_order = build_selector_scene_multi_drop(
                base_pos, base_bearing, all_targets, remaining,
                drop_zones, ball_zone_assignments)
            chosen_idx = predict_selector(sel_model, sel_mean, sel_std,
                                           scene_obs, slot_order, device)

        ball_pos = all_targets[chosen_idx]
        assigned_drop = drop_zones[ball_zone_assignments[chosen_idx]]
        remaining.remove(chosen_idx)
        visit_order.append(chosen_idx)

        print(f"\n>>> Selector chose ball {chosen_idx} ({np.round(ball_pos, 3)}) "
              f"-> zone {ball_zone_assignments[chosen_idx]} | {len(remaining)} remaining")

        # Pick up ball
        result_pick = _run_pick_and_place_target_inline(
            args, sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, target_finger_angles,
            lower_limits, upper_limits,
            ball_pos, assigned_drop, grip_states[chosen_idx],
            device, split_mode, single_mode,
            **policy_kwargs,
        )

        total_steps += result_pick.get("steps", 0)
        total_time += result_pick.get("time", 0.0)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "drop_zone_idx": ball_zone_assignments[chosen_idx],
            "drop_zone_pos": assigned_drop.tolist(),
            **result_pick,
        })

        if not result_pick.get("ok", True):
            break

    # Summary
    n_gripped = sum(1 for r in per_ball_results if r.get("gripped", False))
    n_delivered = sum(1 for r in per_ball_results if r.get("delivered", False))
    print(f"\n{'='*60}")
    print(f"Multi-drop-zone PnP complete: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered")
    print(f"Visit order: {visit_order}")

    # Per-zone delivery counts
    zone_counts = {}
    for r in per_ball_results:
        zi = r.get("drop_zone_idx", 0)
        zone_counts.setdefault(zi, {"total": 0, "delivered": 0})
        zone_counts[zi]["total"] += 1
        if r.get("delivered", False):
            zone_counts[zi]["delivered"] += 1
    for zi in sorted(zone_counts):
        zc = zone_counts[zi]
        print(f"  Zone {zi}: {zc['delivered']}/{zc['total']} delivered")

    print(f"Total steps: {total_steps}  Total time: {total_time:.2f}s")
    print(f"{'='*60}")

    return {
        "ok": all(r.get("ok", True) for r in per_ball_results),
        "balls_gripped": n_gripped,
        "balls_delivered": n_delivered,
        "n_balls": n_balls,
        "visit_order": visit_order,
        "ball_zone_assignments": ball_zone_assignments,
        "total_steps": total_steps,
        "total_time": total_time,
        "per_ball": per_ball_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run BC policy in simulation")
    # Single-model mode
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained .pt checkpoint (single-model mode)")
    # Split-model mode
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to base policy .pt checkpoint (split mode)")
    parser.add_argument("--arm_checkpoint", type=str, default=None,
                        help="Path to arm policy .pt checkpoint (split mode)")
    # Common args
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 1.1, 0.5],
                        metavar=("X", "Y", "Z"),
                        help="Target end-effector position (default: 0.0 1.1 0.5)")
    parser.add_argument("--max_steps", type=int, default=400000,
                        help="Maximum simulation steps")
    parser.add_argument("--gravity_comp", type=str, default=None,
                        help="Path to gravity compensation network checkpoint (adds predicted gravity torques to arm output)")
    parser.add_argument("--arm_damping", type=float, default=0.0,
                        help="Velocity damping coefficient for arm joints (default: 0, e.g. 5.0)")
    parser.add_argument("--headless", action="store_true",
                        help="Skip visualizer updates and sleep for fast evaluation")
    parser.add_argument("--mps", action="store_true",
                        help="Use Apple Metal (MPS) GPU for inference")
    parser.add_argument("--retries", type=int, default=0,
                        help="Number of times to restart simulation on crash (default: 0)")
    parser.add_argument("--heading_threshold", type=float, default=0.15,
                        help="Heading error (rad) for rotate/drive sub-phase transition "
                             "(only used when base checkpoint has 'phase' feature, default: 0.15)")
    # Multi-target mode
    parser.add_argument("--multi_target", action="store_true",
                        help="Run multi-target episode instead of single target")
    parser.add_argument("--num_targets_per_episode", type=int, default=3,
                        help="Number of targets in a multi-target episode (default: 3)")
    parser.add_argument("--selector_checkpoint", type=str, default=None,
                        help="Path to trained selector .pt checkpoint (if omitted, uses expert nearest-neighbor)")
    parser.add_argument("--multi_target_seed", type=int, default=42,
                        help="Random seed for generating multi-target positions (default: 42)")
    parser.add_argument("--per_target_max_steps", type=int, default=100000,
                        help="Max simulation steps per individual target in multi-target mode (default: 100000)")
    # Pick-and-place mode
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Enable pick-and-place mode: grip balls and carry to drop zone")
    parser.add_argument("--drop_zone", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Drop zone position (default: 0.0 0.0 0.5)")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential pick-and-place: pick ball_i, carry to ball_{i+1}, "
                             "last ball goes to drop zone")
    parser.add_argument("--multi_drop_zone", action="store_true",
                        help="Multi-drop-zone mode: balls assigned to 3 different drop zones")
    parser.add_argument("--drop_zones", type=float, nargs=9, default=None,
                        metavar=tuple(f"{c}{i}" for i in range(3) for c in "XYZ"),
                        help="9 floats for 3 drop zone positions (default: triangle layout)")
    args = parser.parse_args()

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = args.checkpoint is not None
    if not split_mode and not single_mode:
        args.checkpoint = "bc_policy_100.pt"
        single_mode = True
    if split_mode and single_mode:
        parser.error("Specify either --checkpoint OR (--base_checkpoint + --arm_checkpoint), not both.")

    # --- Load policy/policies (once, reused across retries) ---
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = dict(device=device)

    if single_mode:
        model, mean, std, label_mean, label_std, _ = load_policy(args.checkpoint, device=device)
        print(f"Loaded policy from {args.checkpoint}  device={device}")
        policy_kwargs.update(model=model, mean=mean, std=std,
                             label_mean=label_mean, label_std=label_std)
    else:
        base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
            load_policy(args.base_checkpoint, device=device)
        arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
            load_policy(args.arm_checkpoint, device=device)
        # Detect phase-as-input model (has 'phase' in feature names)
        has_base_phases = base_fnames is not None and "phase" in list(base_fnames)
        base_desc = args.base_checkpoint
        if has_base_phases:
            phase_desc = f"  phases=input (thr={args.heading_threshold:.3f})"
        else:
            phase_desc = "  phases=no"
        print(f"Split mode: base={base_desc}  arm={args.arm_checkpoint}  "
              f"device={device}"
              f"{phase_desc}")
        policy_kwargs.update(base_model=base_model, base_mean=base_mean, base_std=base_std,
                             base_label_mean=base_lm, base_label_std=base_ls,
                             base_feature_names=base_fnames,
                             has_base_phases=has_base_phases,
                             arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
                             arm_label_mean=arm_lm, arm_label_std=arm_ls,
                             arm_feature_names=arm_fnames)

    # Load gravity compensation network if provided
    if args.gravity_comp:
        grav_model, grav_q_mean, grav_q_std, grav_tau_mean, grav_tau_std = \
            load_gravity_comp(args.gravity_comp, device=device)
        print(f"Gravity comp: {args.gravity_comp}")
        policy_kwargs.update(grav_model=grav_model, grav_q_mean=grav_q_mean,
                             grav_q_std=grav_q_std, grav_tau_mean=grav_tau_mean,
                             grav_tau_std=grav_tau_std)

    # --- Multi-target mode ---
    if args.multi_target:
        # Load selector if checkpoint provided
        sel_model = sel_mean = sel_std = None
        use_expert_selector = args.selector_checkpoint is None
        if args.selector_checkpoint:
            sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
            print(f"Selector policy: {args.selector_checkpoint}")
        else:
            print("Selector: expert (nearest-neighbor)")

        # Generate random targets
        rng = np.random.default_rng(args.multi_target_seed)
        n = args.num_targets_per_episode
        targets = [np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 1.0)])
                    for _ in range(n)]

        if args.multi_drop_zone:
            # Parse drop zones (explicit or randomized)
            if args.drop_zones:
                dz_flat = args.drop_zones
                mdz_zones = [np.array(dz_flat[i*3:(i+1)*3]) for i in range(3)]
            else:
                rng_dz = np.random.default_rng(args.multi_target_seed + 1000)
                mdz_zones = generate_random_drop_zones(rng_dz)

            # Generate zone assignments
            rng_mdz = np.random.default_rng(args.multi_target_seed + 2000)
            ball_zone_assignments = [int(rng_mdz.integers(0, len(mdz_zones))) for _ in range(n)]

            result = run_multi_drop_pick_and_place_sim(
                args, targets,
                split_mode=split_mode, single_mode=single_mode,
                drop_zones=mdz_zones,
                ball_zone_assignments=ball_zone_assignments,
                sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                use_expert_selector=use_expert_selector,
                **policy_kwargs,
            )
            return

        if args.sequential or args.pick_and_place:
            drop_zone = np.array(args.drop_zone) if args.drop_zone else None

            # If no real policy checkpoints were provided, use expert controllers
            has_policy = (split_mode and not single_mode) or \
                         (single_mode and args.checkpoint != "bc_policy_100.pt")
            if not has_policy:
                from robotnik_arm_task import (
                    run_pick_and_place_loop, run_sequential_pick_and_place_loop,
                    setup_simulation, setup_kinematics, add_target_visual,
                    FIXED_FINGER_ANGLES,
                )
                from robotnik_arm_control import configure_gripper_fingers
                print("No policy checkpoints — using expert controllers")

                sim, client, rid, conf_file_name, cur_dir = setup_simulation()
                target_finger_angles = configure_gripper_fingers(sim, client, rid,
                                                                  FIXED_FINGER_ANGLES)
                dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, _, \
                    lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

                loop_fn = run_sequential_pick_and_place_loop if args.sequential \
                    else run_pick_and_place_loop
                combined, selector_samples, per_ball = loop_fn(
                    sim, dyn_model, ee_frame_id, shoulder_frame_id,
                    init_joint_angles, target_finger_angles,
                    lower_limits, upper_limits,
                    target_ee_positions=targets,
                    drop_zone=drop_zone,
                )
                return

            # Use learned policies
            if args.sequential:
                result = run_sequential_pick_and_place_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    drop_zone=drop_zone,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            else:
                result = run_multi_target_pick_and_place_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    drop_zone=drop_zone,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            return

        else:
            result = run_multi_target_sim(
                args, targets,
                split_mode=split_mode, single_mode=single_mode,
                sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                use_expert_selector=use_expert_selector,
                **policy_kwargs,
            )
        return

    target_ee_pos = np.array(args.target)

    # Compute base_threshold from arm geometry (Pythagorean: horizontal reach given vertical offset)
    if split_mode:
        dz = abs(target_ee_pos[2] - SHOULDER_HEIGHT)
        if dz < ARM_REACH_LIMIT:
            base_threshold = 0.75 * np.sqrt(ARM_REACH_LIMIT**2 - dz**2)
        else:
            base_threshold = 0.3  # target z beyond arm reach, get as close as possible
        args.base_threshold = base_threshold
        print(f"Base threshold: {base_threshold:.2f}m "
              f"(arm={ARM_REACH_LIMIT}m, dz={dz:.2f}m)")

    # --- Run with retries ---
    max_attempts = 1 + args.retries
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"\n>>> Restarting simulation (attempt {attempt}/{max_attempts}) <<<\n")
        result = run_sim(args, target_ee_pos, split_mode=split_mode, single_mode=single_mode,
                         **policy_kwargs)
        if result["ok"]:
            break
    else:
        print(f"\nAll {max_attempts} attempts crashed.")


if __name__ == "__main__":
    main()
