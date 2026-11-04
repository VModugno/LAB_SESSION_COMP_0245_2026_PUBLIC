"""Data collection script for behaviour cloning.

Runs the expert controller across many targets, logs input/output pairs
per timestep, and saves them to disk in .npz and .csv formats.
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Force headless mode for batch data collection
os.environ["ROBOENV_HEADLESS"] = "1"

from robotnik_arm_task import (
    FIXED_FINGER_ANGLES,
    TorqueNoise,
    TorqueNoiseParams,
    setup_simulation,
    setup_kinematics,
    run_simulation_loop,
    run_pick_and_place_loop,
    run_sequential_pick_and_place_loop,
    DROP_ZONE_DEFAULT,
)
from robotnik_arm_control import configure_gripper_fingers


# ==============================================================================
# Target generation
# ==============================================================================

def generate_grid_targets(n=100):
    """Produce targets on a fixed grid over the workspace.

    Workspace bounds: X: [-0.8, 0.8], Y: [0.5, 1.5], Z: [0.2, 1.0].
    Grid dimensions scale automatically so that nx * ny * nz >= n.
    """
    points_per_axis = int(np.ceil(n ** (1.0 / 3.0)))
    xs = np.linspace(-5, 5, points_per_axis)
    ys = np.linspace(-5, 5, points_per_axis)
    zs = np.linspace(0.2, 1.0, points_per_axis)
    targets = []
    for x in xs:
        for y in ys:
            for z in zs:
                targets.append(np.array([x, y, z]))
    return targets[:n]


def generate_random_targets(n=100, seed=42):
    """Uniform random sampling from the workspace volume."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-1, 1, n)
    ys = rng.uniform(-1, 1, n)
    zs = rng.uniform(0.2, 1.4, n)
    return [np.array([xs[i], ys[i], zs[i]]) for i in range(n)]


# ==============================================================================
# Simulation runner
# ==============================================================================

def run_single_trajectory(sim, client, rid, dyn_model, ee_frame_id,
                          shoulder_frame_id, init_joint_angles,
                          lower_limits, upper_limits,
                          target_ee_pos, vary_init=False, rng=None,
                          torque_noise=None, hold_time=0.0):
    """Reset the sim and run one trajectory for a given target.

    Returns the data dict from run_simulation_loop or None on failure.
    """
    # Reset to default pose
    sim.ResetPose()

    # Optionally perturb initial base position/orientation
    if vary_init and rng is not None:
        base_pos = list(sim.GetBasePosition())
        base_ori = list(sim.GetBaseOrientation())
        # Small perturbations: ±0.3m in x/y, ±0.1 rad in yaw
        base_pos[0] += rng.uniform(-0.3, 0.3)
        base_pos[1] += rng.uniform(-0.3, 0.3)
        import pybullet as p
        yaw_perturb = rng.uniform(-0.1, 0.1)
        new_ori = p.getQuaternionFromEuler([0, 0, yaw_perturb])
        client.resetBasePositionAndOrientation(
            sim.bot[0].bot_pybullet, base_pos, new_ori,
        )

    # Re-configure gripper fingers
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Re-read joint angles after reset
    init_joint_angles_now = np.array(sim.GetMotorAngles(0))

    try:
        data = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles_now, target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=target_ee_pos,
            torque_noise=torque_noise,
            hold_time=hold_time,
        )
        return data
    except Exception as e:
        print(f"  [WARN] Trajectory failed for target {target_ee_pos}: {e}")
        return None


def run_pick_and_place_trajectory(sim, client, rid, dyn_model, ee_frame_id,
                                   shoulder_frame_id, init_joint_angles,
                                   lower_limits, upper_limits,
                                   ball_positions, drop_zone=None,
                                   vary_init=False, rng=None,
                                   torque_noise_factory=None,
                                   per_target_timeout=None,
                                   sequential=False):
    """Reset the sim and run one pick-and-place episode for multiple balls.

    If *sequential* is True, uses the chain mode (pick at ball_i, drop at ball_{i+1}).
    Returns (combined_data, selector_samples, per_ball_results) or None on failure.
    """
    if drop_zone is None:
        drop_zone = DROP_ZONE_DEFAULT.copy()

    sim.ResetPose()

    if vary_init and rng is not None:
        base_pos = list(sim.GetBasePosition())
        base_ori = list(sim.GetBaseOrientation())
        base_pos[0] += rng.uniform(-0.3, 0.3)
        base_pos[1] += rng.uniform(-0.3, 0.3)
        import pybullet as p
        yaw_perturb = rng.uniform(-0.1, 0.1)
        new_ori = p.getQuaternionFromEuler([0, 0, yaw_perturb])
        client.resetBasePositionAndOrientation(
            sim.bot[0].bot_pybullet, base_pos, new_ori,
        )

    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    init_joint_angles_now = np.array(sim.GetMotorAngles(0))

    loop_fn = run_sequential_pick_and_place_loop if sequential else run_pick_and_place_loop

    try:
        combined_data, selector_samples, per_ball_results = loop_fn(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles_now, target_finger_angles,
            lower_limits, upper_limits,
            target_ee_positions=ball_positions,
            drop_zone=drop_zone,
            torque_noise_factory=torque_noise_factory,
            per_target_timeout=per_target_timeout,
        )
        return combined_data, selector_samples, per_ball_results
    except Exception as e:
        print(f"  [WARN] Pick-and-place episode failed: {e}")
        return None


# ==============================================================================
# Data assembly
# ==============================================================================

def assemble_dataset(all_trajectory_data):
    """Assemble per-timestep rows from all trajectories.

    Returns (inputs, labels, metadata) as numpy arrays.
      inputs:   (N, 44) — base_pos_x, base_pos_y, base_bearing,
                           base_lin_vel_x/y/z, base_ang_vel_x/y/z,
                           q_base_1..4, qd_base_1..4,
                           q_arm_1..7, qd_arm_1..7,
                           ee_pos_x/y/z, target_x, target_y, target_z,
                           tau_gravity_1..7
      labels:   (N, 11) — wheel_cmd_1..4, torque_arm_1..7
      metadata: (N, 2)  — trajectory_id, time
    """
    input_rows = []
    label_rows = []
    meta_rows = []

    for traj_id, data in enumerate(all_trajectory_data):
        n_steps = len(data["time"])
        has_gravity = "tau_gravity" in data and len(data["tau_gravity"]) == n_steps
        for i in range(n_steps):
            base_pos_xy = data["base_pos"][i, :2]            # 2
            base_bearing = data["base_bearing"][i]            # 1
            base_lin_vel = data["base_lin_vel"][i]            # 3
            base_ang_vel = data["base_ang_vel"][i]            # 3
            q_mes = np.array(data["q_mes"][i])
            qd_mes = np.array(data["qd_mes"][i])
            q_base = q_mes[0:4]                               # 4
            qd_base = qd_mes[0:4]                             # 4
            q_arm = q_mes[4:11]                               # 7
            qd_arm = qd_mes[4:11]                             # 7
            ee_pos = data["ee_pos"][i]                         # 3
            target = data["target_ee_pos"][i]                  # 3
            tau_grav = data["tau_gravity"][i] if has_gravity else np.zeros(7)  # 7

            inp = np.concatenate([
                base_pos_xy,
                [base_bearing],
                base_lin_vel,
                base_ang_vel,
                q_base,
                qd_base,
                q_arm,
                qd_arm,
                ee_pos,
                target,
                tau_grav,
            ])  # 44 features

            wheel_cmds = data["wheel_cmds"][i]                 # 4
            torque_arm = data["torque_arm"][i]                  # 7
            lbl = np.concatenate([wheel_cmds, torque_arm])     # 11 outputs

            meta = np.array([traj_id, data["time"][i]])

            input_rows.append(inp)
            label_rows.append(lbl)
            meta_rows.append(meta)

    inputs = np.array(input_rows, dtype=np.float64)
    labels = np.array(label_rows, dtype=np.float64)
    metadata = np.array(meta_rows, dtype=np.float64)
    return inputs, labels, metadata


# ==============================================================================
# Saving
# ==============================================================================

INPUT_COLS = (
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

LABEL_COLS = (
    [f"wheel_cmd_{i}" for i in range(1, 5)]
    + [f"torque_arm_{i}" for i in range(1, 8)]
)

META_COLS = ["trajectory_id", "time"]


def make_torque_noise(noise_params, clean_frac, noise_rng):
    """Return a fresh TorqueNoise for one trajectory, or None for a clean (noise-free) run."""
    if noise_params is None or noise_rng.random() < clean_frac:
        return None
    return TorqueNoise(noise_params, noise_rng)


def flush_chunk(all_trajectory_data, out_dir, chunk_idx):
    """Assemble a batch of trajectories and write to a temporary .npz chunk.

    Returns the path to the chunk file, or None if no data.
    """
    if not all_trajectory_data:
        return None
    inputs, labels, metadata = assemble_dataset(all_trajectory_data)
    os.makedirs(out_dir, exist_ok=True)
    chunk_path = os.path.join(out_dir, f"_chunk_{chunk_idx:04d}.npz")
    np.savez(chunk_path, inputs=inputs, labels=labels, metadata=metadata)
    print(f"  Flushed chunk {chunk_idx} → {chunk_path}  "
          f"({len(all_trajectory_data)} trajs, {inputs.shape[0]} rows)")
    return chunk_path


def merge_chunks(chunk_paths, out_dir):
    """Load all chunk files, concatenate, save final dataset, and clean up."""
    all_inputs, all_labels, all_meta = [], [], []
    for p in chunk_paths:
        # Close the file before cleanup; Windows cannot delete an open file
        with np.load(p) as d:
            all_inputs.append(d["inputs"])
            all_labels.append(d["labels"])
            all_meta.append(d["metadata"])
    inputs = np.concatenate(all_inputs)
    labels = np.concatenate(all_labels)
    metadata = np.concatenate(all_meta)

    # Re-number trajectory IDs so they are globally contiguous
    _, inverse = np.unique(metadata[:, 0], return_inverse=True)
    # Assign new sequential IDs preserving original order of first appearance
    first_occurrence = {}
    new_id = 0
    new_ids = np.empty(len(inverse), dtype=np.float64)
    for i, old_id in enumerate(metadata[:, 0]):
        if old_id not in first_occurrence:
            first_occurrence[old_id] = new_id
            new_id += 1
        new_ids[i] = first_occurrence[old_id]
    metadata[:, 0] = new_ids

    save_dataset(inputs, labels, metadata, out_dir)

    # Remove temporary chunk files
    for p in chunk_paths:
        os.remove(p)


def save_dataset(inputs, labels, metadata, out_dir):
    """Save dataset as .npz and .csv with timestamped filenames."""
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # .npz
    npz_path = os.path.join(out_dir, f"dataset_{timestamp}.npz")
    np.savez(npz_path, inputs=inputs, labels=labels, metadata=metadata)
    print(f"Saved {npz_path}  inputs={inputs.shape}  labels={labels.shape}  meta={metadata.shape}")

    # .csv
    csv_path = os.path.join(out_dir, f"dataset_{timestamp}.csv")
    all_cols = META_COLS + INPUT_COLS + LABEL_COLS
    all_data = np.hstack([metadata, inputs, labels])
    header = ",".join(all_cols)
    np.savetxt(csv_path, all_data, delimiter=",", header=header, comments="")
    print(f"Saved {csv_path}  rows={all_data.shape[0]}  cols={all_data.shape[1]}")


# ==============================================================================
# Worker for parallel collection
# ==============================================================================

def _worker_collect(worker_id, targets, vary_init, seed, flush_every, out_dir,
                    noise_params=None, clean_frac=0.0, hold_time=0.0):
    """Run a batch of trajectories in an independent sim instance.

    Returns list of chunk file paths written to disk.
    """
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, _, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    rng = np.random.default_rng(seed + worker_id) if vary_init else None
    noise_rng = np.random.default_rng(seed + 1000 + worker_id)

    all_data = []
    chunk_paths = []
    chunk_idx = 0
    for idx, target in enumerate(targets):
        torque_noise = make_torque_noise(noise_params, clean_frac, noise_rng)
        print(f"  [W{worker_id}] [{idx+1}/{len(targets)}] Target: {np.round(target, 3)}"
              f"{'  (noisy)' if torque_noise else ''}")
        data = run_single_trajectory(
            sim, client, rid, dyn_model, ee_frame_id, shoulder_frame_id,
            init_joint_angles, lower_limits, upper_limits,
            target_ee_pos=target, vary_init=vary_init, rng=rng,
            torque_noise=torque_noise, hold_time=hold_time,
        )
        if data is not None and len(data["time"]) > 0:
            all_data.append(data)
            print(f"  [W{worker_id}]   Steps: {len(data['time'])}  Final EE error: "
                  f"{np.linalg.norm(target - data['final_ee_pos']):.4f}m")
        else:
            print(f"  [W{worker_id}]   [SKIP] No data for target {target}")

        if len(all_data) >= flush_every:
            path = flush_chunk(all_data, out_dir, chunk_idx * 100 + worker_id)
            if path:
                chunk_paths.append(path)
                chunk_idx += 1
            all_data = []

    if all_data:
        path = flush_chunk(all_data, out_dir, chunk_idx * 100 + worker_id)
        if path:
            chunk_paths.append(path)

    return chunk_paths


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories for behaviour cloning.")
    parser.add_argument("--mode", choices=["grid", "random"], default="random",
                        help="Target generation mode (default: grid)")
    parser.add_argument("--num_targets", type=int, default=100,
                        help="Number of targets to collect (default: 100)")
    parser.add_argument("--vary_init", action="store_true",
                        help="Also randomise initial base pose per trajectory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for random targets and init perturbation")
    parser.add_argument("--flush_every", type=int, default=100,
                        help="Flush trajectories to disk every N runs (default: 100)")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
                        help="Output directory (default: final/robotnik_arm_task/data/)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel sim workers (default: 1)")
    # Torque noise (DART): perturb the executed arm torque, log the clean expert torque
    parser.add_argument("--noise_std", type=float, default=0.0,
                        help="Per-joint torque noise std in N*m during arm reaching; 0 disables (default: 0)")
    parser.add_argument("--noise_prob", type=float, default=0.5,
                        help="Chance each noise segment is active vs quiet (default: 0.5)")
    parser.add_argument("--noise_kick_prob", type=float, default=0.1,
                        help="Chance an active segment is a larger kick (default: 0.1)")
    parser.add_argument("--noise_kick_scale", type=float, default=3.0,
                        help="Kick std as a multiple of --noise_std (default: 3.0)")
    parser.add_argument("--clean_frac", type=float, default=0.25,
                        help="Fraction of trajectories collected without noise (default: 0.25)")
    parser.add_argument("--hold_time", type=float, default=2.0,
                        help="Seconds to keep recording after the target is first reached (default: 2)")
    # Pick-and-place mode
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Collect pick-and-place episodes instead of single-target reach")
    parser.add_argument("--drop_zone", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Drop zone position for pick-and-place (default: 0.0 0.0 0.5)")
    parser.add_argument("--num_balls_per_episode", type=int, default=3,
                        help="Number of balls per pick-and-place episode (default: 3)")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential pick-and-place: pick ball_i, carry to ball_{i+1}")
    args = parser.parse_args()

    noise_params = None
    if args.noise_std > 0:
        noise_params = TorqueNoiseParams(std=args.noise_std, active_prob=args.noise_prob,
                                         kick_prob=args.noise_kick_prob,
                                         kick_scale=args.noise_kick_scale)

    # Generate targets
    if args.mode == "grid":
        targets = generate_grid_targets(n=args.num_targets)
    else:
        targets = generate_random_targets(n=args.num_targets, seed=args.seed)

    # --- Pick-and-place mode ---
    if args.pick_and_place:
        drop_zone = np.array(args.drop_zone) if args.drop_zone else DROP_ZONE_DEFAULT.copy()
        n_balls = args.num_balls_per_episode
        # num_targets controls the number of episodes
        n_episodes = args.num_targets

        print(f"Collecting {n_episodes} pick-and-place episodes "
              f"({n_balls} balls each, drop_zone={drop_zone})")

        sim, client, rid, conf_file_name, cur_dir = setup_simulation()
        dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, _, \
            lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

        rng = np.random.default_rng(args.seed)
        noise_rng = np.random.default_rng(args.seed + 1000)

        all_data = []
        chunk_paths = []
        chunk_idx = 0
        for ep in range(n_episodes):
            # Generate random ball positions for this episode
            ball_positions = [
                np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 1.0)])
                for _ in range(n_balls)
            ]
            tnf = (lambda: make_torque_noise(noise_params, args.clean_frac, noise_rng)) if noise_params else None
            print(f"\n[Episode {ep+1}/{n_episodes}] Balls: {[np.round(b, 2).tolist() for b in ball_positions]}")

            result = run_pick_and_place_trajectory(
                sim, client, rid, dyn_model, ee_frame_id, shoulder_frame_id,
                init_joint_angles, lower_limits, upper_limits,
                ball_positions=ball_positions, drop_zone=drop_zone,
                vary_init=args.vary_init, rng=rng,
                torque_noise_factory=tnf,
                sequential=args.sequential,
            )
            if result is not None:
                combined_data, selector_samples, per_ball_results = result
                if "time" in combined_data and len(combined_data["time"]) > 0:
                    all_data.append(combined_data)
                    n_gripped = sum(1 for r in per_ball_results if r["gripped"])
                    n_delivered = sum(1 for r in per_ball_results if r["delivered"])
                    print(f"  Steps: {len(combined_data['time'])}  "
                          f"Gripped: {n_gripped}/{n_balls}  Delivered: {n_delivered}/{n_balls}")

            if len(all_data) >= args.flush_every:
                path = flush_chunk(all_data, args.out_dir, chunk_idx)
                if path:
                    chunk_paths.append(path)
                    chunk_idx += 1
                all_data = []

        if all_data:
            path = flush_chunk(all_data, args.out_dir, chunk_idx)
            if path:
                chunk_paths.append(path)

        if not chunk_paths:
            print("No data collected. Exiting.")
            sys.exit(1)

        print(f"\nMerging {len(chunk_paths)} chunks into final dataset...")
        merge_chunks(chunk_paths, args.out_dir)
        print("Done.")
        return

    print(f"Collecting {len(targets)} trajectories (mode={args.mode}, vary_init={args.vary_init}, "
          f"workers={args.workers})")
    if noise_params:
        print(f"Torque noise: {noise_params}  clean_frac={args.clean_frac}")
    if args.hold_time > 0:
        print(f"Hold after reach: {args.hold_time}s")

    if args.workers <= 1:
        # Single-process mode (original behavior)
        sim, client, rid, conf_file_name, cur_dir = setup_simulation()
        dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, _, \
            lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

        rng = np.random.default_rng(args.seed) if args.vary_init else None
        noise_rng = np.random.default_rng(args.seed + 1000)

        all_data = []
        chunk_paths = []
        chunk_idx = 0
        for idx, target in enumerate(targets):
            torque_noise = make_torque_noise(noise_params, args.clean_frac, noise_rng)
            print(f"\n[{idx+1}/{len(targets)}] Target: {np.round(target, 3)}"
                  f"{'  (noisy)' if torque_noise else ''}")
            data = run_single_trajectory(
                sim, client, rid, dyn_model, ee_frame_id, shoulder_frame_id,
                init_joint_angles, lower_limits, upper_limits,
                target_ee_pos=target, vary_init=args.vary_init, rng=rng,
                torque_noise=torque_noise, hold_time=args.hold_time,
            )
            if data is not None and len(data["time"]) > 0:
                all_data.append(data)
                print(f"  Steps: {len(data['time'])}  Final EE error: "
                      f"{np.linalg.norm(target - data['final_ee_pos']):.4f}m")
            else:
                print(f"  [SKIP] No data collected for target {target}")

            if len(all_data) >= args.flush_every:
                path = flush_chunk(all_data, args.out_dir, chunk_idx)
                if path:
                    chunk_paths.append(path)
                    chunk_idx += 1
                all_data = []

        if all_data:
            path = flush_chunk(all_data, args.out_dir, chunk_idx)
            if path:
                chunk_paths.append(path)
            all_data = []
    else:
        # Parallel mode: split targets across workers
        from concurrent.futures import ProcessPoolExecutor, as_completed

        splits = np.array_split(targets, args.workers)
        chunk_paths = []

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _worker_collect, wid, list(map(np.array, split)),
                    args.vary_init, args.seed, args.flush_every, args.out_dir,
                    noise_params, args.clean_frac, args.hold_time,
                ): wid
                for wid, split in enumerate(splits) if len(split) > 0
            }
            for future in as_completed(futures):
                wid = futures[future]
                try:
                    paths = future.result()
                    chunk_paths.extend(paths)
                    print(f"Worker {wid} done: {len(paths)} chunks")
                except Exception as e:
                    print(f"Worker {wid} failed: {e}")

    if not chunk_paths:
        print("No trajectories collected. Exiting.")
        sys.exit(1)

    # Merge all chunks into final dataset
    print(f"\nMerging {len(chunk_paths)} chunks into final dataset...")
    merge_chunks(chunk_paths, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
