#!/usr/bin/env python3
"""Collect training data for the target-selector policy.

No simulation needed — we just generate random target sets, run the
shortest-path expert decision sequence, and log the observations.
At each decision point the robot is assumed to be at the last visited
target (or origin for the first decision), facing a random bearing.

Each row is one decision: a fixed-size scene vector (all remaining targets
packed into MAX_TARGETS slots with a validity mask) plus the label (which
slot in the packed order the expert chose).

Usage:
    python collect_selector_data.py --num_episodes 5000 --max_targets 5 --seed 42
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

from robotnik_arm_task import (
    MAX_TARGETS, SELECTOR_OBS_DIM, SELECTOR_INPUT_DIM,
    SELECTOR_OBS_DIM_MDZ, SELECTOR_INPUT_DIM_MDZ,
    NUM_DROP_ZONES, DROP_ZONES_DEFAULT, DROP_ZONE_DEFAULT,
    target_to_base_frame, shortest_path_expert,
    shortest_path_expert_pnp, shortest_path_expert_sequential,
    shortest_path_expert_multi_drop,
    build_selector_scene, build_selector_scene_multi_drop,
    generate_random_drop_zones,
)


def generate_episode_targets(rng, n_targets):
    """Generate n_targets random positions in the workspace."""
    xs = rng.uniform(-5, 5, n_targets)
    ys = rng.uniform(-5, 5, n_targets)
    zs = rng.uniform(0.2, 1.0, n_targets)
    return np.column_stack([xs, ys, zs])


def run_expert_decisions(targets, rng):
    """Simulate the expert decision sequence without any physics.

    The robot starts at the origin (0,0,0) with bearing 0.  The expert solves
    shortest-total-distance (TSP) over all remaining targets at each decision
    point and picks the first target in that optimal order.  After each
    decision the robot is teleported to the chosen target with a random bearing.

    Returns:
        obs_list: list of (SELECTOR_INPUT_DIM,) scene observations
        label_list: list of int — index within the packed slot_order
        decision_ids: list of int (decision index)
    """
    n = len(targets)
    remaining = set(range(n))

    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    obs_list = []
    label_list = []
    decision_ids = []
    decision_id = 0

    while len(remaining) > 1:
        # Build scene observation
        scene_obs, slot_order = build_selector_scene(
            robot_pos, robot_bearing, targets, remaining)

        # Expert picks optimal first target (shortest total path)
        chosen_idx, _ = shortest_path_expert(robot_pos, targets, remaining)

        # Label = position of chosen_idx in slot_order
        label = slot_order.index(chosen_idx)

        obs_list.append(scene_obs)
        label_list.append(label)
        decision_ids.append(decision_id)

        remaining.remove(chosen_idx)
        decision_id += 1

        # Teleport robot to chosen target with random bearing
        robot_pos = targets[chosen_idx].copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, label_list, decision_ids


def run_expert_decisions_pnp(targets, rng, drop_zone):
    """Pick-and-place expert decisions: robot returns to drop_zone after each ball.

    At each decision point the robot is at the drop zone (except the first,
    where it starts at origin).
    """
    n = len(targets)
    remaining = set(range(n))

    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    obs_list = []
    label_list = []
    decision_ids = []
    decision_id = 0

    while len(remaining) > 1:
        scene_obs, slot_order = build_selector_scene(
            robot_pos, robot_bearing, targets, remaining)

        chosen_idx, _ = shortest_path_expert_pnp(robot_pos, targets, remaining, drop_zone)
        label = slot_order.index(chosen_idx)

        obs_list.append(scene_obs)
        label_list.append(label)
        decision_ids.append(decision_id)

        remaining.remove(chosen_idx)
        decision_id += 1

        # After picking + delivering, robot is at the drop zone
        robot_pos = drop_zone.copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, label_list, decision_ids


def run_expert_decisions_sequential(targets, rng, drop_zone):
    """Sequential expert decisions: robot goes ball_i -> ball_{i+1} -> ... -> drop_zone.

    The full order is solved up front. At each decision point the robot is
    at the previous ball's position (or origin for the first).
    """
    n = len(targets)
    remaining = set(range(n))

    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    # Solve full order once
    _, full_order = shortest_path_expert_sequential(robot_pos, targets, remaining, drop_zone)

    obs_list = []
    label_list = []
    decision_ids = []

    remaining_set = set(range(n))
    for decision_id, chosen_idx in enumerate(full_order):
        if len(remaining_set) <= 1:
            break

        scene_obs, slot_order = build_selector_scene(
            robot_pos, robot_bearing, targets, remaining_set)

        label = slot_order.index(chosen_idx)

        obs_list.append(scene_obs)
        label_list.append(label)
        decision_ids.append(decision_id)

        remaining_set.remove(chosen_idx)

        # Robot carries ball to next position (the next ball, or drop zone for last)
        robot_pos = targets[chosen_idx].copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, label_list, decision_ids


def run_expert_decisions_multi_drop(targets, rng, drop_zones):
    """Multi-drop-zone expert decisions: each ball assigned to a random drop zone.

    At each decision point the robot is at the drop zone of the last delivered ball
    (or origin for the first). The expert solves the multi-drop TSP at each step.

    Returns:
        obs_list: list of (SELECTOR_INPUT_DIM_MDZ,) scene observations
        label_list: list of int — index within the packed slot_order
        decision_ids: list of int
    """
    n = len(targets)
    # Randomly assign each ball to a drop zone
    ball_zone_assignments = [int(rng.integers(0, len(drop_zones))) for _ in range(n)]

    remaining = set(range(n))
    robot_pos = np.array([0.0, 0.0, 0.0])
    robot_bearing = 0.0

    obs_list = []
    label_list = []
    decision_ids = []
    decision_id = 0

    while len(remaining) > 1:
        scene_obs, slot_order = build_selector_scene_multi_drop(
            robot_pos, robot_bearing, targets, remaining,
            drop_zones, ball_zone_assignments)

        chosen_idx, _ = shortest_path_expert_multi_drop(
            robot_pos, targets, remaining, drop_zones, ball_zone_assignments)
        label = slot_order.index(chosen_idx)

        obs_list.append(scene_obs)
        label_list.append(label)
        decision_ids.append(decision_id)

        remaining.remove(chosen_idx)
        decision_id += 1

        # After picking + delivering, robot is at the assigned drop zone
        robot_pos = drop_zones[ball_zone_assignments[chosen_idx]].copy()
        robot_bearing = rng.uniform(-np.pi, np.pi)

    return obs_list, label_list, decision_ids


def main():
    parser = argparse.ArgumentParser(description="Collect selector training data (no sim needed)")
    parser.add_argument("--num_episodes", type=int, default=5000,
                        help="Number of episodes (default: 5000)")
    parser.add_argument("--max_targets", type=int, default=5,
                        help="Max targets per episode (sampled from min..max, default: 5)")
    parser.add_argument("--min_targets", type=int, default=2,
                        help="Min targets per episode (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Use pick-and-place expert (robot returns to drop zone between balls)")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential expert (pick ball_i, carry to ball_{i+1})")
    parser.add_argument("--drop_zone", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Drop zone position (default: 0.0 0.0 0.5)")
    parser.add_argument("--multi_drop_zone", action="store_true",
                        help="Use multi-drop-zone expert (3 drop zones, balls randomly assigned)")
    args = parser.parse_args()

    assert args.min_targets >= 2
    rng = np.random.default_rng(args.seed)

    drop_zone = np.array(args.drop_zone) if args.drop_zone else DROP_ZONE_DEFAULT.copy()

    if args.multi_drop_zone:
        mode_label = "multi-drop-zone (randomized zones)"
        # Zones are generated per-episode inside the lambda
        decision_fn = lambda targets, rng: run_expert_decisions_multi_drop(
            targets, rng, generate_random_drop_zones(rng))
    elif args.sequential:
        mode_label = "sequential"
        decision_fn = lambda targets, rng: run_expert_decisions_sequential(targets, rng, drop_zone)
    elif args.pick_and_place:
        mode_label = "pick-and-place"
        decision_fn = lambda targets, rng: run_expert_decisions_pnp(targets, rng, drop_zone)
    else:
        mode_label = "multi-target"
        decision_fn = run_expert_decisions

    if args.multi_drop_zone:
        print(f"Mode: {mode_label}")
    else:
        print(f"Mode: {mode_label}  drop_zone={drop_zone if args.pick_and_place or args.sequential else 'N/A'}")

    all_obs = []
    all_labels = []
    all_eid = []
    all_did = []
    global_did = 0

    for ep in range(args.num_episodes):
        n_targets = rng.integers(args.min_targets, args.max_targets + 1)
        targets = generate_episode_targets(rng, n_targets)

        obs_list, label_list, local_dids = decision_fn(targets, rng)

        for obs, lbl, ldid in zip(obs_list, label_list, local_dids):
            all_obs.append(obs)
            all_labels.append(lbl)
            all_did.append(global_did + ldid)
            all_eid.append(ep)

        global_did += max(local_dids) + 1 if local_dids else 0

        if ep % 500 == 0 or ep == args.num_episodes - 1:
            print(f"Episode {ep+1}/{args.num_episodes}  "
                  f"decisions so far: {len(all_obs)}")

    inputs = np.array(all_obs, dtype=np.float64)      # (N_decisions, 20)
    labels = np.array(all_labels, dtype=np.int64)      # (N_decisions,)
    metadata = np.column_stack([
        np.array(all_eid, dtype=np.float64),
        np.array(all_did, dtype=np.float64),
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.out_dir, f"dataset_selector_{timestamp}.npz")
    np.savez(out_path, inputs=inputs, labels=labels, metadata=metadata)

    n_decisions = len(np.unique(all_did))
    print(f"\nSaved {out_path}")
    print(f"  decision rows: {inputs.shape[0]}  input_dim: {inputs.shape[1]}")
    print(f"  decisions: {n_decisions}")
    print("Done.")


if __name__ == "__main__":
    main()
