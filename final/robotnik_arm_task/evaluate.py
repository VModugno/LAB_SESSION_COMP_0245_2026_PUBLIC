#!/usr/bin/env python3
"""Evaluate BC policy accuracy across multiple random targets.

Reports separate metrics for base navigation and arm reaching phases,
plus overall end-to-end accuracy.

Usage:
  # Single policy
  python evaluate.py --checkpoint model.pt --num_targets 20 --headless

  # Split policy
  python evaluate.py --base_checkpoint base.pt --arm_checkpoint arm.pt --num_targets 20 --headless

  # Base only (skip arm phase)
  python evaluate.py --base_checkpoint base.pt --num_targets 20 --base_only --headless

  # Arm only (skip base, start near target)
  python evaluate.py --arm_checkpoint arm.pt --num_targets 20 --arm_only --headless
"""

import argparse
import os
import sys

if "--headless" in sys.argv:
    os.environ["ROBOENV_HEADLESS"] = "1"

import numpy as np

from run_policy_sim import (
    run_sim, load_policy, load_gravity_comp, load_selector,
    run_multi_target_sim, run_multi_target_pick_and_place_sim,
    run_sequential_pick_and_place_sim,
    run_multi_drop_pick_and_place_sim,
    REACH_TOLERANCE,
)
from robotnik_arm_task import (
    compute_base_goal, DROP_ZONES_DEFAULT, NUM_DROP_ZONES,
    shortest_path_expert_multi_drop, build_selector_scene_multi_drop,
    generate_random_drop_zones,
)


def generate_targets(n, seed=42):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-5, 5, n)
    ys = rng.uniform(-5, 5, n)
    zs = rng.uniform(0.2, 1.0, n)
    return [np.array([xs[i], ys[i], zs[i]]) for i in range(n)]


def print_stats(label, values):
    if not values:
        print(f"  {label:20s}: no data")
        return
    arr = np.array(values)
    print(f"  {label:20s}: mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}  std={arr.std():.4f}")


def print_buckets(label, values, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0)):
    if not values:
        return
    n = len(values)
    for t in thresholds:
        count = sum(1 for v in values if v < t)
        print(f"    {label} < {t}m: {count}/{n} ({100*count/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC policy across multiple targets")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--arm_checkpoint", type=str, default=None)
    parser.add_argument("--base_threshold", type=float, default=None)
    parser.add_argument("--num_targets", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Max sim steps (default: 100000 ≈ 100s at 1ms timestep)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mps", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_only", action="store_true",
                        help="Only evaluate base phase (stop at phase switch)")
    parser.add_argument("--arm_only", action="store_true",
                        help="Only evaluate arm phase (use expert base)")
    parser.add_argument("--gravity_comp", type=str, default=None,
                        help="Path to gravity compensation network checkpoint")
    # Multi-target evaluation
    parser.add_argument("--multi_target", action="store_true",
                        help="Evaluate multi-target episodes")
    parser.add_argument("--num_targets_per_episode", type=int, default=3,
                        help="Targets per multi-target episode (default: 3)")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of multi-target episodes to evaluate (default: 10)")
    parser.add_argument("--selector_checkpoint", type=str, default=None,
                        help="Path to selector checkpoint (omit for expert nearest-neighbor)")
    parser.add_argument("--selector_only", action="store_true",
                        help="Evaluate selector decisions only (no simulation)")
    parser.add_argument("--per_target_max_steps", type=int, default=100000,
                        help="Max steps per target in multi-target mode (default: 100000)")
    parser.add_argument("--heading_threshold", type=float, default=0.15)
    parser.add_argument("--arm_damping", type=float, default=0.0)
    # Pick-and-place evaluation
    parser.add_argument("--pick_and_place", action="store_true",
                        help="Evaluate pick-and-place mode (grip + deliver balls)")
    parser.add_argument("--drop_zone", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Drop zone position for pick-and-place (default: 0.0 0.0 0.5)")
    parser.add_argument("--sequential", action="store_true",
                        help="Sequential pick-and-place: pick ball_i, carry to ball_{i+1}")
    parser.add_argument("--multi_drop_zone", action="store_true",
                        help="Multi-drop-zone mode: balls assigned to 3 different drop zones")
    parser.add_argument("--xy_range", type=float, default=5.0,
                        help="XY workspace half-extent for target/zone generation (default: 5.0, training uses 5.0)")
    args = parser.parse_args()

    # Determine mode
    split_mode = args.base_checkpoint is not None and args.arm_checkpoint is not None
    single_mode = args.checkpoint is not None
    base_only = args.base_only
    arm_only = args.arm_only

    if base_only and not args.base_checkpoint:
        parser.error("--base_only requires --base_checkpoint")
    if arm_only and not args.arm_checkpoint:
        parser.error("--arm_only requires --arm_checkpoint")

    if base_only:
        split_mode = True
        single_mode = False
        if not args.arm_checkpoint:
            args.arm_checkpoint = None
    if arm_only:
        split_mode = True
        single_mode = False
        if not args.base_checkpoint:
            args.base_checkpoint = None

    if not split_mode and not single_mode and not args.selector_only:
        parser.error("Specify --checkpoint or --base_checkpoint/--arm_checkpoint")

    # Load policies
    import torch
    if args.mps and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    policy_kwargs = dict(device=device)

    if single_mode:
        model, mean, std, label_mean, label_std, _ = load_policy(args.checkpoint, device=device)
        policy_kwargs.update(model=model, mean=mean, std=std,
                             label_mean=label_mean, label_std=label_std)
        print(f"Single policy: {args.checkpoint}  device={device}")
    else:
        if args.base_checkpoint:
            base_model, base_mean, base_std, base_lm, base_ls, base_fnames = \
                load_policy(args.base_checkpoint, device=device)
            policy_kwargs.update(base_model=base_model, base_mean=base_mean, base_std=base_std,
                                 base_label_mean=base_lm, base_label_std=base_ls,
                                 base_feature_names=base_fnames)
            print(f"Base policy: {args.base_checkpoint}")

        if args.arm_checkpoint:
            arm_model, arm_mean, arm_std, arm_lm, arm_ls, arm_fnames = \
                load_policy(args.arm_checkpoint, device=device)
            policy_kwargs.update(arm_model=arm_model, arm_mean=arm_mean, arm_std=arm_std,
                                 arm_label_mean=arm_lm, arm_label_std=arm_ls,
                                 arm_feature_names=arm_fnames)
            print(f"Arm policy: {args.arm_checkpoint}")

        print(f"device={device}  switch={'xy<'+str(args.base_threshold)+'m' if args.base_threshold else 'settle'}")

    if args.gravity_comp:
        grav_model, grav_q_mean, grav_q_std, grav_tau_mean, grav_tau_std = \
            load_gravity_comp(args.gravity_comp, device=device)
        print(f"Gravity comp: {args.gravity_comp}")
        policy_kwargs.update(grav_model=grav_model, grav_q_mean=grav_q_mean,
                             grav_q_std=grav_q_std, grav_tau_mean=grav_tau_mean,
                             grav_tau_std=grav_tau_std)

    # --- Selector-only evaluation (no simulation) ---
    if args.selector_only:
        from robotnik_arm_task import shortest_path_expert, build_selector_scene
        from run_policy_sim import predict_selector

        if not args.selector_checkpoint:
            parser.error("--selector_only requires --selector_checkpoint")

        sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
        print(f"Selector: {args.selector_checkpoint}  device={device}")

        rng = np.random.default_rng(args.seed)
        n_per_ep = args.num_targets_per_episode

        is_mdz = args.multi_drop_zone
        xy_r = args.xy_range
        if is_mdz:
            print(f"Mode: multi-drop-zone (randomized zones per episode, xy_range={xy_r})")
        elif xy_r != 5.0:
            print(f"xy_range={xy_r}")

        total_decisions = 0
        correct_decisions = 0
        expert_total_dist = 0.0
        model_total_dist = 0.0

        for ep in range(args.num_episodes):
            targets = np.array([[rng.uniform(-xy_r, xy_r), rng.uniform(-xy_r, xy_r),
                                 rng.uniform(0.2, 1.0)]
                                for _ in range(n_per_ep)])

            robot_pos = np.array([0.0, 0.0, 0.0])
            robot_bearing = 0.0
            remaining = set(range(n_per_ep))

            # Per-episode random zones and assignments for multi-drop mode
            if is_mdz:
                mdz_zones = generate_random_drop_zones(rng, xy_range=xy_r)
                ball_zone_assignments = [int(rng.integers(0, len(mdz_zones)))
                                         for _ in range(n_per_ep)]

            ep_model_dist = 0.0

            while len(remaining) > 1:
                if is_mdz:
                    # Expert choice (multi-drop TSP)
                    expert_idx, _ = shortest_path_expert_multi_drop(
                        robot_pos, targets, remaining, mdz_zones, ball_zone_assignments)
                    # Model choice
                    scene_obs, slot_order = build_selector_scene_multi_drop(
                        robot_pos, robot_bearing, targets, remaining,
                        mdz_zones, ball_zone_assignments)
                else:
                    expert_idx, _ = shortest_path_expert(robot_pos, targets, remaining)
                    scene_obs, slot_order = build_selector_scene(
                        robot_pos, robot_bearing, targets, remaining)

                model_idx = predict_selector(sel_model, sel_mean, sel_std,
                                             scene_obs, slot_order, device)

                if model_idx == expert_idx:
                    correct_decisions += 1
                total_decisions += 1

                # Compute leg distance for model's choice
                ep_model_dist += np.linalg.norm(robot_pos[:2] - targets[model_idx, :2])
                if is_mdz:
                    # Add ball -> drop zone leg
                    dz = mdz_zones[ball_zone_assignments[model_idx]]
                    ep_model_dist += np.linalg.norm(targets[model_idx, :2] - dz[:2])

                # Move robot to next position
                remaining.remove(model_idx)
                if is_mdz:
                    # After delivery, robot is at the assigned drop zone
                    robot_pos = mdz_zones[ball_zone_assignments[model_idx]].copy()
                else:
                    robot_pos = targets[model_idx].copy()
                robot_bearing = rng.uniform(-np.pi, np.pi)

            # Last remaining target (no decision needed, just add distance)
            if remaining:
                last = remaining.pop()
                ep_model_dist += np.linalg.norm(robot_pos[:2] - targets[last, :2])
                if is_mdz:
                    dz = mdz_zones[ball_zone_assignments[last]]
                    ep_model_dist += np.linalg.norm(targets[last, :2] - dz[:2])

            # Compute expert total distance (full optimal path from origin)
            if is_mdz:
                _, full_order = shortest_path_expert_multi_drop(
                    np.array([0, 0, 0]), targets, set(range(n_per_ep)),
                    mdz_zones, ball_zone_assignments)
            else:
                _, full_order = shortest_path_expert(
                    np.array([0, 0, 0]), targets, set(range(n_per_ep)))

            ep_expert_dist_full = 0.0
            pos = np.array([0.0, 0.0, 0.0])
            for oi, idx in enumerate(full_order):
                ep_expert_dist_full += np.linalg.norm(pos[:2] - targets[idx, :2])
                if is_mdz:
                    dz = mdz_zones[ball_zone_assignments[idx]]
                    ep_expert_dist_full += np.linalg.norm(targets[idx, :2] - dz[:2])
                    pos = dz.copy()
                else:
                    pos = targets[idx].copy()

            expert_total_dist += ep_expert_dist_full
            model_total_dist += ep_model_dist

            if (ep + 1) % max(1, args.num_episodes // 10) == 0 or ep == 0:
                print(f"  Episode {ep+1}/{args.num_episodes}  "
                      f"model_dist={ep_model_dist:.2f}  expert_dist={ep_expert_dist_full:.2f}")

        acc = 100.0 * correct_decisions / max(total_decisions, 1)
        avg_expert = expert_total_dist / args.num_episodes
        avg_model = model_total_dist / args.num_episodes
        overhead = 100.0 * (avg_model - avg_expert) / max(avg_expert, 1e-6)

        mode_str = "MULTI-DROP-ZONE " if is_mdz else ""
        print(f"\n{'='*60}")
        print(f"{mode_str}SELECTOR-ONLY EVALUATION  ({args.num_episodes} episodes, "
              f"{n_per_ep} targets each)")
        print(f"{'='*60}")
        print(f"Decision accuracy:     {correct_decisions}/{total_decisions} ({acc:.1f}%)")
        print(f"Avg expert distance:   {avg_expert:.2f}")
        print(f"Avg model distance:    {avg_model:.2f}")
        print(f"Distance overhead:     {overhead:+.1f}%")
        print(f"{'='*60}")
        return

    # --- Multi-target evaluation ---
    if args.multi_target:
        sel_model = sel_mean = sel_std = None
        use_expert_selector = args.selector_checkpoint is None
        if args.selector_checkpoint:
            sel_model, sel_mean, sel_std = load_selector(args.selector_checkpoint, device=device)
            print(f"Selector: {args.selector_checkpoint}")
        else:
            print("Selector: expert (nearest-neighbor)")

        rng = np.random.default_rng(args.seed)
        all_results = []

        if args.multi_drop_zone:
            mode_label = "multi-drop-zone"
        elif args.sequential:
            mode_label = "sequential pick-and-place"
        elif args.pick_and_place:
            mode_label = "pick-and-place"
        else:
            mode_label = "multi-target"
        print(f"\nEvaluating {args.num_episodes} {mode_label} episodes "
              f"({args.num_targets_per_episode} targets each, seed={args.seed})\n")

        for ep in range(args.num_episodes):
            n = args.num_targets_per_episode
            targets = [np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(0.2, 1.0)])
                        for _ in range(n)]
            print(f"\n[Episode {ep+1}/{args.num_episodes}]")

            if args.multi_drop_zone:
                mdz_zones = generate_random_drop_zones(rng)
                ball_zone_assignments = [int(rng.integers(0, len(mdz_zones))) for _ in range(n)]
                result = run_multi_drop_pick_and_place_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    drop_zones=mdz_zones,
                    ball_zone_assignments=ball_zone_assignments,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            elif args.sequential:
                drop_zone = np.array(args.drop_zone) if args.drop_zone else None
                result = run_sequential_pick_and_place_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    drop_zone=drop_zone,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            elif args.pick_and_place:
                drop_zone = np.array(args.drop_zone) if args.drop_zone else None
                result = run_multi_target_pick_and_place_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    drop_zone=drop_zone,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            else:
                result = run_multi_target_sim(
                    args, targets,
                    split_mode=split_mode, single_mode=single_mode,
                    sel_model=sel_model, sel_mean=sel_mean, sel_std=sel_std,
                    use_expert_selector=use_expert_selector,
                    **policy_kwargs,
                )
            all_results.append(result)

        # --- Aggregate report ---
        if args.pick_and_place or args.sequential or args.multi_drop_zone:
            # Pick-and-place / multi-drop-zone report
            print(f"\n{'='*70}")
            print(f"{mode_label.upper()} EVALUATION REPORT  ({args.num_episodes} episodes, "
                  f"{args.num_targets_per_episode} balls each)")
            print(f"{'='*70}")

            gripped_per_ep = [r.get("balls_gripped", 0) for r in all_results]
            delivered_per_ep = [r.get("balls_delivered", 0) for r in all_results]
            n_balls_per_ep = [r.get("n_balls", 0) for r in all_results]
            time_per_ep = [r.get("total_time", 0.0) for r in all_results]
            steps_per_ep = [r.get("total_steps", 0) for r in all_results]

            total_gripped = sum(gripped_per_ep)
            total_delivered = sum(delivered_per_ep)
            total_balls = sum(n_balls_per_ep)
            full_deliver = sum(1 for g, n in zip(delivered_per_ep, n_balls_per_ep) if g == n)

            print(f"\n--- GRIP SUCCESS ---")
            print(f"Total gripped: {total_gripped}/{total_balls} "
                  f"({100*total_gripped/max(total_balls,1):.1f}%)")
            print_stats("Gripped/episode", gripped_per_ep)

            print(f"\n--- DELIVERY SUCCESS ---")
            print(f"Total delivered: {total_delivered}/{total_balls} "
                  f"({100*total_delivered/max(total_balls,1):.1f}%)")
            print(f"Full deliveries (all balls): {full_deliver}/{args.num_episodes} "
                  f"({100*full_deliver/max(args.num_episodes,1):.1f}%)")
            print_stats("Delivered/episode", delivered_per_ep)

            # Per-zone delivery counts (multi-drop-zone only)
            if args.multi_drop_zone:
                zone_delivered = {}
                zone_total = {}
                for r in all_results:
                    for pb in r.get("per_ball", []):
                        zi = pb.get("drop_zone_idx", 0)
                        zone_total[zi] = zone_total.get(zi, 0) + 1
                        if pb.get("delivered", False):
                            zone_delivered[zi] = zone_delivered.get(zi, 0) + 1
                print(f"\n--- PER-ZONE DELIVERY ---")
                for zi in sorted(zone_total):
                    d = zone_delivered.get(zi, 0)
                    t = zone_total[zi]
                    print(f"  Zone {zi}: {d}/{t} delivered ({100*d/max(t,1):.1f}%)")

            print(f"\n--- TIMING ---")
            print_stats("Time/episode (s)", time_per_ep)
            print_stats("Steps/episode", steps_per_ep)

            # Per-ball breakdown
            all_grip_dists = []
            all_drop_dists = []
            for r in all_results:
                for pb in r.get("per_ball", []):
                    if "grip_dist" in pb:
                        all_grip_dists.append(pb["grip_dist"])
                    if "drop_dist" in pb:
                        all_drop_dists.append(pb["drop_dist"])

            if all_grip_dists:
                print(f"\n--- PER-BALL ACCURACY ---")
                print_stats("Grip distance", all_grip_dists)
                print_buckets("grip_dist", all_grip_dists,
                              thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))
            if all_drop_dists:
                print_stats("Drop distance", all_drop_dists)
                print_buckets("drop_dist", all_drop_dists,
                              thresholds=(0.05, 0.08, 0.1, 0.2, 0.5))

            # Distance overhead vs optimal (multi-drop-zone selector evaluation)
            if args.multi_drop_zone and not use_expert_selector:
                print(f"\n--- SELECTOR QUALITY ---")
                print(f"(Compare visit order distance vs optimal multi-drop TSP)")

            print(f"{'='*70}")
            return

        # Standard multi-target report
        print(f"\n{'='*70}")
        print(f"MULTI-TARGET EVALUATION REPORT  ({args.num_episodes} episodes, "
              f"{args.num_targets_per_episode} targets each)")
        print(f"{'='*70}")

        reached_per_ep = [r["targets_reached"] for r in all_results]
        total_per_ep = [r["n_targets"] for r in all_results]
        time_per_ep = [r["total_time"] for r in all_results]
        steps_per_ep = [r["total_steps"] for r in all_results]

        total_reached = sum(reached_per_ep)
        total_targets = sum(total_per_ep)
        full_clear = sum(1 for r, t in zip(reached_per_ep, total_per_ep) if r == t)

        print(f"\n--- TARGETS ---")
        print(f"Total reached: {total_reached}/{total_targets} "
              f"({100*total_reached/total_targets:.1f}%)")
        print(f"Full clears (all targets): {full_clear}/{args.num_episodes} "
              f"({100*full_clear/args.num_episodes:.1f}%)")
        print_stats("Reached/episode", reached_per_ep)
        print_stats("Time/episode (s)", time_per_ep)
        print_stats("Steps/episode", steps_per_ep)

        # Per-target error stats
        all_final_errs = []
        all_best_errs = []
        for r in all_results:
            for pt in r["per_target"]:
                all_final_errs.append(pt.get("final_err", float("inf")))
                all_best_errs.append(pt.get("best_err", float("inf")))

        print(f"\n--- PER-TARGET ACCURACY ---")
        print_stats("Final EE error", all_final_errs)
        print_stats("Best EE error", all_best_errs)
        print_buckets("best_err", all_best_errs)

        # Visit ordering quality vs expert
        if not use_expert_selector:
            from robotnik_arm_task import shortest_path_expert
            order_matches = 0
            order_total = 0
            for ep_i, result in enumerate(all_results):
                n = result["n_targets"]
                per_target = result["per_target"]
                order_total += 1
            if order_total > 0:
                print(f"\n--- ORDERING ---")
                print(f"(Ordering comparison requires replay; "
                      f"compare with --selector_checkpoint=None for expert baseline)")

        print(f"{'='*70}")
        return

    # Generate targets
    targets = generate_targets(args.num_targets, seed=args.seed)
    print(f"\nEvaluating {len(targets)} targets (seed={args.seed})\n")

    # For base_only: reduce max_steps so it doesn't run forever after switch
    if base_only:
        args.max_steps = min(args.max_steps, 100000)

    # Run evaluations
    results = []
    for ti, target in enumerate(targets):
        print(f"[{ti+1}/{len(targets)}] Target: {np.round(target, 3)}", end="  ")
        sys.stdout.flush()

        result = run_sim(args, target, split_mode=split_mode, single_mode=single_mode,
                         **policy_kwargs)
        results.append(result)

        # Compute expert standoff goal for this target
        goal_xy, goal_bearing = compute_base_goal(target)
        result["standoff_xy"] = goal_xy.tolist()
        result["standoff_bearing"] = float(goal_bearing)

        # Base XY distance to standoff at best EE moment
        if result["base_xy_at_best"] is not None:
            bxy = np.array(result["base_xy_at_best"])
            result["base_xy_err"] = float(np.linalg.norm(bxy - goal_xy))
            bearing_err = result["base_bearing_at_best"] - goal_bearing
            result["base_bearing_err"] = float(np.abs(np.arctan2(np.sin(bearing_err), np.cos(bearing_err))))
        else:
            result["base_xy_err"] = None
            result["base_bearing_err"] = None

        # One-line summary
        status = "OK" if result["ok"] else "CRASH"
        xy_str = f"  base_xy_err={result['base_xy_err']:.4f}" if result["base_xy_err"] is not None else ""
        if result["phase_switched"]:
            print(f"{status}  base_best_ee={result['base_best_ee']:.4f}{xy_str}  "
                  f"arm_best={result['arm_best_ee']:.4f}  "
                  f"final={result['final_err']:.4f}")
        else:
            print(f"{status}  best={result['best_err']:.4f}{xy_str}  final={result['final_err']:.4f}")

    # ===== Aggregate Report =====
    ok_results = [r for r in results if r["ok"]]
    crashed = len(results) - len(ok_results)
    n = len(results)

    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT  ({n} targets)")
    print(f"{'='*70}")
    print(f"Completed: {len(ok_results)}/{n}  Crashed: {crashed}/{n}")

    if not ok_results:
        print("All runs crashed.")
        return

    # --- Overall ---
    print(f"\n--- OVERALL ---")
    reached = sum(1 for r in ok_results if r["reached"])
    print(f"Reached target ({REACH_TOLERANCE*1000:.0f}mm): {reached}/{len(ok_results)} "
          f"({100*reached/len(ok_results):.1f}%)")
    print_stats("Final EE error", [r["final_err"] for r in ok_results])
    print_stats("Best EE error", [r["best_err"] for r in ok_results])
    print_buckets("best", [r["best_err"] for r in ok_results])

    # --- Base phase ---
    base_results = [r for r in ok_results if r["base_best_ee"] < np.inf]
    if base_results:
        print(f"\n--- BASE PHASE ({len(base_results)} runs) ---")
        switched = [r for r in base_results if r["phase_switched"]]
        print(f"Phase switched: {len(switched)}/{len(base_results)}")
        if switched:
            print_stats("Switch time (s)", [r["switch_time"] for r in switched])
        print_stats("Best EE at base", [r["base_best_ee"] for r in base_results])
        print_stats("Final EE at base", [r["base_final_ee"] for r in base_results])
        print_buckets("base_best_ee", [r["base_best_ee"] for r in base_results])

        # XY distance to expert standoff position
        xy_errs = [r["base_xy_err"] for r in base_results if r["base_xy_err"] is not None]
        bearing_errs = [r["base_bearing_err"] for r in base_results if r["base_bearing_err"] is not None]
        if xy_errs:
            print(f"\n  Accuracy vs expert standoff position:")
            print_stats("XY dist to standoff", xy_errs)
            print_buckets("xy_err", xy_errs, thresholds=(0.05, 0.1, 0.2, 0.5, 1.0))
        if bearing_errs:
            bearing_degs = [np.degrees(e) for e in bearing_errs]
            print_stats("Bearing error (deg)", bearing_degs)
            print_buckets("bearing_err_deg", bearing_degs, thresholds=(5, 10, 20, 45, 90))

    # --- Arm phase ---
    arm_results = [r for r in ok_results if r["arm_best_ee"] < np.inf]
    if arm_results:
        print(f"\n--- ARM PHASE ({len(arm_results)} runs) ---")
        print_stats("Best EE at arm", [r["arm_best_ee"] for r in arm_results])
        print_stats("Final EE at arm", [r["arm_final_ee"] for r in arm_results])
        print_buckets("arm_best", [r["arm_best_ee"] for r in arm_results])
        # Improvement from base->arm
        improvements = []
        for r in arm_results:
            if r["base_final_ee"] < np.inf:
                improvements.append(r["base_final_ee"] - r["arm_best_ee"])
        if improvements:
            print_stats("Arm improvement", improvements)
            improved = sum(1 for v in improvements if v > 0)
            print(f"  Arm improved over base: {improved}/{len(improvements)} "
                  f"({100*improved/len(improvements):.1f}%)")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
