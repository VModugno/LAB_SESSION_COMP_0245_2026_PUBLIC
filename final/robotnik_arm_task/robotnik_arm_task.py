import numpy as np
import time
import os
from dataclasses import dataclass
from itertools import permutations

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)


# ==============================================================================
# TARGET SPECIFICATION
# ==============================================================================
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.05
MAX_TARGETS = 5

# Pick-and-place constants
GRIP_TOLERANCE = 0.08
DROP_ZONE_DEFAULT = np.array([0.0, 0.0, 0.5])
BALL_RADIUS = 0.03
BALL_MASS = 0.05


# ==============================================================================
# TORQUE NOISE (DART-style data collection)
# ==============================================================================
@dataclass
class TorqueNoiseParams:
    """Random perturbations added to the arm torque command during data collection.

    The sim executes the perturbed torque but the clean expert torque is logged, so a policy
    trained on the data sees the expert recovering from pushes (braking near the target)
    without learning the noise itself.
    """
    std: float = 2.0               # N*m, per-joint std of an active noise segment
    active_prob: float = 0.5       # chance a segment is noisy; quiet segments let the arm settle
    kick_prob: float = 0.1         # chance an active segment is a larger kick
    kick_scale: float = 3.0        # kick std = std * kick_scale
    min_duration: float = 0.05     # s, the noise value is held constant for one segment
    max_duration: float = 0.3      # s


class TorqueNoise:
    """Piecewise-constant random torque: a new value is drawn at the start of each segment."""

    def __init__(self, params, rng):
        self.params = params
        self.rng = rng
        self.value = np.zeros(7)
        self.segment_end = 0.0

    def sample(self, current_time):
        if current_time >= self.segment_end:
            p_ = self.params
            self.segment_end = current_time + self.rng.uniform(p_.min_duration, p_.max_duration)
            if self.rng.random() < p_.active_prob:
                scale = p_.std * (p_.kick_scale if self.rng.random() < p_.kick_prob else 1.0)
                self.value = self.rng.normal(0.0, scale, 7)
            else:
                self.value = np.zeros(7)
        return self.value


def setup_simulation():
    """Initialise the PyBullet simulation, configure friction and motor gains.

    Returns (sim, client, rid).
    """
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet

    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0],
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    return sim, client, rid, conf_file_name, cur_dir


def setup_kinematics(sim, conf_file_name, cur_dir):
    """Build the Pinocchio kinematic/dynamic model and compute initial EE pose.

    Returns (dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles,
             init_ee_pos, lower_limits, upper_limits).
    """
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    lower_limits, upper_limits = sim.GetBotJointsLimit()

    return dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, lower_limits, upper_limits


def add_target_visual(client, target_pos):
    """Draw a red sphere at the target position in the PyBullet viewer."""
    pos_list = target_pos.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85],
    )
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=pos_list)
    client.addUserDebugPoints(pointPositions=[pos_list], pointColorsRGB=[[1.0, 0.0, 0.0]], pointSize=10)


def compute_base_goal(target_ee_pos):
    """Compute the base standoff pose (position + bearing) for a given target.

    Returns (base_goal_pos, base_goal_bearing).
    """
    standoff = float(np.interp(target_ee_pos[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target_ee_pos[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    return base_goal_pos, target_bearing


from dataclasses import dataclass, field


@dataclass
class TorqueNoiseParams:
    """Configuration for DART-style torque perturbation during data collection."""
    std: float = 1.0              # base noise std (N·m per joint)
    active_prob: float = 0.5      # probability each segment is noisy vs quiet
    kick_prob: float = 0.1        # probability an active segment is a larger kick
    kick_scale: float = 3.0       # kick std = kick_scale * std
    segment_steps: int = 50       # steps per noise segment


class TorqueNoise:
    """Generates per-step torque perturbation vectors for 7 arm joints."""

    def __init__(self, params: TorqueNoiseParams, rng=None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self._steps_in_segment = 0
        self._current_noise = np.zeros(7)
        self._active = False
        self._new_segment()

    def _new_segment(self):
        self._steps_in_segment = 0
        self._active = self.rng.random() < self.params.active_prob
        if self._active:
            scale = self.params.std
            if self.rng.random() < self.params.kick_prob:
                scale *= self.params.kick_scale
            self._current_noise = self.rng.normal(0, scale, size=7)
        else:
            self._current_noise = np.zeros(7)

    def sample(self):
        """Return the noise vector for this step, advancing the segment clock."""
        self._steps_in_segment += 1
        if self._steps_in_segment >= self.params.segment_steps:
            self._new_segment()
        return self._current_noise


def run_simulation_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                        init_joint_angles, target_finger_angles,
                        lower_limits, upper_limits, target_ee_pos=None,
                        torque_noise=None, hold_time=0.0):
    """Run the main control loop: base navigation then arm reaching.

    torque_noise: optional TorqueNoise. While the arm is reaching, its value is added to the
        torque sent to the sim; the logged torque_arm stays the clean expert command.
    hold_time: s to keep running after the target is first reached (0 = stop on reach).

    Returns a dict of logged data arrays.
    """
    if target_ee_pos is None:
        target_ee_pos = TARGET_EE_POS

    time_step = sim.GetTimeStep()
    current_time = 0.0

    base_goal_pos, base_goal_bearing = compute_base_goal(target_ee_pos)
    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])

    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    cmd = MotorCommands()
    reach_time = None

    # Data logs
    time_all, ee_pos_all, ee_error_all = [], [], []
    base_pos_all, q_mes_all, qd_mes_all = [], [], []
    q_d_all, qd_d_all = [], []
    torque_arm_all, wheel_cmds_all, base_bearing_all, target_ee_pos_all = [], [], [], []
    base_lin_vel_all, base_ang_vel_all = [], []
    tau_gravity_all = []
    torque_applied_all = []

    while True:
        # ---- Sense ----
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        base_lin_vel = np.array(sim.GetBaseLinVelocity())
        base_ang_vel = np.array(sim.GetBaseAngVelocity())

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = target_ee_pos - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # ---- Gravity compensation (for data logging) ----
        tau_g_full = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        tau_gravity_arm = tau_g_full[6:13].copy()  # 7 arm joints

        # ---- Base control ----
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params,
        )
        base_navigation_done = base_state.done

        # ---- Arm control ----
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target_ee_pos, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        # ---- Torque noise (executed only, never logged as the label) ----
        torque_applied = torque_arm
        if torque_noise is not None and base_navigation_done:
            torque_applied = np.clip(torque_arm + torque_noise.sample(current_time),
                                     -arm_params.max_torque, arm_params.max_torque)

        # ---- Send commands ----
        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_applied, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # ---- Optional visualisation ----
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # ---- Exit on 'q' key ----
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        if ord('q') in keys and keys[ord('q')] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # ---- Stop criteria ----
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if hold_time > 0:
            if reach_time is None and stop_message and stop_message.startswith("Target reached"):
                reach_time = current_time
                print(f"{stop_message} Holding for {hold_time:.1f}s more...")
            if reach_time is not None:
                # After the first reach only the hold window and the step cap end the run
                if current_time - reach_time >= hold_time:
                    stop_message = (f"Hold window over at t={current_time:.2f}s "
                                    f"(EE error {ee_dist_error:.4f} m)")
                elif not (stop_message or "").startswith("Reached step cap"):
                    stop_message = None
        if stop_message:
            print(stop_message)
            break

        # ---- Log data ----
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(np.concatenate((q_mes[:4], q_des_arm, target_finger_angles)))
        qd_d_all.append(np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6))))
        torque_arm_all.append(torque_arm)
        torque_applied_all.append(torque_applied)
        wheel_cmds_all.append(wheel_cmds)
        base_bearing_all.append(base_bearing)
        target_ee_pos_all.append(target_ee_pos.copy())
        base_lin_vel_all.append(base_lin_vel)
        base_ang_vel_all.append(base_ang_vel)
        tau_gravity_all.append(tau_gravity_arm)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | "
                  f"EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | "
                  f"yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")

    return {
        "time": np.array(time_all),
        "ee_pos": np.array(ee_pos_all),
        "ee_error": np.array(ee_error_all),
        "base_pos": np.array(base_pos_all),
        "q_mes": q_mes_all,
        "qd_mes": qd_mes_all,
        "q_d": q_d_all,
        "qd_d": qd_d_all,
        "torque_arm": np.array(torque_arm_all),
        "torque_applied": np.array(torque_applied_all),
        "reach_time": reach_time,
        "wheel_cmds": np.array(wheel_cmds_all),
        "base_bearing": np.array(base_bearing_all),
        "target_ee_pos": np.array(target_ee_pos_all),
        "base_lin_vel": np.array(base_lin_vel_all),
        "base_ang_vel": np.array(base_ang_vel_all),
        "tau_gravity": np.array(tau_gravity_all),
        "final_base_pos": base_pos,
        "final_ee_pos": ee_pos,
        "final_q_mes": q_mes,
        "final_time": current_time,
        "target_finger_angles": target_finger_angles,
    }


# ==============================================================================
# PICK-AND-PLACE PRIMITIVES
# ==============================================================================

@dataclass
class GripState:
    """Tracks the grip lifecycle of a single ball."""
    ball_id: int                # PyBullet body id
    constraint_id: int = -1     # PyBullet constraint id (when gripped)
    is_gripped: bool = False
    is_delivered: bool = False
    drop_zone_idx: int = 0      # assigned drop zone index (for multi-drop-zone mode)


def spawn_ball(client, position, color=None):
    """Spawn a ball at *position*, held in place until gripped.

    The ball is created with mass but immediately has its dynamics disabled
    (activationState=ACTIVATION_STATE_DISABLE_SLEEPING + DISABLE_SIMULATION)
    so it floats at the spawn height instead of falling to the ground.
    Dynamics are re-enabled by ``auto_grip`` just before attaching the constraint.

    Returns the PyBullet body id.
    """
    if color is None:
        color = [0.2, 0.4, 1.0, 1.0]
    col_shape = client.createCollisionShape(p.GEOM_SPHERE, radius=BALL_RADIUS)
    vis_shape = client.createVisualShape(p.GEOM_SPHERE, radius=BALL_RADIUS, rgbaColor=color)
    ball_id = client.createMultiBody(
        baseMass=BALL_MASS,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=position.tolist(),
    )
    # Freeze the ball in place — it stays at spawn position until gripped
    client.changeDynamics(ball_id, -1, mass=0)
    return ball_id


def add_drop_zone_visual(client, drop_pos):
    """Draw a green sphere marker at the drop zone."""
    vis = client.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.0, 1.0, 0.0, 0.6])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=drop_pos.tolist())
    client.addUserDebugPoints(
        pointPositions=[drop_pos.tolist()],
        pointColorsRGB=[[0.0, 1.0, 0.0]],
        pointSize=12,
    )


def find_ee_link_id(client, rid):
    """Find the PyBullet link index for the end-effector.

    Searches joint names for 'end_effector' or falls back to the last link.
    """
    n_joints = client.getNumJoints(rid)
    for j in range(n_joints):
        info = client.getJointInfo(rid, j)
        name = info[12].decode()  # child link name
        if "end_effector" in name:
            return j
    # Fallback: use last link
    return n_joints - 1


def auto_grip(client, rid, ee_link_id, grip_state):
    """Attach ball to the EE via a fixed constraint. Mutates *grip_state* in place."""
    if grip_state.is_gripped:
        return
    # Re-enable dynamics so the ball can move with the EE
    client.changeDynamics(grip_state.ball_id, -1, mass=BALL_MASS)
    cid = client.createConstraint(
        parentBodyUniqueId=rid,
        parentLinkIndex=ee_link_id,
        childBodyUniqueId=grip_state.ball_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )
    grip_state.constraint_id = cid
    grip_state.is_gripped = True


def auto_release(client, grip_state):
    """Remove the grip constraint and mark the ball as delivered."""
    if not grip_state.is_gripped:
        return
    client.removeConstraint(grip_state.constraint_id)
    grip_state.constraint_id = -1
    grip_state.is_gripped = False
    grip_state.is_delivered = True


def run_pick_and_place_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                            init_joint_angles, target_finger_angles,
                            lower_limits, upper_limits,
                            target_ee_positions,
                            drop_zone=None,
                            torque_noise_factory=None,
                            per_target_timeout=None):
    """Expert multi-ball pick-and-place episode.

    For each ball (in shortest-path order):
      1. nav_to_ball — base drives to ball standoff
      2. reach_ball — arm reaches ball, auto-grip on proximity
      3. nav_to_drop — base drives to drop zone standoff (ball attached)
      4. reach_drop — arm reaches drop pos, auto-release on proximity

    Returns (combined_data, selector_samples, per_ball_results).
    """
    if drop_zone is None:
        drop_zone = DROP_ZONE_DEFAULT.copy()

    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    ee_link_id = find_ee_link_id(client, rid)

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    # Spawn balls and drop zone visual
    grip_states = []
    for i in range(n_balls):
        ball_id = spawn_ball(client, all_targets[i])
        grip_states.append(GripState(ball_id=ball_id))
    add_drop_zone_visual(client, drop_zone)

    remaining = set(range(n_balls))
    selector_samples = []
    per_ball_results = []
    all_target_data = []
    episode_time = 0.0

    time_step = sim.GetTimeStep()

    while remaining:
        # --- Decision point: expert picks next ball ---
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        candidates = build_selector_candidates(base_pos, base_bearing, all_targets, remaining)
        chosen_idx, _ = shortest_path_expert_pnp(base_pos, all_targets, remaining, drop_zone)

        selector_samples.append({
            "candidates": candidates,
            "chosen_idx": chosen_idx,
            "episode_time": episode_time,
        })

        ball_pos = all_targets[chosen_idx]
        gs = grip_states[chosen_idx]
        remaining.remove(chosen_idx)
        print(f"\n>>> Pick-and-place: ball {chosen_idx} at {ball_pos} | {len(remaining)} remaining")

        # --- Phase 1: nav_to_ball ---
        noise = torque_noise_factory() if torque_noise_factory else None
        old_max = max_steps
        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(int(per_target_timeout / time_step))

        data_nav_ball = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            np.array(sim.GetMotorAngles(0)), target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=ball_pos,
            torque_noise=noise,
        )
        episode_time += data_nav_ball["final_time"]

        # --- Phase 2: auto-grip check ---
        ee_pos = data_nav_ball["final_ee_pos"]
        ball_world_pos, _ = client.getBasePositionAndOrientation(gs.ball_id)
        grip_dist = np.linalg.norm(ee_pos - np.array(ball_world_pos))
        gripped = grip_dist < GRIP_TOLERANCE
        if gripped:
            auto_grip(client, rid, ee_link_id, gs)
            print(f"    Gripped ball {chosen_idx} (dist={grip_dist:.4f}m)")
        else:
            print(f"    Grip FAILED ball {chosen_idx} (dist={grip_dist:.4f}m)")

        # --- Phase 3: nav_to_drop (only if gripped) ---
        data_nav_drop = None
        if gripped:
            noise2 = torque_noise_factory() if torque_noise_factory else None
            data_nav_drop = run_simulation_loop(
                sim, dyn_model, ee_frame_id, shoulder_frame_id,
                np.array(sim.GetMotorAngles(0)), target_finger_angles,
                lower_limits, upper_limits,
                target_ee_pos=drop_zone,
                torque_noise=noise2,
            )
            episode_time += data_nav_drop["final_time"]

            # --- Phase 4: auto-release check ---
            ee_pos_drop = data_nav_drop["final_ee_pos"]
            drop_dist = np.linalg.norm(ee_pos_drop - drop_zone)
            delivered = drop_dist < GRIP_TOLERANCE
            if delivered:
                auto_release(client, gs)
                print(f"    Delivered ball {chosen_idx} (dist={drop_dist:.4f}m)")
            else:
                auto_release(client, gs)  # release anyway to avoid carrying
                print(f"    Drop imprecise ball {chosen_idx} (dist={drop_dist:.4f}m), released anyway")

        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(old_max)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "gripped": gripped,
            "delivered": gs.is_delivered,
            "grip_dist": grip_dist,
            "nav_ball_time": data_nav_ball["final_time"],
            "nav_drop_time": data_nav_drop["final_time"] if data_nav_drop else 0.0,
        })

        all_target_data.append(data_nav_ball)
        if data_nav_drop is not None:
            all_target_data.append(data_nav_drop)

        print(f"    Ball {chosen_idx}: gripped={gripped}  delivered={gs.is_delivered}")

    # --- Combine trajectory data ---
    combined_data = _combine_trajectory_data(all_target_data)
    combined_data["episode_time"] = episode_time
    combined_data["per_ball_results"] = per_ball_results

    # Summary
    n_gripped = sum(1 for r in per_ball_results if r["gripped"])
    n_delivered = sum(1 for r in per_ball_results if r["delivered"])
    print(f"\n>>> Pick-and-place done: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered, {episode_time:.2f}s total")

    return combined_data, selector_samples, per_ball_results


def shortest_path_expert_sequential(base_pos, target_positions, remaining_indices, drop_zone):
    """Solve TSP for sequential pick-and-place: pick at ball_i, drop at ball_{i+1}.

    Travel cost for permutation (b0, b1, ..., bN-1) is:
      dist(base, b0) + dist(b0, b1) + dist(b1, b2) + ... + dist(bN-1, drop)

    Each leg is a pick-at-source, carry-to-destination trip. The last ball
    is dropped at *drop_zone*.

    Returns:
        chosen_idx: int, the first ball to visit
        full_order: list[int], complete optimal visit sequence
    """
    remaining = list(remaining_indices)
    if len(remaining) == 1:
        return remaining[0], remaining

    drop_xy = drop_zone[:2]
    best_dist = np.inf
    best_order = None
    for perm in permutations(remaining):
        # First leg: base → ball_0
        d = np.linalg.norm(base_pos[:2] - target_positions[perm[0], :2])
        # Intermediate legs: ball_i → ball_{i+1} (carry from pick to next pick)
        for i in range(len(perm) - 1):
            d += np.linalg.norm(target_positions[perm[i], :2] - target_positions[perm[i + 1], :2])
        # Last leg: last ball → drop zone
        d += np.linalg.norm(target_positions[perm[-1], :2] - drop_xy)
        if d < best_dist:
            best_dist = d
            best_order = list(perm)

    return best_order[0], best_order


def run_sequential_pick_and_place_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                                        init_joint_angles, target_finger_angles,
                                        lower_limits, upper_limits,
                                        target_ee_positions,
                                        drop_zone=None,
                                        torque_noise_factory=None,
                                        per_target_timeout=None):
    """Expert sequential pick-and-place: pick ball_i, carry to ball_{i+1}, drop there.

    The last ball is dropped at *drop_zone*.

    For each ball in the visit order:
      1. nav_to_ball — base drives to ball_i standoff
      2. reach_ball — arm reaches ball_i, auto-grip on proximity
      3. nav_to_next — base drives to ball_{i+1} (or drop_zone for last)
      4. reach_next — arm reaches next position, auto-release on proximity

    Returns (combined_data, selector_samples, per_ball_results).
    """
    if drop_zone is None:
        drop_zone = DROP_ZONE_DEFAULT.copy()

    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    ee_link_id = find_ee_link_id(client, rid)

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    # Spawn balls and drop zone visual
    grip_states = []
    for i in range(n_balls):
        ball_id = spawn_ball(client, all_targets[i])
        grip_states.append(GripState(ball_id=ball_id))
    add_drop_zone_visual(client, drop_zone)

    # Determine full visit order up front using sequential TSP
    base_pos = np.array(sim.GetBasePosition())
    _, full_order = shortest_path_expert_sequential(base_pos, all_targets, set(range(n_balls)), drop_zone)

    selector_samples = []
    per_ball_results = []
    all_target_data = []
    episode_time = 0.0
    time_step = sim.GetTimeStep()

    remaining = set(range(n_balls))

    for order_idx, chosen_idx in enumerate(full_order):
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        candidates = build_selector_candidates(base_pos, base_bearing, all_targets, remaining)
        selector_samples.append({
            "candidates": candidates,
            "chosen_idx": chosen_idx,
            "episode_time": episode_time,
        })
        remaining.discard(chosen_idx)

        ball_pos = all_targets[chosen_idx]
        gs = grip_states[chosen_idx]

        # Determine drop destination: next ball's position, or drop_zone for last ball
        if order_idx < n_balls - 1:
            dest_pos = all_targets[full_order[order_idx + 1]]
            dest_label = f"ball {full_order[order_idx + 1]}"
        else:
            dest_pos = drop_zone
            dest_label = "drop zone"

        print(f"\n>>> Sequential: pick ball {chosen_idx} at {ball_pos}, "
              f"carry to {dest_label} ({dest_pos}) | {len(remaining)} remaining")

        # --- Phase 1: nav to ball ---
        noise = torque_noise_factory() if torque_noise_factory else None
        old_max = max_steps
        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(int(per_target_timeout / time_step))

        data_nav_ball = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            np.array(sim.GetMotorAngles(0)), target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=ball_pos,
            torque_noise=noise,
        )
        episode_time += data_nav_ball["final_time"]

        # --- Phase 2: auto-grip check ---
        ee_pos = data_nav_ball["final_ee_pos"]
        ball_world_pos, _ = client.getBasePositionAndOrientation(gs.ball_id)
        grip_dist = np.linalg.norm(ee_pos - np.array(ball_world_pos))
        gripped = grip_dist < GRIP_TOLERANCE
        if gripped:
            auto_grip(client, rid, ee_link_id, gs)
            print(f"    Gripped ball {chosen_idx} (dist={grip_dist:.4f}m)")
        else:
            print(f"    Grip FAILED ball {chosen_idx} (dist={grip_dist:.4f}m)")

        # --- Phase 3: nav to destination ---
        data_nav_dest = None
        if gripped:
            noise2 = torque_noise_factory() if torque_noise_factory else None
            data_nav_dest = run_simulation_loop(
                sim, dyn_model, ee_frame_id, shoulder_frame_id,
                np.array(sim.GetMotorAngles(0)), target_finger_angles,
                lower_limits, upper_limits,
                target_ee_pos=dest_pos,
                torque_noise=noise2,
            )
            episode_time += data_nav_dest["final_time"]

            # --- Phase 4: auto-release ---
            ee_pos_dest = data_nav_dest["final_ee_pos"]
            dest_dist = np.linalg.norm(ee_pos_dest - dest_pos)
            auto_release(client, gs)
            delivered = dest_dist < GRIP_TOLERANCE
            print(f"    Released ball {chosen_idx} at {dest_label} (dist={dest_dist:.4f}m)")

        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(old_max)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "dest_pos": dest_pos.tolist(),
            "dest_label": dest_label,
            "gripped": gripped,
            "delivered": gs.is_delivered,
            "grip_dist": grip_dist,
            "drop_dist": dest_dist if gripped and data_nav_dest else float("inf"),
            "nav_ball_time": data_nav_ball["final_time"],
            "nav_dest_time": data_nav_dest["final_time"] if data_nav_dest else 0.0,
        })

        all_target_data.append(data_nav_ball)
        if data_nav_dest is not None:
            all_target_data.append(data_nav_dest)

    # --- Combine trajectory data ---
    combined_data = _combine_trajectory_data(all_target_data)
    combined_data["episode_time"] = episode_time
    combined_data["per_ball_results"] = per_ball_results

    n_gripped = sum(1 for r in per_ball_results if r["gripped"])
    n_delivered = sum(1 for r in per_ball_results if r["delivered"])
    print(f"\n>>> Sequential pick-and-place done: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered, {episode_time:.2f}s total")

    return combined_data, selector_samples, per_ball_results


SELECTOR_OBS_DIM = 3  # per-candidate: (in_base_x, in_base_y, z)
SELECTOR_INPUT_DIM = MAX_TARGETS * SELECTOR_OBS_DIM + MAX_TARGETS  # 5*3 + 5 = 20

# Multi-drop-zone constants
NUM_DROP_ZONES = 3
DROP_ZONES_DEFAULT = [
    np.array([-2.5, -1.5, 0.5]),
    np.array([2.5, -1.5, 0.5]),
    np.array([0.0, 3.0, 0.5]),
]
DROP_ZONE_COLORS = [
    [1.0, 0.2, 0.2, 1.0],   # red
    [0.2, 0.6, 1.0, 1.0],   # blue
    [0.2, 1.0, 0.3, 1.0],   # green
]
SELECTOR_OBS_DIM_MDZ = 6  # per-candidate: ball_pos(3) + drop_zone_pos(3) in base frame
SELECTOR_INPUT_DIM_MDZ = MAX_TARGETS * SELECTOR_OBS_DIM_MDZ + MAX_TARGETS  # 5*6 + 5 = 35


def generate_random_drop_zones(rng, n=NUM_DROP_ZONES, xy_range=5.0, z=0.5, min_sep=2.0):
    """Generate *n* random drop zone positions with minimum pairwise XY separation.

    Args:
        rng: numpy Generator
        n: number of zones
        xy_range: zones sampled in [-xy_range, xy_range]
        z: fixed height for all zones
        min_sep: minimum pairwise XY distance (rejection-sampled)
    Returns:
        list of (3,) arrays
    """
    for _ in range(200):  # rejection sampling attempts
        xs = rng.uniform(-xy_range, xy_range, n)
        ys = rng.uniform(-xy_range, xy_range, n)
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if np.hypot(xs[i] - xs[j], ys[i] - ys[j]) < min_sep:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return [np.array([xs[i], ys[i], z]) for i in range(n)]
    # Fallback: use default triangle if rejection fails
    return [np.array(dz) for dz in DROP_ZONES_DEFAULT]


def target_to_base_frame(base_pos, base_bearing, target_pos):
    """Express a target's position relative to the robot's base frame.

    Returns (3,): (forward, lateral, z).  XY are rotated into the robot's
    heading frame; Z is kept absolute (robot doesn't move vertically).
    """
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]
    return np.array([c * dx + s * dy,      # forward
                     -s * dx + c * dy,     # lateral
                     target_pos[2]])       # z


def build_selector_candidates(base_pos, base_bearing, target_positions, remaining_indices):
    """Build per-candidate observations for the target selector.

    Args:
        base_pos: (3,) robot base position
        base_bearing: scalar, robot heading (yaw)
        target_positions: (N, 3) all target positions (world frame)
        remaining_indices: iterable of valid target slot indices
    Returns:
        candidates: dict mapping slot_index -> (3,) observation in base frame
    """
    candidates = {}
    for idx in remaining_indices:
        candidates[idx] = target_to_base_frame(base_pos, base_bearing, target_positions[idx])
    return candidates


def build_selector_scene(base_pos, base_bearing, target_positions, remaining_indices):
    """Build a fixed-size scene observation for the target selector.

    Packs all remaining candidates into MAX_TARGETS slots (sorted by slot index)
    with a validity mask.  Empty slots are zero-padded.

    Args:
        base_pos: (3,) robot base position
        base_bearing: scalar, robot heading (yaw)
        target_positions: (N, 3) all target positions (world frame)
        remaining_indices: iterable of valid target slot indices
    Returns:
        obs: (SELECTOR_INPUT_DIM,) = (MAX_TARGETS*3 + MAX_TARGETS,) flattened
        slot_order: list of slot indices in the order they appear in obs
    """
    remaining = sorted(remaining_indices)
    obs_block = np.zeros(MAX_TARGETS * SELECTOR_OBS_DIM)
    mask = np.zeros(MAX_TARGETS)
    slot_order = []

    for i, idx in enumerate(remaining):
        bf = target_to_base_frame(base_pos, base_bearing, target_positions[idx])
        obs_block[i * SELECTOR_OBS_DIM:(i + 1) * SELECTOR_OBS_DIM] = bf
        mask[i] = 1.0
        slot_order.append(idx)

    return np.concatenate([obs_block, mask]), slot_order


def build_selector_scene_multi_drop(base_pos, base_bearing, target_positions,
                                     remaining_indices, drop_zones, ball_zone_assignments):
    """Build a fixed-size scene observation for the multi-drop-zone selector.

    Each slot gets 6 dims: ball_pos in base frame (3) + assigned drop zone pos in base frame (3).

    Args:
        base_pos: (3,) robot base position
        base_bearing: scalar, robot heading (yaw)
        target_positions: (N, 3) all target positions (world frame)
        remaining_indices: iterable of valid target slot indices
        drop_zones: list of (3,) drop zone positions
        ball_zone_assignments: dict or list mapping ball index -> drop zone index
    Returns:
        obs: (SELECTOR_INPUT_DIM_MDZ,) = (MAX_TARGETS*6 + MAX_TARGETS,) flattened
        slot_order: list of slot indices in the order they appear in obs
    """
    remaining = sorted(remaining_indices)
    obs_block = np.zeros(MAX_TARGETS * SELECTOR_OBS_DIM_MDZ)
    mask = np.zeros(MAX_TARGETS)
    slot_order = []

    for i, idx in enumerate(remaining):
        ball_bf = target_to_base_frame(base_pos, base_bearing, target_positions[idx])
        dz_idx = ball_zone_assignments[idx]
        dz_bf = target_to_base_frame(base_pos, base_bearing, drop_zones[dz_idx])
        start = i * SELECTOR_OBS_DIM_MDZ
        obs_block[start:start + 3] = ball_bf
        obs_block[start + 3:start + 6] = dz_bf
        mask[i] = 1.0
        slot_order.append(idx)

    return np.concatenate([obs_block, mask]), slot_order


def shortest_path_expert_multi_drop(base_pos, target_positions, remaining_indices,
                                     drop_zones, ball_zone_assignments):
    """Solve TSP for multi-drop-zone pick-and-place.

    After delivering ball_i to its assigned drop zone, the robot starts the next
    pick from that drop zone's position. Cost:
      dist(base, ball_0) + dist(ball_0, dropzone[ball_0])
      + sum_i( dist(dropzone[ball_{i-1}], ball_i) + dist(ball_i, dropzone[ball_i]) )

    Args:
        base_pos: (3,) robot base position (starting point)
        target_positions: (N, 3) all ball positions
        remaining_indices: set of valid ball slot indices
        drop_zones: list of (3,) drop zone positions
        ball_zone_assignments: dict or list mapping ball index -> drop zone index
    Returns:
        chosen_idx: int, the first ball to visit
        full_order: list[int], complete optimal visit sequence
    """
    remaining = list(remaining_indices)
    if len(remaining) == 1:
        return remaining[0], remaining

    best_dist = np.inf
    best_order = None
    for perm in permutations(remaining):
        # First leg: base → ball_0 → dropzone[ball_0]
        d = np.linalg.norm(base_pos[:2] - target_positions[perm[0], :2])
        dz0 = drop_zones[ball_zone_assignments[perm[0]]]
        d += np.linalg.norm(target_positions[perm[0], :2] - dz0[:2])
        # Subsequent legs: dropzone[ball_{i-1}] → ball_i → dropzone[ball_i]
        for i in range(1, len(perm)):
            prev_dz = drop_zones[ball_zone_assignments[perm[i - 1]]]
            d += np.linalg.norm(prev_dz[:2] - target_positions[perm[i], :2])
            cur_dz = drop_zones[ball_zone_assignments[perm[i]]]
            d += np.linalg.norm(target_positions[perm[i], :2] - cur_dz[:2])
        if d < best_dist:
            best_dist = d
            best_order = list(perm)

    return best_order[0], best_order


def run_multi_drop_pick_and_place_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                                        init_joint_angles, target_finger_angles,
                                        lower_limits, upper_limits,
                                        target_ee_positions,
                                        drop_zones=None,
                                        ball_zone_assignments=None,
                                        torque_noise_factory=None,
                                        per_target_timeout=None):
    """Expert multi-drop-zone pick-and-place episode.

    Each ball is assigned to one of NUM_DROP_ZONES drop zones. After delivering
    ball_i to its zone, the robot starts the next pick from that zone's position.

    Args:
        target_ee_positions: list of (3,) ball positions
        drop_zones: list of (3,) drop zone positions (default: DROP_ZONES_DEFAULT)
        ball_zone_assignments: list mapping ball index -> drop zone index
            (if None, randomly assigned)
        torque_noise_factory: callable() -> TorqueNoise or None
        per_target_timeout: max sim-time per target leg (seconds)

    Returns (combined_data, selector_samples, per_ball_results).
    """
    if drop_zones is None:
        drop_zones = DROP_ZONES_DEFAULT
    drop_zones = [np.array(dz) for dz in drop_zones]

    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    ee_link_id = find_ee_link_id(client, rid)

    n_balls = len(target_ee_positions)
    all_targets = np.array(target_ee_positions)

    if ball_zone_assignments is None:
        rng = np.random.default_rng()
        ball_zone_assignments = [int(rng.integers(0, len(drop_zones))) for _ in range(n_balls)]

    # Spawn balls color-coded by zone
    grip_states = []
    for i in range(n_balls):
        color = DROP_ZONE_COLORS[ball_zone_assignments[i]]
        ball_id = spawn_ball(client, all_targets[i], color=color)
        grip_states.append(GripState(ball_id=ball_id, drop_zone_idx=ball_zone_assignments[i]))

    # Spawn drop zone visuals
    for zi, dz in enumerate(drop_zones):
        color = DROP_ZONE_COLORS[zi]
        vis = client.createVisualShape(p.GEOM_SPHERE, radius=0.08,
                                        rgbaColor=[color[0], color[1], color[2], 0.5])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=dz.tolist())
        client.addUserDebugPoints(
            pointPositions=[dz.tolist()],
            pointColorsRGB=[[color[0], color[1], color[2]]],
            pointSize=14,
        )

    remaining = set(range(n_balls))
    selector_samples = []
    per_ball_results = []
    all_target_data = []
    episode_time = 0.0
    time_step = sim.GetTimeStep()

    while remaining:
        # --- Decision point: expert picks next ball ---
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        candidates_obs, slot_order = build_selector_scene_multi_drop(
            base_pos, base_bearing, all_targets, remaining,
            drop_zones, ball_zone_assignments)
        chosen_idx, _ = shortest_path_expert_multi_drop(
            base_pos, all_targets, remaining, drop_zones, ball_zone_assignments)

        selector_samples.append({
            "candidates": candidates_obs,
            "chosen_idx": chosen_idx,
            "slot_order": slot_order,
            "episode_time": episode_time,
        })

        ball_pos = all_targets[chosen_idx]
        gs = grip_states[chosen_idx]
        assigned_drop = drop_zones[ball_zone_assignments[chosen_idx]]
        remaining.remove(chosen_idx)
        print(f"\n>>> Multi-drop PnP: ball {chosen_idx} at {ball_pos} "
              f"-> zone {ball_zone_assignments[chosen_idx]} at {assigned_drop} | "
              f"{len(remaining)} remaining")

        # --- Phase 1: nav_to_ball ---
        noise = torque_noise_factory() if torque_noise_factory else None
        old_max = max_steps
        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(int(per_target_timeout / time_step))

        data_nav_ball = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            np.array(sim.GetMotorAngles(0)), target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=ball_pos,
            torque_noise=noise,
        )
        episode_time += data_nav_ball["final_time"]

        # --- Phase 2: auto-grip check ---
        ee_pos = data_nav_ball["final_ee_pos"]
        ball_world_pos, _ = client.getBasePositionAndOrientation(gs.ball_id)
        grip_dist = np.linalg.norm(ee_pos - np.array(ball_world_pos))
        gripped = grip_dist < GRIP_TOLERANCE
        if gripped:
            auto_grip(client, rid, ee_link_id, gs)
            print(f"    Gripped ball {chosen_idx} (dist={grip_dist:.4f}m)")
        else:
            print(f"    Grip FAILED ball {chosen_idx} (dist={grip_dist:.4f}m)")

        # --- Phase 3: nav_to_assigned_drop_zone (only if gripped) ---
        data_nav_drop = None
        if gripped:
            noise2 = torque_noise_factory() if torque_noise_factory else None
            data_nav_drop = run_simulation_loop(
                sim, dyn_model, ee_frame_id, shoulder_frame_id,
                np.array(sim.GetMotorAngles(0)), target_finger_angles,
                lower_limits, upper_limits,
                target_ee_pos=assigned_drop,
                torque_noise=noise2,
            )
            episode_time += data_nav_drop["final_time"]

            # --- Phase 4: auto-release check ---
            ee_pos_drop = data_nav_drop["final_ee_pos"]
            drop_dist = np.linalg.norm(ee_pos_drop - assigned_drop)
            auto_release(client, gs)
            delivered = drop_dist < GRIP_TOLERANCE
            if delivered:
                print(f"    Delivered ball {chosen_idx} to zone {ball_zone_assignments[chosen_idx]} "
                      f"(dist={drop_dist:.4f}m)")
            else:
                print(f"    Drop imprecise ball {chosen_idx} zone {ball_zone_assignments[chosen_idx]} "
                      f"(dist={drop_dist:.4f}m), released anyway")

        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(old_max)

        per_ball_results.append({
            "slot": chosen_idx,
            "ball_pos": ball_pos.tolist(),
            "drop_zone_idx": ball_zone_assignments[chosen_idx],
            "drop_zone_pos": assigned_drop.tolist(),
            "gripped": gripped,
            "delivered": gs.is_delivered,
            "grip_dist": grip_dist,
            "nav_ball_time": data_nav_ball["final_time"],
            "nav_drop_time": data_nav_drop["final_time"] if data_nav_drop else 0.0,
        })

        all_target_data.append(data_nav_ball)
        if data_nav_drop is not None:
            all_target_data.append(data_nav_drop)

        print(f"    Ball {chosen_idx}: gripped={gripped}  delivered={gs.is_delivered}")

    # --- Combine trajectory data ---
    combined_data = _combine_trajectory_data(all_target_data)
    combined_data["episode_time"] = episode_time
    combined_data["per_ball_results"] = per_ball_results

    n_gripped = sum(1 for r in per_ball_results if r["gripped"])
    n_delivered = sum(1 for r in per_ball_results if r["delivered"])
    print(f"\n>>> Multi-drop PnP done: {n_gripped}/{n_balls} gripped, "
          f"{n_delivered}/{n_balls} delivered, {episode_time:.2f}s total")

    return combined_data, selector_samples, per_ball_results


def nearest_neighbor_expert(base_pos, target_positions, remaining_indices):
    """Pick the nearest remaining target by XY distance to the robot.

    Args:
        base_pos: (3,) robot base position
        target_positions: (N, 3) all target positions
        remaining_indices: set of valid target slot indices
    Returns:
        int, index of the chosen target slot
    """
    best_idx = None
    best_dist = np.inf
    for idx in remaining_indices:
        dist = np.linalg.norm(base_pos[:2] - target_positions[idx, :2])
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def shortest_path_expert(base_pos, target_positions, remaining_indices):
    """Solve TSP over remaining targets by brute-force to minimise total XY distance.

    Starting from base_pos, evaluate every permutation of remaining targets and
    return the full visit order (list) plus the first target to visit.

    Args:
        base_pos: (3,) robot base position
        target_positions: (N, 3) all target positions
        remaining_indices: set of valid target slot indices
    Returns:
        chosen_idx: int, the first target to visit in the optimal order
        full_order: list[int], complete optimal visit sequence
    """
    remaining = list(remaining_indices)
    if len(remaining) == 1:
        return remaining[0], remaining

    best_dist = np.inf
    best_order = None
    for perm in permutations(remaining):
        d = np.linalg.norm(base_pos[:2] - target_positions[perm[0], :2])
        for i in range(len(perm) - 1):
            d += np.linalg.norm(target_positions[perm[i], :2] - target_positions[perm[i + 1], :2])
        if d < best_dist:
            best_dist = d
            best_order = list(perm)

    return best_order[0], best_order


def shortest_path_expert_pnp(base_pos, target_positions, remaining_indices, drop_zone):
    """Solve TSP for pick-and-place: robot returns to *drop_zone* between every ball.

    The real travel cost for a permutation (b0, b1, b2, ...) is:
      dist(base, b0) + dist(b0, drop) + dist(drop, b1) + dist(b1, drop) + ...

    After the first ball, every subsequent leg starts from the drop zone, so
    inter-ball distance is always dist(drop, b_i) + dist(b_i, drop).

    Args:
        base_pos: (3,) robot base position (starting point)
        target_positions: (N, 3) all ball positions
        remaining_indices: set of valid ball slot indices
        drop_zone: (3,) drop zone position
    Returns:
        chosen_idx: int, the first ball to visit
        full_order: list[int], complete optimal visit sequence
    """
    remaining = list(remaining_indices)
    if len(remaining) == 1:
        return remaining[0], remaining

    drop_xy = drop_zone[:2]
    best_dist = np.inf
    best_order = None
    for perm in permutations(remaining):
        # First leg: base → ball_0 → drop
        d = np.linalg.norm(base_pos[:2] - target_positions[perm[0], :2])
        d += np.linalg.norm(target_positions[perm[0], :2] - drop_xy)
        # Subsequent legs: drop → ball_i → drop
        for i in range(1, len(perm)):
            d += np.linalg.norm(drop_xy - target_positions[perm[i], :2])
            d += np.linalg.norm(target_positions[perm[i], :2] - drop_xy)
        if d < best_dist:
            best_dist = d
            best_order = list(perm)

    return best_order[0], best_order


def run_multi_target_loop(sim, dyn_model, ee_frame_id, shoulder_frame_id,
                          init_joint_angles, target_finger_angles,
                          lower_limits, upper_limits,
                          target_ee_positions,
                          torque_noise_factory=None,
                          per_target_timeout=None):
    """Run an expert multi-target episode: visit each target in nearest-neighbor order.

    Args:
        sim, dyn_model, ...: same as run_simulation_loop
        target_ee_positions: list of target (x,y,z) arrays (any length)
        torque_noise_factory: callable() -> TorqueNoise or None, called per target
        per_target_timeout: max sim-time per target (seconds). None = use default max_steps.

    Returns:
        combined_data: dict with concatenated trajectory data from all targets
        selector_samples: list of dicts, one per decision point.  Each dict has:
            "candidates": dict mapping slot_idx -> (3,) base-frame observation
            "chosen_idx": int, the expert's pick (key into candidates)
            "episode_time": float
        per_target_results: list of dicts with per-target outcomes
    """
    n_targets = len(target_ee_positions)
    assert n_targets >= 1

    all_targets = np.array(target_ee_positions)  # (N, 3)

    remaining = set(range(n_targets))
    selector_samples = []
    per_target_results = []
    all_target_data = []
    episode_time = 0.0

    while remaining:
        # --- Decision point: expert picks nearest target ---
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])

        candidates = build_selector_candidates(base_pos, base_bearing, all_targets, remaining)
        chosen_idx, _ = shortest_path_expert(base_pos, all_targets, remaining)

        selector_samples.append({
            "candidates": candidates,
            "chosen_idx": chosen_idx,
            "episode_time": episode_time,
        })

        current_target = all_targets[chosen_idx]
        remaining.remove(chosen_idx)
        print(f"\n>>> Multi-target: visiting slot {chosen_idx} "
              f"({current_target}) | {len(remaining)} remaining")

        # --- Run single-target loop for this target ---
        noise = torque_noise_factory() if torque_noise_factory else None

        # Override max_steps for per-target timeout if given
        old_max_steps = max_steps
        if per_target_timeout is not None:
            time_step = sim.GetTimeStep()
            os.environ["ROBOENV_MAX_STEPS"] = str(int(per_target_timeout / time_step))

        data = run_simulation_loop(
            sim, dyn_model, ee_frame_id, shoulder_frame_id,
            np.array(sim.GetMotorAngles(0)), target_finger_angles,
            lower_limits, upper_limits,
            target_ee_pos=current_target,
            torque_noise=noise,
        )

        if per_target_timeout is not None:
            os.environ["ROBOENV_MAX_STEPS"] = str(old_max_steps)

        target_time = data["final_time"]
        episode_time += target_time
        final_err = np.linalg.norm(current_target - data["final_ee_pos"])
        reached = final_err < REACH_TOLERANCE

        per_target_results.append({
            "slot": chosen_idx,
            "target": current_target.tolist(),
            "reached": reached,
            "final_err": final_err,
            "time": target_time,
        })
        all_target_data.append(data)
        print(f"    Slot {chosen_idx}: err={final_err:.4f}m  reached={reached}  t={target_time:.2f}s")

    # --- Combine trajectory data across targets ---
    combined_data = _combine_trajectory_data(all_target_data)
    combined_data["episode_time"] = episode_time
    combined_data["per_target_results"] = per_target_results

    return combined_data, selector_samples, per_target_results


def _combine_trajectory_data(data_list):
    """Concatenate trajectory data dicts from multiple single-target runs."""
    if not data_list:
        return {}
    combined = {}
    # Arrays that can be concatenated along axis 0
    array_keys = [
        "time", "ee_pos", "ee_error", "base_pos", "torque_arm", "torque_applied",
        "wheel_cmds", "base_bearing", "target_ee_pos", "base_lin_vel",
        "base_ang_vel", "tau_gravity",
    ]
    # List-of-array keys (q_mes, qd_mes, q_d, qd_d)
    list_keys = ["q_mes", "qd_mes", "q_d", "qd_d"]

    for key in array_keys:
        arrays = [d[key] for d in data_list if key in d and len(d[key]) > 0]
        if arrays:
            # Offset time arrays so they are monotonically increasing
            if key == "time":
                offset = 0.0
                adjusted = []
                for arr in arrays:
                    adjusted.append(arr + offset)
                    offset += arr[-1] + (arr[1] - arr[0] if len(arr) > 1 else 0.001)
                combined[key] = np.concatenate(adjusted)
            else:
                combined[key] = np.concatenate(arrays)

    for key in list_keys:
        lists = [d[key] for d in data_list if key in d and len(d[key]) > 0]
        if lists:
            combined[key] = sum(lists, [])

    # Use the last run's final values
    last = data_list[-1]
    combined["final_base_pos"] = last["final_base_pos"]
    combined["final_ee_pos"] = last["final_ee_pos"]
    combined["final_q_mes"] = last["final_q_mes"]
    combined["final_time"] = sum(d["final_time"] for d in data_list)
    combined["target_finger_angles"] = last["target_finger_angles"]
    combined["reach_time"] = last.get("reach_time")

    return combined


def print_summary(data):
    """Print final simulation results."""
    target_finger_angles = data["target_finger_angles"]
    q_mes = data["final_q_mes"]
    print("==================================================================")
    print(f"Final Base Position          : {np.round(data['final_base_pos'][:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(data['final_ee_pos'], 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(TARGET_EE_POS - data['final_ee_pos']):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")


def plot_results(data):
    """Show EE tracking, error, and joint profile plots."""
    time_all = data["time"]
    ee_pos_all = data["ee_pos"]
    ee_error_all = data["ee_error"]

    # 1. Cartesian EE position tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=TARGET_EE_POS[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in data["q_mes"]], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in data["q_d"]], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


def main():
    # Setup
    sim, client, rid, conf_file_name, cur_dir = setup_simulation()
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
    add_target_visual(client, TARGET_EE_POS)

    dyn_model, ee_frame_id, shoulder_frame_id, init_joint_angles, init_ee_pos, \
        lower_limits, upper_limits = setup_kinematics(sim, conf_file_name, cur_dir)

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(TARGET_EE_POS - init_ee_pos):.4f}")
    print("==================================================================")

    # Run
    data = run_simulation_loop(
        sim, dyn_model, ee_frame_id, shoulder_frame_id,
        init_joint_angles, target_finger_angles,
        lower_limits, upper_limits,
    )


    # Report
    print_summary(data)

    if headless:
        print(f"Headless run finished at t={data['final_time']:.2f}s")
        return

    plot_results(data)


if __name__ == '__main__':
    main()
