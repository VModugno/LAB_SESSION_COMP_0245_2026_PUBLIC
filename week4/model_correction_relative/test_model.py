import os
import numpy as np
import time
from matplotlib import pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, SinusoidalReference, CartesianDiffKin
import torch
from train_model import NeuralNetwork


def feedback_lin_ctrl(dyn_model, q_, qd_, q_d, qd_d, kp, kd):
    """
    Perform feedback linearization control on a robotic system.

    Parameters:
    - dyn_model (pin_wrapper): The dynamics model of the robot encapsulated within a 'pin_wrapper' object,
                               which provides methods for computing robot dynamics such as mass matrices,
                               Coriolis forces, etc.
    - q_ (numpy.ndarray): Measured positions of the robot's joints, indicating the current actual positions
                          as measured by sensors or estimated by observers.
    - qd_ (numpy.ndarray): Measured velocities of the robot's joints, reflecting the current actual velocities.
    - q_d (numpy.ndarray): Desired positions for the robot's joints set by a trajectory generator or a higher-level
                           controller, dictating target positions.
    - qd_d (numpy.ndarray): Desired velocities for the robot's joints, specifying the rate at which the joints
                            should move towards their target positions.
    - kp (float or numpy.ndarray): Proportional gain(s) for the control system, which can be a uniform value across
                                   all joints or unique for each joint, adjusting the response to position error.
    - kd (float or numpy.ndarray): Derivative gain(s), similar to kp, affecting the response to velocity error and
                                   aiding in system stabilization by damping oscillations.

    Returns:
    None

    This function computes the control inputs necessary to achieve desired joint positions and velocities by
    applying feedback linearization, using the robot's dynamic model to appropriately compensate for its
    inherent dynamics. The control law implemented typically combines proportional-derivative (PD) control
    with dynamic compensation to achieve precise and stable motion.
    """
    # here i want to add a control that if the kp gains are a numpy vector ill check if the size is good
    #  and then i will create the P matrix and D matrix using the vector a diagonal of the matrices
    # if the size is not good i will raise an error
    # if the kp is a scalar i will create a diagonal matrix with the scalar value
    # same for kd
    if isinstance(kp, np.ndarray):
        if kp.size != dyn_model.getNumberofActuatedJoints():
            raise ValueError("The size of the kp vector is not correct") 
    else:
        kp = np.array([kp] * dyn_model.getNumberofActuatedJoints())
    
    P = np.diag(kp)
    
    if isinstance(kd, np.ndarray):
        if kd.size != dyn_model.getNumberofActuatedJoints():
            raise ValueError("The size of the kd vector is not correct")
    else:
        kd = np.array([kd] * dyn_model.getNumberofActuatedJoints())
    D = np.diag(kd)
 
    # here i compute the feeback linearization tau // the reordering is already done inside compute all teamrs
    dyn_model.ComputeAllTerms(q_, qd_)
    # reoder measurements in pinocchio order
    q_mes = dyn_model.ReoderJoints2PinVec(q_,"pos")
    qd_mes = dyn_model.ReoderJoints2PinVec(qd_,"vel")
    # control 
    u = P @ (q_d - q_mes) + D @ (qd_d - qd_mes)
    n = dyn_model.res.c + dyn_model.res.g
    
    tau_FL = dyn_model.res.M @ u + n

    tau_FL = dyn_model._FromPinToExtVec(tau_FL)
  
    return tau_FL, u


# Configuration for the simulation
conf_file_name = "pandaconfig.json"  # Configuration file for the robot
cur_dir = os.path.dirname(os.path.abspath(__file__))
# Initialize simulation interface
sim = pb.SimInterface(
    conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)

# Get active joint names from the simulation
ext_names = sim.getNameActiveJoints()
# Adjust the shape for compatibility
ext_names = np.expand_dims(np.array(ext_names), axis=0)

source_names = ["pybullet"]  # Define the source for dynamic modeling

# Create a dynamic model of the robot
dyn_model = PinWrapper(conf_file_name, "pybullet",
                       ext_names, source_names, False, 0, cur_dir)
num_joints = dyn_model.getNumberofActuatedJoints()

init_joint_angles = sim.GetInitMotorAngles()

print(f"Initial joint angles: {init_joint_angles}")


# single joint tuning
# episode_duration is specified in seconds
def simulate_with_given_pid_values(sim_, kp, kd, reference_trajectory, model, episode_duration=10):

    # here we reset the simulator each time we start a new test
    sim_.ResetPose()

    time_step = sim_.GetTimeStep()
    current_time = 0
    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors

    # Initialize data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all,  = [], [], [], []
    qdd_est_all, u_all = [], []

    steps = int(episode_duration/time_step)
    prev_u = None
    # testing loop
    for i in range(steps):
        # measure current state
        q_mes = sim_.GetMotorAngles(0)
        qd_mes = sim_.GetMotorVelocities(0)
        qdd_est = sim_.ComputeMotorAccelerationTMinusOne(0)
        # Compute sinusoidal reference trajectory
        # Ensure q_init is within the range of the amplitude
        q_d, qd_d = reference_trajectory.get_values(current_time)

        # Control command
        tau_cmd, u = feedback_lin_ctrl(
            dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)  # Zero torque command
        
        x_arr = np.hstack((q_mes, qd_mes)).reshape((1, 14))
        tensor_x = torch.Tensor(x_arr)
        err_pred = None
        with torch.no_grad():
            x = tensor_x.to(device)
            pred = model(x)
            err_pred = pred.cpu().detach().numpy()
        adjusted_tau = tau_cmd - err_pred.flatten()
        # Set the torque command do this before step simulation_and_control
        cmd.SetControlCmd(adjusted_tau, ["torque"]*len(q_mes))
        sim_.Step(cmd, "torque")  # Simulation step with torque command

        # Exit logic with 'q' key
        keys = sim_.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim_.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Store data for plotting
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        q_d_all.append(q_d)
        qd_d_all.append(qd_d)
        qdd_est_all.append(qdd_est)
        if prev_u is None:
            u_all.append(u)
        else:
            u_all.append(prev_u)
            prev_u = u

        # get real time
        current_time += time_step

    return q_mes_all, q_d_all, qd_mes_all, qd_d_all, qdd_est_all, u_all, time_step


if __name__ == '__main__':
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    model = NeuralNetwork().to(device)
    model_path = "model.pth"
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    test_duration = 10  # in seconds

    plot=True

    Kps = np.array([13.2473747,  9.61771559,  6.81393286,  1.20059412, 13.26144613, 13.00427159,
                    13.24737841])
    Kds = np.array([2.36536607, 2.00349038, 1.70331287, 0.75029629, 2.36787857, 2.70895233,
                    2.36536673])
    
    Kps[:] = 50


    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi/4, np.pi/6, np.pi/4, np.pi/4, np.pi /
                4, np.pi/4, np.pi/4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    # Example frequencies for joints
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())

    q_mes_all, q_d_all, qd_mes_all, qd_d_all, qdd_est_all, u_all, time_step = simulate_with_given_pid_values(
        sim, Kps, Kds, ref, model, episode_duration=10)

    q_mes_all_np = np.array(q_mes_all)
    q_des_np = np.array(q_d_all)
    qd_mes_np = np.array(qd_mes_all)
    qd_d_np = np.array(qd_d_all)
    qdd_est_np = np.array(qdd_est_all)
    u_all_np = np.array(u_all)

    if plot:
        for i in range(num_joints):
            
            fig, axes = plt.subplots(3, 1, sharex=True)
            fig.suptitle(f"Joint State vs Desired, Joint {i}")
            axes[0].plot(np.arange(np.shape(q_mes_all_np)[
                    0])*time_step, q_mes_all_np[:, i], label="measured_position")
            axes[0].plot(np.arange(np.shape(q_mes_all_np)[
                    0])*time_step, q_des_np[:, i], label="desired_position")
            axes[0].legend()
            axes[0].set_ylabel("position")
            
            axes[1].plot(np.arange(np.shape(qd_mes_np)[
                    0])*time_step, qd_mes_np[:, i], label="measured_velocity")
            axes[1].plot(np.arange(np.shape(qd_mes_np)[
                    0])*time_step, qd_d_np[:, i], label="desired_velocity")
            axes[1].legend()
            axes[1].set_xlabel("time")
            axes[1].set_ylabel("velocity")

            axes[2].plot(np.arange(np.shape(qd_mes_np)[
                    0])*time_step, qdd_est_np[:, i], label="measured_torque")
            axes[2].plot(np.arange(np.shape(qd_mes_np)[
                    0])*time_step, u_all_np[:, i], label="calculated_torque")
            axes[2].legend()
            axes[2].set_xlabel("time")
            axes[2].set_ylabel("torque")
            plt.show()
    
    dataset = np.zeros((6, *q_mes_all_np.shape))
    dataset[0] = q_mes_all_np
    dataset[1] = q_d_all
    dataset[2] = qd_mes_all
    dataset[3] = qd_d_all
    dataset[4] = qdd_est_all
    dataset[5] = u_all

    with open(dataset_name, "wb") as file_handle:
        np.save(file_handle, dataset)