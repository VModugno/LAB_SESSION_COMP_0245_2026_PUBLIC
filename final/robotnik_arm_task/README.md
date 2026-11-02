1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)
python run_policy_sim.py --target -1 -1 0.6 --headless --base_threshold 0.75 --gravity_comp gravity_comp.pt --checkpoint

  python evaluate.py --base_checkpoint 040_bc_base_gelu_e70_h512_l3_15min.pt \
    --arm_checkpoint 041_bc_arm_gelu_e70_h512_l3_6min.pt --gravity_comp gravity_comp.pt \
    --num_targets 20 --headless --base_threshold 0.75
  

python evaluate.py --base_checkpoint 040_bc_base_gelu_e70_h512_l3_15min.pt \
    --arm_checkpoint 041_bc_arm_gelu_e70_h512_l3_6min.pt \
    --num_targets 10 --headless --base_threshold 0.75