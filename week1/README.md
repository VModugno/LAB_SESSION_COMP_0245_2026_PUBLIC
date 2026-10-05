# Week 1 — Robot Simulation and Cartesian Control

This week introduces the pybullet + pinocchio simulation stack (`simulation_and_control`) that we will use throughout the course.

Scripts:

- `induction_simulator.py` — loads a robot arm in pybullet, prints its joint/link/dynamics info, and runs it with zero torques.
- `cartesian_kinematic_controller.py` — Cartesian differential kinematics + feedback linearization to track a sinusoidal end-effector reference.

## Prerequisites (once per machine)

1. Clone RoboEnv with submodules and install its pixi environment:

   ```bash
   git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
   cd RoboEnv && pixi install
   pixi run smoke-test   # → prints "simulation_and_control OK"
   ```

2. If pixi is missing: https://pixi.sh (Windows: `irm https://pixi.sh/install-pixi.ps1 | iex`).

## One-time setup for this folder

`configs/` and `models/` are intentionally not part of the lab repo. Copy or link them from your RoboEnv checkout:

**Windows (PowerShell):**

```powershell
$robo = "C:\path\to\RoboEnv"    # your RoboEnv checkout

Copy-Item "$robo\configs\pandaconfig.json"  .\configs\ -Force
Copy-Item "$robo\configs\elephantconfig.json" .\configs\ -Force
New-Item -ItemType Junction -Path .\models -Target "$robo\models"   # link: no 174 MB copy
# ...or a plain copy instead of the junction:
# Copy-Item "$robo\models" .\models -Recurse
```

**macOS / Linux:**

```bash
robo=/path/to/RoboEnv

mkdir -p configs
cp "$robo"/configs/{pandaconfig,elephantconfig}.json configs/
ln -s "$robo"/models models     # or: cp -r "$robo"/models .
```

## Run

From the **RoboEnv root** (pixi finds its manifest there):

```bash
pixi run python "<full path>/week1/induction_simulator.py"
pixi run python "<full path>/week1/cartesian_kinematic_controller.py"
```

or activate once and work from this folder:

```bash
cd RoboEnv && pixi shell
cd "<full path>/week1"
python induction_simulator.py
```

## What to expect

- A pybullet GUI window opens with the robot loaded (elephant arm for `induction_simulator`, Panda for `cartesian_kinematic_controller`).
- `induction_simulator` prints joint info, link dynamics, and pinocchio model info in the terminal, then steps the sim with zero torques.
- `cartesian_kinematic_controller` runs a closed control loop (~1 kHz) and prints the running simulation time; after you quit it shows per-joint position/velocity tracking plots.

## Stopping

Both scripts loop forever by design. Press **`q` inside the pybullet window** to exit cleanly (Ctrl+C in the terminal can leave the GUI process alive).

---

# Week 1–2 — Dynamic Regression

Script: `dynamic_regression.py` — excites the Panda arm with a sinusoidal joint reference under feedback-linearization control, records joint states/torques, and is the starting point for the dynamics-regression exercise (regressor stacking, least-squares parameter identification).

## Prerequisites (once per machine)

1. Clone RoboEnv with submodules and install its pixi environment:

   ```bash
   git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
   cd RoboEnv && pixi install
   pixi run smoke-test   # → prints "simulation_and_control OK"
   ```

## One-time setup for this folder

`configs/` and `models/` are intentionally not part of the lab repo. Copy or link them from your RoboEnv checkout:

**Windows (PowerShell):**

```powershell
$robo = "C:\path\to\RoboEnv"

Copy-Item "$robo\configs\pandaconfig.json" .\configs\ -Force
New-Item -ItemType Junction -Path .\models -Target "$robo\models"
```

**macOS / Linux:**

```bash
robo=/path/to/RoboEnv
mkdir -p configs && cp "$robo"/configs/pandaconfig.json configs/
ln -s "$robo"/models models
```

## Run

From the **RoboEnv root**:

```bash
pixi run python "<full path>/week1/dynamic_regression.py"
```

or activate once and work from this folder:

```bash
cd RoboEnv && pixi shell
cd "<full path>/week1"
python dynamic_regression.py
```

## What to expect

- The pybullet GUI opens and the Panda arm tracks the sinusoidal reference for 10 seconds while the script collects `q`, `qd`, `qdd`, and measured torques.
- Progress prints every step (`Current time in seconds: ...`), ending at `10.00`.

## Known quirk

After the 10-second data collection completes, the script crashes with `NameError: name 's' is not defined` — there is a **stray `s` character on line 91**, right in the TODO section you are about to implement. Delete that line before running; the data collection part is unaffected.

## Stopping

The script terminates by itself at `max_time = 10` s. The pybullet window closes with the process. The TODOs at the bottom are yours to implement.