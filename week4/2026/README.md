# Week 4 — MLP on MNIST

Script: `mlp_mnist.py` — a small multi-layer perceptron (784 → 32 → 10) trained on MNIST with PyTorch.

## Prerequisites (once per machine)

1. Clone RoboEnv with submodules and install its pixi environment:

   ```bash
   git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
   cd RoboEnv && pixi install
   pixi run smoke-test   # → prints "simulation_and_control OK"
   ```

2. Install the `ml` environment (torch/torchvision) from the same manifest:

   ```bash
   pixi install -e ml
   ```

## Why a separate environment?

torch/torchvision do not fit into RoboEnv's default environment: conda-forge pytorch ships CUDA/MKL builds that clash with the OpenBLAS pin, and the PyPI wheels cover only win-64 / linux-64 / osx-arm64 at RoboEnv's Python version. RoboEnv's manifest therefore defines a separate `ml` environment — it inherits the full simulation stack plus torch/torchvision.

## Run

**Always launch through pixi** (never call `.pixi\envs\ml\python.exe` directly — a Windows OpenMP runtime setting is applied only during activation and torch crashes without it):

From the **RoboEnv root**:

```bash
pixi run -e ml python "<full path>/week_4/2026/mlp_mnist.py"
```

or activate once and work from this folder (recommended — the MNIST download lands in `./data`):

```bash
cd RoboEnv && pixi shell -e ml
cd "<full path>/week_4/2026"
python mlp_mnist.py
```

## What to expect

- First run downloads MNIST into `./data` (~12 MB) and prints download progress bars.
- Trains for **20 epochs** (CPU: a few minutes total), printing per-epoch loss/accuracy:

  ```
  Epoch 1/20, Loss: 0.4694, Accuracy: 87.31%
  ...
  Epoch 20/20, Loss: 0.0764, Accuracy: 97.80%
  ```

- Final evaluation:

  ```
  Test Loss: 0.1022, Test Accuracy: 96.87%
  ```

  (figures from a verified reference run; small variations are normal.)

## Stopping

The script terminates by itself after evaluation. There is no GUI.