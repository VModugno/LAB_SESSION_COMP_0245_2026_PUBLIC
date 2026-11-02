#!/usr/bin/env python3
"""Activation functions available to every network in the project.

Pick one by name (e.g. --activation gelu). Checkpoints store the name they were
trained with, so the sim always rebuilds a model with the right activation.
"""

import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

# Used when training a new model and no activation is given
DEFAULT_ACTIVATION = "gelu"

# Used when loading a checkpoint saved before activations were recorded
# (all of those were trained with ReLU)
LEGACY_ACTIVATION = "relu"


def get_activation(name):
    """Return a new activation layer for the given name."""
    if name not in ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. Available: {list(ACTIVATIONS)}")
    return ACTIVATIONS[name]()
