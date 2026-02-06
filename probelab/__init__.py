"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import datasets, masks, metrics, pool, probes, processing, types, utils

# Primary API
from .processing import Activations

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "Activations",
    # Submodule access
    "processing",
    "probes",
    "masks",
    "datasets",
    "metrics",
    "pool",
    "utils",
    "types",
]
