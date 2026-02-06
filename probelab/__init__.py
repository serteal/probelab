"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import datasets, masks, metrics, pool, probes, processing, types, utils

# Primary API
from .processing import Activations, collect_activations, tokenize_dataset

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "Activations",
    "collect_activations",
    "tokenize_dataset",
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
