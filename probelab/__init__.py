"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import datasets, masks, metrics, pool, probes, processing, types, utils

# Primary API
from .processing import Acts, Activations, collect, collect_activations, load, tokenize_dataset

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "Acts",
    "Activations",
    "collect",
    "load",
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
