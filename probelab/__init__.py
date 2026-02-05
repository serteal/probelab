"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import datasets, masks, metrics, probes, processing

# Primary API
from .processing import Activations, Scores
from .types import Label

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "Activations",
    "Scores",
    "Label",
    # Submodule access
    "processing",
    "probes",
    "masks",
    "datasets",
    "metrics",
]
