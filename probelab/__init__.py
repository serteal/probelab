"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import datasets, metrics, probes
from . import masks  # Single-file module (not a package)

# Primary API
from .logger import logger
from .processing import Activations, Scores, collect_activations
from .types import Label
from .utils import Normalize

__version__ = "0.1.0"

__all__ = [
    # Primary API
    "Activations",
    "Scores",
    "collect_activations",
    "Label",
    "Normalize",
    "logger",
    # Submodule access
    "probes",
    "masks",
    "datasets",
    "metrics",
]
