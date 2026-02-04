"""probelab: A library for training classifiers on LLM activations."""

# Submodules (for pl.submodule.X access)
from . import coordination, datasets, masks, metrics, probes, transforms

# Primary API
from .logger import logger
from .pipeline import Pipeline
from .coordination import PipelineSet
from .processing import Activations, collect_activations
from .types import Label

__version__ = "0.1.0"

__all__ = [
    # Primary API (what 90% of users need)
    "Pipeline",
    "PipelineSet",
    "Activations",
    "collect_activations",
    "Label",
    "logger",
    # Submodule access (use transforms.pre.* and transforms.post.*)
    "transforms",
    "probes",
    "masks",
    "datasets",
    "metrics",
    "coordination",
]
