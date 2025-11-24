"""
probelib: A library for training classifiers on LLM activations.

This library provides tools for:
- Dataset handling
- Activation collection using hooked and pruned models
- Pipeline-based preprocessing and probe training
- Standard evaluation metrics
"""

from . import datasets, masks, metrics, preprocessing, probes, processing, scripts
from .models import HookedModel
from .pipeline import Pipeline
from .processing import Activations, ActivationCollector, collect_activations
from .types import Dialogue, Label, Message
from .visualization import print_metrics, visualize_mask

__version__ = "0.1.0"

__all__ = [
    "Message",
    "Label",
    "Dialogue",
    "collect_activations",
    "ActivationCollector",
    "HookedModel",
    "Activations",
    "Pipeline",
    "print_metrics",
    "visualize_mask",
    "scripts",
    "probes",
    "preprocessing",
    "datasets",
    "processing",
    "metrics",
    "masks",
]
