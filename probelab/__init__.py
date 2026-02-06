"""probelab: A library for training classifiers on LLM activations."""

from . import datasets, masks, metrics, pool, probes, processing, types, utils
from .processing import Activations, collect_activations, tokenize_dataset

__version__ = "0.1.0"

__all__ = [
    "Activations",
    "collect_activations",
    "tokenize_dataset",
    "datasets",
    "masks",
    "metrics",
    "pool",
    "probes",
    "processing",
    "types",
    "utils",
]
