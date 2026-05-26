"""probelab: A library for training classifiers on LLM activations."""

from . import batching, datasets, masks, metrics, pool, probes, storage, tokenization, types, utils
from .activations import Activations
from .datasets import Dataset
from .tokenization import Tokens, tokenize_dataset, tokenize_dialogues

__version__ = "0.1.0"

__all__ = [
    "Activations",
    "batching",
    "Dataset",
    "Tokens",
    "tokenize_dataset",
    "tokenize_dialogues",
    "datasets",
    "masks",
    "metrics",
    "pool",
    "probes",
    "storage",
    "tokenization",
    "types",
    "utils",
]
