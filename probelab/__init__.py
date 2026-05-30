"""probelab: A library for training probes and activation monitors on LLM activations."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from . import (
    batching,
    collection,
    datasets,
    masks,
    metrics,
    pool,
    probes,
    storage,
    tokenization,
    types,
    utils,
)
from .activations import Activations
from .datasets import Dataset
from .tokenization import Tokens, tokenize_dataset, tokenize_dialogues

try:
    # Version is derived from the git tag at build time (hatch-vcs) and read
    # from the installed package metadata at runtime.
    __version__ = _pkg_version("probelab")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "Activations",
    "batching",
    "collection",
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
    "__version__",
]
