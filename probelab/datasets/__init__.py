"""Pre-built datasets for probe training and evaluation.

Usage:
    >>> import probelab as pl
    >>> dataset = pl.datasets.load("circuit_breakers")
    >>> pl.datasets.list_datasets(category="deception")
    ['ai_audit', 'ai_liar', 'dolus_chat', ...]

    # Sparse probing datasets (arXiv:2502.16681)
    >>> ds = pl.datasets.load("sparse_probing_87_glue_cola")
    >>> from probelab.datasets.sparse_probing import load_sparse_probing_all
    >>> all_datasets = load_sparse_probing_all()
"""

from .base import Dataset
from .registry import Topic, info, list_categories, list_datasets, load

__all__ = [
    "Dataset",
    "Topic",
    "load",
    "list_datasets",
    "list_categories",
    "info",
]


def __getattr__(name: str):
    """Lazy import sparse probing utilities to avoid eager registration."""
    if name in (
        "load_sparse_probing_all",
        "load_sparse_probing",
        "list_sparse_probing_categories",
        "SPARSE_PROBING_DATASETS",
    ):
        from . import sparse_probing

        return getattr(sparse_probing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
