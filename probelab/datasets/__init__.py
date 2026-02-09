"""Pre-built datasets for probe training and evaluation.

Usage:
    >>> import probelab as pl
    >>> dataset = pl.datasets.load("circuit_breakers")
    >>> pl.datasets.list_datasets(category="deception")
    ['ai_audit', 'ai_liar', 'dolus_chat', ...]
"""

from .base import Dataset
from .registry import Topic, info, list_categories, list_datasets, load

__all__ = ["Dataset", "Topic", "load", "list_datasets", "list_categories", "info"]
