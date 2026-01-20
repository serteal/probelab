"""High-level pipeline functions for training and evaluation."""

from .workflows import (
    evaluate_from_model,
    evaluate_pipelines,
    train_from_model,
    train_pipelines,
)

__all__ = [
    "train_pipelines",
    "train_from_model",
    "evaluate_pipelines",
    "evaluate_from_model",
]
