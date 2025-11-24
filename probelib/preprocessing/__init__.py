"""Preprocessing transformers for activation pipelines.

This module provides composable transformers that operate on Activations objects:
- PreTransformers: Activations → Activations (pre-probe transforms)
- PostTransformers: Scores → Scores (post-probe transforms)
"""

from .base import PostTransformer, PreTransformer
from .post_transforms import AggregateTokenScores
from .pre_transforms import (
    AggregateLayers,
    AggregateSequences,
    Normalize,
    SelectLayer,
    SelectLayers,
)

__all__ = [
    # Base classes
    "PreTransformer",
    "PostTransformer",
    # Pre-transforms
    "SelectLayer",
    "SelectLayers",
    "AggregateSequences",
    "AggregateLayers",
    "Normalize",
    # Post-transforms
    "AggregateTokenScores",
]
