"""Preprocessing transformers for activation pipelines.

This module provides composable transformers for use in Pipelines:
- SelectLayer/SelectLayers: Layer selection
- Pool: Unified pooling for Activations and Scores
- Normalize: Feature normalization
"""

from .base import PreTransformer
from .pre_transforms import (
    Normalize,
    Pool,
    SelectLayer,
    SelectLayers,
)

__all__ = [
    "PreTransformer",
    "SelectLayer",
    "SelectLayers",
    "Pool",
    "Normalize",
]
