"""Preprocessing transformers for activation pipelines.

This module provides composable transformers for use in Pipelines:
- SelectLayer/SelectLayers: Layer selection
- Pool: Unified pooling for Activations and Scores
- Normalize: Feature normalization
- EMAPool: Exponential moving average pooling (GDM paper)
- RollingPool: Rolling window mean pooling (GDM paper)
"""

from .base import PreTransformer
from .pre_transforms import (
    EMAPool,
    Normalize,
    Pool,
    RollingPool,
    SelectLayer,
    SelectLayers,
)

__all__ = [
    "PreTransformer",
    "SelectLayer",
    "SelectLayers",
    "Pool",
    "Normalize",
    "EMAPool",
    "RollingPool",
]
