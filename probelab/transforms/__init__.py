"""Transform modules for activation and score processing.

This module provides composable transforms for use in Pipelines:
- pre.*: Transforms that operate on Activations (before probe)
- post.*: Transforms that operate on Scores (after probe)

Usage:
    from probelab.transforms import pre, post

    pipeline = Pipeline([
        ("select", pre.SelectLayer(16)),
        ("pool", pre.Pool(dim="sequence", method="mean")),
        ("probe", Logistic()),
        ("agg", post.Pool(method="max")),  # Optional post-probe pooling
    ])

Transform Types:
    pre.SelectLayer: Select single layer, removes LAYER axis
    pre.SelectLayers: Select multiple layers, keeps LAYER axis
    pre.Pool: Pool activations over sequence or layer dimension
    pre.Normalize: Feature normalization

    post.Pool: Pool scores over sequence dimension
    post.EMAPool: Exponential moving average aggregation (GDM paper)
    post.RollingPool: Rolling window aggregation (GDM paper)
"""

from . import post, pre
from .base import ActivationTransform, ScoreTransform

__all__ = [
    # Submodules
    "pre",
    "post",
    # Base classes
    "ActivationTransform",
    "ScoreTransform",
]
