"""Pre-probe transformers that operate on Activations."""

from typing import Literal

import torch

from ..processing.activations import Activations, Axis
from .base import PreTransformer


class SelectLayer(PreTransformer):
    """Select a single layer, removing the LAYER axis.

    This transformer extracts activations from a specific layer,
    reducing the dimensionality from [layers, batch, seq, hidden]
    to [batch, seq, hidden].

    Args:
        layer: Layer index to select

    Example:
        >>> transform = SelectLayer(16)
        >>> acts = transform.transform(acts)  # [batch, seq, hidden]
    """

    def __init__(self, layer: int):
        self.layer = layer

    def transform(self, X: Activations) -> Activations:
        """Select the specified layer."""
        if not X.has_axis(Axis.LAYER):
            raise ValueError(
                f"Activations don't have LAYER axis. Available axes: {X.axes}"
            )
        return X.select(layers=self.layer)

    def __repr__(self) -> str:
        return f"SelectLayer(layer={self.layer})"


class SelectLayers(PreTransformer):
    """Select multiple layers, keeping the LAYER axis.

    This transformer extracts activations from specific layers while
    preserving the LAYER dimension.

    Args:
        layers: List of layer indices to select

    Example:
        >>> transform = SelectLayers([16, 20, 24])
        >>> acts = transform.transform(acts)  # [3, batch, seq, hidden]
    """

    def __init__(self, layers: list[int]):
        self.layers = layers

    def transform(self, X: Activations) -> Activations:
        """Select the specified layers."""
        if not X.has_axis(Axis.LAYER):
            raise ValueError(
                f"Activations don't have LAYER axis. Available axes: {X.axes}"
            )
        return X.select(layers=self.layers)

    def __repr__(self) -> str:
        return f"SelectLayers(layers={self.layers})"


class AggregateSequences(PreTransformer):
    """Pool over the sequence dimension, removing the SEQ axis.

    This transformer aggregates token-level activations into a single
    sequence-level representation, reducing dimensionality from
    [batch, seq, hidden] to [batch, hidden].

    Args:
        method: Pooling method:
            - "mean": Average pooling over detected tokens
            - "max": Max pooling over detected tokens
            - "last_token": Use the last detected token

    Example:
        >>> transform = AggregateSequences("mean")
        >>> acts = transform.transform(acts)  # [batch, hidden]
    """

    def __init__(self, method: Literal["mean", "max", "last_token"] = "mean"):
        if method not in {"mean", "max", "last_token"}:
            raise ValueError(
                f"Invalid aggregation method: {method}. "
                f"Must be one of: 'mean', 'max', 'last_token'"
            )
        self.method = method

    def transform(self, X: Activations) -> Activations:
        """Pool over sequence dimension if present."""
        if X.has_axis(Axis.SEQ):
            return X.pool(dim="sequence", method=self.method)
        return X  # Already pooled

    def __repr__(self) -> str:
        return f"AggregateSequences(method='{self.method}')"


class AggregateLayers(PreTransformer):
    """Pool over the layer dimension, removing the LAYER axis.

    This transformer aggregates activations across multiple layers,
    reducing dimensionality from [layers, batch, seq, hidden] to
    [batch, seq, hidden].

    Args:
        method: Pooling method ("mean" or "max")

    Example:
        >>> transform = AggregateLayers("mean")
        >>> acts = transform.transform(acts)  # [batch, seq, hidden]
    """

    def __init__(self, method: Literal["mean", "max"] = "mean"):
        if method not in {"mean", "max"}:
            raise ValueError(
                f"Invalid aggregation method: {method}. Must be 'mean' or 'max'"
            )
        self.method = method

    def transform(self, X: Activations) -> Activations:
        """Pool over layer dimension if present."""
        if X.has_axis(Axis.LAYER):
            return X.pool(dim="layer", method=self.method)
        return X  # Already pooled

    def __repr__(self) -> str:
        return f"AggregateLayers(method='{self.method}')"


class Normalize(PreTransformer):
    """Normalize activations using training statistics.

    This transformer standardizes features by removing the mean and
    scaling to unit variance. Statistics are computed during fit()
    and applied during transform().

    The normalization preserves the Activations structure and metadata,
    only modifying the activation tensor values.

    Args:
        eps: Small value to avoid division by zero

    Example:
        >>> transform = Normalize()
        >>> transform.fit(train_acts)
        >>> train_acts_norm = transform.transform(train_acts)
        >>> test_acts_norm = transform.transform(test_acts)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, X: Activations, y=None) -> "Normalize":
        """Compute normalization statistics from activations.

        Args:
            X: Training activations
            y: Unused (for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        tensor = X.activations

        # Compute statistics over all but last dimension (features)
        # This handles any input shape: [batch, hidden], [batch, seq, hidden], etc.
        axes = tuple(range(tensor.ndim - 1))

        self.mean_ = tensor.mean(dim=axes, keepdim=True)
        self.std_ = tensor.std(dim=axes, keepdim=True).clamp(min=self.eps)
        self._fitted = True

        return self

    def transform(self, X: Activations) -> Activations:
        """Normalize activations using computed statistics.

        Args:
            X: Activations to normalize

        Returns:
            Normalized activations with same structure as input

        Raises:
            ValueError: If transform called before fit
        """
        if not self._fitted:
            raise ValueError("Normalize must be fitted before transform")

        # Normalize
        normalized = (X.activations - self.mean_) / self.std_

        # Return new Activations with same metadata
        return Activations(
            activations=normalized,
            axes=X.axes,
            layer_meta=X.layer_meta,
            sequence_meta=X.sequence_meta,
            batch_indices=X.batch_indices,
        )

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._fitted else "not fitted"
        return f"Normalize(eps={self.eps}, {fitted_str})"
