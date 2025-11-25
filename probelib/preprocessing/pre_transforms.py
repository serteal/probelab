"""Pre-probe transformers that operate on Activations."""

from typing import overload

import torch

from ..processing.activations import Activations, Axis
from ..processing.scores import Scores
from ..types import AggregationMethod
from .base import PreTransformer


class SelectLayer(PreTransformer):
    """Select a single layer, removing the LAYER axis.

    This transformer extracts activations from a specific layer,
    reducing the dimensionality from [layers, batch, seq, hidden]
    to [batch, seq, hidden].

    Args:
        layer: Layer index to select.

    Example:
        >>> transform = SelectLayer(layer=16)
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
        return X.select(layer=self.layer)

    def __repr__(self) -> str:
        return f"SelectLayer(layer={self.layer})"


class SelectLayers(PreTransformer):
    """Select multiple layers, keeping the LAYER axis.

    This transformer extracts activations from specific layers while
    preserving the LAYER dimension.

    Args:
        layers: List of layer indices to select.

    Example:
        >>> transform = SelectLayers(layers=[16, 20, 24])
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


class Pool(PreTransformer):
    """Unified pooling transform for Activations and Scores.

    Pools over a specified dimension using the given method. Works as both
    a PreTransformer (on Activations) and PostTransformer (on Scores).

    Args:
        dim: Dimension to pool over
            - "sequence": Pool over sequence/token dimension
            - "layer": Pool over layer dimension (Activations only)
        method: Pooling method
            - "mean": Average pooling
            - "max": Max pooling
            - "last_token": Use last token (sequence dimension only)

    Examples:
        >>> # Pre-probe: pool activations over sequence
        >>> pool = Pool(dim="sequence", method="mean")
        >>> pooled_acts = pool.transform(activations)

        >>> # Post-probe: pool token scores to sequence scores
        >>> pool = Pool(dim="sequence", method="max")
        >>> seq_scores = pool.transform(token_scores)

        >>> # Pool over layers
        >>> pool = Pool(dim="layer", method="mean")
        >>> pooled_acts = pool.transform(multi_layer_acts)
    """

    def __init__(self, dim: str, method: str = "mean"):
        valid_dims = {"sequence", "layer"}
        if dim not in valid_dims:
            raise ValueError(f"dim must be one of {valid_dims}, got {dim!r}")

        if isinstance(method, str):
            try:
                method_enum = AggregationMethod(method)
            except ValueError:
                raise ValueError(
                    f"method must be one of {[m.value for m in AggregationMethod]}, got {method!r}"
                )
        else:
            method_enum = method

        if dim == "layer" and method_enum == AggregationMethod.LAST_TOKEN:
            raise ValueError("last_token method not supported for layer dimension")

        self.dim = dim
        self.method = method_enum

    @overload
    def transform(self, X: Activations) -> Activations: ...

    @overload
    def transform(self, X: Scores) -> Scores: ...

    def transform(self, X: Activations | Scores) -> Activations | Scores:
        """Pool over the specified dimension.

        Args:
            X: Activations or Scores to pool

        Returns:
            Pooled Activations or Scores with reduced dimensionality
        """
        if isinstance(X, Activations):
            return self._transform_activations(X)
        elif isinstance(X, Scores):
            return self._transform_scores(X)
        else:
            raise TypeError(f"Expected Activations or Scores, got {type(X).__name__}")

    def _transform_activations(self, X: Activations) -> Activations:
        if self.dim == "sequence":
            if not X.has_axis(Axis.SEQ):
                return X  # Already pooled
            return X.pool(dim="sequence", method=self.method)
        elif self.dim == "layer":
            if not X.has_axis(Axis.LAYER):
                return X  # Already pooled
            return X.pool(dim="layer", method=self.method)

    def _transform_scores(self, X: Scores) -> Scores:
        if self.dim == "layer":
            raise ValueError("Scores don't have a layer dimension")
        return X.pool(dim=self.dim, method=self.method)

    def __repr__(self) -> str:
        return f"Pool(dim={self.dim!r}, method={self.method.value!r})"


class Normalize(PreTransformer):
    """Normalize activations using training statistics.

    This transformer standardizes features by removing the mean and
    scaling to unit variance. Statistics are computed during fit()
    and applied during transform().

    The normalization preserves the Activations structure and metadata,
    only modifying the activation tensor values.

    Supports both batch and streaming (online) learning:
    - fit(): Compute statistics from entire dataset
    - partial_fit(): Update running statistics incrementally (Welford's algorithm)
    - freeze(): Lock statistics for remaining epochs (call after first epoch)

    Args:
        eps: Small value to avoid division by zero

    Example:
        >>> transform = Normalize()
        >>> transform.fit(train_acts)
        >>> train_acts_norm = transform.transform(train_acts)
        >>> test_acts_norm = transform.transform(test_acts)

    Example (streaming):
        >>> transform = Normalize()
        >>> for batch in activation_iterator:
        ...     batch_norm = transform.partial_fit(batch)
        >>> transform.freeze()  # Lock statistics for remaining epochs
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self._fitted = False
        # Online learning state
        self._n_samples_seen = 0
        self._frozen = False

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

    def partial_fit(self, X: Activations, y=None) -> "Normalize":
        """Update running statistics incrementally.

        Uses Welford's online algorithm for numerically stable
        incremental mean and variance computation.

        Args:
            X: Batch of activations
            y: Unused (for sklearn compatibility)

        Returns:
            self: The fitted transformer (sklearn convention)
        """
        if self._frozen:
            # Statistics locked, no update needed
            return self

        tensor = X.activations

        # Flatten to [samples, features] for statistics computation
        # Keep original shape for proper broadcasting during transform
        original_shape = tensor.shape
        flat = tensor.reshape(-1, tensor.shape[-1])
        batch_size = flat.shape[0]

        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)

        if self._n_samples_seen == 0:
            # First batch - initialize statistics
            self.mean_ = batch_mean.unsqueeze(0)
            # Store variance internally, compute std on demand
            self._running_var = batch_var
            self._n_samples_seen = batch_size
        else:
            # Welford's online algorithm for combining batch statistics
            n = self._n_samples_seen
            m = batch_size
            delta = batch_mean - self.mean_.squeeze(0)

            # Update mean
            new_mean = self.mean_.squeeze(0) + delta * m / (n + m)
            self.mean_ = new_mean.unsqueeze(0)

            # Update variance using parallel algorithm
            # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            weight_old = n / (n + m)
            weight_new = m / (n + m)
            new_var = (
                weight_old * self._running_var
                + weight_new * batch_var
                + weight_old * weight_new * delta**2
            )
            self._running_var = new_var
            self._n_samples_seen += m

        self.std_ = torch.sqrt(self._running_var).clamp(min=self.eps).unsqueeze(0)
        self._fitted = True

        return self

    def freeze(self) -> "Normalize":
        """Freeze statistics for remaining epochs.

        Call this after the first epoch to prevent non-stationarity
        as the probe's decision boundary shifts during training.

        Returns:
            self: For method chaining
        """
        self._frozen = True
        return self

    def unfreeze(self) -> "Normalize":
        """Unfreeze statistics to allow further updates.

        Returns:
            self: For method chaining
        """
        self._frozen = False
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

        normalized = (X.activations - self.mean_) / self.std_

        return Activations(
            activations=normalized,
            axes=X.axes,
            layer_meta=X.layer_meta,
            sequence_meta=X.sequence_meta,
            batch_indices=X.batch_indices,
        )

    def __repr__(self) -> str:
        status_parts = []
        if self._fitted:
            status_parts.append("fitted")
        else:
            status_parts.append("not fitted")
        if self._frozen:
            status_parts.append("frozen")
        status_str = ", ".join(status_parts)
        return f"Normalize(eps={self.eps}, {status_str})"
