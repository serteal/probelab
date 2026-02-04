"""Pre-probe transforms that operate on Activations.

These transforms are applied before the probe in a Pipeline.
All transforms inherit from ActivationTransform and have the signature:
    Activations → Activations

Usage:
    from probelab.transforms import pre

    pipeline = Pipeline([
        ("select", pre.SelectLayer(16)),
        ("pool", pre.Pool(dim="sequence", method="mean")),
        ("probe", Logistic()),
    ])
"""

from ..processing.activations import Activations, Axis
from ..types import AggregationMethod
from ..utils.validation import check_activations
from .base import ActivationTransform


class SelectLayer(ActivationTransform):
    """Select a single layer, removing the LAYER axis.

    This transformer extracts activations from a specific layer,
    reducing the dimensionality from [layers, batch, seq, hidden]
    to [batch, seq, hidden].

    Args:
        layer: Layer index to select.

    Example:
        >>> transform = pre.SelectLayer(layer=16)
        >>> acts = transform.transform(acts)  # [batch, seq, hidden]
    """

    def __init__(self, layer: int):
        self.layer = layer

    def transform(self, X: Activations) -> Activations:
        """Select the specified layer."""
        check_activations(X, require_layer=True, estimator_name="SelectLayer")
        return X.select(layer=self.layer)

    def __repr__(self) -> str:
        return f"SelectLayer(layer={self.layer})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectLayer):
            return NotImplemented
        return self.layer == other.layer

    def __hash__(self) -> int:
        return hash(("SelectLayer", self.layer))


class SelectLayers(ActivationTransform):
    """Select multiple layers, keeping the LAYER axis.

    This transformer extracts activations from specific layers while
    preserving the LAYER dimension.

    Args:
        layers: List of layer indices to select.

    Example:
        >>> transform = pre.SelectLayers(layers=[16, 20, 24])
        >>> acts = transform.transform(acts)  # [3, batch, seq, hidden]
    """

    def __init__(self, layers: list[int]):
        self.layers = layers

    def transform(self, X: Activations) -> Activations:
        """Select the specified layers."""
        check_activations(X, require_layer=True, estimator_name="SelectLayers")
        return X.select(layers=self.layers)

    def __repr__(self) -> str:
        return f"SelectLayers(layers={self.layers})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SelectLayers):
            return NotImplemented
        return self.layers == other.layers

    def __hash__(self) -> int:
        return hash(("SelectLayers", tuple(self.layers)))


class Pool(ActivationTransform):
    """Pool activations over a specified dimension.

    Reduces activations over the sequence or layer dimension using the
    given pooling method.

    Args:
        dim: Dimension to pool over
            - "sequence": Pool over sequence/token dimension
            - "layer": Pool over layer dimension
        method: Pooling method
            - "mean": Average pooling
            - "max": Max pooling
            - "last_token": Use last token (sequence dimension only)

    Example:
        >>> # Pool over sequence to get sample-level features
        >>> pool = pre.Pool(dim="sequence", method="mean")
        >>> pooled_acts = pool.transform(activations)

        >>> # Pool over layers
        >>> pool = pre.Pool(dim="layer", method="mean")
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

    def transform(self, X: Activations) -> Activations:
        """Pool over the specified dimension.

        Args:
            X: Activations to pool

        Returns:
            Pooled Activations with reduced dimensionality
        """
        # Validate based on the dimension we're pooling
        if self.dim == "sequence":
            # If no SEQ axis, already pooled - return as-is
            if not X.has_axis(Axis.SEQ):
                return X
            return X.pool(dim="sequence", method=self.method)
        elif self.dim == "layer":
            # If no LAYER axis, already pooled - return as-is
            if not X.has_axis(Axis.LAYER):
                return X
            return X.pool(dim="layer", method=self.method)
        else:
            raise ValueError(f"Unknown dimension: {self.dim}")

    def __repr__(self) -> str:
        return f"Pool(dim={self.dim!r}, method={self.method.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pool):
            return NotImplemented
        return self.dim == other.dim and self.method == other.method

    def __hash__(self) -> int:
        return hash(("Pool", self.dim, self.method))


class Normalize(ActivationTransform):
    """Normalize activations using training statistics.

    This transformer standardizes features by removing the mean and
    scaling to unit variance. Statistics are computed during fit()
    and applied during transform().

    The normalization preserves the Activations structure and metadata,
    only modifying the activation tensor values.

    Args:
        eps: Small value to avoid division by zero

    Example:
        >>> transform = pre.Normalize()
        >>> transform.fit(train_acts)
        >>> train_acts_norm = transform.transform(train_acts)
        >>> test_acts_norm = transform.transform(test_acts)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_ = None
        self.std_ = None
        self._fitted = False
        self._frozen = False

    def fit(self, X: Activations, y=None) -> "Normalize":
        """Compute normalization statistics from activations.

        Args:
            X: Training activations
            y: Unused (for sklearn compatibility)

        Returns:
            self: Fitted transformer
        """
        check_activations(X, estimator_name="Normalize")
        tensor = X.activations

        # Compute statistics over all but last dimension (features)
        # This handles any input shape: [batch, hidden], [batch, seq, hidden], etc.
        axes = tuple(range(tensor.ndim - 1))

        self.mean_ = tensor.mean(dim=axes, keepdim=True)
        self.std_ = tensor.std(dim=axes, keepdim=True).clamp(min=self.eps)
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

        check_activations(X, estimator_name="Normalize")
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Normalize):
            return NotImplemented
        # For fusion purposes, two Normalize with same eps are equivalent
        return self.eps == other.eps

    def __hash__(self) -> int:
        return hash(("Normalize", self.eps))
