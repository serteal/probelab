"""Pre-probe transforms (Activations → Activations).

Usage:
    from probelab.transforms import pre
    pipeline = Pipeline([
        ("select", pre.SelectLayer(16)),
        ("pool", pre.Pool(dim="sequence", method="mean")),
        ("probe", Logistic()),
    ])
"""

from dataclasses import dataclass, field

import torch

from ..processing.activations import Activations, Axis
from ..types import AggregationMethod
from ..utils.validation import check_activations
from .base import ActivationTransform


@dataclass(frozen=True, slots=True)
class SelectLayer(ActivationTransform):
    """Select single layer, removes LAYER axis."""

    layer: int

    def transform(self, X: Activations) -> Activations:
        check_activations(X, require_layer=True, estimator_name="SelectLayer")
        return X.select(layer=self.layer)


@dataclass(frozen=True, slots=True)
class SelectLayers(ActivationTransform):
    """Select multiple layers, keeps LAYER axis."""

    layers: tuple[int, ...]

    def __init__(self, layers: list[int] | tuple[int, ...]):
        object.__setattr__(self, "layers", tuple(layers))

    def transform(self, X: Activations) -> Activations:
        check_activations(X, require_layer=True, estimator_name="SelectLayers")
        return X.select(layers=list(self.layers))

    def __repr__(self) -> str:
        return f"SelectLayers(layers={list(self.layers)})"


@dataclass(frozen=True, slots=True)
class Pool(ActivationTransform):
    """Pool over sequence or layer dimension."""

    dim: str
    method: AggregationMethod

    def __init__(self, dim: str, method: str | AggregationMethod = "mean"):
        valid_dims = {"sequence", "layer"}
        if dim not in valid_dims:
            raise ValueError(f"dim must be one of {valid_dims}, got {dim!r}")
        if isinstance(method, str):
            try:
                m = AggregationMethod(method)
            except ValueError:
                raise ValueError(f"method must be one of {[m.value for m in AggregationMethod]}, got {method!r}")
        else:
            m = method
        if dim == "layer" and m == AggregationMethod.LAST_TOKEN:
            raise ValueError("last_token method not supported for layer dimension")
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "method", m)

    def transform(self, X: Activations) -> Activations:
        axis = {"sequence": Axis.SEQ, "layer": Axis.LAYER}[self.dim]
        return X if not X.has_axis(axis) else X.pool(dim=self.dim, method=self.method)

    def __repr__(self) -> str:
        return f"Pool(dim={self.dim!r}, method={self.method.value!r})"


@dataclass
class Normalize(ActivationTransform):
    """Standardize features to zero mean, unit variance."""

    eps: float = 1e-8
    mean_: torch.Tensor | None = field(default=None, repr=False)
    std_: torch.Tensor | None = field(default=None, repr=False)
    _fitted: bool = field(default=False, repr=False)
    _frozen: bool = field(default=False, repr=False)

    def fit(self, X: Activations, y=None) -> "Normalize":
        if self._frozen:
            return self
        check_activations(X, estimator_name="Normalize")
        axes = tuple(range(X.activations.ndim - 1))
        self.mean_ = X.activations.mean(dim=axes, keepdim=True)
        self.std_ = X.activations.std(dim=axes, keepdim=True).clamp(min=self.eps)
        self._fitted = True
        return self

    def freeze(self) -> "Normalize":
        self._frozen = True
        return self

    def unfreeze(self) -> "Normalize":
        self._frozen = False
        return self

    def transform(self, X: Activations) -> Activations:
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
        status = "fitted" if self._fitted else "not fitted"
        if self._frozen:
            status += ", frozen"
        return f"Normalize(eps={self.eps}, {status})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Normalize) and self.eps == other.eps

    def __hash__(self) -> int:
        return hash(("Normalize", self.eps))
