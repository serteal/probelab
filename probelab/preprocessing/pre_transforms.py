"""Pre-probe transformers that operate on Activations."""

from typing import overload

import torch
import torch.nn.functional as F

from ..processing.activations import Activations, Axis
from ..processing.scores import ScoreAxis, Scores
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


class EMAPool(PreTransformer):
    """Exponential Moving Average pooling over sequence dimension.

    Computes EMA of scores at each position, then takes max.
    Improves long-context generalization for linear probes.

    From GDM paper (Section 3.1.2):
        EMA_0 = 0
        EMA_j = alpha * score_j + (1 - alpha) * EMA_{j-1}
        output = max_j EMA_j

    This is a post-probe transformer that operates on Scores objects
    (token-level predictions) and aggregates them to sequence-level.

    Args:
        alpha: EMA decay factor (default: 0.5, from paper).
               Higher values give more weight to recent tokens.

    Example:
        >>> # Train linear probe with mean pooling, use EMA at inference
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", Logistic()),  # Token-level training
        ...     ("ema", EMAPool(alpha=0.5)),  # EMA + max aggregation
        ... ])
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def transform(self, X: Scores) -> Scores:
        """Apply EMA pooling over sequence dimension.

        Args:
            X: Token-level Scores with shape [batch, seq, 2]

        Returns:
            Sequence-level Scores with shape [batch, 2]
        """
        if not isinstance(X, Scores):
            raise TypeError(f"EMAPool only works on Scores, got {type(X).__name__}")

        # Only works on token-level Scores
        if not X.has_axis(ScoreAxis.SEQ):
            return X  # Already sequence-level

        # Extract positive class probabilities [batch, seq]
        scores = X.scores[:, :, 1]  # Positive class proba
        batch_size, seq_len = scores.shape

        # Get mask for valid tokens
        if X.tokens_per_sample is not None:
            seq_indices = torch.arange(seq_len, device=scores.device)
            tokens_per_sample = X.tokens_per_sample.to(scores.device)
            mask = seq_indices.unsqueeze(0) < tokens_per_sample.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, device=scores.device, dtype=torch.bool)

        # Compute EMA along sequence dimension
        # Paper formula: EMA_0 = 0, EMA_j = alpha * score_j + (1-alpha) * EMA_{j-1}
        # So first observation: EMA_1 = alpha * score_0 + (1-alpha) * 0 = alpha * score_0
        ema = torch.zeros(batch_size, seq_len, device=scores.device, dtype=scores.dtype)
        ema[:, 0] = self.alpha * scores[:, 0] * mask[:, 0].float()

        for j in range(1, seq_len):
            ema[:, j] = (
                self.alpha * scores[:, j] + (1 - self.alpha) * ema[:, j - 1]
            ) * mask[:, j].float() + ema[:, j - 1] * (~mask[:, j]).float()

        # Apply mask and take max
        ema_masked = ema.masked_fill(~mask, float("-inf"))
        max_scores = ema_masked.max(dim=1).values  # [batch]

        # Handle edge case where all tokens are masked
        max_scores = torch.where(
            tokens_per_sample == 0 if X.tokens_per_sample is not None else torch.zeros(batch_size, dtype=torch.bool, device=scores.device),
            torch.zeros_like(max_scores),
            max_scores,
        )

        # Convert to 2-class probabilities
        probs = torch.stack([1 - max_scores, max_scores], dim=-1)

        return Scores.from_sequence_scores(probs, X.batch_indices)

    def __repr__(self) -> str:
        return f"EMAPool(alpha={self.alpha})"


class RollingPool(PreTransformer):
    """Rolling window mean pooling over sequence dimension.

    Computes mean within sliding windows, then takes max across windows.
    Useful for long-context inputs where signal may be localized.

    From GDM paper (Section 3.2.2):
        rolling_mean_t = mean(scores[t-w+1:t])
        output = max_t rolling_mean_t

    This is a post-probe transformer that operates on Scores objects
    (token-level predictions) and aggregates them to sequence-level.

    Args:
        window_size: Size of rolling window (default: 10)

    Example:
        >>> pipeline = Pipeline([
        ...     ("select", SelectLayer(16)),
        ...     ("probe", Logistic()),
        ...     ("rolling", RollingPool(window_size=10)),
        ... ])
    """

    def __init__(self, window_size: int = 10):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self.window_size = window_size

    def transform(self, X: Scores) -> Scores:
        """Apply rolling mean pooling over sequence dimension.

        Args:
            X: Token-level Scores with shape [batch, seq, 2]

        Returns:
            Sequence-level Scores with shape [batch, 2]
        """
        if not isinstance(X, Scores):
            raise TypeError(f"RollingPool only works on Scores, got {type(X).__name__}")

        # Only works on token-level Scores
        if not X.has_axis(ScoreAxis.SEQ):
            return X  # Already sequence-level

        # Extract positive class probabilities [batch, seq]
        scores = X.scores[:, :, 1]  # Positive class proba
        batch_size, seq_len = scores.shape
        w = self.window_size

        # Get mask for valid tokens
        if X.tokens_per_sample is not None:
            seq_indices = torch.arange(seq_len, device=scores.device)
            tokens_per_sample = X.tokens_per_sample.to(scores.device)
            mask = seq_indices.unsqueeze(0) < tokens_per_sample.unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, device=scores.device, dtype=torch.bool)

        # Apply mask to scores
        masked_scores = scores * mask.float()
        masked_counts = mask.float()

        # Use cumsum for efficient rolling window computation
        # Cumulative sums
        cum_scores = torch.cumsum(masked_scores, dim=1)
        cum_counts = torch.cumsum(masked_counts, dim=1)

        # Pad for boundary handling (rolling window starting from position 0)
        cum_scores_padded = F.pad(cum_scores, (w, 0), value=0)
        cum_counts_padded = F.pad(cum_counts, (w, 0), value=0)

        # Rolling sums: roll[t] = cum[t] - cum[t-w]
        roll_scores = cum_scores_padded[:, w:] - cum_scores_padded[:, :-w]
        roll_counts = cum_counts_padded[:, w:] - cum_counts_padded[:, :-w]

        # Rolling means (avoid div by zero)
        rolling_means = roll_scores / roll_counts.clamp(min=1)

        # Mask invalid windows (no tokens in window) and take max
        valid_window_mask = roll_counts > 0
        rolling_means_masked = rolling_means.masked_fill(~valid_window_mask, float("-inf"))
        max_scores = rolling_means_masked.max(dim=1).values  # [batch]

        # Handle edge case where all windows are invalid
        max_scores = torch.where(
            ~valid_window_mask.any(dim=1),
            torch.zeros_like(max_scores),
            max_scores,
        )

        # Convert to 2-class probabilities
        probs = torch.stack([1 - max_scores, max_scores], dim=-1)

        return Scores.from_sequence_scores(probs, X.batch_indices)

    def __repr__(self) -> str:
        return f"RollingPool(window_size={self.window_size})"
