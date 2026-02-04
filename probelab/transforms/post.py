"""Post-probe transforms that operate on Scores.

These transforms are applied after the probe in a Pipeline.
All transforms inherit from ScoreTransform and have the signature:
    Scores → Scores

Usage:
    from probelab.transforms import post

    pipeline = Pipeline([
        ("select", pre.SelectLayer(16)),
        ("probe", Logistic()),  # Token-level predictions
        ("pool", post.Pool(method="mean")),  # Aggregate to sequence-level
    ])
"""

import torch
import torch.nn.functional as F

from ..processing.scores import ScoreAxis, Scores
from ..types import AggregationMethod
from ..utils.validation import check_scores
from .base import ScoreTransform


class Pool(ScoreTransform):
    """Pool scores over the sequence dimension.

    Aggregates token-level scores to sequence-level using the specified
    pooling method.

    Args:
        method: Pooling method
            - "mean": Average pooling
            - "max": Max pooling
            - "last_token": Use last valid token

    Example:
        >>> # Aggregate token predictions to sequence level
        >>> pool = post.Pool(method="mean")
        >>> seq_scores = pool.transform(token_scores)  # [batch, 2]
    """

    def __init__(self, method: str = "mean"):
        if isinstance(method, str):
            try:
                method_enum = AggregationMethod(method)
            except ValueError:
                raise ValueError(
                    f"method must be one of {[m.value for m in AggregationMethod]}, got {method!r}"
                )
        else:
            method_enum = method

        self.method = method_enum

    def transform(self, X: Scores) -> Scores:
        """Pool scores over the sequence dimension.

        Args:
            X: Token-level Scores to pool

        Returns:
            Sequence-level Scores with reduced dimensionality
        """
        check_scores(X, estimator_name="Pool")

        # If no SEQ axis, already pooled - return as-is
        if not X.has_axis(ScoreAxis.SEQ):
            return X

        return X.pool(dim="sequence", method=self.method)

    def __repr__(self) -> str:
        return f"Pool(method={self.method.value!r})"


class EMAPool(ScoreTransform):
    """Exponential Moving Average pooling over sequence dimension.

    Computes EMA of scores at each position, then takes max.
    Improves long-context generalization for linear probes.

    From GDM paper (Section 3.1.2):
        EMA_0 = 0
        EMA_j = alpha * score_j + (1 - alpha) * EMA_{j-1}
        output = max_j EMA_j

    Args:
        alpha: EMA decay factor (default: 0.5, from paper).
               Higher values give more weight to recent tokens.

    Example:
        >>> # Train linear probe with mean pooling, use EMA at inference
        >>> pipeline = Pipeline([
        ...     ("select", pre.SelectLayer(16)),
        ...     ("probe", Logistic()),  # Token-level training
        ...     ("ema", post.EMAPool(alpha=0.5)),  # EMA + max aggregation
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
        check_scores(X, estimator_name="EMAPool")

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
            tokens_per_sample = torch.full(
                (batch_size,), seq_len, device=scores.device, dtype=torch.long
            )

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
            tokens_per_sample == 0,
            torch.zeros_like(max_scores),
            max_scores,
        )

        # Convert to 2-class probabilities
        probs = torch.stack([1 - max_scores, max_scores], dim=-1)

        return Scores.from_sequence_scores(probs, X.batch_indices)

    def __repr__(self) -> str:
        return f"EMAPool(alpha={self.alpha})"


class RollingPool(ScoreTransform):
    """Rolling window mean pooling over sequence dimension.

    Computes mean within sliding windows, then takes max across windows.
    Useful for long-context inputs where signal may be localized.

    From GDM paper (Section 3.2.2):
        rolling_mean_t = mean(scores[t-w+1:t])
        output = max_t rolling_mean_t

    Args:
        window_size: Size of rolling window (default: 10)

    Example:
        >>> pipeline = Pipeline([
        ...     ("select", pre.SelectLayer(16)),
        ...     ("probe", Logistic()),
        ...     ("rolling", post.RollingPool(window_size=10)),
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
        check_scores(X, estimator_name="RollingPool")

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
