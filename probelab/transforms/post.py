"""Post-probe transforms (Scores → Scores).

Usage:
    from probelab.transforms import post
    pipeline = Pipeline([
        ("select", pre.SelectLayer(16)),
        ("probe", Logistic()),
        ("pool", post.Pool(method="mean")),
    ])
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..processing.scores import ScoreAxis, Scores
from ..types import AggregationMethod
from ..utils.validation import check_scores
from .base import ScoreTransform


@dataclass(frozen=True, slots=True)
class Pool(ScoreTransform):
    """Pool scores over sequence dimension."""

    method: AggregationMethod

    def __init__(self, method: str | AggregationMethod = "mean"):
        if isinstance(method, str):
            try:
                m = AggregationMethod(method)
            except ValueError:
                raise ValueError(f"method must be one of {[m.value for m in AggregationMethod]}, got {method!r}")
        else:
            m = method
        object.__setattr__(self, "method", m)

    def transform(self, X: Scores) -> Scores:
        check_scores(X, estimator_name="Pool")
        return X if not X.has_axis(ScoreAxis.SEQ) else X.pool(dim="sequence", method=self.method)

    def __repr__(self) -> str:
        return f"Pool(method={self.method.value!r})"


@dataclass(frozen=True, slots=True)
class EMAPool(ScoreTransform):
    """EMA pooling: EMA_j = alpha * score_j + (1-alpha) * EMA_{j-1}, then max."""

    alpha: float = 0.5

    def __post_init__(self):
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    def transform(self, X: Scores) -> Scores:
        check_scores(X, estimator_name="EMAPool")
        if not X.has_axis(ScoreAxis.SEQ):
            return X

        scores = X.scores[:, :, 1]  # Positive class [batch, seq]
        batch_size, seq_len = scores.shape

        # Build mask
        if X.tokens_per_sample is not None:
            seq_idx = torch.arange(seq_len, device=scores.device)
            mask = seq_idx.unsqueeze(0) < X.tokens_per_sample.to(scores.device).unsqueeze(1)
            tokens_per_sample = X.tokens_per_sample.to(scores.device)
        else:
            mask = torch.ones(batch_size, seq_len, device=scores.device, dtype=torch.bool)
            tokens_per_sample = torch.full((batch_size,), seq_len, device=scores.device, dtype=torch.long)

        # Compute EMA
        ema = torch.zeros_like(scores)
        ema[:, 0] = self.alpha * scores[:, 0] * mask[:, 0].float()
        for j in range(1, seq_len):
            ema[:, j] = (self.alpha * scores[:, j] + (1 - self.alpha) * ema[:, j - 1]) * mask[:, j].float()
            ema[:, j] += ema[:, j - 1] * (~mask[:, j]).float()

        # Max over valid positions
        max_scores = ema.masked_fill(~mask, float("-inf")).max(dim=1).values
        max_scores = torch.where(tokens_per_sample == 0, torch.zeros_like(max_scores), max_scores)

        probs = torch.stack([1 - max_scores, max_scores], dim=-1)
        return Scores.from_sequence_scores(probs, X.batch_indices)


@dataclass(frozen=True, slots=True)
class RollingPool(ScoreTransform):
    """Rolling window mean, then max across windows."""

    window_size: int = 10

    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")

    def transform(self, X: Scores) -> Scores:
        check_scores(X, estimator_name="RollingPool")
        if not X.has_axis(ScoreAxis.SEQ):
            return X

        scores = X.scores[:, :, 1]  # Positive class [batch, seq]
        batch_size, seq_len = scores.shape
        w = self.window_size

        # Build mask
        if X.tokens_per_sample is not None:
            seq_idx = torch.arange(seq_len, device=scores.device)
            mask = seq_idx.unsqueeze(0) < X.tokens_per_sample.to(scores.device).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, seq_len, device=scores.device, dtype=torch.bool)

        # Rolling mean via cumsum
        masked_scores, masked_counts = scores * mask.float(), mask.float()
        cum_scores = F.pad(torch.cumsum(masked_scores, dim=1), (w, 0), value=0)
        cum_counts = F.pad(torch.cumsum(masked_counts, dim=1), (w, 0), value=0)

        roll_scores = cum_scores[:, w:] - cum_scores[:, :-w]
        roll_counts = cum_counts[:, w:] - cum_counts[:, :-w]
        rolling_means = roll_scores / roll_counts.clamp(min=1)

        # Max over valid windows
        valid_mask = roll_counts > 0
        max_scores = rolling_means.masked_fill(~valid_mask, float("-inf")).max(dim=1).values
        max_scores = torch.where(~valid_mask.any(dim=1), torch.zeros_like(max_scores), max_scores)

        probs = torch.stack([1 - max_scores, max_scores], dim=-1)
        return Scores.from_sequence_scores(probs, X.batch_indices)
