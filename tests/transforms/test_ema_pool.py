"""Tests for EMAPool post-probe transform."""

import pytest
import torch

from probelab.processing.scores import ScoreAxis, Scores
from probelab.transforms import post


def create_token_scores(
    batch_size: int = 5,
    seq_len: int = 10,
    tokens_per_sample: list[int] | None = None,
) -> Scores:
    """Create test token-level scores."""
    if tokens_per_sample is None:
        tokens_per_sample = [seq_len] * batch_size

    # Create random scores in [0, 1]
    torch.manual_seed(42)
    scores = torch.rand(batch_size, seq_len, 2)
    # Normalize to valid probabilities
    scores = scores / scores.sum(dim=-1, keepdim=True)

    return Scores.from_token_scores(
        scores=scores,
        tokens_per_sample=torch.tensor(tokens_per_sample),
    )


class TestEMAPool:
    """Test EMAPool transform."""

    def test_initialization(self):
        """Test EMAPool initialization."""
        pool = post.EMAPool(alpha=0.5)
        assert pool.alpha == 0.5

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha values raise errors."""
        with pytest.raises(ValueError, match="alpha must be in"):
            post.EMAPool(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            post.EMAPool(alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            post.EMAPool(alpha=-0.1)

    def test_transform_reduces_sequence_dimension(self):
        """Test that transform removes sequence dimension."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.EMAPool(alpha=0.5)

        result = pool.transform(scores)

        assert result.has_axis(ScoreAxis.BATCH)
        assert not result.has_axis(ScoreAxis.SEQ)
        assert result.has_axis(ScoreAxis.CLASS)
        assert result.shape == (5, 2)

    def test_transform_preserves_batch_size(self):
        """Test that batch size is preserved."""
        for batch_size in [1, 5, 10]:
            scores = create_token_scores(batch_size=batch_size, seq_len=10)
            pool = post.EMAPool(alpha=0.5)

            result = pool.transform(scores)
            assert result.batch_size == batch_size

    def test_transform_respects_variable_length(self):
        """Test that variable-length sequences are handled correctly."""
        scores = create_token_scores(
            batch_size=3, seq_len=10, tokens_per_sample=[10, 5, 2]
        )
        pool = post.EMAPool(alpha=0.5)

        result = pool.transform(scores)

        assert result.shape == (3, 2)
        # All probabilities should be valid
        assert torch.all(result.scores >= 0)
        assert torch.all(result.scores <= 1)

    def test_transform_returns_valid_probabilities(self):
        """Test that output probabilities are valid."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.EMAPool(alpha=0.5)

        result = pool.transform(scores)

        # All values should be between 0 and 1
        assert torch.all(result.scores >= 0)
        assert torch.all(result.scores <= 1)
        # Probabilities should sum to 1
        assert torch.allclose(result.scores.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_different_alpha_values(self):
        """Test behavior with different alpha values."""
        scores = create_token_scores(batch_size=5, seq_len=10)

        pool_low = post.EMAPool(alpha=0.1)
        pool_high = post.EMAPool(alpha=0.9)

        result_low = pool_low.transform(scores)
        result_high = pool_high.transform(scores)

        # Both should produce valid results
        assert result_low.shape == result_high.shape == (5, 2)
        # Results should be different due to different alpha
        assert not torch.allclose(result_low.scores, result_high.scores)

    def test_idempotent_on_sequence_scores(self):
        """Test that EMAPool is idempotent when applied to sequence-level scores."""
        token_scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.EMAPool(alpha=0.5)

        # First transform
        seq_scores = pool.transform(token_scores)

        # Second transform should return the same (no SEQ axis)
        result = pool.transform(seq_scores)

        assert torch.equal(result.scores, seq_scores.scores)

    def test_raises_on_non_scores(self):
        """Test that EMAPool raises error for non-Scores input."""
        pool = post.EMAPool(alpha=0.5)

        with pytest.raises(TypeError, match="Expected Scores"):
            pool.transform(torch.randn(5, 10, 2))

    def test_repr(self):
        """Test string representation."""
        pool = post.EMAPool(alpha=0.7)
        assert "EMAPool" in repr(pool)
        assert "0.7" in repr(pool)

    def test_ema_accumulation_behavior(self):
        """Test that EMA correctly accumulates over sequence per GDM paper formula.

        Paper formula: EMA_0 = 0, EMA_j = alpha * score_j + (1-alpha) * EMA_{j-1}
        """
        # Create scores with increasing positive class probability
        batch_size, seq_len = 1, 5
        # Positive class probs: [0.1, 0.2, 0.3, 0.4, 0.5]
        probs = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]])

        scores = Scores.from_token_scores(
            scores=probs,
            tokens_per_sample=torch.tensor([seq_len]),
        )

        pool = post.EMAPool(alpha=0.5)
        result = pool.transform(scores)

        # EMA with alpha=0.5, starting from 0 (paper formula):
        # ema[0] = 0.5*0.1 + 0.5*0 = 0.05  (first score blended with initial 0)
        # ema[1] = 0.5*0.2 + 0.5*0.05 = 0.125
        # ema[2] = 0.5*0.3 + 0.5*0.125 = 0.2125
        # ema[3] = 0.5*0.4 + 0.5*0.2125 = 0.30625
        # ema[4] = 0.5*0.5 + 0.5*0.30625 = 0.403125
        # max = 0.403125

        assert result.scores[0, 1].item() == pytest.approx(0.403125, rel=1e-4)

    def test_ema_with_spike(self):
        """Test EMA behavior with a spike in scores."""
        batch_size, seq_len = 1, 5
        # Positive class probs: [0.1, 0.2, 0.9, 0.1, 0.1] - spike at position 2
        probs = torch.tensor([[[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]]])

        scores = Scores.from_token_scores(
            scores=probs,
            tokens_per_sample=torch.tensor([seq_len]),
        )

        pool = post.EMAPool(alpha=0.5)
        result = pool.transform(scores)

        # EMA with alpha=0.5, starting from 0:
        # ema[0] = 0.5*0.1 = 0.05
        # ema[1] = 0.5*0.2 + 0.5*0.05 = 0.125
        # ema[2] = 0.5*0.9 + 0.5*0.125 = 0.5125  <-- peak after spike
        # ema[3] = 0.5*0.1 + 0.5*0.5125 = 0.30625
        # ema[4] = 0.5*0.1 + 0.5*0.30625 = 0.203125
        # max = 0.5125 (captures the spike)

        assert result.scores[0, 1].item() == pytest.approx(0.5125, rel=1e-4)

    def test_edge_case_empty_sequence(self):
        """Test handling of empty sequence (0 tokens)."""
        # Create scores with one sample having 0 tokens
        batch_size, seq_len = 3, 10
        scores_tensor = torch.rand(batch_size, seq_len, 2)
        scores_tensor = scores_tensor / scores_tensor.sum(dim=-1, keepdim=True)

        scores = Scores.from_token_scores(
            scores=scores_tensor,
            tokens_per_sample=torch.tensor([10, 0, 5]),  # Middle sample is empty
        )

        pool = post.EMAPool(alpha=0.5)
        result = pool.transform(scores)

        # Should produce valid output
        assert result.shape == (3, 2)
        # Empty sequence should produce 0 probability for positive class
        assert result.scores[1, 1].item() == 0.0
