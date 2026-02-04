"""Tests for post-probe transforms (Scores → Scores)."""

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
    return Scores.from_token_scores(
        scores=torch.randn(batch_size, seq_len, 2).softmax(dim=-1),
        tokens_per_sample=torch.tensor(tokens_per_sample),
    )


def create_sequence_scores(batch_size: int = 5) -> Scores:
    """Create test sequence-level scores."""
    return Scores.from_sequence_scores(
        scores=torch.randn(batch_size, 2).softmax(dim=-1),
    )


class TestPool:
    """Test post.Pool transformer (score pooling)."""

    def test_pool_mean(self):
        """Test mean pooling over sequence dimension."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        transform = post.Pool(method="mean")

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)
        assert not result.has_axis(ScoreAxis.SEQ)

    def test_pool_max(self):
        """Test max pooling over sequence dimension."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        transform = post.Pool(method="max")

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)
        assert not result.has_axis(ScoreAxis.SEQ)

    def test_pool_last_token(self):
        """Test last_token pooling."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        transform = post.Pool(method="last_token")

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)
        assert not result.has_axis(ScoreAxis.SEQ)

    def test_pool_respects_tokens_per_sample(self):
        """Test that pooling respects variable length sequences."""
        # Create scores with known values
        torch.manual_seed(42)
        batch_size = 3
        seq_len = 10
        tokens_per_sample = [5, 8, 3]  # Variable lengths

        scores_tensor = torch.ones(batch_size, seq_len, 2) * 0.5
        # Set positive class to 1.0 for valid tokens
        for i, n_tokens in enumerate(tokens_per_sample):
            scores_tensor[i, :n_tokens, 1] = 1.0

        scores = Scores.from_token_scores(
            scores=scores_tensor,
            tokens_per_sample=torch.tensor(tokens_per_sample),
        )

        transform = post.Pool(method="mean")
        result = transform.transform(scores)

        # Mean should be close to 1.0 for positive class (all valid tokens are 1.0)
        assert torch.allclose(result.scores[:, 1], torch.ones(batch_size), atol=0.01)

    def test_pool_idempotent_when_already_pooled(self):
        """Test that pool is idempotent when already sequence-level."""
        scores = create_sequence_scores(batch_size=5)
        transform = post.Pool(method="mean")

        result = transform.transform(scores)

        assert torch.equal(result.scores, scores.scores)

    def test_pool_invalid_method_raises(self):
        """Test error for invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            post.Pool(method="invalid")

    def test_repr(self):
        """Test string representation."""
        transform = post.Pool(method="mean")
        assert "Pool" in repr(transform)
        assert "mean" in repr(transform)


class TestEMAPool:
    """Test post.EMAPool transformer."""

    def test_ema_pool_basic(self):
        """Test basic EMA pooling."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        transform = post.EMAPool(alpha=0.5)

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)
        assert not result.has_axis(ScoreAxis.SEQ)

    def test_ema_pool_alpha_parameter(self):
        """Test different alpha values produce different results."""
        torch.manual_seed(42)
        scores = create_token_scores(batch_size=5, seq_len=10)

        transform_low = post.EMAPool(alpha=0.1)
        transform_high = post.EMAPool(alpha=0.9)

        result_low = transform_low.transform(scores)
        result_high = transform_high.transform(scores)

        # Different alpha should produce different results
        assert not torch.allclose(result_low.scores, result_high.scores)

    def test_ema_pool_respects_tokens_per_sample(self):
        """Test that EMA respects variable length sequences."""
        torch.manual_seed(42)
        scores = create_token_scores(
            batch_size=3, seq_len=10, tokens_per_sample=[5, 8, 3]
        )
        transform = post.EMAPool(alpha=0.5)

        result = transform.transform(scores)

        # Should produce valid output
        assert result.scores.shape == (3, 2)
        assert torch.isfinite(result.scores).all()

    def test_ema_pool_invalid_alpha_raises(self):
        """Test error for invalid alpha values."""
        with pytest.raises(ValueError, match="alpha must be in"):
            post.EMAPool(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            post.EMAPool(alpha=1.5)

    def test_ema_pool_idempotent_when_already_pooled(self):
        """Test that EMA pool is idempotent when already sequence-level."""
        scores = create_sequence_scores(batch_size=5)
        transform = post.EMAPool(alpha=0.5)

        result = transform.transform(scores)

        assert torch.equal(result.scores, scores.scores)

    def test_repr(self):
        """Test string representation."""
        transform = post.EMAPool(alpha=0.7)
        assert "EMAPool" in repr(transform)
        assert "0.7" in repr(transform)


class TestRollingPool:
    """Test post.RollingPool transformer."""

    def test_rolling_pool_basic(self):
        """Test basic rolling pool."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        transform = post.RollingPool(window_size=5)

        result = transform.transform(scores)

        assert result.scores.shape == (5, 2)
        assert not result.has_axis(ScoreAxis.SEQ)

    def test_rolling_pool_window_size_parameter(self):
        """Test different window sizes produce different results."""
        torch.manual_seed(42)
        scores = create_token_scores(batch_size=5, seq_len=20)

        transform_small = post.RollingPool(window_size=3)
        transform_large = post.RollingPool(window_size=15)

        result_small = transform_small.transform(scores)
        result_large = transform_large.transform(scores)

        # Different window sizes should produce different results
        assert not torch.allclose(result_small.scores, result_large.scores)

    def test_rolling_pool_respects_tokens_per_sample(self):
        """Test that rolling pool respects variable length sequences."""
        torch.manual_seed(42)
        scores = create_token_scores(
            batch_size=3, seq_len=10, tokens_per_sample=[5, 8, 3]
        )
        transform = post.RollingPool(window_size=3)

        result = transform.transform(scores)

        # Should produce valid output
        assert result.scores.shape == (3, 2)
        assert torch.isfinite(result.scores).all()

    def test_rolling_pool_window_larger_than_seq(self):
        """Test rolling pool with window larger than sequence."""
        scores = create_token_scores(batch_size=3, seq_len=5)
        transform = post.RollingPool(window_size=10)

        result = transform.transform(scores)

        # Should still work
        assert result.scores.shape == (3, 2)
        assert torch.isfinite(result.scores).all()

    def test_rolling_pool_invalid_window_raises(self):
        """Test error for invalid window size."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            post.RollingPool(window_size=0)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            post.RollingPool(window_size=-1)

    def test_rolling_pool_idempotent_when_already_pooled(self):
        """Test that rolling pool is idempotent when already sequence-level."""
        scores = create_sequence_scores(batch_size=5)
        transform = post.RollingPool(window_size=5)

        result = transform.transform(scores)

        assert torch.equal(result.scores, scores.scores)

    def test_repr(self):
        """Test string representation."""
        transform = post.RollingPool(window_size=10)
        assert "RollingPool" in repr(transform)
        assert "10" in repr(transform)


class TestTypeChecking:
    """Test type checking for post transforms."""

    def test_pool_rejects_non_scores(self):
        """Test that Pool rejects non-Scores input."""
        transform = post.Pool(method="mean")

        with pytest.raises(TypeError, match="Expected Scores"):
            transform.transform(torch.randn(5, 10, 2))

    def test_ema_pool_rejects_non_scores(self):
        """Test that EMAPool rejects non-Scores input."""
        transform = post.EMAPool(alpha=0.5)

        with pytest.raises(TypeError, match="Expected Scores"):
            transform.transform(torch.randn(5, 10, 2))

    def test_rolling_pool_rejects_non_scores(self):
        """Test that RollingPool rejects non-Scores input."""
        transform = post.RollingPool(window_size=5)

        with pytest.raises(TypeError, match="Expected Scores"):
            transform.transform(torch.randn(5, 10, 2))
