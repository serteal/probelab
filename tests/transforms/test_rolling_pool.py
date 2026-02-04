"""Tests for RollingPool post-probe transform."""

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


class TestRollingPool:
    """Test RollingPool transform."""

    def test_initialization(self):
        """Test RollingPool initialization."""
        pool = post.RollingPool(window_size=10)
        assert pool.window_size == 10

    def test_invalid_window_size_raises(self):
        """Test that invalid window size raises errors."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            post.RollingPool(window_size=0)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            post.RollingPool(window_size=-1)

    def test_transform_reduces_sequence_dimension(self):
        """Test that transform removes sequence dimension."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.RollingPool(window_size=3)

        result = pool.transform(scores)

        assert result.has_axis(ScoreAxis.BATCH)
        assert not result.has_axis(ScoreAxis.SEQ)
        assert result.has_axis(ScoreAxis.CLASS)
        assert result.shape == (5, 2)

    def test_transform_preserves_batch_size(self):
        """Test that batch size is preserved."""
        for batch_size in [1, 5, 10]:
            scores = create_token_scores(batch_size=batch_size, seq_len=10)
            pool = post.RollingPool(window_size=3)

            result = pool.transform(scores)
            assert result.batch_size == batch_size

    def test_transform_respects_variable_length(self):
        """Test that variable-length sequences are handled correctly."""
        scores = create_token_scores(
            batch_size=3, seq_len=10, tokens_per_sample=[10, 5, 2]
        )
        pool = post.RollingPool(window_size=3)

        result = pool.transform(scores)

        assert result.shape == (3, 2)
        # All probabilities should be valid
        assert torch.all(result.scores >= 0)
        assert torch.all(result.scores <= 1)

    def test_transform_returns_valid_probabilities(self):
        """Test that output probabilities are valid."""
        scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.RollingPool(window_size=3)

        result = pool.transform(scores)

        # All values should be between 0 and 1
        assert torch.all(result.scores >= 0)
        assert torch.all(result.scores <= 1)
        # Probabilities should sum to 1
        assert torch.allclose(result.scores.sum(dim=-1), torch.ones(5), atol=1e-5)

    def test_different_window_sizes(self):
        """Test behavior with different window sizes."""
        scores = create_token_scores(batch_size=5, seq_len=20)

        pool_small = post.RollingPool(window_size=3)
        pool_large = post.RollingPool(window_size=10)

        result_small = pool_small.transform(scores)
        result_large = pool_large.transform(scores)

        # Both should produce valid results
        assert result_small.shape == result_large.shape == (5, 2)
        # Results should be different due to different window sizes
        assert not torch.allclose(result_small.scores, result_large.scores)

    def test_window_size_equals_sequence_length(self):
        """Test when window size equals sequence length (equivalent to mean)."""
        batch_size, seq_len = 3, 5
        # Create simple test scores
        probs = torch.tensor([
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
        ])

        scores = Scores.from_token_scores(
            scores=probs,
            tokens_per_sample=torch.tensor([seq_len] * batch_size),
        )

        pool = post.RollingPool(window_size=seq_len)
        result = pool.transform(scores)

        # With window = full sequence, max of rolling means = mean
        # Mean positive class prob = (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 = 0.3
        expected_positive = 0.3
        assert result.scores[0, 1].item() == pytest.approx(expected_positive, rel=1e-4)

    def test_idempotent_on_sequence_scores(self):
        """Test that RollingPool is idempotent when applied to sequence-level scores."""
        token_scores = create_token_scores(batch_size=5, seq_len=10)
        pool = post.RollingPool(window_size=3)

        # First transform
        seq_scores = pool.transform(token_scores)

        # Second transform should return the same (no SEQ axis)
        result = pool.transform(seq_scores)

        assert torch.equal(result.scores, seq_scores.scores)

    def test_raises_on_non_scores(self):
        """Test that RollingPool raises error for non-Scores input."""
        pool = post.RollingPool(window_size=3)

        with pytest.raises(TypeError, match="Expected Scores"):
            pool.transform(torch.randn(5, 10, 2))

    def test_repr(self):
        """Test string representation."""
        pool = post.RollingPool(window_size=15)
        assert "RollingPool" in repr(pool)
        assert "15" in repr(pool)

    def test_rolling_window_behavior(self):
        """Test that rolling window correctly computes means."""
        # Create scores with a spike
        batch_size, seq_len = 1, 10
        # Positive class probs: [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        positive_probs = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        probs = torch.tensor([[
            [1 - p, p] for p in positive_probs
        ]])

        scores = Scores.from_token_scores(
            scores=probs,
            tokens_per_sample=torch.tensor([seq_len]),
        )

        pool = post.RollingPool(window_size=3)
        result = pool.transform(scores)

        # Window of 3 around the spike:
        # max rolling mean should be around positions 3-5 (0.9, 0.9, 0.9)
        # which gives mean = 0.9
        assert result.scores[0, 1].item() == pytest.approx(0.9, rel=1e-4)

    def test_rolling_window_exact_math(self):
        """Test exact rolling window math with known values."""
        batch_size, seq_len = 1, 5
        # Positive class probs: [0.1, 0.2, 0.9, 0.1, 0.1]
        positive_probs = [0.1, 0.2, 0.9, 0.1, 0.1]
        probs = torch.tensor([[
            [1 - p, p] for p in positive_probs
        ]])

        scores = Scores.from_token_scores(
            scores=probs,
            tokens_per_sample=torch.tensor([seq_len]),
        )

        pool = post.RollingPool(window_size=3)
        result = pool.transform(scores)

        # Rolling means with window=3:
        # pos 0: [0.1] / 1 = 0.1 (only 1 element available)
        # pos 1: [0.1, 0.2] / 2 = 0.15 (only 2 elements available)
        # pos 2: [0.1, 0.2, 0.9] / 3 = 0.4
        # pos 3: [0.2, 0.9, 0.1] / 3 = 0.4
        # pos 4: [0.9, 0.1, 0.1] / 3 = 0.366...
        # max = 0.4

        assert result.scores[0, 1].item() == pytest.approx(0.4, rel=1e-4)

    def test_edge_case_empty_sequence(self):
        """Test handling of empty sequence (0 tokens)."""
        batch_size, seq_len = 3, 10
        scores_tensor = torch.rand(batch_size, seq_len, 2)
        scores_tensor = scores_tensor / scores_tensor.sum(dim=-1, keepdim=True)

        scores = Scores.from_token_scores(
            scores=scores_tensor,
            tokens_per_sample=torch.tensor([10, 0, 5]),  # Middle sample is empty
        )

        pool = post.RollingPool(window_size=3)
        result = pool.transform(scores)

        # Should produce valid output
        assert result.shape == (3, 2)
        # Empty sequence should produce 0 probability for positive class
        assert result.scores[1, 1].item() == 0.0

    def test_window_larger_than_sequence(self):
        """Test when window size is larger than some sequence lengths."""
        batch_size, seq_len = 3, 10
        scores_tensor = torch.rand(batch_size, seq_len, 2)
        scores_tensor = scores_tensor / scores_tensor.sum(dim=-1, keepdim=True)

        # Sequences of length [10, 3, 2] with window_size=5
        scores = Scores.from_token_scores(
            scores=scores_tensor,
            tokens_per_sample=torch.tensor([10, 3, 2]),
        )

        pool = post.RollingPool(window_size=5)
        result = pool.transform(scores)

        # Should produce valid output for all sequences
        assert result.shape == (3, 2)
        assert torch.all(result.scores >= 0)
        assert torch.all(result.scores <= 1)
