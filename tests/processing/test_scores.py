"""Tests for Scores container class."""

import pytest
import torch

from probelib.processing.scores import ScoreAxis, Scores


class TestScoresInit:
    """Test Scores initialization."""

    def test_init_with_axes(self):
        """Test basic initialization with explicit axes."""
        scores = torch.randn(5, 2)
        obj = Scores(
            scores=scores,
            axes=(ScoreAxis.BATCH, ScoreAxis.CLASS),
        )

        assert obj.scores is scores
        assert obj.axes == (ScoreAxis.BATCH, ScoreAxis.CLASS)
        assert obj.batch_size == 5

    def test_init_validates_axes_match_dims(self):
        """Test that axes count must match tensor dimensions."""
        scores = torch.randn(5, 10, 2)

        with pytest.raises(ValueError, match="must match tensor dimensions"):
            Scores(
                scores=scores,
                axes=(ScoreAxis.BATCH, ScoreAxis.CLASS),  # Missing SEQ
            )

    def test_init_requires_class_axis_last(self):
        """Test that CLASS axis must be last."""
        scores = torch.randn(5, 2, 10)

        with pytest.raises(ValueError, match="CLASS axis must be the last"):
            Scores(
                scores=scores,
                axes=(ScoreAxis.BATCH, ScoreAxis.CLASS, ScoreAxis.SEQ),
            )


class TestFromSequenceScores:
    """Test Scores.from_sequence_scores factory."""

    def test_basic_creation(self):
        """Test creating sequence-level scores."""
        scores = torch.randn(10, 2)

        obj = Scores.from_sequence_scores(scores)

        assert obj.shape == (10, 2)
        assert obj.axes == (ScoreAxis.BATCH, ScoreAxis.CLASS)
        assert obj.batch_size == 10

    def test_with_batch_indices(self):
        """Test with batch indices."""
        scores = torch.randn(5, 2)
        indices = [0, 2, 4, 6, 8]

        obj = Scores.from_sequence_scores(scores, batch_indices=indices)

        assert obj.batch_indices == indices

    def test_rejects_wrong_dims(self):
        """Test error for non-2D tensor."""
        scores = torch.randn(5, 10, 2)

        with pytest.raises(ValueError, match="Expected 2D tensor"):
            Scores.from_sequence_scores(scores)


class TestFromTokenScores:
    """Test Scores.from_token_scores factory."""

    def test_from_3d_tensor(self):
        """Test creating from 3D [batch, seq, 2] tensor."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        assert obj.shape == (5, 10, 2)
        assert obj.axes == (ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS)
        assert obj.has_axis(ScoreAxis.SEQ)

    def test_from_flattened_tensor(self):
        """Test creating from flattened [n_tokens, 2] tensor."""
        # 3 samples with 5, 3, 2 tokens = 10 total
        tokens_per_sample = torch.tensor([5, 3, 2])
        scores = torch.randn(10, 2)

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        assert obj.shape == (3, 5, 2)  # [batch, max_seq, 2]
        assert obj.axes == (ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS)

    def test_flattened_preserves_values(self):
        """Test that flattened format preserves score values."""
        tokens_per_sample = torch.tensor([3, 2])
        scores = torch.arange(10).reshape(5, 2).float()

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        # First sample: tokens 0, 1, 2
        assert torch.equal(obj.scores[0, 0], scores[0])
        assert torch.equal(obj.scores[0, 1], scores[1])
        assert torch.equal(obj.scores[0, 2], scores[2])

        # Second sample: tokens 3, 4
        assert torch.equal(obj.scores[1, 0], scores[3])
        assert torch.equal(obj.scores[1, 1], scores[4])

    def test_with_batch_indices(self):
        """Test with batch indices."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])
        indices = [1, 3, 5, 7, 9]

        obj = Scores.from_token_scores(scores, tokens_per_sample, batch_indices=indices)

        assert obj.batch_indices == indices

    def test_rejects_wrong_dims(self):
        """Test error for wrong tensor shape."""
        scores = torch.randn(5, 10, 2, 3)  # 4D
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        with pytest.raises(ValueError, match="Expected 2D .* or 3D"):
            Scores.from_token_scores(scores, tokens_per_sample)


class TestScoresPool:
    """Test Scores.pool() method."""

    def test_pool_mean(self):
        """Test mean pooling."""
        scores = torch.ones(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 10, 10, 10, 10])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        pooled = obj.pool(dim="sequence", method="mean")

        assert pooled.shape == (5, 2)
        assert not pooled.has_axis(ScoreAxis.SEQ)
        assert torch.allclose(pooled.scores, torch.ones(5, 2))

    def test_pool_max(self):
        """Test max pooling."""
        scores = torch.arange(100).reshape(5, 10, 2).float()
        tokens_per_sample = torch.tensor([10, 10, 10, 10, 10])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        pooled = obj.pool(dim="sequence", method="max")

        assert pooled.shape == (5, 2)
        # Max should be from last position in each sample
        assert torch.equal(pooled.scores[0], scores[0, 9])

    def test_pool_last_token(self):
        """Test last_token pooling."""
        scores = torch.arange(100).reshape(5, 10, 2).float()
        tokens_per_sample = torch.tensor([5, 3, 8, 1, 10])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        pooled = obj.pool(dim="sequence", method="last_token")

        assert pooled.shape == (5, 2)
        # Check last valid token is selected
        assert torch.equal(pooled.scores[0], scores[0, 4])  # tokens_per_sample=5, last is index 4
        assert torch.equal(pooled.scores[1], scores[1, 2])  # tokens_per_sample=3, last is index 2

    def test_pool_handles_variable_lengths(self):
        """Test pooling with variable-length sequences."""
        scores = torch.ones(3, 10, 2)
        tokens_per_sample = torch.tensor([10, 5, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        pooled = obj.pool(dim="sequence", method="mean")

        # All should be 1.0 since all valid tokens are 1
        assert torch.allclose(pooled.scores, torch.ones(3, 2))

    def test_pool_removes_sequence_axis(self):
        """Test that pool removes SEQ axis."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        assert obj.has_axis(ScoreAxis.SEQ)

        pooled = obj.pool(dim="sequence", method="mean")
        assert not pooled.has_axis(ScoreAxis.SEQ)
        assert pooled.has_axis(ScoreAxis.BATCH)
        assert pooled.has_axis(ScoreAxis.CLASS)

    def test_pool_empty_sequences(self):
        """Test pooling handles empty sequences (tokens_per_sample=0)."""
        scores = torch.randn(3, 10, 2)
        tokens_per_sample = torch.tensor([5, 0, 3])  # Middle sample is empty

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        # Test all methods handle empty sequences
        for method in ["mean", "max", "last_token"]:
            pooled = obj.pool(dim="sequence", method=method)
            # Empty sequence should result in zeros
            assert torch.allclose(pooled.scores[1], torch.zeros(2))

    def test_pool_invalid_method_raises(self):
        """Test error for invalid pooling method."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        with pytest.raises(ValueError, match="Invalid method"):
            obj.pool(dim="sequence", method="invalid")

    def test_pool_invalid_dim_raises(self):
        """Test error for invalid dimension."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        with pytest.raises(ValueError, match="Only 'sequence' dimension"):
            obj.pool(dim="layer", method="mean")

    def test_pool_without_seq_axis_raises(self):
        """Test error when trying to pool already-pooled scores."""
        scores = torch.randn(5, 2)

        obj = Scores.from_sequence_scores(scores)

        with pytest.raises(ValueError, match="don't have SEQ axis"):
            obj.pool(dim="sequence", method="mean")

    def test_pool_fixed_length(self):
        """Test pooling fixed-length sequences (no tokens_per_sample)."""
        scores = torch.ones(5, 10, 2)

        # Create with 3D format but no variable-length handling
        obj = Scores(
            scores=scores,
            axes=(ScoreAxis.BATCH, ScoreAxis.SEQ, ScoreAxis.CLASS),
            tokens_per_sample=None,
        )

        pooled = obj.pool(dim="sequence", method="mean")
        assert pooled.shape == (5, 2)
        assert torch.allclose(pooled.scores, torch.ones(5, 2))


class TestScoresProperties:
    """Test Scores properties."""

    def test_shape_property(self):
        """Test shape property."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        assert obj.shape == (5, 10, 2)

    def test_batch_size_property(self):
        """Test batch_size property."""
        scores = torch.randn(7, 2)
        obj = Scores.from_sequence_scores(scores)

        assert obj.batch_size == 7

    def test_device_property(self):
        """Test device property."""
        scores = torch.randn(5, 2)
        obj = Scores.from_sequence_scores(scores)

        assert obj.device == scores.device

    def test_has_axis(self):
        """Test has_axis method."""
        seq_scores = Scores.from_sequence_scores(torch.randn(5, 2))
        assert seq_scores.has_axis(ScoreAxis.BATCH)
        assert seq_scores.has_axis(ScoreAxis.CLASS)
        assert not seq_scores.has_axis(ScoreAxis.SEQ)

        token_scores = Scores.from_token_scores(
            torch.randn(5, 10, 2), torch.tensor([10, 8, 6, 4, 2])
        )
        assert token_scores.has_axis(ScoreAxis.BATCH)
        assert token_scores.has_axis(ScoreAxis.SEQ)
        assert token_scores.has_axis(ScoreAxis.CLASS)


class TestScoresTo:
    """Test Scores.to() method."""

    def test_to_device(self):
        """Test moving to device."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        # Move to same device (should work)
        moved = obj.to("cpu")
        assert moved.device.type == "cpu"
        assert moved.tokens_per_sample.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda(self):
        """Test moving to CUDA."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        moved = obj.to("cuda")

        assert moved.device.type == "cuda"


class TestScoresRepr:
    """Test Scores string representation."""

    def test_repr_sequence_scores(self):
        """Test repr for sequence-level scores."""
        scores = torch.randn(5, 2)
        obj = Scores.from_sequence_scores(scores)

        repr_str = repr(obj)
        assert "Scores" in repr_str
        assert "(5, 2)" in repr_str
        assert "BATCH" in repr_str
        assert "CLASS" in repr_str

    def test_repr_token_scores(self):
        """Test repr for token-level scores."""
        scores = torch.randn(5, 10, 2)
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])

        obj = Scores.from_token_scores(scores, tokens_per_sample)

        repr_str = repr(obj)
        assert "Scores" in repr_str
        assert "(5, 10, 2)" in repr_str
        assert "SEQ" in repr_str


class TestPoolDeviceHandling:
    """Test pooling with different device scenarios."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_pool_with_gpu_scores(self):
        """Test pooling when scores are on GPU."""
        scores = torch.randn(5, 10, 2, device="cuda")
        tokens_per_sample = torch.tensor([10, 8, 6, 4, 2])  # CPU

        obj = Scores.from_token_scores(scores, tokens_per_sample)
        pooled = obj.pool(dim="sequence", method="mean")

        # Should work without error
        assert pooled.device.type == "cuda"
