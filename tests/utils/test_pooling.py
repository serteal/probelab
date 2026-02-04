"""Tests for the shared masked_pool utility."""

import pytest
import torch

from probelab.types import AggregationMethod
from probelab.utils.pooling import masked_pool


class TestMaskedPoolMean:
    """Tests for mean pooling."""

    def test_mean_pool_basic(self):
        """Test basic mean pooling with all tokens valid."""
        # [batch=2, seq=3, hidden=4]
        tensor = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 4)
        # First batch: mean of [1,2,3], [2,3,4], [3,4,5], [4,5,6] = [2,3,4,5]
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
        # Second batch: mean of [4,5,6], [5,6,7], [6,7,8], [7,8,9] = [5,6,7,8]
        torch.testing.assert_close(result[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_mean_pool_partial_mask(self):
        """Test mean pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        # Only first two tokens valid for batch 0, only first token for batch 1
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: mean of [1,2], [3,4] = [2, 3]
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        # Batch 1: mean of [7,8] = [7, 8]
        torch.testing.assert_close(result[1], torch.tensor([7.0, 8.0]))

    def test_mean_pool_empty_sequence(self):
        """Test mean pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        # All tokens masked for batch 1
        mask = torch.tensor([[True, True], [False, False]])

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: mean of [1,2], [3,4] = [2, 3]
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        # Batch 1: empty, should be 0 (due to clamp(min=1))
        # Note: with clamp(min=1), sum is 0 / 1 = 0
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestMaskedPoolMax:
    """Tests for max pooling."""

    def test_max_pool_basic(self):
        """Test basic max pooling with all tokens valid."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
            [[5.0, 3.0], [4.0, 6.0], [6.0, 5.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: max of [1,3,2], [2,1,4] = [3, 4]
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        # Batch 1: max of [5,4,6], [3,6,5] = [6, 6]
        torch.testing.assert_close(result[1], torch.tensor([6.0, 6.0]))

    def test_max_pool_partial_mask(self):
        """Test max pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [5.0, 1.0], [2.0, 4.0]],  # Max should ignore last
            [[5.0, 3.0], [4.0, 6.0], [10.0, 10.0]],  # Max should ignore last two
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: max of [1,5], [2,1] = [5, 2]
        torch.testing.assert_close(result[0], torch.tensor([5.0, 2.0]))
        # Batch 1: max of [5], [3] = [5, 3]
        torch.testing.assert_close(result[1], torch.tensor([5.0, 3.0]))

    def test_max_pool_empty_sequence(self):
        """Test max pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: max of [1,3], [2,4] = [3, 4]
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        # Batch 1: empty, should be 0
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestMaskedPoolLastToken:
    """Tests for last_token pooling."""

    def test_last_token_basic(self):
        """Test basic last_token pooling with all tokens valid."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "last_token", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Last token for batch 0: [5, 6]
        torch.testing.assert_close(result[0], torch.tensor([5.0, 6.0]))
        # Last token for batch 1: [11, 12]
        torch.testing.assert_close(result[1], torch.tensor([11.0, 12.0]))

    def test_last_token_partial_mask(self):
        """Test last_token pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        # Batch 0: 2 valid tokens, batch 1: 1 valid token
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = masked_pool(tensor, mask, "last_token", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Last valid token for batch 0 (index 1): [3, 4]
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        # Last valid token for batch 1 (index 0): [7, 8]
        torch.testing.assert_close(result[1], torch.tensor([7.0, 8.0]))

    def test_last_token_empty_sequence(self):
        """Test last_token pooling with empty sequence."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = masked_pool(tensor, mask, "last_token", seq_dim=1, batch_dim=0)

        assert result.shape == (2, 2)
        # Batch 0: last valid token (index 1): [3, 4]
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        # Batch 1: empty, should be 0
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestMaskedPoolHigherRank:
    """Tests for pooling with higher-rank tensors (e.g., [layer, batch, seq, hidden])."""

    def test_mean_pool_4d_tensor(self):
        """Test mean pooling with 4D tensor [layer, batch, seq, hidden]."""
        # [layer=2, batch=2, seq=3, hidden=2]
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
             [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
            [[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
             [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=2, batch_dim=1)

        assert result.shape == (2, 2, 2)  # [layer, batch, hidden]
        # Layer 0, Batch 0: mean over seq = [3, 4]
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        # Layer 0, Batch 1: mean over seq = [9, 10]
        torch.testing.assert_close(result[0, 1], torch.tensor([9.0, 10.0]))
        # Layer 1, Batch 0: mean over seq = [4, 5]
        torch.testing.assert_close(result[1, 0], torch.tensor([4.0, 5.0]))
        # Layer 1, Batch 1: mean over seq = [10, 11]
        torch.testing.assert_close(result[1, 1], torch.tensor([10.0, 11.0]))

    def test_max_pool_4d_tensor_with_mask(self):
        """Test max pooling with 4D tensor and partial mask."""
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],  # Max should ignore last
             [[7.0, 8.0], [9.0, 10.0], [100.0, 100.0]]],
            [[[2.0, 3.0], [4.0, 5.0], [100.0, 100.0]],
             [[8.0, 9.0], [10.0, 11.0], [100.0, 100.0]]],
        ])
        mask = torch.tensor([[True, True, False], [True, True, False]])

        result = masked_pool(tensor, mask, "max", seq_dim=2, batch_dim=1)

        assert result.shape == (2, 2, 2)
        # Layer 0, Batch 0: max of [1,3], [2,4] = [3, 4]
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        # Layer 0, Batch 1: max of [7,9], [8,10] = [9, 10]
        torch.testing.assert_close(result[0, 1], torch.tensor([9.0, 10.0]))

    def test_last_token_4d_tensor(self):
        """Test last_token pooling with 4D tensor."""
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
             [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
            [[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
             [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]],
        ])
        # Batch 0: 2 valid, Batch 1: 3 valid
        mask = torch.tensor([[True, True, False], [True, True, True]])

        result = masked_pool(tensor, mask, "last_token", seq_dim=2, batch_dim=1)

        assert result.shape == (2, 2, 2)
        # Layer 0, Batch 0: last valid (index 1) = [3, 4]
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        # Layer 0, Batch 1: last valid (index 2) = [11, 12]
        torch.testing.assert_close(result[0, 1], torch.tensor([11.0, 12.0]))
        # Layer 1, Batch 0: last valid (index 1) = [4, 5]
        torch.testing.assert_close(result[1, 0], torch.tensor([4.0, 5.0]))
        # Layer 1, Batch 1: last valid (index 2) = [12, 13]
        torch.testing.assert_close(result[1, 1], torch.tensor([12.0, 13.0]))


class TestMaskedPoolEdgeCases:
    """Edge cases and error handling."""

    def test_string_method(self):
        """Test that string method names work."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)
        assert result.shape == (1, 2)

        result = masked_pool(tensor, mask, AggregationMethod.MEAN, seq_dim=1, batch_dim=0)
        assert result.shape == (1, 2)

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        tensor = torch.tensor([[[1.0, 2.0]]])
        mask = torch.ones(1, 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="Unknown pooling method"):
            masked_pool(tensor, mask, "invalid", seq_dim=1, batch_dim=0)

    def test_float_mask(self):
        """Test that float mask works (converted to bool internally)."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # Float mask

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (1, 2)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))

    def test_device_transfer(self):
        """Test that mask is moved to tensor's device."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        # Both on CPU should work
        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)
        assert result.device == tensor.device

    def test_single_token(self):
        """Test pooling with single token sequences."""
        tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        mask = torch.ones(2, 1, dtype=torch.bool)

        # Mean of single token = that token
        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # Max of single token = that token
        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        # Last token of single token = that token
        result = masked_pool(tensor, mask, "last_token", seq_dim=1, batch_dim=0)
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
