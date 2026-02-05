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


class TestMaskedPoolEMA:
    """Tests for EMA pooling."""

    def test_ema_basic(self):
        """Test basic EMA pooling with all tokens valid."""
        # [batch=1, seq=3, hidden=1]
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.5)

        assert result.shape == (1, 1)
        # EMA: e0=0.5*1=0.5, e1=0.5*2+0.5*0.5=1.25, e2=0.5*3+0.5*1.25=2.125
        # Max of [0.5, 1.25, 2.125] = 2.125
        torch.testing.assert_close(result[0, 0], torch.tensor(2.125))

    def test_ema_alpha_validation(self):
        """Test that invalid alpha values raise errors."""
        tensor = torch.tensor([[[1.0], [2.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="alpha must be in"):
            masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=1.5)

        with pytest.raises(ValueError, match="alpha must be in"):
            masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=-0.1)

    def test_ema_partial_mask(self):
        """Test EMA pooling with partial mask."""
        # [batch=2, seq=4, hidden=1]
        tensor = torch.tensor([
            [[1.0], [2.0], [100.0], [100.0]],  # Last two masked
            [[5.0], [6.0], [7.0], [100.0]],    # Last one masked
        ])
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.5)

        assert result.shape == (2, 1)
        # Batch 0: e0=0.5, e1=0.5*2+0.5*0.5=1.25, max=1.25
        torch.testing.assert_close(result[0, 0], torch.tensor(1.25))
        # Batch 1: e0=2.5, e1=0.5*6+0.5*2.5=4.25, e2=0.5*7+0.5*4.25=5.625, max=5.625
        torch.testing.assert_close(result[1, 0], torch.tensor(5.625))

    def test_ema_empty_sequence(self):
        """Test EMA pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0], [2.0]],
            [[5.0], [6.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.5)

        assert result.shape == (2, 1)
        # Batch 1: empty, should be 0
        torch.testing.assert_close(result[1, 0], torch.tensor(0.0))

    def test_ema_alpha_1_equals_max(self):
        """Test that alpha=1.0 gives same result as max pooling."""
        tensor = torch.tensor([[[1.0, 2.0], [5.0, 1.0], [3.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        ema_result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=1.0)
        max_result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        # With alpha=1, EMA equals current value, so max(EMA) = max(values)
        torch.testing.assert_close(ema_result, max_result)

    def test_ema_alpha_near_zero(self):
        """Test EMA with alpha close to zero (heavy smoothing)."""
        tensor = torch.tensor([[[10.0], [1.0], [1.0], [1.0]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.1)

        assert result.shape == (1, 1)
        # With low alpha, later values have less weight, early peak persists
        # e0=1.0, e1=0.1+0.9=1.0, e2=0.1+0.9=1.0, e3=0.1+0.9=1.0
        # Actually: e0=0.1*10=1.0, e1=0.1*1+0.9*1.0=1.0, ...
        # Max should be 1.0
        torch.testing.assert_close(result[0, 0], torch.tensor(1.0))

    def test_ema_4d_tensor(self):
        """Test EMA pooling with 4D tensor."""
        # [layer=2, batch=2, seq=3, hidden=1]
        tensor = torch.tensor([
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=2, batch_dim=1, alpha=0.5)

        assert result.shape == (2, 2, 1)


class TestMaskedPoolRolling:
    """Tests for rolling window pooling."""

    def test_rolling_basic(self):
        """Test basic rolling pooling with all tokens valid."""
        # [batch=1, seq=4, hidden=1]
        tensor = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=2)

        assert result.shape == (1, 1)
        # Rolling means with window=2: [1.5, 2.5, 3.5], max=3.5
        torch.testing.assert_close(result[0, 0], torch.tensor(3.5))

    def test_rolling_window_size_validation(self):
        """Test that invalid window_size values raise errors."""
        tensor = torch.tensor([[[1.0], [2.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=0)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=-1)

    def test_rolling_partial_mask(self):
        """Test rolling pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0], [2.0], [100.0], [100.0]],  # Last two masked
            [[5.0], [6.0], [7.0], [100.0]],    # Last one masked
        ])
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=2)

        assert result.shape == (2, 1)
        # Batch 0: rolling means [1, 1.5, 2, 0], valid [T,T,T,F], max=2
        torch.testing.assert_close(result[0, 0], torch.tensor(2.0))
        # Batch 1: rolling means [5, 5.5, 6.5, 7], valid [T,T,T,T], max=7
        # (window at pos 3 includes only pos 2 which has value 7)
        torch.testing.assert_close(result[1, 0], torch.tensor(7.0))

    def test_rolling_empty_sequence(self):
        """Test rolling pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0], [2.0]],
            [[5.0], [6.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=2)

        assert result.shape == (2, 1)
        # Batch 1: empty, should be 0
        torch.testing.assert_close(result[1, 0], torch.tensor(0.0))

    def test_rolling_window_1_equals_max(self):
        """Test that window_size=1 gives same result as max pooling."""
        tensor = torch.tensor([[[1.0, 2.0], [5.0, 1.0], [3.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        rolling_result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=1)
        max_result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        torch.testing.assert_close(rolling_result, max_result)

    def test_rolling_window_larger_than_seq(self):
        """Test rolling with window_size larger than sequence length."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # Window size 10 on seq length 3 - should still work
        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=10)

        assert result.shape == (1, 1)
        # With large window, it's essentially the mean of all: 2.0
        torch.testing.assert_close(result[0, 0], torch.tensor(2.0))

    def test_rolling_4d_tensor(self):
        """Test rolling pooling with 4D tensor."""
        tensor = torch.tensor([
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "rolling", seq_dim=2, batch_dim=1, window_size=2)

        assert result.shape == (2, 2, 1)


class TestMaskedPoolNegativeValues:
    """Tests for pooling with negative tensor values."""

    def test_mean_negative(self):
        """Test mean pooling with negative values."""
        tensor = torch.tensor([[[-1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        # Mean: [(-1-3+5)/3, (2+4-6)/3] = [1/3, 0]
        torch.testing.assert_close(result, torch.tensor([[1/3, 0.0]]))

    def test_max_negative(self):
        """Test max pooling with all negative values."""
        tensor = torch.tensor([[[-5.0], [-3.0], [-1.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        # Max of negative values: -1
        torch.testing.assert_close(result, torch.tensor([[-1.0]]))

    def test_max_negative_with_mask(self):
        """Test max pooling with negative values and partial mask."""
        tensor = torch.tensor([[[-5.0], [-3.0], [100.0]]])  # 100 is masked out
        mask = torch.tensor([[True, True, False]])

        result = masked_pool(tensor, mask, "max", seq_dim=1, batch_dim=0)

        # Max of [-5, -3] = -3
        torch.testing.assert_close(result, torch.tensor([[-3.0]]))

    def test_ema_negative(self):
        """Test EMA pooling with negative values."""
        tensor = torch.tensor([[[-2.0], [1.0], [-1.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.5)

        assert result.shape == (1, 1)
        # EMA should handle negatives correctly
        # e0=-1.0, e1=0.5*1+0.5*(-1.0)=0, e2=0.5*(-1)+0.5*0=-0.5
        # Max of [-1.0, 0, -0.5] = 0
        torch.testing.assert_close(result[0, 0], torch.tensor(0.0))

    def test_rolling_negative(self):
        """Test rolling pooling with negative values."""
        tensor = torch.tensor([[[-4.0], [-2.0], [0.0], [2.0]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=2)

        # Rolling means: [-3, -1, 1], max = 1
        torch.testing.assert_close(result[0, 0], torch.tensor(1.0))


class TestMaskedPoolDtypes:
    """Tests for different tensor dtypes."""

    def test_float16(self):
        """Test pooling with float16 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.dtype == torch.float16
        assert result.shape == (1, 2)

    def test_bfloat16(self):
        """Test pooling with bfloat16 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.dtype == torch.bfloat16
        assert result.shape == (1, 2)

    def test_float64(self):
        """Test pooling with float64 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.dtype == torch.float64
        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0]], dtype=torch.float64))

    def test_ema_float16(self):
        """Test EMA pooling preserves float16 dtype."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float16)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.5)

        assert result.dtype == torch.float16

    def test_rolling_bfloat16(self):
        """Test rolling pooling preserves bfloat16 dtype."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.bfloat16)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=2)

        assert result.dtype == torch.bfloat16


class TestMaskedPoolLargeSequences:
    """Tests for large sequence lengths."""

    def test_large_seq_mean(self):
        """Test mean pooling with large sequence."""
        batch, seq, hidden = 4, 1000, 64
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (batch, hidden)
        # Mean should be close to 0 for random normal
        assert result.abs().mean() < 0.1

    def test_large_seq_ema(self):
        """Test EMA pooling with large sequence."""
        batch, seq, hidden = 2, 500, 32
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = masked_pool(tensor, mask, "ema", seq_dim=1, batch_dim=0, alpha=0.1)

        assert result.shape == (batch, hidden)
        # Should complete without error

    def test_large_seq_rolling(self):
        """Test rolling pooling with large sequence."""
        batch, seq, hidden = 2, 500, 32
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = masked_pool(tensor, mask, "rolling", seq_dim=1, batch_dim=0, window_size=50)

        assert result.shape == (batch, hidden)
        # Should complete without error

    def test_large_seq_variable_mask(self):
        """Test large sequence with variable-length masking."""
        batch, seq, hidden = 4, 1000, 64
        tensor = torch.randn(batch, seq, hidden)
        # Variable lengths: 100, 500, 800, 1000
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[0, :100] = True
        mask[1, :500] = True
        mask[2, :800] = True
        mask[3, :1000] = True

        result = masked_pool(tensor, mask, "mean", seq_dim=1, batch_dim=0)

        assert result.shape == (batch, hidden)

    def test_large_seq_4d(self):
        """Test large sequence with 4D tensor."""
        layers, batch, seq, hidden = 4, 2, 500, 32
        tensor = torch.randn(layers, batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = masked_pool(tensor, mask, "mean", seq_dim=2, batch_dim=1)

        assert result.shape == (layers, batch, hidden)
