"""Tests for pl.pool functions."""

import pytest
import torch

import probelab as pl


class TestPoolMean:
    """Tests for pl.pool.mean."""

    def test_mean_pool_basic(self):
        """Test basic mean pooling with all tokens valid."""
        # [batch=2, seq=3, hidden=4]
        tensor = torch.tensor([
            [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]],
            [[4.0, 5.0, 6.0, 7.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (2, 4)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0, 4.0, 5.0]))
        torch.testing.assert_close(result[1], torch.tensor([5.0, 6.0, 7.0, 8.0]))

    def test_mean_pool_partial_mask(self):
        """Test mean pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(result[1], torch.tensor([7.0, 8.0]))

    def test_mean_pool_empty_sequence(self):
        """Test mean pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestPoolMax:
    """Tests for pl.pool.max."""

    def test_max_pool_basic(self):
        """Test basic max pooling with all tokens valid."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]],
            [[5.0, 3.0], [4.0, 6.0], [6.0, 5.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.max(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[1], torch.tensor([6.0, 6.0]))

    def test_max_pool_partial_mask(self):
        """Test max pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [5.0, 1.0], [2.0, 4.0]],
            [[5.0, 3.0], [4.0, 6.0], [10.0, 10.0]],
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = pl.pool.max(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([5.0, 2.0]))
        torch.testing.assert_close(result[1], torch.tensor([5.0, 3.0]))

    def test_max_pool_empty_sequence(self):
        """Test max pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = pl.pool.max(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestPoolLastToken:
    """Tests for pl.pool.last_token."""

    def test_last_token_basic(self):
        """Test basic last_token pooling with all tokens valid."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.last_token(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([5.0, 6.0]))
        torch.testing.assert_close(result[1], torch.tensor([11.0, 12.0]))

    def test_last_token_partial_mask(self):
        """Test last_token pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])

        result = pl.pool.last_token(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[1], torch.tensor([7.0, 8.0]))

    def test_last_token_empty_sequence(self):
        """Test last_token pooling with empty sequence."""
        tensor = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = pl.pool.last_token(tensor, mask)

        assert result.shape == (2, 2)
        torch.testing.assert_close(result[0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))


class TestPool4D:
    """Tests for pooling with 4D tensors [batch, layer, seq, hidden]."""

    def test_mean_pool_4d_tensor(self):
        """Test mean pooling with 4D tensor [batch, layer, seq, hidden]."""
        # [batch=2, layer=2, seq=3, hidden=2]
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
             [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]],
            [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
             [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask, dim=2)  # Pool over seq dim

        assert result.shape == (2, 2, 2)  # [batch, layer, hidden]
        # Batch 0, Layer 0: mean over seq = [3, 4]
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        # Batch 0, Layer 1: mean over seq = [4, 5]
        torch.testing.assert_close(result[0, 1], torch.tensor([4.0, 5.0]))
        # Batch 1, Layer 0: mean over seq = [9, 10]
        torch.testing.assert_close(result[1, 0], torch.tensor([9.0, 10.0]))
        # Batch 1, Layer 1: mean over seq = [10, 11]
        torch.testing.assert_close(result[1, 1], torch.tensor([10.0, 11.0]))

    def test_max_pool_4d_tensor_with_mask(self):
        """Test max pooling with 4D tensor and partial mask."""
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
             [[2.0, 3.0], [4.0, 5.0], [100.0, 100.0]]],
            [[[7.0, 8.0], [9.0, 10.0], [100.0, 100.0]],
             [[8.0, 9.0], [10.0, 11.0], [100.0, 100.0]]],
        ])
        mask = torch.tensor([[True, True, False], [True, True, False]])

        result = pl.pool.max(tensor, mask, dim=2)

        assert result.shape == (2, 2, 2)
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[1, 0], torch.tensor([9.0, 10.0]))

    def test_last_token_4d_tensor(self):
        """Test last_token pooling with 4D tensor."""
        tensor = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
             [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]],
            [[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
             [[8.0, 9.0], [10.0, 11.0], [12.0, 13.0]]],
        ])
        mask = torch.tensor([[True, True, False], [True, True, True]])

        result = pl.pool.last_token(tensor, mask, dim=2)

        assert result.shape == (2, 2, 2)
        # Batch 0: last valid (index 1)
        torch.testing.assert_close(result[0, 0], torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(result[0, 1], torch.tensor([4.0, 5.0]))
        # Batch 1: last valid (index 2)
        torch.testing.assert_close(result[1, 0], torch.tensor([11.0, 12.0]))
        torch.testing.assert_close(result[1, 1], torch.tensor([12.0, 13.0]))


class TestPoolEdgeCases:
    """Edge cases and error handling."""

    def test_float_mask(self):
        """Test that float mask works (converted to bool internally)."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (1, 2)
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))

    def test_device_transfer(self):
        """Test that mask is moved to tensor's device."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)
        assert result.device == tensor.device

    def test_single_token(self):
        """Test pooling with single token sequences."""
        tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        mask = torch.ones(2, 1, dtype=torch.bool)

        torch.testing.assert_close(pl.pool.mean(tensor, mask), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        torch.testing.assert_close(pl.pool.max(tensor, mask), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        torch.testing.assert_close(pl.pool.last_token(tensor, mask), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


class TestPoolEMA:
    """Tests for pl.pool.ema."""

    def test_ema_basic(self):
        """Test basic EMA pooling with all tokens valid."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.ema(tensor, mask, alpha=0.5)

        assert result.shape == (1, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(2.125))

    def test_ema_alpha_validation(self):
        """Test that invalid alpha values raise errors."""
        tensor = torch.tensor([[[1.0], [2.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="alpha must be in"):
            pl.pool.ema(tensor, mask, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            pl.pool.ema(tensor, mask, alpha=1.5)

    def test_ema_partial_mask(self):
        """Test EMA pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0], [2.0], [100.0], [100.0]],
            [[5.0], [6.0], [7.0], [100.0]],
        ])
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        result = pl.pool.ema(tensor, mask, alpha=0.5)

        assert result.shape == (2, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(1.25))
        torch.testing.assert_close(result[1, 0], torch.tensor(5.625))

    def test_ema_empty_sequence(self):
        """Test EMA pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0], [2.0]],
            [[5.0], [6.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = pl.pool.ema(tensor, mask, alpha=0.5)

        assert result.shape == (2, 1)
        torch.testing.assert_close(result[1, 0], torch.tensor(0.0))

    def test_ema_alpha_1_equals_max(self):
        """Test that alpha=1.0 gives same result as max pooling."""
        tensor = torch.tensor([[[1.0, 2.0], [5.0, 1.0], [3.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        ema_result = pl.pool.ema(tensor, mask, alpha=1.0)
        max_result = pl.pool.max(tensor, mask)

        torch.testing.assert_close(ema_result, max_result)

    def test_ema_4d_tensor(self):
        """Test EMA pooling with 4D tensor."""
        tensor = torch.tensor([
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.ema(tensor, mask, dim=2, alpha=0.5)

        assert result.shape == (2, 2, 1)


class TestPoolRolling:
    """Tests for pl.pool.rolling."""

    def test_rolling_basic(self):
        """Test basic rolling pooling with all tokens valid."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, window_size=2)

        assert result.shape == (1, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(3.5))

    def test_rolling_window_size_validation(self):
        """Test that invalid window_size values raise errors."""
        tensor = torch.tensor([[[1.0], [2.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)

        with pytest.raises(ValueError, match="window_size must be >= 1"):
            pl.pool.rolling(tensor, mask, window_size=0)

    def test_rolling_partial_mask(self):
        """Test rolling pooling with partial mask."""
        tensor = torch.tensor([
            [[1.0], [2.0], [100.0], [100.0]],
            [[5.0], [6.0], [7.0], [100.0]],
        ])
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])

        result = pl.pool.rolling(tensor, mask, window_size=2)

        assert result.shape == (2, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(2.0))
        torch.testing.assert_close(result[1, 0], torch.tensor(7.0))

    def test_rolling_empty_sequence(self):
        """Test rolling pooling with empty sequence (all masked)."""
        tensor = torch.tensor([
            [[1.0], [2.0]],
            [[5.0], [6.0]],
        ])
        mask = torch.tensor([[True, True], [False, False]])

        result = pl.pool.rolling(tensor, mask, window_size=2)

        assert result.shape == (2, 1)
        torch.testing.assert_close(result[1, 0], torch.tensor(0.0))

    def test_rolling_window_1_equals_max(self):
        """Test that window_size=1 gives same result as max pooling."""
        tensor = torch.tensor([[[1.0, 2.0], [5.0, 1.0], [3.0, 4.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        rolling_result = pl.pool.rolling(tensor, mask, window_size=1)
        max_result = pl.pool.max(tensor, mask)

        torch.testing.assert_close(rolling_result, max_result)

    def test_rolling_window_larger_than_seq(self):
        """Test rolling with window_size larger than sequence length."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, window_size=10)

        assert result.shape == (1, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(2.0))

    def test_rolling_4d_tensor(self):
        """Test rolling pooling with 4D tensor."""
        tensor = torch.tensor([
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
            [[[7.0], [8.0], [9.0]], [[10.0], [11.0], [12.0]]],
        ])
        mask = torch.ones(2, 3, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, dim=2, window_size=2)

        assert result.shape == (2, 2, 1)


class TestPoolNegativeValues:
    """Tests for pooling with negative tensor values."""

    def test_mean_negative(self):
        """Test mean pooling with negative values."""
        tensor = torch.tensor([[[-1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        torch.testing.assert_close(result, torch.tensor([[1/3, 0.0]]))

    def test_max_negative(self):
        """Test max pooling with all negative values."""
        tensor = torch.tensor([[[-5.0], [-3.0], [-1.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.max(tensor, mask)

        torch.testing.assert_close(result, torch.tensor([[-1.0]]))

    def test_max_negative_with_mask(self):
        """Test max pooling with negative values and partial mask."""
        tensor = torch.tensor([[[-5.0], [-3.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])

        result = pl.pool.max(tensor, mask)

        torch.testing.assert_close(result, torch.tensor([[-3.0]]))

    def test_ema_negative(self):
        """Test EMA pooling with negative values."""
        tensor = torch.tensor([[[-2.0], [1.0], [-1.0]]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.ema(tensor, mask, alpha=0.5)

        assert result.shape == (1, 1)
        torch.testing.assert_close(result[0, 0], torch.tensor(0.0))

    def test_rolling_negative(self):
        """Test rolling pooling with negative values."""
        tensor = torch.tensor([[[-4.0], [-2.0], [0.0], [2.0]]])
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, window_size=2)

        torch.testing.assert_close(result[0, 0], torch.tensor(1.0))


class TestPoolDtypes:
    """Tests for different tensor dtypes."""

    def test_float16(self):
        """Test pooling with float16 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        assert result.dtype == torch.float16
        assert result.shape == (1, 2)

    def test_bfloat16(self):
        """Test pooling with bfloat16 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        assert result.dtype == torch.bfloat16
        assert result.shape == (1, 2)

    def test_float64(self):
        """Test pooling with float64 tensors."""
        tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float64)
        mask = torch.ones(1, 2, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        assert result.dtype == torch.float64
        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0]], dtype=torch.float64))

    def test_ema_float16(self):
        """Test EMA pooling preserves float16 dtype."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float16)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.ema(tensor, mask, alpha=0.5)

        assert result.dtype == torch.float16

    def test_rolling_bfloat16(self):
        """Test rolling pooling preserves bfloat16 dtype."""
        tensor = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.bfloat16)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, window_size=2)

        assert result.dtype == torch.bfloat16


class TestPoolLargeSequences:
    """Tests for large sequence lengths."""

    def test_large_seq_mean(self):
        """Test mean pooling with large sequence."""
        batch, seq, hidden = 4, 1000, 64
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (batch, hidden)
        assert result.abs().mean() < 0.1

    def test_large_seq_ema(self):
        """Test EMA pooling with large sequence."""
        batch, seq, hidden = 2, 500, 32
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = pl.pool.ema(tensor, mask, alpha=0.1)

        assert result.shape == (batch, hidden)

    def test_large_seq_rolling(self):
        """Test rolling pooling with large sequence."""
        batch, seq, hidden = 2, 500, 32
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = pl.pool.rolling(tensor, mask, window_size=50)

        assert result.shape == (batch, hidden)

    def test_large_seq_variable_mask(self):
        """Test large sequence with variable-length masking."""
        batch, seq, hidden = 4, 1000, 64
        tensor = torch.randn(batch, seq, hidden)
        mask = torch.zeros(batch, seq, dtype=torch.bool)
        mask[0, :100] = True
        mask[1, :500] = True
        mask[2, :800] = True
        mask[3, :1000] = True

        result = pl.pool.mean(tensor, mask)

        assert result.shape == (batch, hidden)

    def test_large_seq_4d(self):
        """Test large sequence with 4D tensor."""
        batch, layers, seq, hidden = 2, 4, 500, 32
        tensor = torch.randn(batch, layers, seq, hidden)
        mask = torch.ones(batch, seq, dtype=torch.bool)

        result = pl.pool.mean(tensor, mask, dim=2)

        assert result.shape == (batch, layers, hidden)
