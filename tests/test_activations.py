"""Tests for Activations container."""

import unittest

import torch

from probelab.processing.activations import Activations, DIMS


def _make_flat(data, det_mask):
    """Build flat+offsets Activations from padded data and det mask.

    Helper that wraps Activations.from_padded for convenience in tests.
    """
    return Activations.from_padded(data, det_mask, dims="bsh")


def _make_flat_blsh(data, det_mask, layers):
    """Build flat+offsets blsh Activations from padded data and det mask."""
    return Activations.from_padded(data, det_mask, dims="blsh", layers=layers)


class TestActivationsConstruction(unittest.TestCase):
    def test_4d_blsh(self):
        """Test flat+offsets [batch, layer, seq, hidden] tensor."""
        t = torch.randn(4, 2, 8, 16)
        a = _make_flat_blsh(t, torch.ones(4, 8), layers=(0, 1))
        # data should be [total_tokens, n_layers, hidden] = [32, 2, 16]
        self.assertEqual(a.data.shape, (32, 2, 16))
        self.assertEqual(a.n_layers, 2)
        self.assertEqual(a.batch_size, 4)
        self.assertEqual(a.seq_len, 8)
        self.assertEqual(a.hidden_size, 16)

    def test_3d_bsh(self):
        """Test flat+offsets [batch, seq, hidden] tensor."""
        t = torch.randn(4, 8, 16)
        a = _make_flat(t, torch.ones(4, 8))
        # data: [32, 16]
        self.assertEqual(a.data.shape, (32, 16))
        self.assertIsNone(a.n_layers)
        self.assertEqual(a.batch_size, 4)
        self.assertEqual(a.seq_len, 8)

    def test_2d_bh(self):
        """Test [batch, hidden] tensor (pooled)."""
        t = torch.randn(4, 16)
        a = Activations(data=t, dims="bh")
        self.assertEqual(a.shape, (4, 16))
        self.assertFalse("s" in a.dims)
        self.assertFalse("l" in a.dims)
        self.assertIsNone(a.n_layers)
        self.assertIsNone(a.seq_len)

    def test_3d_blh(self):
        """Test [batch, layer, hidden] tensor."""
        t = torch.randn(4, 3, 16)
        a = Activations(data=t, dims="blh", layers=(0, 5, 10))
        self.assertEqual(a.shape, (4, 3, 16))
        self.assertEqual(a.n_layers, 3)
        self.assertIsNone(a.seq_len)

    def test_layers_stored(self):
        t = torch.randn(4, 2, 8, 16)
        a = _make_flat_blsh(t, torch.ones(4, 8), layers=(5, 10))
        self.assertEqual(a.layers, (5, 10))

    def test_invalid_dims_raises(self):
        t = torch.randn(4, 8, 16)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="xyz")

    def test_missing_offsets_raises(self):
        t = torch.randn(32, 16)  # flat data
        with self.assertRaises(ValueError):
            Activations(data=t, dims="bsh")  # offsets required

    def test_missing_det_raises(self):
        t = torch.randn(32, 16)
        offsets = torch.tensor([0, 16, 32], dtype=torch.int64)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="bsh", offsets=offsets)  # det required

    def test_missing_layers_raises(self):
        t = torch.randn(32, 2, 16)
        offsets = torch.tensor([0, 16, 32], dtype=torch.int64)
        det = torch.ones(32, dtype=torch.bool)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="blsh", offsets=offsets, det=det)  # layers required

    def test_direct_flat_construction(self):
        """Test directly constructing flat+offsets Activations."""
        data = torch.randn(10, 8)  # 10 total tokens, hidden=8
        offsets = torch.tensor([0, 3, 7, 10], dtype=torch.int64)  # 3 samples
        det = torch.ones(10, dtype=torch.bool)
        a = Activations(data=data, dims="bsh", offsets=offsets, det=det)
        self.assertEqual(a.batch_size, 3)
        self.assertEqual(a.total_tokens, 10)
        self.assertEqual(a.seq_len, 4)  # max(3, 4, 3)


class TestActivationsDims(unittest.TestCase):
    def _acts_blsh(self, n_layers=2, batch=4, seq=8, d_model=16):
        t = torch.randn(batch, n_layers, seq, d_model)
        return _make_flat_blsh(t, torch.ones(batch, seq), layers=tuple(range(n_layers)))

    def test_has_layer(self):
        a = self._acts_blsh(n_layers=2)
        self.assertTrue("l" in a.dims)

    def test_has_batch(self):
        a = self._acts_blsh()
        self.assertTrue("b" in a.dims)

    def test_has_seq(self):
        a = self._acts_blsh()
        self.assertTrue("s" in a.dims)

    def test_has_hidden(self):
        a = self._acts_blsh()
        self.assertTrue("h" in a.dims)

    def test_bh_no_seq_no_layer(self):
        t = torch.randn(4, 16)
        a = Activations(data=t, dims="bh")
        self.assertFalse("s" in a.dims)
        self.assertFalse("l" in a.dims)


class TestActivationsSelect(unittest.TestCase):
    def _acts(self, layer_indices):
        n = len(layer_indices)
        t = torch.arange(4 * n * 8 * 16).reshape(4, n, 8, 16).float()
        return _make_flat_blsh(t, torch.ones(4, 8), layers=tuple(layer_indices))

    def test_select_single_layer(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select_layers(5)
        self.assertFalse("l" in s.dims)
        self.assertIsNone(s.layers)
        # data should be [T=32, hidden=16] (flat, no layer dim)
        self.assertEqual(s.data.ndim, 2)

    def test_select_multiple_layers(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select_layers([5, 15])
        self.assertEqual(s.n_layers, 2)
        self.assertEqual(s.layers, (5, 15))
        self.assertTrue("l" in s.dims)

    def test_select_preserves_offsets(self):
        a = self._acts([0, 5])
        s = a.select_layers(5)
        self.assertTrue(torch.equal(s.offsets, a.offsets))
        self.assertTrue(torch.equal(s.det, a.det))

    def test_select_invalid_layer_raises(self):
        a = self._acts([0, 5, 10])
        with self.assertRaises(ValueError):
            a.select_layers(7)


class TestActivationsPool(unittest.TestCase):
    def _acts(self, det_mask=None):
        t = torch.ones(2, 8, 4)  # [batch, seq, hidden]
        if det_mask is None:
            det_mask = torch.ones(2, 8)
        return Activations.from_padded(t, det_mask, dims="bsh")

    def test_mean_pool_removes_seq(self):
        a = self._acts()
        p = a.mean_pool()
        self.assertFalse("s" in p.dims)
        self.assertEqual(p.shape, (2, 4))

    def test_last_pool_removes_seq(self):
        a = self._acts()
        p = a.last_pool()
        self.assertFalse("s" in p.dims)
        self.assertEqual(p.shape, (2, 4))

    def test_mean_pool_uses_mask(self):
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]).float()
        t = torch.arange(64).reshape(2, 8, 4).float()
        a = Activations.from_padded(t, det, dims="bsh")
        p = a.mean_pool()
        # First sample: mean of tokens 0,1
        expected_0 = t[0, :2].mean(dim=0)
        self.assertTrue(torch.allclose(p.data[0], expected_0))
        # Second sample: mean of tokens 0,1,2,3
        expected_1 = t[1, :4].mean(dim=0)
        self.assertTrue(torch.allclose(p.data[1], expected_1))

    def test_last_pool_uses_mask(self):
        det = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]).float()
        t = torch.arange(64).reshape(2, 8, 4).float()
        a = Activations.from_padded(t, det, dims="bsh")
        p = a.last_pool()
        # First sample: last valid at index 2
        self.assertTrue(torch.allclose(p.data[0], t[0, 2]))
        # Second sample: last valid at index 4
        self.assertTrue(torch.allclose(p.data[1], t[1, 4]))

    def test_mean_pool_multi_layer(self):
        t = torch.randn(2, 3, 8, 4)  # [batch, layer, seq, hidden]
        a = _make_flat_blsh(t, torch.ones(2, 8), layers=(0, 5, 10))
        p = a.mean_pool()
        self.assertEqual(p.shape, (2, 3, 4))
        self.assertTrue("l" in p.dims)

    def test_mean_pool_empty_mask_returns_zeros(self):
        det = torch.zeros(2, 8)
        a = self._acts(det_mask=det)
        p = a.mean_pool()
        self.assertTrue(torch.allclose(p.data, torch.zeros(2, 4)))


class TestActivationsExtractTokens(unittest.TestCase):
    def test_extract_tokens_basic(self):
        t = torch.arange(64).reshape(2, 8, 4).float()
        det = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]).float()
        a = Activations.from_padded(t, det, dims="bsh")
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (5, 4))  # 3 + 2 = 5 tokens
        self.assertEqual(tps.tolist(), [3, 2])

    def test_extract_tokens_requires_single_layer(self):
        t = torch.randn(4, 2, 8, 16)
        a = _make_flat_blsh(t, torch.ones(4, 8), layers=(0, 1))
        with self.assertRaises(ValueError):
            a.extract_tokens()

    def test_extract_tokens_empty_mask(self):
        t = torch.randn(2, 8, 4)
        det = torch.zeros(2, 8)
        a = Activations.from_padded(t, det, dims="bsh")
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (0, 4))
        self.assertEqual(tps.tolist(), [0, 0])

    def test_extract_tokens_correct_values(self):
        t = torch.arange(64).reshape(2, 8, 4).float()
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]).float()
        a = Activations.from_padded(t, det, dims="bsh")
        features, _ = a.extract_tokens()
        # First 2 from batch 0, then 1 from batch 1
        self.assertTrue(torch.equal(features[0], t[0, 0]))
        self.assertTrue(torch.equal(features[1], t[0, 1]))
        self.assertTrue(torch.equal(features[2], t[1, 0]))


class TestActivationsDevice(unittest.TestCase):
    def _acts(self):
        t = torch.randn(2, 4, 8)
        return Activations.from_padded(t, torch.ones(2, 4), dims="bsh")

    def test_to_cpu(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertEqual(c.data.device.type, "cpu")

    def test_to_preserves_offsets_det(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertTrue(torch.equal(c.offsets, a.offsets))
        self.assertTrue(torch.equal(c.det, a.det))


class TestActivationsEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        t = torch.randn(1, 8, 16)
        a = Activations.from_padded(t, torch.ones(1, 8), dims="bsh")
        self.assertEqual(a.batch_size, 1)
        p = a.mean_pool()
        self.assertEqual(p.shape, (1, 16))

    def test_single_token(self):
        t = torch.randn(2, 1, 16)
        a = Activations.from_padded(t, torch.ones(2, 1), dims="bsh")
        self.assertEqual(a.seq_len, 1)
        p = a.mean_pool()
        self.assertEqual(p.shape, (2, 16))

    def test_partial_detection_mask(self):
        t = torch.ones(3, 8, 4)
        det = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]).float()
        a = Activations.from_padded(t, det, dims="bsh")
        p = a.mean_pool()
        self.assertEqual(p.shape, (3, 4))
        # Second sample should be zeros (no valid tokens)
        self.assertTrue(torch.allclose(p.data[1], torch.zeros(4)))


class TestActivationsDtypes(unittest.TestCase):
    """Test dtype handling for activation tensors."""

    def _make_acts(self, tensor, mask=None):
        batch, seq = tensor.shape[0], tensor.shape[-2]
        if mask is None:
            mask = torch.ones(batch, seq)
        return Activations.from_padded(tensor, mask, dims="bsh")

    def test_float32_preserved(self):
        t = torch.randn(1, 4, 8, dtype=torch.float32)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float32)

    def test_float64_preserved(self):
        t = torch.randn(1, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float64)

    def test_float16_preserved(self):
        t = torch.randn(1, 4, 8, dtype=torch.float16)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float16)

    def test_bfloat16_preserved(self):
        t = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.bfloat16)

    def test_int32_cast_to_float(self):
        t = torch.randint(0, 100, (1, 4, 8), dtype=torch.int32)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float32)

    def test_int64_cast_to_float(self):
        t = torch.randint(0, 100, (1, 4, 8), dtype=torch.int64)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float32)

    def test_bool_cast_to_float(self):
        t = torch.zeros(1, 4, 8, dtype=torch.bool)
        a = self._make_acts(t)
        self.assertEqual(a.data.dtype, torch.float32)

    def test_cast_preserves_values(self):
        t = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
        a = self._make_acts(t)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(a.data, expected))

    def test_mean_pool_preserves_dtype_float16(self):
        t = torch.randn(1, 4, 8, dtype=torch.float16)
        a = self._make_acts(t)
        p = a.mean_pool()
        self.assertEqual(p.data.dtype, torch.float16)

    def test_mean_pool_preserves_dtype_float64(self):
        t = torch.randn(1, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        p = a.mean_pool()
        self.assertEqual(p.data.dtype, torch.float64)

    def test_select_preserves_dtype(self):
        t = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        a = _make_flat_blsh(t, torch.ones(2, 4), layers=(0, 1))
        s = a.select_layers(0)
        self.assertEqual(s.data.dtype, torch.float16)

    def test_extract_tokens_preserves_dtype(self):
        t = torch.randn(1, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        features, _ = a.extract_tokens()
        self.assertEqual(features.dtype, torch.float64)


class TestIterLayers(unittest.TestCase):
    def test_iter_layers_blsh(self):
        t = torch.randn(2, 3, 4, 8)
        a = _make_flat_blsh(t, torch.ones(2, 4), layers=(5, 10, 15))
        layers_seen = []
        for layer_idx, acts in a.iter_layers():
            layers_seen.append(layer_idx)
            self.assertFalse("l" in acts.dims)
            # Each single layer acts: [T, hidden] where T=8 total tokens
            self.assertEqual(acts.data.shape, (8, 8))
        self.assertEqual(layers_seen, [5, 10, 15])

    def test_iter_layers_bsh(self):
        t = torch.randn(2, 4, 8)
        a = Activations.from_padded(t, torch.ones(2, 4), dims="bsh")
        layers_seen = []
        for layer_idx, acts in a.iter_layers():
            layers_seen.append(layer_idx)
            self.assertEqual(acts.data.shape, (8, 8))  # [T=8, hidden=8]
        self.assertEqual(layers_seen, [0])


class TestFromPadded(unittest.TestCase):
    def test_from_padded_roundtrip(self):
        """Test that from_padded -> to_padded round-trips correctly."""
        t = torch.arange(48).reshape(2, 6, 4).float()
        det = torch.tensor([
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ]).float()
        a = Activations.from_padded(t, det, dims="bsh")

        padded, padded_det = a.to_padded()
        # Without attention_mask, only det=True tokens are kept:
        # sample 0: 3 tokens, sample 1: 5 tokens → max_seq=5
        self.assertEqual(padded.shape, (2, 5, 4))
        self.assertEqual(padded_det.shape, (2, 5))
        # Within each sample's valid range, all tokens should be det=True
        self.assertTrue(padded_det[0, :3].all())
        self.assertTrue(padded_det[1, :5].all())
        # Padding positions should be False
        self.assertFalse(padded_det[0, 3:].any())
        # Check values match
        self.assertTrue(torch.equal(padded[0, :3], t[0, :3]))
        self.assertTrue(torch.equal(padded[1, :5], t[1, :5]))

    def test_from_padded_with_attention_mask(self):
        """Test from_padded with separate attention and detection masks."""
        t = torch.arange(24).reshape(2, 3, 4).float()
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).float()
        det_mask = torch.tensor([[0, 1, 1], [1, 0, 0]]).float()

        a = Activations.from_padded(t, det_mask, dims="bsh", attention_mask=attention_mask)

        # Sample 0: 3 tokens kept (all attended), 2 are det
        # Sample 1: 2 tokens kept, 1 is det
        self.assertEqual(a.total_tokens, 5)  # 3 + 2
        self.assertEqual(a.batch_size, 2)
        self.assertEqual(a.det.sum().item(), 3)  # 2 + 1

    def test_from_padded_blsh_roundtrip(self):
        """Test blsh round-trip."""
        t = torch.randn(2, 3, 4, 8)
        det = torch.ones(2, 4)
        a = Activations.from_padded(t, det, dims="blsh", layers=(0, 5, 10))
        padded, padded_det = a.to_padded()
        # Should recover the original data
        self.assertTrue(torch.allclose(padded, t, atol=1e-6))


class TestPadBatch(unittest.TestCase):
    def test_pad_batch_subset(self):
        """Test pad_batch with a subset of samples."""
        t = torch.arange(48).reshape(3, 4, 4).float()
        det = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
        ]).float()
        a = Activations.from_padded(t, det, dims="bsh")

        # Select samples 0 and 2
        padded, padded_det = a.pad_batch([0, 2])
        # Local max_seq = max(2, 1) = 2
        self.assertEqual(padded.shape, (2, 2, 4))
        self.assertEqual(padded_det.shape, (2, 2))

    def test_pad_batch_tensor_indices(self):
        """Test pad_batch with tensor indices."""
        t = torch.randn(4, 6, 8)
        a = Activations.from_padded(t, torch.ones(4, 6), dims="bsh")
        padded, padded_det = a.pad_batch(torch.tensor([1, 3]))
        self.assertEqual(padded.shape[0], 2)


class TestTotalTokens(unittest.TestCase):
    def test_total_tokens_bsh(self):
        data = torch.randn(10, 8)
        offsets = torch.tensor([0, 3, 7, 10], dtype=torch.int64)
        det = torch.ones(10, dtype=torch.bool)
        a = Activations(data=data, dims="bsh", offsets=offsets, det=det)
        self.assertEqual(a.total_tokens, 10)

    def test_total_tokens_none_for_bh(self):
        a = Activations(data=torch.randn(4, 8), dims="bh")
        self.assertIsNone(a.total_tokens)


class TestValidDims(unittest.TestCase):
    def test_all_valid_dims(self):
        self.assertEqual(DIMS, {"bh", "bsh", "blh", "blsh"})


if __name__ == "__main__":
    unittest.main()
