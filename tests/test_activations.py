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
        s = a.select("l", 5)
        self.assertFalse("l" in s.dims)
        self.assertIsNone(s.layers)
        # data should be [T=32, hidden=16] (flat, no layer dim)
        self.assertEqual(s.data.ndim, 2)

    def test_select_multiple_layers(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select("l", [5, 15])
        self.assertEqual(s.n_layers, 2)
        self.assertEqual(s.layers, (5, 15))
        self.assertTrue("l" in s.dims)

    def test_select_preserves_offsets(self):
        a = self._acts([0, 5])
        s = a.select("l", 5)
        self.assertTrue(torch.equal(s.offsets, a.offsets))
        self.assertTrue(torch.equal(s.det, a.det))

    def test_select_invalid_layer_raises(self):
        a = self._acts([0, 5, 10])
        with self.assertRaises(ValueError):
            a.select("l", 7)


class TestActivationsPool(unittest.TestCase):
    def _acts(self, det_mask=None):
        t = torch.ones(2, 8, 4)  # [batch, seq, hidden]
        if det_mask is None:
            det_mask = torch.ones(2, 8)
        return Activations.from_padded(t, det_mask, dims="bsh")

    def test_mean_pool_removes_seq(self):
        a = self._acts()
        p = a.mean("s")
        self.assertFalse("s" in p.dims)
        self.assertEqual(p.shape, (2, 4))

    def test_last_pool_removes_seq(self):
        a = self._acts()
        p = a.last()
        self.assertFalse("s" in p.dims)
        self.assertEqual(p.shape, (2, 4))

    def test_mean_pool_uses_mask(self):
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]).float()
        t = torch.arange(64).reshape(2, 8, 4).float()
        a = Activations.from_padded(t, det, dims="bsh")
        p = a.mean("s")
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
        p = a.last()
        # First sample: last valid at index 2
        self.assertTrue(torch.allclose(p.data[0], t[0, 2]))
        # Second sample: last valid at index 4
        self.assertTrue(torch.allclose(p.data[1], t[1, 4]))

    def test_mean_pool_multi_layer(self):
        t = torch.randn(2, 3, 8, 4)  # [batch, layer, seq, hidden]
        a = _make_flat_blsh(t, torch.ones(2, 8), layers=(0, 5, 10))
        p = a.mean("s")
        self.assertEqual(p.shape, (2, 3, 4))
        self.assertTrue("l" in p.dims)

    def test_mean_pool_empty_mask_returns_zeros(self):
        det = torch.zeros(2, 8)
        a = self._acts(det_mask=det)
        p = a.mean("s")
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
        p = a.mean("s")
        self.assertEqual(p.shape, (1, 16))

    def test_single_token(self):
        t = torch.randn(2, 1, 16)
        a = Activations.from_padded(t, torch.ones(2, 1), dims="bsh")
        self.assertEqual(a.seq_len, 1)
        p = a.mean("s")
        self.assertEqual(p.shape, (2, 16))

    def test_partial_detection_mask(self):
        t = torch.ones(3, 8, 4)
        det = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
        ]).float()
        a = Activations.from_padded(t, det, dims="bsh")
        p = a.mean("s")
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
        p = a.mean("s")
        self.assertEqual(p.data.dtype, torch.float16)

    def test_mean_pool_preserves_dtype_float64(self):
        t = torch.randn(1, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        p = a.mean("s")
        self.assertEqual(p.data.dtype, torch.float64)

    def test_select_preserves_dtype(self):
        t = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        a = _make_flat_blsh(t, torch.ones(2, 4), layers=(0, 1))
        s = a.select("l", 0)
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


class TestMeanLayer(unittest.TestCase):
    """Tests for mean("l") — mean over layer dimension."""

    def test_mean_layer_blh(self):
        t = torch.randn(4, 3, 16)
        a = Activations(data=t, dims="blh", layers=(0, 5, 10))
        m = a.mean("l")
        self.assertEqual(m.dims, "bh")
        self.assertEqual(m.shape, (4, 16))
        self.assertTrue(torch.allclose(m.data, t.mean(dim=1)))

    def test_mean_layer_blsh(self):
        t = torch.randn(2, 3, 8, 16)
        a = _make_flat_blsh(t, torch.ones(2, 8), layers=(0, 5, 10))
        # data: [T=16, 3, 16]
        m = a.mean("l")
        self.assertEqual(m.dims, "bsh")
        self.assertIsNone(m.layers)
        self.assertEqual(m.data.shape[1], 16)  # hidden

    def test_mean_layer_no_layer_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.mean("l")


class TestMaxLayer(unittest.TestCase):
    """Tests for max("l") — max over layer dimension."""

    def test_max_layer_blh(self):
        t = torch.randn(4, 3, 16)
        a = Activations(data=t, dims="blh", layers=(0, 5, 10))
        m = a.max("l")
        self.assertEqual(m.dims, "bh")
        self.assertEqual(m.shape, (4, 16))
        self.assertTrue(torch.allclose(m.data, t.max(dim=1).values))

    def test_max_layer_blsh(self):
        t = torch.randn(2, 3, 8, 16)
        a = _make_flat_blsh(t, torch.ones(2, 8), layers=(0, 5, 10))
        m = a.max("l")
        self.assertEqual(m.dims, "bsh")
        self.assertIsNone(m.layers)

    def test_max_layer_no_layer_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.max("l")


class TestSelectSeq(unittest.TestCase):
    """Tests for select("s", idx) — token selection from flat+offsets."""

    def test_select_first_token(self):
        t = torch.arange(48).reshape(2, 6, 4).float()
        a = Activations.from_padded(t, torch.ones(2, 6), dims="bsh")
        s = a.select("s", 0)
        self.assertEqual(s.dims, "bh")
        self.assertEqual(s.shape, (2, 4))
        self.assertTrue(torch.equal(s.data[0], t[0, 0]))
        self.assertTrue(torch.equal(s.data[1], t[1, 0]))

    def test_select_last_token(self):
        det = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]]).float()
        t = torch.arange(48).reshape(2, 6, 4).float()
        a = Activations.from_padded(t, det, dims="bsh")
        s = a.select("s", -1)
        self.assertEqual(s.dims, "bh")
        # Sample 0: 3 tokens → last is t[0,2], sample 1: 5 tokens → last is t[1,4]
        self.assertTrue(torch.allclose(s.data[0], t[0, 2]))
        self.assertTrue(torch.allclose(s.data[1], t[1, 4]))

    def test_select_second_token(self):
        t = torch.arange(24).reshape(2, 3, 4).float()
        a = Activations.from_padded(t, torch.ones(2, 3), dims="bsh")
        s = a.select("s", 1)
        self.assertTrue(torch.equal(s.data[0], t[0, 1]))
        self.assertTrue(torch.equal(s.data[1], t[1, 1]))

    def test_select_negative_index(self):
        t = torch.arange(32).reshape(2, 4, 4).float()
        a = Activations.from_padded(t, torch.ones(2, 4), dims="bsh")
        s = a.select("s", -2)
        # -2 from 4 tokens = index 2
        self.assertTrue(torch.equal(s.data[0], t[0, 2]))
        self.assertTrue(torch.equal(s.data[1], t[1, 2]))

    def test_select_seq_no_seq_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.select("s", 0)

    def test_select_seq_list_raises(self):
        t = torch.randn(2, 4, 8)
        a = Activations.from_padded(t, torch.ones(2, 4), dims="bsh")
        with self.assertRaises(ValueError):
            a.select("s", [0, 1])

    def test_last_sugar(self):
        """last() is sugar for select("s", -1)."""
        t = torch.randn(2, 4, 8)
        a = Activations.from_padded(t, torch.ones(2, 4), dims="bsh")
        s1 = a.last()
        s2 = a.select("s", -1)
        self.assertTrue(torch.allclose(s1.data, s2.data))


class TestFlatten(unittest.TestCase):
    """Tests for flatten() — concat layers into hidden dim."""

    def test_flatten_blh(self):
        t = torch.randn(4, 3, 16)
        a = Activations(data=t, dims="blh", layers=(0, 5, 10))
        f = a.flatten()
        self.assertEqual(f.dims, "bh")
        self.assertEqual(f.shape, (4, 48))  # 3 * 16

    def test_flatten_blsh(self):
        t = torch.randn(2, 3, 8, 16)
        a = _make_flat_blsh(t, torch.ones(2, 8), layers=(0, 5, 10))
        # data: [T=16, 3, 16]
        f = a.flatten()
        self.assertEqual(f.dims, "bsh")
        self.assertEqual(f.data.shape[1], 48)  # 3 * 16
        self.assertIsNotNone(f.offsets)

    def test_flatten_no_layer_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.flatten()


class TestEma(unittest.TestCase):
    """Tests for ema() — EMA + max over sequence."""

    def test_ema_removes_seq(self):
        t = torch.randn(2, 8, 4)
        a = Activations.from_padded(t, torch.ones(2, 8), dims="bsh")
        e = a.ema(alpha=0.5)
        self.assertEqual(e.dims, "bh")
        self.assertEqual(e.shape, (2, 4))

    def test_ema_no_seq_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.ema()


class TestRolling(unittest.TestCase):
    """Tests for rolling() — rolling mean + max over sequence."""

    def test_rolling_removes_seq(self):
        t = torch.randn(2, 8, 4)
        a = Activations.from_padded(t, torch.ones(2, 8), dims="bsh")
        r = a.rolling(window=3)
        self.assertEqual(r.dims, "bh")
        self.assertEqual(r.shape, (2, 4))

    def test_rolling_no_seq_raises(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.rolling()


class TestDimValidation(unittest.TestCase):
    """Tests for invalid dim arguments."""

    def test_mean_invalid_dim(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.mean("x")

    def test_max_invalid_dim(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.max("x")

    def test_select_invalid_dim(self):
        a = Activations(data=torch.randn(4, 16), dims="bh")
        with self.assertRaises(ValueError):
            a.select("x", 0)


class TestCat(unittest.TestCase):
    """Tests for Activations.cat() — batch concatenation."""

    # --- bh ---

    def test_cat_bh_basic(self):
        a = Activations(data=torch.randn(3, 8), dims="bh")
        b = Activations(data=torch.randn(2, 8), dims="bh")
        c = Activations.cat([a, b])
        self.assertEqual(c.dims, "bh")
        self.assertEqual(c.batch_size, 5)
        self.assertEqual(c.hidden_size, 8)

    def test_cat_bh_three_items(self):
        items = [Activations(data=torch.randn(i + 1, 4), dims="bh") for i in range(3)]
        c = Activations.cat(items)
        self.assertEqual(c.batch_size, 6)  # 1 + 2 + 3

    def test_cat_bh_data_order(self):
        a = Activations(data=torch.ones(2, 4), dims="bh")
        b = Activations(data=torch.zeros(3, 4), dims="bh")
        c = Activations.cat([a, b])
        self.assertTrue(torch.equal(c.data[:2], torch.ones(2, 4)))
        self.assertTrue(torch.equal(c.data[2:], torch.zeros(3, 4)))

    # --- blh ---

    def test_cat_blh_basic(self):
        a = Activations(data=torch.randn(3, 2, 8), dims="blh", layers=(5, 10))
        b = Activations(data=torch.randn(4, 2, 8), dims="blh", layers=(5, 10))
        c = Activations.cat([a, b])
        self.assertEqual(c.dims, "blh")
        self.assertEqual(c.batch_size, 7)
        self.assertEqual(c.n_layers, 2)
        self.assertEqual(c.layers, (5, 10))

    def test_cat_blh_layers_mismatch_raises(self):
        a = Activations(data=torch.randn(2, 2, 8), dims="blh", layers=(5, 10))
        b = Activations(data=torch.randn(2, 2, 8), dims="blh", layers=(5, 15))
        with self.assertRaises(ValueError):
            Activations.cat([a, b])

    # --- bsh ---

    def test_cat_bsh_basic(self):
        t1 = torch.randn(2, 4, 8)
        a = Activations.from_padded(t1, torch.ones(2, 4), dims="bsh")
        t2 = torch.randn(3, 6, 8)
        b = Activations.from_padded(t2, torch.ones(3, 6), dims="bsh")
        c = Activations.cat([a, b])
        self.assertEqual(c.dims, "bsh")
        self.assertEqual(c.batch_size, 5)
        self.assertEqual(c.total_tokens, 2 * 4 + 3 * 6)  # 8 + 18 = 26
        self.assertEqual(c.hidden_size, 8)

    def test_cat_bsh_offsets_correct(self):
        """Numerically verify merged offsets."""
        data_a = torch.randn(7, 4)
        off_a = torch.tensor([0, 3, 7], dtype=torch.int64)
        det_a = torch.ones(7, dtype=torch.bool)
        a = Activations(data=data_a, dims="bsh", offsets=off_a, det=det_a)

        data_b = torch.randn(8, 4)
        off_b = torch.tensor([0, 5, 8], dtype=torch.int64)
        det_b = torch.ones(8, dtype=torch.bool)
        b = Activations(data=data_b, dims="bsh", offsets=off_b, det=det_b)

        c = Activations.cat([a, b])
        expected_offsets = torch.tensor([0, 3, 7, 12, 15], dtype=torch.int64)
        self.assertTrue(torch.equal(c.offsets, expected_offsets))

    def test_cat_bsh_data_det_order(self):
        """Data and det tensors are concatenated in order."""
        data_a = torch.ones(5, 4)
        off_a = torch.tensor([0, 2, 5], dtype=torch.int64)
        det_a = torch.ones(5, dtype=torch.bool)
        a = Activations(data=data_a, dims="bsh", offsets=off_a, det=det_a)

        data_b = torch.zeros(3, 4)
        off_b = torch.tensor([0, 3], dtype=torch.int64)
        det_b = torch.zeros(3, dtype=torch.bool)
        b = Activations(data=data_b, dims="bsh", offsets=off_b, det=det_b)

        c = Activations.cat([a, b])
        self.assertTrue(torch.equal(c.data[:5], torch.ones(5, 4)))
        self.assertTrue(torch.equal(c.data[5:], torch.zeros(3, 4)))
        self.assertTrue(c.det[:5].all())
        self.assertFalse(c.det[5:].any())

    def test_cat_bsh_variable_lengths(self):
        """Samples with different sequence lengths."""
        # 3 tokens + 1 token
        data_a = torch.randn(4, 8)
        off_a = torch.tensor([0, 3, 4], dtype=torch.int64)
        det_a = torch.ones(4, dtype=torch.bool)
        a = Activations(data=data_a, dims="bsh", offsets=off_a, det=det_a)

        # 5 tokens
        data_b = torch.randn(5, 8)
        off_b = torch.tensor([0, 5], dtype=torch.int64)
        det_b = torch.ones(5, dtype=torch.bool)
        b = Activations(data=data_b, dims="bsh", offsets=off_b, det=det_b)

        c = Activations.cat([a, b])
        self.assertEqual(c.batch_size, 3)
        self.assertEqual(c.total_tokens, 9)
        expected_offsets = torch.tensor([0, 3, 4, 9], dtype=torch.int64)
        self.assertTrue(torch.equal(c.offsets, expected_offsets))

    # --- blsh ---

    def test_cat_blsh_basic(self):
        t1 = torch.randn(2, 3, 4, 8)
        a = _make_flat_blsh(t1, torch.ones(2, 4), layers=(0, 5, 10))
        t2 = torch.randn(3, 3, 6, 8)
        b = _make_flat_blsh(t2, torch.ones(3, 6), layers=(0, 5, 10))
        c = Activations.cat([a, b])
        self.assertEqual(c.dims, "blsh")
        self.assertEqual(c.batch_size, 5)
        self.assertEqual(c.n_layers, 3)
        self.assertEqual(c.layers, (0, 5, 10))
        self.assertEqual(c.total_tokens, 2 * 4 + 3 * 6)

    # --- Validation ---

    def test_cat_empty_raises(self):
        with self.assertRaises(ValueError):
            Activations.cat([])

    def test_cat_single_returns_same(self):
        a = Activations(data=torch.randn(3, 8), dims="bh")
        c = Activations.cat([a])
        self.assertIs(c, a)

    def test_cat_dims_mismatch_raises(self):
        a = Activations(data=torch.randn(3, 8), dims="bh")
        b = Activations(data=torch.randn(3, 2, 8), dims="blh", layers=(0, 1))
        with self.assertRaises(ValueError):
            Activations.cat([a, b])

    def test_cat_hidden_size_mismatch_raises(self):
        a = Activations(data=torch.randn(3, 8), dims="bh")
        b = Activations(data=torch.randn(3, 16), dims="bh")
        with self.assertRaises(ValueError):
            Activations.cat([a, b])

    # --- Dtype ---

    def test_cat_preserves_float16(self):
        a = Activations(data=torch.randn(2, 8, dtype=torch.float16), dims="bh")
        b = Activations(data=torch.randn(3, 8, dtype=torch.float16), dims="bh")
        c = Activations.cat([a, b])
        self.assertEqual(c.data.dtype, torch.float16)

    # --- Integration ---

    def test_cat_then_mean_s(self):
        """cat followed by mean('s') produces correct result."""
        t1 = torch.ones(2, 4, 8)
        a = Activations.from_padded(t1, torch.ones(2, 4), dims="bsh")
        t2 = torch.ones(3, 6, 8) * 2
        b = Activations.from_padded(t2, torch.ones(3, 6), dims="bsh")
        c = Activations.cat([a, b])
        pooled = c.mean("s")
        self.assertEqual(pooled.dims, "bh")
        self.assertEqual(pooled.batch_size, 5)
        # First 2 samples had value 1.0, next 3 had value 2.0
        self.assertTrue(torch.allclose(pooled.data[:2], torch.ones(2, 8)))
        self.assertTrue(torch.allclose(pooled.data[2:], torch.ones(3, 8) * 2))


class TestValidDims(unittest.TestCase):
    def test_all_valid_dims(self):
        self.assertEqual(DIMS, {"bh", "bsh", "blh", "blsh"})


if __name__ == "__main__":
    unittest.main()
