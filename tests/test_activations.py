"""Tests for Activations container."""

import unittest

import torch

from probelab.processing.activations import Activations, DIMS


class TestActivationsConstruction(unittest.TestCase):
    def test_4d_blsh(self):
        """Test [batch, layer, seq, hidden] tensor."""
        t = torch.randn(4, 2, 8, 16)
        a = Activations(data=t, dims="blsh", mask=torch.ones(4, 8), layers=(0, 1))
        self.assertEqual(a.shape, (4, 2, 8, 16))
        self.assertEqual(a.n_layers, 2)
        self.assertEqual(a.batch_size, 4)
        self.assertEqual(a.seq_len, 8)
        self.assertEqual(a.hidden_size, 16)

    def test_3d_bsh(self):
        """Test [batch, seq, hidden] tensor."""
        t = torch.randn(4, 8, 16)
        a = Activations(data=t, dims="bsh", mask=torch.ones(4, 8))
        self.assertEqual(a.shape, (4, 8, 16))
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
        a = Activations(data=t, dims="blsh", mask=torch.ones(4, 8), layers=(5, 10))
        self.assertEqual(a.layers, (5, 10))

    def test_invalid_dims_raises(self):
        t = torch.randn(4, 8, 16)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="xyz", mask=torch.ones(4, 8))

    def test_missing_mask_raises(self):
        t = torch.randn(4, 8, 16)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="bsh")  # mask required

    def test_missing_layers_raises(self):
        t = torch.randn(4, 2, 8, 16)
        with self.assertRaises(ValueError):
            Activations(data=t, dims="blsh", mask=torch.ones(4, 8))  # layers required


class TestActivationsDims(unittest.TestCase):
    def _acts_blsh(self, n_layers=2, batch=4, seq=8, d_model=16):
        t = torch.randn(batch, n_layers, seq, d_model)
        return Activations(
            data=t, dims="blsh",
            mask=torch.ones(batch, seq),
            layers=tuple(range(n_layers)),
        )

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
        return Activations(
            data=t, dims="blsh",
            mask=torch.ones(4, 8),
            layers=tuple(layer_indices),
        )

    def test_select_single_layer(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select_layers(5)
        self.assertFalse("l" in s.dims)
        self.assertIsNone(s.layers)
        self.assertEqual(len(s.shape), 3)  # [batch, seq, hidden]

    def test_select_multiple_layers(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select_layers([5, 15])
        self.assertEqual(s.n_layers, 2)
        self.assertEqual(s.layers, (5, 15))
        self.assertTrue("l" in s.dims)

    def test_select_preserves_data(self):
        a = self._acts([0, 5, 10])
        s = a.select_layers(5)
        expected = a.data[:, 1]  # layer 5 is at index 1
        self.assertTrue(torch.equal(s.data, expected))

    def test_select_invalid_layer_raises(self):
        a = self._acts([0, 5, 10])
        with self.assertRaises(ValueError):
            a.select_layers(7)

    def test_select_preserves_mask(self):
        a = self._acts([0, 5])
        s = a.select_layers(5)
        self.assertTrue(torch.equal(s.mask, a.mask))


class TestActivationsPool(unittest.TestCase):
    def _acts(self, det_mask=None):
        t = torch.ones(2, 8, 4)  # [batch, seq, hidden]
        if det_mask is None:
            det_mask = torch.ones(2, 8)
        return Activations(data=t, dims="bsh", mask=det_mask)

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
        a = Activations(data=t, dims="bsh", mask=det)
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
        a = Activations(data=t, dims="bsh", mask=det)
        p = a.last_pool()
        # First sample: last valid at index 2
        self.assertTrue(torch.allclose(p.data[0], t[0, 2]))
        # Second sample: last valid at index 4
        self.assertTrue(torch.allclose(p.data[1], t[1, 4]))

    def test_mean_pool_multi_layer(self):
        t = torch.randn(2, 3, 8, 4)  # [batch, layer, seq, hidden]
        a = Activations(data=t, dims="blsh", mask=torch.ones(2, 8), layers=(0, 5, 10))
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
        a = Activations(data=t, dims="bsh", mask=det)
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (5, 4))  # 3 + 2 = 5 tokens
        self.assertEqual(tps.tolist(), [3, 2])

    def test_extract_tokens_requires_single_layer(self):
        t = torch.randn(4, 2, 8, 16)
        a = Activations(data=t, dims="blsh", mask=torch.ones(4, 8), layers=(0, 1))
        with self.assertRaises(ValueError):
            a.extract_tokens()

    def test_extract_tokens_empty_mask(self):
        t = torch.randn(2, 8, 4)
        det = torch.zeros(2, 8)
        a = Activations(data=t, dims="bsh", mask=det)
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (0, 4))
        self.assertEqual(tps.tolist(), [0, 0])

    def test_extract_tokens_correct_values(self):
        t = torch.arange(64).reshape(2, 8, 4).float()
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]).float()
        a = Activations(data=t, dims="bsh", mask=det)
        features, _ = a.extract_tokens()
        # First 2 from batch 0, then 1 from batch 1
        self.assertTrue(torch.equal(features[0], t[0, 0]))
        self.assertTrue(torch.equal(features[1], t[0, 1]))
        self.assertTrue(torch.equal(features[2], t[1, 0]))


class TestActivationsDevice(unittest.TestCase):
    def _acts(self):
        t = torch.randn(2, 4, 8)
        return Activations(data=t, dims="bsh", mask=torch.ones(2, 4))

    def test_to_cpu(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertEqual(c.data.device.type, "cpu")

    def test_to_preserves_mask(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertTrue(torch.equal(c.mask, a.mask))


class TestActivationsEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        t = torch.randn(1, 8, 16)
        a = Activations(data=t, dims="bsh", mask=torch.ones(1, 8))
        self.assertEqual(a.batch_size, 1)
        p = a.mean_pool()
        self.assertEqual(p.shape, (1, 16))

    def test_single_token(self):
        t = torch.randn(2, 1, 16)
        a = Activations(data=t, dims="bsh", mask=torch.ones(2, 1))
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
        a = Activations(data=t, dims="bsh", mask=det)
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
        return Activations(data=tensor, dims="bsh", mask=mask)

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
        expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
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
        a = Activations(data=t, dims="blsh", mask=torch.ones(2, 4), layers=(0, 1))
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
        a = Activations(data=t, dims="blsh", mask=torch.ones(2, 4), layers=(5, 10, 15))
        layers_seen = []
        for layer_idx, acts in a.iter_layers():
            layers_seen.append(layer_idx)
            self.assertFalse("l" in acts.dims)
            self.assertEqual(acts.shape, (2, 4, 8))
        self.assertEqual(layers_seen, [5, 10, 15])

    def test_iter_layers_bsh(self):
        t = torch.randn(2, 4, 8)
        a = Activations(data=t, dims="bsh", mask=torch.ones(2, 4))
        layers_seen = []
        for layer_idx, acts in a.iter_layers():
            layers_seen.append(layer_idx)
            self.assertEqual(acts.shape, (2, 4, 8))
        self.assertEqual(layers_seen, [0])


class TestValidDims(unittest.TestCase):
    def test_all_valid_dims(self):
        self.assertEqual(DIMS, {"bh", "bsh", "blh", "blsh"})


if __name__ == "__main__":
    unittest.main()
