"""Tests for Activations container."""

import unittest

import torch

from probelab.processing.activations import Activations, Axis


class TestActivationsConstruction(unittest.TestCase):
    def test_from_tensor_4d(self):
        t = torch.randn(2, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
            layer_indices=[0, 1],
        )
        self.assertEqual(a.shape, (2, 4, 8, 16))
        self.assertEqual(a.n_layers, 2)
        self.assertEqual(a.batch_size, 4)
        self.assertEqual(a.seq_len, 8)
        self.assertEqual(a.d_model, 16)

    def test_from_tensor_3d(self):
        t = torch.randn(4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
        )
        self.assertEqual(a.shape, (1, 4, 8, 16))
        self.assertEqual(a.n_layers, 1)

    def test_from_tensor_2d_raises(self):
        t = torch.randn(4, 16)
        with self.assertRaises(ValueError):
            Activations.from_tensor(
                t,
                attention_mask=torch.ones(4, 1),
                input_ids=torch.ones(4, 1, dtype=torch.long),
                detection_mask=torch.ones(4, 1),
            )

    def test_layer_indices_default(self):
        t = torch.randn(3, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
        )
        self.assertEqual(a.layer_indices, [0, 1, 2])

    def test_layer_indices_custom(self):
        t = torch.randn(2, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
            layer_indices=[5, 10],
        )
        self.assertEqual(a.layer_indices, [5, 10])

    def test_batch_indices_default(self):
        t = torch.randn(1, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
        )
        # batch_indices is None by default when not provided
        self.assertIsNone(a.batch_indices)

    def test_batch_indices_custom(self):
        t = torch.randn(1, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
            batch_indices=torch.tensor([10, 20, 30, 40]),
        )
        self.assertEqual(a.batch_indices.tolist(), [10, 20, 30, 40])


class TestActivationsAxes(unittest.TestCase):
    def _acts(self, n_layers=2, batch=4, seq=8, d_model=16, layer_indices=None):
        t = torch.randn(n_layers, batch, seq, d_model)
        return Activations.from_tensor(
            t,
            attention_mask=torch.ones(batch, seq),
            input_ids=torch.ones(batch, seq, dtype=torch.long),
            detection_mask=torch.ones(batch, seq),
            layer_indices=layer_indices or list(range(n_layers)),
        )

    def test_has_axis_layer(self):
        a = self._acts(n_layers=2)
        self.assertTrue(a.has_axis(Axis.LAYER))

    def test_has_axis_batch(self):
        a = self._acts()
        self.assertTrue(a.has_axis(Axis.BATCH))

    def test_has_axis_seq(self):
        a = self._acts()
        self.assertTrue(a.has_axis(Axis.SEQ))

    def test_has_axis_hidden(self):
        a = self._acts()
        self.assertTrue(a.has_axis(Axis.HIDDEN))

    def test_axis_size_layer(self):
        a = self._acts(n_layers=3)
        self.assertEqual(a.axis_size(Axis.LAYER), 3)

    def test_axis_size_batch(self):
        a = self._acts(batch=5)
        self.assertEqual(a.axis_size(Axis.BATCH), 5)

    def test_axis_size_seq(self):
        a = self._acts(seq=10)
        self.assertEqual(a.axis_size(Axis.SEQ), 10)

    def test_axis_size_hidden(self):
        a = self._acts(d_model=32)
        self.assertEqual(a.axis_size(Axis.HIDDEN), 32)


class TestActivationsSelect(unittest.TestCase):
    def _acts(self, layer_indices):
        n = len(layer_indices)
        t = torch.arange(n * 4 * 8 * 16).reshape(n, 4, 8, 16).float()
        return Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
            layer_indices=layer_indices,
        )

    def test_select_single_layer(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select(layer=5)
        # After selecting single layer, LAYER axis is removed
        self.assertFalse(s.has_axis(Axis.LAYER))
        # When LAYER axis is removed, layer_indices is empty
        self.assertEqual(s.layer_indices, [])
        # Shape should be [batch, seq, hidden] instead of [layer, batch, seq, hidden]
        self.assertEqual(len(s.shape), 3)

    def test_select_multiple_layers(self):
        a = self._acts([0, 5, 10, 15])
        s = a.select(layers=[5, 15])
        self.assertEqual(s.n_layers, 2)
        self.assertEqual(s.layer_indices, [5, 15])
        self.assertTrue(s.has_axis(Axis.LAYER))

    def test_select_preserves_data(self):
        a = self._acts([0, 5, 10])
        s = a.select(layer=5)
        expected = a.activations[1]  # layer 5 is at index 1
        self.assertTrue(torch.equal(s.activations, expected))

    def test_select_invalid_layer_raises(self):
        a = self._acts([0, 5, 10])
        with self.assertRaises(ValueError):
            a.select(layer=7)

    def test_select_preserves_metadata(self):
        a = self._acts([0, 5])
        s = a.select(layer=5)
        self.assertTrue(torch.equal(s.attention_mask, a.attention_mask))
        self.assertTrue(torch.equal(s.detection_mask, a.detection_mask))
        self.assertTrue(torch.equal(s.input_ids, a.input_ids))


class TestActivationsPool(unittest.TestCase):
    def _acts(self, det_mask=None):
        t = torch.ones(1, 2, 8, 4)  # 1 layer, 2 batch, 8 seq, 4 hidden
        if det_mask is None:
            det_mask = torch.ones(2, 8)
        return Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det_mask,
            layer_indices=[0],
        )

    def test_pool_mean_removes_seq_axis(self):
        a = self._acts()
        p = a.pool(dim="sequence", method="mean")
        self.assertFalse(p.has_axis(Axis.SEQ))
        self.assertEqual(p.shape, (1, 2, 4))

    def test_pool_max_removes_seq_axis(self):
        a = self._acts()
        p = a.pool(dim="sequence", method="max")
        self.assertFalse(p.has_axis(Axis.SEQ))
        self.assertEqual(p.shape, (1, 2, 4))

    def test_pool_last_token_removes_seq_axis(self):
        a = self._acts()
        p = a.pool(dim="sequence", method="last_token")
        self.assertFalse(p.has_axis(Axis.SEQ))
        self.assertEqual(p.shape, (1, 2, 4))

    def test_pool_mean_uses_detection_mask(self):
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]).float()
        t = torch.arange(64).reshape(1, 2, 8, 4).float()
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        p = a.pool(dim="sequence", method="mean")
        # First sample: mean of tokens 0,1
        expected_0 = t[0, 0, :2].mean(dim=0)
        self.assertTrue(torch.allclose(p.activations[0, 0], expected_0))
        # Second sample: mean of tokens 0,1,2,3
        expected_1 = t[0, 1, :4].mean(dim=0)
        self.assertTrue(torch.allclose(p.activations[0, 1], expected_1))

    def test_pool_last_token_uses_detection_mask(self):
        det = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]).float()
        t = torch.arange(64).reshape(1, 2, 8, 4).float()
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        p = a.pool(dim="sequence", method="last_token")
        # First sample: last valid at index 2
        self.assertTrue(torch.allclose(p.activations[0, 0], t[0, 0, 2]))
        # Second sample: last valid at index 4
        self.assertTrue(torch.allclose(p.activations[0, 1], t[0, 1, 4]))

    def test_pool_multi_layer(self):
        t = torch.randn(3, 2, 8, 4)  # 3 layers
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8),
            layer_indices=[0, 5, 10],
        )
        p = a.pool(dim="sequence", method="mean")
        self.assertEqual(p.shape, (3, 2, 4))
        self.assertTrue(p.has_axis(Axis.LAYER))

    def test_pool_empty_mask_returns_zeros(self):
        det = torch.zeros(2, 8)
        a = self._acts(det_mask=det)
        p = a.pool(dim="sequence", method="mean")
        self.assertTrue(torch.allclose(p.activations, torch.zeros(1, 2, 4)))

    def test_pool_layer_mean(self):
        t = torch.randn(3, 2, 8, 4)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=torch.ones(2, 8),
            layer_indices=[0, 1, 2],
        )
        p = a.pool(dim="layer", method="mean")
        self.assertFalse(p.has_axis(Axis.LAYER))
        expected = t.mean(dim=0)
        self.assertTrue(torch.allclose(p.activations, expected))


class TestActivationsExtractTokens(unittest.TestCase):
    def test_extract_tokens_basic(self):
        t = torch.arange(64).reshape(1, 2, 8, 4).float()
        det = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]).float()
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (5, 4))  # 3 + 2 = 5 tokens
        self.assertEqual(tps.tolist(), [3, 2])

    def test_extract_tokens_requires_single_layer(self):
        t = torch.randn(2, 4, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(4, 8),
            input_ids=torch.ones(4, 8, dtype=torch.long),
            detection_mask=torch.ones(4, 8),
            layer_indices=[0, 1],
        )
        with self.assertRaises(ValueError):
            a.extract_tokens()

    def test_extract_tokens_empty_mask(self):
        t = torch.randn(1, 2, 8, 4)
        det = torch.zeros(2, 8)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        features, tps = a.extract_tokens()
        self.assertEqual(features.shape, (0, 4))
        self.assertEqual(tps.tolist(), [0, 0])

    def test_extract_tokens_correct_values(self):
        t = torch.arange(64).reshape(1, 2, 8, 4).float()
        det = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]).float()
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 8),
            input_ids=torch.ones(2, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        features, _ = a.extract_tokens()
        # First 2 from batch 0, then 1 from batch 1
        self.assertTrue(torch.equal(features[0], t[0, 0, 0]))
        self.assertTrue(torch.equal(features[1], t[0, 0, 1]))
        self.assertTrue(torch.equal(features[2], t[0, 1, 0]))


class TestActivationsDevice(unittest.TestCase):
    def _acts(self):
        t = torch.randn(1, 2, 4, 8)
        return Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 4),
            input_ids=torch.ones(2, 4, dtype=torch.long),
            detection_mask=torch.ones(2, 4),
            layer_indices=[0],
        )

    def test_to_cpu(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertEqual(c.activations.device.type, "cpu")

    def test_to_dtype(self):
        a = self._acts()
        f16 = a.to(torch.float16)
        self.assertEqual(f16.activations.dtype, torch.float16)

    def test_to_preserves_metadata(self):
        a = self._acts()
        c = a.to("cpu")
        self.assertTrue(torch.equal(c.attention_mask, a.attention_mask))
        self.assertEqual(c.layer_indices, a.layer_indices)
        # batch_indices is None by default
        self.assertIsNone(c.batch_indices)
        self.assertIsNone(a.batch_indices)


class TestActivationsEdgeCases(unittest.TestCase):
    def test_single_sample(self):
        t = torch.randn(1, 1, 8, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(1, 8),
            input_ids=torch.ones(1, 8, dtype=torch.long),
            detection_mask=torch.ones(1, 8),
            layer_indices=[0],
        )
        self.assertEqual(a.batch_size, 1)
        p = a.pool(dim="sequence", method="mean")
        self.assertEqual(p.shape, (1, 1, 16))

    def test_single_token(self):
        t = torch.randn(1, 2, 1, 16)
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 1),
            input_ids=torch.ones(2, 1, dtype=torch.long),
            detection_mask=torch.ones(2, 1),
            layer_indices=[0],
        )
        self.assertEqual(a.seq_len, 1)
        p = a.pool(dim="sequence", method="max")
        self.assertEqual(p.shape, (1, 2, 16))

    def test_partial_detection_mask(self):
        t = torch.ones(1, 3, 8, 4)
        det = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
            ]
        ).float()
        a = Activations.from_tensor(
            t,
            attention_mask=torch.ones(3, 8),
            input_ids=torch.ones(3, 8, dtype=torch.long),
            detection_mask=det,
            layer_indices=[0],
        )
        p = a.pool(dim="sequence", method="mean")
        self.assertEqual(p.shape, (1, 3, 4))
        # Second sample should be zeros (no valid tokens)
        self.assertTrue(torch.allclose(p.activations[0, 1], torch.zeros(4)))


class TestActivationsDtypes(unittest.TestCase):
    """Test dtype handling for activation tensors."""

    def _make_acts(self, tensor, **kwargs):
        """Helper to create activations with given tensor."""
        batch, seq = tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0], tensor.shape[-2]
        defaults = {
            "attention_mask": torch.ones(batch, seq),
            "input_ids": torch.ones(batch, seq, dtype=torch.long),
            "detection_mask": torch.ones(batch, seq),
        }
        defaults.update(kwargs)
        return Activations.from_tensor(tensor, **defaults)

    # --- Activation tensor dtype tests ---

    def test_float32_preserved(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float32)

    def test_float64_preserved(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float64)

    def test_float16_preserved(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float16)

    def test_bfloat16_preserved(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.bfloat16)

    def test_int32_cast_to_float(self):
        t = torch.randint(0, 100, (1, 2, 4, 8), dtype=torch.int32)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float32)

    def test_int64_cast_to_float(self):
        t = torch.randint(0, 100, (1, 2, 4, 8), dtype=torch.int64)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float32)

    def test_bool_cast_to_float(self):
        t = torch.zeros(1, 2, 4, 8, dtype=torch.bool)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float32)

    def test_int8_cast_to_float(self):
        t = torch.randint(0, 100, (1, 2, 4, 8), dtype=torch.int8)
        a = self._make_acts(t)
        self.assertEqual(a.activations.dtype, torch.float32)

    def test_cast_preserves_values(self):
        t = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.int32)
        a = self._make_acts(t)
        expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        self.assertTrue(torch.equal(a.activations, expected))

    # --- Mask dtype tests ---

    def test_detection_mask_float32(self):
        t = torch.randn(1, 2, 4, 8)
        det = torch.ones(2, 4, dtype=torch.float32)
        a = self._make_acts(t, detection_mask=det)
        self.assertEqual(a.detection_mask.dtype, torch.float32)

    def test_detection_mask_float64(self):
        t = torch.randn(1, 2, 4, 8)
        det = torch.ones(2, 4, dtype=torch.float64)
        a = self._make_acts(t, detection_mask=det)
        self.assertEqual(a.detection_mask.dtype, torch.float64)

    def test_detection_mask_bool(self):
        t = torch.randn(1, 2, 4, 8)
        det = torch.ones(2, 4, dtype=torch.bool)
        a = self._make_acts(t, detection_mask=det)
        self.assertEqual(a.detection_mask.dtype, torch.bool)

    def test_detection_mask_int32(self):
        t = torch.randn(1, 2, 4, 8)
        det = torch.ones(2, 4, dtype=torch.int32)
        a = self._make_acts(t, detection_mask=det)
        self.assertEqual(a.detection_mask.dtype, torch.int32)

    def test_attention_mask_float32(self):
        t = torch.randn(1, 2, 4, 8)
        attn = torch.ones(2, 4, dtype=torch.float32)
        a = self._make_acts(t, attention_mask=attn)
        self.assertEqual(a.attention_mask.dtype, torch.float32)

    # --- input_ids dtype tests ---

    def test_input_ids_int64(self):
        t = torch.randn(1, 2, 4, 8)
        ids = torch.ones(2, 4, dtype=torch.int64)
        a = self._make_acts(t, input_ids=ids)
        self.assertEqual(a.input_ids.dtype, torch.int64)

    def test_input_ids_int32(self):
        t = torch.randn(1, 2, 4, 8)
        ids = torch.ones(2, 4, dtype=torch.int32)
        a = self._make_acts(t, input_ids=ids)
        self.assertEqual(a.input_ids.dtype, torch.int32)

    def test_input_ids_float_raises(self):
        t = torch.randn(1, 2, 4, 8)
        ids = torch.ones(2, 4, dtype=torch.float32)
        with self.assertRaises(ValueError):
            self._make_acts(t, input_ids=ids)

    # --- Operations preserve/handle dtypes ---

    def test_pool_preserves_dtype_float16(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        a = self._make_acts(t)
        p = a.pool(dim="sequence", method="mean")
        self.assertEqual(p.activations.dtype, torch.float16)

    def test_pool_preserves_dtype_float64(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        p = a.pool(dim="sequence", method="max")
        self.assertEqual(p.activations.dtype, torch.float64)

    def test_select_preserves_dtype(self):
        t = torch.randn(2, 2, 4, 8, dtype=torch.float16)
        a = self._make_acts(t, layer_indices=[0, 1])
        s = a.select(layer=0)
        self.assertEqual(s.activations.dtype, torch.float16)

    def test_extract_tokens_preserves_dtype(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float64)
        a = self._make_acts(t)
        features, _ = a.extract_tokens()
        self.assertEqual(features.dtype, torch.float64)

    def test_to_dtype_conversion(self):
        t = torch.randn(1, 2, 4, 8, dtype=torch.float32)
        a = self._make_acts(t)
        a16 = a.to(torch.float16)
        self.assertEqual(a16.activations.dtype, torch.float16)
        a64 = a.to(torch.float64)
        self.assertEqual(a64.activations.dtype, torch.float64)


class TestLayerTensorIndices(unittest.TestCase):
    def _acts(self, layer_indices):
        n = len(layer_indices)
        t = torch.randn(n, 2, 4, 8)
        return Activations.from_tensor(
            t,
            attention_mask=torch.ones(2, 4),
            input_ids=torch.ones(2, 4, dtype=torch.long),
            detection_mask=torch.ones(2, 4),
            layer_indices=layer_indices,
        )

    def test_single_layer_index(self):
        a = self._acts([0, 5, 10, 15])
        self.assertEqual(a.get_layer_tensor_indices(5), [1])

    def test_multiple_layer_indices(self):
        a = self._acts([0, 5, 10, 15])
        self.assertEqual(a.get_layer_tensor_indices([0, 10]), [0, 2])

    def test_invalid_layer_raises(self):
        a = self._acts([0, 5, 10])
        with self.assertRaises(ValueError):
            a.get_layer_tensor_indices(7)


if __name__ == "__main__":
    unittest.main()
