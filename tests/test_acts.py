"""Tests for lazy Acts container."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from probelab.processing.acts import Acts, load
from probelab.probes.logistic import Logistic


try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class TestActsBasics(unittest.TestCase):
    def test_requires_seq_mask_when_seq_dim_present(self):
        t = torch.randn(2, 3, 4)
        with self.assertRaises(ValueError):
            Acts(t, dims="bsh")

    def test_chain_select_and_pool_matches_manual(self):
        torch.manual_seed(0)
        t = torch.randn(3, 2, 4, 5)
        mask = torch.tensor(
            [[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
            dtype=torch.float32,
        )

        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[10, 20])
        got = acts.select_layers(20).mean_pool().realize().squeeze(1)

        masked = t[:, 1] * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        expected = masked.sum(dim=1) / denom

        self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_slice_batch_and_iter_batches(self):
        torch.manual_seed(0)
        t = torch.randn(5, 2, 4, 3)
        mask = torch.ones(5, 4)
        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[0, 1])

        sliced = acts.slice_batch([3, 1, 4])
        self.assertEqual(sliced.shape, (3, 2, 4, 3))
        self.assertTrue(torch.allclose(sliced.realize(), t[[3, 1, 4]], atol=0, rtol=0))

        chunks = list(sliced.iter_batches(2))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].shape, (2, 2, 4, 3))
        self.assertEqual(chunks[1].shape, (1, 2, 4, 3))

    def test_iter_layers_returns_expected_views(self):
        torch.manual_seed(0)
        t = torch.randn(4, 3, 6, 7)
        mask = torch.ones(4, 6)
        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[2, 4, 6]).mean_pool()

        pairs = list(acts.iter_layers())
        self.assertEqual([lid for lid, _ in pairs], [2, 4, 6])
        self.assertEqual(pairs[0][1].shape, (4, 7))


@unittest.skipUnless(HAS_H5PY, "h5py is required for disk-backed Acts tests")
class TestActsDiskIO(unittest.TestCase):
    def test_save_load_roundtrip_and_select_pushdown(self):
        torch.manual_seed(0)
        t = torch.randn(8, 4, 5, 6)
        mask = torch.randint(0, 2, (8, 5)).float()

        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[10, 11, 12, 13])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acts.h5"
            acts.save(path)

            loaded = load(path)
            self.assertEqual(loaded.dims, "blsh")
            self.assertTrue(torch.allclose(loaded.realize(), t, atol=1e-6))

            l12 = loaded.select_layers(12).realize().squeeze(1)
            self.assertTrue(torch.allclose(l12, t[:, 2], atol=1e-6))

    def test_cache_creates_disk_backed_plan(self):
        torch.manual_seed(0)
        t = torch.randn(6, 2, 4, 5)
        mask = torch.ones(6, 4)
        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[0, 1]).mean_pool()

        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "pooled.h5"
            cached = acts.cache(cache_path)
            self.assertEqual(cached.dims, "blh")
            self.assertTrue(torch.allclose(cached.realize(), acts.realize(), atol=1e-6))

    def test_disk_pushdown_mean_pool_matches_manual(self):
        torch.manual_seed(0)
        t = torch.randn(7, 3, 5, 4)
        mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.float32,
        )
        acts = Acts(t, dims="blsh", seq_mask=mask, layer_ids=[0, 1, 2])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "acts.h5"
            acts.save(path)

            pooled = load(path).mean_pool().realize()
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(1)
            manual = (t * mask.unsqueeze(1).unsqueeze(-1)).sum(dim=2) / denom
            self.assertTrue(torch.allclose(pooled, manual, atol=1e-6))


class TestActsLogisticIntegration(unittest.TestCase):
    def test_logistic_fit_predict_sequence_level_on_acts(self):
        torch.manual_seed(0)
        n = 80
        d = 16
        y = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])

        x = torch.randn(n, d) * 0.1
        x[: n // 2, 0] += 4.0
        x[n // 2 :, 0] -= 4.0

        acts = Acts(x.unsqueeze(1), dims="blh", layer_ids=[0])
        probe = Logistic(max_iter=100, device="cpu").fit(acts, y)
        probs = probe.predict(acts)

        self.assertEqual(tuple(probs.shape), (n,))
        self.assertGreater(probs[: n // 2].mean().item(), probs[n // 2 :].mean().item())

    def test_logistic_fit_predict_token_level_on_acts(self):
        torch.manual_seed(0)
        b, s, d = 40, 6, 12
        y = torch.cat([torch.ones(b // 2), torch.zeros(b // 2)])

        x = torch.randn(b, s, d) * 0.2
        x[: b // 2, :, 0] += 1.5
        x[b // 2 :, :, 0] -= 1.5
        mask = torch.ones(b, s)

        acts = Acts(x, dims="bsh", seq_mask=mask)
        probe = Logistic(max_iter=20, device="cpu", stream_batch_size=16).fit(acts, y)
        probs = probe.predict(acts)

        self.assertEqual(tuple(probs.shape), (b, s))
        self.assertGreater(probs[: b // 2].mean().item(), probs[b // 2 :].mean().item())


if __name__ == "__main__":
    unittest.main()
