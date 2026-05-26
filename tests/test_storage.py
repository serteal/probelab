"""Tests for explicit activation storage helpers."""

import tempfile
import unittest
from pathlib import Path

import torch

from probelab import storage
from probelab.activations import Activations


class TestHDF5Storage(unittest.TestCase):
    def test_round_trip_hdf5(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not installed")

        acts = Activations.from_padded(
            torch.randn(2, 3, 4),
            detection_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
            dims="bsh",
            metadata={"model": "demo", "layers": [1], "split": "train"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.h5"
            storage.save_hdf5(acts, str(path), dtype="float32")
            loaded = storage.load_hdf5(str(path))

        self.assertEqual(loaded.dims, "bsh")
        torch.testing.assert_close(loaded.data, acts.data.float())
        self.assertTrue(torch.equal(loaded.offsets, acts.offsets.cpu()))
        self.assertTrue(torch.equal(loaded.detection_mask, acts.detection_mask.cpu()))
        self.assertEqual(loaded.metadata, acts.metadata)

    def test_hdf5_rejects_non_json_metadata(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not installed")

        acts = Activations.from_tensor(
            torch.randn(2, 3),
            dims="bh",
            metadata={"bad": object()},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.h5"
            with self.assertRaisesRegex(TypeError, "JSON-serializable"):
                storage.save_hdf5(acts, str(path))


class TestMemmapStorage(unittest.TestCase):
    def test_round_trip_memmap_all_layers(self):
        acts = Activations.from_padded(
            torch.randn(2, 2, 3, 4),
            detection_mask=torch.ones(2, 3, dtype=torch.bool),
            dims="blsh",
            layers=(4, 8),
            metadata={"model": "demo", "layers": [4, 8]},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.mm"
            storage.save_memmap(acts, str(path))
            self.assertTrue(storage.has_memmap(str(path)))
            loaded = storage.load_memmap(str(path))

        self.assertEqual(loaded.dims, "blsh")
        self.assertEqual(loaded.layers, (4, 8))
        torch.testing.assert_close(loaded.data.float(), acts.data.bfloat16().float())
        self.assertTrue(torch.equal(loaded.offsets, acts.offsets.cpu()))
        self.assertTrue(torch.equal(loaded.detection_mask, acts.detection_mask.cpu()))
        self.assertEqual(loaded.metadata, acts.metadata)

    def test_round_trip_memmap_single_layer(self):
        acts = Activations.from_padded(
            torch.randn(2, 2, 3, 4),
            detection_mask=torch.ones(2, 3, dtype=torch.bool),
            dims="blsh",
            layers=(4, 8),
            metadata={"model": "demo", "layers": [4, 8]},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.mm"
            storage.save_memmap(acts, str(path))
            loaded = storage.load_memmap(str(path), layer=8)

        self.assertEqual(loaded.dims, "bsh")
        torch.testing.assert_close(
            loaded.data.float(),
            acts.select("l", 8).data.bfloat16().float(),
        )
        self.assertEqual(loaded.metadata, acts.metadata)

    def test_memmap_rejects_non_json_metadata(self):
        acts = Activations.from_padded(
            torch.randn(2, 2, 3, 4),
            detection_mask=torch.ones(2, 3, dtype=torch.bool),
            dims="blsh",
            layers=(4, 8),
            metadata={"bad": object()},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.mm"
            with self.assertRaisesRegex(TypeError, "JSON-serializable"):
                storage.save_memmap(acts, str(path))


if __name__ == "__main__":
    unittest.main()
