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

    def test_stream_hdf5_round_trips_chunks(self):
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not installed")

        data = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        detection_mask = torch.tensor([
            [True, False, True, False],
            [False, True, True, True],
            [True, True, False, True],
        ])
        acts = Activations.from_padded(
            data,
            detection_mask=detection_mask,
            attention_mask=torch.ones(3, 4, dtype=torch.bool),
            dims="bsh",
            metadata={"model": "demo", "split": "stream"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.h5"
            storage.save_hdf5(acts, str(path), dtype="float32")
            chunks = list(storage.stream_hdf5(str(path), chunk_tokens=5))

        streamed = Activations.cat([chunk for chunk, _ in chunks])
        indices = [idx for _, chunk_indices in chunks for idx in chunk_indices]
        self.assertEqual(indices, [0, 1, 2])
        torch.testing.assert_close(streamed.data, acts.data.float())
        self.assertTrue(torch.equal(streamed.offsets, acts.offsets))
        self.assertTrue(torch.equal(streamed.detection_mask, acts.detection_mask))
        self.assertEqual(streamed.metadata, acts.metadata)

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

    def test_stream_memmap_single_layer_round_trips_chunks(self):
        data = torch.arange(72, dtype=torch.float32).reshape(3, 2, 4, 3)
        detection_mask = torch.tensor([
            [True, False, True, True],
            [True, True, False, False],
            [False, True, True, True],
        ])
        acts = Activations.from_padded(
            data,
            detection_mask=detection_mask,
            attention_mask=torch.ones(3, 4, dtype=torch.bool),
            dims="blsh",
            layers=(4, 8),
            metadata={"model": "demo", "split": "stream"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.mm"
            storage.save_memmap(acts, str(path))
            chunks = list(storage.stream_memmap(str(path), layer=8, chunk_tokens=5))

        streamed = Activations.cat([chunk for chunk, _ in chunks])
        expected = acts.select("l", 8)
        indices = [idx for _, chunk_indices in chunks for idx in chunk_indices]
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(streamed.dims, "bsh")
        torch.testing.assert_close(streamed.data.float(), expected.data.bfloat16().float())
        self.assertTrue(torch.equal(streamed.offsets, expected.offsets))
        self.assertTrue(torch.equal(streamed.detection_mask, expected.detection_mask))
        self.assertEqual(streamed.metadata, expected.metadata)

    def test_load_memmap_rejects_unknown_layer(self):
        acts = Activations.from_padded(
            torch.randn(2, 2, 3, 4),
            detection_mask=torch.ones(2, 3, dtype=torch.bool),
            dims="blsh",
            layers=(4, 8),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acts.mm"
            storage.save_memmap(acts, str(path))
            with self.assertRaisesRegex(ValueError, "Layer 99 not in stored layers"):
                storage.load_memmap(str(path), layer=99)

    def test_save_memmap_requires_sequence_and_layer_axes(self):
        acts = Activations.from_tensor(torch.randn(2, 3), dims="bh")

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "requires dims with 's' and 'l'"):
                storage.save_memmap(acts, str(Path(tmp) / "acts.mm"))

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
