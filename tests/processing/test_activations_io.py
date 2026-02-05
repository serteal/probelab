"""Tests for Activations save/load (HDF5 storage)."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab.processing.activations import Activations, Axis


# Check if h5py is available
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def create_test_activations(
    n_layers: int = 3,
    batch_size: int = 10,
    seq_len: int = 20,
    d_model: int = 32,
    layer_indices: list[int] | None = None,
) -> Activations:
    """Create test activations with all metadata."""
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    acts = torch.randn(n_layers, batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len)
    # Make some tokens padded
    attention_mask[:, -5:] = 0
    detection_mask = torch.ones(batch_size, seq_len)
    detection_mask[:, :3] = 0  # Skip first 3 tokens
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    return Activations.from_tensor(
        activations=acts,
        attention_mask=attention_mask,
        input_ids=input_ids,
        detection_mask=detection_mask,
        layer_indices=layer_indices,
    )


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestActivationsSave:
    """Test Activations.save() method."""

    def test_save_creates_file(self):
        """Test that save creates an HDF5 file."""
        acts = create_test_activations()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            acts.save(str(path))

            assert path.exists()

    def test_save_with_compression(self):
        """Test saving with different compression settings."""
        acts = create_test_activations()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test gzip compression
            path_gzip = Path(tmpdir) / "acts_gzip.h5"
            acts.save(str(path_gzip), compression="gzip", compression_opts=9)
            assert path_gzip.exists()

            # Test lzf compression
            path_lzf = Path(tmpdir) / "acts_lzf.h5"
            acts.save(str(path_lzf), compression="lzf")
            assert path_lzf.exists()

            # Test no compression
            path_none = Path(tmpdir) / "acts_none.h5"
            acts.save(str(path_none), compression=None)
            assert path_none.exists()

            # All files should be valid HDF5
            for path in [path_gzip, path_lzf, path_none]:
                loaded = Activations.load(str(path))
                assert torch.allclose(loaded.activations, acts.activations)

    def test_save_stores_all_components(self):
        """Test that all components are stored in the file."""
        acts = create_test_activations(layer_indices=[5, 10, 15])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            acts.save(str(path))

            with h5py.File(path, "r") as f:
                # Check all datasets exist
                assert "activations" in f
                assert "axes" in f
                assert "layer_indices" in f
                assert "attention_mask" in f
                assert "detection_mask" in f
                assert "input_ids" in f

                # Check metadata
                assert "probelab_version" in f.attrs

                # Check layer indices
                assert list(f["layer_indices"][:]) == [5, 10, 15]


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestActivationsLoad:
    """Test Activations.load() method."""

    def test_load_roundtrip(self):
        """Test that save/load preserves all data."""
        original = create_test_activations(
            n_layers=3, batch_size=8, seq_len=16, d_model=32, layer_indices=[4, 8, 12]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            loaded = Activations.load(str(path))

            # Check activations match
            assert torch.allclose(loaded.activations, original.activations)

            # Check metadata matches
            assert loaded.axes == original.axes
            assert loaded.layer_indices == original.layer_indices

            # Check sequence metadata
            assert torch.allclose(loaded.attention_mask, original.attention_mask)
            assert torch.allclose(loaded.detection_mask, original.detection_mask)
            assert torch.equal(loaded.input_ids, original.input_ids)

    def test_load_specific_layers(self):
        """Test loading only specific layers."""
        original = create_test_activations(
            n_layers=5, batch_size=4, layer_indices=[2, 4, 6, 8, 10]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            # Load only layers 4 and 8
            loaded = Activations.load(str(path), layers=[4, 8])

            assert loaded.n_layers == 2
            assert loaded.layer_indices == [4, 8]

            # Check data matches for loaded layers
            orig_idx_4 = original.layer_indices.index(4)
            orig_idx_8 = original.layer_indices.index(8)
            assert torch.allclose(
                loaded.activations[0], original.activations[orig_idx_4]
            )
            assert torch.allclose(
                loaded.activations[1], original.activations[orig_idx_8]
            )

    def test_load_batch_slice(self):
        """Test loading a slice of the batch dimension."""
        original = create_test_activations(n_layers=2, batch_size=20, layer_indices=[0, 1])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            # Load first 5 samples
            loaded = Activations.load(str(path), batch_slice=slice(0, 5))

            assert loaded.batch_size == 5
            assert torch.allclose(loaded.activations, original.activations[:, :5])

            # Check sequence metadata is sliced too
            assert loaded.attention_mask.shape[0] == 5
            assert loaded.detection_mask.shape[0] == 5
            assert loaded.input_ids.shape[0] == 5

    def test_load_layers_and_batch_slice(self):
        """Test loading specific layers and batch slice together."""
        original = create_test_activations(
            n_layers=4, batch_size=20, layer_indices=[4, 8, 12, 16]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            # Load layer 8 and 16, first 10 samples
            loaded = Activations.load(str(path), layers=[8, 16], batch_slice=slice(0, 10))

            assert loaded.n_layers == 2
            assert loaded.batch_size == 10
            assert loaded.layer_indices == [8, 16]

    def test_load_to_device(self):
        """Test loading to specific device."""
        original = create_test_activations()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            loaded = Activations.load(str(path), device="cpu")
            assert loaded.activations.device == torch.device("cpu")

    def test_load_invalid_layer_raises(self):
        """Test that loading non-existent layer raises error."""
        original = create_test_activations(layer_indices=[0, 1, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            with pytest.raises(ValueError, match="Layer 99 not found"):
                Activations.load(str(path), layers=[99])


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestActivationsIOEdgeCases:
    """Test edge cases for save/load."""

    def test_save_load_without_layer_axis(self):
        """Test save/load for activations without LAYER axis."""
        # Create activations and remove layer axis
        original = create_test_activations(n_layers=1, layer_indices=[5])
        pooled = original.select(layer=5)  # Removes LAYER axis

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            pooled.save(str(path))

            loaded = Activations.load(str(path))

            assert Axis.LAYER not in loaded.axes
            assert torch.allclose(loaded.activations, pooled.activations)

    def test_save_load_without_seq_axis(self):
        """Test save/load for activations without SEQ axis (pooled)."""
        original = create_test_activations(n_layers=2, layer_indices=[0, 1])
        pooled = original.pool(dim="sequence", method="mean")  # Removes SEQ axis

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            pooled.save(str(path))

            loaded = Activations.load(str(path))

            assert Axis.SEQ not in loaded.axes
            assert loaded.sequence_meta is None
            assert torch.allclose(loaded.activations, pooled.activations)

    def test_save_load_with_batch_indices(self):
        """Test that batch_indices are preserved."""
        original = create_test_activations(n_layers=2, batch_size=10, layer_indices=[0, 1])
        # Manually set batch indices
        original.batch_indices = torch.tensor([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            loaded = Activations.load(str(path))

            assert loaded.batch_indices is not None
            assert torch.equal(loaded.batch_indices, original.batch_indices)

    def test_import_error_without_h5py(self, monkeypatch):
        """Test that helpful error is raised when h5py not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "h5py":
                raise ImportError("No module named 'h5py'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        acts = create_test_activations()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"

            with pytest.raises(ImportError, match="h5py required"):
                acts.save(str(path))


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestActivationsIOCompression:
    """Test compression efficiency."""

    def test_layer_chunking_enables_efficient_partial_loads(self):
        """Test that chunking by layer allows efficient partial reads."""
        # Create large activations
        original = create_test_activations(
            n_layers=10, batch_size=100, seq_len=128, d_model=256, layer_indices=list(range(10))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "activations.h5"
            original.save(str(path))

            # Loading single layer should be fast (only reads that layer's chunk)
            loaded_single = Activations.load(str(path), layers=[5])
            assert loaded_single.n_layers == 1
            assert loaded_single.layer_indices == [5]

            # Verify data is correct
            assert torch.allclose(
                loaded_single.activations[0], original.activations[5]
            )
