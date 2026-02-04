"""Tests for MultiMax probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab.probes.multimax import MultiMax
from probelab.processing.activations import Activations
from probelab.types import Label


def create_test_activations(n_samples=10, seq_len=20, d_model=16, layer=0):
    """Create test activations."""
    acts = torch.randn(1, n_samples, seq_len, d_model)
    detection_mask = torch.ones(n_samples, seq_len)

    return Activations.from_tensor(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )


def create_separable_data(n_samples=20, seq_len=10, d_model=8, layer=0):
    """Create linearly separable data for testing."""
    acts = torch.zeros(1, n_samples, seq_len, d_model)

    # First half: positive class - specific pattern in sequences
    for i in range(n_samples // 2):
        # Strong signal in early tokens
        acts[0, i, :3, 0] = 2.0
        acts[0, i, :3, 1] = 1.5
        # Noise elsewhere
        acts[0, i, 3:, :] = torch.randn(seq_len - 3, d_model) * 0.1

    # Second half: negative class - different pattern
    for i in range(n_samples // 2, n_samples):
        # Strong signal in late tokens
        acts[0, i, -3:, 0] = -2.0
        acts[0, i, -3:, 1] = -1.5
        # Noise elsewhere
        acts[0, i, :-3, :] = torch.randn(seq_len - 3, d_model) * 0.1

    activations = Activations.from_tensor(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=torch.ones(n_samples, seq_len),
        layer_indices=[layer],
    )

    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestMultiMax:
    """Test MultiMax probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = MultiMax(
            n_heads=8,
            mlp_hidden_dim=64,
            learning_rate=0.001,
            weight_decay=0.01,
            n_epochs=100,
            patience=5,
            device="cpu",
        )

        assert probe.n_heads == 8
        assert probe.mlp_hidden_dim == 64
        assert probe.learning_rate == 0.001
        assert probe.weight_decay == 0.01
        assert probe.n_epochs == 100
        assert probe.patience == 5
        assert probe._fitted is False

    def test_fit(self):
        """Test fitting the MultiMax probe."""
        activations, labels = create_separable_data(n_samples=20)

        # MultiMax handles sequences internally - just select layer
        prepared = activations.select(layer=0)
        probe = MultiMax(
            n_heads=4,
            mlp_hidden_dim=32,
            n_epochs=50,
            device="cpu",
        )

        fitted_probe = probe.fit(prepared, labels)

        assert fitted_probe is probe
        assert probe._fitted is True
        assert probe._network is not None
        assert probe._d_model == 8

    def test_predict(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        prepared = activations.select(layer=0)
        probe = MultiMax(
            n_heads=8,
            mlp_hidden_dim=64,
            n_epochs=100,
            patience=10,
            device="cpu",
        )
        probe.fit(prepared, labels)

        scores = probe.predict(prepared)
        probs = scores.scores

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20))

        # Just check that the probe is working
        predictions = probs.argmax(dim=1)
        assert len(predictions) == 20
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        prepared = activations.select(layer=0)
        probe = MultiMax(device="cpu")

        with pytest.raises(RuntimeError, match="must be fit before predict"):
            probe.predict(prepared)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        prepared = activations.select(layer=0)
        probe = MultiMax(
            n_heads=4,
            mlp_hidden_dim=32,
            learning_rate=0.001,
            n_epochs=50,
            device="cpu",
        )
        probe.fit(prepared, labels)

        scores_before = probe.predict(prepared)
        probs_before = scores_before.scores

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            probe.save(save_path)

            loaded_probe = MultiMax.load(save_path)

            assert loaded_probe.n_heads == 4
            assert loaded_probe.mlp_hidden_dim == 32
            assert loaded_probe._fitted is True

            scores_after = loaded_probe.predict(prepared)
            probs_after = scores_after.scores
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_no_aggregation_needed(self):
        """Test that MultiMax probe doesn't need aggregation - it handles sequences internally."""
        activations, labels = create_separable_data(n_samples=20)

        # MultiMax probe works without sequence pooling - just select layer
        prepared = activations.select(layer=0)
        probe = MultiMax(
            n_heads=4,
            mlp_hidden_dim=32,
            n_epochs=50,
            device="cpu",
        )

        # Should work normally
        probe.fit(prepared, labels)
        scores = probe.predict(prepared)
        probs = scores.scores
        assert probs.shape == (20, 2)

    def test_different_n_heads(self):
        """Test probe with different number of heads."""
        activations, labels = create_separable_data(n_samples=20)

        for n_heads in [2, 5, 10]:
            prepared = activations.select(layer=0)
            probe = MultiMax(
                n_heads=n_heads,
                n_epochs=30,
                device="cpu",
            )

            probe.fit(prepared, labels)
            scores = probe.predict(prepared)
            probs = scores.scores

            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_raises_on_layer_axis_present(self):
        """Test that probe raises error if LAYER axis is present."""
        # Create activations with LAYER axis
        acts = torch.randn(2, 10, 5, 8)  # [layers, batch, seq, hidden]
        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(10, 5),
            input_ids=torch.ones(10, 5, dtype=torch.long),
            detection_mask=torch.ones(10, 5),
            layer_indices=[0, 1],
        )

        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5

        probe = MultiMax(device="cpu")

        with pytest.raises(ValueError, match="expects no LAYER axis"):
            probe.fit(activations, labels)

    def test_repr(self):
        """Test string representation."""
        probe = MultiMax(n_heads=8, mlp_hidden_dim=64)
        repr_str = repr(probe)
        assert "MultiMax" in repr_str
        assert "n_heads=8" in repr_str
        assert "fitted=False" in repr_str
