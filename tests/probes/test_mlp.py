"""Tests for MLP probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab.probes.mlp import MLP
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

    # First half: positive class
    for i in range(n_samples // 2):
        acts[0, i, :, :4] = 1.0
        acts[0, i, :, 4:] = 0.0

    # Second half: negative class
    for i in range(n_samples // 2, n_samples):
        acts[0, i, :, :4] = 0.0
        acts[0, i, :, 4:] = 1.0

    # Add small noise
    acts += torch.randn_like(acts) * 0.1

    activations = Activations.from_tensor(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=torch.ones(n_samples, seq_len),
        layer_indices=[layer],
    )

    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestMLP:
    """Test MLP probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = MLP(
            hidden_dim=256,
            dropout=0.2,
            activation="gelu",
            learning_rate=0.002,
            weight_decay=0.05,
            n_epochs=200,
            device="cpu",
        )

        assert probe.hidden_dim == 256
        assert probe.dropout == 0.2
        assert probe.activation == "gelu"
        assert probe.learning_rate == 0.002
        assert probe.weight_decay == 0.05
        assert probe.n_epochs == 200
        assert probe._fitted is False
        assert probe._network is None

    def test_fit(self):
        """Test fitting the MLP probe."""
        activations, labels = create_separable_data(n_samples=20)

        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = MLP(hidden_dim=16, n_epochs=50, device="cpu")

        fitted_probe = probe.fit(prepared, labels)

        assert fitted_probe is probe
        assert probe._fitted is True
        assert probe._network is not None
        assert probe._d_model == 8

    def test_predict(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = MLP(hidden_dim=32, n_epochs=100, device="cpu")
        probe.fit(prepared, labels)

        scores = probe.predict(prepared)
        probs = scores.scores

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20), atol=1e-6)

        # Check that it learned something (should get at least 50% accuracy on this simple data)
        predictions = probs.argmax(dim=1)
        true_labels = torch.tensor([label.value for label in labels])
        accuracy = (predictions == true_labels).float().mean()
        assert accuracy >= 0.5

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = MLP(device="cpu")

        with pytest.raises(RuntimeError, match="must be fit before predict"):
            probe.predict(prepared)

    def test_token_level_training(self):
        """Test training on token-level activations with score aggregation."""
        activations, labels = create_separable_data(n_samples=20)

        # Select layer but don't pool - trains on token level
        prepared = activations.select(layer=0)
        probe = MLP(hidden_dim=16, n_epochs=50, device="cpu")
        probe.fit(prepared, labels)

        # Predict returns token-level scores, pool them to sequence level
        scores = probe.predict(prepared)
        pooled_scores = scores.pool(dim="sequence", method="mean")
        probs = pooled_scores.scores
        assert probs.shape == (20, 2)

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        activations, labels = create_separable_data(n_samples=20)

        for method in ["mean", "max", "last_token"]:
            prepared = activations.select(layer=0).pool("sequence", method)
            probe = MLP(hidden_dim=16, n_epochs=50, device="cpu")
            probe.fit(prepared, labels)

            scores = probe.predict(prepared)
            probs = scores.scores
            assert probs.shape == (20, 2)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        prepared = activations.select(layer=0).pool("sequence", "mean")
        probe = MLP(
            hidden_dim=32,
            dropout=0.1,
            activation="gelu",
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

            loaded_probe = MLP.load(save_path)

            assert loaded_probe.hidden_dim == 32
            assert loaded_probe._fitted is True

            scores_after = loaded_probe.predict(prepared)
            probs_after = scores_after.scores
            probs_before = probs_before.to(probs_after.device)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_different_activations(self):
        """Test different activation functions."""
        activations, labels = create_separable_data(n_samples=20)

        for activation in ["relu", "gelu"]:
            prepared = activations.select(layer=0).pool("sequence", "mean")
            probe = MLP(
                hidden_dim=16,
                activation=activation,
                n_epochs=50,
                device="cpu",
            )
            probe.fit(prepared, labels)

            scores = probe.predict(prepared)
            probs = scores.scores
            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
