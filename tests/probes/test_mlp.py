"""Tests for MLP probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab import Pipeline
from probelab.transforms import pre, post
from probelab.probes.mlp import MLP
from probelab.processing.activations import Activations
from probelab.types import Label, AggregationMethod


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
        pipeline = Pipeline([
            ("select", pre.SelectLayer(5)),
            ("agg", pre.Pool(dim="sequence", method="max")),
            ("probe", MLP(
                hidden_dim=256,
                dropout=0.2,
                activation="gelu",
                learning_rate=0.002,
                weight_decay=0.05,
                n_epochs=200,
                device="cpu",
                random_state=42,
                verbose=False,
            )),
        ])

        assert pipeline["select"].layer == 5
        assert pipeline["agg"].method == AggregationMethod.MAX
        assert pipeline["probe"].hidden_dim == 256
        assert pipeline["probe"].dropout == 0.2
        assert pipeline["probe"].activation == "gelu"
        assert pipeline["probe"].learning_rate == 0.002
        assert pipeline["probe"].weight_decay == 0.05
        assert pipeline["probe"].n_epochs == 200
        assert pipeline["probe"]._fitted is False
        assert pipeline["probe"]._network is None

    def test_fit(self):
        """Test fitting the MLP probe."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("agg", pre.Pool(dim="sequence", method="mean")),
            ("probe", MLP(hidden_dim=16, n_epochs=50, device="cpu", random_state=42)),
        ])

        fitted_pipeline = pipeline.fit(activations, labels)

        assert fitted_pipeline is pipeline
        assert pipeline["probe"]._fitted is True
        assert pipeline["probe"]._network is not None
        assert pipeline["probe"]._d_model == 8

    def test_predict(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("agg", pre.Pool(dim="sequence", method="mean")),
            ("probe", MLP(hidden_dim=32, n_epochs=100, device="cpu", random_state=42)),
        ])
        pipeline.fit(activations, labels)

        probs = pipeline.predict(activations)

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
        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("agg", pre.Pool(dim="sequence", method="mean")),
            ("probe", MLP(device="cpu")),
        ])

        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.predict(activations)

    def test_token_level_training(self):
        """Test training on token-level activations with score aggregation."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", MLP(hidden_dim=16, n_epochs=50, device="cpu", random_state=42)),
            ("agg_scores", post.Pool(method="mean")),
        ])

        pipeline.fit(activations, labels)

        # Should predict at sequence level
        probs = pipeline.predict(activations)
        assert probs.shape == (20, 2)

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        activations, labels = create_separable_data(n_samples=20)

        for method in [AggregationMethod.MEAN, AggregationMethod.MAX, AggregationMethod.LAST_TOKEN]:
            pipeline = Pipeline([
                ("select", pre.SelectLayer(0)),
                ("pool", pre.Pool(dim="sequence", method=method)),
                ("probe", MLP(hidden_dim=16, n_epochs=50, device="cpu", random_state=42)),
            ])
            pipeline.fit(activations, labels)

            probs = pipeline.predict(activations)
            assert probs.shape == (20, 2)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("agg", pre.Pool(dim="sequence", method="mean")),
            ("probe", MLP(
                hidden_dim=32,
                dropout=0.1,
                activation="gelu",
                learning_rate=0.001,
                n_epochs=50,
                device="cpu",
                random_state=42,
            )),
        ])
        pipeline.fit(activations, labels)

        probs_before = pipeline.predict(activations)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            pipeline["probe"].save(save_path)

            loaded_probe = MLP.load(save_path)

            assert loaded_probe.hidden_dim == 32
            assert loaded_probe._fitted is True

            # Create new pipeline with loaded probe
            loaded_pipeline = Pipeline([
                ("select", pre.SelectLayer(0)),
                ("agg", pre.Pool(dim="sequence", method="mean")),
                ("probe", loaded_probe),
            ])

            probs_after = loaded_pipeline.predict(activations)
            probs_before = probs_before.to(probs_after.device)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_different_activations(self):
        """Test different activation functions."""
        activations, labels = create_separable_data(n_samples=20)

        for activation in ["relu", "gelu"]:
            pipeline = Pipeline([
                ("select", pre.SelectLayer(0)),
                ("agg", pre.Pool(dim="sequence", method="mean")),
                ("probe", MLP(
                    hidden_dim=16,
                    activation=activation,
                    n_epochs=50,
                    device="cpu",
                    random_state=42,
                )),
            ])
            pipeline.fit(activations, labels)

            probs = pipeline.predict(activations)
            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)
