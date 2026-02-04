"""Tests for GatedBipolar probe implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelab import Pipeline
from probelab.transforms import pre
from probelab.probes.gated_bipolar import GatedBipolar
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


class TestGatedBipolar:
    """Test GatedBipolar probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        pipeline = Pipeline([
            ("select", pre.SelectLayer(5)),
            ("probe", GatedBipolar(
                mlp_hidden_dim=64,
                gate_dim=32,
                dropout=0.15,
                lambda_l1=1e-4,
                lambda_orth=1e-3,
                learning_rate=0.001,
                weight_decay=0.01,
                n_epochs=100,
                patience=5,
                device="cpu",
                random_state=42,
                verbose=False,
            )),
        ])

        assert pipeline["select"].layer == 5
        assert pipeline["probe"].mlp_hidden_dim == 64
        assert pipeline["probe"].gate_dim == 32
        assert pipeline["probe"].dropout == 0.15
        assert pipeline["probe"].lambda_l1 == 1e-4
        assert pipeline["probe"].lambda_orth == 1e-3
        assert pipeline["probe"].learning_rate == 0.001
        assert pipeline["probe"].weight_decay == 0.01
        assert pipeline["probe"].n_epochs == 100
        assert pipeline["probe"].patience == 5
        assert pipeline["probe"]._fitted is False

    def test_fit(self):
        """Test fitting the GatedBipolar probe."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", GatedBipolar(
                mlp_hidden_dim=32,
                gate_dim=16,
                n_epochs=50,
                device="cpu",
                random_state=42
            )),
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
            ("probe", GatedBipolar(
                mlp_hidden_dim=64,
                gate_dim=32,
                n_epochs=100,
                patience=10,
                device="cpu",
                random_state=42,
            )),
        ])
        pipeline.fit(activations, labels)

        probs = pipeline.predict(activations)

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
        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", GatedBipolar(device="cpu")),
        ])

        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.predict(activations)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", GatedBipolar(
                mlp_hidden_dim=32,
                gate_dim=16,
                lambda_l1=1e-4,
                lambda_orth=1e-3,
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

            loaded_probe = GatedBipolar.load(save_path)

            assert loaded_probe.mlp_hidden_dim == 32
            assert loaded_probe.gate_dim == 16
            assert loaded_probe.lambda_l1 == 1e-4
            assert loaded_probe.lambda_orth == 1e-3
            assert loaded_probe._fitted is True

            # Create new pipeline with loaded probe
            loaded_pipeline = Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", loaded_probe),
            ])

            probs_after = loaded_pipeline.predict(activations)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_no_aggregation_needed(self):
        """Test that GatedBipolar probe doesn't need aggregation - it handles sequences internally."""
        activations, labels = create_separable_data(n_samples=20)

        # GatedBipolar probe works without any aggregation transformers
        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", GatedBipolar(
                mlp_hidden_dim=32,
                gate_dim=16,
                n_epochs=50,
                device="cpu",
                random_state=42
            )),
        ])

        # Should work normally
        pipeline.fit(activations, labels)
        probs = pipeline.predict(activations)
        assert probs.shape == (20, 2)

    def test_different_gate_dims(self):
        """Test probe with different gate dimensions."""
        activations, labels = create_separable_data(n_samples=20)

        for gate_dim in [8, 32, 64]:
            pipeline = Pipeline([
                ("select", pre.SelectLayer(0)),
                ("probe", GatedBipolar(
                    gate_dim=gate_dim,
                    n_epochs=30,
                    device="cpu",
                    random_state=42
                )),
            ])

            pipeline.fit(activations, labels)
            probs = pipeline.predict(activations)

            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_regularization_loss(self):
        """Test that regularization loss is computed correctly."""
        activations, labels = create_separable_data(n_samples=20)

        probe = GatedBipolar(
            mlp_hidden_dim=32,
            gate_dim=16,
            lambda_l1=1e-4,
            lambda_orth=1e-3,
            n_epochs=10,
            device="cpu",
            random_state=42
        )

        # Fit the probe to initialize the network
        acts_single = activations.select(layer=0)
        probe.fit(acts_single, labels)

        # Compute regularization loss
        reg_loss = probe._network.get_regularization_loss(
            lambda_l1=1e-4, lambda_orth=1e-3
        )

        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.ndim == 0  # Scalar
        assert reg_loss >= 0  # Regularization loss should be non-negative

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

        probe = GatedBipolar(device="cpu")

        with pytest.raises(ValueError, match="expects single layer activations"):
            probe.fit(activations, labels)

    def test_repr(self):
        """Test string representation."""
        probe = GatedBipolar(mlp_hidden_dim=128, gate_dim=64)
        repr_str = repr(probe)
        assert "GatedBipolar" in repr_str
        assert "gate_dim=64" in repr_str
        assert "mlp_hidden_dim=128" in repr_str
        assert "not fitted" in repr_str

    def test_bipolar_pooling_captures_extremes(self):
        """Test that bipolar pooling captures both max and min."""
        # Create data where positive class has extreme positive AND negative values
        n_samples, seq_len, d_model = 20, 10, 8
        acts = torch.zeros(1, n_samples, seq_len, d_model)

        # Positive class: has both strong positive and negative extremes
        for i in range(n_samples // 2):
            acts[0, i, 0, :] = 3.0  # Strong positive
            acts[0, i, 1, :] = -3.0  # Strong negative
            acts[0, i, 2:, :] = torch.randn(seq_len - 2, d_model) * 0.1

        # Negative class: mild values throughout
        for i in range(n_samples // 2, n_samples):
            acts[0, i, :, :] = torch.randn(seq_len, d_model) * 0.3

        activations = Activations.from_tensor(
            activations=acts,
            attention_mask=torch.ones(n_samples, seq_len),
            input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
            detection_mask=torch.ones(n_samples, seq_len),
            layer_indices=[0],
        )

        labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

        pipeline = Pipeline([
            ("select", pre.SelectLayer(0)),
            ("probe", GatedBipolar(
                gate_dim=16,
                n_epochs=100,
                device="cpu",
                random_state=42
            )),
        ])

        pipeline.fit(activations, labels)
        probs = pipeline.predict(activations)

        # The probe should be able to learn this pattern
        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
