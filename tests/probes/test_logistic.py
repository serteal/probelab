"""Tests for Logistic implementation."""

import tempfile
from pathlib import Path

import pytest
import torch

from probelib import Pipeline
from probelib.preprocessing import SelectLayer, Pool
from probelib.probes.logistic import Logistic
from probelib.processing.activations import (
    ActivationIterator,
    Activations,
)
from probelib.types import Label, AggregationMethod


class MockActivationIterator(ActivationIterator):
    """Mock iterator for testing."""

    def __init__(self, batches):
        # Add batch_indices to each batch if not present
        self.batches = []
        start_idx = 0
        for batch in batches:
            if batch.batch_indices is None:
                # Create batch indices for this batch
                batch_size = batch.batch_size
                indices = torch.arange(start_idx, start_idx + batch_size, dtype=torch.long)
                # Create new batch with indices
                from probelib.processing.activations import Axis
                new_batch = Activations.from_tensor(
                    activations=batch.activations,
                    layer_indices=batch.layer_indices,
                    attention_mask=batch.attention_mask,
                    detection_mask=batch.detection_mask,
                    input_ids=batch.input_ids,
                    batch_indices=indices,
                )
                self.batches.append(new_batch)
                start_idx += batch_size
            else:
                self.batches.append(batch)
        self._layers = [0]  # Single layer

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    @property
    def layers(self):
        return self._layers

    # No longer need with_labels method - use zip instead


def create_test_activations(n_samples=10, seq_len=20, d_model=16, layer=0):
    """Create test activations with controlled properties."""
    # Create activations with some structure for testing
    acts = torch.randn(1, n_samples, seq_len, d_model)

    # Create detection mask with varying lengths
    detection_mask = torch.zeros(n_samples, seq_len)
    for i in range(n_samples):
        # Each sample has different number of valid tokens
        valid_len = 5 + i % 10
        detection_mask[i, :valid_len] = 1

    return Activations.from_tensor(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )


def create_separable_data(n_samples=20, seq_len=10, d_model=8, layer=0):
    """Create linearly separable data for testing."""
    # Create two clear clusters
    acts = torch.zeros(1, n_samples, seq_len, d_model)

    # First half: positive class
    acts[0, : n_samples // 2, :, 0] = 1.0  # High value in first dimension
    acts[0, : n_samples // 2, :, 1:] = (
        torch.randn(n_samples // 2, seq_len, d_model - 1) * 0.1
    )

    # Second half: negative class
    acts[0, n_samples // 2 :, :, 0] = -1.0  # Low value in first dimension
    acts[0, n_samples // 2 :, :, 1:] = (
        torch.randn(n_samples // 2, seq_len, d_model - 1) * 0.1
    )

    detection_mask = torch.ones(n_samples, seq_len)

    activations = Activations.from_tensor(
        activations=acts,
        attention_mask=torch.ones(n_samples, seq_len),
        input_ids=torch.ones(n_samples, seq_len, dtype=torch.long),
        detection_mask=detection_mask,
        layer_indices=[layer],
    )

    # Create corresponding labels
    labels = [Label.POSITIVE] * (n_samples // 2) + [Label.NEGATIVE] * (n_samples // 2)

    return activations, labels


class TestLogistic:
    """Test Logistic probe implementation."""

    def test_initialization(self):
        """Test probe initialization."""
        pipeline = Pipeline([
            ("select", SelectLayer(5)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(
                C=10.0,  # C is inverse of l2_penalty (C=10 ≈ l2_penalty=0.1)
                device="cpu",
                random_state=42,
                verbose=False,
            )),
        ])

        assert pipeline["select"].layer == 5
        assert pipeline["agg"].method == AggregationMethod.MEAN
        assert pipeline["probe"].C == 10.0
        assert pipeline["probe"].device == "cpu"
        assert pipeline["probe"].random_state == 42
        assert pipeline["probe"]._fitted is False

    def test_fit_with_aggregation(self):
        """Test fitting with sequence aggregation."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(device="cpu")),
        ])

        fitted_pipeline = pipeline.fit(activations, labels)

        assert fitted_pipeline is pipeline  # Should return self
        assert pipeline["probe"]._fitted is True
        assert pipeline["probe"]._network is not None
        assert pipeline["probe"]._network.linear.weight.shape == (1, 8)  # [1, d_model]

    def test_fit_token_level(self):
        """Test fitting at token level with score aggregation."""
        activations, labels = create_separable_data(n_samples=10, seq_len=5)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("probe", Logistic(device="cpu")),
            ("agg_scores", Pool(axis="sequence", method="mean")),
        ])

        pipeline.fit(activations, labels)

        assert pipeline["probe"]._fitted is True
        assert pipeline["probe"]._network is not None

    def test_predict_proba(self):
        """Test probability prediction."""
        activations, labels = create_separable_data(n_samples=20)

        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(device="cpu", random_state=42)),
        ])
        pipeline.fit(activations, labels)

        # Predict on same data
        probs = pipeline.predict_proba(activations)

        assert probs.shape == (20, 2)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(20))

        # Should predict high probability for correct classes
        pos_probs = probs[:10, 1]  # P(y=1) for positive samples
        neg_probs = probs[10:, 0]  # P(y=0) for negative samples

        assert pos_probs.mean() > 0.65  # Should be confident on positive
        assert neg_probs.mean() > 0.65  # Should be confident on negative

    def test_predict_before_fit(self):
        """Test that prediction fails before fitting."""
        activations = create_test_activations()
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(device="cpu")),
        ])

        with pytest.raises(ValueError, match="Pipeline must be fitted"):
            pipeline.predict_proba(activations)

    def test_different_aggregations(self):
        """Test different aggregation methods."""
        activations, labels = create_separable_data(n_samples=20)

        for method in [AggregationMethod.MEAN, AggregationMethod.MAX, AggregationMethod.LAST_TOKEN]:
            pipeline = Pipeline([
                ("select", SelectLayer(0)),
                ("pool", Pool(axis="sequence", method=method)),
                ("probe", Logistic(device="cpu", random_state=42)),
            ])

            pipeline.fit(activations, labels)
            probs = pipeline.predict_proba(activations)

            assert probs.shape == (20, 2)
            assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_save_and_load(self):
        """Test saving and loading probe."""
        activations, labels = create_separable_data(n_samples=20)

        # Train pipeline
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("agg", Pool(axis="sequence", method="max")),
            ("probe", Logistic(C=2.0, device="cpu", random_state=42)),
        ])
        pipeline.fit(activations, labels)

        # Get predictions before saving
        probs_before = pipeline.predict_proba(activations)

        # Save and load probe
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            pipeline["probe"].save(save_path)

            loaded_probe = Logistic.load(save_path)

            # Check attributes preserved
            assert loaded_probe.C == 2.0
            assert loaded_probe._fitted is True

            # Create new pipeline with loaded probe
            loaded_pipeline = Pipeline([
                ("select", SelectLayer(0)),
                ("agg", Pool(axis="sequence", method="max")),
                ("probe", loaded_probe),
            ])

            # Check predictions are the same
            probs_after = loaded_pipeline.predict_proba(activations)
            probs_before = probs_before.to(probs_after.device)
            assert torch.allclose(probs_before, probs_after, atol=1e-6)

    def test_save_unfitted_probe(self):
        """Test that saving unfitted probe raises error."""
        probe = Logistic(device="cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            with pytest.raises(RuntimeError, match="Cannot save unfitted probe"):
                probe.save(save_path)

    def test_load_with_different_device(self):
        """Test loading probe to different device."""
        activations, labels = create_separable_data(n_samples=20)

        # Train on CPU
        pipeline = Pipeline([
            ("select", SelectLayer(0)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(device="cpu")),
        ])
        pipeline.fit(activations, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "probe.pt"
            pipeline["probe"].save(save_path)

            # Load to different device (still CPU in tests)
            loaded_probe = Logistic.load(save_path, device="cpu")
            assert loaded_probe.device == "cpu"

    def test_validation_errors(self):
        """Test that appropriate errors are raised."""
        activations = create_test_activations(n_samples=10)

        # GPU version doesn't check for single class
        # (handled by PyTorch loss function)

        # Test with wrong layer - should error during transform
        pipeline = Pipeline([
            ("select", SelectLayer(5)),
            ("agg", Pool(axis="sequence", method="mean")),
            ("probe", Logistic(device="cpu")),
        ])
        with pytest.raises(ValueError, match="Layer 5 is not available"):
            pipeline.fit(activations, [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5)


