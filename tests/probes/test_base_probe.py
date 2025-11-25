"""Tests for BaseProbe abstract class."""

import pytest
import torch

from probelib.probes.base import BaseProbe
from probelib.processing.activations import Activations
from probelib.types import Label


class ConcreteProbe(BaseProbe):
    """Concrete implementation of BaseProbe for testing."""

    def fit(self, X, y):
        """Mock fit implementation."""
        self._fitted = True
        return self

    def predict_proba(self, X):
        """Mock predict_proba implementation."""
        if not self._fitted:
            raise RuntimeError("Probe not fitted")
        # Return dummy probabilities
        if hasattr(X, "batch_size"):
            n_samples = X.batch_size
        else:
            n_samples = 10
        probs = torch.ones(n_samples, 2) * 0.5
        return probs

    def save(self, path):
        """Mock save implementation."""
        pass

    @classmethod
    def load(cls, path, device=None):
        """Mock load implementation."""
        return cls()


class TestBaseProbe:
    """Test BaseProbe abstract class and common functionality."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = ConcreteProbe(
            device="cpu",
            random_state=42,
            verbose=False,
        )

        # Note: layer and sequence_pooling are no longer probe parameters
        # They're handled by Pipeline transformers
        assert probe.device == "cpu"
        assert probe.random_state == 42
        assert probe.verbose is False
        assert probe._fitted is False

    def test_initialization_auto_device(self):
        """Test automatic device detection."""
        probe = ConcreteProbe()
        # Should default to cuda if available, else cpu
        assert probe.device in ["cuda", "cpu"]

    def test_to_tensor_from_list(self):
        """Test label conversion from list."""
        probe = ConcreteProbe()

        # Test with Label enum
        labels = [Label.POSITIVE, Label.NEGATIVE, Label.POSITIVE]
        y = probe._to_tensor(labels)
        assert y.tolist() == [1, 0, 1]

        # Test with raw values
        labels = [0, 1, 1, 0]
        y = probe._to_tensor(labels)
        assert y.tolist() == [0, 1, 1, 0]

    def test_to_tensor_from_tensor(self):
        """Test label conversion from tensor."""
        probe = ConcreteProbe(device="cpu")

        labels = torch.tensor([0, 1, 1, 0])
        y = probe._to_tensor(labels)
        assert torch.equal(y, labels)

    def test_to_tensor_invalid_classes(self):
        """Test that non-binary labels raise error."""
        probe = ConcreteProbe()

        # Test with invalid class values
        labels = [0, 1, 2]
        with pytest.raises(ValueError, match="Only binary classification is supported"):
            probe._to_tensor(labels)

        labels = [-1, 0, 1]
        with pytest.raises(ValueError, match="Only binary classification is supported"):
            probe._to_tensor(labels)

    def test_abstract_methods_required(self):
        """Test that abstract methods must be implemented."""

        class IncompleteProbe(BaseProbe):
            """Probe missing required methods."""

            pass

        # Should not be able to instantiate
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteProbe()

    def test_fit_sets_fitted_flag(self):
        """Test that fitting sets the _fitted flag."""
        probe = ConcreteProbe()
        assert not probe._fitted

        # Create simple activations
        acts = torch.randn(4, 16)  # 4 batch, 16 dim
        from probelib.processing.activations import Axis

        activations = Activations(
            activations=acts,
            axes=(Axis.BATCH, Axis.HIDDEN),
            layer_meta=None,
            sequence_meta=None,
            batch_indices=None,
        )

        labels = [Label.POSITIVE, Label.POSITIVE, Label.NEGATIVE, Label.NEGATIVE]
        probe.fit(activations, labels)
        assert probe._fitted

    def test_predict_before_fit_raises(self):
        """Test that predict fails before fitting."""
        probe = ConcreteProbe()

        acts = torch.randn(4, 16)
        from probelib.processing.activations import Axis

        activations = Activations(
            activations=acts,
            axes=(Axis.BATCH, Axis.HIDDEN),
            layer_meta=None,
            sequence_meta=None,
            batch_indices=None,
        )

        with pytest.raises(RuntimeError, match="Probe not fitted"):
            probe.predict_proba(activations)
