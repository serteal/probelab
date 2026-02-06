"""Tests for probe implementations."""
import tempfile
import unittest
from pathlib import Path

import torch

from probelab.processing.activations import Activations
from probelab.probes.logistic import Logistic
from probelab.probes.mlp import MLP
from probelab.types import Label

# =============================================================================
# Test Helpers
# =============================================================================

def _separable_acts(n_samples=20, seq=8, d_model=8, gap=2.0):
  """Create linearly separable activations."""
  half = n_samples // 2
  t = torch.zeros(1, n_samples, seq, d_model)
  t[0, :half, :, 0] = gap
  t[0, half:, :, 0] = -gap
  t[:, :, :, 1:] = torch.randn(1, n_samples, seq, d_model - 1) * 0.1
  return Activations.from_tensor(
    t, attention_mask=torch.ones(n_samples, seq),
    input_ids=torch.ones(n_samples, seq, dtype=torch.long),
    detection_mask=torch.ones(n_samples, seq), layer_indices=[0]
  ), [Label.POSITIVE] * half + [Label.NEGATIVE] * half

def _acts(n_samples=10, seq=8, d_model=16):
  """Create random activations."""
  t = torch.randn(1, n_samples, seq, d_model)
  return Activations.from_tensor(
    t, attention_mask=torch.ones(n_samples, seq),
    input_ids=torch.ones(n_samples, seq, dtype=torch.long),
    detection_mask=torch.ones(n_samples, seq), layer_indices=[0]
  )

# =============================================================================
# Logistic Probe Tests
# =============================================================================

class TestLogisticInit(unittest.TestCase):
  def test_default_params(self):
    p = Logistic()
    self.assertEqual(p.C, 1.0)
    self.assertEqual(p.max_iter, 100)
    self.assertFalse(p._fitted)

  def test_custom_params(self):
    p = Logistic(C=10.0, max_iter=50, device="cpu")
    self.assertEqual(p.C, 10.0)
    self.assertEqual(p.max_iter, 50)
    self.assertEqual(p.device, "cpu")

class TestLogisticFit(unittest.TestCase):
  def test_fit_sequence_level(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertIsNotNone(p._network)

  def test_fit_token_level(self):
    acts, labels = _separable_acts(n_samples=10, seq=5)
    prepared = acts.select(layer=0)
    p = Logistic(device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertTrue(p._trained_on_tokens)

  def test_fit_returns_self(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu")
    result = p.fit(prepared, labels)
    self.assertIs(result, p)

class TestLogisticPredict(unittest.TestCase):
  def test_predict_returns_scores(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu").fit(prepared, labels)
    scores = p.predict(prepared)
    self.assertEqual(scores.shape, (20,))  # [batch] for sequence-level

  def test_predict_valid_probabilities(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu").fit(prepared, labels)
    probs = p.predict(prepared)
    self.assertTrue(torch.all(probs >= 0))
    self.assertTrue(torch.all(probs <= 1))

  def test_predict_separable_accuracy(self):
    acts, labels = _separable_acts(n_samples=20, gap=3.0)
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu").fit(prepared, labels)
    probs = p.predict(prepared)
    pos_correct = (probs[:10] > 0.5).sum()  # Positive class should have high prob
    neg_correct = (probs[10:] < 0.5).sum()  # Negative class should have low prob
    self.assertGreater(pos_correct + neg_correct, 14)  # >70% accuracy

  def test_predict_before_fit_raises(self):
    acts = _acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(device="cpu")
    with self.assertRaises(RuntimeError):
      p.predict(prepared)

class TestLogisticSaveLoad(unittest.TestCase):
  def test_save_load_roundtrip(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = Logistic(C=2.0, device="cpu").fit(prepared, labels)
    probs_before = p.predict(prepared)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "probe.pt"
      p.save(path)
      loaded = Logistic.load(path)

    self.assertEqual(loaded.C, 2.0)
    self.assertTrue(loaded._fitted)
    probs_after = loaded.predict(prepared)
    self.assertTrue(torch.allclose(probs_before, probs_after, atol=1e-5))

  def test_save_unfitted_raises(self):
    p = Logistic(device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "probe.pt"
      with self.assertRaises(RuntimeError):
        p.save(path)

# =============================================================================
# MLP Probe Tests
# =============================================================================

class TestMLPInit(unittest.TestCase):
  def test_default_params(self):
    p = MLP()
    self.assertEqual(p.hidden_dim, 128)
    self.assertEqual(p.n_epochs, 100)
    self.assertIsNone(p.dropout)
    self.assertFalse(p._fitted)

  def test_custom_params(self):
    p = MLP(hidden_dim=64, dropout=0.1, n_epochs=50, device="cpu")
    self.assertEqual(p.hidden_dim, 64)
    self.assertEqual(p.dropout, 0.1)
    self.assertEqual(p.n_epochs, 50)

class TestMLPFit(unittest.TestCase):
  def test_fit_sequence_level(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertIsNotNone(p._network)

  def test_fit_token_level(self):
    acts, labels = _separable_acts(n_samples=10, seq=5)
    prepared = acts.select(layer=0)
    p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertTrue(p._trained_on_tokens)

  def test_fit_with_dropout(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, dropout=0.2, n_epochs=10, device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)

  def test_fit_gelu_activation(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, activation="gelu", n_epochs=10, device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)

class TestMLPPredict(unittest.TestCase):
  def test_predict_returns_scores(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
    scores = p.predict(prepared)
    self.assertEqual(scores.shape, (20,))  # [batch] for sequence-level

  def test_predict_valid_probabilities(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
    probs = p.predict(prepared)
    self.assertTrue(torch.all(probs >= 0))
    self.assertTrue(torch.all(probs <= 1))

  def test_predict_before_fit_raises(self):
    acts = _acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(device="cpu")
    with self.assertRaises(RuntimeError):
      p.predict(prepared)

class TestMLPSaveLoad(unittest.TestCase):
  def test_save_load_roundtrip(self):
    acts, labels = _separable_acts()
    prepared = acts.select(layer=0).pool("sequence", "mean")
    p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
    probs_before = p.predict(prepared)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "probe.pt"
      p.save(path)
      loaded = MLP.load(path)

    self.assertEqual(loaded.hidden_dim, 32)
    self.assertTrue(loaded._fitted)
    probs_after = loaded.predict(prepared)
    self.assertTrue(torch.allclose(probs_before, probs_after, atol=1e-5))

  def test_save_unfitted_raises(self):
    p = MLP(device="cpu")
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "probe.pt"
      with self.assertRaises(RuntimeError):
        p.save(path)

# =============================================================================
# Common Probe Tests (Test all probes with same interface)
# =============================================================================

class TestProbeInterface(unittest.TestCase):
  """Test that all probes follow the same interface."""

  def _test_probe_interface(self, ProbeClass, **init_kwargs):
    acts, labels = _separable_acts(n_samples=20)
    prepared = acts.select(layer=0).pool("sequence", "mean")

    # Test fit returns self
    p = ProbeClass(device="cpu", **init_kwargs)
    result = p.fit(prepared, labels)
    self.assertIs(result, p)
    self.assertTrue(p._fitted)

    # Test predict returns tensor with batch dimension
    probs = p.predict(prepared)
    self.assertEqual(probs.shape[0], 20)  # batch size
    self.assertEqual(len(probs.shape), 1)  # [batch] for sequence-level

    # Test save/load
    with tempfile.TemporaryDirectory() as tmpdir:
      path = Path(tmpdir) / "probe.pt"
      p.save(path)
      loaded = ProbeClass.load(path, device="cpu")
      self.assertTrue(loaded._fitted)
      probs_loaded = loaded.predict(prepared)
      probs_orig = p.predict(prepared)
      self.assertTrue(torch.allclose(probs_orig, probs_loaded, atol=1e-5))

  def test_logistic_interface(self):
    self._test_probe_interface(Logistic)

  def test_mlp_interface(self):
    self._test_probe_interface(MLP, hidden_dim=32, n_epochs=10)

# =============================================================================
# Probe Error Handling
# =============================================================================

class TestProbeErrors(unittest.TestCase):
  def test_logistic_rejects_multi_layer(self):
    t = torch.randn(2, 10, 8, 16)  # 2 layers
    acts = Activations.from_tensor(
      t, attention_mask=torch.ones(10, 8),
      input_ids=torch.ones(10, 8, dtype=torch.long),
      detection_mask=torch.ones(10, 8), layer_indices=[0, 1]
    )
    labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5
    p = Logistic(device="cpu")
    with self.assertRaises(ValueError):
      p.fit(acts, labels)

  def test_mlp_rejects_multi_layer(self):
    t = torch.randn(2, 10, 8, 16)
    acts = Activations.from_tensor(
      t, attention_mask=torch.ones(10, 8),
      input_ids=torch.ones(10, 8, dtype=torch.long),
      detection_mask=torch.ones(10, 8), layer_indices=[0, 1]
    )
    labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5
    p = MLP(device="cpu")
    with self.assertRaises(ValueError):
      p.fit(acts, labels)

# =============================================================================
# Probe Edge Cases
# =============================================================================

class TestProbeEdgeCases(unittest.TestCase):
  def test_small_batch(self):
    t = torch.randn(1, 4, 8, 8)
    acts = Activations.from_tensor(
      t, attention_mask=torch.ones(4, 8),
      input_ids=torch.ones(4, 8, dtype=torch.long),
      detection_mask=torch.ones(4, 8), layer_indices=[0]
    )
    labels = [Label.POSITIVE, Label.POSITIVE, Label.NEGATIVE, Label.NEGATIVE]
    prepared = acts.select(layer=0).pool("sequence", "mean")

    p = Logistic(device="cpu").fit(prepared, labels)
    scores = p.predict(prepared)
    self.assertEqual(scores.shape, (4,))  # [batch]

  def test_integer_labels(self):
    acts, _ = _separable_acts(n_samples=10)
    prepared = acts.select(layer=0).pool("sequence", "mean")
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # int labels
    p = Logistic(device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)

  def test_tensor_labels(self):
    acts, _ = _separable_acts(n_samples=10)
    prepared = acts.select(layer=0).pool("sequence", "mean")
    labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = Logistic(device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)

class TestProbeDeviceHandling(unittest.TestCase):
  """Test device handling edge cases for probes."""

  def test_logistic_token_level_device_consistency(self):
    """Test that token-level training handles device correctly."""
    # Create activations (on CPU)
    acts, labels = _separable_acts(n_samples=10, seq=5)
    prepared = acts.select(layer=0)  # Keep SEQ axis for token-level

    # Train on CPU - should work without device mismatch
    p = Logistic(device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertTrue(p._trained_on_tokens)

    # Verify prediction works - returns [batch, seq] for token-level
    probs = p.predict(prepared)
    self.assertEqual(probs.shape, (10, 5))  # [batch, seq]

  def test_mlp_token_level_device_consistency(self):
    """Test that MLP token-level training handles device correctly."""
    acts, labels = _separable_acts(n_samples=10, seq=5)
    prepared = acts.select(layer=0)

    p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(prepared, labels)
    self.assertTrue(p._fitted)
    self.assertTrue(p._trained_on_tokens)

    # Returns [batch, seq] for token-level
    probs = p.predict(prepared)
    self.assertEqual(probs.shape, (10, 5))  # [batch, seq]

  def test_logistic_repeat_interleave_same_device(self):
    """Verify repeat_interleave uses consistent device for labels and tokens_per_sample."""
    # This test specifically checks the fix for device mismatch in repeat_interleave
    acts, labels = _separable_acts(n_samples=6, seq=4)
    prepared = acts.select(layer=0)

    p = Logistic(device="cpu")
    p.fit(prepared, labels)

    # If device handling is wrong, this would fail during fit
    self.assertTrue(p._fitted)

if __name__ == '__main__':
  unittest.main()
