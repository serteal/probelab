"""Tests for probe implementations."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from probelab.processing.activations import Activations
from probelab.probes.logistic import Logistic
from probelab.probes.mlp import MLP
from probelab.probes.attention import Attention
from probelab.probes.multimax import MultiMax
from probelab.probes.gated_bipolar import GatedBipolar
from probelab.types import Label

# =============================================================================
# Test Helpers
# =============================================================================

def _separable_acts(n_samples=20, seq=8, d_model=8, gap=2.0):
    """Create linearly separable activations."""
    half = n_samples // 2
    # [batch, seq, hidden]
    t = torch.zeros(n_samples, seq, d_model)
    t[:half, :, 0] = gap
    t[half:, :, 0] = -gap
    t[:, :, 1:] = torch.randn(n_samples, seq, d_model - 1) * 0.1
    return Activations.from_padded(
        data=t, detection_mask=torch.ones(n_samples, seq), dims="bsh",
    ), [Label.POSITIVE] * half + [Label.NEGATIVE] * half

def _acts(n_samples=10, seq=8, d_model=16):
    """Create random activations."""
    t = torch.randn(n_samples, seq, d_model)
    return Activations.from_padded(
        data=t, detection_mask=torch.ones(n_samples, seq), dims="bsh",
    )

# =============================================================================
# Logistic Probe Tests
# =============================================================================

class TestLogisticInit(unittest.TestCase):
    def test_default_params(self):
        p = Logistic()
        self.assertEqual(p.C, 1.0)
        self.assertEqual(p.max_iter, 500)
        self.assertFalse(p.fitted)

    def test_custom_params(self):
        p = Logistic(C=10.0, max_iter=50, device="cpu")
        self.assertEqual(p.C, 10.0)
        self.assertEqual(p.max_iter, 50)
        self.assertEqual(p.device, "cpu")

class TestLogisticFit(unittest.TestCase):
    def test_fit_sequence_level(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)
        self.assertIsNotNone(p.net)

    def test_fit_token_level(self):
        acts, labels = _separable_acts(n_samples=10, seq=5)
        # Keep SEQ axis for token-level training
        p = Logistic(device="cpu").fit(acts, labels)
        self.assertTrue(p.fitted)

    def test_fit_returns_self(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu")
        result = p.fit(prepared, labels)
        self.assertIs(result, p)

class TestLogisticPredict(unittest.TestCase):
    def test_predict_returns_scores(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu").fit(prepared, labels)
        scores = p.predict(prepared)
        self.assertEqual(scores.shape, (20,))  # [batch] for sequence-level

    def test_predict_valid_probabilities(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu").fit(prepared, labels)
        probs = p.predict(prepared)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_predict_separable_accuracy(self):
        acts, labels = _separable_acts(n_samples=20, gap=5.0)
        prepared = acts.mean("s")
        p = Logistic(device="cpu").fit(prepared, labels)
        probs = p.predict(prepared)
        pos_correct = (probs[:10] > 0.5).sum()  # Positive class should have high prob
        neg_correct = (probs[10:] < 0.5).sum()  # Negative class should have low prob
        self.assertGreater(pos_correct + neg_correct, 14)  # >70% accuracy

    def test_predict_before_fit_raises(self):
        acts = _acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu")
        with self.assertRaises(RuntimeError):
            p.predict(prepared)

class TestLogisticSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(C=2.0, device="cpu").fit(prepared, labels)
        probs_before = p.predict(prepared)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            loaded = Logistic.load(path)

        self.assertEqual(loaded.C, 2.0)
        self.assertTrue(loaded.fitted)
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
        self.assertFalse(p.fitted)

    def test_custom_params(self):
        p = MLP(hidden_dim=64, dropout=0.1, n_epochs=50, device="cpu")
        self.assertEqual(p.hidden_dim, 64)
        self.assertEqual(p.dropout, 0.1)
        self.assertEqual(p.n_epochs, 50)

class TestMLPFit(unittest.TestCase):
    def test_fit_sequence_level(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)
        self.assertIsNotNone(p.net)

    def test_fit_token_level(self):
        acts, labels = _separable_acts(n_samples=10, seq=5)
        # Keep SEQ axis for token-level training
        p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(acts, labels)
        self.assertTrue(p.fitted)

    def test_fit_with_dropout(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, dropout=0.2, n_epochs=10, device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)

    def test_fit_gelu_activation(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, activation="gelu", n_epochs=10, device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)

class TestMLPPredict(unittest.TestCase):
    def test_predict_returns_scores(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
        scores = p.predict(prepared)
        self.assertEqual(scores.shape, (20,))  # [batch] for sequence-level

    def test_predict_valid_probabilities(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
        probs = p.predict(prepared)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_predict_before_fit_raises(self):
        acts = _acts()
        prepared = acts.mean("s")
        p = MLP(device="cpu")
        with self.assertRaises(RuntimeError):
            p.predict(prepared)

class TestMLPSaveLoad(unittest.TestCase):
    def test_save_load_roundtrip(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=32, n_epochs=10, device="cpu").fit(prepared, labels)
        probs_before = p.predict(prepared)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            loaded = MLP.load(path)

        self.assertEqual(loaded.hidden_dim, 32)
        self.assertTrue(loaded.fitted)
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
        prepared = acts.mean("s")

        # Test fit returns self
        p = ProbeClass(device="cpu", **init_kwargs)
        result = p.fit(prepared, labels)
        self.assertIs(result, p)
        self.assertTrue(p.fitted)

        # Test predict returns tensor with batch dimension
        probs = p.predict(prepared)
        self.assertEqual(probs.shape[0], 20)  # batch size
        self.assertEqual(len(probs.shape), 1)  # [batch] for sequence-level

        # Test save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            loaded = ProbeClass.load(path, device="cpu")
            self.assertTrue(loaded.fitted)
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
        # [batch, layer, seq, hidden] - 2 layers
        t = torch.randn(10, 2, 8, 16)
        acts = Activations.from_padded(
            data=t, detection_mask=torch.ones(10, 8),
            dims="blsh", layers=(0, 1),
        )
        labels = [Label.POSITIVE] * 5 + [Label.NEGATIVE] * 5
        p = Logistic(device="cpu")
        with self.assertRaises(ValueError):
            p.fit(acts, labels)

    def test_mlp_rejects_multi_layer(self):
        # [batch, layer, seq, hidden] - 2 layers
        t = torch.randn(10, 2, 8, 16)
        acts = Activations.from_padded(
            data=t, detection_mask=torch.ones(10, 8),
            dims="blsh", layers=(0, 1),
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
        t = torch.randn(4, 8, 8)  # [batch, seq, hidden]
        acts = Activations.from_padded(
            data=t, detection_mask=torch.ones(4, 8), dims="bsh",
        )
        labels = [Label.POSITIVE, Label.POSITIVE, Label.NEGATIVE, Label.NEGATIVE]
        prepared = acts.mean("s")

        p = Logistic(device="cpu").fit(prepared, labels)
        scores = p.predict(prepared)
        self.assertEqual(scores.shape, (4,))  # [batch]

    def test_integer_labels(self):
        acts, _ = _separable_acts(n_samples=10)
        prepared = acts.mean("s")
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # int labels
        p = Logistic(device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)

    def test_tensor_labels(self):
        acts, _ = _separable_acts(n_samples=10)
        prepared = acts.mean("s")
        labels = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        p = Logistic(device="cpu").fit(prepared, labels)
        self.assertTrue(p.fitted)

class TestProbeDeviceHandling(unittest.TestCase):
    """Test device handling edge cases for probes."""

    def test_logistic_token_level_device_consistency(self):
        """Test that token-level training handles device correctly."""
        # Create activations (on CPU)
        acts, labels = _separable_acts(n_samples=10, seq=5)

        # Train on CPU - should work without device mismatch
        p = Logistic(device="cpu").fit(acts, labels)
        self.assertTrue(p.fitted)

        # Verify prediction works - returns [batch, seq] for token-level
        probs = p.predict(acts)
        self.assertEqual(probs.shape, (10, 5))  # [batch, seq]

    def test_mlp_token_level_device_consistency(self):
        """Test that MLP token-level training handles device correctly."""
        acts, labels = _separable_acts(n_samples=10, seq=5)

        p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(acts, labels)
        self.assertTrue(p.fitted)

        # Returns [batch, seq] for token-level
        probs = p.predict(acts)
        self.assertEqual(probs.shape, (10, 5))  # [batch, seq]

    def test_logistic_repeat_interleave_same_device(self):
        """Verify repeat_interleave uses consistent device for labels and tokens_per_sample."""
        # This test specifically checks the fix for device mismatch in repeat_interleave
        acts, labels = _separable_acts(n_samples=6, seq=4)

        p = Logistic(device="cpu")
        p.fit(acts, labels)

        # If device handling is wrong, this would fail during fit
        self.assertTrue(p.fitted)

# =============================================================================
# Seed Reproducibility Tests
# =============================================================================

class TestSeedReproducibility(unittest.TestCase):
    def test_mlp_same_seed_same_predictions(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p1 = MLP(hidden_dim=16, n_epochs=5, seed=42, device="cpu").fit(prepared, labels)
        p2 = MLP(hidden_dim=16, n_epochs=5, seed=42, device="cpu").fit(prepared, labels)
        self.assertTrue(torch.allclose(p1.predict(prepared), p2.predict(prepared), atol=1e-5))

    def test_logistic_same_seed_same_predictions(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p1 = Logistic(seed=42, device="cpu").fit(prepared, labels)
        p2 = Logistic(seed=42, device="cpu").fit(prepared, labels)
        self.assertTrue(torch.allclose(p1.predict(prepared), p2.predict(prepared), atol=1e-5))

    def test_different_seed_different_predictions(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p1 = MLP(hidden_dim=16, n_epochs=20, seed=42, device="cpu").fit(prepared, labels)
        p2 = MLP(hidden_dim=16, n_epochs=20, seed=99, device="cpu").fit(prepared, labels)
        self.assertFalse(torch.allclose(p1.predict(prepared), p2.predict(prepared), atol=1e-6))

# =============================================================================
# Optimizer Factory Tests
# =============================================================================

class TestOptimizerFactory(unittest.TestCase):
    def test_mlp_custom_sgd(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(
            hidden_dim=16, n_epochs=10,
            optimizer_fn=lambda params: SGD(params, lr=0.01),
            device="cpu",
        ).fit(prepared, labels)
        self.assertTrue(p.fitted)

    def test_mlp_with_scheduler(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(
            hidden_dim=16, n_epochs=10,
            optimizer_fn=lambda params: SGD(params, lr=0.01),
            scheduler_fn=lambda opt: StepLR(opt, step_size=5, gamma=0.5),
            device="cpu",
        ).fit(prepared, labels)
        self.assertTrue(p.fitted)

    def test_logistic_custom_adam(self):
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = Logistic(
            n_epochs=50,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.01),
            device="cpu",
        ).fit(prepared, labels)
        self.assertTrue(p.fitted)

    def test_save_load_roundtrip_custom_optimizer(self):
        """Custom optimizer not serialized; loaded probe still works for predict."""
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(
            hidden_dim=16, n_epochs=10, seed=42,
            optimizer_fn=lambda params: SGD(params, lr=0.01),
            device="cpu",
        ).fit(prepared, labels)
        probs_before = p.predict(prepared)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            loaded = MLP.load(path)

        self.assertTrue(loaded.fitted)
        self.assertEqual(loaded.seed, 42)
        self.assertIsNone(loaded._optimizer_fn)
        probs_after = loaded.predict(prepared)
        self.assertTrue(torch.allclose(probs_before, probs_after, atol=1e-5))

    def test_attention_custom_optimizer(self):
        acts, labels = _separable_acts(n_samples=20, seq=8)
        p = Attention(
            hidden_dim=16, n_epochs=20,
            optimizer_fn=lambda params: SGD(params, lr=0.01),
            device="cpu",
        ).fit(acts, labels)
        self.assertTrue(p.fitted)

# =============================================================================
# Exposed Params Tests
# =============================================================================

class TestExposedParams(unittest.TestCase):
    def test_attention_val_split_eval_interval(self):
        acts, labels = _separable_acts(n_samples=20, seq=8)
        p = Attention(
            hidden_dim=16, n_epochs=20, val_split=0.3, eval_interval=5,
            device="cpu",
        ).fit(acts, labels)
        self.assertTrue(p.fitted)
        self.assertEqual(p.val_split, 0.3)
        self.assertEqual(p.eval_interval, 5)

    def test_multimax_batch_size_val_split(self):
        acts, labels = _separable_acts(n_samples=20, seq=8)
        p = MultiMax(
            n_epochs=5, batch_size=8, val_split=0.1,
            device="cpu",
        ).fit(acts, labels)
        self.assertTrue(p.fitted)
        self.assertEqual(p.batch_size, 8)
        self.assertEqual(p.val_split, 0.1)

    def test_gated_bipolar_batch_size_val_split(self):
        acts, labels = _separable_acts(n_samples=20, seq=8)
        p = GatedBipolar(
            n_epochs=5, batch_size=4, val_split=0.1,
            device="cpu",
        ).fit(acts, labels)
        self.assertTrue(p.fitted)
        self.assertEqual(p.batch_size, 4)
        self.assertEqual(p.val_split, 0.1)


# =============================================================================
# Dtype Policy Tests
# =============================================================================

def _bf16_separable_acts(n_samples=20, seq=8, d_model=8, gap=2.0):
    """Create linearly separable activations in bfloat16."""
    half = n_samples // 2
    t = torch.zeros(n_samples, seq, d_model, dtype=torch.bfloat16)
    t[:half, :, 0] = gap
    t[half:, :, 0] = -gap
    t[:, :, 1:] = torch.randn(n_samples, seq, d_model - 1, dtype=torch.bfloat16) * 0.1
    return Activations.from_padded(
        data=t, detection_mask=torch.ones(n_samples, seq), dims="bsh",
    ), [Label.POSITIVE] * half + [Label.NEGATIVE] * half


class TestDtypePolicy(unittest.TestCase):
    """Test unified dtype policy across probes."""

    def test_mlp_cast_none_preserves_bf16(self):
        """bf16 activations + MLP cast=None → network is bf16, predict returns bf16."""
        acts, labels = _bf16_separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(prepared, labels)
        net_dtype = next(p.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.bfloat16)
        probs = p.predict(prepared)
        self.assertEqual(probs.dtype, torch.bfloat16)

    def test_mlp_cast_float32(self):
        """bf16 activations + MLP cast='float32' → network is float32."""
        acts, labels = _bf16_separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=16, n_epochs=5, device="cpu", cast="float32").fit(prepared, labels)
        net_dtype = next(p.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.float32)

    def test_logistic_lbfgs_preserves_bf16(self):
        """bf16 activations + Logistic (LBFGS) → preserves bf16."""
        acts, labels = _bf16_separable_acts()
        prepared = acts.mean("s")
        p = Logistic(device="cpu").fit(prepared, labels)
        net_dtype = next(p.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.bfloat16)

    def test_logistic_custom_optimizer_preserves_bf16(self):
        """bf16 activations + Logistic (custom optimizer, cast=None) → network is bf16."""
        acts, labels = _bf16_separable_acts()
        prepared = acts.mean("s")
        p = Logistic(
            n_epochs=50,
            optimizer_fn=lambda params: torch.optim.Adam(params, lr=0.01),
            device="cpu",
        ).fit(prepared, labels)
        net_dtype = next(p.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.bfloat16)

    def test_predict_autocasts_mismatched_dtype(self):
        """Train float32, predict bf16 input → works (predict auto-casts)."""
        acts, labels = _separable_acts()  # float32
        prepared = acts.mean("s")
        p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(prepared, labels)
        # Verify network is float32
        self.assertEqual(next(p.net.parameters()).dtype, torch.float32)

        # Create bf16 test acts — predict() should auto-cast to float32
        bf16_acts, _ = _bf16_separable_acts()
        bf16_prepared = bf16_acts.mean("s")
        probs = p.predict(bf16_prepared)
        self.assertEqual(probs.shape[0], 20)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_save_load_preserves_training_dtype(self):
        """Train bf16, save, load → still bf16."""
        acts, labels = _bf16_separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(prepared, labels)
        self.assertEqual(p._training_dtype, torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            loaded = MLP.load(path)

        self.assertEqual(loaded._training_dtype, torch.bfloat16)
        net_dtype = next(loaded.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.bfloat16)

    def test_old_checkpoint_defaults_to_float32(self):
        """Old checkpoint without training_dtype key → loads as float32."""
        acts, labels = _separable_acts()
        prepared = acts.mean("s")
        p = MLP(hidden_dim=16, n_epochs=5, device="cpu").fit(prepared, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "probe.pt"
            p.save(path)
            # Simulate old checkpoint by removing the new keys
            state = torch.load(path, map_location="cpu")
            del state["training_dtype"]
            del state["cast"]
            torch.save(state, path)
            loaded = MLP.load(path)

        self.assertEqual(loaded._training_dtype, torch.float32)
        net_dtype = next(loaded.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.float32)

    def test_attention_bf16_trains_and_predicts(self):
        """Attention with bf16 activations → trains and predicts correctly."""
        acts, labels = _bf16_separable_acts(n_samples=20, seq=8)
        p = Attention(
            hidden_dim=16, n_epochs=20,
            device="cpu",
        ).fit(acts, labels)
        self.assertTrue(p.fitted)
        net_dtype = next(p.net.parameters()).dtype
        self.assertEqual(net_dtype, torch.bfloat16)
        probs = p.predict(acts)
        self.assertEqual(probs.shape, (20,))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))


if __name__ == '__main__':
    unittest.main()
