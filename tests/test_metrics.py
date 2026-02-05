"""Tests for metrics module."""
import unittest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from probelab.metrics import (
  auroc, partial_auroc, accuracy, balanced_accuracy, precision, recall, f1,
  recall_at_fpr, fpr, fnr, mean_score, std_score, percentile,
  weighted_error_rate, optimal_threshold, bootstrap, get_metric_by_name,
)

# =============================================================================
# Test Data
# =============================================================================

def _perfect():
  """Perfect separation."""
  y_true = np.array([1, 1, 1, 0, 0, 0])
  y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
  return y_true, y_pred

def _random():
  """Random (no separation)."""
  y_true = np.array([1, 1, 1, 0, 0, 0])
  y_pred = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
  return y_true, y_pred

def _realistic():
  """Realistic overlapping scores."""
  np.random.seed(42)
  y_true = np.concatenate([np.zeros(50), np.ones(50)])
  y_pred = np.concatenate([
    np.random.normal(0.3, 0.1, 50),
    np.random.normal(0.7, 0.1, 50),
  ])
  return y_true, y_pred

# =============================================================================
# AUROC Tests
# =============================================================================

class TestAUROC(unittest.TestCase):
  def test_perfect_separation(self):
    y_true, y_pred = _perfect()
    self.assertEqual(auroc(y_true, y_pred), 1.0)

  def test_random(self):
    y_true, y_pred = _random()
    self.assertAlmostEqual(auroc(y_true, y_pred), 0.5, places=1)

  def test_single_class_raises(self):
    with self.assertRaises(ValueError):
      auroc(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))

  def test_accepts_tensor(self):
    y_true = torch.tensor([1, 1, 0, 0])
    y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1])
    self.assertEqual(auroc(y_true, y_pred), 1.0)

  def test_accepts_2d_proba(self):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]])
    self.assertEqual(auroc(y_true, y_pred), 1.0)

class TestPartialAUROC(unittest.TestCase):
  def test_perfect_at_low_fpr(self):
    y_true, y_pred = _perfect()
    self.assertEqual(partial_auroc(y_true, y_pred, max_fpr=0.1), 1.0)

  def test_single_class_raises(self):
    with self.assertRaises(ValueError):
      partial_auroc(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]), max_fpr=0.1)

# =============================================================================
# Classification Metrics
# =============================================================================

class TestAccuracy(unittest.TestCase):
  def test_perfect(self):
    y_true, y_pred = _perfect()
    self.assertEqual(accuracy(y_true, y_pred), 1.0)

  def test_custom_threshold(self):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.6, 0.4, 0.1])
    self.assertEqual(accuracy(y_true, y_pred, threshold=0.5), 1.0)
    self.assertEqual(accuracy(y_true, y_pred, threshold=0.8), 0.75)

class TestBalancedAccuracy(unittest.TestCase):
  def test_perfect(self):
    y_true, y_pred = _perfect()
    self.assertEqual(balanced_accuracy(y_true, y_pred), 1.0)

  def test_imbalanced(self):
    y_true = np.array([1, 0, 0, 0, 0, 0])
    y_pred = np.array([0.9, 0.4, 0.4, 0.4, 0.4, 0.4])
    self.assertEqual(balanced_accuracy(y_true, y_pred), 1.0)

class TestPrecision(unittest.TestCase):
  def test_perfect(self):
    y_true, y_pred = _perfect()
    self.assertEqual(precision(y_true, y_pred), 1.0)

  def test_with_fp(self):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.6, 0.7, 0.1])  # one FP
    self.assertAlmostEqual(precision(y_true, y_pred), 2/3, places=3)

class TestRecall(unittest.TestCase):
  def test_perfect(self):
    y_true, y_pred = _perfect()
    self.assertEqual(recall(y_true, y_pred), 1.0)

  def test_with_fn(self):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.3, 0.4, 0.1])  # one FN
    self.assertEqual(recall(y_true, y_pred), 0.5)

class TestF1(unittest.TestCase):
  def test_perfect(self):
    y_true, y_pred = _perfect()
    self.assertEqual(f1(y_true, y_pred), 1.0)

  def test_harmonic_mean(self):
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.6])
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    expected = 2 * p * r / (p + r)
    self.assertAlmostEqual(f1(y_true, y_pred), expected, places=5)

# =============================================================================
# Recall at FPR
# =============================================================================

class TestRecallAtFPR(unittest.TestCase):
  def test_perfect_at_1pct(self):
    y_true, y_pred = _perfect()
    self.assertEqual(recall_at_fpr(y_true, y_pred, fpr=0.01), 1.0)

  def test_perfect_at_5pct(self):
    y_true, y_pred = _perfect()
    self.assertEqual(recall_at_fpr(y_true, y_pred, fpr=0.05), 1.0)

  def test_monotonic_with_fpr(self):
    y_true, y_pred = _realistic()
    r1 = recall_at_fpr(y_true, y_pred, fpr=0.01)
    r5 = recall_at_fpr(y_true, y_pred, fpr=0.05)
    r10 = recall_at_fpr(y_true, y_pred, fpr=0.10)
    self.assertLessEqual(r1, r5)
    self.assertLessEqual(r5, r10)

  def test_bounds(self):
    y_true, y_pred = _realistic()
    for fpr_val in [0.01, 0.05, 0.10]:
      r = recall_at_fpr(y_true, y_pred, fpr=fpr_val)
      self.assertGreaterEqual(r, 0.0)
      self.assertLessEqual(r, 1.0)

  def test_no_positives(self):
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    self.assertEqual(recall_at_fpr(y_true, y_pred, fpr=0.05), 0.0)

  def test_no_negatives(self):
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.6, 0.7, 0.8, 0.9])
    self.assertEqual(recall_at_fpr(y_true, y_pred, fpr=0.05), 1.0)

# =============================================================================
# FPR / FNR
# =============================================================================

class TestFPR(unittest.TestCase):
  def test_all_negatives_correct(self):
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    self.assertEqual(fpr(y_true, y_pred), 0.0)

  def test_some_fp(self):
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
    self.assertEqual(fpr(y_true, y_pred), 0.4)

  def test_custom_threshold(self):
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8])
    self.assertEqual(fpr(y_true, y_pred, threshold=0.7), 0.2)

class TestFNR(unittest.TestCase):
  def test_all_positives_correct(self):
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.6, 0.7, 0.8, 0.9])
    self.assertEqual(fnr(y_true, y_pred), 0.0)

  def test_some_fn(self):
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.3, 0.4, 0.6, 0.8])
    self.assertEqual(fnr(y_true, y_pred), 0.5)

  def test_custom_threshold(self):
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.3, 0.4, 0.6, 0.8])
    self.assertEqual(fnr(y_true, y_pred, threshold=0.7), 0.75)

# =============================================================================
# Distribution Metrics
# =============================================================================

class TestDistributionMetrics(unittest.TestCase):
  def test_mean_score(self):
    y_true = np.zeros(5)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    self.assertAlmostEqual(mean_score(y_true, y_pred), 0.3, places=5)

  def test_std_score(self):
    y_true = np.zeros(5)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    self.assertAlmostEqual(std_score(y_true, y_pred), np.std(y_pred), places=5)

  def test_percentile_50(self):
    y_true = np.zeros(9)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    self.assertAlmostEqual(percentile(y_true, y_pred, q=50), 0.5, places=3)

  def test_percentile_90(self):
    y_true = np.zeros(9)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    self.assertAlmostEqual(percentile(y_true, y_pred, q=90), 0.82, places=1)

# =============================================================================
# Weighted Error
# =============================================================================

class TestWeightedError(unittest.TestCase):
  def test_perfect_classifier(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    self.assertEqual(weighted_error_rate(y_true, y_pred), 0.0)

  def test_one_fp_one_fn(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.3, 0.6, 0.4, 0.9])  # 1 FP, 1 FN
    result = weighted_error_rate(y_true, y_pred, fnr_weight=5.0, fpr_weight=50.0)
    self.assertAlmostEqual(result, 0.5, places=3)

  def test_all_fp(self):
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.8, 0.9, 0.6, 0.7])
    result = weighted_error_rate(y_true, y_pred, fnr_weight=5.0, fpr_weight=50.0)
    self.assertAlmostEqual(result, 50/55, places=3)

  def test_all_fn(self):
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    result = weighted_error_rate(y_true, y_pred, fnr_weight=5.0, fpr_weight=50.0)
    self.assertAlmostEqual(result, 5/55, places=3)

class TestOptimalThreshold(unittest.TestCase):
  def test_returns_tuple(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    result = optimal_threshold(y_true, y_pred)
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 2)

  def test_perfect_classifier(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.3, 0.7, 0.9])
    threshold, error = optimal_threshold(y_true, y_pred)
    self.assertGreater(threshold, 0.3)
    self.assertLess(threshold, 0.7)
    self.assertEqual(error, 0.0)

# =============================================================================
# Bootstrap
# =============================================================================

class TestBootstrap(unittest.TestCase):
  def test_returns_tuple(self):
    y_true, y_pred = _perfect()
    result = bootstrap(lambda yt, yp: float(np.mean(yp)), y_true, y_pred, n=100, seed=42)
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 3)

  def test_ci_contains_point(self):
    y_true, y_pred = _realistic()
    point, ci_low, ci_high = bootstrap(
      lambda yt, yp: float(roc_auc_score(yt, yp)),
      y_true, y_pred, n=100, seed=42
    )
    self.assertLessEqual(ci_low, point)
    self.assertLessEqual(point, ci_high)

  def test_reproducible(self):
    y_true, y_pred = _perfect()
    r1 = bootstrap(lambda yt, yp: float(np.mean(yp)), y_true, y_pred, n=100, seed=42)
    r2 = bootstrap(lambda yt, yp: float(np.mean(yp)), y_true, y_pred, n=100, seed=42)
    self.assertEqual(r1, r2)

# =============================================================================
# Registry
# =============================================================================

class TestRegistry(unittest.TestCase):
  def test_basic_lookup(self):
    self.assertEqual(get_metric_by_name("auroc"), auroc)
    self.assertEqual(get_metric_by_name("accuracy"), accuracy)
    self.assertEqual(get_metric_by_name("f1"), f1)

  def test_recall_at_syntax(self):
    metric = get_metric_by_name("recall@5")
    y_true, y_pred = _perfect()
    self.assertEqual(metric(y_true, y_pred), 1.0)

  def test_tpr_at_syntax(self):
    metric = get_metric_by_name("tpr@5")
    y_true, y_pred = _perfect()
    self.assertEqual(metric(y_true, y_pred), 1.0)

  def test_percentile_syntax(self):
    metric = get_metric_by_name("percentile95")
    y_true = np.zeros(100)
    y_pred = np.linspace(0, 1, 100)
    result = metric(y_true, y_pred)
    self.assertGreater(result, 0.9)
    self.assertLess(result, 1.0)

  def test_unknown_raises(self):
    with self.assertRaises(ValueError):
      get_metric_by_name("unknown_metric")

# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases(unittest.TestCase):
  def test_empty_arrays(self):
    with self.assertRaises((ValueError, IndexError)):
      auroc(np.array([]), np.array([]))

  def test_identical_scores(self):
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    self.assertAlmostEqual(auroc(y_true, y_pred), 0.5, places=1)

  def test_mixed_dtypes(self):
    y_true = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
    y_pred = torch.tensor([0.9, 0.8, 0.2, 0.1], dtype=torch.float64)
    result = auroc(y_true, y_pred)
    self.assertGreater(result, 0.5)
    self.assertLessEqual(result, 1.0)

  def test_2d_proba_with_error(self):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]])
    result = weighted_error_rate(y_true, y_pred)
    self.assertAlmostEqual(result, 0.5, places=3)

if __name__ == '__main__':
  unittest.main()
