"""Tests for probelab.metrics module."""

import numpy as np
import pytest
import torch

from probelab.metrics import (
    accuracy,
    auroc,
    bootstrap,
    f1,
    fpr,
    fnr,
    get_metric_by_name,
    mean_score,
    optimal_threshold,
    partial_auroc,
    percentile,
    precision,
    recall,
    recall_at_fpr,
    std_score,
    weighted_error_rate,
)


class TestBasicMetrics:
    """Test basic metric functions."""

    def test_perfect_separation_auroc(self):
        """Test AUROC with perfect separation."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])
        assert auroc(y_true, y_pred) == 1.0

    def test_random_classification_auroc(self):
        """Test AUROC with random classification (should be ~0.5)."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6])
        assert abs(auroc(y_true, y_pred) - 0.5) < 0.1

    def test_partial_auroc(self):
        """Test partial AUROC with max FPR constraint."""
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.85, 0.2, 0.3, 0.1, 0.25])

        assert auroc(y_true, y_pred) == 1.0
        assert partial_auroc(y_true, y_pred, max_fpr=0.1) == 1.0

    def test_recall_at_fpr_perfect_separation(self):
        """Test recall@FPR with perfect separation."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        assert recall_at_fpr(y_true, y_pred, fpr=0.05) == 1.0
        assert recall_at_fpr(y_true, y_pred, fpr=0.10) == 1.0

    def test_realistic_recall_at_fpr(self):
        """Test recall@FPR with realistic overlapping scores."""
        np.random.seed(42)
        y_true = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        y_pred = np.concatenate([
            np.random.normal(0.3, 0.1, 50),  # Negative scores
            np.random.normal(0.7, 0.1, 50),  # Positive scores
        ])

        recall_1 = recall_at_fpr(y_true, y_pred, fpr=0.01)
        recall_5 = recall_at_fpr(y_true, y_pred, fpr=0.05)
        recall_10 = recall_at_fpr(y_true, y_pred, fpr=0.10)

        # Higher FPR should give higher or equal recall
        assert recall_1 <= recall_5 <= recall_10
        assert 0 <= recall_1 <= 1
        assert 0 <= recall_5 <= 1
        assert 0 <= recall_10 <= 1

    def test_accuracy_metric(self):
        """Test accuracy metric."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.2, 0.3])
        assert accuracy(y_true, y_pred) == 1.0

    def test_precision_recall_f1(self):
        """Test precision, recall, and F1 metrics."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.3, 0.2, 0.1, 0.6])  # One FN, one FP

        assert precision(y_true, y_pred) == pytest.approx(2 / 3)
        assert recall(y_true, y_pred) == pytest.approx(2 / 3)
        assert f1(y_true, y_pred) == pytest.approx(2 / 3)


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_bootstrap_function(self):
        """Test bootstrap function directly."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        def simple_auroc(yt, yp):
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(yt, yp))

        point, ci_low, ci_high = bootstrap(simple_auroc, y_true, y_pred, n=100, seed=42)
        assert point == 1.0
        assert ci_low <= point <= ci_high

    def test_bootstrap_with_custom_metric(self):
        """Test bootstrap with custom metric function."""
        def custom_mean(yt, yp):
            return float(np.mean(yp))

        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.8, 0.7, 0.2, 0.3])

        point, ci_low, ci_high = bootstrap(custom_mean, y_true, y_pred, n=100, seed=42)
        assert point == pytest.approx(0.5)
        assert ci_low <= point <= ci_high

    def test_bootstrap_with_builtin_metrics(self):
        """Test bootstrap with built-in metrics."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])

        # Default metrics return raw floats
        result = auroc(y_true, y_pred)
        assert isinstance(result, float)

        # Use bootstrap() function for CIs
        point, ci_low, ci_high = bootstrap(
            lambda yt, yp: float(__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(yt, yp)),
            y_true, y_pred, n=100, seed=42
        )
        assert isinstance(point, float)
        assert ci_low <= point <= ci_high


class TestDistributionMetrics:
    """Test distribution statistics metrics."""

    def test_mean_std_score(self):
        """Test mean and std score metrics."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        assert mean_score(y_true, y_pred) == pytest.approx(0.3)
        assert std_score(y_true, y_pred) == pytest.approx(np.std(y_pred))

    def test_percentile_metrics(self):
        """Test percentile metrics."""
        y_true = np.array([0] * 9)
        y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        assert percentile(y_true, y_pred, q=50) == pytest.approx(0.5)
        assert percentile(y_true, y_pred, q=90) == pytest.approx(0.82)
        assert percentile(y_true, y_pred, q=10) == pytest.approx(0.18)


class TestFPRFNRMetrics:
    """Test false positive/negative rate metrics."""

    def test_fpr_default_threshold(self):
        """Test FPR at default threshold."""
        y_true = np.array([0, 0, 0, 0, 0])  # All negative
        y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8])

        result = fpr(y_true, y_pred)
        assert result == 0.4  # 2 out of 5 > 0.5

    def test_fpr_custom_threshold(self):
        """Test FPR at custom threshold."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8])

        result = fpr(y_true, y_pred, threshold=0.7)
        assert result == 0.2  # 1 out of 5 > 0.7

    def test_fpr_with_mixed_dataset(self):
        """Test FPR with mixed positive/negative examples."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

        result = fpr(y_true, y_pred, threshold=0.5)
        assert result == 0.25  # 1 out of 4 negatives > 0.5

    def test_fnr_default_threshold(self):
        """Test FNR at default threshold."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.3, 0.4, 0.6, 0.8])

        result = fnr(y_true, y_pred)
        assert result == 0.5  # 2 out of 4 <= 0.5

    def test_fnr_custom_threshold(self):
        """Test FNR at custom threshold."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.3, 0.4, 0.6, 0.8])

        result = fnr(y_true, y_pred, threshold=0.7)
        assert result == 0.75  # 3 out of 4 <= 0.7


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        with pytest.raises(ValueError):
            auroc(np.array([]), np.array([]))

    def test_single_class(self):
        """Test behavior with only one class."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0.8, 0.9, 0.7])

        with pytest.raises(ValueError, match="Cannot compute AUROC with only one class"):
            auroc(y_true, y_pred)

    def test_identical_scores(self):
        """Test with identical scores."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        assert abs(auroc(y_true, y_pred) - 0.5) < 0.1

    def test_mixed_tensor_types(self):
        """Test with mixed tensor types."""
        y_true = torch.tensor([1, 1, 0, 0], dtype=torch.int32)
        y_pred = torch.tensor([0.8, 0.9, 0.2, 0.3], dtype=torch.float64)

        result = auroc(y_true, y_pred)
        assert 0.5 <= result <= 1.0


class TestGetMetricByName:
    """Test metric lookup by name."""

    def test_registry_lookup(self):
        """Test looking up metrics from registry."""
        assert get_metric_by_name("auroc") == auroc
        assert get_metric_by_name("accuracy") == accuracy
        assert get_metric_by_name("f1") == f1

    def test_recall_at_fpr_syntax(self):
        """Test recall@X syntax."""
        metric = get_metric_by_name("recall@5")
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])
        assert metric(y_true, y_pred) == 1.0

    def test_tpr_at_fpr_syntax(self):
        """Test tpr@X syntax (alias for recall@X)."""
        metric = get_metric_by_name("tpr@5")
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.7, 0.2, 0.3, 0.1])
        assert metric(y_true, y_pred) == 1.0

    def test_percentile_syntax(self):
        """Test percentileX syntax."""
        metric = get_metric_by_name("percentile95")
        y_true = np.array([0] * 100)
        y_pred = np.linspace(0, 1, 100)
        assert 0.9 < metric(y_true, y_pred) < 1.0

    def test_unknown_metric(self):
        """Test unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric_by_name("unknown_metric")


class TestWeightedErrorMetrics:
    """Test weighted error rate metrics from GDM paper."""

    def test_weighted_error_rate_basic(self):
        """Test basic weighted error rate calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.3, 0.6, 0.4, 0.9])  # One FP, one FN

        result = weighted_error_rate(y_true, y_pred, threshold=0.5, fnr_weight=5.0, fpr_weight=50.0)
        assert result == pytest.approx(0.5)

    def test_weighted_error_rate_perfect_classifier(self):
        """Test weighted error rate with perfect classification."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert weighted_error_rate(y_true, y_pred, threshold=0.5) == 0.0

    def test_weighted_error_rate_all_false_positives(self):
        """Test weighted error rate with all false positives."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0.8, 0.9, 0.6, 0.7])

        result = weighted_error_rate(y_true, y_pred, threshold=0.5, fnr_weight=5.0, fpr_weight=50.0)
        assert result == pytest.approx(50 / 55)

    def test_weighted_error_rate_all_false_negatives(self):
        """Test weighted error rate with all false negatives."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        result = weighted_error_rate(y_true, y_pred, threshold=0.5, fnr_weight=5.0, fpr_weight=50.0)
        assert result == pytest.approx(5 / 55)

    def test_optimal_threshold_basic(self):
        """Test optimal threshold finding."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])

        threshold, error = optimal_threshold(y_true, y_pred)
        assert 0.3 < threshold < 0.7
        assert error == 0.0

    def test_optimal_threshold_returns_tuple(self):
        """Test that optimal_threshold returns a tuple."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])

        result = optimal_threshold(y_true, y_pred)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_weighted_error_rate_with_2d_proba(self):
        """Test weighted error rate with 2D probability array."""
        y_true = np.array([0, 0, 1, 1])
        y_pred_2d = np.array([[0.7, 0.3], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]])

        result = weighted_error_rate(y_true, y_pred_2d, threshold=0.5)
        assert result == pytest.approx(0.5)
