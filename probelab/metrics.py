"""Function-based metrics for probe evaluation."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _prep(y_true, y_pred):
    """Convert inputs to numpy, extract positive class probabilities."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().float().numpy()
    else:
        y_true = np.asarray(y_true)
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().float().numpy()
    else:
        y_pred = np.asarray(y_pred)
    proba = y_pred[:, 1] if y_pred.ndim == 2 else y_pred
    return y_true, proba


def _to_binary(proba, threshold=0.5):
    """Convert probabilities to binary predictions."""
    return (proba > threshold).astype(int)


# =============================================================================
# Bootstrap
# =============================================================================


def bootstrap(metric_fn, y_true, y_pred, n=1000, confidence=0.95, seed=None):
    """Compute metric with bootstrap confidence intervals.

    Returns: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    y_true, proba = _prep(y_true, y_pred)

    point = metric_fn(y_true, proba)

    samples = []
    for _ in range(n):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            samples.append(metric_fn(y_true[idx], proba[idx]))
        except (ValueError, ZeroDivisionError):
            continue

    if len(samples) < n * 0.5:
        return point, np.nan, np.nan

    alpha = (1 - confidence) / 2
    return point, float(np.percentile(samples, alpha * 100)), float(np.percentile(samples, (1 - alpha) * 100))


# =============================================================================
# Core Metrics
# =============================================================================


def auroc(y_true, y_pred):
    """Area under ROC curve."""
    y_true, proba = _prep(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot compute AUROC with only one class")
    return float(roc_auc_score(y_true, proba))


def partial_auroc(y_true, y_pred, max_fpr=0.1):
    """Partial AUROC up to max_fpr."""
    y_true, proba = _prep(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot compute partial AUROC with only one class")
    return float(roc_auc_score(y_true, proba, max_fpr=max_fpr))


def accuracy(y_true, y_pred, threshold=0.5):
    """Classification accuracy."""
    y_true, proba = _prep(y_true, y_pred)
    return float(accuracy_score(y_true, _to_binary(proba, threshold)))


def balanced_accuracy(y_true, y_pred, threshold=0.5):
    """Balanced accuracy (average recall per class)."""
    y_true, proba = _prep(y_true, y_pred)
    return float(balanced_accuracy_score(y_true, _to_binary(proba, threshold)))


def precision(y_true, y_pred, threshold=0.5):
    """Precision (TP / (TP + FP))."""
    y_true, proba = _prep(y_true, y_pred)
    return float(precision_score(y_true, _to_binary(proba, threshold), zero_division=0.0))


def recall(y_true, y_pred, threshold=0.5):
    """Recall (TP / (TP + FN))."""
    y_true, proba = _prep(y_true, y_pred)
    return float(recall_score(y_true, _to_binary(proba, threshold), zero_division=0.0))


def f1(y_true, y_pred, threshold=0.5):
    """F1 score (harmonic mean of precision and recall)."""
    y_true, proba = _prep(y_true, y_pred)
    return float(f1_score(y_true, _to_binary(proba, threshold), zero_division=0.0))


def average_precision(y_true, y_pred):
    """Average precision, i.e. area under the precision-recall curve."""
    y_true, proba = _prep(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot compute average precision with only one class")
    return float(average_precision_score(y_true, proba))


def brier_score(y_true, y_pred):
    """Brier score for probabilistic binary predictions."""
    y_true, proba = _prep(y_true, y_pred)
    return float(brier_score_loss(y_true, proba))


# =============================================================================
# Recall at FPR
# =============================================================================


def recall_at_fpr(y_true, y_pred, fpr=0.05):
    """Recall at a specified false positive rate."""
    assert 0.0 <= fpr <= 1.0, "fpr must be between 0 and 1"
    y_true, proba = _prep(y_true, y_pred)

    pos_scores = proba[y_true == 1]
    neg_scores = proba[y_true == 0]

    if len(pos_scores) == 0:
        return 0.0
    if len(neg_scores) == 0:
        return 1.0

    neg_sorted = np.sort(neg_scores)
    threshold_idx = min(int((1 - fpr) * len(neg_sorted)), len(neg_sorted) - 1)
    threshold = neg_sorted[threshold_idx]

    return float(np.sum(pos_scores > threshold) / len(pos_scores))


# =============================================================================
# FPR / FNR
# =============================================================================


def fpr(y_true, y_pred, threshold=0.5):
    """False positive rate (FP / (FP + TN))."""
    y_true, proba = _prep(y_true, y_pred)
    neg = proba[y_true == 0]
    return float(np.mean(neg > threshold)) if len(neg) else np.nan


def fnr(y_true, y_pred, threshold=0.5):
    """False negative rate (FN / (FN + TP))."""
    y_true, proba = _prep(y_true, y_pred)
    pos = proba[y_true == 1]
    return float(np.mean(pos <= threshold)) if len(pos) else np.nan


def confusion_counts(y_true, y_pred, threshold=0.5):
    """Return binary confusion counts as ``{"tn", "fp", "fn", "tp"}``."""
    y_true, proba = _prep(y_true, y_pred)
    y_bool = y_true.astype(bool)
    p_bool = _to_binary(proba, threshold).astype(bool)
    return {
        "tn": int((~y_bool & ~p_bool).sum()),
        "fp": int((~y_bool & p_bool).sum()),
        "fn": int((y_bool & ~p_bool).sum()),
        "tp": int((y_bool & p_bool).sum()),
    }


def true_negatives(y_true, y_pred, threshold=0.5):
    """Number of true negatives."""
    return confusion_counts(y_true, y_pred, threshold)["tn"]


def false_positives(y_true, y_pred, threshold=0.5):
    """Number of false positives."""
    return confusion_counts(y_true, y_pred, threshold)["fp"]


def false_negatives(y_true, y_pred, threshold=0.5):
    """Number of false negatives."""
    return confusion_counts(y_true, y_pred, threshold)["fn"]


def true_positives(y_true, y_pred, threshold=0.5):
    """Number of true positives."""
    return confusion_counts(y_true, y_pred, threshold)["tp"]


def max_balanced_accuracy_threshold(y_true, y_pred):
    """Find the threshold that maximizes balanced accuracy.

    Returns:
        ``(threshold, score)`` using the unique predicted probabilities as
        candidate thresholds.
    """
    y_true, proba = _prep(y_true, y_pred)
    if len(np.unique(y_true)) < 2:
        raise ValueError("Cannot optimize balanced accuracy with only one class")
    thresholds = np.unique(proba)
    best_threshold = float(thresholds[0])
    best_score = -1.0
    for threshold in thresholds:
        score = balanced_accuracy(y_true, proba, threshold=float(threshold))
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def max_balanced_accuracy(y_true, y_pred):
    """Best balanced accuracy over all unique score thresholds."""
    return max_balanced_accuracy_threshold(y_true, y_pred)[1]


# =============================================================================
# Distribution Statistics
# =============================================================================


def mean_score(y_true, y_pred):
    """Mean predicted probability."""
    _, proba = _prep(y_true, y_pred)
    return float(np.mean(proba))


def std_score(y_true, y_pred):
    """Standard deviation of predicted probabilities."""
    _, proba = _prep(y_true, y_pred)
    return float(np.std(proba))


def percentile(y_true, y_pred, q=95):
    """Q-th percentile of predicted probabilities."""
    _, proba = _prep(y_true, y_pred)
    return float(np.percentile(proba, q))


# =============================================================================
# Weighted Error (GDM Paper)
# =============================================================================


def weighted_error_rate(y_true, y_pred, threshold=0.5, fnr_weight=5.0, fpr_weight=50.0):
    """Weighted combination of FPR and FNR (GDM paper)."""
    y_true, proba = _prep(y_true, y_pred)
    y_bin = _to_binary(proba, threshold)

    pos_mask, neg_mask = y_true == 1, y_true == 0
    fnr_val = np.mean(y_bin[pos_mask] == 0) if pos_mask.any() else 0.0
    fpr_val = np.mean(y_bin[neg_mask] == 1) if neg_mask.any() else 0.0

    return (fnr_weight * fnr_val + fpr_weight * fpr_val) / (fnr_weight + fpr_weight)


def optimal_threshold(y_true, y_pred, fnr_weight=5.0, fpr_weight=50.0, n_thresholds=100):
    """Find threshold that minimizes weighted error. Returns (threshold, error)."""
    y_true, proba = _prep(y_true, y_pred)

    best_t, best_err = 0.5, float("inf")
    for t in np.linspace(0, 1, n_thresholds):
        err = weighted_error_rate(y_true, proba, t, fnr_weight, fpr_weight)
        if err < best_err:
            best_t, best_err = t, err

    return float(best_t), float(best_err)


# =============================================================================
# Registry
# =============================================================================

METRICS_REGISTRY = {
    "auroc": auroc,
    "accuracy": accuracy,
    "average_precision": average_precision,
    "balanced_accuracy": balanced_accuracy,
    "brier_score": brier_score,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "fpr": fpr,
    "fnr": fnr,
    "tn": true_negatives,
    "fp": false_positives,
    "fn": false_negatives,
    "tp": true_positives,
    "max_balanced_accuracy": max_balanced_accuracy,
    "mean_score": mean_score,
    "std_score": std_score,
    "weighted_error": weighted_error_rate,
}
