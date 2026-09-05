"""Decision-threshold optimisation for imbalanced binary classification.

Phase 4 left us with three trained models that all rank failures well (ROC-AUC
≈ 0.86 – 0.88) but make poor decisions at the default 0.5 threshold —
especially XGBoost, which only flags ~20% of true failures. The functions in
this module sweep over candidate thresholds and pick the one that maximises a
chosen criterion (F1 on the failure class, Youden's J, balanced accuracy,
macro F1) or that minimises a stated business cost.

The module is deliberately lightweight — it operates on a 1-D
``y_proba`` array of failure probabilities plus a 0/1 ``y_true`` array, so it
can be plugged into any existing pipeline without dependency on the training
code path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


_THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
DEFAULT_THRESHOLD = 0.5

# Cost defaults — Phase 5 spec values, interpreted as "$ per outcome".
DEFAULT_FP_COST = 2.50    # false alarm: 30 s of unnecessary DevOps triage
DEFAULT_FN_COST = 18.75   # missed failure: full 5-minute manual triage @ $75/h


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


def _ensure_binary(y_true: Any) -> np.ndarray:
    arr = np.asarray(y_true)
    if arr.dtype.kind in "iu":  # already integer
        return arr.astype(int)
    return (arr.astype(str) == "failure").astype(int)


def _metric_value(
    y_true_bin: np.ndarray, y_pred_bin: np.ndarray, metric: str
) -> float:
    if metric == "f1":
        return float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    if metric == "f1_macro":
        return float(
            f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        )
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true_bin, y_pred_bin))
    if metric == "youden_j":
        try:
            tn, fp, fn, tp = confusion_matrix(
                y_true_bin, y_pred_bin, labels=[0, 1]
            ).ravel()
        except ValueError:
            return 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(tpr - fpr)
    raise ValueError(f"Unsupported metric: {metric}")


def compute_metrics_at_threshold(
    y_true: Any,
    y_proba_positive: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Return the standard metric bundle at a specific decision threshold."""
    y_true_bin = _ensure_binary(y_true)
    y_pred_bin = (np.asarray(y_proba_positive) >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_bin, y_pred_bin)
        ),
        "precision_failure": float(
            precision_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "recall_failure": float(
            recall_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "f1_failure": float(
            f1_score(y_true_bin, y_pred_bin, zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true_bin, y_pred_bin, labels=[0, 1]
        ).tolist(),
    }


# --------------------------------------------------------------------------- #
# Threshold search
# --------------------------------------------------------------------------- #


def find_optimal_threshold(
    y_true: Any,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> dict[str, Any]:
    """Find the threshold in [0.05, 0.95] that maximises ``metric``.

    ``metric`` must be one of ``"f1"`` (default), ``"youden_j"``,
    ``"balanced_accuracy"``, ``"f1_macro"``. ``y_proba`` must be the
    probability assigned to the positive (``failure``) class.

    Returns a dictionary matching the Phase 5 spec, plus a ``metric`` key
    for traceability.
    """
    y_true_bin = _ensure_binary(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = _THRESHOLD_GRID
    values: list[float] = []
    for thr in thresholds:
        y_pred_bin = (y_proba >= thr).astype(int)
        values.append(_metric_value(y_true_bin, y_pred_bin, metric))

    idx_opt = int(np.argmax(values))
    optimal_threshold = float(thresholds[idx_opt])
    metric_at_optimal = float(values[idx_opt])

    default_pred = (y_proba >= DEFAULT_THRESHOLD).astype(int)
    default_value = _metric_value(y_true_bin, default_pred, metric)

    return {
        "metric": metric,
        "optimal_threshold": optimal_threshold,
        "metric_value_at_optimal": metric_at_optimal,
        "default_threshold_metric_value": float(default_value),
        "improvement": float(metric_at_optimal - default_value),
        "all_thresholds": [float(t) for t in thresholds],
        "all_metric_values": [float(v) for v in values],
    }


def find_threshold_by_business_cost(
    y_true: Any,
    y_proba: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> dict[str, Any]:
    """Find the threshold that minimises ``fp_cost*FP + fn_cost*FN``."""
    y_true_bin = _ensure_binary(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = _THRESHOLD_GRID
    costs: list[float] = []
    for thr in thresholds:
        y_pred_bin = (y_proba >= thr).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(
                y_true_bin, y_pred_bin, labels=[0, 1]
            ).ravel()
        except ValueError:
            costs.append(float("inf"))
            continue
        costs.append(float(fp * fp_cost + fn * fn_cost))

    idx_opt = int(np.argmin(costs))
    optimal_threshold = float(thresholds[idx_opt])
    optimal_cost = float(costs[idx_opt])

    default_pred = (y_proba >= DEFAULT_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, default_pred, labels=[0, 1]
    ).ravel()
    default_cost = float(fp * fp_cost + fn * fn_cost)

    return {
        "optimal_threshold": optimal_threshold,
        "min_cost": optimal_cost,
        "default_threshold_cost": default_cost,
        "savings": default_cost - optimal_cost,
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
        "all_thresholds": [float(t) for t in thresholds],
        "all_costs": costs,
    }


__all__ = [
    "DEFAULT_FN_COST",
    "DEFAULT_FP_COST",
    "DEFAULT_THRESHOLD",
    "compute_metrics_at_threshold",
    "find_optimal_threshold",
    "find_threshold_by_business_cost",
]
