# Phase 5: Threshold Optimization for Imbalanced Binary Classification

Our Phase 4 results show that XGBoost has excellent ranking ability (ROC-AUC 88.4%, PR-AUC 58.7%) but poor recall at the default 0.5 threshold (only 20%). This is a classic problem under heavy class imbalance.

Goal: find the optimal decision threshold that maximizes F1 on the failure class, and report the improved metrics.

This is a 30-minute task, not a full re-training.

## Task 1: Create `src/threshold_optimization.py`

```python
def find_optimal_threshold(y_true, y_proba, metric: str = "f1") -> dict:
    """Find the threshold that maximizes the chosen metric.

    Returns: {
        "optimal_threshold": float,
        "metric_value_at_optimal": float,
        "default_threshold_metric_value": float,
        "improvement": float,
        "all_thresholds": list,
        "all_metric_values": list,
    }
    """
```

The function must:
1. Sweep thresholds from 0.05 to 0.95 in steps of 0.01
2. For each threshold, compute the metric (default: F1 on failure class)
3. Return the threshold that maximizes the metric
4. Also support: "youden_j" (TPR - FPR), "balanced_accuracy", "f1_macro"

Additionally implement `find_threshold_by_business_cost(y_true, y_proba, fp_cost, fn_cost) -> float`:
- Sweeps thresholds and minimizes total business cost
- fp_cost = cost of false alarm (default: $2.50 per false alarm)
- fn_cost = cost of missed failure (default: $18.75 per missed failure)

## Task 2: Apply to all three models

Create `src/run_phase5.py` that:

1. Loads the 3 trained models from Phase 4 (stratified split)
2. Loads test_stratified.csv
3. For each model, computes y_proba on the test set
4. Finds optimal threshold by F1
5. Finds optimal threshold by Youden's J
6. Finds optimal threshold by business cost
7. Computes the new metrics (accuracy, balanced accuracy, precision, recall, F1) at each optimal threshold
8. Saves results to `results/threshold_optimization.json`

Also apply the SAME thresholds (found on stratified test) to the CHRONOLOGICAL test set for sanity check.

## Task 3: Generate charts

### `fig_20_threshold_optimization.png`

Title: "Threshold Optimization: F1 vs Decision Threshold"
Caption: "F1-score on the failure class as a function of the classification threshold. The optimal threshold (red dot) substantially outperforms the default 0.5 threshold (gray line), particularly for XGBoost where the imbalanced default decision rule suppresses failure predictions."

Layout: single plot with 3 curves (one per model), Y = F1 score, X = threshold (0.05 to 0.95). Mark optimal points with red dots and annotate values.

### `fig_21_metrics_before_after_threshold.png`

Title: "Model Performance: Default vs Optimized Threshold"
Caption: "Comparison of accuracy, F1, precision, recall, and balanced accuracy at the default 0.5 threshold versus the F1-optimized threshold. Threshold tuning is particularly impactful for XGBoost, lifting failure-class F1 from 0.32 to 0.50+."

Layout: 1 row × 3 columns (one per model), grouped bar chart of 5 metrics × 2 conditions (default / optimized).

Append captions to figures/captions.md.

## Task 4: Update business impact

Recompute business impact with the NEW optimized thresholds. The improved recall means more failures caught and higher annual savings.

Save to `results/business_impact_optimized.json`.

## Task 5: Save best optimized model

The winning model + its optimal threshold should be saved together:

```python
best_optimized = {
    "model_name": "XGBoost" or whichever wins post-tuning,
    "optimal_threshold": float,
    "f1_at_optimal": float,
    "model_path": "models/best_optimized.joblib",
}
```

Save metadata to `models/best_optimized_metadata.json`.

## After completion give me

1. Optimal threshold per model (by F1)
2. Before / after metrics table for each model
3. The NEW winner (best F1 post-tuning)
4. Updated business impact ($ saved annually)
5. Confirmation that the optimized thresholds work on the chronological test set too
6. List of new files

## Constraints

- All code in English with type hints
- Reuse existing trained models — do NOT retrain
- Random seed 42 where applicable
- Saves intermediate results