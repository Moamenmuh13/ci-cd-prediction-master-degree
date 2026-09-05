# Phase 4: Full Training, Dual Evaluation, and Ablation Study

This is the most important phase. We do FULL training on 7,817 rows for each of 3 models, evaluate on BOTH splits (stratified primary, chronological secondary), run an ablation study to test whether the Hybrid pipeline beats single-modality baselines, and generate publication-quality charts for the Results chapter.

## Critical context

- Two splits available:
    - `data/processed/train_stratified.csv` + `test_stratified.csv` (PRIMARY)
    - `data/processed/train_chronological.csv` + `test_chronological.csv` (SECONDARY)
- 3 models: Logistic Regression, Random Forest, XGBoost
- Binary target: conclusion (1=failure, 0=success)
- Class imbalance: 89/11 in primary, 88/12 train vs 93/7 test in chronological
- Failure is the POSITIVE class (the thing we want to predict)

## Task 1: Create `src/train_evaluate.py`

A comprehensive training and evaluation module.

### 1.1 Metrics function

```python
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute comprehensive binary classification metrics."""
```

Compute:
- accuracy
- balanced_accuracy (critical for imbalanced data)
- precision (failure class)
- recall (failure class)
- f1_score (failure class)
- macro_f1
- weighted_f1
- roc_auc
- pr_auc (Precision-Recall AUC) — most honest under imbalance
- specificity (recall on success class)
- confusion_matrix as nested list

### 1.2 Train and evaluate one model

```python
def train_and_evaluate_one(
    pipeline, X_train, y_train, X_test, y_test, model_name: str
) -> dict:
    """Fit pipeline, predict, compute metrics, time everything."""
```

Returns dict with: metrics, fit_time, predict_time, confusion_matrix, y_pred, y_proba (for later use in charts).

### 1.3 Run all models on one split

```python
def run_all_models(X_train, y_train, X_test, y_test, split_name: str) -> dict:
    """Train and evaluate all 3 models on the given split."""
```

Save trained models to `models/{model}_{split_name}.joblib` with compression=3.

### 1.4 Ablation study function

```python
def run_ablation_study(X_train, y_train, X_test, y_test) -> dict:
    """Test whether Hybrid > single-modality baselines.

    Uses XGBoost only (the winning model from main training).
    Tests 4 configurations:
        1. Text only (drop numerical + categorical + binary)
        2. Categorical only (drop numerical + binary + text)
        3. Structured only (numerical + categorical + binary, no text)
        4. Hybrid full (all 4 branches)
    """
```

For each configuration, create a modified preprocessor (rebuild ColumnTransformer with only the relevant branches), wrap in XGBClassifier with the same hyperparameters, train, evaluate, record metrics.

Save results to `results/ablation_study.json`.

This is the most important table in the thesis.

### 1.5 Business impact metrics

```python
def compute_business_metrics(metrics_dict) -> dict:
    """Translate technical metrics into business value."""
```

Assumptions (document them):
- Mid-sized organization: 1,000 builds per day
- Failure rate: 11% (observed)
- Average failed build wastes 8 minutes of compute at $0.008/min = $0.064 per failed build
- Average failed build costs developer 15 minutes of context-switching at $75/hr = $18.75 per failed build
- If model precision on failure = P and recall = R:
    - Daily failures caught = 110 × R
    - Daily false alarms = (caught / P) - caught
    - Compute saved per day = caught × $0.064
    - Developer time saved per day = caught × $18.75
    - False alarm cost = false_alarms × 2min × $75/60 = false_alarms × $2.50
    - Net daily savings, monthly, annual

Save to `results/business_impact.json`.

## Task 2: Run the experiment

Create `src/run_phase4.py` that:

1. Loads stratified train/test (PRIMARY)
2. Trains all 3 models on stratified
3. Records metrics in `results/main_metrics_stratified.json`

4. Loads chronological train/test (SECONDARY)
5. Trains all 3 models on chronological
6. Records metrics in `results/main_metrics_chronological.json`

7. Runs the ablation study on stratified data (the primary)
8. Computes business metrics for the best model

9. Saves the best model (by F1 on failure class, stratified split) as `models/best_model.joblib`

## Task 3: Generate evaluation charts

Use ThesisPlotter, 300 DPI. Save to figures/.

### `fig_12_confusion_matrices.png`

Title: "Confusion Matrices Across Models (Stratified Test Set)"
Caption: "Confusion matrices for the three classifiers on the stratified test set. Diagonal dominance indicates correct predictions. Class imbalance is reflected in the off-diagonal counts."

Layout: 1 row × 3 columns. Annotate counts and percentages.

### `fig_13_roc_curves.png`

Title: "ROC Curves Comparison"
Caption: "Receiver Operating Characteristic curves for the three models on the stratified test set. ROC-AUC values quantify discriminative ability independent of threshold."

Single plot, 3 curves + random baseline.

### `fig_14_pr_curves.png`

Title: "Precision-Recall Curves (Failure Class)"
Caption: "Precision-Recall curves on the failure class for the three models. PR-AUC is the most honest metric under 89:11 class imbalance."

Single plot, 3 curves + baseline (failure prevalence).

### `fig_15_metrics_comparison.png`

Title: "Comparative Performance Across Evaluation Metrics"
Caption: "Comparative bar chart of accuracy, balanced accuracy, F1, ROC-AUC, and PR-AUC across the three models on the stratified test set. XGBoost achieves the highest scores on the metrics that matter most under class imbalance."

Grouped bars: 5 metrics × 3 models.

### `fig_16_ablation_study.png`

Title: "Ablation Study: Feature Modality Contributions"
Caption: "Performance of XGBoost across four feature configurations: Text-only, Categorical-only, Structured-only, and Hybrid (all). Empirical comparison validates or challenges the hybrid framing."

Grouped bars: F1, PR-AUC, ROC-AUC for each of the 4 configurations.

### `fig_17_feature_importance.png`

Title: "Top 30 Feature Importances (XGBoost Hybrid)"
Caption: "Top 30 features ranked by XGBoost importance, color-coded by modality. Reveals which feature categories drive predictions in the hybrid model."

Horizontal bar chart, top 30 from xgb.feature_importances_. Color: numerical=blue, categorical=green, binary=orange, text=purple.

### `fig_18_stratified_vs_chronological.png`

Title: "Stratified vs Chronological Evaluation"
Caption: "Comparison of model performance under stratified random split (primary) and chronological split (secondary). The performance gap quantifies sensitivity to temporal data drift in deployment scenarios."

Grouped bars: 3 metrics (F1, PR-AUC, balanced accuracy) × 3 models × 2 splits.

### `fig_19_business_impact.png`

Title: "Estimated Business Impact of the Predictive System"
Caption: "Estimated daily, monthly, and annual cost savings from deploying the best model (XGBoost) at the assumed scale. Net savings account for both compute reclaimed from correctly predicted failures and the cost of false alarms."

Bar chart or pie chart showing the cost breakdown.

Append all 8 captions to `figures/captions.md`.

## Task 4: Reporting

Print a clean summary to console and save to `results/phase4_summary.md`:

1. Stratified results table (3 models × all metrics)
2. Chronological results table (3 models × all metrics)
3. Ablation study table (4 configs × XGBoost × all metrics)
4. Business impact summary
5. Winner declaration: "Best model: X, F1: Y, PR-AUC: Z"
6. Cross-split comparison: "Stratified F1: A, Chronological F1: B, Drift: C pp"

## Constraints

- All code in English with type hints
- Use logging for status (training takes time)
- Save intermediate JSON results so we don't lose work
- Random seed 42 everywhere
- Use joblib compression=3 to keep model files reasonable

## After completion, give me

1. Full metrics table (3 models × 2 splits) — markdown formatted
2. Ablation study table (4 configs)
3. Business impact summary ($ saved per year)
4. The winner and its metrics on BOTH splits
5. Drift analysis: how much does performance drop on chronological vs stratified?
6. List of all new files in models/, results/, figures/
7. Any warnings or surprises

## Expectations for sanity-checking results

If F1 on failure class > 0.40 on stratified test set: solid result, can defend
If F1 = 0.20–0.40: acceptable but needs hyperparameter tuning later
If F1 < 0.20: fundamental data issue (we'll discuss)

If ablation shows Text-only ≈ Hybrid (within 2 pp): be honest about it in the thesis discussion.
If Categorical-only is competitive: that means repository identity drives predictions, NOT commit content.