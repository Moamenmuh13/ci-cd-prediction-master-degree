# Phase 3: Build Hybrid Pipeline for Binary Classification

We just completed Phase 2.5 with clean features ready for modeling. Now we build the pipeline architecture but DO NOT do full training (that's Phase 4).

## Critical context

- TARGET = "conclusion" (binary: 1=failure, 0=success)
- 5 numerical, 4 categorical, 6 binary, 1 text feature
- Class imbalance: 89% success / 11% failure
- Two splits available: stratified (primary) and chronological (secondary)

## Task 1: Update `src/hybrid_pipeline.py`

REPLACE the existing multi-output pipeline with a clean binary classification version.

### 1.1 Imports and constants

```python
from src.data_preparation import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
    TEXT_FEATURE, TARGET,
)
RANDOM_SEED = 42
```

### 1.2 Preprocessor

```python
def build_preprocessor() -> ColumnTransformer:
    """Hybrid preprocessor with 4 parallel branches."""
```

- Numerical branch: StandardScaler on NUMERICAL_FEATURES
- Categorical branch: OneHotEncoder(handle_unknown='ignore', sparse_output=True) on CATEGORICAL_FEATURES
- Binary branch: passthrough on BINARY_FEATURES
- Text branch: TfidfVectorizer on TEXT_FEATURE with:
    - max_features=3000
    - ngram_range=(1, 2)
    - min_df=5
    - max_df=0.95
    - sublinear_tf=True
    - lowercase=True (already lowercased but defensive)
- sparse_threshold=0.3, n_jobs=-1

### 1.3 Three classifier pipelines

```python
def build_logistic_regression_pipeline() -> Pipeline:
    # LogisticRegression(max_iter=1000, class_weight='balanced',
    #                    solver='liblinear', C=1.0, random_state=42)

def build_random_forest_pipeline() -> Pipeline:
    # RandomForestClassifier(n_estimators=200, max_depth=25,
    #                        min_samples_split=10, min_samples_leaf=4,
    #                        class_weight='balanced', n_jobs=-1, random_state=42)

def build_xgboost_pipeline() -> Pipeline:
    # XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
    #               subsample=0.85, colsample_bytree=0.85,
    #               reg_alpha=0.1, reg_lambda=1.0,
    #               scale_pos_weight=8.12,
    #               eval_metric='logloss', n_jobs=-1, random_state=42,
    #               tree_method='hist')
```

These are SINGLE-OUTPUT binary classifiers. NO MultiOutputClassifier wrapper this time.

### 1.4 Helper functions

```python
def get_all_pipelines() -> dict[str, Pipeline]:
    """Return all 3 pipelines in canonical order."""

def prepare_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) from a prepared dataframe."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + [TEXT_FEATURE]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y
```

## Task 2: Sanity Check

Create `src/sanity_check.py` that:

1. Loads `data/processed/train_stratified.csv` (use stratified as primary)
2. Takes a 1,000-row sample from train, 200 from test
3. For each of the 3 pipelines:
   - Fit on the sample
   - Predict on the test sample
   - Print: model name, fit time, predict time, total transformed feature count
4. Verify the preprocessor output shape
5. Print per-branch feature count breakdown

## Task 3: Architecture Diagram

Regenerate `fig_11_hybrid_architecture_diagram.png` for the BINARY classification version.

The diagram should show:

```
[Raw Commit Data]
       |
       v
[ColumnTransformer]
   /    |    |    \
  v     v    v     v
[Num] [Cat] [Bin] [Text]
[Scaler][OHE][Pass][TFIDF]
   \    |    |    /
    v   v    v   v
  [Fused Feature Matrix (sparse)]
       |
       v
[Binary Classifier: LR / RF / XGBoost]
       |
       v
[P(failure) in [0, 1]]
       |
       v
[Threshold = 0.5]
       |
       v
[Output: success / failure]
```

Use rounded boxes with branch-specific colors:
- Numerical: blue
- Categorical: green
- Binary: orange
- Text: purple

Include hyperparameter labels in each branch box (StandardScaler, OneHotEncoder, etc.).

Caption: "Hybrid Machine Learning Pipeline architecture for binary classification of CI/CD workflow outcomes. Four parallel preprocessing branches (numerical scaling, categorical one-hot encoding, binary passthrough, and TF-IDF text vectorization) are fused via ColumnTransformer into a unified sparse feature matrix, then consumed by one of three comparative binary classifiers."

Update `figures/captions.md` accordingly.

## Task 4: Create `src/run_phase3.py`

Orchestrator that:
1. Runs sanity_check
2. Regenerates architecture diagram
3. Confirms all 3 pipelines are joblib-serializable
4. Prints comprehensive summary

## After completion give me

1. Sanity check results (3 models × {fit time, predict time, accuracy on small sample})
2. Total transformed feature matrix shape
3. Per-branch feature counts
4. Confirmation each pipeline serializes correctly
5. Any errors or warnings

## Constraints

- Binary classification (NOT multi-output)
- No full training in this phase
- All code in English with type hints
- Reuse the new data preparation constants