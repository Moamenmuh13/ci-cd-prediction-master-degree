# Phase 2: Data Preparation, Feature Engineering, and Missing EDA Charts

Phase 1 EDA is complete with 6 figures. Now we (a) prepare and engineer
features for modeling, and (b) generate 4 critical EDA charts that were
skipped because the schema mapping consumed Phase 1's chart budget.

## Critical decisions established from Phase 1

1. Class imbalance: 89% success / 11% failure. Handle via class_weight (LR/RF) and scale_pos_weight (XGB) — no resampling.
2. High-cardinality categoricals: workflow_name (623), branch (1396) need Top-K bucketing (keep top 20, rest = "other").
3. Drop post-execution features: run_duration_sec, status (always "completed").
4. run_attempt has signal: convert to binary `is_retry`.
5. Numerical features are heavily right-skewed: apply log1p transformation.

## Task 1: Create `src/data_preparation.py`

### 1.1 Data Cleaning

Function `clean_data(df: pd.DataFrame) -> pd.DataFrame`:

- Drop `author_association` column (100% null)
- Drop `status` column (always "completed")
- Drop `updated_at` (post-execution)
- Fill `commit_author` null with "unknown" (82 rows, 0.8%)
- Strip whitespace from `commit_message`
- Drop rows where commit_message is null or empty (defensive)
- Convert `conclusion` to binary integer:
    - 1 if "failure" (POSITIVE class)
    - 0 if "success"
- Parse `created_at` and `commit_date` to datetime

### 1.2 Feature Engineering

Function `engineer_features(df: pd.DataFrame) -> pd.DataFrame`:

Add these derived features:

1. `commit_message_length` (character count)
2. `commit_message_word_count` (token count after split)
3. `was_truncated` = 1 if commit_message_length >= 999 else 0
4. `is_retry` = 1 if run_attempt > 1 else 0  (4% of data, 41% failure rate)
5. `log_lines_added` = np.log1p(lines_added)
6. `log_lines_deleted` = np.log1p(lines_deleted)
7. `log_total_changes` = np.log1p(total_changes)
8. `log_files_changed` = np.log1p(files_changed)
9. `add_delete_ratio` = (lines_added + 1) / (lines_deleted + 1)
10. `avg_change_per_file` = total_changes / (files_changed + 1)
11. `is_large_commit` = 1 if total_changes > median(total_changes) else 0
12. `is_many_files` = 1 if files_changed > 10 else 0
13. `commit_hour` = created_at.dt.hour
14. `commit_day_of_week` = created_at.dt.dayofweek (0=Mon, 6=Sun)
15. `is_weekend_commit` = 1 if commit_day_of_week >= 5 else 0
16. `is_off_hours_commit` = 1 if commit_hour < 8 or commit_hour >= 18 else 0

### 1.3 High-Cardinality Categorical Handling

Function `bucket_high_cardinality(df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame`:

For columns `workflow_name`, `branch`, `commit_author`:
- Keep top K most frequent values as is
- Replace all others with "other"

Print before/after cardinality for each.

### 1.4 Drop Post-Execution and Redundant Columns

Drop these columns explicitly (they should NOT be features):

```python
COLUMNS_TO_DROP_AFTER_FEATURE_ENG = [
    "run_id", "run_duration_sec", "run_attempt",   # post-execution
    "updated_at", "commit_sha",                     # IDs
    "created_at", "commit_date",                    # already extracted
    "commit_hour", "commit_day_of_week",            # encoded into is_weekend/is_off_hours
    "lines_added", "lines_deleted",                 # log versions kept
    "total_changes", "files_changed",               # log versions kept
]
```

### 1.5 Chronological Train/Test Split

Function `chronological_split(df, test_size=0.2) -> tuple`:

- Sort by `created_at` BEFORE dropping it
- First 80% = train, last 20% = test
- Verify class balance is similar in both splits
- Print date ranges
- Save to `data/processed/train.csv` and `data/processed/test.csv`

### 1.6 Feature Columns Constants

At module top:

```python
NUMERICAL_FEATURES: list[str] = [
    "log_lines_added", "log_lines_deleted",
    "log_total_changes", "log_files_changed",
    "commit_message_length", "commit_message_word_count",
    "add_delete_ratio", "avg_change_per_file",
]
CATEGORICAL_FEATURES: list[str] = [
    "repository", "workflow_name", "branch", "event",
    "commit_author",
]
BINARY_FEATURES: list[str] = [
    "was_truncated", "is_retry", "is_large_commit",
    "is_many_files", "is_weekend_commit", "is_off_hours_commit",
]
TEXT_FEATURE: str = "commit_message"
TARGET: str = "conclusion"  # 1=failure (POSITIVE), 0=success
```

## Task 2: Generate the 4 missing EDA charts

Use the existing ThesisPlotter (300 DPI, academic style).

### `phase1_fig_07_failure_rate_per_event.png`

Title: "Failure Rate per Workflow Trigger Event"
Caption: "Failure rate stratified by the GitHub Actions trigger event. Schedule events exhibit the highest failure rate (23.2%), followed by pull_request (13.3%) and push (8.6%), demonstrating that the event type carries strong discriminative signal for failure prediction."

Horizontal bar chart: event on Y, failure rate (%) on X, sorted descending.
Show counts inside each bar. Filter to events with >= 50 occurrences.

### `phase1_fig_08_top_words_success_vs_failure.png`

Title: "Most Discriminative Words in Commit Messages by Outcome"
Caption: "Top 20 words appearing more frequently in commit messages of failed builds (red) versus successful builds (green), computed using TF-IDF feature relevance. The divergent vocabularies — particularly the prominence of 'fix', 'bug', 'revert', and 'broken' in failure messages — empirically validate the use of NLP-based text features for prediction."

This is THE most important chart for the NLP claim. Implementation:
1. Lower-case commit_messages, remove URLs and SHA hashes, basic tokenization
2. Compute relative frequency of each word in failure vs success class
3. Rank by frequency_in_failures - frequency_in_successes (most "failure-typical")
4. Plot top 20 most-failure-typical and top 20 most-success-typical side by side
5. Use 2-panel horizontal bar chart (failure on left, success on right)

Filter stopwords. Min word length 3. Min occurrence 10.

### `phase1_fig_09_commit_msg_length_by_outcome.png`

Title: "Commit Message Length Distribution by Outcome"
Caption: "Distribution of commit message lengths (log scale) stratified by build outcome. The KDE curves are nearly indistinguishable, suggesting that message length alone is not a strong predictor — content matters more than length."

Overlaid KDE plots (success vs failure), log-scale X axis.

### `phase1_fig_10_temporal_failure_trend.png`

Title: "Build Failure Rate Over Time (Monthly)"
Caption: "Monthly evolution of the build failure rate across the data collection window. The temporal variability justifies the use of chronological train/test splitting rather than random splitting to prevent temporal leakage."

Line chart: X = month, Y = failure rate (%). Add a horizontal dashed line for overall mean failure rate.
Annotate the highest and lowest months.

Append the 4 captions to `figures/captions.md`.

## Task 3: Generate Phase 2 validation charts (new)

### `phase2_fig_01_engineered_features_distribution.png`

Title: "Distribution of Engineered Features"
Caption: "Distribution of the 16 engineered features. Log-transformed numerical features show approximately Gaussian shape suitable for linear models, while binary indicators capture domain heuristics relevant to CI/CD failure analysis."

4x4 grid of histograms (or bar charts for binary).

### `phase2_fig_02_correlation_after_engineering.png`

Title: "Pearson Correlation Matrix After Feature Engineering"
Caption: "Correlation structure after feature engineering. The log-transformed features preserve the signal of their raw counterparts while reducing skew, and the derived ratio features introduce new low-correlation predictors."

Heatmap of all NUMERICAL_FEATURES + BINARY_FEATURES (NOT the categorical).

### `phase2_fig_03_top_categorical_buckets.png`

Title: "Top Categorical Values After Bucketing"
Caption: "Distribution of the top 20 values for workflow_name and branch after Top-K bucketing, with all other values aggregated into 'other'. This reduces cardinality from 623 and 1,396 unique values to 21 each, making one-hot encoding tractable."

2 horizontal bar charts side by side.

Append captions.

## Task 4: Create `src/run_phase2.py`

Orchestrator:
1. Load raw data
2. Run clean_data
3. Run engineer_features
4. Run bucket_high_cardinality
5. Drop unneeded columns
6. Run chronological_split
7. Generate the 4 missing Phase 1 charts
8. Generate the 3 Phase 2 charts
9. Save processed data
10. Save `results/phase2_summary.json` with stats
11. Print comprehensive console summary

## After completion, give me:

1. Train/test sizes and class distribution in each
2. Top 10 words MORE common in failures (from the words chart)
3. Failure rates per event (table)
4. Top 5 repositories by failure rate (table)
5. Cardinality before/after Top-K bucketing for workflow_name and branch
6. Final feature counts: how many numerical, categorical (after bucketing), binary, text
7. List of all new figures generated
8. Any warnings or surprises

## Important constraints

- All code in English with type hints
- Use logging in modules
- Set random_state=42 everywhere
- Don't modify raw data
- Reuse ThesisPlotter
- Verify chronological split doesn't break class balance (within ±2pp)