# Phase 1: Comprehensive EDA + Data Preparation on REAL GitHub Actions Data

We just collected 9,772 real workflow runs from 18 popular open-source repositories on GitHub. Now we do thorough EDA and prepare the data for modeling.

## Context

- Dataset: `data/raw/github_actions_real.csv`
- Target column: `conclusion` (binary: success / failure)
- Class imbalance: 89% success, 11% failure
- Mix of numerical, categorical, and textual features
- Real commit messages with diverse vocabulary

## Task 1: Comprehensive EDA

Create `src/full_eda.py` that loads `data/raw/github_actions_real.csv` and produces:

### Console output (saved to `results/eda_report.txt`):

1. Dataset overview
   - Shape, dtypes, missing values per column
   - Date range of data (min/max created_at)
   - Number of unique values per categorical column

2. Target analysis
   - Conclusion distribution (success vs failure counts and %)
   - Conclusion distribution PER REPOSITORY (which repos have higher failure rates?)
   - Conclusion distribution PER EVENT (push vs pull_request)

3. Numerical features analysis
   - Statistics for: lines_added, lines_deleted, total_changes, files_changed, run_duration_sec
   - Compare numerical features for success vs failure (mean, median)
   - Check correlations between numerical features

4. Textual features analysis
   - Commit message length statistics (min, max, median, mean, std)
   - How many messages are at the 1000-char cap (truncated)?
   - Top 20 most common WORDS in success vs failure commit messages
   - Average message length per conclusion class

5. Temporal analysis
   - Distribution of runs over time (by month)
   - Failure rate trend over time

### Generate 10 publication-quality charts (300 DPI):

Use the existing ThesisPlotter. Save to `figures/` with the `phase1_` prefix.

1. `phase1_fig_01_target_distribution.png`
   Title: "CI/CD Build Outcome Distribution in Real GitHub Actions Data"
   Bar chart of conclusion (success/failure) with counts and percentages.
   Caption: "Distribution of 9,772 workflow runs collected from 18 active open-source repositories. The 89:11 success-to-failure ratio reflects the typical class imbalance observed in production CI/CD environments."

2. `phase1_fig_02_failure_rate_per_repo.png`
   Title: "Build Failure Rate per Repository (Top 18)"
   Horizontal bar chart, repos sorted by failure rate (descending).
   Caption: "Per-repository failure rates reveal substantial heterogeneity across projects, with some repositories exhibiting failure rates above 25% and others below 5%. This variability validates the use of repository as a categorical feature in the model."

3. `phase1_fig_03_lines_changed_by_outcome.png`
   Title: "Code Change Volume by Build Outcome"
   Boxplot of total_changes split by conclusion. Use log scale on Y axis.
   Caption: "Box-and-whisker comparison of code change volume between successful and failed builds. Failed builds exhibit a wider distribution and higher upper quartile, supporting the hypothesis that larger commits are more failure-prone."

4. `phase1_fig_04_files_changed_by_outcome.png`
   Title: "Files Modified per Commit by Build Outcome"
   Same style as fig_03 but for files_changed.
   Caption: "Distribution of files modified per commit, by outcome. Larger and broader changes correlate with elevated failure risk."

5. `phase1_fig_05_correlation_heatmap.png`
   Title: "Pearson Correlation Matrix of Numerical Features"
   Heatmap of correlations between: lines_added, lines_deleted, total_changes, files_changed, run_duration_sec, commit_message_length.
   Caption: "Pearson correlation matrix of the available numerical features. The strong positive correlations between lines_added, lines_deleted, total_changes, and files_changed are expected and indicate that any one of them can serve as a proxy for commit size."

6. `phase1_fig_06_commit_msg_length_by_outcome.png`
   Title: "Commit Message Length Distribution by Build Outcome"
   Overlaid histograms (or KDE plot) of commit_message_length split by conclusion.
   Caption: "Commit message length distribution stratified by outcome. The visible peak at 1,000 characters reflects the truncation cap applied during data collection."

7. `phase1_fig_07_event_type_distribution.png`
   Title: "Workflow Triggers and Their Failure Rates"
   Stacked bar chart: event type (push/pull_request/etc.) split by success/failure.
   Caption: "Distribution of workflow trigger events with their corresponding outcome breakdown. Pull request events show measurably different failure characteristics from push events."

8. `phase1_fig_08_top_words_success_vs_failure.png`
   Title: "Most Frequent Words in Commit Messages by Outcome"
   Two horizontal bar charts side by side: top 20 words in success commits vs top 20 in failure commits.
   Use NLTK or simple tokenization. Filter common English stopwords.
   Caption: "Top 20 most frequent words in commit messages for successful and failed builds. The divergent vocabularies, particularly the prominence of 'fix', 'bug', and 'revert' in failed builds, support the use of NLP-based text features for prediction."

9. `phase1_fig_09_temporal_failure_trend.png`
   Title: "Build Failure Rate Over Time"
   Line chart: monthly failure rate over the collection period.
   Caption: "Monthly evolution of the build failure rate across the data collection window. The variability over time motivates the use of chronological train-test splitting rather than random splitting to prevent temporal leakage."

10. `phase1_fig_10_run_duration_by_outcome.png`
    Title: "Workflow Run Duration by Outcome"
    Boxplot of run_duration_sec by conclusion (log scale Y).
    Caption: "Workflow run duration distribution by outcome. NOTE: This is a POST-EXECUTION feature and will not be used as a predictor (only shown here for descriptive purposes)."

Append all 10 captions to `figures/captions.md`.

## Task 2: Data Preparation

Create `src/data_preparation.py`:

### 2.1 Data Cleaning

Function `clean_data(df: pd.DataFrame) -> pd.DataFrame`:
- Drop column `author_association` (100% NaN per Phase 0)
- Fill `commit_author` NaN with the string "unknown"
- Add feature `was_truncated`: 1 if commit_message length == 1000, else 0
- Add feature `commit_message_length` (chars)
- Add feature `commit_message_word_count` (tokens after split)
- Convert `conclusion` to binary: success=0, failure=1 (failure is the POSITIVE class)
- Parse `created_at` and `commit_date` to datetime
- Drop rows where commit_message is null OR empty
- Strip whitespace from commit_message

### 2.2 Feature Engineering

Function `engineer_features(df: pd.DataFrame) -> pd.DataFrame`:

Add these derived features:

1. `is_large_commit`: 1 if total_changes > median, else 0
2. `is_many_files`: 1 if files_changed > 10, else 0
3. `add_delete_ratio`: lines_added / (lines_deleted + 1)
4. `avg_change_per_file`: total_changes / (files_changed + 1)
5. `commit_hour`: hour from commit_date
6. `commit_day_of_week`: 0-6
7. `is_weekend_commit`: 1 if Saturday/Sunday, else 0
8. `is_off_hours_commit`: 1 if hour < 8 OR hour >= 18

Critical: drop run_duration_sec, status, run_attempt, updated_at from features  used for prediction (these are post-execution and would cause data leakage).

### 2.3 Train/Test Split

Function `chronological_split(df, test_size=0.2) -> Tuple`:
- Sort by `created_at` ascending
- First 80% = train, last 20% = test
- Print class distribution per split
- Save to `data/processed/train.csv` and `data/processed/test.csv`

Document this prevents temporal leakage (model never sees future data during training).

### 2.4 Feature columns definition

At module level, define:

```python
NUMERICAL_FEATURES = [
    "lines_added", "lines_deleted", "total_changes", "files_changed",
    "commit_message_length", "commit_message_word_count",
    "add_delete_ratio", "avg_change_per_file"
]
CATEGORICAL_FEATURES = [
    "repository", "workflow_name", "branch", "event", "commit_author"
]
BINARY_FEATURES = [
    "was_truncated", "is_large_commit", "is_many_files",
    "is_weekend_commit", "is_off_hours_commit"
]
TEXT_FEATURE = "commit_message"
TARGET = "conclusion"  # 1 = failure, 0 = success
EXCLUDED_LEAKAGE = [
    "run_id", "run_duration_sec", "status", "run_attempt",
    "updated_at", "commit_sha", "commit_date", "created_at",
    "commit_hour", "commit_day_of_week"  # these were used in feature eng but no longer needed
]
```

Note: `commit_hour` and `commit_day_of_week` are encoded into `is_weekend_commit` and `is_off_hours_commit`, so we drop the raw versions.

## Task 3: Create `src/run_phase1.py`

The orchestrator that:
1. Loads raw data
2. Runs `full_eda.py` logic
3. Runs cleaning + feature engineering
4. Runs chronological split
5. Saves processed train/test CSVs
6. Prints a comprehensive summary

## After completion, give me:

1. The contents of `results/eda_report.txt` (truncated if huge)
2. The contents of `figures/captions.md` (just the new 10 entries)
3. Train/test split details:
   - Train size, test size
   - Class distribution in each (should be similar)
   - Date ranges of each
4. The top 5 repositories by failure rate
5. The top 5 commit message tokens that appear MORE in failures than in successes
6. List all 10 figures generated
7. Any warnings or surprises

## Important constraints

- All code in English, type hints, PEP 8
- Use logging in modules; print is OK in standalone scripts
- ThesisPlotter is already in src/visualization.py — reuse it
- Set random seed 42 everywhere relevant
- Don't modify raw data