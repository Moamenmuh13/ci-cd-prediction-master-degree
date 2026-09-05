# Phase 2.5: Data Quality Refinement

We just reviewed the Phase 2 outputs and found two issues that need fixing BEFORE moving to model training:

1. **Discriminative vocabulary contains author names and project-specific tokens** (e.g., "anna", "kamat", "yagiz", "trivikr"). The text model is leaking author identity instead of learning commit content.

2. **Severe multicollinearity remains** in the feature set:
   - lines_added ↔ total_changes = 0.99
   - lines_added ↔ lines_change_ratio = 0.99
   - commit_message_length ↔ commit_message_word_count = 0.95
   - run_attempt ↔ is_retry = 0.96

Also: `run_attempt` and `run_duration_sec` are POST-EXECUTION features and must be excluded from the model entirely (they leak the outcome).

## Task 1: Refine Text Preprocessing

Update `src/data_preparation.py` to add a function:

```python
def clean_commit_message_for_nlp(text: str) -> str:
    """Aggressive cleaning to prevent author/project leakage in TF-IDF."""
```

The function must:

1. Lowercase
2. Remove URLs (http://, https://, www.)
3. Remove email addresses
4. Remove SHA hashes (40-char hex strings, 8-char hex strings — common in commits)
5. Remove PR/issue references: patterns like `#123`, `(#4567)`, `gh-12345`, `GH-12345`
6. Remove version numbers: patterns like `1.2.3`, `v1.2`, `2024.1.0`
7. Remove `Co-authored-by:` and `Signed-off-by:` lines entirely
8. Remove `Bumps X from Y to Z` boilerplate (Dependabot pattern)
9. Remove file paths (anything matching `[a-z_/]+\.[a-z]+`)
10. Remove non-alphabetic tokens (keep letters and spaces only)
11. Collapse multiple whitespace into single space
12. Strip

Apply this function to `commit_message` BEFORE TF-IDF vectorization. Save the cleaned column as `commit_message_clean` and KEEP the original `commit_message` for inspection.

## Task 2: Build a Stoplist of Author/Project Identifiers

Create a function that:

1. Scans commit_author column, extracts unique GitHub logins
2. Scans repository column, extracts org and repo names
3. Combines them into a stoplist (e.g., "anna", "kamat", "facebook", "react", "vercel", "elasticsearch")
4. Also adds common bot signatures: "dependabot", "renovate", "bors", "github-actions"
5. Removes ALL these tokens from `commit_message_clean`

Print the size of the generated stoplist.

## Task 3: Final Feature Set (FIX multicollinearity)

Update the feature constants in `src/data_preparation.py` to this EXACT final set:

```python
NUMERICAL_FEATURES = [
    "log_lines_added",        # representative of commit size
    "log_lines_deleted",      # independent signal
    "log_files_changed",      # independent signal
    "commit_message_length",  # representative of message verbosity
    "avg_lines_per_file",     # complexity metric
]
CATEGORICAL_FEATURES = [
    "repository",
    "workflow_name",
    "branch",
    "event",
]
BINARY_FEATURES = [
    "is_large_commit",
    "is_many_files",
    "is_weekend_commit",
    "is_off_hours_commit",
    "is_bot_author",
    "was_truncated",
]
TEXT_FEATURE = "commit_message_clean"  # USE THE CLEANED VERSION
TARGET = "conclusion"
```

EXPLICITLY DROP these columns from the processed data (they cause leakage or redundancy):

```python
LEAKAGE_OR_REDUNDANT = [
    # POST-EXECUTION (leakage):
    "run_duration_sec", "run_attempt", "is_retry",
    "status", "updated_at",
    # REDUNDANT (multicollinearity):
    "total_changes", "lines_change_ratio",
    "commit_message_word_count",
    "log_total_changes",
    # IDs / non-features:
    "run_id", "commit_sha",
    # Already encoded:
    "commit_hour", "commit_day_of_week",
    "lines_added", "lines_deleted", "files_changed",
    # Raw text replaced by cleaned version:
    "commit_message",
    # Personal identifier (use is_bot_author instead):
    "commit_author",
]
```

## Task 4: Create BOTH split strategies

Update `src/data_preparation.py` to produce TWO sets of splits:

```python
def stratified_split(df, test_size=0.2, random_state=42):
    """Stratified random split preserving class ratio. PRIMARY evaluation."""
    # Use sklearn.model_selection.train_test_split with stratify=df['conclusion']
    # Save to: data/processed/train_stratified.csv, test_stratified.csv

def chronological_split(df, test_size=0.2):
    """Chronological split. SECONDARY evaluation (deployment realism)."""
    # Sort by commit_date, take last 20% as test
    # Save to: data/processed/train_chronological.csv, test_chronological.csv
```

Print class distribution for both splits side by side.

## Task 5: Regenerate fig_10 with Cleaned Vocabulary

Re-run the discriminative vocabulary analysis on the CLEANED text and regenerate `fig_10_discriminative_vocabulary.png`.

Expected outcome: failure tokens should now contain meaningful words like "fix", "broken", "error", "fail", "revert", "hotfix", "regression", "bug", "rollback". Success tokens should contain words like "add", "update", "implement", "test", "doc", "refactor".

If the vocabulary STILL contains author names or project-specific gibberish, add an iteration: print the top 50 failure tokens, identify any that are clearly identifiers (manual review by the script — look for tokens with no English-dictionary match), and add them to the stoplist. Repeat until clean.

## Task 6: Run and Summarize

Create `src/run_phase2_5.py` that runs everything and prints:

1. Stoplist size and top 20 entries
2. Sample of 5 cleaned commit messages (before / after)
3. Final feature count by branch:
   - Numerical: N
   - Categorical: N (after Top-K bucketing)
   - Binary: N
   - Text: 1 (TF-IDF)
4. Class distribution for stratified split (train / test)
5. Class distribution for chronological split (train / test)
6. Top 20 FAILURE tokens (after cleaning) — must be meaningful words
7. Top 20 SUCCESS tokens (after cleaning) — must be meaningful words

## Constraints

- All code in English with type hints
- Reuse existing ThesisPlotter
- Don't run any training in this phase — preparation only
- Set random_state=42 everywhere