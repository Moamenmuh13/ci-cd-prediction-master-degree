# CI/CD Build-Failure Prediction — MSc thesis project

Predict GitHub Actions workflow failure from **pre-execution** commit features.
Binary target `conclusion`; **`failure` is the positive class**. Labels are
strings (`"success"` / `"failure"`), not 0/1 — XGBoost is wrapped in
`LabelEncoderForBinary` (`src/hybrid_pipeline.py`) to keep the fit/predict
contract uniform across the three estimators.

Read [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) and [DATA_PROFILE.md](DATA_PROFILE.md)
before changing anything. They encode verified facts that are expensive to
re-derive and easy to get wrong.

## Environment

- Python: `.venv/bin/python` — **never** bare `python` (no pandas on the system
  interpreter). The venv has pandas, scikit-learn 1.8.0, xgboost 3.2.0.
- Run modules from this directory: `.venv/bin/python -m src.run_phase4`
- `random_state = 42` everywhere. Never modify `data/raw/`.

## Layout

```
src/
  collect_github_data.py   GitHub API → data/raw/github_actions_real.csv
  data_preparation.py      cleaning, feature engineering, text cleaning, splits
  hybrid_pipeline.py       4-branch ColumnTransformer + LR / RF / XGB
  train_evaluate.py        metrics, ablation, business model, best-model selection
  threshold_optimization.py  threshold sweep
  eda.py visualization.py  ThesisPlotter, 300 DPI figures
  run_phase{2,2_5,3,4,5}.py  orchestrators
data/raw/        immutable source CSV
data/processed/  cicd_prepared.csv + {train,test}_{stratified,chronological}.csv
results/         every reported metric as JSON
figures/         19 PNGs + captions.md
models/          *.joblib + metadata sidecars
```

Phase specs live at the repository root (`phase0.md` … `phase5.md`). They are
**intent**, not a description of the code — several have drifted (F-6).

## Ground truth — verified, do not recompute unless asked

9,772 runs · 18 repositories · 89.03 / 10.97 success/failure · **2,835 unique
commits** · `created_at` spans 2025-11-25 → 2026-05-29.

Reported headline: XGBoost @ threshold 0.06 → failure F1 **0.5924** (stratified
test), ROC-AUC 0.884, PR-AUC 0.587. This reproduces exactly from raw data.

## Known defects — assume true, do not re-litigate

Full detail and measurements in REVIEW_FINDINGS.md.

1. **F-1 Duplicate-commit leakage.** 3.45 runs per commit; 88.3% of stratified
   test rows share a commit with train. On a commit-grouped split the honest
   number is F1 0.533 / PR-AUC 0.540. **Any new split must group on `commit_sha`.**
2. **F-2 `chronological_split()` sorts by `commit_date`, not `created_at`.** Test
   window is 11 hours; 95.8% of test rows ran *before* train's last run. It is
   not a temporal holdout. Thesis TC-001 validates the wrong column.
3. **F-3 Threshold 0.06 was selected on the test set and reported on it.** No
   validation fold; the +27 pp is partly a selection effect.
4. **F-4 Business model is unsupported** — hardcodes a 30% failure rate against
   an observed 11%, omits false-alarm cost, and the "optimized" model reports
   *lower* savings ($382,802) than the unoptimized one ($428,854).
5. **F-5 Ablation lacks `categorical_only`** — the configuration that would show
   whether repository identity, not commit content, drives predictions.
   Structured-only (F1 0.379) already beats the full hybrid (0.322).
6. **F-6 Spec drift**: `is_many_files` uses the median (spec: `> 10`),
   `is_off_hours` uses `< 6` (spec: `< 8`), `branch` is bucketed top-15 while
   captions claim 21.
7. **F-7** Medians, bucket vocabularies and the stoplist are fit on train + test.
8. **F-8** `is_off_hours_commit` / `is_weekend_commit` use UTC across globally
   distributed projects — near-noise as defined.
9. **F-9** `files_changed` is censored at 300 by the GitHub API (110 rows).

## Working rules

- **Quote metrics from `results/*.json`, never from thesis prose.** The two have
  already diverged.
- A changed number must propagate to four places: `results/*.json`, the figure,
  its entry in `figures/captions.md`, and
  `../final-version-doc/files/4_Thesis_Source_Markdown.md`. State which of the
  four you updated.
- **Do not retrain without asking.** A full Phase 4 run overwrites `models/` and
  every `results/phase4_*.json`.
- **Flag any metric that improves without a mechanism.** This project has already
  been burned once by leakage; a jump with no causal story is a bug until proven
  otherwise.
- Report failures with the actual output. Never present a number as clean when it
  came from the stratified split without saying so.
- Do not add `run_duration_sec`, `run_attempt`, `is_retry`, `status` or
  `updated_at` to any feature set — they are post-execution and the strictly
  pre-execution framing is the thesis's central claim.

## Thesis prose conventions

Academic register: no contractions, no bullet lists inside chapter body text,
IEEE-numbered citations, figures referenced as "Figure 7.4" in text. Edit
`4_Thesis_Source_Markdown.md`; the `.docx` is a derived artifact.

## Do not commit or upload

`.venv/`, `*.joblib`, any CSV over 1 MB, `~$*.docx` lock files, and
**`phase0.md` in its current state — line 13 contains a live GitHub PAT (F-10).**
