# Predicting CI/CD Pipeline Build Failures Using Machine Learning

MSc Software Engineering thesis project — Moamen Mohamed Aly Hussein.

Predicts GitHub Actions workflow failure from **pre-execution** commit features
(commit metadata + the commit message), using a four-branch hybrid pipeline:
numerical scaling, one-hot categoricals, binary flags, and TF-IDF text, fused via
`ColumnTransformer` and consumed by Logistic Regression / Random Forest / XGBoost.

## Dataset

9,772 real workflow runs from 18 active open-source repositories, collected from
the GitHub Actions API. 89.0% success / 11.0% failure. **`failure` is the positive
class.** Full profile: [`cicd-failure-prediction/DATA_PROFILE.md`](cicd-failure-prediction/DATA_PROFILE.md).

## Reported results

| Model | Threshold | Failure F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.06 (tuned) | 0.5924 | 0.884 | 0.587 |
| Random Forest | 0.5 | 0.4753 | 0.861 | 0.473 |
| Logistic Regression | 0.5 | 0.4603 | 0.875 | 0.499 |

> **Read [`cicd-failure-prediction/REVIEW_FINDINGS.md`](cicd-failure-prediction/REVIEW_FINDINGS.md) before citing these numbers.**
> An independent review found duplicate-commit leakage across the stratified
> split (88.3% of test rows share a commit with train). On a commit-grouped
> split the honest figure is **F1 0.533 / PR-AUC 0.540**. Two further issues —
> a mis-sorted "chronological" split and threshold selection on the test set —
> are documented there with measurements and fixes.

## Picking up the work

[`cicd-failure-prediction/HANDOFF.md`](cicd-failure-prediction/HANDOFF.md) —
current state, prioritized work queue with file:line references, and settled
decisions. Read it before starting a session on a new machine.

## Layout

```
cicd-failure-prediction/     the pipeline (see its CLAUDE.md for conventions)
  src/                       collection → preparation → training → threshold tuning
  data/raw/                  immutable source CSV (tracked)
  results/                   every reported metric as JSON
  figures/                   19 figures at 300 DPI + captions.md
  models/                    trained pipelines + metadata sidecars
final-version-doc/files/     thesis, defense deck, Q&A guide, markdown source
dataset/                     reference papers and auxiliary datasets
phase0.md … phase5.md        original phase specifications (intent, not a spec of the code)
```

## Reproducing

```bash
cd cicd-failure-prediction
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_phase2_5    # prepare + split (writes data/processed/)
.venv/bin/python -m src.run_phase4      # train, evaluate, ablation
.venv/bin/python -m src.run_phase5      # threshold optimization
```

Seed is fixed at 42 throughout; the reported metrics reproduce exactly.
Re-collecting the dataset requires `export GITHUB_TOKEN=...` before
`src/collect_github_data.py`.

## License / use

Academic work submitted for an MSc degree. The `dataset/` directory contains
third-party published papers included for reference only.
