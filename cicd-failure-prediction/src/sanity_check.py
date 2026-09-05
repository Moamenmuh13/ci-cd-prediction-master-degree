"""Sanity check for the binary hybrid pipelines on a small sample.

Not a training/evaluation run — the goal is to confirm that the four-branch
preprocessor + the three classifier flavours fit and predict end-to-end on
1,000 rows of the real GitHub Actions data, with the correct output shape
and serialisability before Phase 4 commits real compute.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_preparation import TARGET  # noqa: E402
from src.hybrid_pipeline import (  # noqa: E402
    LabelEncoderForBinary,
    get_all_pipelines,
    prepare_features_targets,
)
from src.utils import PROCESSED_DATA_DIR, get_logger  # noqa: E402


_LOGGER = get_logger(__name__)


_TRAIN_SAMPLE_SIZE = 1_000
_TEST_SAMPLE_SIZE = 200
_RANDOM_STATE = 42


def _inner_pipeline(pipeline: Any) -> Any:
    return (
        pipeline.estimator
        if isinstance(pipeline, LabelEncoderForBinary)
        else pipeline
    )


def _feature_counts_per_branch(pipeline: Any) -> dict[str, int]:
    preprocessor = _inner_pipeline(pipeline).named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    counts = {"numerical": 0, "categorical": 0, "binary": 0, "text": 0}
    for name in feature_names:
        prefix = name.split("__", 1)[0] if "__" in name else "other"
        if prefix in counts:
            counts[prefix] += 1
    return counts


def _is_serializable(pipeline: Any) -> bool:
    buffer = io.BytesIO()
    try:
        joblib.dump(pipeline, buffer)
        buffer.seek(0)
        joblib.load(buffer)
    except Exception as exc:  # pragma: no cover — diagnostic only
        _LOGGER.warning("Pipeline failed joblib round-trip: %s", exc)
        return False
    return True


def _sample_predictions(
    y_pred: Any, y_true: pd.Series, n: int = 3
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    limit = min(n, len(y_true))
    for i in range(limit):
        rows.append(
            {
                "row": i,
                "predicted": str(y_pred[i]),
                "actual": str(y_true.iloc[i]),
            }
        )
    return rows


def run_sanity_check() -> dict[str, Any]:
    train_path = PROCESSED_DATA_DIR / "train_stratified.csv"
    test_path = PROCESSED_DATA_DIR / "test_stratified.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Phase 2.5 splits missing — run `python src/run_phase2_5.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_sample = train_df.sample(
        n=min(_TRAIN_SAMPLE_SIZE, len(train_df)), random_state=_RANDOM_STATE
    ).reset_index(drop=True)
    test_sample = test_df.head(_TEST_SAMPLE_SIZE).reset_index(drop=True)

    x_train, y_train = prepare_features_targets(train_sample)
    x_test, y_test = prepare_features_targets(test_sample)

    report: dict[str, Any] = {
        "train_sample_rows": int(len(x_train)),
        "test_sample_rows": int(len(x_test)),
        "feature_counts": None,
        "models": {},
    }

    for name, pipeline in get_all_pipelines().items():
        _LOGGER.info("Fitting %s on %d sample rows ...", name, len(x_train))
        t0 = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = pipeline.predict(x_test)
        predict_time = time.perf_counter() - t0

        counts = _feature_counts_per_branch(pipeline)
        if report["feature_counts"] is None:
            report["feature_counts"] = counts

        report["models"][name] = {
            "fit_time_sec": round(fit_time, 3),
            "predict_time_sec": round(predict_time, 4),
            "feature_total": sum(counts.values()),
            "feature_counts": counts,
            "serializable": _is_serializable(pipeline),
            "sample_predictions": _sample_predictions(y_pred, y_test, n=3),
        }
        _LOGGER.info(
            "%s ready — fit=%.2fs predict=%.3fs features=%d",
            name, fit_time, predict_time, sum(counts.values()),
        )

    return report


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    header_line = "  ".join(
        f"{str(headers[i]):<{widths[i]}}" for i in range(len(headers))
    )
    print(header_line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(f"{str(row[i]):<{widths[i]}}" for i in range(len(headers))))


def print_results(report: dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("Phase 3 — Sanity Check Results (binary 'conclusion' target)")
    print("=" * 78)
    print(
        f"Train sample: {report['train_sample_rows']} rows  |  "
        f"Test sample: {report['test_sample_rows']} rows"
    )

    print("\nModel summary")
    print("-" * 78)
    rows: list[list[str]] = []
    for name, info in report["models"].items():
        rows.append(
            [
                name,
                f"{info['fit_time_sec']:.3f}",
                f"{info['predict_time_sec']:.4f}",
                str(info["feature_total"]),
                "yes" if info["serializable"] else "no",
            ]
        )
    _print_table(
        ["Model", "Fit (s)", "Predict (s)", "# features", "Serializable"], rows
    )

    print("\nFeature breakdown per branch")
    print("-" * 78)
    for branch, count in report["feature_counts"].items():
        print(f"  {branch:<14}: {count:>5} features")
    print(f"  {'TOTAL':<14}: {sum(report['feature_counts'].values()):>5} features")

    print("\nSample predictions on the first 3 test rows")
    print("-" * 78)
    for name, info in report["models"].items():
        print(f"\n  [{name}]")
        for sp in info["sample_predictions"]:
            marker = "MATCH " if sp["predicted"] == sp["actual"] else "MISS  "
            print(
                f"    row {sp['row']}:  {marker} "
                f"pred={sp['predicted']:<8} actual={sp['actual']}"
            )


if __name__ == "__main__":
    report = run_sanity_check()
    print_results(report)
