"""Phase 1 exploratory data analysis for the *real* GitHub Actions dataset.

The synthetic dataset's three targets (failure_stage / severity /
rollback_triggered) do not exist in the GitHub-collected data, so this EDA
adapts the original Phase 1 plan to the new schema while preserving its
spirit:

* one figure for the (now binary) target distribution,
* one figure for the categorical feature that matters most for thesis context
  (workflow ``event``),
* one figure for the per-repository row counts (already informative thanks to
  Phase 0's deliberate stratification),
* one figure for the distribution of the four core numerical features,
* one figure for the Pearson correlation matrix of those features,
* one figure for the conclusion × repository cross-tabulation, which is the
  most thesis-relevant joint distribution we have.

Produces ``results/eda_report.txt`` and writes six 300-DPI PNGs to
``figures/``. Captions are appended to ``figures/captions.md``.
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    FIGURES_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
    ensure_dir,
)
from src.visualization import ThesisPlotter  # noqa: E402


DATASET_PATH = RAW_DATA_DIR / "github_actions_real.csv"
REPORT_PATH = RESULTS_DIR / "eda_report.txt"


CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "repository",
    "workflow_name",
    "event",
    "branch",
    "conclusion",
    "status",
    "commit_author",
)

NUMERICAL_BOXPLOT_COLUMNS: tuple[str, ...] = (
    "run_duration_sec",
    "lines_added",
    "lines_deleted",
    "files_changed",
)

NUMERICAL_HEATMAP_COLUMNS: tuple[str, ...] = (
    "run_duration_sec",
    "run_attempt",
    "lines_added",
    "lines_deleted",
    "total_changes",
    "files_changed",
)

CONCLUSION_ORDER: tuple[str, ...] = ("success", "failure")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["created_at", "updated_at", "commit_date"],
    )
    # author_association is always NaN in the commits API; drop early.
    if "author_association" in df.columns and df["author_association"].isna().all():
        df = df.drop(columns=["author_association"])
    return df


def section(title: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n{title}\n{bar}"


def subsection(title: str) -> str:
    return f"\n--- {title} ---"


def _format_value_counts(series: pd.Series, top_n: int | None = None) -> str:
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    pct = (counts / len(series) * 100).round(2)
    frame = pd.DataFrame({"count": counts, "percent": pct})
    return frame.to_string()


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def print_overview(df: pd.DataFrame) -> None:
    print(section("CI/CD Failure Prediction — EDA Report (real GitHub data)"))
    print(f"Source file : {DATASET_PATH}")
    print(f"Rows        : {len(df):,}")
    print(f"Columns     : {df.shape[1]}")

    print(section("Schema (dtypes)"))
    dtype_df = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "n_unique": df.nunique(dropna=False),
        }
    )
    print(dtype_df.to_string())

    print(section("Missing values"))
    missing = df.isna().sum()
    missing_df = pd.DataFrame(
        {"missing": missing, "percent": (missing / len(df) * 100).round(2)}
    )
    missing_df = missing_df[missing_df["missing"] > 0].sort_values(
        "missing", ascending=False
    )
    if missing_df.empty:
        print("No missing values detected.")
    else:
        print(missing_df.to_string())

    print(section("Numerical column statistics"))
    numerical = df.select_dtypes(include=[np.number])
    if numerical.empty:
        print("(no numerical columns)")
    else:
        print(numerical.describe().T.round(3).to_string())

    print(section("Categorical column distributions"))
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        print(subsection(col))
        unique = df[col].nunique(dropna=False)
        if unique > 20:
            print(f"({unique} unique values — showing top 15)")
            print(_format_value_counts(df[col], top_n=15))
        else:
            print(_format_value_counts(df[col]))

    print(section("Target distribution: conclusion (binary)"))
    print(_format_value_counts(df["conclusion"]))
    n_success = int((df["conclusion"] == "success").sum())
    n_failure = int((df["conclusion"] == "failure").sum())
    if n_failure > 0:
        ratio = n_success / n_failure
        print(f"\nImbalance ratio (success:failure) = {ratio:.2f} : 1")

    print(section("Cross-tabulation: conclusion × event"))
    if "event" in df.columns:
        crosstab = pd.crosstab(df["event"], df["conclusion"], dropna=False)
        crosstab["failure_rate_%"] = (
            crosstab.get("failure", 0)
            / crosstab.sum(axis=1).replace(0, np.nan)
            * 100
        ).round(2)
        print(crosstab.to_string())

    print(section("Cross-tabulation: conclusion × repository"))
    crosstab_repo = pd.crosstab(df["repository"], df["conclusion"], dropna=False)
    crosstab_repo["failure_rate_%"] = (
        crosstab_repo.get("failure", 0)
        / crosstab_repo.sum(axis=1).replace(0, np.nan)
        * 100
    ).round(2)
    crosstab_repo = crosstab_repo.sort_values(
        "failure_rate_%", ascending=False, na_position="last"
    )
    print(crosstab_repo.to_string())

    print(section("Cross-tabulation: conclusion × run_attempt"))
    crosstab_attempt = pd.crosstab(
        df["run_attempt"], df["conclusion"], dropna=False
    )
    print(crosstab_attempt.to_string())

    print(section("End of report"))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _ordered_value_counts(series: pd.Series, order: Sequence[str]) -> pd.Series:
    counts = series.value_counts()
    return counts.reindex([v for v in order if v in counts.index])


def plot_conclusion_distribution(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    counts = _ordered_value_counts(df["conclusion"], CONCLUSION_ORDER)
    colors = ["#2e7d32", "#c62828"][: len(counts)]

    fig, ax = plotter.new_figure(figsize=(7.5, 5.0))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Distribution of Workflow-Run Conclusions (Real GitHub Data)")
    ax.set_xlabel("Conclusion")
    ax.set_ylabel("Number of workflow runs")
    ax.set_ylim(0, counts.values.max() * 1.15)

    plotter.save_figure(
        fig,
        "fig_01_conclusion_distribution.png",
        caption=(
            "Distribution of the binary target ``conclusion`` across the 9,772 "
            "GitHub Actions workflow runs collected from 18 open-source "
            "repositories. The pronounced majority of successful runs (~89%) "
            "reflects the natural success bias of merged code and motivates "
            "the use of class-weighted training in subsequent phases."
        ),
        title="Figure 1 — Conclusion distribution",
    )


def plot_event_distribution(df: pd.DataFrame, plotter: ThesisPlotter) -> None:
    counts = df["event"].value_counts().sort_values(ascending=True)
    colors = plotter.palette(len(counts))

    fig, ax = plotter.new_figure(figsize=(9.0, 5.5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="black")
    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_title("Distribution of Workflow Trigger Events")
    ax.set_xlabel("Number of workflow runs")
    ax.set_ylabel("Trigger event")
    ax.set_xlim(0, counts.values.max() * 1.18)
    ax.grid(axis="y", visible=False)

    plotter.save_figure(
        fig,
        "fig_02_event_distribution.png",
        caption=(
            "Frequency of workflow trigger events (push, pull_request, "
            "schedule, etc.). The mix of event types confirms that the "
            "dataset covers both human-driven and automated CI activity, "
            "ensuring the model learns from realistic CI/CD workloads."
        ),
        title="Figure 2 — Trigger event distribution",
    )


def plot_repository_distribution(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    counts = df["repository"].value_counts().sort_values(ascending=True)
    colors = plotter.palette(len(counts))

    fig, ax = plotter.new_figure(figsize=(10.0, 7.5))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="black")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_title("Workflow Runs Collected per Repository")
    ax.set_xlabel("Number of workflow runs")
    ax.set_ylabel("Repository")
    ax.set_xlim(0, counts.values.max() * 1.15)
    ax.grid(axis="y", visible=False)

    plotter.save_figure(
        fig,
        "fig_03_rows_per_repository.png",
        caption=(
            "Number of workflow runs collected per repository. The "
            "deliberate 600-row cap per repository — combined with two "
            "repositories that did not reach the cap — confirms that the "
            "dataset is broadly distributed rather than dominated by any "
            "single project."
        ),
        title="Figure 3 — Rows per repository",
    )


def plot_numerical_boxplot(df: pd.DataFrame, plotter: ThesisPlotter) -> None:
    columns = NUMERICAL_BOXPLOT_COLUMNS
    palette = plotter.palette(len(columns))

    pretty_names = {
        "run_duration_sec": "Run duration (seconds, log)",
        "lines_added": "Lines added (log)",
        "lines_deleted": "Lines deleted (log)",
        "files_changed": "Files changed (log)",
    }

    fig, axes = plotter.new_figure(figsize=(11.5, 8.0), nrows=2, ncols=2)
    axes_flat = axes.flatten()

    for ax, col, color in zip(axes_flat, columns, palette):
        series = df[col].dropna()
        # Real data is heavy-tailed — show on a log scale via log-transform.
        series_log = np.log1p(series.astype(float).clip(lower=0))
        sns.boxplot(
            x=series_log,
            ax=ax,
            color=color,
            fliersize=2.5,
            linewidth=1.0,
            width=0.55,
        )
        ax.set_title(pretty_names.get(col, col))
        ax.set_xlabel(f"log(1 + {col})")
        ax.set_ylabel("")
        ax.grid(axis="x", visible=True, alpha=0.3)

    fig.suptitle(
        "Distribution of Core Numerical Features (Log-Transformed)", y=1.02
    )

    plotter.save_figure(
        fig,
        "fig_04_numerical_features_boxplot.png",
        caption=(
            "Box-plot distribution of the four core numerical features "
            "after a log(1+x) transform. The transformation tames the heavy "
            "right tails of commit-level statistics (lines added/deleted, "
            "files changed) and makes the underlying inter-quartile "
            "structure visible for downstream modelling decisions."
        ),
        title="Figure 4 — Numerical feature boxplots (log-scaled)",
    )


def plot_correlation_heatmap(df: pd.DataFrame, plotter: ThesisPlotter) -> None:
    numerical = df[list(NUMERICAL_HEATMAP_COLUMNS)].copy()
    corr = numerical.corr(method="pearson")

    fig, ax = plotter.new_figure(figsize=(8.5, 7.0))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.85},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix of Numerical Features")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.grid(False)

    plotter.save_figure(
        fig,
        "fig_05_correlation_heatmap.png",
        caption=(
            "Pearson correlation coefficients between the six numerical "
            "features. Strong correlations among ``lines_added``, "
            "``lines_deleted``, ``total_changes`` and ``files_changed`` are "
            "expected because they share commit-size semantics; ``run_attempt`` "
            "and ``run_duration_sec`` remain comparatively independent and "
            "thus contribute distinct predictive signal."
        ),
        title="Figure 5 — Correlation heatmap",
    )


def plot_conclusion_vs_repo_heatmap(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    crosstab = pd.crosstab(df["repository"], df["conclusion"])
    crosstab = crosstab.reindex(
        columns=[c for c in CONCLUSION_ORDER if c in crosstab.columns]
    )
    # Sort by failure rate descending for visual signal.
    failure_rate = crosstab.get("failure", pd.Series(0)) / crosstab.sum(axis=1)
    crosstab = crosstab.loc[failure_rate.sort_values(ascending=False).index]

    fig, ax = plotter.new_figure(figsize=(8.5, 8.5))
    sns.heatmap(
        crosstab,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Number of workflow runs", "shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Cross-tabulation: Repository vs Conclusion (sorted by failure rate)")
    ax.set_xlabel("Conclusion")
    ax.set_ylabel("Repository")
    ax.grid(False)

    plotter.save_figure(
        fig,
        "fig_06_conclusion_vs_repository_heatmap.png",
        caption=(
            "Joint frequency of (repository, conclusion) pairs, with rows "
            "ordered by descending per-repository failure rate. The marked "
            "variation in failure rates across projects validates the "
            "inclusion of ``repository`` as a categorical predictor: the "
            "base failure probability differs substantially between codebases."
        ),
        title="Figure 6 — Conclusion vs repository",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _reset_captions_file(plotter: ThesisPlotter) -> None:
    if plotter.captions_path.exists():
        plotter.captions_path.unlink()


def run_eda() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)

    if not DATASET_PATH.exists():
        print(f"ERROR: dataset not found at {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {DATASET_PATH} ...")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_overview(df)
    report_text = buffer.getvalue()
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"Wrote EDA report → {REPORT_PATH}")
    sys.stdout.write(report_text)

    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    _reset_captions_file(plotter)

    print("\nGenerating publication-quality figures ...")
    plot_conclusion_distribution(df, plotter)
    plot_event_distribution(df, plotter)
    plot_repository_distribution(df, plotter)
    plot_numerical_boxplot(df, plotter)
    plot_correlation_heatmap(df, plotter)
    plot_conclusion_vs_repo_heatmap(df, plotter)

    print(f"\nAll figures saved to {FIGURES_DIR} (300 DPI).")
    print(f"Captions written to    {plotter.captions_path}.")


if __name__ == "__main__":
    run_eda()
