"""Sanity-check EDA on the freshly collected GitHub Actions dataset.

This is *not* the full Phase 1 EDA — it's a 3-figure quick look so we can
confirm the collection produced usable data before iterating further.

Run from project root::

    python src/quick_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import FIGURES_DIR, RAW_DATA_DIR, ensure_dir  # noqa: E402
from src.visualization import ThesisPlotter  # noqa: E402


DATASET_PATH = RAW_DATA_DIR / "github_actions_real.csv"


def _print_overview(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("Quick EDA — GitHub Actions real dataset")
    print("=" * 72)
    print(f"Shape  : {df.shape}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string() or "  (none)")


def _plot_conclusion_distribution(
    df: pd.DataFrame, plotter: ThesisPlotter
) -> None:
    counts = df["conclusion"].value_counts()
    palette = {
        "success": "#2e7d32",
        "failure": "#c62828",
        "cancelled": "#757575",
        "skipped": "#9e9e9e",
        "neutral": "#bdbdbd",
    }
    colors = [palette.get(str(k), "#1565c0") for k in counts.index]

    fig, ax = plotter.new_figure(figsize=(7.5, 5.0))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Workflow Run Conclusion Distribution (GitHub Actions)")
    ax.set_xlabel("Conclusion")
    ax.set_ylabel("Number of workflow runs")
    ax.set_ylim(0, counts.values.max() * 1.15)

    plotter.save_figure(
        fig,
        "fig_eda_01_conclusion_distribution.png",
        caption=(
            "Distribution of workflow-run conclusions collected from the "
            "GitHub Actions API. The success/failure ratio drives the class "
            "balance for the downstream classification model."
        ),
        title="Quick EDA Figure 1 — Conclusion distribution",
    )


def _plot_commit_msg_length(df: pd.DataFrame, plotter: ThesisPlotter) -> None:
    lengths = df["commit_message"].dropna().astype(str).str.len()
    fig, ax = plotter.new_figure(figsize=(8.5, 5.2))
    ax.hist(
        lengths,
        bins=40,
        color="#1565c0",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.axvline(
        lengths.median(),
        color="#c62828",
        linestyle="--",
        linewidth=1.2,
        label=f"median = {int(lengths.median())} chars",
    )
    ax.axvline(
        lengths.mean(),
        color="#2e7d32",
        linestyle="--",
        linewidth=1.2,
        label=f"mean = {lengths.mean():.1f} chars",
    )
    ax.set_title("Distribution of Commit Message Lengths")
    ax.set_xlabel("Commit message length (characters)")
    ax.set_ylabel("Number of commits")
    ax.legend(frameon=False)

    plotter.save_figure(
        fig,
        "fig_eda_02_commit_msg_length.png",
        caption=(
            "Histogram of commit message lengths in the collected dataset. "
            "The distribution informs the TF-IDF vectoriser parameters used "
            "downstream."
        ),
        title="Quick EDA Figure 2 — Commit message lengths",
    )


def _plot_rows_per_repo(df: pd.DataFrame, plotter: ThesisPlotter) -> None:
    counts = df["repository"].value_counts().head(20).sort_values()
    palette = plotter.palette(len(counts))

    fig, ax = plotter.new_figure(figsize=(10.0, 7.5))
    bars = ax.barh(counts.index, counts.values, color=palette, edgecolor="black")
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {value:,}",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_title("Top 20 Repositories by Number of Collected Workflow Runs")
    ax.set_xlabel("Number of workflow runs")
    ax.set_ylabel("Repository")
    ax.set_xlim(0, counts.values.max() * 1.15)
    ax.grid(axis="y", visible=False)

    plotter.save_figure(
        fig,
        "fig_eda_03_rows_per_repo.png",
        caption=(
            "Number of workflow runs collected per repository. The bar lengths "
            "confirm that the collector distributes its budget across multiple "
            "projects rather than over-sampling one."
        ),
        title="Quick EDA Figure 3 — Rows per repository",
    )


def main() -> None:
    ensure_dir(FIGURES_DIR)

    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} does not exist. Run collector first.")
        return

    df = pd.read_csv(DATASET_PATH)
    _print_overview(df)

    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    _plot_conclusion_distribution(df, plotter)
    _plot_commit_msg_length(df, plotter)
    _plot_rows_per_repo(df, plotter)

    # Quick summary stats that the user asked for in the deliverables.
    print("\n" + "=" * 72)
    print("Deliverables snapshot")
    print("=" * 72)
    print(f"Total rows                       : {len(df):,}")
    if "conclusion" in df.columns:
        conc_counts = df["conclusion"].value_counts(dropna=False)
        if "success" in conc_counts and "failure" in conc_counts:
            ratio = conc_counts["success"] / max(conc_counts["failure"], 1)
            print(
                f"Success/Failure ratio            : "
                f"{conc_counts['success']:,} / {conc_counts['failure']:,} "
                f"({ratio:.2f}x)"
            )

    print("\nTop 5 repositories by row count:")
    print(df["repository"].value_counts().head(5).to_string())

    print("\n5 random commit messages:")
    sample = df.sample(n=min(5, len(df)), random_state=42)[
        ["repository", "conclusion", "commit_message"]
    ]
    for _, row in sample.iterrows():
        message = " ".join(str(row["commit_message"]).split())[:200]
        print(
            f"  [{row['repository']} · {row['conclusion']}] "
            f"{message}"
        )

    if "lines_added" in df.columns:
        avg_added = df["lines_added"].dropna().mean()
        avg_files = df["files_changed"].dropna().mean()
        print(
            f"\nAverage lines_added              : {avg_added:.1f}"
            f"\nAverage files_changed            : {avg_files:.1f}"
        )


if __name__ == "__main__":
    main()
