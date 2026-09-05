"""Phase 3 runner: sanity check + architecture diagram + summary.

The architecture diagram is a single-target version of the synthetic
project's flow chart: input → ColumnTransformer → four parallel branches →
fused sparse matrix → classifier → one binary output (``conclusion``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sanity_check import print_results, run_sanity_check  # noqa: E402
from src.utils import FIGURES_DIR, RESULTS_DIR, ensure_dir  # noqa: E402
from src.visualization import ThesisPlotter  # noqa: E402


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #


_COLOR_INPUT = "#37474f"
_COLOR_TRANSFORMER = "#1a237e"
_COLOR_NUM = "#1565c0"
_COLOR_CAT = "#2e7d32"
_COLOR_BIN = "#ef6c00"
_COLOR_TEXT = "#6a1b9a"
_COLOR_FUSED = "#00695c"
_COLOR_CLASSIFIER = "#b71c1c"
_COLOR_OUT_SUCCESS = "#2e7d32"
_COLOR_OUT_FAILURE = "#c62828"
_TEXT_LIGHT = "#ffffff"
_ARROW_COLOR = "#37474f"


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    fontsize: int = 10,
    text_color: str = _TEXT_LIGHT,
) -> tuple[float, float, float, float, float]:
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=1.3,
        facecolor=facecolor,
        edgecolor="#1c1c1c",
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )
    center_x = x + width / 2.0
    return center_x, y + height, center_x, y, center_x


def _draw_arrow(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    *,
    color: str = _ARROW_COLOR,
) -> None:
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle="-|>",
        mutation_scale=16,
        color=color,
        linewidth=1.5,
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)


def draw_architecture_diagram(plotter: ThesisPlotter) -> None:
    """Render ``fig_11_hybrid_architecture_diagram.png``."""
    fig, ax = plotter.new_figure(figsize=(14.0, 13.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.grid(False)

    # Row 1 — raw input
    raw = _draw_box(
        ax,
        x=3.5,
        y=12.4,
        width=5.0,
        height=0.95,
        text=(
            "GitHub Actions Workflow Run\n"
            "(5 numerical · 4 categorical · 6 binary · 1 text column)"
        ),
        facecolor=_COLOR_INPUT,
        fontsize=10,
    )

    # Row 2 — ColumnTransformer
    ct = _draw_box(
        ax,
        x=2.0,
        y=10.5,
        width=8.0,
        height=0.95,
        text="ColumnTransformer  (n_jobs=-1, sparse_threshold=0.3)",
        facecolor=_COLOR_TRANSFORMER,
        fontsize=11,
    )
    _draw_arrow(ax, raw[4], raw[3], ct[4], ct[1])

    # Row 3 — 4 parallel branches
    branch_y = 7.9
    branch_h = 1.7
    branch_w = 2.55
    n_branches = 4
    margin = 0.4
    available = 12.0 - 2 * margin
    spacing = (available - branch_w * n_branches) / (n_branches - 1)

    branches = [
        (
            "Numerical Branch\n5 features\nStandardScaler",
            _COLOR_NUM,
        ),
        (
            "Categorical Branch\n4 columns\nOneHotEncoder\nhandle_unknown='ignore'",
            _COLOR_CAT,
        ),
        (
            "Binary Branch\n6 features\npassthrough",
            _COLOR_BIN,
        ),
        (
            "Text Branch · 1 column\nTfidfVectorizer\nngram=(1,2), max_feat=3000\nmin_df=5, sublinear_tf=True",
            _COLOR_TEXT,
        ),
    ]
    branch_anchors_bottom: list[tuple[float, float]] = []
    for i, (label, color) in enumerate(branches):
        x = margin + i * (branch_w + spacing)
        anchors = _draw_box(
            ax,
            x=x,
            y=branch_y,
            width=branch_w,
            height=branch_h,
            text=label,
            facecolor=color,
            fontsize=9,
        )
        _draw_arrow(ax, ct[4], ct[3], anchors[0], anchors[1])
        branch_anchors_bottom.append((anchors[2], anchors[3]))

    # Row 4 — fused matrix
    fused = _draw_box(
        ax,
        x=2.0,
        y=5.5,
        width=8.0,
        height=0.95,
        text="Fused Feature Matrix  (sparse, ≈ 3,090 columns)",
        facecolor=_COLOR_FUSED,
        fontsize=11,
    )
    for bx, by in branch_anchors_bottom:
        _draw_arrow(ax, bx, by, fused[4], fused[1])

    # Row 5 — classifier
    classifier = _draw_box(
        ax,
        x=1.5,
        y=3.3,
        width=9.0,
        height=1.15,
        text=(
            "Binary Classifier (no MultiOutputClassifier wrapper)\n"
            "Logistic Regression   |   Random Forest   |   XGBoost"
        ),
        facecolor=_COLOR_CLASSIFIER,
        fontsize=11,
    )
    _draw_arrow(ax, fused[4], fused[3], classifier[4], classifier[1])

    # Row 6 — binary target with two class boxes
    out_y = 1.2
    out_h = 1.3
    out_w = 3.6
    spacing_out = (12.0 - 2 * margin - out_w * 2) / 1
    outputs = [
        (
            "Predicted: SUCCESS\n(~89% prior probability)\np(success | features)",
            _COLOR_OUT_SUCCESS,
        ),
        (
            "Predicted: FAILURE\n(~11% prior probability)\np(failure | features)",
            _COLOR_OUT_FAILURE,
        ),
    ]
    for i, (label, color) in enumerate(outputs):
        x = margin + i * (out_w + spacing_out)
        anchors = _draw_box(
            ax,
            x=x,
            y=out_y,
            width=out_w,
            height=out_h,
            text=label,
            facecolor=color,
            fontsize=10,
        )
        _draw_arrow(ax, classifier[4], classifier[3], anchors[0], anchors[1])

    # Subtitle bar
    ax.text(
        6.0,
        0.4,
        "Target: conclusion ∈ {success, failure}",
        ha="center",
        va="center",
        fontsize=11,
        fontstyle="italic",
        color="#37474f",
    )

    ax.set_title(
        "Hybrid Binary-Classification Pipeline Architecture (Real CI/CD Data)",
        fontsize=14,
        pad=15,
    )

    plotter.save_figure(
        fig,
        "fig_11_hybrid_architecture_diagram.png",
        caption=(
            "Architecture of the binary hybrid pipeline used on the real "
            "GitHub Actions dataset. Four parallel preprocessing branches "
            "— numerical scaling, one-hot encoding of bucketed categoricals, "
            "passthrough of derived binary flags, and TF-IDF of the cleaned "
            "``commit_message_clean`` column — are fused via "
            "ColumnTransformer into a sparse feature matrix, then consumed "
            "by a single binary classifier (Logistic Regression, Random "
            "Forest, or XGBoost) to predict ``conclusion``."
        ),
        title="Figure 11 — Hybrid architecture diagram (binary)",
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _save_summary(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> None:
    ensure_dir(FIGURES_DIR)
    ensure_dir(RESULTS_DIR)

    print("[Phase 3] Running sanity check on all three binary pipelines ...")
    report = run_sanity_check()
    print_results(report)

    summary_path = RESULTS_DIR / "phase3_summary.json"
    _save_summary(report, summary_path)
    print(f"\n[Phase 3] Summary written → {summary_path}")

    print("\n[Phase 3] Rendering architecture diagram ...")
    plotter = ThesisPlotter(figures_dir=FIGURES_DIR)
    draw_architecture_diagram(plotter)

    print("\n[Phase 3] Final summary")
    print("-" * 78)
    for name, info in report["models"].items():
        print(
            f"  - {name}: fit={info['fit_time_sec']:.2f}s · "
            f"predict={info['predict_time_sec']:.4f}s · "
            f"features={info['feature_total']} · "
            f"serializable={info['serializable']}"
        )

    print("\n[Phase 3] Done.")


if __name__ == "__main__":
    main()
