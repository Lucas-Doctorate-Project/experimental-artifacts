"""Matplotlib rendering for when Greenfilling allows backfilling."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from results_plots import (
    ALPHA_COLORS,
    RESULT_STYLE,
    _draw_boxes,
    _figure_legend,
    _results_typography,
)

# flat steps cover under 0.3% of the median run time, so they are not drawn
PANELS = [
    (
        "level",
        {"Q1": "Lowest 25%", "Q2": "25-50%", "Q3": "50-75%", "Q4": "Highest 25%"},
        "By grid intensity level",
        "Rank within the 4-week window",
    ),
    (
        "direction",
        {"falling": "Fell", "rising": "Rose"},
        "By recent change in grid intensity",
        "Change over the last 15 minutes",
    ),
]


def _allowed_panel(
    ax: Axes, shares: pd.DataFrame, machine: str, kind: str, categories: list[str]
) -> None:
    positions = np.arange(len(categories))
    for label, offset in zip(ALPHA_COLORS, [-0.27, -0.09, 0.09, 0.27]):
        objective, load = label.split()
        subset = shares.loc[
            (shares.kind == kind)
            & (shares.objective == objective)
            & (shares.workload == f"{machine}_{load}")
        ]
        values = [
            subset.loc[subset.category == category, "open_share_pct"].dropna().to_numpy()
            for category in categories
        ]
        _draw_boxes(ax, values, positions + offset, ALPHA_COLORS[label], 0.15, "vertical", 0.55)
    ax.set_xlim(-0.55, len(categories) - 0.45)
    ax.set_ylim(-5, 105)


def _allowed_figure(shares: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.2),
        layout="constrained",
        sharey=True,
        sharex="col",
        width_ratios=[len(labels) for _, labels, _, _ in PANELS],
    )
    for row, machine in enumerate(("mustang", "trinity")):
        for column, (kind, labels, title, xlabel) in enumerate(PANELS):
            ax = axes[row, column]
            _allowed_panel(ax, shares, machine, kind, list(labels))
            ax.set_xticks(np.arange(len(labels)), list(labels.values()))
            if row == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel(xlabel)
        axes[row, 0].set_ylabel(machine.capitalize())
    _figure_legend(fig, ALPHA_COLORS, 0.55, 4)
    fig.set_layout_engine("constrained", rect=(0, 0.08, 1, 0.92))
    shared_ylabel = fig.supylabel("Backfilling allowed (% of time)")
    fig.suptitle("When Greenfilling allows backfilling")
    _results_typography(fig)
    shared_ylabel.set_fontsize(14)
    return fig


def save_backfill_allowed_figure(shares: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_backfill_allowed_figure(shares, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _allowed_figure(shares)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"backfill_allowed.{suffix}", bbox_inches="tight")
        plt.close(figure)
