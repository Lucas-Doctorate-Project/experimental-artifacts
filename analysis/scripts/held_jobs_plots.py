"""Matplotlib rendering for the oracle's saving from held jobs and from all other jobs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plot_style import POLICY_COLORS, pale
from results_plots import RESULT_STYLE, _draw_boxes, _results_typography

SCORINGS = {
    "own": (POLICY_COLORS["Head"], "Own intensity window"),
    "unrelated": ("#999999", "Another intensity window"),
}


def _held_figure(splits: pd.DataFrame) -> Figure:
    own = splits.loc[splits.scoring == "own"]
    groups = [
        (
            "held_pct",
            f"Jobs deliberately held\n(median {own.held_jobs.median():.0f} per run)",
        ),
        (
            "others_pct",
            f"All other jobs\n(median {own.other_moved_jobs.median():,.0f} moved per run)",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for position, (column, _) in enumerate(groups):
        for offset, (scoring, (color, _)) in zip((-0.18, 0.18), SCORINGS.items()):
            values = splits.loc[splits.scoring == scoring, column].to_numpy()
            _draw_boxes(ax, values, [position + offset], color, 0.3, "horizontal", 0.55)
    ax.set_yticks(range(len(groups)), [label for _, label in groups])
    ax.invert_yaxis()
    ax.axvline(0, color=".5", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to footprint saving (% of EASY footprint)")
    ax.legend(
        handles=[
            Patch(facecolor=pale(color, 0.55), edgecolor=color, label=label)
            for color, label in SCORINGS.values()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Where head-job shifting reduces the footprint")
    ax.set_title("24 h limit", fontsize=11, color=".35", fontweight="normal")
    _results_typography(fig)
    return fig


def save_held_jobs_figure(splits: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_held_jobs_figure(splits, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _held_figure(splits)
        for suffix in ("pdf", "png"):
            figure.savefig(
                destination / suffix / f"held_jobs.{suffix}", bbox_inches="tight"
            )
        plt.close(figure)
