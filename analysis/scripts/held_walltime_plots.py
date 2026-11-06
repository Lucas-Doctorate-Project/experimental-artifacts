"""Matplotlib rendering for the intensity drop of the oracle's held jobs by runtime share."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plot_style import POLICY_COLORS, pale
from results_plots import RESULT_STYLE, _results_typography

# runtime share of the requested walltime: lower edge, upper edge, label
SHARES = [
    (0, 0.1, "Under 10%"),
    (0.1, 0.5, "10 to 50%"),
    (0.5, 0.9, "50 to 90%"),
    (0.9, np.inf, "90% or more"),
]
DROPS = {
    "target": ("white", "Predicted from walltime"),
    "got": (POLICY_COLORS["Head"], "Achieved at actual runtime"),
    "best": (pale(POLICY_COLORS["Head"]), "Best with actual runtime"),
}


def _held_walltime_figure(drops: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), layout="constrained")
    labels = []
    for position, (low, high, label) in enumerate(SHARES):
        rows = drops.loc[(drops.runtime_share > low) & (drops.runtime_share <= high)]
        for offset, (column, (color, _)) in zip((-0.27, 0, 0.27), DROPS.items()):
            value = 100 * np.average(rows[column], weights=rows.work)
            ax.bar(
                position + offset,
                value,
                0.27,
                color=color,
                edgecolor=POLICY_COLORS["Head"],
                linewidth=0.8,
                hatch="///" if column == "best" else None,
                zorder=2,
            )
        labels.append(
            f"{label}\n{100 * len(rows) / len(drops):.0f}% of jobs, {100 * rows.work.sum() / drops.work.sum():.0f}% of work"
        )
    ax.set_xticks(range(len(SHARES)), labels)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Actual runtime / requested walltime")
    ax.set_ylabel("Reduction in mean intensity (%)")
    ax.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=color,
                ec=POLICY_COLORS["Head"],
                lw=0.8,
                hatch="///" if column == "best" else None,
                label=label,
            )
            for column, (color, label) in DROPS.items()
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Short runtimes limit the benefit of holding jobs")
    ax.set_title(
        "Head-job shifting, 24 h limit | Weighted by job area",
        fontsize=11,
        color=".35",
        fontweight="normal",
    )
    _results_typography(fig)
    return fig


def save_held_walltime_figure(drops: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_held_walltime_figure(drops, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _held_walltime_figure(drops)
        for suffix in ("pdf", "png"):
            figure.savefig(
                destination / suffix / f"held_walltime.{suffix}", bbox_inches="tight"
            )
        plt.close(figure)
