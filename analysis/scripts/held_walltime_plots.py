"""Matplotlib rendering for the intensity drop of the oracle's held jobs by runtime share."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from results_plots import RESULT_STYLE, _results_typography

# runtime share of the requested walltime: lower edge, upper edge, label
SHARES = [(0, 0.1, "Under 10%"), (0.1, 0.5, "10 to 50%"), (0.5, 0.9, "50 to 90%"), (0.9, np.inf, "90% or more")]
DROPS = {
    "target": ("#9cc3de", "Target (requested walltime)"),
    "got": ("#2677ad", "Got (real runtime)"),
    "best": ("#a6acb1", "Best start knowing the runtime"),
}


def _held_walltime_figure(drops: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
    labels = []
    for position, (low, high, label) in enumerate(SHARES):
        rows = drops.loc[(drops.runtime_share > low) & (drops.runtime_share <= high)]
        for offset, (column, (color, _)) in zip((-0.27, 0, 0.27), DROPS.items()):
            value = 100 * np.average(rows[column], weights=rows.work)
            ax.bar(position + offset, value, 0.27, color=color, edgecolor=".3", linewidth=0.5)
        labels.append(
            f"{label}\n{100 * len(rows) / len(drops):.0f}% of jobs, {100 * rows.work.sum() / drops.work.sum():.0f}% of work"
        )
    ax.set_xticks(range(len(SHARES)), labels)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Real runtime as a share of the requested walltime")
    ax.set_ylabel("Drop in mean intensity (%)")
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color, ec=".3", lw=0.5, label=label) for color, label in DROPS.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Held jobs: chosen start against starting at the decision, weighted by work")
    _results_typography(fig)
    return fig


def save_held_walltime_figure(drops: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_held_walltime_figure(drops, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _held_walltime_figure(drops)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"held_walltime.{suffix}", bbox_inches="tight")
        plt.close(figure)
