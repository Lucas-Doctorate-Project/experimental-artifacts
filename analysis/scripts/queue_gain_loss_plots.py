"""Matplotlib rendering for the oracle's gain and loss inside the queue."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from plot_style import POLICY_COLORS
from results_plots import RESULT_STYLE, _results_typography

SCORINGS = {
    "unrelated": ("#999999", "Another intensity window"),
    "own": (POLICY_COLORS["Head"], "Own intensity window"),
}


def _gain_loss_figure(scores: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6.4), layout="constrained")
    for scoring, (color, label) in SCORINGS.items():
        rows = scores.loc[scores.scoring == scoring]
        ax.scatter(
            rows.gain_pct,
            rows.loss_pct,
            s=14,
            color=color,
            alpha=0.75,
            linewidths=0,
            label=label,
        )
    top = 1.05 * max(scores.gain_pct.max(), scores.loss_pct.max())
    ax.plot(
        [0, top],
        [0, top],
        color=".3",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
        label="Loss equals gain",
    )
    ax.set(xlim=(0, top), ylim=(0, top), aspect="equal")
    ax.set_xlabel("Gain from jobs that ran cleaner (% of EASY footprint)")
    ax.set_ylabel("Loss from jobs that ran dirtier (% of EASY footprint)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[i] for i in (1, 0, 2)],
        [labels[i] for i in (1, 0, 2)],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
    )
    ax.text(0.05, 0.94, "Net footprint increase", transform=ax.transAxes, color=".35")
    ax.text(
        0.95,
        0.08,
        "Net footprint saving",
        transform=ax.transAxes,
        ha="right",
        color=".35",
    )
    fig.suptitle("Cleaner starts can displace other jobs")
    ax.set_title(
        "Head-job shifting, 24 h limit", fontsize=11, color=".35", fontweight="normal"
    )
    _results_typography(fig)
    return fig


def save_queue_gain_loss_figure(scores: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_queue_gain_loss_figure(scores, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _gain_loss_figure(scores)
        for suffix in ("pdf", "png"):
            figure.savefig(
                destination / suffix / f"queue_gain_loss.{suffix}", bbox_inches="tight"
            )
        plt.close(figure)
