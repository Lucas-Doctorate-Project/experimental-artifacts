"""Matplotlib rendering for the oracle's gain and loss inside the queue."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from results_plots import RESULT_STYLE, _results_typography

SCORINGS = {
    "unrelated": ("#a6acb1", "Same moves, scored on another window"),
    "own": ("#2677ad", "Oracle 24h, scored on its own window"),
}


def _gain_loss_figure(scores: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 6.2), layout="constrained")
    for scoring, (color, label) in SCORINGS.items():
        rows = scores.loc[scores.scoring == scoring]
        ax.scatter(rows.gain_pct, rows.loss_pct, s=14, color=color, alpha=0.75, linewidths=0, label=label)
    top = 1.05 * max(scores.gain_pct.max(), scores.loss_pct.max())
    ax.plot([0, top], [0, top], color=".3", linewidth=0.8, linestyle="--", label="Loss equals gain")
    ax.set(xlim=(0, top), ylim=(0, top), aspect="equal")
    ax.set_xlabel("Gain from jobs that ran cleaner (% of EASY footprint)")
    ax.set_ylabel("Loss from jobs that ran dirtier (% of EASY footprint)")
    ax.legend(loc="upper left", frameon=False)
    fig.suptitle("Gains and losses inside the queue, one point per pair")
    _results_typography(fig)
    return fig


def save_queue_gain_loss_figure(scores: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_queue_gain_loss_figure(scores, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _gain_loss_figure(scores)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"queue_gain_loss.{suffix}", bbox_inches="tight")
        plt.close(figure)
