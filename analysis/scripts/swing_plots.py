"""Matplotlib rendering for savings against the daily swing of each window."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from results_plots import RESULT_STYLE, _results_typography

METHODS = {
    "bound": "Delay-only bound, 24h budget",
    "offline": "Offline shifting, 24h budget",
    "oracle": "Oracle, 24h horizon",
    "greenfilling": "Greenfilling",
}
# colors not used for policies, workloads, or objectives in the other figures
ZONES = {"DE": ("#3d8b5a", "Germany"), "FR": ("#8a5ea8", "France"), "PL": ("#c9a227", "Poland")}


def _faded(color: str) -> tuple[float, float, float]:
    """The color mixed with light gray, so the fitted lines stand out over the points."""
    return tuple(0.45 * np.array(to_rgb(color)) + 0.55 * np.array(to_rgb("#d9d9d9")))


def _spearman(rows: pd.DataFrame) -> float:
    return rows.saving_pct.rank().corr(rows.swing.rank())


def _swing_figure(savings: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.4), sharex=True, layout="constrained")
    for ax, (method, title) in zip(axes.flat, METHODS.items()):
        rows = savings.loc[savings.method == method]
        for zone, (color, _) in ZONES.items():
            part = rows.loc[rows.zone == zone]
            ax.scatter(part.swing, part.saving_pct, s=12, color=_faded(color), alpha=0.7, linewidths=0, zorder=1)
        for zone, (color, _) in ZONES.items():
            part = rows.loc[rows.zone == zone]
            slope, intercept = np.polyfit(part.swing, part.saving_pct, 1)
            span = np.array([part.swing.min(), part.swing.max()])
            ax.plot(span, intercept + slope * span, color=color, linewidth=2.5, zorder=3)
        ax.axhline(0, color=".3", linewidth=0.8)
        ax.set_title(f"{title}\nSpearman, all pairs {_spearman(rows):.2f}")
    for ax in axes[1]:
        ax.set_xlabel("Daily swing (mean daily range / window mean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Saving against EASY (%)")
    handles = [Patch(color=color, label=country) for color, country in ZONES.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False)
    fig.suptitle("Saving against daily swing, one point per pair, line fitted per country")
    _results_typography(fig)
    return fig


def save_swing_figure(savings: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_swing_figure(savings, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _swing_figure(savings)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"swing_savings.{suffix}", bbox_inches="tight")
        plt.close(figure)
