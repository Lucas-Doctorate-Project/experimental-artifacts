"""Matplotlib rendering for savings against the daily swing of each window."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from results_plots import RESULT_STYLE, _results_typography

METHODS = {
    "bound": "Delay-only bound, 24 h",
    "offline": "Offline shifting, 24 h",
    "oracle": "Head-job shifting, 24 h",
    "greenfilling": "Greenfilling",
}
ZONES = {
    "DE": ("#3d8b5a", "Germany"),
    "FR": ("#8a5ea8", "France"),
    "PL": ("#c9a227", "Poland"),
}
ZONE_MARKERS = {"DE": "o", "FR": "s", "PL": "^"}
ZONE_LINES = {"DE": "-", "FR": "--", "PL": ":"}


def _faded(color: str) -> tuple[float, float, float]:
    """The color mixed with light gray, so the fitted lines stand out over the points."""
    return tuple(0.45 * np.array(to_rgb(color)) + 0.55 * np.array(to_rgb("#d9d9d9")))


def _spearman(rows: pd.DataFrame) -> float:
    return rows.saving_pct.rank().corr(rows.swing.rank())


def _swing_figure(savings: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(
        2, 2, figsize=(9.5, 7.4), sharex=True, layout="constrained"
    )
    for ax, (method, title) in zip(axes.flat, METHODS.items()):
        rows = savings.loc[savings.method == method]
        for zone, (color, _) in ZONES.items():
            part = rows.loc[rows.zone == zone]
            ax.scatter(
                part.swing,
                part.saving_pct,
                s=16,
                marker=ZONE_MARKERS[zone],
                color=_faded(color),
                alpha=0.7,
                linewidths=0,
                zorder=1,
            )
        for zone, (color, _) in ZONES.items():
            part = rows.loc[rows.zone == zone]
            slope, intercept = np.polyfit(part.swing, part.saving_pct, 1)
            span = np.array([part.swing.min(), part.swing.max()])
            ax.plot(
                span,
                intercept + slope * span,
                color=color,
                linestyle=ZONE_LINES[zone],
                linewidth=2,
                zorder=3,
            )
        ax.axhline(0, color=".5", linewidth=0.8, zorder=0)
        ax.set_title(title, pad=25)
        ax.text(
            0,
            1.025,
            f"Spearman correlation: {_spearman(rows):.2f}",
            transform=ax.transAxes,
            fontsize=10,
            color=".35",
        )
    for ax in axes[1]:
        ax.set_xlabel("Daily swing (mean daily range / window mean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Saving against EASY (%)")
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=ZONE_MARKERS[zone],
            linestyle=ZONE_LINES[zone],
            label=country,
        )
        for zone, (color, country) in ZONES.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(handles),
        frameon=False,
    )
    fig.set_layout_engine("constrained", rect=(0, 0.065, 1, 0.935))
    fig.suptitle("Daily variation offers more room for shifting")
    _results_typography(fig)
    return fig


def save_swing_figure(savings: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_swing_figure(savings, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _swing_figure(savings)
        for suffix in ("pdf", "png"):
            figure.savefig(
                destination / suffix / f"swing_savings.{suffix}", bbox_inches="tight"
            )
        plt.close(figure)
