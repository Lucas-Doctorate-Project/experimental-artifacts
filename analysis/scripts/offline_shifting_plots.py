"""Matplotlib rendering for offline shifting against Greenfilling and the oracle."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from results_plots import RESULT_STYLE, _draw_boxes, _results_typography

GREENFILLING = "#7a7a7a"
ORACLE = "#2677ad"
OFFLINE = "#b85d18"


def _groups(results: pd.DataFrame, points: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Name, color, and per-pair results of each policy, in display order."""
    original = results.loc[results.group == "original"]
    groups = [("Greenfilling", GREENFILLING, points.loc[points.policy == "Greenfilling"])]
    for hours in (6, 12, 24):
        rows = points.loc[points.policy == f"Oracle head shift {hours} h"]
        groups.append((f"Oracle, {hours}h horizon", ORACLE, rows))
    for hours in (3, 6, 12, 24):
        groups.append((f"Offline shifting, {hours}h budget", OFFLINE, original.loc[original.budget_hours == hours]))
    return groups


def _boxes(ax: Axes, groups: list[tuple[str, str, pd.DataFrame]], column: str, label: str) -> None:
    for position, (_, color, rows) in enumerate(groups):
        _draw_boxes(ax, rows[column].to_numpy(), [position], color, 0.65, "horizontal", 0.55)
    ax.set_yticks(range(len(groups)), [name for name, _, _ in groups])
    ax.set_xlabel(label)
    ax.grid(axis="y", visible=False)


def _offline_figure(results: pd.DataFrame, points: pd.DataFrame) -> Figure:
    groups = _groups(results, points)
    fig, (saving, wait) = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, layout="constrained")
    _boxes(saving, groups, "saving_pct", "Targeted footprint saving relative to EASY (%)")
    _boxes(wait, groups, "added_wait_hours", "Mean added wait per job relative to EASY (h)")
    saving.axvline(0, color=".3", linewidth=0.8)
    wait.axvline(0, color=".3", linewidth=0.8)
    saving.invert_yaxis()
    fig.suptitle("Larger savings need much longer waits")
    _results_typography(fig)
    return fig


def save_offline_shifting_figure(results: pd.DataFrame, points: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_offline_shifting_figure(results, points, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _offline_figure(results, points)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"offline_shifting.{suffix}", bbox_inches="tight")
        plt.close(figure)
