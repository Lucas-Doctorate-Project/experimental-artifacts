"""Matplotlib rendering for each policy's gain and loss from jobs that ran cleaner and dirtier."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from plot_style import POLICY_COLORS, pale
from results_plots import RESULT_STYLE, _results_typography

POLICIES = {"Job shifting": POLICY_COLORS["Head"], "CATS": POLICY_COLORS["CATS"]}
CONTROL = "#999999"
# size class: marker and how much of the policy color it keeps
SIZES = {"small": ("o", 0.35), "medium": ("s", 0.65), "large": ("^", 1.0)}
SIZE_LABELS = {
    "small": "Small jobs: the rest, 10% of node-hours",
    "medium": "Medium jobs: the next 40% of node-hours",
    "large": "Large jobs: the biggest, half of node-hours",
}


def _diagonal(ax, top: float) -> None:
    ax.plot([0, top], [0, top], color=".3", linewidth=0.8, linestyle="--", zorder=0)
    ax.set(xlim=(0, top), ylim=(0, top), aspect="equal")
    ax.text(0.04, 0.93, "Net increase", transform=ax.transAxes, color=".35", fontsize=10)
    ax.text(0.96, 0.05, "Net saving", transform=ax.transAxes, ha="right", color=".35", fontsize=10)


def _gain_loss_figure(scores: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6), layout="constrained")
    whole = scores.loc[scores["size"] == "all"]
    sized = scores.loc[scores["size"] != "all"]
    tops = [1.05 * rows[["gain_pct", "loss_pct"]].to_numpy().max() for rows in (whole, sized)]
    for column, (policy, color) in enumerate(POLICIES.items()):
        rows = scores.loc[scores.policy == policy]
        ax = axes[0, column]
        for scoring, point_color in (("unrelated", CONTROL), ("real", color)):
            selected = rows.loc[(rows.scoring == scoring) & (rows["size"] == "all")]
            ax.scatter(selected.gain_pct, selected.loss_pct, s=14, color=point_color, alpha=0.75, linewidths=0)
        ax.set_title(f"{policy}, all jobs", color=color, fontsize=13, fontweight="bold")
        ax = axes[1, column]
        for size, (marker, amount) in SIZES.items():
            selected = rows.loc[rows["size"] == size]
            ax.scatter(
                selected.gain_pct,
                selected.loss_pct,
                s=16,
                marker=marker,
                color=pale(color, amount),
                edgecolor=color,
                linewidths=0.4,
            )
        ax.set_title(f"{policy}, by job size", color=color, fontsize=13, fontweight="bold")
    for row, top in enumerate(tops):
        for ax in axes[row]:
            _diagonal(ax, top)
    for ax in axes[1]:
        ax.set_xlabel("Gain from jobs that ran cleaner\n(% of total emissions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Loss from jobs that ran dirtier\n(% of total emissions)")
    axes[0, 1].legend(
        handles=[
            *(
                Line2D([], [], marker="o", linestyle="", color=color, label=f"{policy}, real intensity data")
                for policy, color in POLICIES.items()
            ),
            Line2D([], [], marker="o", linestyle="", color=CONTROL, label="Unrelated intensity data (control)"),
            Line2D([], [], color=".3", linewidth=0.8, linestyle="--", label="Loss equals gain"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
    )
    axes[1, 1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markerfacecolor=pale("#555555", amount),
                markeredgecolor="#555555",
                label=SIZE_LABELS[size],
            )
            for size, (marker, amount) in SIZES.items()
        ][::-1],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=False,
        title="Real intensity data, per workload",
        alignment="left",
    )
    fig.suptitle("Delaying jobs moves other jobs into cleaner and dirtier hours")
    axes[0, 1].text(
        1,
        1.09,
        "6 h limit, known future intensity, 144 workload and window pairs",
        transform=axes[0, 1].transAxes,
        ha="right",
        fontsize=10.5,
        color=".35",
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
            figure.savefig(destination / suffix / f"queue_gain_loss.{suffix}", bbox_inches="tight")
        plt.close(figure)
