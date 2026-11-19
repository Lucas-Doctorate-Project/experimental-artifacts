"""Matplotlib rendering for the saving of delayed jobs by share of requested time used."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from plot_style import POLICY_COLORS, pale
from results_plots import RESULT_STYLE, _results_typography

# share of requested time used: lower edge, upper edge, label
SHARES = [
    (-np.inf, 0.1, "Under 10%"),
    (0.1, 0.5, "10 to 50%"),
    (0.5, 0.9, "50 to 90%"),
    (0.9, np.inf, "90% or more"),
]
POLICIES = {"Job shifting": POLICY_COLORS["Head"], "CATS": POLICY_COLORS["CATS"]}
VALUES = {
    "expected": "Expected, with requested time",
    "real": "Real, with actual run time",
    "best": "Best possible, with actual run time",
}


def _bar_style(value: str, color: str) -> dict:
    face = {"expected": "white", "real": color, "best": pale(color)}[value]
    return {"facecolor": face, "edgecolor": color, "linewidth": 0.8, "hatch": "///" if value == "best" else None}


def _held_walltime_figure(jobs: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, layout="constrained")
    for ax, (policy, color) in zip(axes, POLICIES.items()):
        rows = jobs.loc[jobs.policy == policy]
        runs = rows[["workload", "trace"]].drop_duplicates().shape[0]
        labels = []
        for position, (_, _, label) in enumerate(SHARES):
            group = rows.loc[rows.share_class == label]
            for offset, value in zip((-0.27, 0, 0.27), VALUES):
                ax.bar(position + offset, group[value].sum() / runs, 0.27, zorder=2, **_bar_style(value, color))
            labels.append(
                f"{label}\n{100 * len(group) / len(rows):.0f}% of jobs\n"
                f"{100 * group.node_hours.sum() / rows.node_hours.sum():.0f}% of node-hours"
            )
        ax.set_xticks(range(len(SHARES)), labels)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=".5", linewidth=0.8, zorder=1)
        ax.set_title(policy, color=color, fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Share of requested time used")
    axes[0].set_ylabel("Saving from jobs delayed on purpose\n(% of total emissions, mean per run)")
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, label=label, **_bar_style(value, "#555555"))
            for value, label in VALUES.items()
        ],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Knowing actual run times would add little saving")
    axes[1].text(
        1,
        1.06,
        "6 h limit, known future intensity, 144 workload and window pairs",
        transform=axes[1].transAxes,
        ha="right",
        fontsize=10.5,
        color=".35",
    )
    _results_typography(fig)
    return fig


def save_held_walltime_figure(jobs: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_held_walltime_figure(jobs, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _held_walltime_figure(jobs)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"held_walltime.{suffix}", bbox_inches="tight")
        plt.close(figure)
