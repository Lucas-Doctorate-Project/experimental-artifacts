"""Matplotlib rendering for each policy's saving from jobs delayed on purpose and from all other jobs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plot_style import POLICY_COLORS, pale
from results_plots import RESULT_STYLE, _draw_boxes, _results_typography

POLICIES = {"Job shifting": POLICY_COLORS["Head"], "CATS": POLICY_COLORS["CATS"]}
CONTROL = "#999999"
# vertical position of each policy block; a block holds two rows of two boxes
BLOCKS = {"Job shifting": 0.0, "CATS": 2.6}


def _row_labels(rows: pd.DataFrame) -> tuple[str, str]:
    jobs, job_pct, node_hour_pct = (
        rows.delayed_jobs.median(),
        rows.delayed_job_pct.median(),
        rows.delayed_node_hour_pct.median(),
    )
    share = f"{job_pct:.0f}%" if job_pct >= 1 else f"{job_pct:.1f}%"
    return (
        f"Jobs delayed on purpose\nmedian {jobs:,.0f} jobs ({share}), {node_hour_pct:.0f}% of node-hours",
        "All other jobs",
    )


def _held_figure(splits: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(9, 6.2), layout="constrained")
    ticks, labels = [], []
    for policy, color in POLICIES.items():
        rows = splits.loc[splits.policy == policy]
        base = BLOCKS[policy]
        for row, (column, label) in enumerate(zip(("delayed_pct", "others_pct"), _row_labels(rows.loc[rows.scoring == "real"]))):
            center = base + row
            for offset, scoring, box_color in ((-0.17, "real", color), (0.17, "unrelated", CONTROL)):
                values = rows.loc[rows.scoring == scoring, column].to_numpy()
                _draw_boxes(ax, values, [center + offset], box_color, 0.3, "horizontal", 0.55)
            ticks.append(center)
            labels.append(label)
        ax.text(
            -0.02,
            base - 0.62,
            policy,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            color=color,
            fontsize=13,
            fontweight="bold",
        )
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(BLOCKS["CATS"] + 1.6, BLOCKS["Job shifting"] - 0.9)
    ax.axvline(0, color=".5", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Contribution to carbon saving (% of total emissions)")
    ax.text(0, 1.02, r"More saving $\longrightarrow$", transform=ax.transAxes, fontsize=11, color=".35")
    ax.text(
        1,
        1.02,
        "6 h limit, known future intensity, 144 workload and window pairs",
        transform=ax.transAxes,
        ha="right",
        fontsize=10.5,
        color=".35",
    )
    ax.legend(
        handles=[
            *(
                Patch(facecolor=pale(color, 0.55), edgecolor=color, label=f"{policy}, real intensity data")
                for policy, color in POLICIES.items()
            ),
            Patch(facecolor=pale(CONTROL, 0.55), edgecolor=CONTROL, label="Unrelated intensity data (control)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Where job shifting and CATS save carbon")
    _results_typography(fig)
    return fig


def save_held_jobs_figure(splits: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_held_jobs_figure(splits, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _held_figure(splits)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"held_jobs.{suffix}", bbox_inches="tight")
        plt.close(figure)
