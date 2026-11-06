"""Matplotlib rendering for the job-class response-time diagnostic."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from results_plots import RESULT_STYLE, _results_typography

POLICIES = {
    "Greenfilling": "Greenfilling",
    "Oracle head shift 6 h": "Head-job shifting",
}
CLASS_POLICIES = ["Greenfilling", "Oracle head shift 6 h"]
CLASS_COLORS = {"narrow-short": "#d98637", "other": "#4792bd", "wide": "#8660a8"}
CLASS_HATCHES = {"narrow-short": "///", "other": "", "wide": "..."}


def _class_figure(classes: pd.DataFrame) -> Figure:
    replay = classes.loc[
        (classes["class"] != "context") & classes.policy.isin(CLASS_POLICIES)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained", sharey=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("slowdown_contribution", "Bounded slowdown increase"),
            ("wrt_contribution_seconds", "Area-weighted response-time increase"),
        ],
    ):
        totals = (
            replay.groupby(["policy", "class"])[metric]
            .sum()
            .unstack()
            .loc[CLASS_POLICIES]
        )
        shares = 100 * totals.div(totals.sum(axis=1), axis=0)
        names = [POLICIES[policy] for policy in CLASS_POLICIES]
        bottom = np.zeros(len(shares))
        for job_class, color in CLASS_COLORS.items():
            ax.bar(
                names,
                shares[job_class],
                bottom=bottom,
                color=color,
                edgecolor="white",
                hatch=CLASS_HATCHES[job_class],
                label=job_class.capitalize(),
            )
            bottom += shares[job_class].to_numpy()
        ax.set_title(title)
    axes[0].set_ylabel("Share of the increase (%)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=3)
    fig.set_layout_engine("constrained", rect=(0, 0.09, 1, 0.91))
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle("Which jobs absorb the added delay")
    _results_typography(fig)
    return fig


def save_class_figure(classes: pd.DataFrame, destination: Path) -> None:
    """Render the job-class diagnostic from the paired schedules."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _class_figure(classes)
        for suffix in ("pdf", "png"):
            figure.savefig(
                destination / suffix / f"delay_by_job_class.{suffix}",
                bbox_inches="tight",
            )
        plt.close(figure)
