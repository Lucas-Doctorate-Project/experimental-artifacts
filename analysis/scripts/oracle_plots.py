"""Matplotlib rendering for the Greenfilling and oracle comparison figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from results_plots import RESULT_STYLE, _draw_boxes, _results_typography

# display name, color
POLICIES = {
    "Greenfilling": ("Greenfilling", "#7a7a7a"),
    "Oracle head shift 6 h": ("Oracle 6h", "#2677ad"),
    "Oracle head shift 12 h": ("Oracle 12h", "#2677ad"),
    "Oracle head shift 24 h": ("Oracle 24h", "#2677ad"),
}
AVAILABLE = ("Available, 24h budget", "#b85d18")
CLASS_POLICIES = ["Greenfilling", "Oracle head shift 6 h"]
CLASS_COLORS = {"narrow-short": "#56b4c9", "other": "#a6acb1", "wide": "#b85d18"}


def _quartiles(values: pd.Series) -> tuple[float, float, float]:
    return values.quantile(0.25), values.median(), values.quantile(0.75)


def _ladder_figure(savings: pd.DataFrame, bounds: pd.DataFrame) -> Figure:
    rows = [(name, color, savings.loc[savings.policy == policy, "saving_pct"]) for policy, (name, color) in POLICIES.items()]
    rows.append((*AVAILABLE, bounds.loc[bounds.delay_hours == 24, "savings_pct"]))
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    for position, (name, color, values) in enumerate(rows):
        low, median, high = _quartiles(values)
        ax.errorbar(median, position, xerr=[[median - low], [high - median]], fmt="o", color=color, markersize=9, capsize=3)
        ax.annotate(f"{median:.2f}%", (median, position), xytext=(0, 9), textcoords="offset points", ha="center", color=".3")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xticks([0, 0.1, 0.5, 1, 5, 10], ["0", "0.1", "0.5", "1", "5", "10"])
    ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Targeted footprint saving relative to EASY (%)")
    fig.suptitle("How much saving each policy captures")
    _results_typography(fig)
    return fig


def _moved_figure(moved: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for position, (policy, (name, color)) in enumerate(POLICIES.items()):
        values = moved.loc[moved.policy == policy, "intensity_change_pct"].dropna().to_numpy()
        _draw_boxes(ax, values, [position], color, 0.5, "vertical", 0.55)
    ax.axhline(0, color=".35", lw=0.7, ls="--")
    ax.set_xticks(range(len(POLICIES)), [name for name, _ in POLICIES.values()])
    ax.set_xlim(-0.6, len(POLICIES) - 0.4)
    ax.set_ylabel("Intensity change of moved jobs\nrelative to their EASY slot (%)")
    fig.suptitle("Whether moved jobs run in cleaner hours")
    _results_typography(fig)
    return fig


def _class_figure(classes: pd.DataFrame) -> Figure:
    replay = classes.loc[(classes["class"] != "context") & classes.policy.isin(CLASS_POLICIES)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", sharey=True)
    for ax, (metric, title) in zip(
        axes,
        [("slowdown_contribution", "Bounded slowdown increase"), ("wrt_contribution_seconds", "Response time increase")],
    ):
        totals = replay.groupby(["policy", "class"])[metric].sum().unstack().loc[CLASS_POLICIES]
        shares = 100 * totals.div(totals.sum(axis=1), axis=0)
        names = [POLICIES[policy][0] for policy in CLASS_POLICIES]
        bottom = np.zeros(len(shares))
        for job_class, color in CLASS_COLORS.items():
            ax.bar(names, shares[job_class], bottom=bottom, color=color, label=job_class.capitalize())
            bottom += shares[job_class].to_numpy()
        ax.set_title(title)
    axes[0].set_ylabel("Share of the increase (%)")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Which jobs absorb the added delay")
    _results_typography(fig)
    return fig


def save_oracle_figures(
    savings: pd.DataFrame, bounds: pd.DataFrame, moved: pd.DataFrame, classes: pd.DataFrame, destination: Path
) -> None:
    """Render the three figures without a display. Example: save_oracle_figures(savings, bounds, moved, classes, Path('/tmp/results'))."""
    unknown = set(savings.policy) - set(POLICIES)
    if unknown:
        raise ValueError(f"unsupported policies {sorted(unknown)}, expected {sorted(POLICIES)}")
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        for name, figure in [
            ("oracle_saving_ladder", _ladder_figure(savings, bounds)),
            ("moved_job_intensity", _moved_figure(moved)),
            ("delay_by_job_class", _class_figure(classes)),
        ]:
            for suffix in ("pdf", "png"):
                figure.savefig(destination / suffix / f"{name}.{suffix}", bbox_inches="tight")
            plt.close(figure)
