"""Matplotlib rendering for the available saving and the saving Greenfilling achieved."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from results_plots import ALPHA_COLORS, RESULT_STYLE, _legend_patches, _results_typography

MACHINE_MARKERS = {"mustang": "o", "trinity": "s"}
POLICIES = {"Greenfilling"}
BUDGET_LABELS = (1, 3, 6, 12, 24)
# inset positions in axes fractions, in areas without bound markers
INSETS = {"mustang": [0.57, 0.06, 0.2, 0.3], "trinity": [0.05, 0.6, 0.18, 0.32]}
ZOOM_PADDING = 0.3


def _quartiles(values: pd.Series) -> pd.Series:
    return pd.Series({"q1": values.quantile(0.25), "median": values.median(), "q3": values.quantile(0.75)})


def _marker_style(color: str, marker: str, achieved: bool) -> dict:
    """Hollow outline for the available saving, filled with a dark edge for Greenfilling."""
    return {
        "fmt": marker,
        "color": color,
        "markerfacecolor": color if achieved else "white",
        "markeredgecolor": ".15" if achieved else color,
        "markeredgewidth": 1 if achieved else 1.6,
        "markersize": 8 if achieved else 6,
        "elinewidth": 1 if achieved else 0.8,
        "capsize": 2,
        "zorder": 3 if achieved else 2,
    }


def _bound_markers(ax: Axes, bounds: pd.DataFrame, color: str, marker: str) -> None:
    by_delay = bounds.groupby("delay_hours")
    shift = by_delay.mean_shift_hours.median()
    saving = by_delay.savings_pct.apply(_quartiles).unstack()
    ax.errorbar(
        shift,
        saving["median"],
        yerr=[saving["median"] - saving.q1, saving.q3 - saving["median"]],
        **_marker_style(color, marker, achieved=False),
    )


def _realized_marker(ax: Axes, points: pd.DataFrame, color: str, marker: str) -> None:
    wait = _quartiles(points.added_wait_hours)
    saving = _quartiles(points.saving_pct)
    ax.errorbar(
        wait["median"],
        saving["median"],
        xerr=[[wait["median"] - wait.q1], [wait.q3 - wait["median"]]],
        yerr=[[saving["median"] - saving.q1], [saving.q3 - saving["median"]]],
        **_marker_style(color, marker, achieved=True),
    )


def _budget_labels(ax: Axes, bounds: pd.DataFrame) -> None:
    by_case = bounds.groupby(["delay_hours", "workload", "objective"])
    shift = by_case.mean_shift_hours.median().groupby("delay_hours").mean()
    top = by_case.savings_pct.quantile(0.75).groupby("delay_hours").max()
    for delay in BUDGET_LABELS:
        ax.annotate(
            f"{delay:g}h",
            (shift[delay], top[delay]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            color=".4",
        )


def _draw(ax: Axes, bounds: pd.DataFrame, points: pd.DataFrame) -> None:
    for (objective, workload), curve in bounds.groupby(["objective", "workload"]):
        machine, load = workload.split("_")
        color = ALPHA_COLORS[f"{objective} {load}"]
        marker = MACHINE_MARKERS[machine]
        _bound_markers(ax, curve, color, marker)
        realized = points.loc[(points.objective == objective) & (points.workload == workload)]
        if len(realized):
            _realized_marker(ax, realized, color, marker)


def _zoom_limits(points: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """Box around the interquartile ranges of the given points, including zero saving."""
    by_case = points.groupby(["workload", "objective"])
    x0, x1 = by_case.added_wait_hours.quantile(0.25).min(), by_case.added_wait_hours.quantile(0.75).max()
    y0 = min(by_case.saving_pct.quantile(0.25).min(), 0)
    y1 = max(by_case.saving_pct.quantile(0.75).max(), 0)
    (dx, dy) = (ZOOM_PADDING * (x1 - x0), ZOOM_PADDING * (y1 - y0))
    return (max(x0 - dx, 0), x1 + dx), (y0 - dy, y1 + dy)


def _zoom_inset(ax: Axes, bounds: pd.DataFrame, points: pd.DataFrame, machine: str) -> None:
    inset = ax.inset_axes(INSETS[machine])
    _draw(inset, bounds, points)
    xlim, ylim = _zoom_limits(points.loc[points.workload.str.startswith(f"{machine}_")])
    inset.set(xlim=xlim, ylim=ylim)
    # reuse the main axes ticks so the inset grid lines continue the main grid
    inset.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
    inset.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
    inset.spines[["top", "right"]].set_visible(True)
    inset.tick_params(labelleft=False, labelbottom=False)
    ax.indicate_inset_zoom(inset, edgecolor=".4")


def _style_handles() -> list[Line2D]:
    machines = [
        Line2D([], [], color=".3", marker=marker, ls="", markersize=8, label=machine.capitalize())
        for machine, marker in MACHINE_MARKERS.items()
    ]
    return machines + [
        Line2D([], [], color=".3", marker="o", ls="", markerfacecolor="white", markeredgewidth=1.6, label="Available saving"),
        Line2D([], [], color=".6", marker="o", ls="", markeredgecolor=".15", markersize=8, label="Greenfilling"),
    ]


def _opportunity_figure(bounds: pd.DataFrame, points: pd.DataFrame) -> Figure:
    unknown = set(points.policy) - POLICIES
    if unknown:
        raise ValueError(f"unsupported policies {sorted(unknown)}, expected {sorted(POLICIES)}")
    fig, ax = plt.subplots(figsize=(11.5, 6.4), layout="constrained")
    _draw(ax, bounds, points)
    _budget_labels(ax, bounds)
    ax.set_xlim(left=0)
    for machine in INSETS:
        _zoom_inset(ax, bounds, points, machine)
    ax.set_xlabel("Mean added wait per job relative to EASY (h)")
    ax.set_ylabel("Targeted footprint saving relative to EASY (%)")
    fig.legend(
        handles=_legend_patches(ALPHA_COLORS, 0.55),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=4,
        frameon=False,
    )
    fig.legend(handles=_style_handles(), loc="lower center", bbox_to_anchor=(0.5, 0), ncol=4, frameon=False)
    fig.set_layout_engine("constrained", rect=(0, 0.14, 1, 0.86))
    fig.suptitle("Footprint saving from delaying jobs: available versus Greenfilling")
    _results_typography(fig)
    return fig


def save_opportunity_figure(bounds: pd.DataFrame, points: pd.DataFrame, destination: Path) -> None:
    """Render the figure without a display. Example: save_opportunity_figure(bounds, points, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        figure = _opportunity_figure(bounds, points)
        for suffix in ("pdf", "png"):
            figure.savefig(destination / suffix / f"opportunity_capture.{suffix}", bbox_inches="tight")
        plt.close(figure)
