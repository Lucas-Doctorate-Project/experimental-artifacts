"""Matplotlib rendering for footprint and service results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plot_style import CONDITION_COLORS, RESULT_STYLE, pale
from plot_style import typography as _results_typography

PAIRED_GROUPS = [
    ("Mustang", "original", "mustang"),
    ("Mustang, resampled\n(more large jobs)", "resampled", "mustang"),
    ("Trinity", "original", "trinity"),
]
PLOT_METRICS = [
    ("target_delta_pct", "Targeted footprint"),
    ("bounded_slowdown_delta_pct", "Mean bounded slowdown"),
    ("weighted_response_seconds_delta_pct", "Area-weighted response time"),
]


def _box_style(color: str, opacity: float) -> dict[str, dict[str, str | float]]:
    return {
        "boxprops": {"facecolor": pale(color, opacity), "edgecolor": color},
        "medianprops": {"color": color, "linewidth": 2},
        "whiskerprops": {"color": color},
        "capprops": {"color": color},
    }


def _draw_boxes(
    ax: Axes,
    values: np.ndarray | list[np.ndarray],
    positions: list[float] | np.ndarray,
    color: str,
    width: float = 0.24,
    orientation: str = "horizontal",
    opacity: float = 0.45,
) -> None:
    ax.boxplot(
        values,
        positions=positions,
        widths=width,
        orientation=orientation,
        whis=1.5,
        showfliers=True,
        zorder=2,
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "markeredgewidth": 0.6,
        },
        patch_artist=True,
        manage_ticks=False,
        **_box_style(color, opacity),
    )


def _paired_metric(ax: Axes, pairs: pd.DataFrame, metric: str) -> None:
    for position, load in enumerate(("slack", "stress")):
        subset = pairs.loc[pairs.workload.str.endswith(f"_{load}")]
        for objective, offset in (("carbon", -0.16), ("water", 0.16)):
            values = subset.loc[subset.objective == objective, metric].to_numpy()
            center = position + offset
            _draw_boxes(
                ax,
                values,
                [center],
                ALPHA_COLORS[f"{objective} {load}"],
                0.24,
                "horizontal",
                0.55,
            )


def _paired_panel(ax: Axes, pairs: pd.DataFrame, metric: str, title: str) -> None:
    _paired_metric(ax, pairs, metric)
    ax.axvline(0, color=".5", lw=0.8, ls="--", zorder=0)
    ax.set_yticks([])
    ax.set_ylim(1.55, -0.55)
    ax.set_title(title)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def _paired_group_row(axes: np.ndarray, pairs: pd.DataFrame, row: int) -> None:
    label, group, machine = PAIRED_GROUPS[row]
    subset = pairs.loc[
        (pairs.group == group) & pairs.workload.str.startswith(f"{machine}_")
    ]
    for ax, (metric, title) in zip(axes, PLOT_METRICS):
        _paired_panel(ax, subset, metric, title if row == 0 else "")
    axes[0].set_ylabel(label)


def _paired_figure(pairs: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 8), layout="constrained", sharex="col", sharey="row"
    )
    for row in range(len(PAIRED_GROUPS)):
        _paired_group_row(axes[row], pairs, row)
    _figure_legend(fig, ALPHA_COLORS, 0.55, 4)
    fig.set_layout_engine("constrained", rect=(0, 0.12, 1, 0.88))
    fig.supxlabel("Change from EASY (%)   |   Lower is better", y=0.065)
    fig.suptitle("Greenfilling's footprint and service costs")
    _results_typography(fig)
    return fig


def _figure_legend(
    fig: Figure, colors: dict[str, str], opacity: float, columns: int
) -> None:
    fig.legend(
        handles=_legend_patches(colors, opacity),
        loc="lower center",
        ncol=columns,
        frameon=False,
    )


def _legend_patches(colors: dict[str, str], opacity: float) -> list[Patch]:
    return [
        Patch(facecolor=pale(color, opacity), edgecolor=color, label=label.capitalize())
        for label, color in colors.items()
    ]


ALPHA_COLORS = CONDITION_COLORS
ALPHAS = [0.3, 0.5, 0.8]


def _alpha_panel(
    ax: Axes, sensitivity: pd.DataFrame, machine: str, metric: str
) -> None:
    positions = np.arange(len(ALPHAS))
    for label, offset in zip(ALPHA_COLORS, [-0.27, -0.09, 0.09, 0.27]):
        objective, load = label.split()
        subset = sensitivity.loc[
            (sensitivity.objective == objective)
            & (sensitivity.workload == f"{machine}_{load}")
        ]
        values = [
            subset.loc[subset.alpha == alpha, metric].to_numpy() for alpha in ALPHAS
        ]
        _draw_boxes(
            ax, values, positions + offset, ALPHA_COLORS[label], 0.15, "vertical", 0.55
        )
    ax.axhline(0, color=".5", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(positions, [str(alpha) for alpha in ALPHAS])
    ax.set_xlim(-0.55, len(ALPHAS) - 0.45)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _alpha_figure(sensitivity: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(
        2, 3, figsize=(11.5, 6.2), layout="constrained", sharex="col"
    )
    for row, machine in enumerate(("mustang", "trinity")):
        _alpha_row(axes[row], sensitivity, machine, row == 0)
    _figure_legend(fig, ALPHA_COLORS, 0.55, 4)
    fig.set_layout_engine("constrained", rect=(0, 0.13, 1, 0.87))
    shared_xlabel = fig.supxlabel(
        "Smoothing factor α of Greenfilling's moving average", y=0.075
    )
    fig.suptitle("Greenfilling sensitivity to its moving average")
    _results_typography(fig)
    shared_xlabel.set_fontsize(12)
    return fig


def _alpha_row(
    axes: np.ndarray, sensitivity: pd.DataFrame, machine: str, show_title: bool
) -> None:
    for ax, (metric, title) in zip(axes, PLOT_METRICS):
        _alpha_panel(ax, sensitivity, machine, metric)
        if show_title:
            ax.set_title(title)
    axes[0].set_ylabel(machine.capitalize() + "\nChange from EASY (%)")


def save_results_figures(
    pairs: pd.DataFrame, sensitivity: pd.DataFrame, destination: Path
) -> None:
    """Render both figures without a display. Example: save_results_figures(pairs, sensitivity, Path('/tmp/results'))."""
    for suffix in ("pdf", "png"):
        (destination / suffix).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        for name, figure in [
            ("paired_results", _paired_figure(pairs)),
            ("alpha_sensitivity_combined", _alpha_figure(sensitivity)),
        ]:
            for suffix in ("pdf", "png"):
                figure.savefig(
                    destination / suffix / f"{name}.{suffix}", bbox_inches="tight"
                )
            plt.close(figure)
