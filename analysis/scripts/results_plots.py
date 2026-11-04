"""Matplotlib rendering for footprint and service results."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.text import Text

RESULT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}

PAIRED_GROUPS = [
    ("Mustang", "original", "mustang"),
    ("Mustang, resampled\n(more large jobs)", "resampled", "mustang"),
    ("Trinity", "original", "trinity"),
]
PLOT_METRICS = [
    ("target_delta_pct", "Targeted footprint"),
    ("bounded_slowdown_delta_pct", "Mean bounded slowdown"),
    ("weighted_response_seconds_delta_pct", "Work-weighted response time"),
]


def _results_typography(fig: Figure) -> None:
    """Apply the bundled Roboto Condensed font to results figures only."""
    from matplotlib import font_manager
    from matplotlib.text import Text

    font_path = Path(__file__).resolve().parents[1] / "fonts/RobotoCondensed-Regular.ttf"
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=font_path).get_name()
    for text in fig.findobj(Text):
        text.set_fontfamily(family)
        text.set_fontsize(12)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.title.set_fontsize(14)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(14)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(16)


def _title_block(fig: Figure, title: str, subtitle: str, top: float) -> Text:
    """Place a title and a gray explanatory subtitle above axes that end at top."""
    fig.suptitle(title, y=0.995)
    lines = subtitle.count("\n") + 1
    return fig.text(0.5, top + 0.014 * lines + 0.012, subtitle, ha="center", va="center", color=".3")


def _box_style(color: str, opacity: float) -> dict[str, dict[str, str | float]]:
    return {
        "boxprops": {"facecolor": color, "edgecolor": color, "alpha": opacity},
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
        whis=(0, 100),
        showfliers=False,
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
    ax.axvline(0, color=".35", lw=0.7, ls="--")
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
    # rect is (left, bottom, width, height), so the axes end at 0.9 below the subtitle
    fig.set_layout_engine("constrained", rect=(0, 0.12, 1, 0.78))
    shared_xlabel = fig.supxlabel(
        "Change from EASY on the same workload and grid window (%)", y=0.065
    )
    subtitle = _title_block(
        fig,
        "Greenfilling relative to EASY backfilling",
        "Each box spans 36 four-week grid windows, whiskers at the full range."
        " Negative footprint change is a saving.",
        0.9,
    )
    _results_typography(fig)
    subtitle.set_fontsize(13)
    shared_xlabel.set_fontsize(14)
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
        Patch(facecolor=color, edgecolor=color, alpha=opacity, label=label.capitalize())
        for label, color in colors.items()
    ]


ALPHA_COLORS = {
    "carbon slack": "#dba34d",
    "carbon stress": "#b85d18",
    "water slack": "#56b4c9",
    "water stress": "#2677ad",
}
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
    ax.axhline(0, color=".3", lw=0.7, ls="--")
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
    shared_xlabel = fig.supxlabel("Smoothing factor α of Greenfilling's moving average", y=0.075)
    fig.suptitle("Greenfilling smoothing factor relative to EASY")
    _results_typography(fig)
    shared_xlabel.set_fontsize(14)
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
