"""Render the shared policy comparison, execution-intensity diagnostic, and horizon sensitivity."""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import POLICY_COLORS, RESULT_STYLE, pale

POLICIES = [
    ("Greenfilling", "Greenfilling", POLICY_COLORS["Greenfilling"], 0),
    ("CATS", "CATS", POLICY_COLORS["CATS"], 1.3),
    ("Head", "Head-job shifting", POLICY_COLORS["Head"], 2.6),
    ("Offline", "Offline shifting", POLICY_COLORS["Offline"], 3.9),
]
FORECASTS = [("previous_day", -0.18), ("known_future", 0.18)]


def main_groups(results):
    for policy, label, color, position in POLICIES:
        rows = results.loc[
            (results.policy == policy) & ((results.hours == 6) | (results.hours == 0))
        ]
        if policy in ("CATS", "Head"):
            for forecast, offset in FORECASTS:
                yield (
                    label,
                    color,
                    position + offset,
                    forecast,
                    rows.loc[rows.forecast == forecast],
                )
        else:
            yield label, color, position, "reference", rows


def boxplot(ax, values, position, color, forecast):
    return ax.boxplot(
        values,
        positions=[position],
        widths=0.26,
        orientation="horizontal",
        whis=1.5,
        zorder=2,
        showfliers=True,
        patch_artist=True,
        manage_ticks=False,
        boxprops={
            "facecolor": "white" if forecast == "previous_day" else pale(color),
            "edgecolor": color,
            "linewidth": 1.4,
        },
        medianprops={"color": color, "linewidth": 2.3},
        whiskerprops={"color": color, "linewidth": 1.1},
        capprops={"color": color, "linewidth": 1.1},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "markeredgewidth": 0.7,
            "alpha": 0.65,
        },
    )


def forecast_legend(fig, anchor=0.59):
    handles = [
        Patch(facecolor="white", edgecolor="#555555", label="Previous 24 h repeated"),
        Patch(facecolor="#cccccc", edgecolor="#555555", label="Known future intensity"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(anchor, 0.01),
        ncol=2,
        frameon=False,
        fontsize=11,
        handlelength=1.4,
        columnspacing=2.5,
    )


def policy_axes(ax, policies):
    ax.set_yticks(
        [position for _, _, _, position in policies],
        [label for _, label, _, _ in policies],
    )
    ax.tick_params(axis="y", length=0, pad=12, labelsize=12)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(".7")
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=".5", lw=0.9, ls="--", zorder=0)
    ax.margins(x=0.06)


def heading(fig, title):
    fig.suptitle(title, x=0.025, y=0.98, ha="left", fontsize=20, fontweight="bold")


def panel_heading(ax, title, cue):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=27)
    ax.text(0, 1.025, cue, transform=ax.transAxes, fontsize=11, color=".35")


def comparison_figure(results, bounds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.83, bottom=0.18, wspace=0.18)
    for ax, column, title, cue in zip(
        axes,
        ["saving_pct", "response_pct"],
        ["Carbon saving", "Area-weighted response time"],
        [r"More saving $\longrightarrow$", r"$\longleftarrow$ Less time in system"],
    ):
        for _, color, position, forecast, rows in main_groups(results):
            boxplot(ax, rows[column], position, color, forecast)
        policy_axes(ax, POLICIES)
        panel_heading(ax, title, cue)
        ax.set_xlabel(
            "Reduction from EASY (%)"
            if column == "saving_pct"
            else "Increase from EASY (%)",
            labelpad=10,
        )
    bound = bounds.loc[bounds.hours == 6, "bound_saving_pct"]
    boxplot(axes[0], bound, 5.3, POLICY_COLORS["Bound"], "reference")
    axes[0].set_yticks(
        [0, 1.3, 2.6, 3.9, 5.3],
        [
            "Greenfilling",
            "CATS",
            "Head-job shifting",
            "Offline shifting",
            "Delay-only bound",
        ],
    )
    axes[0].set_ylim(5.85, -0.55)
    heading(fig, "Carbon savings and their response-time cost")
    forecast_legend(fig)
    return fig


def moved_figure(results):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.23, right=0.97, top=0.82, bottom=0.23)
    for label, color, position, forecast, rows in main_groups(results):
        if label != "Offline shifting":
            boxplot(ax, -rows.moved_intensity_pct, position, color, forecast)
    policy_axes(ax, POLICIES[:3])
    ax.set_ylim(3.15, -0.5)
    ax.set_xlabel(
        "Reduction in shifted jobs' mean execution intensity vs. EASY (%)", labelpad=10
    )
    ax.text(
        0, 1.06, r"$\longleftarrow$ Dirtier hours", transform=ax.transAxes, color=".35"
    )
    ax.text(
        1,
        1.06,
        r"Cleaner hours $\longrightarrow$",
        transform=ax.transAxes,
        ha="right",
        color=".35",
    )
    heading(fig, "CATS shifts jobs into cleaner hours")
    forecast_legend(fig)
    return fig


def horizon_figure(results):
    hours = [3, 6, 12, 24]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.80, bottom=0.20, wspace=0.23)
    handles = []
    for (policy, label, color, _), offset in zip(POLICIES[2:], [-0.18, 0.18]):
        rows = results.loc[
            (results.policy == policy)
            & results.forecast.isin(["known_future", "known_future_runtime"])
        ]
        for position, limit in enumerate(hours):
            group = rows.loc[rows.hours == limit]
            if group.empty:
                continue
            for ax, column in zip(axes, ["saving_pct", "response_pct"]):
                boxplot(ax, group[column], position + offset, color, "reference")
        handles.append(Patch(facecolor=pale(color), edgecolor=color, label=label))
    for ax in axes:
        ax.set_yticks(range(len(hours)), [f"{limit} h" for limit in hours])
        ax.set_ylim(3.6, -0.6)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=".5", lw=0.9, ls="--", zorder=0)
        ax.margins(x=0.06)
    axes[0].set_ylabel("Shifting limit", labelpad=12)
    panel_heading(axes[0], "Carbon saving", r"More saving $\longrightarrow$")
    panel_heading(
        axes[1], "Area-weighted response time", r"$\longleftarrow$ Less time in system"
    )
    axes[0].set_xlabel("Reduction from EASY (%)", labelpad=10)
    axes[1].set_xlabel("Increase from EASY (%)", labelpad=10)
    heading(fig, "Longer shifting limits increase time in the system")
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.01),
        ncol=2,
        frameon=False,
    )
    return fig


def save_comparison_figures(results, bounds, destination):
    for fmt in ("png", "pdf"):
        (destination / fmt).mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RESULT_STYLE):
        for name, figure in [
            ("policy_comparison", comparison_figure(results, bounds)),
            ("moved_job_intensity", moved_figure(results)),
            ("horizon_sensitivity", horizon_figure(results)),
        ]:
            for fmt in ("png", "pdf"):
                figure.savefig(destination / fmt / f"{name}.{fmt}", bbox_inches="tight")
            plt.close(figure)
