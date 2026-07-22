"""Shared loading, metrics, and plotting for the greenfilling trade-off notebook.

`greenfilling_tradeoffs.ipynb` imports this module for loading the campaign,
computing per-window paired deltas versus the EASY baseline, and plotting.
"""

import re

import numpy as np
import pandas as pd
import tomllib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# Average bounded slowdown uses this floor on execution time so very short jobs
# do not dominate the comparison: max(turnaround / max(execution, floor), 1).
BOUNDED_SLOWDOWN_FLOOR_SECONDS = 10.0
JOULES_PER_KWH = 3_600_000.0

METADATA_COLUMNS = [
    "name",
    "variant",
    "objective",
    "workload_label",
    "dataset",
    "regime",
    "zone",
    "start_date",
]

# Paired-delta columns reported in the summary table and per-window head.
DELTA_COLUMNS = [
    "greenfilling_variant",
    "total_carbon_footprint_delta_pct",
    "total_water_footprint_delta_pct",
    "consumed_energy_kwh_delta_pct",
    "energy_weighted_carbon_intensity_delta_pct",
    "energy_weighted_water_intensity_delta_pct",
    "makespan_delta_pct",
    "replay_median_waiting_time_delta_pct",
    "replay_p95_waiting_time_delta_pct",
    "replay_mean_bounded_slowdown_delta_pct",
]


# --- Parsing -----------------------------------------------------------------

def parse_experiment_name(name):
    """Split an experiment name into its variant/objective/workload/zone/date.

    The workload is a (dataset, regime) pair, e.g. `mustang_slack`. It is kept
    combined in `workload_label` so each dataset pairs against its own EASY
    baseline, and also split into `dataset` and `regime` for slicing.
    """
    baseline = re.fullmatch(
        r"easy_bf_([^_]+)_([^_]+)_([A-Z]{2})_(\d{4}-\d{2}-\d{2})", name
    )
    if baseline:
        dataset, regime, zone, start_date = baseline.groups()
        return {
            "variant": "easy_bf",
            "objective": "baseline",
            "workload_label": f"{dataset}_{regime}",
            "dataset": dataset,
            "regime": regime,
            "zone": zone,
            "start_date": start_date,
        }

    greenfilling = re.fullmatch(
        r"greenfilling_(carbon|water)_([^_]+)_([^_]+)_([A-Z]{2})_(\d{4}-\d{2}-\d{2})",
        name,
    )
    if greenfilling:
        objective, dataset, regime, zone, start_date = greenfilling.groups()
        return {
            "variant": f"greenfilling_{objective}",
            "objective": objective,
            "workload_label": f"{dataset}_{regime}",
            "dataset": dataset,
            "regime": regime,
            "zone": zone,
            "start_date": start_date,
        }

    raise ValueError(f"Unexpected experiment name: {name}")


# --- Loading -----------------------------------------------------------------

def load_campaign(campaign_path, windows_path):
    """Return (campaign, windows). `campaign` has the parsed name parts joined on."""
    with campaign_path.open("rb") as file:
        campaign = pd.DataFrame(tomllib.load(file)["experiment"])

    name_parts = pd.DataFrame(
        [parse_experiment_name(name) for name in campaign["name"]]
    )
    campaign = pd.concat([campaign, name_parts], axis=1)

    windows = pd.read_csv(windows_path)
    return campaign, windows


def check_completeness(campaign, out_dir):
    """One row per experiment, flagging which expected output files exist."""
    result_files = []
    for row in campaign.itertuples(index=False):
        experiment_dir = out_dir / row.name
        result_files.append(
            {
                "name": row.name,
                "variant": row.variant,
                "workload_label": row.workload_label,
                "zone": row.zone,
                "start_date": row.start_date,
                "has_directory": experiment_dir.exists(),
                "has_schedule": (experiment_dir / "out_schedule.csv").exists(),
                "has_jobs": (experiment_dir / "out_jobs.csv").exists(),
                "has_environmental_footprint": (
                    experiment_dir / "out_environmental_footprint.csv"
                ).exists(),
            }
        )
    return pd.DataFrame(result_files)


def completeness_summary(result_files):
    """Per (variant, workload) counts of present output files."""
    return result_files.groupby(["variant", "workload_label"])[
        ["has_directory", "has_schedule", "has_jobs", "has_environmental_footprint"]
    ].sum()


def missing_results(result_files):
    """Experiments lacking either the schedule or the jobs output."""
    return result_files[~(result_files["has_schedule"] & result_files["has_jobs"])]


def load_results(campaign, result_files, out_dir, windows):
    """Build the `(schedules, jobs)` tables for experiments with complete output.

    `schedules` has one row per experiment (windows merged in). `jobs` has one row
    per simulated job, labelled `context` (initial backlog, ids start with `ctx_`)
    or `replay`, with bounded slowdown precomputed.
    """
    campaign_metadata = campaign[METADATA_COLUMNS].set_index("name")
    complete_results = result_files[
        result_files["has_schedule"] & result_files["has_jobs"]
    ]

    schedule_frames = []
    job_frames = []

    for experiment_name in complete_results["name"]:
        experiment_dir = out_dir / experiment_name
        metadata = campaign_metadata.loc[experiment_name].to_dict()

        schedule = pd.read_csv(experiment_dir / "out_schedule.csv")
        schedule.insert(0, "name", experiment_name)
        for key, value in metadata.items():
            schedule[key] = value
        schedule_frames.append(schedule)

        experiment_jobs = pd.read_csv(experiment_dir / "out_jobs.csv")
        experiment_jobs.insert(0, "name", experiment_name)
        for key, value in metadata.items():
            experiment_jobs[key] = value
        job_frames.append(experiment_jobs)

    schedules = pd.concat(schedule_frames, ignore_index=True)
    jobs = pd.concat(job_frames, ignore_index=True)

    jobs["job_kind"] = jobs["job_id"].str.startswith("ctx_").map(
        {True: "context", False: "replay"}
    )
    jobs["bounded_slowdown"] = np.maximum(
        jobs["turnaround_time"]
        / jobs["execution_time"].clip(lower=BOUNDED_SLOWDOWN_FLOOR_SECONDS),
        1.0,
    )

    schedules = schedules.merge(
        windows,
        on=["zone", "start_date"],
        how="left",
        validate="many_to_one",
    )
    return schedules, jobs


# --- Metrics -----------------------------------------------------------------

def build_job_metrics(jobs):
    """Aggregate per (experiment, job_kind) job metrics."""
    return (
        jobs.groupby(
            ["name", "variant", "workload_label", "zone", "start_date", "job_kind"],
            observed=True,
        )
        .agg(
            jobs=("job_id", "size"),
            median_waiting_time=("waiting_time", "median"),
            p95_waiting_time=("waiting_time", lambda values: values.quantile(0.95)),
            mean_bounded_slowdown=("bounded_slowdown", "mean"),
        )
        .reset_index()
    )


def summarize_job_metrics(job_metrics):
    """Per (workload, variant, job_kind) overview of the job metrics."""
    return (
        job_metrics.groupby(["workload_label", "variant", "job_kind"], observed=True)
        .agg(
            runs=("name", "nunique"),
            jobs_per_run=("jobs", "median"),
            median_bounded_slowdown=("mean_bounded_slowdown", "median"),
        )
        .round(3)
    )


def build_run_metrics(schedules, job_metrics):
    """One row per experiment: schedule aggregates joined with replay job metrics."""
    replay_metrics = (
        job_metrics[job_metrics["job_kind"].eq("replay")]
        .drop(columns=["job_kind"])
        .rename(
            columns={
                "jobs": "replay_jobs",
                "median_waiting_time": "replay_median_waiting_time",
                "p95_waiting_time": "replay_p95_waiting_time",
                "mean_bounded_slowdown": "replay_mean_bounded_slowdown",
            }
        )
    )

    schedule_columns = [
        "name",
        "variant",
        "workload_label",
        "zone",
        "start_date",
        "season",
        "swing_carbon",
        "swing_water",
        "total_carbon_footprint",
        "total_carbon_operational",
        "total_water_footprint",
        "total_water_offsite",
        "consumed_joules",
        "makespan",
    ]

    schedule_metrics = schedules[schedule_columns].copy()
    schedule_metrics["consumed_energy_kwh"] = (
        schedule_metrics["consumed_joules"] / JOULES_PER_KWH
    )
    if schedule_metrics["consumed_energy_kwh"].le(0).any():
        raise ValueError("Consumed energy must be positive for every experiment")

    schedule_metrics["energy_weighted_carbon_intensity"] = (
        schedule_metrics["total_carbon_operational"]
        / schedule_metrics["consumed_energy_kwh"]
    )
    schedule_metrics["energy_weighted_water_intensity"] = (
        schedule_metrics["total_water_offsite"]
        / schedule_metrics["consumed_energy_kwh"]
    )

    return schedule_metrics.merge(
        replay_metrics,
        on=["name", "variant", "workload_label", "zone", "start_date"],
        how="left",
        validate="one_to_one",
    )


def paired_deltas(metrics, greenfilling_variant, baseline_variant="easy_bf"):
    """Per-window deltas of one greenfilling variant against EASY in the same window."""
    keys = ["workload_label", "zone", "start_date"]
    green = metrics[metrics["variant"].eq(greenfilling_variant)].copy()
    baseline = metrics[metrics["variant"].eq(baseline_variant)].copy()

    paired = green.merge(
        baseline,
        on=keys,
        suffixes=("_greenfilling", "_baseline"),
        validate="one_to_one",
    )
    paired["greenfilling_variant"] = greenfilling_variant
    paired["baseline_variant"] = baseline_variant
    for column in ["season", "swing_carbon", "swing_water"]:
        paired[column] = paired[f"{column}_greenfilling"]

    percent_metrics = [
        "total_carbon_footprint",
        "total_water_footprint",
        "consumed_energy_kwh",
        "energy_weighted_carbon_intensity",
        "energy_weighted_water_intensity",
        "makespan",
        "replay_median_waiting_time",
        "replay_p95_waiting_time",
        "replay_mean_bounded_slowdown",
    ]
    for metric in percent_metrics:
        paired[f"{metric}_delta_pct"] = (
            (paired[f"{metric}_greenfilling"] - paired[f"{metric}_baseline"])
            / paired[f"{metric}_baseline"]
            * 100
        )

    return paired


def build_paired_comparisons(run_metrics):
    """Paired deltas for both greenfilling objectives, stacked."""
    return pd.concat(
        [
            paired_deltas(run_metrics, greenfilling_variant)
            for greenfilling_variant in ["greenfilling_carbon", "greenfilling_water"]
        ],
        ignore_index=True,
    )


def summarize_paired(paired_comparisons):
    """Median/min/max of each paired delta, per (workload, greenfilling variant)."""
    return (
        paired_comparisons[["workload_label"] + DELTA_COLUMNS]
        .groupby(["workload_label", "greenfilling_variant"], observed=True)
        .agg(["median", "min", "max"])
        .round(2)
    )


# --- Plotting style ----------------------------------------------------------

VARIANT_ORDER = ["greenfilling_carbon", "greenfilling_water"]
VARIANT_LABELS = {
    "greenfilling_carbon": "Carbon objective",
    "greenfilling_water": "Water objective",
}
VARIANT_COLORS = {
    "greenfilling_carbon": "#3973ac",
    "greenfilling_water": "#2f8f6b",
}

# Workloads are (dataset, regime) pairs. Hue encodes the dataset, shade the
# regime; marker shape distinguishes all four in pooled scatters.
WORKLOAD_ORDER = ["mustang_stress", "mustang_slack", "trinity_stress", "trinity_slack"]
WORKLOAD_LABELS = {
    "mustang_stress": "Mustang stress",
    "mustang_slack": "Mustang slack",
    "trinity_stress": "Trinity stress",
    "trinity_slack": "Trinity slack",
}
WORKLOAD_COLORS = {
    "mustang_stress": "#2f5d8a",
    "mustang_slack": "#7ba7d0",
    "trinity_stress": "#c8571f",
    "trinity_slack": "#e6a05c",
}
WORKLOAD_MARKERS = {
    "mustang_stress": "o",
    "mustang_slack": "s",
    "trinity_stress": "^",
    "trinity_slack": "D",
}
WORKLOAD_LINESTYLES = {
    "mustang_stress": "--",
    "mustang_slack": ":",
    "trinity_stress": "-.",
    "trinity_slack": (0, (3, 1, 1, 1)),
}

SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_COLORS = {
    "winter": "#4c78a8",
    "spring": "#54a24b",
    "summer": "#f2a541",
    "autumn": "#b279a2",
}
OBJECTIVE_MARKERS = {
    "greenfilling_carbon": "o",
    "greenfilling_water": "^",
}


# --- Plotting ----------------------------------------------------------------

def plot_delta_boxes(data, metrics, titles, ylabels, figure_title):
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(4.6 * len(metrics), 4.4), squeeze=False
    )
    axes = axes[0]

    # One box per (objective, workload). Workloads sit side by side within each objective group.
    group_centers = {variant: 1.0 + index * 1.7 for index, variant in enumerate(VARIANT_ORDER)}
    within_offsets = {
        workload: (position - (len(WORKLOAD_ORDER) - 1) / 2) * 0.42
        for position, workload in enumerate(WORKLOAD_ORDER)
    }

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        for workload in WORKLOAD_ORDER:
            positions = [group_centers[variant] + within_offsets[workload] for variant in VARIANT_ORDER]
            series = [
                data.loc[
                    data["greenfilling_variant"].eq(variant) & data["workload_label"].eq(workload),
                    metric,
                ].dropna()
                for variant in VARIANT_ORDER
            ]
            box = ax.boxplot(
                series,
                positions=positions,
                widths=0.36,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.4},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(WORKLOAD_COLORS[workload])
                patch.set_alpha(0.35)

            for position, values in zip(positions, series):
                offsets = np.linspace(-0.08, 0.08, len(values)) if len(values) else []
                ax.scatter(
                    position + offsets,
                    values,
                    s=16,
                    alpha=0.65,
                    color=WORKLOAD_COLORS[workload],
                    edgecolor="none",
                )

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([group_centers[variant] for variant in VARIANT_ORDER])
        ax.set_xticklabels([VARIANT_LABELS[variant] for variant in VARIANT_ORDER])
        ax.set_xlim(
            group_centers[VARIANT_ORDER[0]] - 0.95,
            group_centers[VARIANT_ORDER[-1]] + 0.95,
        )

    workload_handles = [
        Patch(facecolor=WORKLOAD_COLORS[workload], alpha=0.35, edgecolor="black", label=WORKLOAD_LABELS[workload])
        for workload in WORKLOAD_ORDER
    ]
    fig.legend(handles=workload_handles, loc="upper center", ncol=len(WORKLOAD_ORDER), bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(figure_title, y=1.09)
    fig.tight_layout()
    return fig, axes


def plot_environmental_deltas(data):
    """Footprint deltas (carbon, water) versus EASY, per sampled window."""
    return plot_delta_boxes(
        data,
        ["total_carbon_footprint_delta_pct", "total_water_footprint_delta_pct"],
        ["Carbon footprint", "Water footprint"],
        [r"$\Delta$ [% vs EASY]"] * 2,
        r"Environmental $\Delta$ by sampled window",
    )


def plot_energy_exposure_deltas(data):
    """Energy and effective operational intensity deltas versus EASY."""
    return plot_delta_boxes(
        data,
        [
            "consumed_energy_kwh_delta_pct",
            "energy_weighted_carbon_intensity_delta_pct",
            "energy_weighted_water_intensity_delta_pct",
        ],
        ["Consumed energy", "Effective carbon intensity", "Effective water intensity"],
        [r"$\Delta$ [% vs EASY]"] * 3,
        "Energy and intensity exposure deltas by sampled window",
    )


def plot_performance_deltas(data):
    """Replay-job performance deltas versus EASY, per sampled window."""
    return plot_delta_boxes(
        data,
        [
            "replay_median_waiting_time_delta_pct",
            "replay_p95_waiting_time_delta_pct",
            "replay_mean_bounded_slowdown_delta_pct",
        ],
        ["Median waiting time", "P95 waiting time", "Average bounded slowdown"],
        [r"$\Delta$ [% vs EASY]"] * 3,
        "Job performance deltas by sampled window",
    )


def plot_tradeoff_scatter(data, environmental_delta_column, x_label, title):
    plot_data = data.copy()
    plot_data["environmental_saving_pct"] = -plot_data[environmental_delta_column]
    plot_data["bounded_slowdown_penalty_pct"] = plot_data[
        "replay_mean_bounded_slowdown_delta_pct"
    ]

    zones = sorted(plot_data["zone"].unique())
    fig, axes = plt.subplots(
        len(WORKLOAD_ORDER),
        len(zones),
        figsize=(4.4 * len(zones), 4.0 * len(WORKLOAD_ORDER)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_values = plot_data["environmental_saving_pct"]
    y_values = plot_data["bounded_slowdown_penalty_pct"]
    x_padding = max((x_values.max() - x_values.min()) * 0.12, 0.25)
    y_padding = max((y_values.max() - y_values.min()) * 0.12, 1.0)

    for row, workload in enumerate(WORKLOAD_ORDER):
        workload_data = plot_data[plot_data["workload_label"].eq(workload)]
        for ax, zone in zip(axes[row], zones):
            zone_data = workload_data[workload_data["zone"].eq(zone)]
            for variant in VARIANT_ORDER:
                variant_data = zone_data[zone_data["greenfilling_variant"].eq(variant)]
                for season in SEASON_ORDER:
                    points = variant_data[variant_data["season"].eq(season)]
                    ax.scatter(
                        points["environmental_saving_pct"],
                        points["bounded_slowdown_penalty_pct"],
                        marker=OBJECTIVE_MARKERS[variant],
                        s=58,
                        color=SEASON_COLORS[season],
                        edgecolor="black",
                        linewidth=0.55,
                        alpha=0.85,
                    )

            ax.axvline(0, color="black", linewidth=0.9)
            ax.axhline(0, color="black", linewidth=0.9)
            ax.set_title(f"{WORKLOAD_LABELS[workload]} - {zone}")
            ax.set_xlabel(x_label)
            ax.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
            ax.set_ylim(min(0, y_values.min() - y_padding), y_values.max() + y_padding)
        axes[row][0].set_ylabel("Average bounded slowdown penalty [% vs EASY]")

    season_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=SEASON_COLORS[season], label=season.title(), markersize=7)
        for season in SEASON_ORDER
    ]
    objective_handles = [
        Line2D(
            [0],
            [0],
            marker=OBJECTIVE_MARKERS[variant],
            linestyle="",
            color="black",
            label=VARIANT_LABELS[variant],
            markersize=7,
        )
        for variant in VARIANT_ORDER
    ]

    fig.legend(handles=season_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.legend(handles=objective_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle(title, y=1.12)
    fig.tight_layout()
    return fig, axes


def plot_swing_relationship(data):
    specs = [
        ("greenfilling_carbon", "swing_carbon", "total_carbon_footprint_delta_pct", "Carbon objective", "Carbon swing"),
        ("greenfilling_water", "swing_water", "total_water_footprint_delta_pct", "Water objective", "Water swing"),
    ]

    fig, axes = plt.subplots(1, len(specs), figsize=(5.6 * len(specs), 4.6), squeeze=False)
    axes = axes[0]

    for ax, (variant, swing_column, delta_column, title, x_label) in zip(axes, specs):
        variant_data = data[data["greenfilling_variant"].eq(variant)]

        for workload in WORKLOAD_ORDER:
            subset = variant_data[variant_data["workload_label"].eq(workload)]
            for season in SEASON_ORDER:
                points = subset[subset["season"].eq(season)]
                ax.scatter(
                    points[swing_column],
                    -points[delta_column],
                    color=SEASON_COLORS[season],
                    edgecolor="black",
                    linewidth=0.55,
                    s=58,
                    alpha=0.85,
                    marker=WORKLOAD_MARKERS[workload],
                )

        # Fit and correlation over all points, both workloads pooled.
        swing = variant_data[swing_column].to_numpy()
        saving = -variant_data[delta_column].to_numpy()  # positive = footprint reduction vs EASY
        if len(swing) > 1:
            slope, intercept = np.polyfit(swing, saving, 1)
            grid = np.linspace(swing.min(), swing.max(), 50)
            ax.plot(grid, slope * grid + intercept, color="black", linestyle="--", linewidth=1.1)
            r = np.corrcoef(swing, saving)[0, 1]
            ax.set_title(f"{title}  (r = {r:.2f})")
        else:
            ax.set_title(title)

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Footprint saving [% vs EASY]")

    season_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=SEASON_COLORS[season], label=season.title(), markersize=7)
        for season in SEASON_ORDER
    ]
    workload_handles = [
        Line2D([0], [0], marker=WORKLOAD_MARKERS[workload], linestyle="", color="black", label=WORKLOAD_LABELS[workload], markersize=7)
        for workload in WORKLOAD_ORDER
    ]
    fig.legend(handles=season_handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.legend(handles=workload_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("Intra-day swing versus greenfilling footprint saving", y=1.14)
    fig.tight_layout()
    return fig, axes
