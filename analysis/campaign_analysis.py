"""Shared loading, metrics, and plotting for the green-window campaign notebooks.

`green_window_scheduling.ipynb` imports this module for loading the campaign,
computing per-window paired deltas versus the EASY baseline, and plotting.

The green scheduler is swept over two axes, an intensity `signal` (carbon or
water) and a `planning horizon` (how far ahead it may look for a greener window
to displace a job into). A `variant` folds both together, so each (signal,
horizon) pair pairs against the same EASY baseline through one code path.

The older two-variant `greenfilling_{carbon,water}` naming is still parsed, so a
campaign TOML describing that campaign keeps loading.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import tomllib

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle


# Average bounded slowdown uses this floor on execution time so very short jobs
# do not dominate the comparison: max(turnaround / max(execution, floor), 1).
BOUNDED_SLOWDOWN_FLOOR_SECONDS = 10.0
JOULES_PER_KWH = 3_600_000.0
SECONDS_PER_DAY = 86_400.0
IEEE_DOUBLE_COLUMN_WIDTH_INCHES = 7.16
IEEE_MAX_FIGURE_HEIGHT_INCHES = 8.8
IEEE_LINE_ART_DPI = 600
BOX_FIGURE_SIZE = (IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 5.0)
TRADEOFF_FIGURE_SIZE = (IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 8.6)
SWING_FIGURE_SIZE = (IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 3.35)
TREND_FIGURE_SIZE = (IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 4.6)
COVERAGE_FIGURE_SIZE = (IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.9)
DATA_MARKER_SIZE = 4.0
LEGEND_MARKER_SIZE = 4.5
BOX_POINT_AREA = 7.0

METADATA_COLUMNS = [
    "name",
    "variant",
    "objective",
    "horizon_seconds",
    "workload_label",
    "dataset",
    "regime",
    "zone",
    "start_date",
]

# Paired-delta columns reported in the summary table and per-window head.
DELTA_COLUMNS = [
    "green_variant",
    "total_carbon_footprint_delta_pct",
    "total_water_footprint_delta_pct",
    "consumed_energy_kwh_delta_pct",
    "energy_weighted_carbon_intensity_delta_pct",
    "energy_weighted_water_intensity_delta_pct",
    "makespan_delta_pct",
    "replay_median_waiting_time_delta_seconds",
    "replay_p95_waiting_time_delta_pct",
    "replay_mean_bounded_slowdown_delta_pct",
]

# The delta whose sign answers "did the scheduler hit its own target", per signal.
TARGET_DELTA_COLUMN = {
    "carbon": "total_carbon_footprint_delta_pct",
    "water": "total_water_footprint_delta_pct",
}
TARGET_INTENSITY_COLUMN = {
    "carbon": "energy_weighted_carbon_intensity_delta_pct",
    "water": "energy_weighted_water_intensity_delta_pct",
}
SWING_COLUMN = {"carbon": "swing_carbon", "water": "swing_water"}


# --- Parsing -----------------------------------------------------------------

GREEN_WINDOW_NAME_RE = re.compile(
    r"green_window_scheduling_(?P<objective>carbon|water)_(?P<horizon>\d+)"
    r"_(?P<dataset>[^_]+)_(?P<regime>[^_]+)"
    r"_(?P<zone>[A-Z]{2})_(?P<start_date>\d{4}-\d{2}-\d{2})"
)
GREENFILLING_NAME_RE = re.compile(
    r"greenfilling_(?P<objective>carbon|water)"
    r"_(?P<dataset>[^_]+)_(?P<regime>[^_]+)"
    r"_(?P<zone>[A-Z]{2})_(?P<start_date>\d{4}-\d{2}-\d{2})"
)
BASELINE_NAME_RE = re.compile(
    r"easy_bf_(?P<dataset>[^_]+)_(?P<regime>[^_]+)"
    r"_(?P<zone>[A-Z]{2})_(?P<start_date>\d{4}-\d{2}-\d{2})"
)


def parse_experiment_name(name):
    """Split an experiment name into its variant/objective/horizon/workload/window.

    The workload is a (dataset, regime) pair, e.g. `mustang_slack`. It is kept
    combined in `workload_label` so each dataset pairs against its own EASY
    baseline, and also split into `dataset` and `regime` for slicing.

    `horizon_seconds` is the green scheduler's planning horizon, and is <NA> for
    variants that have no horizon axis (the baseline and the old greenfilling
    runs).
    """
    green_window = GREEN_WINDOW_NAME_RE.fullmatch(name)
    if green_window:
        parts = green_window.groupdict()
        horizon = int(parts["horizon"])
        return {
            "variant": f"green_window_scheduling_{parts['objective']}_{horizon}",
            "objective": parts["objective"],
            "horizon_seconds": horizon,
            "workload_label": f"{parts['dataset']}_{parts['regime']}",
            "dataset": parts["dataset"],
            "regime": parts["regime"],
            "zone": parts["zone"],
            "start_date": parts["start_date"],
        }

    greenfilling = GREENFILLING_NAME_RE.fullmatch(name)
    if greenfilling:
        parts = greenfilling.groupdict()
        return {
            "variant": f"greenfilling_{parts['objective']}",
            "objective": parts["objective"],
            "horizon_seconds": pd.NA,
            "workload_label": f"{parts['dataset']}_{parts['regime']}",
            "dataset": parts["dataset"],
            "regime": parts["regime"],
            "zone": parts["zone"],
            "start_date": parts["start_date"],
        }

    baseline = BASELINE_NAME_RE.fullmatch(name)
    if baseline:
        parts = baseline.groupdict()
        return {
            "variant": "easy_bf",
            "objective": "baseline",
            "horizon_seconds": pd.NA,
            "workload_label": f"{parts['dataset']}_{parts['regime']}",
            "dataset": parts["dataset"],
            "regime": parts["regime"],
            "zone": parts["zone"],
            "start_date": parts["start_date"],
        }

    raise ValueError(f"Unexpected experiment name: {name}")


def green_variants(metrics, baseline_variant="easy_bf"):
    """Every non-baseline variant present, ordered by signal then horizon."""
    variants = [
        variant
        for variant in metrics["variant"].unique()
        if variant != baseline_variant
    ]

    def sort_key(variant):
        match = re.search(r"_(carbon|water)(?:_(\d+))?$", variant)
        if not match:
            return (2, variant, 0)
        signal, horizon = match.groups()
        return (0 if signal == "carbon" else 1, "", int(horizon or 0))

    return sorted(variants, key=sort_key)


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
    return add_accounting_horizon(schedules, jobs), jobs


# --- Accounting horizon ------------------------------------------------------

def add_accounting_horizon(schedules, jobs):
    """Per run: the footprint's accounting horizon, and energy over that horizon.

    Batsim's footprint totals run to the last trace timestamp, because the trace
    updates keep the simulation alive, while `consumed_joules` stops at the last
    job completion. With always-on nodes, PUE 1 and no embodied impact, a run
    accounted to the end of the current trace satisfies

        total = compute * (1 - P_idle / P_compute) + P_idle * nodes * integral of I

    over the whole trace, for carbon and water. Adds `trace_end_seconds`,
    `accounted_to_trace_end` (whether that identity holds, checked by
    `assert_common_horizon`) and `consumed_joules_to_trace_end`.
    """
    import shift_ceiling_results as scr

    compute = jobs.groupby("name")[["consumed_carbon", "consumed_water"]].sum()
    platforms, windows = {}, {}
    trace_ends, accounted, energies = [], [], []
    for run in schedules.itertuples(index=False):
        if run.dataset not in platforms:
            platforms[run.dataset] = scr.load_platform(run.dataset)
        platform = platforms[run.dataset]
        key = (run.zone, run.start_date)
        if key not in windows:
            trace = scr.load_trace(run)
            end = (len(trace.carbon) - 1) * scr.DELTA_SECONDS
            windows[key] = end, {
                signal: scr.integral_at(
                    np.array([end], dtype=float), values, scr.prefix_integral(values)
                )[0]
                for signal, values in [("carbon", trace.carbon), ("water", trace.water)]
            }
        end, whole_trace = windows[key]

        idle_ratio = platform.idle_power_w / platform.compute_power_w
        # W x (intensity x h) / 1000 gives g for carbon and L for water.
        accounted.append(all(
            np.isclose(
                total,
                compute.loc[run.name, f"consumed_{signal}"] * (1 - idle_ratio)
                + platform.idle_power_w * run.nb_computing_machines * whole_trace[signal] / 1000,
                rtol=1e-6,
            )
            for signal, total in [
                ("carbon", run.total_carbon_operational),
                ("water", run.total_water_offsite),
            ]
        ))
        trace_ends.append(end)
        energies.append(
            run.consumed_joules
            + platform.idle_power_w * run.nb_computing_machines * (end - run.makespan)
        )

    return schedules.assign(
        trace_end_seconds=trace_ends,
        accounted_to_trace_end=accounted,
        consumed_joules_to_trace_end=energies,
    )


def assert_common_horizon(schedules):
    """Raise unless every run was accounted to the end of its current trace.

    A run that fails was accounted against another trace, for instance an output
    kept from before the traces were extended, so its footprint cannot be paired
    with the other runs of its window. Rerun it, do not rescale the results.
    """
    stale = sorted(schedules.loc[~schedules["accounted_to_trace_end"], "name"])
    if stale:
        raise ValueError(
            f"{len(stale)} of {len(schedules)} runs were not accounted to the end of "
            f"their current trace, so they cannot be paired with the others. Rerun "
            f"them: {', '.join(stale[:10])}{' ...' if len(stale) > 10 else ''}"
        )


# --- Trace coverage ----------------------------------------------------------

def trace_extent(windows, intensities_dir):
    """Per window: when the intensity trace ends, and its last vs mean intensity.

    Batsim holds the last trace sample for the rest of the simulation, so any
    footprint accrued past `trace_end_seconds` is priced at a frozen intensity.
    A schedule that runs well past the window is therefore not being scored
    against a real signal, which is what `trace_coverage` measures.
    """
    rows = []
    for window in windows.itertuples(index=False):
        trace = pd.read_csv(Path(intensities_dir) / window.file)
        carbon = trace[trace["property"].eq("carbon_intensity")].sort_values("timestamp")
        water = trace[trace["property"].eq("water_intensity")].sort_values("timestamp")
        rows.append(
            {
                "zone": window.zone,
                "start_date": window.start_date,
                "trace_end_seconds": carbon["timestamp"].max(),
                "last_carbon_intensity": carbon["value"].iloc[-1],
                "last_water_intensity": water["value"].iloc[-1],
                "mean_carbon_intensity": carbon["value"].mean(),
                "mean_water_intensity": water["value"].mean(),
            }
        )
    return pd.DataFrame(rows)


def trace_coverage(run_metrics, out_dir, extent):
    """Per run: how much footprint is accrued after the intensity trace ends.

    A run whose makespan exceeds the trace is not scored against a real signal,
    and its footprint numbers cannot be recovered by re-weighting them after the
    fact: the schedule itself was chosen against a signal that stopped varying.
    Such runs are rejected by `invalid_footprint_runs`, not corrected.
    """
    extent = extent.set_index(["zone", "start_date"])
    rows = []
    for run in run_metrics.itertuples(index=False):
        window = extent.loc[(run.zone, run.start_date)]
        footprint = pd.read_csv(
            Path(out_dir) / run.name / "out_environmental_footprint.csv"
        )
        footprint = footprint.drop_duplicates(subset="time", keep="last").sort_values(
            "time"
        )
        inside = footprint[footprint["time"] <= window["trace_end_seconds"]]
        carbon_inside = (
            inside["carbon_operational(gCO2e)"].iloc[-1] if len(inside) else 0.0
        )
        water_inside = inside["water_offsite(L)"].iloc[-1] if len(inside) else 0.0
        carbon_total = footprint["carbon_operational(gCO2e)"].iloc[-1]
        water_total = footprint["water_offsite(L)"].iloc[-1]

        rows.append(
            {
                "name": run.name,
                "variant": run.variant,
                "workload_label": run.workload_label,
                "zone": run.zone,
                "start_date": run.start_date,
                "trace_end_seconds": window["trace_end_seconds"],
                "makespan": run.makespan,
                "makespan_over_trace": run.makespan / window["trace_end_seconds"],
                "tail_carbon_share_pct": (carbon_total - carbon_inside)
                / carbon_total
                * 100,
                "tail_water_share_pct": (water_total - water_inside) / water_total * 100,
                "footprint_is_valid": run.makespan <= window["trace_end_seconds"],
            }
        )
    return pd.DataFrame(rows)


def summarize_trace_coverage(coverage):
    """Per variant: how far past the trace each schedule runs, and the tail share."""
    return (
        coverage.groupby("variant")
        .agg(
            runs=("name", "size"),
            median_makespan_over_trace=("makespan_over_trace", "median"),
            max_makespan_over_trace=("makespan_over_trace", "max"),
            median_tail_carbon_share_pct=("tail_carbon_share_pct", "median"),
            max_tail_carbon_share_pct=("tail_carbon_share_pct", "max"),
        )
        .round(2)
    )


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


def build_energy_decomposition(jobs):
    """Per run: the energy and footprint that running jobs account for.

    The schedule totals cover the whole platform, so subtracting these leaves
    the idle part. Splitting the two matters because displacement moves compute
    out of dirty hours and, by the same action, leaves idle sitting in them.
    """
    return (
        jobs.groupby("name", observed=True)
        .agg(
            compute_joules=("consumed_energy", "sum"),
            compute_carbon=("consumed_carbon", "sum"),
            compute_water=("consumed_water", "sum"),
        )
        .reset_index()
    )


def build_run_metrics(schedules, job_metrics, jobs=None):
    """One row per experiment: schedule aggregates joined with replay job metrics.

    Passing `jobs` adds the compute/idle split of energy and footprint.
    """
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
        "objective",
        "horizon_seconds",
        "workload_label",
        "dataset",
        "regime",
        "zone",
        "start_date",
        "season",
        "swing_carbon",
        "swing_water",
        "total_carbon_footprint",
        "total_carbon_operational",
        "total_water_footprint",
        "total_water_offsite",
        "consumed_joules_to_trace_end",
        "trace_end_seconds",
        "makespan",
        "time_computing",
        "time_idle",
        "nb_computing_machines",
        "nb_jobs",
    ]

    schedule_metrics = schedules[schedule_columns].copy()
    # Same horizon as the footprint totals, so intensities and pairs are consistent.
    schedule_metrics["consumed_energy_kwh"] = (
        schedule_metrics["consumed_joules_to_trace_end"] / JOULES_PER_KWH
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
    # Nodes are never powered down in this platform, so idle time is pure loss.
    schedule_metrics["idle_time_share"] = schedule_metrics["time_idle"] / (
        schedule_metrics["time_idle"] + schedule_metrics["time_computing"]
    )
    # Share of the platform actually doing work over the run. A scheduler that
    # holds the cluster to wait for a greener window shows up here immediately.
    schedule_metrics["node_utilisation"] = schedule_metrics["time_computing"] / (
        schedule_metrics["makespan"] * schedule_metrics["nb_computing_machines"]
    )

    if jobs is not None:
        schedule_metrics = schedule_metrics.merge(
            build_energy_decomposition(jobs), on="name", validate="one_to_one"
        )
        compute_kwh = schedule_metrics["compute_joules"] / JOULES_PER_KWH
        idle_kwh = schedule_metrics["consumed_energy_kwh"] - compute_kwh
        if idle_kwh.le(0).any():
            raise ValueError("Idle energy must be positive for every experiment")

        schedule_metrics["compute_energy_kwh"] = compute_kwh
        schedule_metrics["idle_energy_kwh"] = idle_kwh
        schedule_metrics["compute_energy_share"] = (
            compute_kwh / schedule_metrics["consumed_energy_kwh"]
        )
        for signal, total, compute in [
            ("carbon", "total_carbon_operational", "compute_carbon"),
            ("water", "total_water_offsite", "compute_water"),
        ]:
            schedule_metrics[f"compute_{signal}_intensity"] = (
                schedule_metrics[compute] / compute_kwh
            )
            schedule_metrics[f"idle_{signal}_intensity"] = (
                schedule_metrics[total] - schedule_metrics[compute]
            ) / idle_kwh

    return schedule_metrics.merge(
        replay_metrics,
        on=["name", "variant", "workload_label", "zone", "start_date"],
        how="left",
        validate="one_to_one",
    )


def summarize_phase_exposure(run_metrics):
    """Compute/idle exposure by workload and variant, including EASY once.

    Amounts are per-run means; shares are ratios of sums across windows.
    Machine times are node-seconds, not elapsed schedule durations.
    Assumes always-on nodes: exported idle counters omit final idle intervals,
    so idle time is capacity-time minus computing time through the end of the
    trace, the horizon of the footprint totals.
    """
    run_metrics = run_metrics.copy()
    run_metrics["time_idle"] = (
        run_metrics["trace_end_seconds"] * run_metrics["nb_computing_machines"]
        - run_metrics["time_computing"]
    )
    keys = ["workload_label", "variant"]
    columns = [
        "time_computing", "time_idle", "compute_energy_kwh", "idle_energy_kwh",
        "compute_carbon", "total_carbon_operational",
        "compute_water", "total_water_offsite",
    ]
    grouped = run_metrics.groupby(keys, observed=True)
    totals = grouped[columns].sum()
    runs = grouped.size()
    phases = []
    for phase, time_column in [("Compute", "time_computing"), ("Idle", "time_idle")]:
        compute = phase == "Compute"
        energy = totals["compute_energy_kwh" if compute else "idle_energy_kwh"]
        frame = pd.DataFrame(index=totals.index)
        frame["phase"] = phase
        frame["runs"] = runs
        frame["node_hours_per_run"] = totals[time_column] / runs / 3600
        frame["node_time_share_pct"] = 100 * totals[time_column] / (
            totals["time_computing"] + totals["time_idle"]
        )
        frame["energy_kwh_per_run"] = energy / runs
        frame["energy_share_pct"] = 100 * energy / (
            totals["compute_energy_kwh"] + totals["idle_energy_kwh"]
        )
        for signal, total_column in [
            ("carbon", "total_carbon_operational"), ("water", "total_water_offsite")
        ]:
            footprint = totals[f"compute_{signal}"]
            if not compute:
                footprint = totals[total_column] - footprint
            frame[f"{signal}_share_pct"] = 100 * footprint / totals[total_column]
        phases.append(frame.reset_index())
    return pd.concat(phases, ignore_index=True).set_index(keys + ["phase"]).sort_index()


def build_deferral_diagnostics(jobs, run_metrics):
    """Per run: how job waiting times compare with the scheduler's own horizon.

    The planning horizon is meant to bound how far a job may be displaced. If
    most jobs wait far longer than it, the bound is not holding and the campaign
    is measuring a runaway scheduler rather than a green-scheduling policy.
    """
    horizons = run_metrics.set_index("name")["horizon_seconds"]
    replay = jobs[jobs["job_kind"].eq("replay")]

    rows = []
    for name, group in replay.groupby("name", observed=True):
        horizon = horizons.get(name, pd.NA)
        waiting = group["waiting_time"]
        row = {
            "name": name,
            "horizon_seconds": horizon,
            "median_waiting_time": waiting.median(),
            "max_waiting_time": waiting.max(),
        }
        if pd.notna(horizon):
            row["over_horizon_pct"] = waiting.gt(horizon).mean() * 100
            row["over_10x_horizon_pct"] = waiting.gt(10 * float(horizon)).mean() * 100
            row["median_waiting_over_horizon"] = waiting.median() / float(horizon)
        rows.append(row)

    diagnostics = pd.DataFrame(rows)
    return diagnostics.merge(
        run_metrics[["name", "variant", "workload_label", "zone", "start_date"]],
        on="name",
    )


def summarize_deferral(diagnostics):
    """Per variant: is the planning horizon actually bounding displacement?"""
    green = diagnostics[diagnostics["horizon_seconds"].notna()]
    return (
        green.groupby("variant")
        .agg(
            runs=("name", "size"),
            median_over_horizon_pct=("over_horizon_pct", "median"),
            median_over_10x_horizon_pct=("over_10x_horizon_pct", "median"),
            median_waiting_over_horizon=("median_waiting_over_horizon", "median"),
        )
        .round(2)
    )

PERCENT_DELTA_METRICS = [
    "total_carbon_footprint",
    "total_water_footprint",
    "consumed_energy_kwh",
    "energy_weighted_carbon_intensity",
    "energy_weighted_water_intensity",
    "compute_carbon_intensity",
    "idle_carbon_intensity",
    "compute_water_intensity",
    "idle_water_intensity",
    "makespan",
    "replay_p95_waiting_time",
    "replay_mean_bounded_slowdown",
    "node_utilisation",
]

# One convention for the whole module: every `*_improvement_pct` column is a
# percentage against EASY in the same window where POSITIVE MEANS GREEN IS
# BETTER. For everything except utilisation that is the negated delta, because
# lower footprint, energy, makespan, waiting and slowdown are all better.
# `signal` picks carbon or water, so the same column answers both campaigns.
LOWER_IS_BETTER = {
    "energy": "consumed_energy_kwh",
    "makespan": "makespan",
    "waiting": "replay_p95_waiting_time",
    "slowdown": "replay_mean_bounded_slowdown",
}
SIGNAL_METRICS = {
    "footprint": "total_{signal}_footprint",
    "platform_intensity": "energy_weighted_{signal}_intensity",
    "compute_intensity": "compute_{signal}_intensity",
    "idle_intensity": "idle_{signal}_intensity",
}

# The headline set, in reading order: did it help, where did the carbon go,
# what did it cost.
IMPROVEMENT_COLUMNS = [
    "footprint_improvement_pct",
    "compute_intensity_improvement_pct",
    "idle_intensity_improvement_pct",
    "platform_intensity_improvement_pct",
    "energy_improvement_pct",
    "makespan_improvement_pct",
    "waiting_improvement_pct",
    "slowdown_improvement_pct",
    "utilisation_improvement_pct",
]


def paired_deltas(metrics, green_variant, baseline_variant="easy_bf"):
    """Per-window comparison of one green variant against EASY in the same window.

    Returns the raw `*_delta_pct` columns plus the standardized
    `*_improvement_pct` columns, where positive always means green won.
    """
    keys = ["workload_label", "zone", "start_date"]
    green = metrics[metrics["variant"].eq(green_variant)].copy()
    baseline = metrics[metrics["variant"].eq(baseline_variant)].copy()

    paired = green.merge(
        baseline,
        on=keys,
        suffixes=("_green", "_baseline"),
        validate="one_to_one",
    )
    paired["green_variant"] = green_variant
    paired["baseline_variant"] = baseline_variant
    # Window and green-side attributes that describe the pair as a whole.
    for column in [
        "season",
        "swing_carbon",
        "swing_water",
        "objective",
        "horizon_seconds",
        "dataset",
        "regime",
    ]:
        paired[column] = paired[f"{column}_green"]
    paired["signal"] = paired["objective"]
    paired["horizon_hours"] = (
        pd.to_numeric(paired["horizon_seconds"], errors="coerce") / 3600
    )

    for metric in PERCENT_DELTA_METRICS:
        if f"{metric}_green" not in paired.columns:
            continue
        paired[f"{metric}_delta_pct"] = (
            (paired[f"{metric}_green"] - paired[f"{metric}_baseline"])
            / paired[f"{metric}_baseline"]
            * 100
        )

    paired["replay_median_waiting_time_delta_seconds"] = (
        paired["replay_median_waiting_time_green"]
        - paired["replay_median_waiting_time_baseline"]
    )

    for name, metric in LOWER_IS_BETTER.items():
        paired[f"{name}_improvement_pct"] = -paired[f"{metric}_delta_pct"]
    # Utilisation is the one metric where more is better.
    paired["utilisation_improvement_pct"] = paired["node_utilisation_delta_pct"]

    is_carbon = paired["signal"].eq("carbon")
    for name, template in SIGNAL_METRICS.items():
        carbon = template.format(signal="carbon") + "_delta_pct"
        water = template.format(signal="water") + "_delta_pct"
        if carbon not in paired.columns:
            continue
        paired[f"{name}_improvement_pct"] = -np.where(
            is_carbon, paired[carbon], paired[water]
        )

    paired["target_swing"] = np.where(
        is_carbon, paired["swing_carbon"], paired["swing_water"]
    )

    return paired


def build_paired_comparisons(run_metrics, baseline_variant="easy_bf"):
    """Paired deltas for every green variant present, stacked.

    Variants are discovered from the data, so adding a signal or a planning
    horizon to the campaign needs no change here.
    """
    return pd.concat(
        [
            paired_deltas(run_metrics, variant, baseline_variant)
            for variant in green_variants(run_metrics, baseline_variant)
        ],
        ignore_index=True,
    )


def invalid_footprint_runs(coverage):
    """Runs whose schedule outlived the intensity trace, so their footprint is void.

    Batsim holds the last trace sample once the trace ends, so these runs were
    both scored and scheduled against a signal that had stopped varying. The
    energy, makespan, waiting-time and slowdown metrics are unaffected.
    """
    return coverage[~coverage["footprint_is_valid"]]


def assert_footprint_validity(coverage):
    """Raise unless every run finished inside its intensity trace."""
    invalid = invalid_footprint_runs(coverage)
    if len(invalid):
        worst = invalid["makespan_over_trace"].max()
        raise ValueError(
            f"{len(invalid)} of {len(coverage)} runs outlive their intensity trace "
            f"(worst: {worst:.1f}x the trace length). Their footprint metrics are void. "
            "Fix the schedules or extend the traces, do not reweight the results."
        )


def summarize_paired(paired_comparisons):
    """Median/min/max of each paired delta, per (workload, green variant)."""
    return (
        paired_comparisons[["workload_label"] + DELTA_COLUMNS]
        .groupby(["workload_label", "green_variant"], observed=True)
        .agg(["median", "min", "max"])
        .round(2)
    )


def summarize_by_horizon(paired_comparisons, group_columns=()):
    """Median improvement over EASY per signal and horizon. Positive means better.

    `group_columns` adds further slicing, e.g. `("regime",)` or `("dataset",)`.
    """
    keys = ["signal", "horizon_hours", *group_columns]
    available = [c for c in IMPROVEMENT_COLUMNS if c in paired_comparisons.columns]
    summary = (
        paired_comparisons.groupby(keys, observed=True)[available].median().round(2)
    )
    summary.insert(0, "windows", paired_comparisons.groupby(keys, observed=True).size())
    summary.insert(
        1,
        "windows_better_pct",
        paired_comparisons.groupby(keys, observed=True)["footprint_improvement_pct"]
        .apply(lambda values: (values > 0).mean() * 100)
        .round(1),
    )
    return summary


def summarize_swing_correlations(paired_comparisons):
    """Pearson correlations between signal swing and target-footprint saving."""
    rows = []
    for (signal, horizon), variant_data in paired_comparisons.groupby(
        ["signal", "horizon_hours"], observed=True
    ):
        groups = [("pooled", variant_data)] + [
            (workload, variant_data[variant_data["workload_label"].eq(workload)])
            for workload in WORKLOAD_ORDER
        ]
        for workload, group in groups:
            swing = group["target_swing"]
            saving = group["footprint_improvement_pct"]
            correlation = (
                swing.corr(saving)
                if len(group) > 1 and swing.nunique() > 1 and saving.nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "signal": signal,
                    "horizon_hours": horizon,
                    "workload_label": workload,
                    "observations": len(group),
                    "pearson_r": correlation,
                }
            )
    return pd.DataFrame(rows).round({"pearson_r": 3})


# --- Plotting style ----------------------------------------------------------

SIGNAL_ORDER = ["carbon", "water"]
SIGNAL_LABELS = {"carbon": "Carbon objective", "water": "Water objective"}
SIGNAL_MARKERS = {"carbon": "o", "water": "^"}

# Planning horizons, in hours. Derived from the data by `horizon_order`, with
# this as the display order for the ones the campaign actually swept.
HORIZON_LABELS = {6.0: "6 h", 12.0: "12 h", 24.0: "24 h"}

# Workloads are (dataset, regime) pairs. Hatches and marker shapes keep them
# distinguishable using only black and white.
WORKLOAD_ORDER = ["mustang_stress", "mustang_slack"]
WORKLOAD_LABELS = {
    "mustang_stress": "Mustang stress",
    "mustang_slack": "Mustang slack",
}
WORKLOAD_HATCHES = {
    "mustang_stress": "///",
    "mustang_slack": "\\\\\\",
}
WORKLOAD_MARKERS = {
    "mustang_stress": "o",
    "mustang_slack": "s",
}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]
SEASON_FILLSTYLES = {
    "winter": "full",
    "spring": "none",
    "summer": "left",
    "autumn": "right",
}


def horizon_order(data):
    """Planning horizons present, in hours, ascending."""
    return sorted(data["horizon_hours"].dropna().unique())


def horizon_label(hours):
    return HORIZON_LABELS.get(hours, f"{hours:g} h")


# --- Plotting ----------------------------------------------------------------

def monochrome_marker(marker, fillstyle):
    """Return a black-and-white marker with the requested fill style."""
    return MarkerStyle(marker, fillstyle=fillstyle)


def export_figure(fig, output_dir, stem):
    """Export an IEEE-sized figure as vector PDF and 600 dpi PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", dpi=IEEE_LINE_ART_DPI)
    fig.savefig(png_path, format="png", dpi=IEEE_LINE_ART_DPI)
    return pdf_path, png_path


def _workload_legend(fig, y=0.5):
    handles = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=WORKLOAD_HATCHES[workload],
            label=WORKLOAD_LABELS[workload],
        )
        for workload in WORKLOAD_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(WORKLOAD_ORDER),
        bbox_to_anchor=(0.5, y),
        frameon=False,
    )


def plot_delta_boxes(data, metrics, titles, ylabels, figure_title, log_scale=()):
    """Grid of paired-delta boxes: one row per signal, one column per metric.

    Within each panel the x axis is the planning horizon and the four workloads
    sit side by side, so the horizon trend reads left to right.
    """
    horizons = horizon_order(data)
    fig, axes = plt.subplots(
        len(SIGNAL_ORDER),
        len(metrics),
        figsize=BOX_FIGURE_SIZE,
        squeeze=False,
        sharex=True,
    )

    group_centers = {horizon: 1.0 + index * 2.3 for index, horizon in enumerate(horizons)}
    within_offsets = {
        workload: (position - (len(WORKLOAD_ORDER) - 1) / 2) * 0.42
        for position, workload in enumerate(WORKLOAD_ORDER)
    }

    for row, signal in enumerate(SIGNAL_ORDER):
        signal_data = data[data["signal"].eq(signal)]
        for column, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = axes[row][column]
            for workload in WORKLOAD_ORDER:
                positions = [group_centers[horizon] + within_offsets[workload] for horizon in horizons]
                series = [
                    signal_data.loc[
                        signal_data["horizon_hours"].eq(horizon)
                        & signal_data["workload_label"].eq(workload),
                        metric,
                    ].dropna()
                    for horizon in horizons
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
                    patch.set_facecolor("white")
                    patch.set_edgecolor("black")
                    patch.set_hatch(WORKLOAD_HATCHES[workload])

            ax.axhline(0, color="black", linewidth=0.9)
            if metric in log_scale:
                ax.set_yscale("symlog")
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"{SIGNAL_LABELS[signal]}\n{ylabel}")
            ax.set_xticks([group_centers[horizon] for horizon in horizons])
            ax.set_xticklabels([horizon_label(horizon) for horizon in horizons])
            ax.set_xlim(
                group_centers[horizons[0]] - 0.95,
                group_centers[horizons[-1]] + 0.95,
            )

    _workload_legend(fig, y=0.955)
    fig.suptitle(figure_title, y=0.995)
    fig.supxlabel("Planning horizon", y=0.01)
    fig.tight_layout(rect=(0, 0.03, 1, 0.90), pad=0.3, w_pad=0.5, h_pad=0.5)
    return fig, axes


def plot_improvement_vs_easy(data):
    """Did green beat EASY. Positive is better on every panel."""
    return plot_delta_boxes(
        data,
        [
            "footprint_improvement_pct",
            "energy_improvement_pct",
            "slowdown_improvement_pct",
        ],
        ["Target footprint", "Energy", "Bounded slowdown"],
        ["Improvement [% vs EASY]"] * 3,
        "Improvement over EASY by planning horizon (positive is better)",
    )


def plot_cost_vs_easy(data):
    """What displacement cost. Positive is better on every panel."""
    return plot_delta_boxes(
        data,
        [
            "makespan_improvement_pct",
            "utilisation_improvement_pct",
        ],
        ["Makespan", "Node utilisation"],
        ["Improvement [% vs EASY]"] * 2,
        "Scheduling cost against EASY by planning horizon (positive is better)",
    )


def plot_phase_exposure(run_metrics):
    """Distributions of per-run compute/idle shares by workload and variant."""
    data = run_metrics.copy()
    capacity_time = data["trace_end_seconds"] * data["nb_computing_machines"]
    data["compute_time_share_pct"] = 100 * data["time_computing"] / capacity_time
    data["idle_time_share_pct"] = 100 - data["compute_time_share_pct"]
    data["compute_energy_share_pct"] = 100 * data["compute_energy_share"]
    data["idle_energy_share_pct"] = 100 - data["compute_energy_share_pct"]
    green = data.loc[data["variant"].ne("easy_bf")].sort_values(
        ["objective", "horizon_seconds"]
    ).drop_duplicates("variant")
    variants = ["easy_bf"] + green["variant"].tolist()
    labels = ["EASY"] + [
        f"{row.objective[0].upper()}{row.horizon_seconds / 3600:g}h"
        for row in green.itertuples()
    ]
    panels = [("compute", "time", "Compute node-time [%]"),
              ("idle", "time", "Idle node-time [%]"),
              ("compute", "energy", "Compute energy [%]"),
              ("idle", "energy", "Idle energy [%]")]
    fig, axes = plt.subplots(len(panels), len(WORKLOAD_ORDER), figsize=(11, 9),
                             squeeze=False, sharex=True)
    for column, workload in enumerate(WORKLOAD_ORDER):
        subset = data.loc[data["workload_label"].eq(workload)]
        for row, (phase, measure, ylabel) in enumerate(panels):
            ax = axes[row, column]
            values = [subset.loc[subset["variant"].eq(variant),
                                 f"{phase}_{measure}_share_pct"].dropna()
                      for variant in variants]
            boxes = ax.boxplot(values, positions=np.arange(len(variants)),
                               widths=0.55, patch_artist=True, showfliers=True,
                               medianprops={"color": "black", "linewidth": 1.2},
                               flierprops={"marker": ".", "markersize": 2,
                                           "markeredgecolor": "black"})
            for box in boxes["boxes"]:
                box.set_facecolor("0.75" if phase == "compute" else "white")
            # Zoom each workload/phase to its full observed range, including
            # outliers. A small minimum span avoids magnifying numerical noise.
            observed = np.concatenate([v.to_numpy() for v in values])
            low, high = observed.min(), observed.max()
            span = max(high - low, 0.2)
            midpoint = (low + high) / 2
            ax.set_ylim(max(0, midpoint - span * 0.65),
                        min(100, midpoint + span * 0.65))
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.set_xticks(range(len(variants)), labels, rotation=90, ha="right")
            ax.set_xlim(-0.6, len(variants) - 0.4)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(WORKLOAD_LABELS[workload])
            if column == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle("Compute and idle exposure across environmental windows", y=0.995)
    fig.supxlabel("Scheduler variant: C = carbon objective; W = water objective; h = horizon hours",
                  y=0.01)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97), h_pad=1.0, w_pad=1.0)
    return fig, axes


def plot_intensity_decomposition(data):
    """Where the footprint change comes from: compute placement versus idle.

    Displacement moves compute out of dirty hours and leaves idle sitting in
    them, so the compute and idle lines usually pull in opposite directions and
    platform intensity is their energy-weighted blend. Percentage improvements
    themselves do not obey that blend when energy shares change.
    """
    horizons = horizon_order(data)
    series = [
        ("compute_intensity_improvement_pct", "Compute", "o", "-"),
        ("idle_intensity_improvement_pct", "Idle", "s", "--"),
        ("platform_intensity_improvement_pct", "Platform", "^", ":"),
    ]

    fig, axes = plt.subplots(
        len(SIGNAL_ORDER), len(WORKLOAD_ORDER),
        figsize=TREND_FIGURE_SIZE, squeeze=False, sharex=True,
    )
    for row, signal in enumerate(SIGNAL_ORDER):
        signal_data = data[data["signal"].eq(signal)]
        for column, workload in enumerate(WORKLOAD_ORDER):
            ax = axes[row][column]
            subset = signal_data[signal_data["workload_label"].eq(workload)]
            for metric, label, marker, style in series:
                medians = [
                    subset.loc[subset["horizon_hours"].eq(horizon), metric].median()
                    for horizon in horizons
                ]
                ax.plot(horizons, medians, color="black", marker=marker,
                        linestyle=style, markersize=DATA_MARKER_SIZE, linewidth=1.0,
                        markerfacecolor="white", label=label)
            ax.axhline(0, color="black", linewidth=0.9)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.margins(y=0.12)
            ax.set_xticks(horizons)
            ax.set_xticklabels([horizon_label(horizon) for horizon in horizons])
            if row == 0:
                ax.set_title(WORKLOAD_LABELS[workload])
            if column == 0:
                ax.set_ylabel(f"{SIGNAL_LABELS[signal]}\nimprovement [%]")

    handles = [
        Line2D([0], [0], color="black", marker=marker, linestyle=style,
               markerfacecolor="white", markersize=LEGEND_MARKER_SIZE, label=label)
        for _, label, marker, style in series
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.955), frameon=False)
    fig.suptitle("Intensity improvement over EASY, by phase", y=0.998)
    fig.supxlabel("Planning horizon", y=0.01)
    fig.tight_layout(rect=(0.01, 0.03, 1, 0.88), pad=0.3, w_pad=0.6, h_pad=0.5)
    return fig, axes


def plot_trace_coverage(coverage):
    """How far each variant's schedule runs past the end of the intensity trace.

    Beyond `makespan_over_trace = 1` the intensity signal is frozen at its last
    sample, so footprint accrued there is not scored against a real trace.
    """
    variants = [variant for variant in coverage["variant"].unique() if variant != "easy_bf"]
    variants = sorted(variants, key=lambda name: (name.split("_")[-2], int(name.split("_")[-1])))
    order = ["easy_bf"] + variants

    fig, axes = plt.subplots(1, 2, figsize=COVERAGE_FIGURE_SIZE, squeeze=False)
    axes = axes[0]
    panels = [
        ("makespan_over_trace", "Makespan / trace length", True),
        ("tail_carbon_share_pct", "Carbon accrued past trace end [%]", False),
    ]
    positions = range(1, len(order) + 1)
    for ax, (metric, ylabel, log) in zip(axes, panels):
        series = [coverage.loc[coverage["variant"].eq(variant), metric].dropna() for variant in order]
        box = ax.boxplot(series, positions=list(positions), widths=0.6,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "black", "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor("black")
        if log:
            ax.set_yscale("log")
            ax.axhline(1, color="black", linewidth=0.9, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(
            ["EASY"] + [f"{v.split('_')[-2][:1].upper()}{int(v.split('_')[-1])//3600}h" for v in variants],
            rotation=90, ha="right",
        )
    fig.suptitle("Trace coverage of completed schedules", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.3, w_pad=0.5)
    return fig, axes


def plot_tradeoff_scatter(data, improvement_column, x_label, title):
    """Environmental saving against bounded-slowdown penalty, per window.

    Rows are workloads, columns are planning horizons, marker shape is the
    signal and fill style is the season.
    """
    plot_data = data.copy()
    plot_data["environmental_saving_pct"] = plot_data[improvement_column]
    plot_data["bounded_slowdown_penalty_pct"] = plot_data[
        "slowdown_improvement_pct"
    ]

    horizons = horizon_order(plot_data)
    fig, axes = plt.subplots(
        len(WORKLOAD_ORDER),
        len(horizons),
        figsize=TRADEOFF_FIGURE_SIZE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_values = plot_data["environmental_saving_pct"]
    x_padding = max((x_values.max() - x_values.min()) * 0.12, 0.25)

    for row, workload in enumerate(WORKLOAD_ORDER):
        workload_data = plot_data[plot_data["workload_label"].eq(workload)]
        for ax, horizon in zip(axes[row], horizons):
            horizon_data = workload_data[workload_data["horizon_hours"].eq(horizon)]
            for signal in SIGNAL_ORDER:
                signal_data = horizon_data[horizon_data["signal"].eq(signal)]
                for season in SEASON_ORDER:
                    points = signal_data[signal_data["season"].eq(season)]
                    ax.plot(
                        points["environmental_saving_pct"],
                        points["bounded_slowdown_penalty_pct"],
                        linestyle="",
                        marker=monochrome_marker(
                            SIGNAL_MARKERS[signal], SEASON_FILLSTYLES[season]
                        ),
                        color="black",
                        markerfacecolor="black",
                        markerfacecoloralt="white",
                        markeredgecolor="black",
                        markeredgewidth=0.7,
                        markersize=DATA_MARKER_SIZE,
                        alpha=0.85,
                    )

            ax.axvline(0, color="black", linewidth=0.9)
            ax.set_yscale("symlog")
            ax.set_title(f"{WORKLOAD_LABELS[workload]} - {horizon_label(horizon)}")
            ax.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)

    season_handles = [
        Line2D([0], [0], marker=monochrome_marker("o", SEASON_FILLSTYLES[season]), linestyle="",
               markerfacecolor="black", markerfacecoloralt="white", markeredgecolor="black",
               color="black", label=season.title(), markersize=LEGEND_MARKER_SIZE)
        for season in SEASON_ORDER
    ]
    signal_handles = [
        Line2D([0], [0], marker=SIGNAL_MARKERS[signal], linestyle="", color="black",
               label=SIGNAL_LABELS[signal], markersize=LEGEND_MARKER_SIZE)
        for signal in SIGNAL_ORDER
    ]

    fig.legend(handles=season_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.legend(handles=signal_handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle(title, y=0.995)
    fig.supxlabel(x_label, y=0.01)
    fig.supylabel("Bounded slowdown improvement [% vs EASY]", x=0.005)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.905), pad=0.3, w_pad=0.4, h_pad=0.5)
    return fig, axes


def plot_swing_relationship(data):
    """Intra-day swing of the target signal against the saving it produced."""
    horizons = horizon_order(data)
    fig, axes = plt.subplots(
        len(SIGNAL_ORDER), len(horizons), figsize=SWING_FIGURE_SIZE,
        squeeze=False, sharex=True, sharey=True,
    )

    for row, signal in enumerate(SIGNAL_ORDER):
        signal_data = data[data["signal"].eq(signal)]
        for ax, horizon in zip(axes[row], horizons):
            cell = signal_data[signal_data["horizon_hours"].eq(horizon)]
            for workload in WORKLOAD_ORDER:
                subset = cell[cell["workload_label"].eq(workload)]
                for season in SEASON_ORDER:
                    points = subset[subset["season"].eq(season)]
                    ax.plot(
                        points["target_swing"],
                        points["footprint_improvement_pct"],
                        linestyle="",
                        color="black",
                        marker=monochrome_marker(
                            WORKLOAD_MARKERS[workload], SEASON_FILLSTYLES[season]
                        ),
                        markerfacecolor="black",
                        markerfacecoloralt="white",
                        markeredgecolor="black",
                        markeredgewidth=0.7,
                        markersize=DATA_MARKER_SIZE,
                        alpha=0.85,
                    )

            swing = cell["target_swing"].to_numpy()
            saving = cell["footprint_improvement_pct"].to_numpy()
            if len(swing) > 1:
                slope, intercept = np.polyfit(swing, saving, 1)
                grid = np.linspace(swing.min(), swing.max(), 50)
                ax.plot(grid, slope * grid + intercept, color="black", linestyle="-",
                        marker="", linewidth=1.1)
                r = np.corrcoef(swing, saving)[0, 1]
                ax.set_title(f"{SIGNAL_LABELS[signal]}, {horizon_label(horizon)} (r = {r:.2f})")
            else:
                ax.set_title(f"{SIGNAL_LABELS[signal]}, {horizon_label(horizon)}")

            ax.axhline(0, color="black", linewidth=0.9)

    workload_handles = [
        Line2D([0], [0], marker=WORKLOAD_MARKERS[workload], linestyle="", color="black",
               label=WORKLOAD_LABELS[workload], markersize=LEGEND_MARKER_SIZE)
        for workload in WORKLOAD_ORDER
    ]
    fig.legend(handles=workload_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.suptitle("Intra-day swing versus footprint improvement", y=0.98)
    fig.supxlabel("Intra-day swing of the target signal", y=0.10)
    fig.supylabel("Footprint improvement [% vs EASY]", x=0.005)
    fig.tight_layout(rect=(0.03, 0.14, 1, 0.93), pad=0.3, w_pad=0.5, h_pad=0.6)
    return fig, axes


# --- Comparison against the offline shifting ceiling -------------------------

def aggregate_saving_pct(frame, keys, baseline_column, achieved_column):
    """Saving as a ratio of summed footprints, the estimator the ceiling uses.

    A median of per-window percentages weights a tiny window the same as a huge
    one. Summing first keeps the two analyses on the same estimator.
    """
    # dropna=False keeps the EASY baseline, which has no planning horizon.
    totals = frame.groupby(keys, observed=True, dropna=False)[
        [baseline_column, achieved_column]
    ].sum()
    totals["saving_pct"] = (
        100.0
        * (totals[baseline_column] - totals[achieved_column])
        / totals[baseline_column]
    )
    return totals


def build_ceiling_costs(jobs):
    """Per run: the shifting-ceiling cost function evaluated on achieved starts.

    Reuses `shift_ceiling_results` so the expression is identical to the offline
    bound: non-context jobs only, submission time as the release date, and each
    job charged on its own nodes at compute power while it runs and at idle
    power while it waits. `baseline_*` puts every job at its submission instant,
    which is exactly the ceiling's own baseline, so the savings are comparable.

    This deliberately ignores platform-wide idle, which the ceiling cannot model.
    """
    import shift_ceiling_results as scr

    replay = jobs[jobs["job_kind"].eq("replay")]
    traces, prefixes = {}, {}
    for row in scr.load_windows().itertuples():
        key = (str(row.zone), str(row.start_date))
        trace = scr.load_trace(row)
        traces[key] = trace
        prefixes[key] = (
            scr.prefix_integral(trace.carbon),
            scr.prefix_integral(trace.water),
        )

    platforms = {}
    rows = []
    for name, group in replay.groupby("name", observed=True):
        key = (str(group["zone"].iloc[0]), str(group["start_date"].iloc[0]))
        dataset = str(group["dataset"].iloc[0])
        platform = platforms.setdefault(dataset, scr.load_platform(dataset))
        trace = traces[key]
        carbon_prefix, water_prefix = prefixes[key]

        submission = group["submission_time"].to_numpy(dtype=float)
        execution = group["execution_time"].to_numpy(dtype=float)
        nodes = group["requested_number_of_resources"].to_numpy(dtype=float)
        achieved = group["starting_time"].to_numpy(dtype=float)

        record = {"name": name}
        for label, starts in [("baseline", submission), ("achieved", achieved)]:
            at_submission_c = scr.integral_at(submission, trace.carbon, carbon_prefix)
            at_start_c = scr.integral_at(starts, trace.carbon, carbon_prefix)
            at_finish_c = scr.integral_at(starts + execution, trace.carbon, carbon_prefix)
            at_submission_w = scr.integral_at(submission, trace.water, water_prefix)
            at_start_w = scr.integral_at(starts, trace.water, water_prefix)
            at_finish_w = scr.integral_at(starts + execution, trace.water, water_prefix)
            record[f"{label}_carbon"] = float(
                np.sum(
                    nodes
                    * (
                        platform.compute_power_w * (at_finish_c - at_start_c)
                        + platform.idle_power_w * (at_start_c - at_submission_c)
                    )
                )
                / 1_000_000.0
            )
            record[f"{label}_water"] = float(
                np.sum(
                    nodes
                    * (
                        platform.compute_power_w * (at_finish_w - at_start_w)
                        + platform.idle_power_w * (at_start_w - at_submission_w)
                    )
                )
                / 1_000.0
            )
        rows.append(record)

    return pd.DataFrame(rows)


def summarize_ceiling_capture(ceiling_costs, run_metrics):
    """Ceiling-comparable saving per variant, signal and horizon.

    Positive means the schedule beat the "every job starts at submission"
    baseline that the offline bound uses. EASY usually scores negative, because
    that baseline assumes no queueing at all.
    """
    metrics = run_metrics[
        ["name", "variant", "objective", "horizon_seconds", "workload_label"]
    ]
    costs = ceiling_costs.merge(metrics, on="name", validate="one_to_one")
    costs["horizon_hours"] = (
        pd.to_numeric(costs["horizon_seconds"], errors="coerce") / 3600
    )
    # EASY displaces nothing, so it belongs against the bound at d_m = 0, where
    # the ceiling saving is 0 by construction. Its score is then pure queueing cost.
    costs.loc[costs["objective"].eq("baseline"), "horizon_hours"] = 0.0

    frames = []
    for signal in SIGNAL_ORDER:
        # The baseline runs on no signal, so report it against both.
        subset = costs[costs["objective"].isin([signal, "baseline"])].copy()
        subset["signal"] = signal
        table = aggregate_saving_pct(
            subset,
            ["signal", "variant", "horizon_hours"],
            f"baseline_{signal}",
            f"achieved_{signal}",
        )
        frames.append(table)
    return pd.concat(frames)[["saving_pct"]].round(2)


def load_ceiling_bound(results_path):
    """The offline bound as ratio-of-sums saving per signal and horizon.

    Returns None when `shift_ceiling_results.csv.gz` has not been generated.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        return None

    import shift_ceiling_analysis as sca

    analysis = sca.load_results(results_path)
    bound = aggregate_saving_pct(
        analysis.pair_summary,
        ["signal", "d_m_seconds"],
        "baseline",
        "shifted",
    ).reset_index()
    bound["horizon_hours"] = bound["d_m_seconds"] / 3600
    return bound[["signal", "horizon_hours", "saving_pct"]].round(2)


def compare_with_ceiling(ceiling_costs, run_metrics, results_path):
    """Achieved saving beside the offline bound, on one estimator and one scale."""
    achieved = summarize_ceiling_capture(ceiling_costs, run_metrics).reset_index()
    bound = load_ceiling_bound(results_path)
    if bound is None:
        return achieved

    merged = achieved.merge(
        bound.rename(columns={"saving_pct": "ceiling_saving_pct"}),
        on=["signal", "horizon_hours"],
        how="left",
    )
    # Percentage points short of the bound. A ratio would be meaningless here,
    # since a schedule that loses to its own baseline has a negative saving.
    merged["gap_to_ceiling_pp"] = (
        merged["ceiling_saving_pct"] - merged["saving_pct"]
    ).round(2)
    merged["captured_pct_of_ceiling"] = np.where(
        merged["saving_pct"] > 0,
        merged["saving_pct"] / merged["ceiling_saving_pct"] * 100,
        np.nan,
    ).round(1)
    return merged
