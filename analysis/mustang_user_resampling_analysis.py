"""User-level bootstrap of the fixed Mustang extracts, used by the notebook.

CSV timestamps and wallclock limits retain their historical values. Derived
runtime/walltime columns are used only for characterisation and Batsim export.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

import shift_ceiling_analysis as sca


CSV_COLUMNS = [
    "user_ID", "group_ID", "submit_time", "start_time", "end_time",
    "wallclock_limit", "job_status", "node_count", "tasks_requested",
]
NUM_NODES = 1600
NODE_SPEED = 24 * 2.3e9 * 2
WIDE_NODES = 160
LONG_SECONDS = 2 * 3600
DEFAULT_REGIMES = {
    "slack": ("2015-04-13", "2015-05-11"),
    "stress": ("2013-01-07", "2013-02-04"),
}
DEFAULT_WEIGHTS = {"random": 1.0, "heavy_2x": 2.0, "heavy_4x": 4.0, "heavy_8x": 8.0}
DEFAULT_SEEDS = list(range(20260926, 20260931))
DEFAULT_JOB_NODE_HOURS_QUANTILE = 0.75
DEFAULT_MIN_JOBS_FOR_HEAVY = 10
DEFAULT_HEAVY_USER_THRESHOLD_QUANTILE = 0.75
FIGURE_DIR = Path(__file__).with_name("figures")
FIGURE_PREFIX = "mustang_user_resampling"
SCENARIO_ORDER = ("original", "random", "heavy_2x", "heavy_4x", "heavy_8x")
SCENARIO_LABELS = {
    "original": "Original",
    "random": "Random",
    "heavy_2x": "Heavy 2x",
    "heavy_4x": "Heavy 4x",
    "heavy_8x": "Heavy 8x",
}
SCENARIO_COLORS = {
    "original": "black",
    "random": "#8C8C8C",
    "heavy_2x": "#4C78A8",
    "heavy_4x": "#F58518",
    "heavy_8x": "#54A24B",
}
REGIME_COLORS = {
    "slack": "#4C78A8",
    "stress": "#E45756",
}


def configure_plot_style():
    """Apply the repository's publication figure style with a font fallback."""
    sca.configure_plot_style()
    try:
        font_manager.findfont("Roboto Condensed", fallback_to_default=False)
    except ValueError:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "dejavusans",
            }
        )


def require(condition, message):
    """Keep scientific validation active even when Python runs with -O."""
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_window(bounds):
    t0, t1 = (pd.to_datetime(value, utc=True) for value in bounds)
    require(t1 > t0, "Window end must be after its start")
    return t0, t1


def prepare_jobs(raw, source_row_start=1):
    """Apply Mustang hygiene; source_row is a 1-based data-row ordinal.

    Keep original string fields for CSV round trips. Reject missing/noncausal
    submissions as well as the original notebook's invalid execution records.
    Exclusion reason counts can overlap and are reported explicitly.
    """
    require(list(raw.columns) == CSV_COLUMNS, "Unexpected Mustang CSV schema/order")
    frame = raw.copy()
    frame["source_row"] = np.arange(source_row_start, source_row_start + len(raw))
    for name in ["submit", "start", "end"]:
        frame[f"{name}_utc"] = pd.to_datetime(
            frame[f"{name}_time"], utc=True, errors="coerce", format="mixed"
        )
    nodes = pd.to_numeric(frame["node_count"], errors="coerce")
    users = pd.to_numeric(frame["user_ID"], errors="coerce")
    valid_nodes = nodes.between(1, NUM_NODES) & nodes.mod(1).eq(0)
    valid_users = users.ge(0) & np.isfinite(users) & users.mod(1).eq(0)
    valid_execution = frame["start_utc"].notna() & frame["start_utc"].le(frame["end_utc"])
    valid_submission = frame["submit_utc"].notna() & frame["submit_utc"].le(frame["start_utc"])
    keep = valid_nodes & valid_users & valid_execution & valid_submission
    audit = {
        "input_rows": len(raw), "kept_rows": int(keep.sum()),
        "excluded_rows": int((~keep).sum()),
        "invalid_nodes": int((~valid_nodes).sum()),
        "invalid_user": int((~valid_users).sum()),
        "invalid_execution_times": int((~valid_execution).sum()),
        "invalid_submission_times": int((~valid_submission).sum()),
    }
    frame = frame.loc[keep].copy()
    frame["node_count"] = nodes.loc[keep].astype("int64")
    frame["user_ID"] = users.loc[keep].astype("int64")
    frame["source_user_ID"] = frame["user_ID"]
    duration = (frame["end_utc"] - frame["start_utc"]).dt.total_seconds()
    frame["runtime_sec"] = duration.clip(lower=1.0)
    wallclock = pd.to_timedelta(frame["wallclock_limit"], errors="coerce").dt.total_seconds()
    frame["walltime_sec"] = np.ceil(
        wallclock.where(wallclock > 0, frame["runtime_sec"]).clip(lower=frame["runtime_sec"])
    )
    frame["node_hours"] = frame["node_count"] * frame["runtime_sec"] / 3600
    audit["runtime_clipped_rows"] = int(duration.lt(1).sum())
    audit["walltime_fallback_rows"] = int((~wallclock.gt(0)).sum())
    audit["walltime_raised_rows"] = int((wallclock.gt(0) & wallclock.lt(frame["runtime_sec"])).sum())
    return frame, audit


def load_source(path, regimes, chunksize=200_000):
    """Audit the full CSV, retaining only replay and context rows of interest."""
    require(Path(path).is_file(), f"Missing Mustang dataset: {path}")
    windows = [utc_window(bounds) for bounds in regimes.values()]
    frames, counts, offset = [], Counter(), 1
    for raw in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunksize):
        frame, audit = prepare_jobs(raw, offset)
        offset += len(raw)
        counts.update(audit)
        needed = pd.Series(False, index=frame.index)
        for t0, t1 in windows:
            needed |= (
                (frame["submit_utc"].ge(t0) & frame["submit_utc"].lt(t1))
                | (frame["start_utc"].lt(t0) & frame["end_utc"].ge(t0))
                | (frame["submit_utc"].lt(t0) & frame["start_utc"].ge(t0))
            )
        frames.append(frame.loc[needed])
    require(bool(frames), "Empty source CSV")
    source = pd.concat(frames, ignore_index=True)
    require(not source.empty, "No valid jobs in the configured extracts")
    counts["retained_for_extracts"] = len(source)
    return source, dict(counts)


def add_job(jobs, profiles, job_id, subtime, nodes, runtime, walltime):
    profiles[job_id] = {"type": "parallel_homogeneous", "cpu": float(runtime) * NODE_SPEED, "com": 0.0}
    jobs.append({
        "id": job_id, "subtime": float(subtime), "res": int(nodes),
        "walltime": int(max(walltime, np.ceil(runtime))), "profile": job_id,
    })


def get_context(workload):
    jobs = [job for job in workload["jobs"] if job["id"].startswith("ctx_")]
    return deepcopy({
        "nb_res": NUM_NODES, "jobs": jobs,
        "profiles": {job["profile"]: workload["profiles"][job["profile"]] for job in jobs},
    })


def build_workload(replay, context, t0):
    """Replay rows are already ordered; append them to the exact fixed context."""
    result = deepcopy(context)
    for i, row in enumerate(replay.itertuples(index=False), 1):
        add_job(result["jobs"], result["profiles"], f"job{i}",
                (row.submit_utc - t0).total_seconds(), row.node_count,
                row.runtime_sec, row.walltime_sec)
    return result


def job_signature(job, profiles):
    profile = profiles[job["profile"]]
    return (job["subtime"], job["res"], job["walltime"],
            profile["type"], profile["cpu"], profile["com"])


def prepare_regime(source, reference, bounds):
    """Check the raw extract and retain the reference's exact replay tie order.

    Match jobs by all simulated attributes, using source-row order for identical
    duplicates. This avoids dependence on pandas' unstable legacy tie sorting.
    Context is verified against the raw trace before it is copied verbatim.
    """
    t0, t1 = utc_window(bounds)
    validate_workload(reference, t1 - t0)
    context = get_context(reference)
    reconstructed = {"nb_res": NUM_NODES, "jobs": [], "profiles": {}}
    running = source.loc[source["start_utc"].lt(t0) & source["end_utc"].ge(t0)]
    waiting = source.loc[source["submit_utc"].lt(t0) & source["start_utc"].ge(t0)]
    for i, row in enumerate(running.sort_values(["start_utc", "source_row"]).itertuples(), 1):
        elapsed = (t0 - row.start_utc).total_seconds()
        add_job(reconstructed["jobs"], reconstructed["profiles"], f"ctx_run{i}", 0,
                row.node_count, max((row.end_utc - t0).total_seconds(), 1), row.walltime_sec - elapsed)
    for i, row in enumerate(waiting.sort_values(["submit_utc", "source_row"]).itertuples(), 1):
        add_job(reconstructed["jobs"], reconstructed["profiles"], f"ctx_queued{i}", 0,
                row.node_count, row.runtime_sec, row.walltime_sec)
    def context_signatures(workload):
        return Counter((job["id"].rstrip("0123456789"), job_signature(job, workload["profiles"]))
                       for job in workload["jobs"])
    require(context_signatures(reconstructed) == context_signatures(context),
            "Raw-trace warm-up differs from the existing baseline")
    pool = source_replay_pool(source, bounds)
    candidate = build_workload(pool, {"nb_res": NUM_NODES, "jobs": [], "profiles": {}}, t0)
    positions = defaultdict(deque)
    for i, job in enumerate(candidate["jobs"]):
        positions[job_signature(job, candidate["profiles"])].append(i)
    order = []
    for job in reference["jobs"]:
        if job["id"].startswith("ctx_"):
            continue
        signature = job_signature(job, reference["profiles"])
        require(bool(positions[signature]), "Raw replay differs from the existing baseline")
        order.append(positions[signature].popleft())
    require(len(order) == len(pool), "Baseline is missing raw replay jobs")
    pool = pool.iloc[order].reset_index(drop=True)
    require(build_workload(pool, context, t0) == reference,
            "Baseline IDs/order differ from the canonical Mustang JSON")
    return pool, context


def source_replay_pool(source, bounds):
    """Return valid replay jobs in a half-open source window.

    This is intentionally independent from the canonical Batsim tie ordering:
    it supports pre-resampling source characterisation only, without reading or
    writing a workload artifact.
    """
    t0, t1 = utc_window(bounds)
    pool = source.loc[source["submit_utc"].ge(t0) & source["submit_utc"].lt(t1)].copy()
    pool = pool.sort_values(["submit_utc", "source_row"]).reset_index(drop=True)
    require(not pool.empty, "Configured source window has no eligible replay jobs")
    return pool


def measure_user_activity(pool, job_node_hours_quantile=DEFAULT_JOB_NODE_HOURS_QUANTILE,
                          min_jobs_for_heavy=DEFAULT_MIN_JOBS_FOR_HEAVY):
    """Measure the two inputs used later to classify source users.

    This deliberately stops before choosing the heavy-user percentile so the
    notebook can inspect the high-job-share distribution first. The production
    classifier below calls the same helper, keeping calibration and generation
    on one implementation path.
    """
    require(0 < job_node_hours_quantile < 1,
            "job_node_hours_quantile must be between 0 and 1")
    require(isinstance(min_jobs_for_heavy, (int, np.integer)) and min_jobs_for_heavy >= 1,
            "min_jobs_for_heavy must be a positive integer")
    require(not pool.empty, "Cannot characterize an empty user population")
    users = pool.groupby("source_user_ID", sort=True).agg(
        jobs=("source_row", "size"), total_node_hours=("node_hours", "sum"),
        mean_node_count=("node_count", "mean"), max_node_count=("node_count", "max"),
        mean_runtime_sec=("runtime_sec", "mean"), max_runtime_sec=("runtime_sec", "max"),
    )
    users["wide_job_share"] = pool["node_count"].ge(WIDE_NODES).groupby(pool["source_user_ID"]).mean()
    users["long_job_share"] = pool["runtime_sec"].gt(LONG_SECONDS).groupby(pool["source_user_ID"]).mean()
    job_threshold = float(pool["node_hours"].quantile(job_node_hours_quantile, interpolation="linear"))
    high_jobs = pool["node_hours"].gt(job_threshold)
    users["high_node_hours_jobs"] = high_jobs.groupby(pool["source_user_ID"]).sum().astype("int64")
    users["high_node_hours_job_share"] = high_jobs.groupby(pool["source_user_ID"]).mean()
    users["heavy_eligible"] = users["jobs"].ge(min_jobs_for_heavy)
    users["job_node_hours_quantile"] = job_node_hours_quantile
    users["job_node_hours_threshold"] = job_threshold
    users["min_jobs_for_heavy"] = min_jobs_for_heavy
    return users


def characterize_users(pool, job_node_hours_quantile=DEFAULT_JOB_NODE_HOURS_QUANTILE,
                       min_jobs_for_heavy=DEFAULT_MIN_JOBS_FOR_HEAVY,
                       heavy_user_threshold_quantile=DEFAULT_HEAVY_USER_THRESHOLD_QUANTILE):
    """Classify users by a sustained share of high-node-hour jobs.

    A high-node-hour job exceeds the source regime's empirical job-level
    quantile. Users need a minimum number of jobs to be eligible; heavy users
    are the upper user quantile of the eligible users' high-job share. Ties at
    the user threshold remain heavy.
    """
    require(0 < heavy_user_threshold_quantile < 1,
            "heavy_user_threshold_quantile must be between 0 and 1")
    users = measure_user_activity(
        pool,
        job_node_hours_quantile=job_node_hours_quantile,
        min_jobs_for_heavy=min_jobs_for_heavy,
    )
    eligible_shares = users.loc[users["heavy_eligible"], "high_node_hours_job_share"]
    require(not eligible_shares.empty,
            "No users meet min_jobs_for_heavy; lower the minimum or expand the window")
    user_threshold = float(
        eligible_shares.quantile(heavy_user_threshold_quantile, interpolation="linear")
    )
    users["is_heavy"] = users["heavy_eligible"] & users["high_node_hours_job_share"].ge(user_threshold)
    users["heavy_user_threshold_quantile"] = heavy_user_threshold_quantile
    users["heavy_threshold_high_node_hours_job_share"] = user_threshold
    return users


def format_hygiene_summary(hygiene):
    """Produce the compact source-validation table displayed by the notebook."""
    labels = {
        "input_rows": "Source rows",
        "kept_rows": "Eligible execution rows",
        "excluded_rows": "Excluded rows",
        "invalid_nodes": "Rows with invalid node count",
        "invalid_user": "Rows with invalid user ID",
        "invalid_execution_times": "Rows with invalid execution times",
        "invalid_submission_times": "Rows with invalid submission times",
        "runtime_clipped_rows": "Runtimes raised to 1 second",
        "walltime_fallback_rows": "Rows using runtime as walltime",
        "walltime_raised_rows": "Walltimes raised to runtime",
        "retained_for_extracts": "Rows retained for both extracts",
    }
    return pd.DataFrame(
        [(labels[key], int(hygiene[key])) for key in labels if key in hygiene],
        columns=["Validation check", "Rows"],
    ).set_index("Validation check")


def format_source_job_summary(summary):
    """Format original job-distribution statistics with explicit units."""
    report = summary.copy()
    for column in [
        "jobs_gt_2h_share", "jobs_ge_160_nodes_share",
        "jobs_gt_2h_and_ge_160_nodes_share",
    ]:
        report[column] *= 100
    report = report.set_index("regime").rename(
        index=lambda value: value.title(),
        columns={
            "jobs": "Replay jobs",
            "runtime_p50_h": "Runtime P50 [h]",
            "runtime_p90_h": "Runtime P90 [h]",
            "runtime_p99_h": "Runtime P99 [h]",
            "node_count_p50": "Nodes P50",
            "node_count_p90": "Nodes P90",
            "node_count_p99": "Nodes P99",
            "jobs_gt_2h_share": "Jobs >2 h [%]",
            "jobs_ge_160_nodes_share": "Jobs >=160 nodes [%]",
            "jobs_gt_2h_and_ge_160_nodes_share": "Both [%]",
        },
    )
    report.index.name = "Regime"
    return report.round(2)


def summarize_user_classification(user_profiles):
    """Describe the fixed source-user classification once per regime."""
    rows = []
    for regime, users in user_profiles.groupby("regime", sort=False):
        job_quantile = float(users["job_node_hours_quantile"].iloc[0])
        user_quantile = float(users["heavy_user_threshold_quantile"].iloc[0])
        rows.append({
            "regime": regime,
            "replay_jobs": int(users["jobs"].sum()),
            "source_users": len(users),
            "eligible_users": int(users["heavy_eligible"].sum()),
            "heavy_users": int(users["is_heavy"].sum()),
            "job_threshold_quantile": job_quantile,
            "job_node_hours_threshold": float(users["job_node_hours_threshold"].iloc[0]),
            "minimum_jobs": int(users["min_jobs_for_heavy"].iloc[0]),
            "user_threshold_quantile": user_quantile,
            "high_job_share_threshold": float(
                users["heavy_threshold_high_node_hours_job_share"].iloc[0]
            ),
            "nominal_upper_tail_share": 1 - user_quantile,
        })
    return pd.DataFrame(rows)


def format_user_classification(summary):
    """Format source-user classification with percentiles and shares visible."""
    report = summary.copy()
    for column in [
        "job_threshold_quantile", "user_threshold_quantile",
        "high_job_share_threshold", "nominal_upper_tail_share",
    ]:
        report[column] *= 100
    report = report.set_index("regime").rename(
        index=lambda value: value.title(),
        columns={
            "replay_jobs": "Replay jobs",
            "source_users": "Source users",
            "eligible_users": "Eligible users",
            "heavy_users": "Heavy users",
            "job_threshold_quantile": "High-job percentile",
            "job_node_hours_threshold": "High-job threshold [node-h]",
            "minimum_jobs": "Minimum jobs",
            "user_threshold_quantile": "User threshold percentile",
            "high_job_share_threshold": "Required high-job share [%]",
            "nominal_upper_tail_share": "Nominal upper tail [%]",
        },
    )
    report.index.name = "Regime"
    return report.round(2)


def format_expected_composition(expected):
    """Format analytical expectations without repeating source classifications."""
    report = expected.copy()
    report["expected_heavy_user_share"] *= 100
    report = report.set_index(["regime", "scenario"]).rename(
        index=lambda value: SCENARIO_LABELS.get(value, value.replace("_", " ").title()),
        columns={
            "expected_heavy_user_share": "Expected heavy draws [%]",
            "expected_jobs": "Expected replay jobs",
            "expected_node_hours": "Expected node-hours",
        },
    )
    report.index = report.index.set_levels(
        [report.index.levels[0].str.title(), report.index.levels[1]]
    )
    report.index.names = ["Regime", "Scenario"]
    return report.round(2)


def format_collection_summary(summary):
    """Summarize persisted workloads at regime/scenario granularity."""
    report = summary.groupby(["regime", "scenario"], sort=False).agg(
        Workloads=("workload", "size"),
        **{
            "Replay jobs (mean)": ("jobs", "mean"),
            "Replay jobs (min)": ("jobs", "min"),
            "Replay jobs (max)": ("jobs", "max"),
            "Node-hours (mean)": ("total_node_hours", "mean"),
            "Heavy draws [%]": ("heavy_user_share", lambda values: 100 * values.mean()),
            "Wide jobs [%]": ("wide_job_share", lambda values: 100 * values.mean()),
            "Long jobs [%]": ("long_job_share", lambda values: 100 * values.mean()),
            "Hourly submission Pearson r": ("hourly_submission_pearson_r", "mean"),
            "Work / capacity": ("offered_work_capacity_ratio", "mean"),
            "Validated": ("validation_passed", "all"),
        },
    )
    report.index = pd.MultiIndex.from_tuples(
        [
            (regime.title(), SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title()))
            for regime, scenario in report.index
        ],
        names=["Regime", "Scenario"],
    )
    return report.round(2)


def format_hourly_submission_correlations(summary):
    """Present one selectable workload ID and hourly correlation per trace."""
    report = summary[[
        "workload", "regime", "scenario", "seed", "hourly_submission_pearson_r",
    ]].copy()
    report["Scenario"] = report["scenario"].map(
        lambda value: SCENARIO_LABELS.get(value, value.replace("_", " ").title())
    )
    report["Seed"] = report["seed"].map(
        lambda value: "Baseline" if pd.isna(value) else str(int(value))
    )
    report["Regime"] = report["regime"].str.title()
    report["_scenario_order"] = pd.Categorical(
        report["scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    report = report.sort_values(
        ["regime", "_scenario_order", "seed"], na_position="first"
    )
    return (
        report.rename(columns={
            "workload": "Workload ID",
            "hourly_submission_pearson_r": "Pearson r vs original",
        })
        .set_index("Workload ID")[["Regime", "Scenario", "Seed", "Pearson r vs original"]]
        .round(4)
    )


def format_manifest_preview(manifest, rows=10):
    """Show one representative user draw per regime/scenario without null labels."""
    representatives = manifest.groupby(
        ["regime", "scenario"], sort=False, group_keys=False
    ).head(1).head(rows)
    report = representatives[
        [
            "workload", "synthetic_user_ID", "source_user_ID", "jobs",
            "high_node_hours_jobs", "high_node_hours_job_share",
            "heavy_eligible", "is_heavy", "selection_weight",
            "selection_probability", "seed",
        ]
    ].head(rows).copy()
    report["high_node_hours_job_share"] *= 100
    report["selection_probability"] *= 100
    report["selection_probability"] = report["selection_probability"].map(
        lambda value: "Identity" if pd.isna(value) else f"{value:.2f}%"
    )
    report["seed"] = report["seed"].map(
        lambda value: "Baseline" if pd.isna(value) else str(int(value))
    )
    return report.rename(columns={
        "workload": "Workload",
        "synthetic_user_ID": "Synthetic user",
        "source_user_ID": "Source user",
        "jobs": "Jobs",
        "high_node_hours_jobs": "High jobs",
        "high_node_hours_job_share": "High jobs [%]",
        "heavy_eligible": "Eligible",
        "is_heavy": "Heavy",
        "selection_weight": "Weight",
        "selection_probability": "Draw probability",
        "seed": "Seed",
    }).round(2)


def random_stream(seed, regime, scenario):
    """Stable named streams: reordering configurations cannot change a trace."""
    stream_id = int.from_bytes(hashlib.sha256(f"{regime}/{scenario}".encode()).digest()[:4], "big")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), stream_id]))), stream_id


def resample(pool, users, regime, scenario, heavy_weight=1.0, seed=None):
    require(math.isfinite(heavy_weight) and heavy_weight >= 1, "Heavy weight must be finite and >= 1")
    original = scenario == "original"
    require(original or (seed is not None and int(seed) == seed and seed >= 0), "A nonnegative integer seed is required")
    ids = users.index.to_numpy()
    weights = np.where(users["is_heavy"], heavy_weight, 1.0)
    probabilities = weights / weights.sum()
    if original:
        selected, stream_id = ids, None
    else:
        rng, stream_id = random_stream(seed, regime, scenario)
        selected = rng.choice(ids, size=len(ids), replace=True, p=probabilities)
    manifest = pd.DataFrame({
        "synthetic_user_ID": selected if original else np.arange(1, len(ids) + 1),
        "source_user_ID": selected, "draw_order": np.arange(1, len(ids) + 1),
    })
    manifest = manifest.join(users, on="source_user_ID")
    manifest["selection_weight"] = np.where(manifest["is_heavy"], heavy_weight, 1.0)
    manifest["selection_probability"] = manifest["selection_weight"] / weights.sum() if not original else np.nan
    manifest["seed"] = pd.array([seed] * len(ids), dtype="Int64")
    manifest["rng_stream_id"] = pd.array([stream_id] * len(ids), dtype="Int64")
    manifest["regime"] = regime
    manifest["scenario"] = scenario
    manifest["heavy_weight"] = heavy_weight
    if original:
        replay = pool.copy()
        replay["draw_order"] = replay["source_user_ID"].map(manifest.set_index("source_user_ID")["draw_order"])
    else:
        groups = {user: rows for user, rows in pool.groupby("source_user_ID", sort=False)}
        copies = []
        for row in manifest.itertuples(index=False):
            copy = groups[row.source_user_ID].copy()
            copy["user_ID"] = row.synthetic_user_ID
            copy["draw_order"] = row.draw_order
            copies.append(copy)
        replay = pd.concat(copies, ignore_index=True)
        replay = replay.sort_values(["submit_utc", "source_row", "draw_order"]).reset_index(drop=True)
    replay["is_heavy"] = replay["source_user_ID"].map(users["is_heavy"])
    return replay, manifest


def validate_resampling(replay, pool, manifest, users):
    """Check full copied subtraces, identity, provenance and all job attributes."""
    require(len(manifest) == len(users), "Wrong number of user draws")
    require(manifest["synthetic_user_ID"].is_unique, "Synthetic user IDs collide")
    require(set(replay["user_ID"]) == set(manifest["synthetic_user_ID"]), "Replay/manifest users differ")
    require(not replay.duplicated(["user_ID", "source_row"]).any(), "A copy repeats a source job")
    require(replay["submit_utc"].is_monotonic_increasing, "Submissions are not sorted")
    source = pool.set_index("source_row")
    expected = source.loc[replay["source_row"]].reset_index(drop=True)
    preserved = [c for c in CSV_COLUMNS if c != "user_ID"] + [
        "source_user_ID", "submit_utc", "start_utc", "end_utc", "runtime_sec", "walltime_sec", "node_hours",
    ]
    pd.testing.assert_frame_equal(replay[preserved].reset_index(drop=True), expected[preserved])
    by_id = manifest.set_index("synthetic_user_ID")
    require(np.array_equal(replay["source_user_ID"], replay["user_ID"].map(by_id["source_user_ID"])),
            "A copied user contains jobs of another source user")
    counts = replay.groupby("user_ID").size().sort_index()
    pd.testing.assert_series_equal(counts, by_id["jobs"].sort_index(), check_names=False, check_dtype=False)
    np.testing.assert_allclose(replay.groupby("user_ID")["node_hours"].sum().sort_index(),
                               by_id["total_node_hours"].sort_index(), rtol=1e-12)
    required = [
        "source_user_ID", "draw_order", "is_heavy", "heavy_eligible",
        "high_node_hours_jobs", "high_node_hours_job_share", "total_node_hours",
        "job_node_hours_quantile", "job_node_hours_threshold", "min_jobs_for_heavy",
        "heavy_user_threshold_quantile", "heavy_threshold_high_node_hours_job_share",
        "selection_weight", "regime", "scenario",
    ]
    require(not manifest[required].isna().any().any(), "Incomplete manifest provenance")
    require(np.array_equal(manifest["is_heavy"], manifest["source_user_ID"].map(users["is_heavy"])),
            "Manifest heavy classification differs from the source")
    if not manifest["scenario"].eq("original").all():
        require(not manifest[["seed", "rng_stream_id", "selection_probability"]].isna().any().any(),
                "Missing stochastic provenance")
        weights = np.where(users["is_heavy"], manifest["heavy_weight"].iloc[0], 1.0)
        np.testing.assert_allclose(manifest["selection_probability"], manifest["selection_weight"] / weights.sum())


def validate_workload(workload, window, context=None, expected_replay=None):
    require(workload["nb_res"] == NUM_NODES, "Wrong Batsim capacity")
    jobs, profiles = workload["jobs"], workload["profiles"]
    require(len({job["id"] for job in jobs}) == len(jobs), "Duplicate Batsim job IDs")
    require(len(profiles) == len(jobs) and {j["profile"] for j in jobs} == set(profiles),
            "Missing, shared or orphan Batsim profiles")
    replay = []
    for job in jobs:
        profile = profiles[job["profile"]]
        require(profile["type"] == "parallel_homogeneous" and profile["com"] == 0, "Unexpected Batsim profile")
        runtime = profile["cpu"] / NODE_SPEED
        require(math.isfinite(runtime) and runtime > 0, "Invalid runtime")
        require(1 <= job["res"] <= NUM_NODES and int(job["res"]) == job["res"], "Invalid node count")
        require(math.isfinite(job["walltime"]) and int(job["walltime"]) == job["walltime"]
                and job["walltime"] + 1e-8 >= runtime, "Invalid effective walltime")
        require(math.isfinite(job["subtime"]) and 0 <= job["subtime"] < window.total_seconds(), "Invalid submission")
        if job["id"].startswith("ctx_"):
            require(job["subtime"] == 0, "Context must arrive at t=0")
        else:
            require(job["id"].startswith("job"), "Unexpected job ID kind")
            replay.append(job)
    require([job["subtime"] for job in jobs] == sorted(job["subtime"] for job in jobs), "Unsorted workload")
    if context is not None:
        require(get_context(workload) == context, "Warm-up changed")
        require(jobs[:len(context["jobs"])] == context["jobs"], "Warm-up order changed")
    if expected_replay is not None:
        require(len(replay) == expected_replay, "Wrong replay count")


def summarize(replay, manifest, users, window):
    total = float(replay["node_hours"].sum())
    amounts = replay.groupby("user_ID")["node_hours"].sum()
    source_amounts = replay.groupby("source_user_ID")["node_hours"].sum()
    weight = float(manifest["heavy_weight"].iloc[0])
    weights = np.where(users["is_heavy"], weight, 1.0)
    original = manifest["scenario"].iloc[0] == "original"
    return {
        "jobs": len(replay), "users": len(manifest),
        "unique_source_users": manifest["source_user_ID"].nunique(),
        "heavy_eligible_users": int(manifest["heavy_eligible"].sum()),
        "heavy_users": int(manifest["is_heavy"].sum()),
        "heavy_user_share": float(manifest["is_heavy"].mean()),
        "heavy_job_share": float(replay["is_heavy"].mean()),
        "heavy_node_hours_share": float(replay.loc[replay["is_heavy"], "node_hours"].sum() / total),
        "total_node_hours": total,
        "wide_job_share": float(replay["node_count"].ge(WIDE_NODES).mean()),
        "long_job_share": float(replay["runtime_sec"].gt(LONG_SECONDS).mean()),
        "top_user_node_hours_share": float(amounts.max() / total),
        "top_source_user_node_hours_share": float(source_amounts.max() / total),
        "max_source_copies": int(manifest["source_user_ID"].value_counts().max()),
        "expected_jobs": float(len(users) * np.average(users["jobs"], weights=weights)),
        "expected_node_hours": float(len(users) * np.average(users["total_node_hours"], weights=weights)),
        "expected_heavy_user_share": float(np.average(users["is_heavy"], weights=weights)),
        "offered_work_capacity_ratio": total / (NUM_NODES * window.total_seconds() / 3600),
        "sampling": "identity" if original else "with_replacement",
    }


def hourly_submission_counts(replay, bounds):
    """Return one aligned submission count for every hour in a replay window."""
    t0, t1 = utc_window(bounds)
    hourly_index = pd.date_range(t0, t1, freq="h", inclusive="left")
    counts = (
        replay.set_index("submit_utc")
        .resample("h")
        .size()
        .reindex(hourly_index, fill_value=0)
        .astype("int64")
    )
    require(len(counts) == int((t1 - t0).total_seconds() / 3600),
            "Hourly submission series does not cover the complete window")
    return counts


def hourly_submission_pearson_r(replay, original, bounds):
    """Correlate aligned hourly submissions with the regime's original trace."""
    candidate_counts = hourly_submission_counts(replay, bounds).to_numpy(dtype=float)
    original_counts = hourly_submission_counts(original, bounds).to_numpy(dtype=float)
    require(candidate_counts.std() > 0 and original_counts.std() > 0,
            "Hourly submission correlation requires nonconstant series")
    correlation = float(np.corrcoef(candidate_counts, original_counts)[0, 1])
    require(math.isfinite(correlation) and -1 <= correlation <= 1,
            "Invalid hourly submission correlation")
    return correlation


def validate_csv(path, expected):
    actual = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(list(actual.columns) == CSV_COLUMNS, "Exported CSV schema differs")
    pd.testing.assert_frame_equal(actual, expected[CSV_COLUMNS].astype(str).reset_index(drop=True))
    prepared, audit = prepare_jobs(actual)
    require(audit["excluded_rows"] == 0, "Exported CSV contains invalid jobs")
    require(prepared["walltime_sec"].ge(prepared["runtime_sec"]).all(), "Invalid effective CSV walltime")
    return prepared


def generate_artifacts(source, workload_dir, output_dir, regimes=None,
                       job_node_hours_quantile=DEFAULT_JOB_NODE_HOURS_QUANTILE,
                       min_jobs_for_heavy=DEFAULT_MIN_JOBS_FOR_HEAVY,
                       heavy_user_threshold_quantile=DEFAULT_HEAVY_USER_THRESHOLD_QUANTILE,
                       scenario_weights=None, seeds=None, source_path=None, hygiene=None):
    """Generate/audit baseline + replicas, then publish collection manifests.

    All raw/baseline comparisons run before any output files are created.
    Reruns replace the same named artifacts; existing unrelated files remain.
    Only workloads listed in summary.csv belong to the current configuration.
    """
    regimes = DEFAULT_REGIMES if regimes is None else regimes
    scenario_weights = DEFAULT_WEIGHTS if scenario_weights is None else scenario_weights
    seeds = DEFAULT_SEEDS if seeds is None else list(seeds)
    require(bool(regimes) and bool(scenario_weights) and bool(seeds), "Configuration cannot be empty")
    require(len(set(seeds)) == len(seeds), "Seeds must be unique")
    require(all(isinstance(s, (int, np.integer)) and s >= 0 for s in seeds), "Seeds must be nonnegative integers")
    require("original" not in scenario_weights, "original is reserved for the identity baseline")
    for name in [*regimes, *scenario_weights]:
        require(name and all(c.isascii() and (c.isalnum() or c == "_") for c in name), "Unsafe regime/scenario name")
    require(all(math.isfinite(w) and w >= 1 for w in scenario_weights.values()), "Weights must be finite and >= 1")
    require(scenario_weights.get("random", 1) == 1, "random must use uniform weights")
    output_dir, workload_dir = Path(output_dir), Path(workload_dir)
    prepared = {}
    for regime, bounds in regimes.items():
        reference = json.loads((workload_dir / f"mustang_{regime}.json").read_text())
        pool, context = prepare_regime(source, reference, bounds)
        users = characterize_users(
            pool,
            job_node_hours_quantile=job_node_hours_quantile,
            min_jobs_for_heavy=min_jobs_for_heavy,
            heavy_user_threshold_quantile=heavy_user_threshold_quantile,
        )
        prepared[regime] = (pool, context, users, reference)
    # One flat output directory: CSV/JSON pairs share their workload identifier.
    require(output_dir.resolve() != workload_dir.resolve(), "Use a separate generated output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, manifests, profiles, frames = [], [], [], {}
    for regime, (pool, context, users, reference) in prepared.items():
        t0, t1 = utc_window(regimes[regime])
        profiles.append(users.reset_index().assign(regime=regime))
        configurations = [("original", 1.0, None)] + [
            (scenario, weight, seed) for scenario, weight in scenario_weights.items() for seed in seeds
        ]
        for scenario, weight, seed in configurations:
            stem = f"mustang_{regime}_{scenario}" + (f"_seed{seed}" if seed is not None else "")
            replay, manifest = resample(pool, users, regime, scenario, weight, seed)
            manifest.insert(0, "workload", stem)
            manifest["window_start"] = t0.isoformat()
            manifest["window_end"] = t1.isoformat()
            manifest["job_node_hours_quantile"] = job_node_hours_quantile
            manifest["min_jobs_for_heavy"] = min_jobs_for_heavy
            manifest["heavy_user_threshold_quantile"] = heavy_user_threshold_quantile
            validate_resampling(replay, pool, manifest, users)
            workload = build_workload(replay, context, t0)
            validate_workload(workload, t1 - t0, context, len(replay))
            if scenario == "original":
                require(workload == reference, "Generated original must exactly match the existing JSON")
            csv_path, json_path = output_dir / f"{stem}.csv", output_dir / f"{stem}.json"
            replay[CSV_COLUMNS].to_csv(csv_path, index=False)
            json_path.write_text(json.dumps(workload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            reloaded = validate_csv(csv_path, replay)
            require(build_workload(reloaded, context, t0) == workload, "CSV round trip changed the Batsim workload")
            require(json.loads(json_path.read_text()) == workload, "JSON round trip changed the workload")
            stats = summarize(replay, manifest, users, t1 - t0)
            stats["first_replay_subtime"] = float((replay["submit_utc"].min() - t0).total_seconds())
            summaries.append({
                "workload": stem, "regime": regime, "scenario": scenario, "seed": seed,
                "heavy_weight": weight,
                "job_node_hours_quantile": job_node_hours_quantile,
                "job_node_hours_threshold": float(users["job_node_hours_threshold"].iloc[0]),
                "min_jobs_for_heavy": min_jobs_for_heavy,
                "heavy_user_threshold_quantile": heavy_user_threshold_quantile,
                "heavy_threshold_high_node_hours_job_share": float(
                    users["heavy_threshold_high_node_hours_job_share"].iloc[0]
                ),
                "window_start": t0.isoformat(), "window_end": t1.isoformat(), **stats,
                "context_jobs": len(context["jobs"]),
                "csv_file": csv_path.name, "json_file": json_path.name,
                "csv_sha256": sha256_file(csv_path), "json_sha256": sha256_file(json_path),
                "validation_passed": True,
            })
            manifests.append(manifest)
            frames[stem] = replay
        print(f"{regime}: {len(pool):,} original jobs, {len(users)} users "
              f"({int(users['is_heavy'].sum())} heavy); {len(configurations)} CSV/JSON pairs validated")
    summary = pd.DataFrame(summaries)
    summary["seed"] = pd.array(summary["seed"], dtype="Int64")
    summary["hourly_submission_pearson_r"] = np.nan
    for regime, bounds in regimes.items():
        original_name = f"mustang_{regime}_original"
        original = frames[original_name]
        regime_rows = summary["regime"].eq(regime)
        summary.loc[regime_rows, "hourly_submission_pearson_r"] = [
            hourly_submission_pearson_r(frames[workload], original, bounds)
            for workload in summary.loc[regime_rows, "workload"]
        ]
    require(summary["hourly_submission_pearson_r"].notna().all(),
            "Every workload needs an hourly submission correlation")
    manifest = pd.concat(manifests, ignore_index=True)
    user_profiles = pd.concat(profiles, ignore_index=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    user_profiles.to_csv(output_dir / "user_profiles.csv", index=False)
    metadata = {
        "method": "static user-level weighted bootstrap within fixed windows; original context held fixed",
        "regimes": {k: [v.isoformat() for v in utc_window(b)] for k, b in regimes.items()},
        "heavy_rule": (
            "job node-hours > source job-level linear empirical quantile; users with at least "
            "min_jobs_for_heavy are ranked by their high-node-hours-job share; user-threshold ties included"
        ),
        "job_node_hours_quantile": job_node_hours_quantile,
        "min_jobs_for_heavy": min_jobs_for_heavy,
        "heavy_user_threshold_quantile": heavy_user_threshold_quantile,
        "scenario_weights": scenario_weights, "seeds": [int(s) for s in seeds],
        "rng": "PCG64(SeedSequence([seed, big-endian uint32 SHA256(regime/scenario)[:4]]))",
        "num_nodes": NUM_NODES, "node_speed_flops_per_second": NODE_SPEED,
        "wide_min_nodes": WIDE_NODES, "long_runtime_gt_seconds": LONG_SECONDS,
        "source": {"path": str(source_path), "sha256": sha256_file(source_path)} if source_path else None,
        "source_row_numbering": "1-based data row, excluding CSV header",
        "baseline_sha256": {k: sha256_file(workload_dir / f"mustang_{k}.json") for k in regimes},
        "implementation_sha256": sha256_file(__file__),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "hygiene": hygiene,
        "artifacts": {name: sha256_file(output_dir / name)
                      for name in ["summary.csv", "manifest.csv", "user_profiles.csv"]},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary, manifest, user_profiles, frames


def verify_artifacts(output_dir, workload_dir):
    """Independently reload CSVs, JSONs and provenance after generation."""
    output_dir, workload_dir = Path(output_dir), Path(workload_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text())
    classification_keys = [
        "job_node_hours_quantile", "min_jobs_for_heavy", "heavy_user_threshold_quantile",
    ]
    require(all(key in metadata for key in classification_keys),
            "Classification provenance is incomplete")
    for name, digest in metadata["artifacts"].items():
        require(sha256_file(output_dir / name) == digest, f"Collection checksum differs: {name}")
    summary = pd.read_csv(output_dir / "summary.csv", dtype={"seed": "Int64"})
    manifest = pd.read_csv(output_dir / "manifest.csv", dtype={"seed": "Int64", "rng_stream_id": "Int64"})
    require(summary["workload"].is_unique, "Duplicate summary workloads")
    require("hourly_submission_pearson_r" in summary.columns
            and summary["hourly_submission_pearson_r"].notna().all(),
            "Summary is missing hourly submission correlations")
    expected_count = len(metadata["regimes"]) * (1 + len(metadata["scenario_weights"]) * len(metadata["seeds"]))
    require(len(summary) == expected_count, "Incomplete workload collection")
    require(set(summary["workload"]) == set(manifest["workload"]), "Manifest/summary workloads differ")
    references, original_replays = {}, {}
    for regime, digest in metadata["baseline_sha256"].items():
        path = workload_dir / f"mustang_{regime}.json"
        require(sha256_file(path) == digest, f"Baseline changed since generation: {regime}")
        references[regime] = json.loads(path.read_text())
        original_row = summary.loc[
            summary["regime"].eq(regime) & summary["scenario"].eq("original")
        ]
        require(len(original_row) == 1, f"Missing unique original summary row: {regime}")
        original_csv = output_dir / original_row.iloc[0]["csv_file"]
        original_replays[regime], original_audit = prepare_jobs(
            pd.read_csv(original_csv, dtype=str, keep_default_na=False)
        )
        require(original_audit["excluded_rows"] == 0,
                f"Original CSV contains invalid jobs: {regime}")
    for row in summary.itertuples(index=False):
        csv_path, json_path = output_dir / row.csv_file, output_dir / row.json_file
        require(sha256_file(csv_path) == row.csv_sha256, f"CSV checksum differs: {row.workload}")
        require(sha256_file(json_path) == row.json_sha256, f"JSON checksum differs: {row.workload}")
        replay, audit = prepare_jobs(pd.read_csv(csv_path, dtype=str, keep_default_na=False))
        require(audit["excluded_rows"] == 0, "Exported trace has invalid rows")
        t0, t1 = utc_window((row.window_start, row.window_end))
        require(replay["submit_utc"].ge(t0).all() and replay["submit_utc"].lt(t1).all(), "CSV includes jobs outside replay window")
        correlation = hourly_submission_pearson_r(
            replay, original_replays[row.regime], (row.window_start, row.window_end)
        )
        require(math.isclose(correlation, row.hourly_submission_pearson_r,
                             rel_tol=1e-12, abs_tol=1e-12),
                "Hourly submission correlation differs")
        context = get_context(references[row.regime])
        workload = json.loads(json_path.read_text())
        validate_workload(workload, t1 - t0, context, len(replay))
        require(build_workload(replay, context, t0) == workload, "Persisted CSV/JSON disagree")
        if row.scenario == "original":
            require(workload == references[row.regime], "Original baseline differs")
        draws = manifest.loc[manifest["workload"].eq(row.workload)].set_index("synthetic_user_ID")
        require(draws.index.is_unique and len(draws) == row.users, "Invalid persisted user instances")
        for key in classification_keys:
            require(key in draws and draws[key].notna().all(),
                    f"Missing persisted classification setting: {key}")
            require(np.allclose(draws[key].to_numpy(dtype=float), metadata[key]),
                    f"Manifest classification setting differs: {key}")
            require(math.isclose(float(getattr(row, key)), float(metadata[key]), rel_tol=1e-12),
                    f"Summary classification setting differs: {key}")
        pd.testing.assert_series_equal(replay.groupby("user_ID").size().sort_index(),
                                       draws["jobs"].sort_index(), check_names=False, check_dtype=False)
        np.testing.assert_allclose(replay.groupby("user_ID")["node_hours"].sum().sort_index(),
                                   draws["total_node_hours"].sort_index(), rtol=1e-12)
        require(len(replay) == row.jobs, "Summary job count differs")
        require(math.isclose(replay["node_hours"].sum(), row.total_node_hours, rel_tol=1e-12), "Summary node-hours differ")
    return {"workloads_verified": len(summary), "synthetics": int(summary["scenario"].ne("original").sum()),
            "baselines": int(summary["scenario"].eq("original").sum()), "user_draws_verified": len(manifest)}


def _selected_and_original_rows(summary, workload_id):
    """Return one generated workload and the original row for its regime."""
    require(isinstance(workload_id, str) and workload_id,
            "Choose a nonempty generated workload ID")
    require(Path(workload_id).name == workload_id,
            "Workload ID must be a plain filename stem")
    selected = summary.loc[summary["workload"].eq(workload_id)]
    require(len(selected) == 1, f"Unknown or duplicate workload ID: {workload_id}")
    row = selected.iloc[0]
    require(row["scenario"] != "original",
            "Choose a generated workload, not the original baseline")
    original = summary.loc[
        summary["regime"].eq(row["regime"]) & summary["scenario"].eq("original")
    ]
    require(len(original) == 1, f"Missing unique original workload: {row['regime']}")
    return row, original.iloc[0]


def plot_static_workload_comparison(summary, frames, workload_id):
    """Compare one generated workload with its original regime in one figure."""
    required = {
        "workload", "regime", "scenario", "seed", "window_start", "window_end",
        "jobs", "total_node_hours", "wide_job_share", "long_job_share",
        "hourly_submission_pearson_r",
    }
    require(required.issubset(summary.columns),
            "Summary lacks static comparison fields")
    row, original_row = _selected_and_original_rows(summary, workload_id)
    require(row["workload"] in frames and original_row["workload"] in frames,
            "Comparison replay frames are incomplete")

    candidate = frames[row["workload"]]
    original = frames[original_row["workload"]]
    frame_columns = {
        "runtime_sec", "node_count", "submit_utc", "user_ID",
        "source_user_ID", "node_hours",
    }
    require(frame_columns.issubset(candidate.columns),
            "Generated replay lacks comparison columns")
    require(frame_columns.issubset(original.columns),
            "Original replay lacks comparison columns")

    configure_plot_style()
    figure = plt.figure(figsize=(13.5, 15), constrained_layout=True)
    grid = figure.add_gridspec(4, 2, height_ratios=[1, 0.78, 1, 1.12])
    runtime_axis = figure.add_subplot(grid[0, 0])
    node_axis = figure.add_subplot(grid[0, 1])
    daily_axis = figure.add_subplot(grid[1, 0])
    job_share_axis = figure.add_subplot(grid[1, 1])
    instance_axis = figure.add_subplot(grid[2, 0])
    source_axis = figure.add_subplot(grid[2, 1])
    hourly_axis = figure.add_subplot(grid[3, :])

    scenario_label = SCENARIO_LABELS.get(
        row["scenario"], row["scenario"].replace("_", " ").title()
    )
    seed_label = "" if pd.isna(row["seed"]) else f", seed {int(row['seed'])}"
    candidate_label = f"{scenario_label}{seed_label}"
    candidate_color = SCENARIO_COLORS.get(row["scenario"], "#4C78A8")
    series = [
        (original, "Original Mustang", "black", 2.2),
        (candidate, candidate_label, candidate_color, 1.7),
    ]

    runtime_max = max(frame["runtime_sec"].max() for frame, *_ in series) / 3600
    runtime_grid = np.geomspace(1 / 3600, runtime_max, 400)
    node_max = max(frame["node_count"].max() for frame, *_ in series)
    node_grid = np.geomspace(1, node_max, 400)
    for frame, label, color, width in series:
        runtime_axis.plot(
            runtime_grid, 100 * _ecdf(frame["runtime_sec"] / 3600, runtime_grid),
            label=label, color=color, linewidth=width,
        )
        node_axis.plot(
            node_grid, 100 * _ecdf(frame["node_count"], node_grid),
            label=label, color=color, linewidth=width,
        )
    runtime_axis.set(xscale="log", xlabel="Actual runtime [h]",
                     ylabel="Cumulative job share [%]", title="Runtime distribution")
    node_axis.set(xscale="log", xlabel="Requested nodes",
                  ylabel="Cumulative job share [%]", title="Node-count distribution")

    t0, t1 = utc_window((row["window_start"], row["window_end"]))
    daily_index = pd.date_range(t0, t1, freq="D", inclusive="left")
    hourly_index = pd.date_range(t0, t1, freq="h", inclusive="left")
    days = np.arange(1, len(daily_index) + 1)
    for frame, label, color, width in series:
        daily = (
            frame.set_index("submit_utc").resample("D").size()
            .reindex(daily_index, fill_value=0)
        )
        hourly = hourly_submission_counts(
            frame, (row["window_start"], row["window_end"])
        ).reindex(hourly_index, fill_value=0)
        daily_axis.plot(days, daily, label=label, color=color, linewidth=width)
        hourly_axis.plot(
            hourly.index, hourly.to_numpy(), label=label,
            color=color, linewidth=width, alpha=0.88,
        )
    daily_axis.set(xlabel="Days since T0", ylabel="Jobs/day",
                   title="Daily submissions")
    daily_axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    job_share_categories = ["Long\n(> 2 h)", "Wide\n(>= 160 nodes)"]
    original_job_shares = 100 * np.array([
        original_row["long_job_share"], original_row["wide_job_share"],
    ], dtype=float)
    candidate_job_shares = 100 * np.array([
        row["long_job_share"], row["wide_job_share"],
    ], dtype=float)
    job_share_deltas = candidate_job_shares - original_job_shares
    positions = np.arange(len(job_share_categories))
    bar_width = 0.36
    original_bars = job_share_axis.bar(
        positions - bar_width / 2, original_job_shares, bar_width,
        label="Original Mustang", color="#555555",
    )
    candidate_bars = job_share_axis.bar(
        positions + bar_width / 2, candidate_job_shares, bar_width,
        label=candidate_label, color=candidate_color,
    )
    for bar, value in zip(original_bars, original_job_shares, strict=True):
        job_share_axis.annotate(
            f"{value:.2f}%", (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=9,
        )
    for bar, value, delta in zip(
        candidate_bars, candidate_job_shares, job_share_deltas, strict=True
    ):
        job_share_axis.annotate(
            f"{value:.2f}%\n({delta:+.2f} pp)",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
            fontsize=9,
        )
    job_share_axis.set(
        xticks=positions, xticklabels=job_share_categories,
        ylabel="Replay jobs [%]", title="Long/wide job-share comparison",
    )
    job_share_axis.set_ylim(
        0, min(110, max(10, 1.35 * candidate_job_shares.max(),
                        1.25 * original_job_shares.max()))
    )
    job_share_axis.legend(loc="upper left")

    hourly_axis.set(xlabel="Submission hour", ylabel="Jobs/hour",
                    title=("Hourly submissions "
                           f"(Pearson r = {float(row['hourly_submission_pearson_r']):.4f})"))
    hourly_axis.tick_params(axis="x", labelrotation=25)

    fraction_grid = np.linspace(0, 100, 400)
    for frame, label, color, width in series:
        instance_axis.plot(
            fraction_grid,
            100 * _lorenz(
                frame.groupby("user_ID")["node_hours"].sum(), fraction_grid / 100
            ),
            label=label, color=color, linewidth=width,
        )
        source_axis.plot(
            fraction_grid,
            100 * _lorenz(
                frame.groupby("source_user_ID")["node_hours"].sum(),
                fraction_grid / 100,
            ),
            label=label, color=color, linewidth=width,
        )
    for axis, title, xlabel in [
        (instance_axis, "Synthetic-instance concentration", "Cumulative instance share [%]"),
        (source_axis, "Source-user concentration", "Cumulative source share [%]"),
    ]:
        axis.plot([0, 100], [0, 100], color="#999", linestyle="--", linewidth=1)
        axis.set(xlabel=xlabel, ylabel="Cumulative node-hour share [%]", title=title,
                 xlim=(0, 100), ylim=(0, 100))

    for axis in figure.axes:
        axis.grid(True, which="major", alpha=0.22)
    runtime_axis.legend(loc="lower right")
    hourly_axis.legend(loc="upper right")
    figure.suptitle(
        f"{row['workload']} compared with {original_row['workload']}\n"
        f"{int(row['jobs']):,} jobs | {float(row['total_node_hours']):,.0f} node-hours | "
        f"long Δ {job_share_deltas[0]:+.2f} pp | "
        f"wide Δ {job_share_deltas[1]:+.2f} pp",
        fontsize=16,
    )
    return figure


def export_selected_workload(summary, manifest, artifact_dir, selection_root, workload_id):
    """Copy one verified generated workload, provenance, and comparison figure.

    The export is deterministic and deliberately does not edit an experiment
    campaign. Re-exporting the same workload replaces only its own five files.
    """
    row, original_row = _selected_and_original_rows(summary, workload_id)
    draws = manifest.loc[manifest["workload"].eq(workload_id)].copy()
    require(len(draws) == int(row["users"]), "Selected workload provenance is incomplete")

    artifact_dir = Path(artifact_dir).resolve()
    selection_root = Path(selection_root).resolve()

    def source_file(name, expected_digest):
        require(Path(name).name == name, "Artifact filename must not contain a directory")
        path = artifact_dir / name
        require(path.is_file(), f"Missing generated artifact: {name}")
        require(sha256_file(path) == expected_digest, f"Artifact checksum differs: {name}")
        return path

    csv_source = source_file(row["csv_file"], row["csv_sha256"])
    json_source = source_file(row["json_file"], row["json_sha256"])
    destination = selection_root / workload_id
    destination.mkdir(parents=True, exist_ok=True)
    csv_target = destination / csv_source.name
    json_target = destination / json_source.name
    draw_target = destination / f"{workload_id}_user_manifest.csv"
    comparison_target = destination / "comparison.png"
    selection_target = destination / "selection.json"
    shutil.copy2(csv_source, csv_target)
    shutil.copy2(json_source, json_target)
    draws.to_csv(draw_target, index=False)
    require(sha256_file(csv_target) == row["csv_sha256"], "Copied CSV checksum differs")
    require(sha256_file(json_target) == row["json_sha256"], "Copied JSON checksum differs")

    original_csv_source = source_file(
        original_row["csv_file"], original_row["csv_sha256"]
    )

    def comparison_frame(csv_path, workload):
        replay, audit = prepare_jobs(pd.read_csv(
            csv_path, dtype=str, keep_default_na=False
        ))
        require(audit["excluded_rows"] == 0,
                f"Comparison CSV contains invalid jobs: {workload}")
        workload_draws = manifest.loc[manifest["workload"].eq(workload)]
        require(not workload_draws.empty, f"Missing comparison provenance: {workload}")
        source_by_instance = workload_draws.set_index("synthetic_user_ID")["source_user_ID"]
        replay["source_user_ID"] = replay["user_ID"].map(source_by_instance)
        require(replay["source_user_ID"].notna().all(),
                f"Incomplete source-user mapping: {workload}")
        return replay

    comparison_frames = {
        row["workload"]: comparison_frame(csv_source, row["workload"]),
        original_row["workload"]: comparison_frame(
            original_csv_source, original_row["workload"]
        ),
    }
    comparison_figure = plot_static_workload_comparison(
        summary, comparison_frames, workload_id
    )
    comparison_figure.savefig(comparison_target, dpi=180, bbox_inches="tight")
    plt.close(comparison_figure)
    require(comparison_target.is_file() and comparison_target.stat().st_size > 0,
            "Comparison PNG was not written")

    seed = None if pd.isna(row["seed"]) else int(row["seed"])
    selection = {
        "workload": workload_id,
        "regime": row["regime"],
        "scenario": row["scenario"],
        "seed": seed,
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "heavy_weight": float(row["heavy_weight"]),
        "classification": {
            "job_node_hours_quantile": float(row["job_node_hours_quantile"]),
            "job_node_hours_threshold": float(row["job_node_hours_threshold"]),
            "min_jobs_for_heavy": int(row["min_jobs_for_heavy"]),
            "heavy_user_threshold_quantile": float(row["heavy_user_threshold_quantile"]),
            "heavy_threshold_high_node_hours_job_share": float(
                row["heavy_threshold_high_node_hours_job_share"]
            ),
        },
        "diagnostics": {
            "jobs": int(row["jobs"]),
            "users": int(row["users"]),
            "total_node_hours": float(row["total_node_hours"]),
            "wide_job_share": float(row["wide_job_share"]),
            "long_job_share": float(row["long_job_share"]),
            "hourly_submission_pearson_r": float(row["hourly_submission_pearson_r"]),
        },
        "files": {
            "csv": {"name": csv_target.name, "sha256": sha256_file(csv_target)},
            "json": {"name": json_target.name, "sha256": sha256_file(json_target)},
            "user_manifest": {
                "name": draw_target.name, "sha256": sha256_file(draw_target),
            },
            "comparison_plot": {
                "name": comparison_target.name,
                "sha256": sha256_file(comparison_target),
            },
        },
        "campaign_updated": False,
    }
    selection_target.write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "workload": workload_id,
        "directory": destination,
        "csv": csv_target,
        "json": json_target,
        "user_manifest": draw_target,
        "comparison_plot": comparison_target,
        "selection_manifest": selection_target,
    }


def export_selected_workload_pair(summary, manifest, artifact_dir, selection_root,
                                  workload_ids):
    """Export exactly one generated Slack workload and one Stress workload."""
    expected_regimes = {"slack", "stress"}
    require(isinstance(workload_ids, dict),
            "Selected workload IDs must be a Slack/Stress dictionary")
    require(set(workload_ids) == expected_regimes,
            "Provide exactly one workload ID for Slack and one for Stress")
    require(len(set(workload_ids.values())) == len(expected_regimes),
            "Slack and Stress workload IDs must be different")

    # Validate the complete pair before creating either export directory.
    for regime in ("slack", "stress"):
        row, _ = _selected_and_original_rows(summary, workload_ids[regime])
        require(row["regime"] == regime,
                f"Selected {regime} ID belongs to the {row['regime']} regime")

    return {
        regime: export_selected_workload(
            summary=summary,
            manifest=manifest,
            artifact_dir=artifact_dir,
            selection_root=selection_root,
            workload_id=workload_ids[regime],
        )
        for regime in ("slack", "stress")
    }


def _ordered_scenarios(group):
    present = set(group["scenario"])
    return [scenario for scenario in SCENARIO_ORDER if scenario in present]


def _scenario_frames(group, frames, scenario):
    return [frames[row.workload] for row in group.itertuples(index=False) if row.scenario == scenario]


def plot_interactive_hourly_submissions(summary, frames):
    """Compare one workload at a time with its original hourly submission series."""
    import plotly.graph_objects as go

    required = {
        "workload", "regime", "scenario", "seed", "window_start", "window_end",
        "hourly_submission_pearson_r",
    }
    require(required.issubset(summary.columns), "Summary lacks interactive hourly fields")
    require(set(summary["workload"]) == set(frames), "Summary and replay frames differ")

    figure = go.Figure()
    trace_index, original_index = {}, {}
    original_rows = summary.loc[summary["scenario"].eq("original")]
    require(original_rows["regime"].is_unique, "Each regime needs one original workload")

    def add_trace(row, *, original):
        counts = hourly_submission_counts(
            frames[row.workload], (row.window_start, row.window_end)
        )
        scenario_label = SCENARIO_LABELS.get(
            row.scenario, row.scenario.replace("_", " ").title()
        )
        seed_label = "baseline" if pd.isna(row.seed) else f"seed {int(row.seed)}"
        name = "Original Mustang" if original else f"{scenario_label}, {seed_label}"
        color = "black" if original else SCENARIO_COLORS.get(row.scenario, "#4C78A8")
        figure.add_trace(go.Scattergl(
            x=counts.index,
            y=counts.to_numpy(),
            mode="lines",
            visible=False,
            name=name,
            line={"color": color, "width": 2.0 if original else 1.25},
            opacity=1.0 if original else 0.82,
            hovertemplate=(
                f"<b>{row.workload}</b><br>"
                "%{x|%Y-%m-%d %H:%M UTC}<br>"
                "%{y} jobs/hour<br>"
                f"Pearson r = {row.hourly_submission_pearson_r:.4f}"
                "<extra></extra>"
            ),
        ))
        trace_index[row.workload] = len(figure.data) - 1

    for row in original_rows.itertuples(index=False):
        add_trace(row, original=True)
        original_index[row.regime] = trace_index[row.workload]
    for row in summary.loc[summary["scenario"].ne("original")].itertuples(index=False):
        add_trace(row, original=False)

    buttons = []
    for row in summary.itertuples(index=False):
        visible = [False] * len(figure.data)
        visible[original_index[row.regime]] = True
        visible[trace_index[row.workload]] = True
        scenario_label = SCENARIO_LABELS.get(
            row.scenario, row.scenario.replace("_", " ").title()
        )
        seed_label = "baseline" if pd.isna(row.seed) else f"seed {int(row.seed)}"
        label = row.workload
        title = (
            f"{row.regime.title()}: {scenario_label}, {seed_label} "
            f"(Pearson r = {row.hourly_submission_pearson_r:.4f})"
        )
        buttons.append({
            "label": label,
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title": {"text": title, "x": 0.5},
                    "xaxis": {
                        "title": {"text": "Submission hour"},
                        "rangeslider": {"visible": True, "thickness": 0.10},
                        "autorange": True,
                    },
                },
            ],
        })

    candidates = summary.loc[summary["scenario"].ne("original")]
    default_row = candidates.iloc[0] if not candidates.empty else summary.iloc[0]
    figure.data[original_index[default_row["regime"]]].visible = True
    figure.data[trace_index[default_row["workload"]]].visible = True
    default_scenario = SCENARIO_LABELS.get(
        default_row["scenario"], default_row["scenario"].replace("_", " ").title()
    )
    default_seed = (
        "baseline" if pd.isna(default_row["seed"])
        else f"seed {int(default_row['seed'])}"
    )
    active_button = next(
        position for position, workload in enumerate(summary["workload"])
        if workload == default_row["workload"]
    )
    figure.update_layout(
        template="plotly_white",
        height=520,
        title={
            "text": (
                f"{default_row['regime'].title()}: {default_scenario}, {default_seed} "
                f"(Pearson r = {default_row['hourly_submission_pearson_r']:.4f})"
            ),
            "x": 0.5,
        },
        xaxis={
            "title": "Submission hour",
            "rangeslider": {"visible": True, "thickness": 0.10},
        },
        yaxis={"title": "Jobs/hour", "rangemode": "tozero"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        updatemenus=[{
            "buttons": buttons,
            "active": int(active_button),
            "direction": "down",
            "showactive": True,
            "x": 0,
            "xanchor": "left",
            "y": 1.16,
            "yanchor": "top",
        }],
        margin={"l": 70, "r": 25, "b": 70, "t": 115},
    )
    return figure


def plot_interactive_trace_overview(summary):
    """Place every trace in one meaningful long-vs-wide workload map."""
    import plotly.graph_objects as go

    required = {
        "workload", "regime", "scenario", "seed", "jobs", "total_node_hours",
        "heavy_user_share", "wide_job_share", "long_job_share",
        "top_source_user_node_hours_share", "hourly_submission_pearson_r",
    }
    require(required.issubset(summary.columns), "Summary lacks trace-overview fields")
    require(summary["workload"].is_unique, "Workload IDs must be unique")

    figure = go.Figure()
    indices_by_regime = {}
    for regime, group in summary.groupby("regime", sort=False):
        indices_by_regime[regime] = []
        maximum_work = float(group["total_node_hours"].max())
        for scenario in _ordered_scenarios(group):
            selected = group.loc[group["scenario"].eq(scenario)]
            scenario_label = SCENARIO_LABELS.get(
                scenario, scenario.replace("_", " ").title()
            )
            seed_labels = [
                "baseline" if pd.isna(seed) else str(int(seed))
                for seed in selected["seed"]
            ]
            marker_sizes = 11 + 22 * np.sqrt(
                selected["total_node_hours"].to_numpy(dtype=float) / maximum_work
            )
            customdata = [
                [
                    row.workload, scenario_label, seed_label, int(row.jobs),
                    float(row.total_node_hours), 100 * float(row.heavy_user_share),
                    100 * float(row.top_source_user_node_hours_share),
                    float(row.hourly_submission_pearson_r),
                ]
                for row, seed_label in zip(
                    selected.itertuples(index=False), seed_labels, strict=True
                )
            ]
            figure.add_trace(go.Scatter(
                x=100 * selected["long_job_share"],
                y=100 * selected["wide_job_share"],
                mode="markers", name=scenario_label,
                legendgroup=scenario, visible=False,
                marker={
                    "color": SCENARIO_COLORS.get(scenario, "#4C78A8"),
                    "symbol": "diamond" if scenario == "original" else "circle",
                    "size": marker_sizes,
                    "opacity": 0.88,
                    "line": {"color": "white", "width": 1.0},
                },
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Scenario: %{customdata[1]}<br>"
                    "Seed: %{customdata[2]}<br>"
                    "Long jobs: %{x:.2f}%<br>"
                    "Wide jobs: %{y:.2f}%<br>"
                    "Replay jobs: %{customdata[3]:,.0f}<br>"
                    "Node-hours: %{customdata[4]:,.0f}<br>"
                    "Heavy draws: %{customdata[5]:.2f}%<br>"
                    "Largest source-user share: %{customdata[6]:.2f}%<br>"
                    "Hourly Pearson r: %{customdata[7]:.4f}"
                    "<extra></extra>"
                ),
            ))
            indices_by_regime[regime].append(len(figure.data) - 1)

    regimes = list(indices_by_regime)
    default_regime = regimes[0]
    for index in indices_by_regime[default_regime]:
        figure.data[index].visible = True
    buttons = []
    for regime in regimes:
        visible = [False] * len(figure.data)
        for index in indices_by_regime[regime]:
            visible[index] = True
        buttons.append({
            "label": regime.title(),
            "method": "update",
            "args": [
                {"visible": visible},
                {
                    "title.text": f"{regime.title()}: generated trace overview",
                    "xaxis.autorange": True,
                    "yaxis.autorange": True,
                },
            ],
        })
    figure.update_layout(
        template="plotly_white", height=600,
        title={"text": f"{default_regime.title()}: generated trace overview", "x": 0.5},
        xaxis={"title": "Long jobs (>2 h) [%]", "rangemode": "tozero"},
        yaxis={"title": "Wide jobs (>=160 nodes) [%]", "rangemode": "tozero"},
        hovermode="closest",
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        updatemenus=[{
            "buttons": buttons, "active": 0, "direction": "down", "showactive": True,
            "x": 0, "xanchor": "left", "y": 1.14, "yanchor": "top",
        }],
        margin={"l": 75, "r": 30, "b": 70, "t": 115},
    )
    figure.add_annotation(
        text="Bubble area increases with submitted node-hours; hover for the exact workload ID and metrics",
        x=0.5, y=-0.14, xref="paper", yref="paper", showarrow=False,
        font={"color": "#555", "size": 11},
    )
    return figure


def plot_interactive_trace_distributions(summary, frames):
    """Build one synchronized distribution selector for all generated traces."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    required = {
        "workload", "regime", "scenario", "seed", "window_start", "window_end",
    }
    require(required.issubset(summary.columns), "Summary lacks distribution-dashboard fields")
    require(set(summary["workload"]) == set(frames), "Summary and replay frames differ")

    summary = summary.reset_index(drop=True)
    all_frames = list(frames.values())
    runtime_grid = np.geomspace(
        1 / 3600,
        max(frame["runtime_sec"].max() for frame in all_frames) / 3600,
        400,
    )
    node_grid = np.geomspace(
        1, max(frame["node_count"].max() for frame in all_frames), 400
    )
    fraction_grid = np.linspace(0, 100, 400)
    daily_x = np.arange(1, 29)
    daily_indices = {
        regime: pd.date_range(*utc_window((group["window_start"].iloc[0], group["window_end"].iloc[0])),
                              freq="D", inclusive="left")
        for regime, group in summary.groupby("regime", sort=False)
    }
    figure = make_subplots(
        rows=2, cols=6,
        specs=[
            [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
            [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
        ],
        subplot_titles=[
            "Runtime distribution", "Node-count distribution",
            "Daily submissions", "Instance concentration", "Source concentration",
        ],
        horizontal_spacing=0.08, vertical_spacing=0.20,
    )
    panels = [
        (1, 1, "Actual runtime [h]", "Cumulative job share [%]"),
        (1, 4, "Requested nodes", "Cumulative job share [%]"),
        (2, 1, "Days since T0", "Jobs/day"),
        (2, 3, "Cumulative instance share [%]", "Node-hour share [%]"),
        (2, 5, "Cumulative source share [%]", "Node-hour share [%]"),
    ]
    trace_indices = {}
    for workload_row in summary.itertuples(index=False):
            replay = frames[workload_row.workload]
            daily = (
                replay.set_index("submit_utc").resample("D").size()
                .reindex(daily_indices[workload_row.regime], fill_value=0).to_numpy()
            )
            panel_data = [
                (
                    runtime_grid,
                    100 * _ecdf(replay["runtime_sec"] / 3600, runtime_grid),
                    "Runtime: %{x:.3g} h<br>Cumulative jobs: %{y:.2f}%",
                ),
                (
                    node_grid,
                    100 * _ecdf(replay["node_count"], node_grid),
                    "Nodes: %{x:.3g}<br>Cumulative jobs: %{y:.2f}%",
                ),
                (
                    daily_x, daily,
                    "Day %{x:.0f}<br>Jobs/day: %{y:,.0f}",
                ),
                (
                    fraction_grid,
                    100 * _lorenz(
                        replay.groupby("user_ID")["node_hours"].sum(),
                        fraction_grid / 100,
                    ),
                    "Instances: %{x:.1f}%<br>Node-hours: %{y:.1f}%",
                ),
                (
                    fraction_grid,
                    100 * _lorenz(
                        replay.groupby("source_user_ID")["node_hours"].sum(),
                        fraction_grid / 100,
                    ),
                    "Sources: %{x:.1f}%<br>Node-hours: %{y:.1f}%",
                ),
            ]
            scenario_label = SCENARIO_LABELS.get(
                workload_row.scenario,
                workload_row.scenario.replace("_", " ").title(),
            )
            seed_label = (
                "baseline" if pd.isna(workload_row.seed)
                else f"seed {int(workload_row.seed)}"
            )
            name = (
                "Original Mustang" if workload_row.scenario == "original"
                else f"{scenario_label}, {seed_label}"
            )
            indices = []
            for panel_index, ((row, col, _, _), (x, y, hover)) in enumerate(
                zip(panels, panel_data, strict=True)
            ):
                figure.add_trace(
                    go.Scattergl(
                        x=x, y=y, mode="lines", visible=False,
                        name=name, legendgroup=workload_row.workload,
                        showlegend=panel_index == 0,
                        line={
                            "color": SCENARIO_COLORS.get(workload_row.scenario, "#4C78A8"),
                            "width": 2.2 if workload_row.scenario == "original" else 1.5,
                        },
                        hovertemplate=(
                            f"<b>{workload_row.workload}</b><br>{hover}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )
                indices.append(len(figure.data) - 1)
            trace_indices[workload_row.workload] = indices

    original_workloads = {}
    for regime, group in summary.groupby("regime", sort=False):
        original_row = group.loc[group["scenario"].eq("original")]
        require(len(original_row) == 1, f"{regime} needs one original workload")
        original_workloads[regime] = original_row.iloc[0]["workload"]
    buttons = []
    for workload_row in summary.itertuples(index=False):
        visible = [False] * len(figure.data)
        original_workload = original_workloads[workload_row.regime]
        for index in trace_indices[original_workload]:
            visible[index] = True
        for index in trace_indices[workload_row.workload]:
            visible[index] = True
        scenario_label = SCENARIO_LABELS.get(
            workload_row.scenario,
            workload_row.scenario.replace("_", " ").title(),
        )
        seed_label = (
            "baseline" if pd.isna(workload_row.seed)
            else f"seed {int(workload_row.seed)}"
        )
        buttons.append({
            "label": workload_row.workload,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title.text": (
                    f"{workload_row.regime.title()}: {scenario_label}, {seed_label} "
                    "compared with original Mustang"
                )},
            ],
        })

    candidates = summary.loc[summary["scenario"].ne("original")]
    default_row = candidates.iloc[0] if not candidates.empty else summary.iloc[0]
    default_original = original_workloads[default_row["regime"]]
    for index in trace_indices[default_original]:
        figure.data[index].visible = True
    for index in trace_indices[default_row["workload"]]:
        figure.data[index].visible = True
    active_button = next(
        position for position, workload in enumerate(summary["workload"])
        if workload == default_row["workload"]
    )
    default_scenario = SCENARIO_LABELS.get(
        default_row["scenario"], default_row["scenario"].replace("_", " ").title()
    )
    default_seed = (
        "baseline" if pd.isna(default_row["seed"])
        else f"seed {int(default_row['seed'])}"
    )
    for row, col, xlabel, ylabel in panels:
        figure.update_xaxes(title_text=xlabel, row=row, col=col)
        figure.update_yaxes(title_text=ylabel, row=row, col=col)
    figure.update_xaxes(type="log", row=1, col=1)
    figure.update_xaxes(type="log", row=1, col=4)
    for col in (3, 5):
        figure.add_shape(
            type="line", x0=0, y0=0, x1=100, y1=100,
            line={"color": "#999", "dash": "dash", "width": 1},
            row=2, col=col,
        )
    figure.update_layout(
        template="plotly_white", height=720,
        title={
            "text": (
                f"{default_row['regime'].title()}: {default_scenario}, {default_seed} "
                "compared with original Mustang"
            ),
            "x": 0.5,
        },
        hovermode="closest",
        legend={"orientation": "h", "y": 1.02, "x": 0.5, "xanchor": "center"},
        updatemenus=[{
            "buttons": buttons, "active": int(active_button),
            "direction": "down", "showactive": True,
            "x": 0, "xanchor": "left", "y": 1.15, "yanchor": "top",
        }],
        margin={"l": 70, "r": 25, "b": 70, "t": 120},
    )
    return figure


def _ecdf(values, grid):
    values = np.sort(np.asarray(values, dtype=float))
    return np.searchsorted(values, grid, side="right") / len(values)


def summarize_source_job_distributions(pools):
    """Summarize original replay-job shapes before any user resampling.

    The fixed 2-hour and 160-node references are descriptive aids used
    elsewhere in the notebook; they are not a heavy-user classification.
    """
    rows = []
    for regime, pool in pools.items():
        require(not pool.empty, f"{regime} source pool is empty")
        runtime_hours = pool["runtime_sec"] / 3600
        nodes = pool["node_count"]
        rows.append({
            "regime": regime,
            "jobs": len(pool),
            "runtime_p50_h": runtime_hours.quantile(0.50),
            "runtime_p90_h": runtime_hours.quantile(0.90),
            "runtime_p99_h": runtime_hours.quantile(0.99),
            "node_count_p50": nodes.quantile(0.50),
            "node_count_p90": nodes.quantile(0.90),
            "node_count_p99": nodes.quantile(0.99),
            "jobs_gt_2h_share": runtime_hours.gt(2).mean(),
            "jobs_ge_160_nodes_share": nodes.ge(160).mean(),
            "jobs_gt_2h_and_ge_160_nodes_share": runtime_hours.gt(2).mul(nodes.ge(160)).mean(),
        })
    return pd.DataFrame(rows)


def plot_source_job_distributions(pools, figure_dir=FIGURE_DIR):
    """Plot original job distributions, before classification or resampling.

    The top panels are marginal empirical CDFs. The bottom panels retain the
    runtime/node-count pairing for every job, which is useful when choosing
    separate or joint definitions of large and long jobs.
    """
    require(bool(pools), "At least one source pool is required")
    configure_plot_style()
    runtime_max_hours = max(pool["runtime_sec"].max() / 3600 for pool in pools.values())
    node_max = max(pool["node_count"].max() for pool in pools.values())
    runtime_grid_hours = np.geomspace(1 / 3600, runtime_max_hours, 500)
    node_grid = np.geomspace(1, node_max, 500)
    fig, axes = plt.subplots(2, 2, figsize=(sca.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 4.7))

    for regime, pool in pools.items():
        color = REGIME_COLORS.get(regime, None)
        label = regime.title()
        axes[0, 0].plot(
            runtime_grid_hours,
            _ecdf(pool["runtime_sec"] / 3600, runtime_grid_hours),
            color=color,
            linewidth=1.3,
            label=label,
        )
        axes[0, 1].plot(
            node_grid,
            _ecdf(pool["node_count"], node_grid),
            color=color,
            linewidth=1.3,
            label=label,
        )

    axes[0, 0].axvline(2, color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    axes[0, 1].axvline(WIDE_NODES, color="0.5", linestyle="--", linewidth=0.8, zorder=0)
    axes[0, 0].set(
        xscale="log", title="Runtime distribution", xlabel="Actual runtime [h]",
        ylabel="Cumulative job share",
    )
    axes[0, 1].set(
        xscale="log", title="Node-count distribution", xlabel="Requested nodes",
        ylabel="Cumulative job share",
    )
    for axis in axes[0]:
        axis.set_ylim(0, 1.02)

    for axis, (regime, pool) in zip(axes[1], pools.items(), strict=True):
        axis.scatter(
            pool["runtime_sec"] / 3600,
            pool["node_count"],
            color=REGIME_COLORS.get(regime, None),
            s=5,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
        )
        axis.axvline(2, color="0.5", linestyle="--", linewidth=0.8, zorder=0)
        axis.axhline(WIDE_NODES, color="0.5", linestyle="--", linewidth=0.8, zorder=0)
        axis.set(
            xscale="log", yscale="log", title=f"{regime.title()}: job shape",
            xlabel="Actual runtime [h]", ylabel="Requested nodes",
        )

    regime_handles = [
        Line2D([], [], color=REGIME_COLORS.get(regime, "black"), label=regime.title())
        for regime in pools
    ]
    reference_handles = [
        Line2D([], [], color="0.5", linestyle="--", linewidth=0.8, label="2 h / 160-node reference")
    ]
    fig.legend(
        handles=[*regime_handles, *reference_handles], loc="upper center",
        ncol=len(regime_handles) + 1, frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.13, top=0.84, wspace=0.35, hspace=0.56)
    export_figure(fig, f"{FIGURE_PREFIX}_source_job_distributions", figure_dir)
    return fig


def _calibration_axes(pools, *, sharex=False):
    """Create one compact calibration panel per regime."""
    require(bool(pools), "At least one source pool is required")
    configure_plot_style()
    fig, axes = plt.subplots(
        1,
        len(pools),
        figsize=(sca.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.35),
        sharex=sharex,
        sharey=True,
        squeeze=False,
    )
    return fig, axes.ravel()


def plot_job_threshold_decision(pools, figure_dir=FIGURE_DIR,
                                quantile_range=(0.40, 0.95),
                                reference_quantile=None):
    """Show how the candidate percentile changes selected job shapes."""
    require(bool(pools), "At least one source pool is required")
    configure_plot_style()
    quantiles = np.linspace(*quantile_range, 111)
    fig, axes = _calibration_axes(pools, sharex=True)
    shape_colors = ["#4C78A8", "#F58518", "#54A24B", "#BAB0AC"]
    shape_labels = ["Long only", "Wide only", "Long and wide", "Neither"]

    for axis, (regime, pool) in zip(axes, pools.items(), strict=True):
        require(not pool.empty, f"{regime} source pool is empty")
        cutoffs = pool["node_hours"].quantile(quantiles, interpolation="linear").to_numpy()
        long = pool["runtime_sec"].gt(LONG_SECONDS).to_numpy()
        wide = pool["node_count"].ge(WIDE_NODES).to_numpy()
        work = pool["node_hours"].to_numpy()
        composition = []
        for cutoff in cutoffs:
            selected = work > cutoff
            denominator = selected.sum()
            require(denominator > 0, "A candidate job percentile selected no jobs")
            composition.append([
                np.sum(selected & long & ~wide) / denominator,
                np.sum(selected & ~long & wide) / denominator,
                np.sum(selected & long & wide) / denominator,
                np.sum(selected & ~long & ~wide) / denominator,
            ])
        axis.stackplot(
            100 * quantiles, 100 * np.asarray(composition).T,
            colors=shape_colors, labels=shape_labels, alpha=0.9,
        )
        if reference_quantile is not None:
            axis.axvline(
                100 * reference_quantile, color="#E45756",
                linestyle="--", linewidth=0.9,
            )
        axis.set(
            title=regime.title(),
            xlabel="Candidate high-job percentile", ylabel="Jobs above cutoff [%]",
            xlim=(100 * quantile_range[0], 100 * quantile_range[1]), ylim=(0, 100),
        )
        axis.xaxis.set_major_formatter(PercentFormatter(100))

    handles = [
        Line2D([], [], color=color, linewidth=6, label=label)
        for color, label in zip(shape_colors, shape_labels, strict=True)
    ]
    if reference_quantile is not None:
        handles.append(
            Line2D([], [], color="#E45756", linestyle="--", linewidth=0.9,
                   label=f"P{100 * reference_quantile:.0f} selection")
        )
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.22, top=0.72, wspace=0.16)
    export_figure(fig, f"{FIGURE_PREFIX}_calibration_job_threshold_decision", figure_dir)
    return fig


def plot_minimum_jobs_decision(pools, figure_dir=FIGURE_DIR,
                               maximum=50, reference_minimum=None):
    """Show evidence-versus-coverage tradeoffs for minimum submitted jobs."""
    fig, axes = _calibration_axes(pools, sharex=True)
    for axis, (regime, pool) in zip(axes, pools.items(), strict=True):
        require(not pool.empty, f"{regime} source pool is empty")
        by_user = pool.groupby("source_user_ID").agg(
            jobs=("source_row", "size"), node_hours=("node_hours", "sum")
        )
        minima = np.arange(1, min(maximum, int(by_user["jobs"].max())) + 1)
        eligible_user_share = []
        covered_job_share = []
        covered_work_share = []
        for minimum in minima:
            eligible = by_user["jobs"].ge(minimum)
            eligible_user_share.append(eligible.mean())
            covered_job_share.append(by_user.loc[eligible, "jobs"].sum() / by_user["jobs"].sum())
            covered_work_share.append(
                by_user.loc[eligible, "node_hours"].sum() / by_user["node_hours"].sum()
            )
        axis.plot(minima, 100 * np.asarray(eligible_user_share), label="Eligible users", linewidth=1.4)
        axis.plot(minima, 100 * np.asarray(covered_job_share), label="Replay jobs covered", linewidth=1.4)
        axis.plot(minima, 100 * np.asarray(covered_work_share), label="Node-hours covered", linewidth=1.4)
        if reference_minimum is not None:
            axis.axvline(
                reference_minimum, color="0.35", linestyle="--", linewidth=0.9
            )
            axis.annotate(
                f"{reference_minimum}-job selection", (reference_minimum, 4),
                xytext=(4, 0), textcoords="offset points", rotation=90,
                va="bottom", ha="left", fontsize=7,
            )
        axis.set(
            title=regime.title(), xlabel="Candidate minimum jobs per user",
            xlim=(1, maximum), ylim=(0, 102),
        )
    axes[0].set_ylabel("Users or covered workload [%]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.22, top=0.72, wspace=0.16)
    export_figure(fig, f"{FIGURE_PREFIX}_calibration_minimum_jobs_decision", figure_dir)
    return fig


def expected_resampling_metrics(pool, users, heavy_user_threshold_quantile,
                                scenario_weights):
    """Compute ratio-of-expectations metrics before drawing any replica."""
    require(0 < heavy_user_threshold_quantile < 1,
            "heavy_user_threshold_quantile must be between 0 and 1")
    eligible_shares = users.loc[users["heavy_eligible"], "high_node_hours_job_share"]
    require(not eligible_shares.empty, "No eligible users for resampling sensitivity")
    threshold = float(
        eligible_shares.quantile(heavy_user_threshold_quantile, interpolation="linear")
    )
    heavy = users["heavy_eligible"] & users["high_node_hours_job_share"].ge(threshold)
    job_metrics = pool.assign(
        _long=pool["runtime_sec"].gt(LONG_SECONDS),
        _wide=pool["node_count"].ge(WIDE_NODES),
        _long_or_wide=(
            pool["runtime_sec"].gt(LONG_SECONDS) | pool["node_count"].ge(WIDE_NODES)
        ),
    ).groupby("source_user_ID", sort=True).agg(
        jobs=("source_row", "size"),
        long_jobs=("_long", "sum"),
        wide_jobs=("_wide", "sum"),
        long_or_wide_jobs=("_long_or_wide", "sum"),
        node_hours=("node_hours", "sum"),
    ).reindex(users.index)
    require(not job_metrics.isna().any().any(), "User metrics do not align with source users")

    rows = []
    draws = len(users)
    for scenario, heavy_weight in scenario_weights.items():
        require(math.isfinite(heavy_weight) and heavy_weight >= 1,
                "Scenario weights must be finite and >= 1")
        weights = np.where(heavy, heavy_weight, 1.0)
        probabilities = weights / weights.sum()
        expected_jobs_per_draw = float(np.dot(probabilities, job_metrics["jobs"]))
        expected_node_hours = draws * float(np.dot(probabilities, job_metrics["node_hours"]))
        rows.append({
            "scenario": scenario,
            "heavy_user_threshold_quantile": heavy_user_threshold_quantile,
            "high_job_share_threshold": threshold,
            "heavy_users": int(heavy.sum()),
            "expected_heavy_draw_share": float(np.dot(probabilities, heavy)),
            "expected_long_job_share": float(
                np.dot(probabilities, job_metrics["long_jobs"]) / expected_jobs_per_draw
            ),
            "expected_wide_job_share": float(
                np.dot(probabilities, job_metrics["wide_jobs"]) / expected_jobs_per_draw
            ),
            "expected_long_or_wide_job_share": float(
                np.dot(probabilities, job_metrics["long_or_wide_jobs"]) / expected_jobs_per_draw
            ),
            "expected_node_hours_ratio": expected_node_hours / pool["node_hours"].sum(),
            "effective_user_diversity_share": (
                1 / np.square(probabilities).sum() / len(users)
            ),
        })
    return pd.DataFrame(rows)


def format_expected_resampling_metrics(metrics):
    """Format the selected sensitivity point for notebook display."""
    report = metrics.copy()
    percentage_columns = [
        "heavy_user_threshold_quantile", "high_job_share_threshold",
        "expected_heavy_draw_share", "expected_long_job_share",
        "expected_wide_job_share", "expected_long_or_wide_job_share",
        "expected_node_hours_ratio", "effective_user_diversity_share",
    ]
    report[percentage_columns] *= 100
    report = report.set_index(["regime", "scenario"]).rename(
        index=lambda value: SCENARIO_LABELS.get(value, str(value).title()),
        columns={
            "heavy_user_threshold_quantile": "Heavy-user percentile",
            "high_job_share_threshold": "Required high-job share [%]",
            "heavy_users": "Heavy users",
            "expected_heavy_draw_share": "Expected heavy draws [%]",
            "expected_long_job_share": "Expected long jobs [%]",
            "expected_wide_job_share": "Expected wide jobs [%]",
            "expected_long_or_wide_job_share": "Expected long or wide jobs [%]",
            "expected_node_hours_ratio": "Expected node-hours / original [%]",
            "effective_user_diversity_share": "Effective user diversity [%]",
        },
    )
    report.index = report.index.set_levels(
        [report.index.levels[0].str.title(), report.index.levels[1]]
    )
    report.index.names = ["Regime", "Scenario"]
    return report[[
        "Expected heavy draws [%]",
        "Expected long jobs [%]",
        "Expected wide jobs [%]",
        "Expected long or wide jobs [%]",
        "Expected node-hours / original [%]",
        "Effective user diversity [%]",
    ]].round(2)


def plot_heavy_user_decision(pools, user_profiles, scenario_weights,
                             figure_dir=FIGURE_DIR, quantile_range=(0.40, 0.90),
                             reference_quantile=None):
    """Map each candidate user percentile directly to expected enrichment."""
    require(set(pools) == set(user_profiles), "Pools and user profiles must cover the same regimes")
    configure_plot_style()
    quantiles = np.linspace(*quantile_range, 51)
    weighted_scenarios = {
        scenario: weight for scenario, weight in scenario_weights.items() if weight > 1
    }
    require(bool(weighted_scenarios), "At least one weight above 1 is required")
    fig, axes = plt.subplots(
        2, len(pools), figsize=(sca.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 4.05),
        squeeze=False, sharex="col", sharey="row",
    )
    for column, regime in enumerate(pools):
        enrichment_axis, count_axis = axes[:, column]
        pool, users = pools[regime], user_profiles[regime]
        sensitivity = pd.concat(
            [
                expected_resampling_metrics(pool, users, quantile, weighted_scenarios)
                for quantile in quantiles
            ],
            ignore_index=True,
        )
        original_target = 100 * (
            pool["runtime_sec"].gt(LONG_SECONDS) | pool["node_count"].ge(WIDE_NODES)
        ).mean()
        for scenario in weighted_scenarios:
            selected = sensitivity.loc[sensitivity["scenario"].eq(scenario)]
            color = SCENARIO_COLORS.get(scenario, None)
            enrichment_axis.step(
                100 * selected["heavy_user_threshold_quantile"],
                100 * selected["expected_long_or_wide_job_share"],
                where="post", color=color, linewidth=1.35,
                label=SCENARIO_LABELS.get(scenario, scenario),
            )
        enrichment_axis.axhline(
            original_target, color="black", linestyle="--", linewidth=0.9,
            label="Original / uniform expectation",
        )
        count_series = sensitivity.loc[
            sensitivity["scenario"].eq(next(iter(weighted_scenarios)))
        ]
        count_axis.step(
            100 * count_series["heavy_user_threshold_quantile"],
            count_series["heavy_users"],
            where="post", color="#5F5F5F", linewidth=1.35,
            label="Heavy-user count",
        )
        if reference_quantile is not None:
            for axis in (enrichment_axis, count_axis):
                axis.axvline(
                    100 * reference_quantile, color="#E45756",
                    linestyle=":", linewidth=1.0,
                )
        enrichment_axis.set(
            title=regime.title(),
            xlim=(100 * quantile_range[0], 100 * quantile_range[1]),
        )
        count_axis.set(
            xlabel="Candidate heavy-user percentile",
            xlim=(100 * quantile_range[0], 100 * quantile_range[1]),
        )
        count_axis.xaxis.set_major_formatter(PercentFormatter(100))
        count_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
        count_axis.annotate(
            "Higher percentile → fewer heavy users",
            xy=(0.98, 0.04), xycoords="axes fraction",
            ha="right", va="bottom", color="0.35", fontsize=7,
        )
        count_axis.annotate(
            f"{int(users['heavy_eligible'].sum())} eligible users",
            xy=(0.02, 0.94), xycoords="axes fraction",
            ha="left", va="top", color="0.35", fontsize=7,
        )
    axes[0, 0].set_ylabel("Expected long or wide jobs [%]")
    axes[1, 0].set_ylabel("Heavy source users [count]")
    for axis in axes[:, 1:].ravel():
        axis.tick_params(labelleft=False)
    scenario_handles = [
        Line2D([], [], color=SCENARIO_COLORS.get(scenario, "black"), label=SCENARIO_LABELS.get(scenario, scenario))
        for scenario in weighted_scenarios
    ]
    reference_handles = [
        Line2D([], [], color="black", linestyle="--", linewidth=0.9,
               label="Original / uniform expectation"),
        Line2D([], [], color="#5F5F5F", linewidth=1.35,
               label="Heavy-user count"),
    ]
    if reference_quantile is not None:
        reference_handles.append(
            Line2D([], [], color="#E45756", linestyle=":", linewidth=1.0,
                   label=f"P{100 * reference_quantile:.0f} selection")
        )
    fig.legend(
        handles=[*scenario_handles, *reference_handles], loc="upper center",
        ncol=len(scenario_handles) + len(reference_handles), frameon=False,
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.13, top=0.82,
        wspace=0.15, hspace=0.18,
    )
    export_figure(fig, f"{FIGURE_PREFIX}_calibration_heavy_user_decision", figure_dir)
    return fig


def _lorenz(values, grid):
    values = np.sort(np.asarray(values, dtype=float))
    cumulative = np.r_[0.0, np.cumsum(values) / values.sum()]
    positions = np.linspace(0.0, 1.0, len(cumulative))
    return np.interp(grid, positions, cumulative)


def _plot_replicate_band(axis, values, x, scenario):
    """Plot the original trace or the range and mean of its five replicas."""
    stacked = np.vstack(values)
    color = SCENARIO_COLORS[scenario]
    if scenario == "original":
        axis.plot(x, stacked[0], color=color, linewidth=1.5, zorder=3)
        return
    axis.fill_between(x, stacked.min(axis=0), stacked.max(axis=0), color=color, alpha=0.16, linewidth=0)
    axis.plot(x, stacked.mean(axis=0), color=color, linewidth=1.15)


def export_figure(fig, stem, figure_dir=FIGURE_DIR):
    """Save publication figures as vector PDF and 600-dpi PNG."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = [figure_dir / f"{stem}.{suffix}" for suffix in ("pdf", "png")]
    fig.savefig(paths[0], format="pdf", dpi=sca.IEEE_LINE_ART_DPI)
    fig.savefig(paths[1], format="png", dpi=sca.IEEE_LINE_ART_DPI)
    return paths


def _plot_composition(group):
    metric_panels = [
        ("jobs", "Replay jobs", "Jobs"),
        ("total_node_hours", "Submitted work", "Node-hours"),
        ("heavy_user_share", "Heavy-user selection", "User instances [%]"),
        ("wide_job_share", "Wide jobs (>=160 nodes)", "Replay jobs [%]"),
        ("long_job_share", "Long jobs (>2 h)", "Replay jobs [%]"),
        ("top_source_user_node_hours_share", "Largest source-user share", "Node-hours [%]"),
    ]
    scenarios = _ordered_scenarios(group)
    positions = np.arange(len(scenarios))
    fig, axes = plt.subplots(2, 3, figsize=(sca.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 4.65))
    for axis, (metric, title, ylabel) in zip(axes.flat, metric_panels, strict=True):
        is_share = metric.endswith("share")
        for position, scenario in zip(positions, scenarios, strict=True):
            values = group.loc[group["scenario"].eq(scenario), metric].to_numpy(
                dtype=float, copy=True
            )
            if is_share:
                values *= 100
            color = SCENARIO_COLORS[scenario]
            if scenario == "original":
                axis.scatter(position, values[0], color=color, marker="D", s=19, zorder=3)
            else:
                offsets = np.linspace(-0.13, 0.13, len(values))
                axis.scatter(position + offsets, values, color=color, s=15, alpha=0.85, zorder=2)
                axis.plot([position - 0.17, position + 0.17], [values.mean(), values.mean()], color=color, linewidth=1.2)
        axis.set_title(title)
        axis.set_xticks(positions, [SCENARIO_LABELS[s] for s in scenarios], rotation=30, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="x", visible=False)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    handles = [
        Line2D([], [], color="black", marker="D", linestyle="", markersize=4, label="Original"),
        Line2D([], [], color="0.3", marker="o", linestyle="", markersize=4, label="One replicate"),
        Line2D([], [], color="0.3", linewidth=1.2, label="Replicate mean"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.84, wspace=0.38, hspace=0.54)
    return fig


def _plot_distribution_panels(group, frames):
    scenarios = _ordered_scenarios(group)
    t0, t1 = utc_window((group["window_start"].iloc[0], group["window_end"].iloc[0]))
    daily_index = pd.date_range(t0, t1, freq="D", inclusive="left")
    runtime_max = max(frame["runtime_sec"].max() for frame in frames.values())
    node_max = max(frame["node_count"].max() for frame in frames.values())
    runtime_grid = np.geomspace(1, runtime_max, 400)
    node_grid = np.geomspace(1, node_max, 400)
    fraction_grid = np.linspace(0, 1, 400)
    fig = plt.figure(figsize=(sca.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 4.85))
    grid = fig.add_gridspec(2, 6)
    runtime_axis = fig.add_subplot(grid[0, :3])
    node_axis = fig.add_subplot(grid[0, 3:])
    daily_axis = fig.add_subplot(grid[1, :2])
    instance_axis = fig.add_subplot(grid[1, 2:4])
    source_axis = fig.add_subplot(grid[1, 4:], sharey=instance_axis)
    for scenario in scenarios:
        scenario_frames = _scenario_frames(group, frames, scenario)
        _plot_replicate_band(
            runtime_axis,
            [_ecdf(frame["runtime_sec"], runtime_grid) for frame in scenario_frames],
            runtime_grid, scenario,
        )
        _plot_replicate_band(
            node_axis,
            [_ecdf(frame["node_count"], node_grid) for frame in scenario_frames],
            node_grid, scenario,
        )
        daily = []
        instance_lorenz = []
        source_lorenz = []
        for frame in scenario_frames:
            arrivals = frame.set_index("submit_utc")
            daily.append(arrivals.resample("D").size().reindex(daily_index, fill_value=0).to_numpy())
            instance_lorenz.append(_lorenz(frame.groupby("user_ID")["node_hours"].sum(), fraction_grid))
            source_lorenz.append(_lorenz(frame.groupby("source_user_ID")["node_hours"].sum(), fraction_grid))
        _plot_replicate_band(daily_axis, daily, np.arange(len(daily_index)), scenario)
        _plot_replicate_band(instance_axis, instance_lorenz, fraction_grid, scenario)
        _plot_replicate_band(source_axis, source_lorenz, fraction_grid, scenario)
    runtime_axis.set(
        xscale="log", title="Runtime distribution",
        xlabel="Runtime [s]", ylabel="Cumulative job share",
    )
    node_axis.set(
        xscale="log", title="Node-count distribution",
        xlabel="Requested nodes", ylabel="Cumulative job share",
    )
    daily_axis.set(title="Daily submissions", xlabel="Days since T0", ylabel="Jobs/day")
    instance_axis.set(
        title="Instance concentration",
        xlabel="Cumulative instance share", ylabel="Node-hour share",
    )
    source_axis.set(
        title="Source concentration", xlabel="Cumulative source share", ylabel="",
    )
    source_axis.tick_params(labelleft=False)
    for axis in (instance_axis, source_axis):
        axis.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=0.8, zorder=0)
    handles = [Line2D([], [], color=SCENARIO_COLORS[s], label=SCENARIO_LABELS[s]) for s in scenarios]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.12, top=0.85,
        wspace=1.20, hspace=0.58,
    )
    return fig


def plot_diagnostics(summary, frames, figure_dir=FIGURE_DIR):
    """Export compact publication figures for each source regime.

    Lines show the original workload or the mean of five replicas. Shaded bands
    span the replica range and are descriptive rather than confidence intervals.
    """
    configure_plot_style()
    figures = []
    for regime, group in summary.groupby("regime", sort=False):
        composition = _plot_composition(group)
        distributions = _plot_distribution_panels(group, frames)
        export_figure(composition, f"{FIGURE_PREFIX}_{regime}_composition", figure_dir)
        export_figure(distributions, f"{FIGURE_PREFIX}_{regime}_distributions", figure_dir)
        figures.extend([composition, distributions])
    return figures
