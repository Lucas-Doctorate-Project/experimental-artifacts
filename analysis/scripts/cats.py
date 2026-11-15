"""Verify CATS admissions and compare paired schedules on the original carbon traces."""

import argparse
import tomllib
from pathlib import Path

import helpers
import numpy as np
import pandas as pd
from results_metrics import HORIZONS


def previous_day_costs(release, duration, history, offsets):
    grid = 900
    # history[0] is the sample at -86400
    first = np.floor(release / grid).astype(int)
    daily = history[first[:, None] + np.arange(96)]
    prefix = np.column_stack((np.zeros(len(release)), np.cumsum(daily * grid, axis=1)))
    begins = (release % grid)[:, None] + offsets
    rows = np.arange(len(release))[:, None]

    def cumulative(seconds):
        cycles, phase = np.divmod(seconds, 86400)
        index = (phase // grid).astype(int)
        return (
            cycles * prefix[:, -1:]
            + prefix[rows, index]
            + (phase % grid) * daily[rows, index]
        )

    return cumulative(begins + duration[:, None]) - cumulative(begins)


def check_admissions(jobs, path, history, known_future=False):
    """Independently score the 25 admission candidates using the selected forecast."""
    log = pd.read_csv(path, sep="\t", dtype={"job_id": str}).set_index("job_id")
    if not log.index.is_unique or set(log.index) != set(jobs.job_id):
        raise ValueError(f"{path}: admission IDs differ from the output jobs")
    log = log.loc[jobs.job_id]
    arrival = jobs.submission_time.to_numpy()
    eligible = log.eligible_time.to_numpy()
    if not np.allclose(arrival, log.submission_time, atol=1e-6, rtol=0):
        raise ValueError(f"{path}: submission times changed")
    if np.any(jobs.starting_time.to_numpy() + 1e-6 < eligible):
        raise ValueError(f"{path}: job started before admission")
    if np.any(eligible[jobs.is_context] != arrival[jobs.is_context]):
        raise ValueError(f"{path}: context job deferred")
    replay = ~jobs.is_context.to_numpy()
    release = arrival[replay]
    duration = jobs.requested_time.to_numpy()[replay]
    grid = 900
    offsets = np.arange(25) * grid
    if known_future:
        times = np.arange(len(history) + 1) * grid - 86400
        prefix = np.r_[0, np.cumsum(history * grid)]
        begins = release[:, None] + offsets
        ends = begins + duration[:, None]
        if ends.max() > times[-1]:
            raise ValueError(f"{path}: future window out of range")
        costs = np.interp(ends, times, prefix) - np.interp(begins, times, prefix)
    else:
        costs = previous_day_costs(release, duration, history, offsets)
    selected = (eligible[replay] - release) / grid
    which = np.rint(selected).astype(int)
    if np.any(np.abs(which - selected) > 1e-8) or np.any(
        (which < 0) | (which >= len(offsets))
    ):
        raise ValueError(f"{path}: admission outside candidate set")
    # allow for floating point subtraction of long integrals
    best = costs.min(axis=1)
    if not np.allclose(costs[np.arange(len(which)), which], best, rtol=2e-9, atol=2e-5):
        raise ValueError(f"{path}: admission differs from independent forecast minimum")
    return eligible[replay] - release


def check_schedule(jobs, reference, capacity):
    """Check paired work and exact aggregate node occupancy at exported event times."""
    for field in (
        "job_id",
        "profile",
        "submission_time",
        "requested_time",
        "requested_number_of_resources",
    ):
        if not np.array_equal(jobs[field], reference[field]):
            raise ValueError(f"paired {field} differs")
    if not np.allclose(
        jobs.execution_time, reference.execution_time, rtol=0, atol=2e-6
    ):
        raise ValueError("paired execution durations differ")
    terminal = {"COMPLETED_SUCCESSFULLY", "COMPLETED_WALLTIME_REACHED"}
    if not set(jobs.final_state) <= terminal:
        raise ValueError("jobs rejected or unfinished")
    different = jobs.final_state != reference.final_state
    if not np.allclose(
        jobs.execution_time[different],
        jobs.requested_time[different],
        atol=2e-6,
        rtol=0,
    ):
        raise ValueError(
            "paired terminal states differ away from the walltime boundary"
        )
    nodes = jobs.requested_number_of_resources.to_numpy()
    times = np.r_[jobs.starting_time, jobs.finish_time]
    changes = np.r_[nodes, -nodes]
    peak = np.cumsum(changes[np.lexsort((changes, times))]).max()
    if peak > capacity:
        raise ValueError(f"peak allocation {peak} exceeds {capacity}")
    return peak


def load(path):
    jobs = (
        pd.read_csv(path, dtype={"job_id": str})
        .sort_values("job_id")
        .reset_index(drop=True)
    )
    if not jobs.job_id.is_unique:
        raise ValueError(f"duplicate job ID in {path}")
    jobs["is_context"] = jobs.job_id.str.startswith("ctx_")
    return jobs


def forecast_errors(history):
    """MAE by lead, with one issue every 15 minutes across the 28-day arrival window."""
    issues = 96 + np.arange(28 * 96)
    errors = {}
    for low, high in ((0, 6), (6, 24), (24, 42)):
        leads = np.arange(4 * low, 4 * high)
        actual = history[issues[:, None] + leads]
        forecast = history[issues[:, None] - 96 + leads % 96]
        errors[f"forecast_mae_{low}_{high}h"] = float(np.abs(actual - forecast).mean())
    return errors


def analyze(config, outputs):
    experiments = tomllib.loads(config.read_text())["experiment"]
    original = helpers.ROOT / "analysis/outputs/original"
    input_root = helpers.ROOT / "experiments"
    platforms = {p: helpers.platform(p) for p in {e["platform"] for e in experiments}}
    signals = {
        p: helpers.load_signals(p)
        for p in {e["environmental_trace"] for e in experiments}
    }
    histories = {}
    errors = {}
    rows = []
    for experiment in experiments:
        name = experiment["name"]
        case = name.removeprefix("cats_carbon_")
        workload = Path(experiment["workload"]).stem
        machine = platforms[experiment["platform"]]
        target = signals[experiment["environmental_trace"]]
        trace = Path(experiment["environmental_trace"]).stem
        if trace not in histories:
            history_path = (input_root / experiment["variant_options"]).with_name(
                f"{trace}.csv"
            )
            history = pd.read_csv(history_path)
            if not np.array_equal(
                history.timestamp, np.arange(len(history)) * 900 - 86400
            ):
                raise ValueError(f"irregular history grid in {history_path}")
            histories[trace] = history.value.to_numpy()
            errors[trace] = forecast_errors(histories[trace])
        easy = load(original / f"easy_bf_{case}" / "out_jobs.csv")
        green = load(original / f"greenfilling_carbon_{case}" / "out_jobs.csv")
        cats = load(outputs / name / "out_jobs.csv")
        intentional = check_admissions(
            cats, outputs / name / "admissions.tsv", histories[trace]
        )
        for policy, jobs in (("EASY", easy), ("Greenfilling", green), ("CATS", cats)):
            peak = check_schedule(jobs, easy, machine["nodes"])
            totals = helpers.physical_totals(
                jobs, machine, target, HORIZONS["original"]
            )
            performance = helpers.performance(jobs)
            wait = jobs.loc[~jobs.is_context, "waiting_time"].mean()
            delay = intentional.mean() if policy == "CATS" else 0.0
            rows.append(
                {
                    "workload": workload,
                    "trace": trace,
                    "policy": policy,
                    "peak_nodes": peak,
                    "terminal_jobs": len(jobs),
                    "successful_jobs": jobs.success.sum(),
                    "walltime_status_differences_vs_easy": (
                        jobs.final_state != easy.final_state
                    ).sum(),
                    "mean_wait_seconds": wait,
                    "mean_admission_delay_seconds": delay,
                    "mean_queue_wait_seconds": wait - delay,
                    "admission_delayed_jobs": np.count_nonzero(intentional)
                    if policy == "CATS"
                    else 0,
                    **performance,
                    **totals,
                    **(errors[trace] if policy == "CATS" else {}),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("outputs", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.results.exists():
        parser.error(f"results file already exists: {args.results}")
    metrics = analyze(args.config, args.outputs)
    metrics.to_csv(args.results, index=False, mode="x")
    print(f"Verified {len(metrics) // 3} paired schedules")
