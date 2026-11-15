"""Analyze all four six-hour carbon schedules for each original workload and trace."""

import argparse
import json
import tomllib
from pathlib import Path

import cats
import helpers
import numpy as np
import pandas as pd
from oracle import oracle_output
from results_metrics import HORIZONS


def check_head(log_path, jobs, machine, observations):
    """Recompute each logged head-job score with the forecast frozen at its decision."""
    choices = pd.read_csv(log_path, sep="\t", dtype={"job_id": str})
    if choices.empty or not choices.job_id.isin(jobs.job_id).all():
        raise ValueError(f"{log_path}: missing or unknown head decisions")
    jobs_by_id = jobs.set_index("job_id")
    details = jobs_by_id.loc[choices.job_id]
    issued = choices.issued.to_numpy()
    begin = choices.begin.to_numpy()
    end = choices.end.to_numpy()
    origin = choices.groupby("job_id").issued.transform("first").to_numpy()
    if np.any(begin < issued - 1e-6) or np.any(
        (begin > issued + 1e-6) & (begin > origin + 21600 + 1e-6)
    ):
        raise ValueError(f"{log_path}: head displaced outside horizon")
    if not np.allclose(end - begin, details.requested_time + 5, rtol=0, atol=2e-6):
        raise ValueError(f"{log_path}: scored walltime changed")

    step = 900
    times = np.arange(len(observations) + 1) * step - 86400
    prefix = np.r_[0.0, np.cumsum(observations * step)]
    cutoff = np.floor(issued / step) * step
    if np.any(cutoff - 86400 < times[0]) or np.any(cutoff > times[-1]):
        raise ValueError(f"{log_path}: missing historical observations")
    first = np.interp(cutoff - 86400, times, prefix)
    daily_sum = np.interp(cutoff, times, prefix) - first

    def integral_to(date):
        cycles, phase = np.divmod(np.maximum(0, date - cutoff), 86400)
        forecast = (
            cycles * daily_sum
            + np.interp(cutoff - 86400 + phase, times, prefix)
            - first
        )
        return np.interp(np.minimum(date, cutoff), times, prefix) + forecast

    busy_cost = machine["busy_w"] * (integral_to(end) - integral_to(begin))
    idle_cost = machine["idle_w"] * (integral_to(begin) - integral_to(origin))
    cost = busy_cost + idle_cost
    cost *= details.requested_number_of_resources.to_numpy() / helpers.J_PER_KWH
    if not np.allclose(cost, choices.impact, rtol=2e-8, atol=1e-7):
        raise ValueError(
            f"{log_path}: logged scores do not match previous-day forecast"
        )
    last = choices.drop_duplicates("job_id", keep="last").set_index("job_id")
    if not np.allclose(
        last.begin,
        jobs_by_id.loc[last.index, "starting_time"],
        rtol=0,
        atol=2e-6,
    ):
        raise ValueError(f"{log_path}: selected and realized head starts differ")
    return int(((last.begin > last.issued) & ~last.index.str.startswith("ctx_")).sum())


def analyze(config, archive):
    input_root = helpers.ROOT / "experiments"
    outputs = helpers.ROOT / "analysis/outputs"
    runs = tomllib.loads(config.read_text())["experiment"]
    experiments_by_name = {experiment["name"]: experiment for experiment in runs}
    cases = {
        experiment["name"].removeprefix("cats_known_future_")
        for experiment in runs
        if experiment["variant_name"] == "cats"
    }
    if len(cases) != 144 or len(experiments_by_name) != 288 or len(runs) != 288:
        raise ValueError("expected 144 cases with two new schedules each")
    signals, histories, machines = {}, {}, {}
    rows = []
    for case in sorted(cases):
        cats_name = f"cats_known_future_{case}"
        head_name = f"head_previous_day_{case}"
        cats_config = experiments_by_name[cats_name]
        head_config = experiments_by_name[head_name]
        if any(
            cats_config[field] != head_config[field]
            for field in ("workload", "platform", "environmental_trace")
        ):
            raise ValueError(f"{case}: unpaired inputs")
        workload = Path(cats_config["workload"]).stem
        trace = Path(cats_config["environmental_trace"]).stem
        if trace not in histories:
            options = json.loads(
                (input_root / head_config["variant_options"]).read_text()
            )
            history = pd.read_csv(input_root / options["intensity_trace"])
            if not np.array_equal(
                history.timestamp, np.arange(len(history)) * 900 - 86400
            ):
                raise ValueError(f"{trace}: irregular history")
            histories[trace] = history.value.to_numpy()
            signals[trace] = helpers.load_signals(cats_config["environmental_trace"])
            if not np.array_equal(
                histories[trace][96:], signals[trace]["carbon"]["values"]
            ):
                raise ValueError(f"{trace}: scheduling and accounting traces differ")
        platform = cats_config["platform"]
        if platform not in machines:
            machines[platform] = helpers.platform(platform)
        machine = machines[platform]
        old_head = f"green_window_scheduling_carbon_h21600_{case}"
        old_cats = outputs / "cats-carbon-20261104/full" / f"cats_carbon_{case}"
        easy = cats.load(outputs / "original" / f"easy_bf_{case}" / "out_jobs.csv")
        baseline = helpers.physical_totals(
            easy, machine, signals[trace], HORIZONS["original"]
        )
        replay = ~easy.is_context
        easy_wait = easy.loc[replay, "waiting_time"].mean()
        easy_slowdown = helpers.performance(easy)["bounded_slowdown"]
        for policy, forecast, folder in [
            ("CATS", "previous_day", old_cats),
            ("CATS", "known_future", archive / cats_name),
            ("Head", "previous_day", archive / head_name),
            ("Head", "known_future", outputs / oracle_output(old_head) / old_head),
        ]:
            jobs = cats.load(folder / "out_jobs.csv")
            peak = cats.check_schedule(jobs, easy, machine["nodes"])
            held = None
            if policy == "CATS":
                delay = cats.check_admissions(
                    jobs,
                    folder / "admissions.tsv",
                    histories[trace],
                    known_future=forecast == "known_future",
                )
                held = int(np.count_nonzero(delay))
            elif forecast == "previous_day":
                held = check_head(folder / "head.tsv", jobs, machine, histories[trace])
            totals = helpers.physical_totals(
                jobs, machine, signals[trace], HORIZONS["original"]
            )
            performance = helpers.performance(jobs)
            wait = jobs.loc[replay, "waiting_time"].mean()
            rows.append(
                {
                    "case": case,
                    "workload": workload,
                    "trace": trace,
                    "policy": policy,
                    "forecast": forecast,
                    "deliberately_delayed_replay_jobs": held,
                    "peak_nodes": peak,
                    "same_runtime_max_error": np.abs(
                        jobs.execution_time - easy.execution_time
                    ).max(),
                    "walltime_label_changes": (
                        jobs.final_state != easy.final_state
                    ).sum(),
                    "carbon_g": totals["carbon_g"],
                    "easy_carbon_g": baseline["carbon_g"],
                    "saving_pct": 100 * (1 - totals["carbon_g"] / baseline["carbon_g"]),
                    "mean_wait_seconds": wait,
                    "added_wait_h": (wait - easy_wait) / 3600,
                    "slowdown_ratio": performance["bounded_slowdown"] / easy_slowdown,
                    **performance,
                }
            )
    return pd.DataFrame(rows)


def paired_differences(results):
    by_case = results.pivot(
        index=["case", "workload", "trace"],
        columns=["policy", "forecast"],
        values=["saving_pct", "added_wait_h", "bounded_slowdown"],
    )
    contrasts = [
        (("CATS", "known_future"), ("CATS", "previous_day")),
        (("Head", "known_future"), ("Head", "previous_day")),
        (("CATS", "previous_day"), ("Head", "previous_day")),
        (("CATS", "known_future"), ("Head", "known_future")),
    ]
    rows = []
    for first, second in contrasts:
        data = by_case["saving_pct"][first] - by_case["saving_pct"][second]
        pairs = data.rename("extra_saving_pp").to_frame()
        pairs["extra_wait_h"] = (
            by_case["added_wait_h"][first] - by_case["added_wait_h"][second]
        )
        pairs["slowdown_ratio"] = (
            by_case["bounded_slowdown"][first] / by_case["bounded_slowdown"][second]
        )
        rows.append(
            pairs.assign(first=" ".join(first), second=" ".join(second)).reset_index()
        )
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    metrics = analyze(args.config, args.archive)
    pairs = paired_differences(metrics)
    args.destination.mkdir(exist_ok=False, parents=True)
    metrics.to_csv(args.destination / "metrics.csv", index=False)
    pairs.to_csv(args.destination / "contrasts.csv", index=False)
    print(f"Verified {len(metrics) // 4} complete four-way comparisons")
