"""Compare carbon policies on the same original workload/window pairs."""

import argparse
from pathlib import Path

import cats
import helpers
import numpy as np
import pandas as pd
from oracle import moved_intensity_change, oracle_output
from results_metrics import HORIZONS

OUTPUTS = helpers.ROOT / "analysis/outputs"
KEYS = ["workload", "trace"]
OFFLINE_RESULTS = OUTPUTS / "offline-shifting/results.csv"


def policy_schedules(case):
    """Locate online schedules, preserving the forecast and horizon of each run."""
    yield (
        "Greenfilling",
        "observed",
        0,
        OUTPUTS / "original" / f"greenfilling_carbon_{case}",
    )
    yield (
        "CATS",
        "previous_day",
        6,
        OUTPUTS / "cats-carbon-20261104/full" / f"cats_carbon_{case}",
    )
    crossed = OUTPUTS / "forecast-cross-20261104/full"
    yield "CATS", "known_future", 6, crossed / f"cats_known_future_{case}"
    yield "Head", "previous_day", 6, crossed / f"head_previous_day_{case}"
    for hours in (6, 12, 24):
        name = f"green_window_scheduling_carbon_h{hours * 3600}_{case}"
        yield "Head", "known_future", hours, OUTPUTS / oracle_output(name) / name


def check_pairs(rows, expected):
    """Require one complete set of original cases for each configuration."""
    for configuration, group in rows.groupby(["policy", "forecast", "hours"]):
        index = group.set_index(KEYS).index
        if not index.is_unique or not index.sort_values().equals(
            expected.sort_values()
        ):
            raise ValueError(
                f"{configuration}: expected the same 144 workload/window pairs"
            )
    if not np.isfinite(rows[["saving_pct", "response_pct"]].to_numpy()).all():
        raise ValueError("nonfinite footprint or response-time result")


def comparison_data(offline_results=OFFLINE_RESULTS):
    runs = helpers.experiments("original")
    easy_runs = runs.loc[runs.variant == "easy_bf"]
    expected = easy_runs.set_index(KEYS).index
    if len(expected) != 144 or not expected.is_unique:
        raise ValueError("expected 144 distinct original EASY cases")
    platforms = {path: helpers.platform(path) for path in easy_runs.platform.unique()}
    signals = {
        path: helpers.load_signals(path)
        for path in easy_runs.environmental_trace.unique()
    }
    rows = []
    for _, run in easy_runs.iterrows():
        machine = platforms[run.platform]
        carbon = {"carbon": signals[run.environmental_trace]["carbon"]}
        easy = cats.load(OUTPUTS / "original" / run["name"] / "out_jobs.csv")
        easy_total = helpers.physical_totals(
            easy, machine, carbon, HORIZONS["original"]
        )["carbon_g"]
        easy_response = helpers.performance(easy)["weighted_response_seconds"]
        case = f"{run.workload}_{run.trace}"
        for policy, forecast, hours, folder in policy_schedules(case):
            jobs = cats.load(folder / "out_jobs.csv")
            cats.check_schedule(jobs, easy, machine["nodes"])
            total = helpers.physical_totals(
                jobs, machine, carbon, HORIZONS["original"]
            )["carbon_g"]
            response = helpers.performance(jobs)["weighted_response_seconds"]
            rows.append(
                {
                    "workload": run.workload,
                    "trace": run.trace,
                    "policy": policy,
                    "forecast": forecast,
                    "hours": hours,
                    "saving_pct": 100 * (1 - total / easy_total),
                    "response_pct": 100 * (response / easy_response - 1),
                    "weighted_response_seconds": response,
                    "easy_weighted_response_seconds": easy_response,
                    "moved_intensity_pct": moved_intensity_change(
                        easy, jobs, carbon["carbon"]
                    ),
                }
            )
    online = pd.DataFrame(rows)
    check_pairs(online, expected)
    offline = (
        pd.read_csv(offline_results)
        .query("group == 'original' and objective == 'carbon'")
        .copy()
    )
    offline["policy"] = "Offline"
    offline["forecast"] = "known_future_runtime"
    offline["hours"] = offline.budget_hours
    offline["response_pct"] = 100 * (
        offline.weighted_response_seconds / offline.easy_weighted_response_seconds - 1
    )
    check_pairs(offline, expected)
    if set(offline.hours) != {3, 6, 12, 24}:
        raise ValueError("expected offline budgets of 3, 6, 12, and 24 hours")
    if not np.isfinite(offline.bound_saving_pct).all():
        raise ValueError("nonfinite delay-only bound")
    baseline = online.loc[
        online.policy == "Greenfilling", KEYS + ["easy_weighted_response_seconds"]
    ]
    paired = offline.merge(
        baseline, on=KEYS, suffixes=("", "_checked"), validate="many_to_one"
    )
    if not np.allclose(
        paired.easy_weighted_response_seconds,
        paired.easy_weighted_response_seconds_checked,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise ValueError("offline and online EASY response-time baselines differ")
    bounds = offline[KEYS + ["hours", "bound_saving_pct"]].copy()
    combined = pd.concat(
        [online, offline.reindex(columns=online.columns)], ignore_index=True
    )
    return combined, bounds


def main_rows(results):
    return results.loc[(results.hours == 6) | (results.policy == "Greenfilling")]


def generate_figures(destination, offline_results=OFFLINE_RESULTS, data_dir=None):
    from policy_comparison_plots import save_comparison_figures

    if data_dir is not None and data_dir.exists():
        raise FileExistsError(f"data directory already exists: {data_dir}")
    results, bounds = comparison_data(offline_results)
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=False)
        results.to_csv(data_dir / "policy_comparison.csv", index=False)
        bounds.to_csv(data_dir / "delay_only_bound.csv", index=False)
    save_comparison_figures(results, bounds, destination)
    print(
        main_rows(results)
        .groupby(["policy", "forecast"])[
            ["saving_pct", "response_pct", "moved_intensity_pct"]
        ]
        .median()
        .to_string()
    )
    print(
        f"Verified 144 matched carbon cases per configuration; saved figures to {destination.resolve()}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=helpers.ROOT / "analysis/figures"
    )
    parser.add_argument("--offline-results", type=Path, default=OFFLINE_RESULTS)
    parser.add_argument("--data-dir", type=Path, help="new directory for per-case CSVs")
    args = parser.parse_args()
    generate_figures(args.output_dir, args.offline_results, args.data_dir)


if __name__ == "__main__":
    main()
