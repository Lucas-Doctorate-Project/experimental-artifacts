"""Compare the delay-only savings bound with realized savings: python analysis/opportunity.py."""

import argparse
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from opportunity_plots import save_opportunity_figure
from results_metrics import HORIZONS

BASELINE_GROUP = "original"
HORIZON = HORIZONS[BASELINE_GROUP]
DELAY_HOURS = (0, 1, 3, 6, 12, 24)


def bound_curves(group: str, source: ModuleType = helpers) -> pd.DataFrame:
    """Delay-only bound per EASY run and objective. Example: bound_curves('original')."""
    config = source.experiments(group)
    signals = {path: source.load_signals(path) for path in config.environmental_trace.unique()}
    platforms = {path: source.platform(path) for path in config.platform.unique()}
    rows = []
    for _, run in config.loc[config.variant == "easy_bf"].iterrows():
        machine = platforms[run.platform]
        jobs = source.load_jobs(group, run["name"], machine["nodes"])
        for objective, signal in signals[run.environmental_trace].items():
            bound = source.relaxed_opportunity(
                jobs, signal, machine, HORIZON, objective, DELAY_HOURS, allow_advance=False
            )
            rows.append(bound.assign(workload=run.workload, trace=run.trace, objective=objective))
    return pd.concat(rows, ignore_index=True)


def policy_label(run: pd.Series) -> str:
    if run.variant.startswith("greenfilling"):
        return "Greenfilling"
    if run.variant.startswith("green_window_scheduling"):
        return f"Oracle head shift {run.planning_horizon_seconds / 3600:g} h"
    raise ValueError(f"unknown variant {run.variant!r} in run {run['name']!r}")


def realized_points(
    group: str, baseline_group: str, source: ModuleType = helpers
) -> pd.DataFrame:
    """Targeted saving and mean added wait against EASY. Example: realized_points('original', 'original')."""
    baseline = source.experiments(baseline_group)
    baseline = baseline.loc[baseline.variant == "easy_bf"].set_index(["workload", "trace"])
    config = source.experiments(group)
    signals = {path: source.load_signals(path) for path in config.environmental_trace.unique()}
    platforms = {path: source.platform(path) for path in config.platform.unique()}
    rows = []
    for _, run in config.loc[config.variant != "easy_bf"].iterrows():
        machine = platforms[run.platform]
        easy_run = baseline.loc[(run.workload, run.trace)]
        easy = source.load_jobs(baseline_group, easy_run["name"], machine["nodes"])
        policy = source.load_jobs(group, run["name"], machine["nodes"])
        if not easy.job_id.equals(policy.job_id):
            raise ValueError(f"run {run['name']!r} and its EASY baseline contain different jobs")
        signal = {run.objective: signals[run.environmental_trace][run.objective]}
        unit = "g" if run.objective == "carbon" else "l"
        key = f"{run.objective}_{unit}"
        easy_total = source.physical_totals(easy, machine, signal, HORIZON)[key]
        policy_total = source.physical_totals(policy, machine, signal, HORIZON)[key]
        replay = ~policy.is_context
        added_wait = policy.starting_time[replay] - easy.starting_time[replay]
        rows.append(
            {
                "group": group,
                "policy": policy_label(run),
                "workload": run.workload,
                "trace": run.trace,
                "objective": run.objective,
                "saving_pct": 100 * (1 - policy_total / easy_total),
                "added_wait_hours": float(np.mean(added_wait)) / 3600,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Render the opportunity figure. Example: python analysis/opportunity.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    bounds = bound_curves(BASELINE_GROUP)
    save_opportunity_figure(bounds, realized_points(BASELINE_GROUP, BASELINE_GROUP), arguments.output_dir)
    print(f"Opportunity figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
