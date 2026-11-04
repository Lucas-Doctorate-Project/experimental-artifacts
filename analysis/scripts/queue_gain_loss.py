"""Split the oracle's saving into jobs run cleaner and dirtier: python analysis/scripts/queue_gain_loss.py."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from opportunity import BASELINE_GROUP, HORIZON
from oracle import ORACLE_GROUP, OracleSource
from queue_gain_loss_plots import save_queue_gain_loss_figure

PLANNING_HORIZON_SECONDS = 86400
WORKERS = 8


def gain_and_loss(easy: pd.DataFrame, policy: pd.DataFrame, machine: dict, signal: dict) -> dict:
    """Footprint change of jobs run cleaner and dirtier than under EASY, in % of the EASY footprint.

    A job's change is its nodes times the busy power above idle times the change
    in its intensity integral. Idle power is fixed, so gain minus loss equals the
    pair's saving. Example: gain_and_loss(easy, policy, machine, signal).
    """
    nodes = easy.requested_number_of_resources.to_numpy()
    increment = (machine["busy_w"] - machine["idle_w"]) / helpers.J_PER_KWH
    change = (
        increment
        * nodes
        * (
            helpers.integral(signal, policy.starting_time.to_numpy(), policy.finish_time.to_numpy())
            - helpers.integral(signal, easy.starting_time.to_numpy(), easy.finish_time.to_numpy())
        )
    )
    total = helpers.physical_totals(easy, machine, {"scored": signal}, HORIZON)["scored_l"]
    return {
        "gain_pct": 100 * -change[change < 0].sum() / total,
        "loss_pct": 100 * change[change > 0].sum() / total,
        "net_pct": 100 * -change.sum() / total,
    }


def unrelated_traces(config: pd.DataFrame) -> dict[str, str]:
    """For each window, the next window of the same zone, cyclically. Its moves were not chosen for it."""
    traces = config[["zone", "trace", "environmental_trace"]].drop_duplicates().sort_values("trace")
    unrelated = {}
    for _, zone in traces.groupby("zone"):
        paths = zone.environmental_trace.tolist()
        for i, trace in enumerate(zone.trace):
            unrelated[trace] = paths[(i + 1) % len(paths)]
    return unrelated


def score_pair(task: dict) -> list[dict]:
    """Score one oracle run against its own window and an unrelated one."""
    source = OracleSource()
    machine = source.platform(task["platform"])
    easy = source.load_jobs(BASELINE_GROUP, task["easy_name"], machine["nodes"])
    policy = source.load_jobs(ORACLE_GROUP, task["name"], machine["nodes"])
    if not easy.job_id.equals(policy.job_id):
        raise ValueError(f"run {task['name']!r} and its EASY baseline contain different jobs")
    rows = []
    for scoring, path in (("own", task["environmental_trace"]), ("unrelated", task["unrelated_trace"])):
        signal = source.load_signals(path)[task["objective"]]
        rows.append(
            {
                "workload": task["workload"],
                "trace": task["trace"],
                "objective": task["objective"],
                "scoring": scoring,
                "scoring_trace": path,
                **gain_and_loss(easy, policy, machine, signal),
            }
        )
    return rows


def oracle_tasks() -> list[dict]:
    """One task per oracle run at the 24 h horizon, with its EASY baseline and unrelated window."""
    source = OracleSource()
    baseline = source.experiments(BASELINE_GROUP)
    easy = baseline.loc[baseline.variant == "easy_bf"].set_index(["workload", "trace"])["name"]
    unrelated = unrelated_traces(baseline)
    oracle = source.experiments(ORACLE_GROUP)
    oracle = oracle.loc[oracle.planning_horizon_seconds == PLANNING_HORIZON_SECONDS]
    columns = ["name", "workload", "trace", "objective", "platform", "environmental_trace"]
    return [
        run | {"easy_name": easy[(run["workload"], run["trace"])], "unrelated_trace": unrelated[run["trace"]]}
        for run in oracle[columns].to_dict("records")
    ]


def main() -> None:
    """Render the gain and loss figure. Example: python analysis/scripts/queue_gain_loss.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        scores = pd.DataFrame([row for rows in pool.map(score_pair, oracle_tasks()) for row in rows])
    save_queue_gain_loss_figure(scores, arguments.output_dir)
    medians = scores.groupby("scoring")[["gain_pct", "loss_pct", "net_pct"]].median()
    print(f"{len(scores) // 2} pairs, medians in % of the EASY footprint:\n{medians.round(2).to_string()}")
    print(f"Gain and loss figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
