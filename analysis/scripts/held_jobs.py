"""Split the oracle's saving into the jobs it held and all other jobs: python analysis/scripts/held_jobs.py."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from held_jobs_plots import save_held_jobs_figure
from opportunity import BASELINE_GROUP, HORIZON
from oracle import ORACLE_GROUP, OracleSource
from queue_gain_loss import oracle_tasks

DECISIONS = helpers.ROOT / "analysis/outputs/green-window-scheduling-debug"
WORKERS = 8


def held_jobs(name: str) -> set[str]:
    """Jobs whose last window choice starts later than the decision date. Example: held_jobs(run_name).

    decisions.txt holds `now <date>` for each Batsim event received and the JSON
    of each `green_window_selected` log line, in log order, so a decision's date
    is the last `now` before it.
    """
    date, last = 0.0, {}
    with (DECISIONS / name / "decisions.txt").open() as lines:
        for line in lines:
            if line.startswith("now "):
                date = float(line[4:])
                continue
            choice = json.loads(line)
            last[choice["job"].split("!", 1)[1]] = (date, choice["begin"])
    return {job for job, (decided, begin) in last.items() if begin > decided + 1e-6}


def split_pair(task: dict) -> list[dict]:
    """Saving of held and other jobs in one oracle run, scored on its own window and an unrelated one."""
    source = OracleSource()
    machine = source.platform(task["platform"])
    easy = source.load_jobs(BASELINE_GROUP, task["easy_name"], machine["nodes"])
    policy = source.load_jobs(ORACLE_GROUP, task["name"], machine["nodes"])
    logged = pd.read_csv(DECISIONS / task["name"] / "out_jobs.csv", dtype={"job_id": str})
    if not easy.job_id.equals(policy.job_id):
        raise ValueError(f"run {task['name']!r} and its EASY baseline contain different jobs")
    if not np.array_equal(
        logged.sort_values("job_id").starting_time.to_numpy(), policy.starting_time.to_numpy()
    ):
        raise ValueError(f"run {task['name']!r} has a different schedule in its decision-log rerun")
    held = easy.job_id.isin(held_jobs(task["name"])).to_numpy()
    moved = np.abs(policy.starting_time.to_numpy() - easy.starting_time.to_numpy()) > 1
    nodes = easy.requested_number_of_resources.to_numpy()
    increment = (machine["busy_w"] - machine["idle_w"]) / helpers.J_PER_KWH
    rows = []
    for scoring, path in (("own", task["environmental_trace"]), ("unrelated", task["unrelated_trace"])):
        signal = source.load_signals(path)[task["objective"]]
        change = (
            increment
            * nodes
            * (
                helpers.integral(signal, policy.starting_time.to_numpy(), policy.finish_time.to_numpy())
                - helpers.integral(signal, easy.starting_time.to_numpy(), easy.finish_time.to_numpy())
            )
        )
        total = helpers.physical_totals(easy, machine, {"scored": signal}, HORIZON)["scored_l"]
        rows.append(
            {
                "workload": task["workload"],
                "trace": task["trace"],
                "objective": task["objective"],
                "scoring": scoring,
                "held_jobs": int(held.sum()),
                "other_moved_jobs": int((moved & ~held).sum()),
                "held_pct": 100 * -change[held].sum() / total,
                "others_pct": 100 * -change[~held].sum() / total,
                "net_pct": 100 * -change.sum() / total,
            }
        )
    return rows


def main() -> None:
    """Render the held-jobs figure. Example: python analysis/scripts/held_jobs.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        splits = pd.DataFrame([row for rows in pool.map(split_pair, oracle_tasks()) for row in rows])
    save_held_jobs_figure(splits, arguments.output_dir)
    medians = splits.groupby("scoring")[["held_pct", "others_pct", "net_pct"]].median()
    print(f"{len(splits) // 2} pairs, medians in % of the EASY footprint:\n{medians.round(2).to_string()}")
    print(f"Held-jobs figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
