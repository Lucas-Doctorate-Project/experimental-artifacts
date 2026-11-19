"""Split each policy's saving into jobs delayed on purpose and all other jobs: python analysis/scripts/held_jobs.py."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from delayed_jobs import delayed_on_purpose, easy_emissions, emission_change, load_pair, policy_tasks
from held_jobs_plots import save_held_jobs_figure

WORKERS = 8


def split_pair(task: dict) -> list[dict]:
    """Saving of delayed and other jobs in one run, scored on real and unrelated intensity data."""
    machine, easy, policy = load_pair(task)
    delayed = easy.job_id.isin(delayed_on_purpose(task).index).to_numpy()
    node_hours = (easy.requested_number_of_resources * easy.execution_time).to_numpy()
    rows = []
    for scoring, path in (("real", task["environmental_trace"]), ("unrelated", task["unrelated_trace"])):
        signal = helpers.load_signals(path)["carbon"]
        change = emission_change(easy, policy, machine, signal)
        total = easy_emissions(easy, machine, signal)
        rows.append(
            {
                "policy": task["policy"],
                "workload": task["workload"],
                "trace": task["trace"],
                "scoring": scoring,
                "delayed_jobs": int(delayed.sum()),
                "delayed_job_pct": 100 * delayed.mean(),
                "delayed_node_hour_pct": 100 * node_hours[delayed].sum() / node_hours.sum(),
                "delayed_pct": 100 * -change[delayed].sum() / total,
                "others_pct": 100 * -change[~delayed].sum() / total,
                "net_pct": 100 * -change.sum() / total,
            }
        )
    return rows


def main() -> None:
    """Render the delayed-jobs figure. Example: python analysis/scripts/held_jobs.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        splits = pd.DataFrame([row for rows in pool.map(split_pair, policy_tasks()) for row in rows])
    gap = (splits.delayed_pct + splits.others_pct - splits.net_pct).abs().max()
    if gap > 1e-9:
        raise ValueError(f"delayed and other jobs do not add up to the saving (gap {gap:g} points)")
    save_held_jobs_figure(splits, arguments.output_dir)
    columns = ["delayed_pct", "others_pct", "net_pct", "delayed_jobs", "delayed_job_pct", "delayed_node_hour_pct"]
    medians = splits.groupby(["policy", "scoring"])[columns].median()
    print(f"{len(splits) // 4} cases, medians (savings in % of total emissions):\n{medians.round(2).to_string()}")
    print(f"Delayed-jobs figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
