"""Compare the saving each policy expected from its delayed jobs with the real and best possible saving.

python analysis/scripts/held_walltime.py
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from delayed_jobs import LIMIT_SECONDS, delayed_on_purpose, easy_emissions, load_pair, policy_tasks
from held_walltime_plots import SHARES, save_held_walltime_figure

GRID_SECONDS = 900
WORKERS = 8


def avoided(signal: dict, decided: np.ndarray, begin: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Intensity integral avoided over `length` by starting at `begin` instead of `decided`."""
    return helpers.integral(signal, decided, decided + length) - helpers.integral(signal, begin, begin + length)


def best_avoided(signal: dict, decided: np.ndarray, begin: np.ndarray, runtime: np.ndarray) -> np.ndarray:
    """Most intensity integral avoidable for the actual run time within the 6 h limit, ignoring capacity.

    Candidates are the decision date, every 15-minute grid point up to the limit,
    and the start the policy chose, so the best is never below the real value.
    """
    first = np.floor(decided / GRID_SECONDS) * GRID_SECONDS
    grid = first[:, None] + GRID_SECONDS * np.arange(1, LIMIT_SECONDS // GRID_SECONDS + 2)
    grid = np.where(grid <= decided[:, None] + LIMIT_SECONDS, grid, decided[:, None])
    candidates = np.column_stack([decided, begin, grid])
    return avoided(signal, decided[:, None], candidates, runtime[:, None]).max(axis=1)


def delayed_contributions(task: dict) -> pd.DataFrame:
    """Expected, real, and best possible saving of each delayed replay job, in % of total emissions.

    Each job is compared with itself started at its decision date, the start it
    would have had without the delay. Real uses the actual run time at the chosen
    start. Expected applies the reduction in carbon intensity the scheduler saw
    over the requested time to the job's emissions over its actual run time, so
    all three values describe the same energy. Idle power during the wait is left out.
    """
    machine, easy, policy = load_pair(task)
    signal = helpers.load_signals(task["environmental_trace"])["carbon"]
    scale = 100 * (machine["busy_w"] - machine["idle_w"]) / helpers.J_PER_KWH / easy_emissions(easy, machine, signal)
    chosen = delayed_on_purpose(task)
    chosen = chosen.loc[~chosen.index.str.startswith("ctx_")]
    jobs = policy.set_index("job_id").loc[chosen.index]
    decided, begin = chosen.decided.to_numpy(), chosen.begin.to_numpy()
    requested, runtime = jobs.requested_time.to_numpy(), jobs.execution_time.to_numpy()
    nodes = jobs.requested_number_of_resources.to_numpy()
    expected_reduction = avoided(signal, decided, begin, requested) / helpers.integral(
        signal, decided, decided + requested
    )
    unshifted = helpers.integral(signal, decided, decided + runtime)
    return pd.DataFrame(
        {
            "policy": task["policy"],
            "workload": task["workload"],
            "trace": task["trace"],
            "node_hours": nodes * runtime / 3600,
            "used_share": runtime / requested,
            "expected": scale * nodes * expected_reduction * unshifted,
            "real": scale * nodes * avoided(signal, decided, begin, runtime),
            "best": scale * nodes * best_avoided(signal, decided, begin, runtime),
        }
    )


def share_class(used_share: pd.Series) -> pd.Series:
    """Label of the share of requested time used. Example: share_class(jobs.used_share)."""
    labels = pd.Series(pd.NA, index=used_share.index, dtype="object")
    for low, high, label in SHARES:
        labels[(used_share > low) & (used_share <= high)] = label
    return labels


def main() -> None:
    """Render the requested-time figure. Example: python analysis/scripts/held_walltime.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = pd.concat(pool.map(delayed_contributions, policy_tasks()), ignore_index=True)
    jobs["share_class"] = share_class(jobs.used_share)
    if jobs.share_class.isna().any():
        raise ValueError(f"{jobs.share_class.isna().sum()} delayed jobs fall outside every share class")
    save_held_walltime_figure(jobs, arguments.output_dir)
    runs = jobs.groupby(["policy", "workload", "trace"])[["expected", "real", "best"]].sum()
    print(f"{len(jobs)} delayed replay jobs; mean saving per run in % of total emissions:")
    print(runs.groupby("policy").mean().round(3).to_string())
    print(f"Requested-time figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
