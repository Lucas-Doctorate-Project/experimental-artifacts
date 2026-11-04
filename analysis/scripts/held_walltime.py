"""Compare the intensity drop the oracle targeted with the drop its held jobs got: python analysis/scripts/held_walltime.py."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from held_jobs import last_choices
from held_walltime_plots import save_held_walltime_figure
from oracle import ORACLE_GROUP, OracleSource
from queue_gain_loss import PLANNING_HORIZON_SECONDS, oracle_tasks

WINDOW_STEP_SECONDS = 900
WORKERS = 8


def mean_drop(signal: dict, decided: np.ndarray, begin: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Relative drop in mean intensity over `length` when starting at `begin` instead of `decided`."""
    now = helpers.integral(signal, decided, decided + length)
    return (now - helpers.integral(signal, begin, begin + length)) / now


def held_job_drops(task: dict) -> pd.DataFrame:
    """Target, got, and best intensity drop of each held replay job in one oracle run.

    Each held job is compared with itself started at the decision date, which is
    when it would have started without the hold. Target uses the requested
    walltime, as the oracle does, and got uses the real runtime. Best is the
    cleanest start for the real runtime on the oracle's candidate grid, ignoring
    capacity. Idle power during the wait is left out.
    """
    source = OracleSource()
    machine = source.platform(task["platform"])
    jobs = source.load_jobs(ORACLE_GROUP, task["name"], machine["nodes"]).set_index("job_id")
    signal = source.load_signals(task["environmental_trace"])[task["objective"]]
    choices = pd.DataFrame.from_dict(last_choices(task["name"]), orient="index", columns=["decided", "begin"])
    held = choices.loc[(choices.begin > choices.decided + 1e-6) & ~choices.index.str.startswith("ctx_")]
    jobs = jobs.loc[held.index]
    decided, begin = held.decided.to_numpy(), held.begin.to_numpy()
    walltime, runtime = jobs.requested_time.to_numpy(), jobs.execution_time.to_numpy()
    candidates = decided[:, None] + WINDOW_STEP_SECONDS * np.arange(PLANNING_HORIZON_SECONDS // WINDOW_STEP_SECONDS + 1)
    return pd.DataFrame(
        {
            "workload": task["workload"],
            "objective": task["objective"],
            "trace": task["trace"],
            "work": jobs.requested_number_of_resources.to_numpy() * runtime,
            "runtime_share": runtime / walltime,
            "target": mean_drop(signal, decided, begin, walltime),
            "got": mean_drop(signal, decided, begin, runtime),
            "best": mean_drop(signal, decided[:, None], candidates, runtime[:, None]).max(axis=1),
        }
    )


def main() -> None:
    """Render the held-walltime figure. Example: python analysis/scripts/held_walltime.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        drops = pd.concat(pool.map(held_job_drops, oracle_tasks()), ignore_index=True)
    save_held_walltime_figure(drops, arguments.output_dir)
    columns = ["target", "got", "best"]
    weighted = {column: np.average(drops[column], weights=drops.work) for column in columns}
    print(f"{len(drops)} held replay jobs, relative intensity drop in %:")
    print(f"  median per job:  {', '.join(f'{c} {100 * drops[c].median():.1f}' for c in columns)}")
    print(f"  weighted by work: {', '.join(f'{c} {100 * v:.1f}' for c, v in weighted.items())}")
    print(f"Held-walltime figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
