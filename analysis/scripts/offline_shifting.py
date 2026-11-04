"""Move EASY jobs into cleaner free capacity after the fact: python analysis/scripts/offline_shifting.py."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from offline_shifting_plots import save_offline_shifting_figure
from opportunity import BASELINE_GROUP, realized_points
from oracle import ORACLE_GROUP, OracleSource
from results_metrics import HORIZONS

GROUPS = ("original", "resampled")
BUDGET_HOURS = (3, 6, 12, 24)
GRID_SECONDS = 60
MAX_PASSES = 20
MIN_RELATIVE_GAIN = 1e-9
WORKERS = 8


def start_candidates(signal: dict, start: float, duration: float, budget: float, horizon: int) -> np.ndarray:
    """Starts where a job's footprint can change slope, from its EASY start through the budget."""
    upper = min(start + budget, horizon - duration)
    times = signal["times"]
    starts = times[np.searchsorted(times, start, side="left") : np.searchsorted(times, upper, side="right")]
    ends = times[
        np.searchsorted(times, start + duration, side="left") : np.searchsorted(
            times, upper + duration, side="right"
        )
    ]
    return np.unique(np.r_[start, upper, starts, ends - duration])


def grid_span(start: float, duration: float) -> tuple[int, int]:
    return int(start // GRID_SECONDS), int(np.ceil((start + duration) / GRID_SECONDS))


def peak_nodes(starts: np.ndarray, durations: np.ndarray, nodes: np.ndarray) -> int:
    """Exact peak allocation. At equal times, a finish frees nodes before a start takes them."""
    times = np.r_[starts, starts + durations]
    changes = np.r_[nodes, -nodes]
    return int(np.cumsum(changes[np.lexsort((changes, times))]).max())


def shift_jobs(
    jobs: pd.DataFrame, signal: dict, machine: dict, horizon: int, budget_hours: float
) -> tuple[np.ndarray, int]:
    """Greedy delay-only repair of a finished schedule. Example: shift_jobs(jobs, signal, machine, 3196800, 24).

    Jobs with the largest remaining gain move first, each to its cleanest start
    with enough free nodes on a grid that counts every touched minute as
    occupied. Budgets stay relative to the EASY start. Context jobs stay fixed.
    Passes repeat because each move can free capacity for another job.
    """
    easy_starts = jobs.starting_time.to_numpy()
    durations = jobs.execution_time.to_numpy()
    nodes = jobs.requested_number_of_resources.to_numpy()
    starts = easy_starts.copy()
    slots = int(np.ceil(horizon / GRID_SECONDS)) + 1
    changes = np.zeros(slots + 1)
    np.add.at(changes, (starts // GRID_SECONDS).astype(int), nodes)
    np.add.at(changes, np.ceil((starts + durations) / GRID_SECONDS).astype(int), -nodes)
    occupied = np.cumsum(changes)[:slots]
    shiftable = np.flatnonzero(~jobs.is_context.to_numpy() & (durations > 0))
    candidates, costs = {}, {}
    for i in shiftable:
        candidates[i] = start_candidates(signal, easy_starts[i], durations[i], budget_hours * 3600, horizon)
        costs[i] = nodes[i] * helpers.integral(signal, candidates[i], candidates[i] + durations[i])
    best = np.array([costs[i].min() for i in shiftable])
    for passes in range(1, MAX_PASSES + 1):
        current = nodes[shiftable] * helpers.integral(
            signal, starts[shiftable], starts[shiftable] + durations[shiftable]
        )
        moves = 0
        for position in np.argsort(best - current):
            i = shiftable[position]
            limit = current[position] * (1 - MIN_RELATIVE_GAIN)
            if best[position] >= limit:
                break
            first, last = grid_span(starts[i], durations[i])
            occupied[first:last] -= nodes[i]
            for choice in np.argsort(costs[i], kind="stable"):
                if costs[i][choice] >= limit:
                    occupied[first:last] += nodes[i]
                    break
                first_new, last_new = grid_span(candidates[i][choice], durations[i])
                if occupied[first_new:last_new].max() + nodes[i] <= machine["nodes"]:
                    occupied[first_new:last_new] += nodes[i]
                    starts[i] = candidates[i][choice]
                    moves += 1
                    break
            else:
                occupied[first:last] += nodes[i]
        if moves == 0:
            break
    peak = peak_nodes(starts, durations, nodes)
    if peak > machine["nodes"]:
        raise ValueError(f"shifted schedule uses {peak} nodes on a {machine['nodes']}-node machine")
    return starts, passes


def shift_run(task: dict) -> list[dict]:
    """Shift one EASY run for each objective and budget, reporting EASY and shifted metrics."""
    group, horizon = task["group"], HORIZONS[task["group"]]
    machine = helpers.platform(task["platform"])
    jobs = helpers.load_jobs(group, task["name"], machine["nodes"])
    signals = helpers.load_signals(task["environmental_trace"])
    easy_service = helpers.performance(jobs)
    replay = ~jobs.is_context.to_numpy()
    rows = []
    for objective, signal in signals.items():
        selected = {objective: signal}
        key = f"{objective}_{'g' if objective == 'carbon' else 'l'}"
        easy_total = helpers.physical_totals(jobs, machine, selected, horizon)[key]
        bounds = helpers.relaxed_opportunity(
            jobs, signal, machine, horizon, objective, BUDGET_HOURS, allow_advance=False
        ).set_index("delay_hours")
        for budget in BUDGET_HOURS:
            starts, passes = shift_jobs(jobs, signal, machine, horizon, budget)
            shifted = jobs.assign(
                starting_time=starts,
                finish_time=starts + jobs.execution_time,
                waiting_time=starts - jobs.submission_time,
            )
            service = helpers.performance(shifted)
            added_wait = starts[replay] - jobs.starting_time.to_numpy()[replay]
            rows.append(
                {
                    "group": group,
                    "workload": task["workload"],
                    "trace": task["trace"],
                    "objective": objective,
                    "budget_hours": budget,
                    "saving_pct": 100
                    * (1 - helpers.physical_totals(shifted, machine, selected, horizon)[key] / easy_total),
                    "bound_saving_pct": bounds.savings_pct[float(budget)],
                    "added_wait_hours": added_wait.mean() / 3600,
                    "moved_jobs": int(np.count_nonzero(added_wait)),
                    "replay_jobs": int(replay.sum()),
                    "passes": passes,
                    "easy_bounded_slowdown": easy_service["bounded_slowdown"],
                    "bounded_slowdown": service["bounded_slowdown"],
                    "easy_weighted_response_seconds": easy_service["weighted_response_seconds"],
                    "weighted_response_seconds": service["weighted_response_seconds"],
                }
            )
    return rows


def campaign_tasks(groups: tuple[str, ...] = GROUPS) -> list[dict]:
    """One task per EASY run. Example: campaign_tasks(('original',))."""
    tasks = []
    for group in groups:
        config = helpers.experiments(group)
        columns = ["group", "name", "workload", "trace", "platform", "environmental_trace"]
        tasks += config.loc[config.variant == "easy_bf", columns].to_dict("records")
    return tasks


def main() -> None:
    """Shift every EASY run. Example: python analysis/scripts/offline_shifting.py --output-dir /tmp/shifting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "offline-shifting",
        help="new directory for results.csv (default: analysis/outputs/offline-shifting)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="render the figure here from an existing results.csv instead of shifting jobs",
    )
    arguments = parser.parse_args()
    if arguments.figure_dir:
        source = OracleSource()
        points = pd.concat(
            [
                realized_points(BASELINE_GROUP, BASELINE_GROUP, source),
                realized_points(ORACLE_GROUP, BASELINE_GROUP, source),
            ],
            ignore_index=True,
        )
        results = pd.read_csv(arguments.output_dir / "results.csv")
        save_offline_shifting_figure(results, points, arguments.figure_dir)
        print(f"Offline shifting figure saved to {arguments.figure_dir.resolve()}")
        return
    arguments.output_dir.mkdir(parents=True)
    tasks = campaign_tasks()
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = [row for result in pool.map(shift_run, tasks) for row in result]
    pd.DataFrame(rows).to_csv(arguments.output_dir / "results.csv", index=False)
    print(f"{len(rows)} results from {len(tasks)} EASY runs saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
