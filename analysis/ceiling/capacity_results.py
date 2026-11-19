"""Generate T3, the capacity step of the ceiling ladder, for the intensity windows.

T3 asks how much any scheduler could cut the footprint of a Mustang workload
on its machine, for each amount of delay it accepts against EASY. The answer is
a curve of reduction against delay. A scheduling rule, the pricing heuristic
of the campaign or any other, is a point under that curve.

Each point of the curve solves a linear program (LP) on hourly slots:

    variables  x[j, t], the node-seconds job j runs in slot t, from the slot of
               its submission to its EASY finish + MAX_LATENESS_H
    work       the sum over t of x[j, t] is nodes_j x execution_time_j
    width      0 <= x[j, t] <= nodes_j x slot length
    machine    the sum over j of x[j, t] fits in the nodes the context jobs
               leave free in slot t
    objective  minimize the sum of (intensity_t + price x t) x[j, t]

The LP may pause a job, change its width and see the whole trace, which no
real scheduler can. Each of these freedoms only adds schedules, so the LP
reduction bounds that of every rigid schedule with the same delay. The EASY
schedule is one of the LP's choices, so a point with a positive delay never
has a negative reduction.

Delay is the change in the area-weighted response time against EASY, the
measure of greenfilling_weighted_response_time_analysis.py. With each job
weighted by its area, nodes x execution time, the mean response time is the
center of mass of the busy profile, the sum over t of t x busy_t over the total
area, plus a constant of the workload. The delay is then linear in x and
enters the objective as the price term. The lowest price gives the largest
reduction, the highest price the smallest delay, and the prices in between
trace the curve. The price is counted in standard deviations of the hourly
intensity per hour of delay, so it means the same in every signal.

A delay budget, one more constraint row instead of the price, would put every
window at the same delays, but on mustang_slack it took 540 to 1370 s per LP
against 25 to 120 s for the prices. The points of the prices still bound the
curve at any budget B. A point with price p, in the signal's units, and its
delay and best gives best + p x (delay - B) as a floor on the best intensity
within B, so the highest such floor over the points is a strict ceiling, and
the line between two neighboring points is a reachable value.

The machine and the jobs come from the EASY run of the campaign, including the
context jobs, which keep their EASY nodes. EASY ignores the signal, so its
schedule is the same in every window. Facility scenarios that are affine in
the grid signal, every carbon scenario and the dry and continuous water ones,
have the same LP solutions. Only the economizer water needs its own.

Columns of t3_capacity.csv, besides the window labels of T0 to T2:

- delay_price: the price, in standard deviations of the intensity per hour.
- intensity_std: the standard deviation of the hourly intensity over the
  scheduling window, so delay_price x intensity_std is the price in the
  signal's units per hour.
- delay_h: the area-weighted response time change against EASY, in hours.
- at_easy, best: the busy-weighted mean intensity of EASY and of the LP
  schedule, per kWh of compute energy, as at_arrival and best in T2.
- reduction: 1 - best / at_easy, on the same base as T1 and T2. With the
  nodes always on, only the compute minus idle power moves with a job, so the
  facility footprint falls by (compute - idle) / compute of this.

Usage, from the experimental-artifacts repository root, after the campaign
has written experiments/out:

    python analysis/ceiling/capacity_results.py

Writes t3_capacity.csv next to this script.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

import ladder_results as ladder
from ladder_results import scr


SLOT_SECONDS = 3600
INSTANTS_PER_SLOT = SLOT_SECONDS // scr.DELTA_SECONDS
# The pricing campaign finishes jobs up to 243 h after their EASY finish.
MAX_LATENESS_H = 11 * 24
# On PL_2020-07-20, 1e-3 is within 0.1 pp of the largest reduction and 1.0 gives the smallest delay;
# the delay falls from about 80 h to 0 between 2e-3 and 1.5e-2, so the prices are densest there.
DELAY_PRICES = (1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7.5e-3, 1e-2, 1.5e-2, 3e-2, 1.0)
# The LP of mustang_slack, about 2.9 million variables, peaks at 3.4 GB of memory per worker.
WORKERS = 2
EASY_RUN_ROOT = scr.REPOSITORY_ROOT / "experiments" / "out"
T3_OUTPUT = ladder.CEILING_DIR / "t3_capacity.csv"


@dataclass(frozen=True)
class CapacityLP:
    """The constraints of T3 for one EASY schedule. Only the costs change with the signal and the price."""

    slot_seconds: int
    slot_of_variable: np.ndarray
    work_matrix: sp.csr_matrix  # one row per job, summing its variables
    machine_matrix: sp.csr_matrix  # one row per slot, summing the jobs in it
    work: np.ndarray  # node-seconds of each job
    free_capacity: np.ndarray  # node-seconds the context jobs leave free in each slot
    width_limit: np.ndarray  # node-seconds a variable's job can run in one slot
    easy_busy: np.ndarray  # node-seconds EASY runs in each slot

    @property
    def slots(self) -> int:
        return len(self.free_capacity)

    @property
    def slot_centers_h(self) -> np.ndarray:
        return (np.arange(self.slots) + 0.5) * self.slot_seconds / 3600


def load_machine_nodes(workload_name: str) -> int:
    with (scr.WORKLOAD_ROOT / f"{workload_name}.json").open(encoding="utf-8") as source:
        return int(json.load(source)["nb_res"])


def load_easy_schedule(workload_name: str, trace_name: str) -> pd.DataFrame:
    return pd.read_csv(EASY_RUN_ROOT / f"easy_bf_{workload_name}_{trace_name}" / "out_jobs.csv")


def compute_busy_by_slot(schedule: pd.DataFrame, slots: int, slot_seconds: int) -> np.ndarray:
    """Node-seconds the jobs of a schedule run in each slot, from their exact start and finish."""
    start, finish, nodes = (
        schedule[column].to_numpy(dtype=float)
        for column in ("starting_time", "finish_time", "requested_number_of_resources")
    )
    slot_edges = np.arange(slots + 1) * slot_seconds
    node_seconds_until_edge = [(nodes * np.clip(edge - start, 0.0, finish - start)).sum() for edge in slot_edges]
    return np.diff(node_seconds_until_edge)


def build_capacity_lp(
    schedule: pd.DataFrame,
    machine_nodes: int,
    max_lateness_h: float = MAX_LATENESS_H,
    slot_seconds: int = SLOT_SECONDS,
) -> CapacityLP:
    """The T3 constraints for the jobs of an EASY schedule, with its context jobs held in place.

    A job may run from the slot of its submission, even before the submission
    inside that slot, which only loosens the bound.
    """
    is_context = schedule["job_id"].str.startswith("ctx_").to_numpy()
    jobs = schedule[~is_context]
    submission, start, execution, nodes = (
        jobs[column].to_numpy(dtype=float)
        for column in ("submission_time", "starting_time", "execution_time", "requested_number_of_resources")
    )

    first_slot = (submission // slot_seconds).astype(int)
    end_slot = np.ceil((start + execution + max_lateness_h * 3600) / slot_seconds).astype(int)
    slots = int(end_slot.max())
    slots_per_job = end_slot - first_slot

    variables = int(slots_per_job.sum())
    job_of_variable = np.repeat(np.arange(len(jobs)), slots_per_job)
    slot_of_variable = np.concatenate([np.arange(first, end) for first, end in zip(first_slot, end_slot)])
    ones = np.ones(variables)

    return CapacityLP(
        slot_seconds=slot_seconds,
        slot_of_variable=slot_of_variable,
        work_matrix=sp.csr_matrix((ones, (job_of_variable, np.arange(variables))), shape=(len(jobs), variables)),
        machine_matrix=sp.csr_matrix((ones, (slot_of_variable, np.arange(variables))), shape=(slots, variables)),
        work=nodes * execution,
        free_capacity=machine_nodes * slot_seconds - compute_busy_by_slot(schedule[is_context], slots, slot_seconds),
        width_limit=nodes[job_of_variable] * slot_seconds,
        easy_busy=compute_busy_by_slot(jobs, slots, slot_seconds),
    )


def solve_busy_profile(lp: CapacityLP, intensity_by_slot: np.ndarray, price_per_hour: float) -> np.ndarray:
    """Node-seconds per slot of the schedule that minimizes intensity plus the delay price."""
    if len(intensity_by_slot) < lp.slots:
        raise ValueError(f"The LP needs {lp.slots} slots of intensity, the trace has {len(intensity_by_slot)}")

    cost_by_slot = intensity_by_slot[: lp.slots] + price_per_hour * lp.slot_centers_h
    result = linprog(
        cost_by_slot[lp.slot_of_variable],
        A_ub=lp.machine_matrix,
        b_ub=lp.free_capacity,
        A_eq=lp.work_matrix,
        b_eq=lp.work,
        bounds=np.column_stack([np.zeros_like(lp.width_limit), lp.width_limit]),
        method="highs",
    )
    if result.status != 0:
        raise RuntimeError(f"Capacity LP failed: {result.message}")

    return np.bincount(lp.slot_of_variable, weights=result.x, minlength=lp.slots)


def compute_delay_h(lp: CapacityLP, busy_by_slot: np.ndarray) -> float:
    """Area-weighted response time change against EASY: the shift of the center of mass of the busy profile."""
    return float(lp.slot_centers_h @ (busy_by_slot - lp.easy_busy) / lp.work.sum())


def average_by_slot(intensity_by_instant: np.ndarray) -> np.ndarray:
    """Mean intensity of each hourly slot.

    shortcut: the LP takes the hourly mean, as if a job spread evenly over the
    four instants of its hour, which hides the moves inside an hour. On
    PL_2020-07-20 the largest reduction of the hourly LP is 0.04 pp below the
    15-minute one for the jobs of the first two days of mustang_stress, and the
    same to 0.01 pp for all of them; switch to 15-minute slots if a step needs
    that precision.
    """
    slots = len(intensity_by_instant) // INSTANTS_PER_SLOT
    return intensity_by_instant[: slots * INSTANTS_PER_SLOT].reshape(slots, INSTANTS_PER_SLOT).mean(axis=1)


def select_scheduling_window(intensity_by_slot: np.ndarray) -> np.ndarray:
    """The slots of the 28 days in which jobs arrive, the window of the T0 mean."""
    return intensity_by_slot[: ladder.SCHEDULING_INSTANTS // INSTANTS_PER_SLOT]


def standardize(intensity_by_slot: np.ndarray) -> np.ndarray:
    """Intensity in standard deviations from the mean of the scheduling window.

    The LP solution does not change with this transform, since every schedule
    runs the same total work. It makes the delay price comparable across
    signals and lets affine facility scenarios share one solution.
    """
    window = select_scheduling_window(intensity_by_slot)
    return (intensity_by_slot - window.mean()) / window.std()


Frontier = dict[tuple[str, float], np.ndarray]
"""Busy profile of the LP schedule by workload name and delay price, for one signal."""


def build_window_lps(window: pd.Series) -> dict[str, CapacityLP]:
    """One LP per workload, from the EASY run of the campaign in this window."""
    trace_name = Path(window.file).stem
    return {
        workload_name: build_capacity_lp(load_easy_schedule(workload_name, trace_name), load_machine_nodes(workload_name))
        for workload_name in ladder.WORKLOAD_NAMES
    }


def solve_frontier(lps: dict[str, CapacityLP], standardized_by_slot: np.ndarray) -> Frontier:
    """Busy profile of every workload at every delay price, for one standardized signal."""
    return {
        (workload_name, price): solve_busy_profile(lp, standardized_by_slot, price)
        for workload_name, lp in lps.items()
        for price in DELAY_PRICES
    }


def reuse_or_solve_frontier(
    lps: dict[str, CapacityLP],
    standardized_by_slot: np.ndarray,
    solved_frontiers: list[tuple[np.ndarray, Frontier]],
) -> Frontier:
    """The frontier of a signal, reusing one already solved for the same standardized signal.

    A facility scenario that is affine in another signal standardizes to the
    same values and has the same LP solutions, so it skips the solver. New
    frontiers are appended to solved_frontiers.
    """
    for known_standardized, frontier in solved_frontiers:
        if np.allclose(known_standardized, standardized_by_slot):
            return frontier

    frontier = solve_frontier(lps, standardized_by_slot)
    solved_frontiers.append((standardized_by_slot, frontier))
    return frontier


def summarize_frontier(
    lps: dict[str, CapacityLP], frontier: Frontier, signal: str, intensity_by_slot: np.ndarray
) -> list[dict]:
    """One row per workload and price, with the delay and the intensities in the signal's own units."""
    intensity_std = select_scheduling_window(intensity_by_slot).std()
    rows = []
    for (workload_name, price), busy_by_slot in frontier.items():
        lp = lps[workload_name]
        intensity = intensity_by_slot[: lp.slots]
        at_easy = intensity @ lp.easy_busy / lp.work.sum()
        best = intensity @ busy_by_slot / lp.work.sum()
        rows.append(dict(
            workload=workload_name,
            signal=signal,
            delay_price=price,
            intensity_std=intensity_std,
            delay_h=compute_delay_h(lp, busy_by_slot),
            at_easy=at_easy,
            best=best,
            reduction=1.0 - best / at_easy,
        ))

    return rows


def compute_capacity_ceiling(window: pd.Series) -> list[dict]:
    """T3 rows of one row of windows.csv, one per facility, signal, workload and delay price."""
    lps = build_window_lps(window)
    solved_frontiers: list[tuple[np.ndarray, Frontier]] = []

    rows = []
    for facility in ladder.FACILITIES:
        trace = scr.load_trace(window, facility=None if facility == "grid" else facility)
        labels = dict(zone=trace.zone, intensity_trace=trace.name, season=trace.season,
                      year=trace.year, facility=facility)

        for signal in ladder.SIGNALS:
            intensity_by_slot = average_by_slot(getattr(trace, signal))
            frontier = reuse_or_solve_frontier(lps, standardize(intensity_by_slot), solved_frontiers)
            rows += [{**labels, **row} for row in summarize_frontier(lps, frontier, signal, intensity_by_slot)]

    print(window.file, flush=True)
    return rows


def main() -> None:
    # Rows as Series, since the named tuples of itertuples cannot be sent to the workers.
    windows = [window for _, window in scr.load_windows().iterrows()]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = [row for window_rows in pool.map(compute_capacity_ceiling, windows) for row in window_rows]
    ladder.write_rows(rows, T3_OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
