"""Generate the first steps of the ceiling ladder for the intensity windows.

T0, inherent footprint. The footprint of a workload if timing did not matter:
its IT energy times the mean intensity of the 28-day scheduling window. It is
the reference that turns the other steps into absolute values.

T1, signal ceiling, after Sukprasert et al. (EuroSys 2024, Sec. 3.2). A lone
job arrives at every instant of the scheduling window, and the ceiling is the
cleanest placement it can reach. Workload, capacity and queue are ignored.

    deferrable     runs as one block that starts anywhere in
                   [arrival, arrival + slack]
    interruptible  pauses at no cost and runs in its cleanest instants before
                   the same deadline, arrival + job length + slack

The reduction is 1 - sum(best) / sum(at arrival) over the arrivals: the
footprint reduction of the whole set of jobs, not the mean of the per-job
reductions.

Both steps cover carbon and water, on the grid trace alone ("grid", the
campaign's PUE = 1 and WUE = 0) and on the three facility scenarios of
intensities/facility, where the intensities are per kWh of IT energy.

Terms used throughout:

- instant: one 15-minute step of an intensity trace. Job lengths and slacks
  are counted in instants unless the name ends in _h (hours).
- arrival: the instant a job is submitted. Without shifting, it runs from
  there.
- job length: how many instants the job runs, L in the paper.
- slack: how many instants the job may wait after its arrival, S in the paper
  and the horizon H in the campaign.
- placement: the instants a job runs in. Power is constant while it runs, so
  the footprint per kWh of a placement is the mean intensity over it.

Usage, from the experimental-artifacts repository root:

    python analysis/ceiling/ladder_results.py

Writes t0_inherent.csv and t1_signal.csv next to this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


CEILING_DIR = Path(__file__).resolve().parent
# analysis/ is not an importable package, so load the shared loaders from their location.
sys.path.insert(0, str(CEILING_DIR.parent))
import shift_ceiling_results as scr  # noqa: E402


INSTANTS_PER_HOUR = 3600 // scr.DELTA_SECONDS
# Jobs arrive in the first 28 days; the rest of the extract only holds late jobs.
SCHEDULING_INSTANTS = 28 * 24 * INSTANTS_PER_HOUR
WORKLOAD_NAMES = ("mustang_slack", "mustang_stress")
SIGNALS = ("carbon", "water")
FACILITIES = ("grid", *scr.FACILITY_SCENARIOS)
LENGTH_HOURS = (0.25, 1, 3, 6, 12, 16, 24)
SLACK_HOURS = (6, 12, 24)  # the horizons H of the campaign
T0_OUTPUT = CEILING_DIR / "t0_inherent.csv"
T1_OUTPUT = CEILING_DIR / "t1_signal.csv"


def convert_hours_to_instants(hours: float) -> int:
    return int(round(hours * INSTANTS_PER_HOUR))


def compute_mean_by_start(intensity_by_instant: np.ndarray, job_length: int) -> np.ndarray:
    """Mean intensity of a block of job_length instants, for every start instant.

    Entry s is the mean over [s, s + job_length). Each block sum is the
    difference of two entries of the cumulative sum, so a mean costs one
    subtraction whatever the job length.
    """
    cumulative_intensity = np.concatenate(([0.0], np.cumsum(intensity_by_instant)))
    block_sums = cumulative_intensity[job_length:] - cumulative_intensity[:-job_length]
    return block_sums / job_length


def find_best_deferrable_placement(
    intensity_by_instant: np.ndarray,
    arrival_instants: np.ndarray,
    job_length: int,
    slack: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity at arrival and at the best start, for each arrival.

    The job runs as one block and starts at its arrival or up to slack instants
    later. Each pass of the loop tries the same delay for every arrival at once,
    so the loop has only `slack` passes.
    """
    mean_by_start = compute_mean_by_start(intensity_by_instant, job_length)
    mean_at_arrival = mean_by_start[arrival_instants]

    best_mean = mean_at_arrival.copy()
    for delay in range(1, slack + 1):
        np.minimum(best_mean, mean_by_start[arrival_instants + delay], out=best_mean)

    return mean_at_arrival, best_mean


def find_best_interruptible_placement(
    intensity_by_instant: np.ndarray,
    arrival_instants: np.ndarray,
    job_length: int,
    slack: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean intensity at arrival and over the cleanest reachable instants, for each arrival.

    The job pauses and resumes at no cost, so it may run in any job_length
    instants before the deferrable deadline, arrival + job_length + slack. The
    contiguous blocks of the deferrable job are among these choices, so this
    placement is never worse. np.partition moves the job_length lowest
    intensities of each row to its front without sorting the whole row.
    """
    # One row per arrival, holding the intensity of every instant the job can reach (a view, not a copy).
    reachable_intensities = sliding_window_view(intensity_by_instant, job_length + slack)[arrival_instants]
    cleanest_intensities = np.partition(reachable_intensities, job_length - 1, axis=1)[:, :job_length]
    mean_at_arrival = compute_mean_by_start(intensity_by_instant, job_length)[arrival_instants]

    return mean_at_arrival, cleanest_intensities.mean(axis=1)


PLACEMENT_FINDERS = {
    "deferrable": find_best_deferrable_placement,
    "interruptible": find_best_interruptible_placement,
}


def compute_inherent_footprint(trace: scr.IntensityTrace, workloads: list[scr.Workload]) -> list[dict]:
    """T0 rows, one per signal and workload, with the footprint in kg CO2eq or L.

    The energy is that of the jobs at compute power, the part of the footprint
    that the higher steps shift in time.
    """
    rows = []
    for signal in SIGNALS:
        mean_intensity = getattr(trace, signal)[:SCHEDULING_INSTANTS].mean()
        unit_factor = 1e-3 if signal == "carbon" else 1.0  # g/kWh gives g, reported in kg; L/kWh gives L

        for workload in workloads:
            node_seconds = float(np.sum(workload.nodes * workload.execution_time_s))
            energy_kwh = node_seconds * workload.platform.compute_power_w / 3.6e6
            rows.append(dict(
                workload=workload.name,
                signal=signal,
                mean_intensity=mean_intensity,
                energy_kwh=energy_kwh,
                footprint=energy_kwh * mean_intensity * unit_factor,
            ))

    return rows


def compute_signal_ceiling(trace: scr.IntensityTrace, signal: str) -> list[dict]:
    """T1 rows, one per job length, slack and mode, with mean intensities per kWh."""
    intensity_by_instant = getattr(trace, signal)

    # The last arrival, waiting the longest slack, must still finish the longest job inside the extract.
    longest_job = convert_hours_to_instants(max(LENGTH_HOURS))
    longest_slack = convert_hours_to_instants(max(SLACK_HOURS))
    required_instants = SCHEDULING_INSTANTS + longest_job + longest_slack - 1
    if len(intensity_by_instant) < required_instants:
        raise ValueError(
            f"{trace.name} has {len(intensity_by_instant)} instants, the signal ceiling needs {required_instants}"
        )

    arrival_instants = np.arange(SCHEDULING_INSTANTS)
    rows = []
    for length_h in LENGTH_HOURS:
        for slack_h in SLACK_HOURS:
            for mode, find_best_placement in PLACEMENT_FINDERS.items():
                mean_at_arrival, best_mean = find_best_placement(
                    intensity_by_instant,
                    arrival_instants,
                    convert_hours_to_instants(length_h),
                    convert_hours_to_instants(slack_h),
                )
                rows.append(dict(
                    signal=signal,
                    length_h=length_h,
                    slack_h=slack_h,
                    mode=mode,
                    at_arrival=mean_at_arrival.mean(),
                    best=best_mean.mean(),
                    reduction=1.0 - best_mean.sum() / mean_at_arrival.sum(),
                ))

    return rows


def compute_window_rows(window: tuple, workloads: list[scr.Workload]) -> tuple[list[dict], list[dict]]:
    """T0 and T1 rows of one row of windows.csv, for every facility scenario."""
    inherent_rows, signal_rows = [], []
    for facility in FACILITIES:
        trace = scr.load_trace(window, facility=None if facility == "grid" else facility)
        labels = dict(zone=trace.zone, intensity_trace=trace.name, season=trace.season,
                      year=trace.year, facility=facility)

        inherent_rows += [{**labels, **row} for row in compute_inherent_footprint(trace, workloads)]

        for signal in SIGNALS:
            signal_rows += [{**labels, **row} for row in compute_signal_ceiling(trace, signal)]

    return inherent_rows, signal_rows


def write_rows(rows: list[dict], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")
    print(f"Wrote {len(rows):,} rows to {path.name}")


def main() -> None:
    workloads = [scr.load_workload(name) for name in WORKLOAD_NAMES]

    inherent_rows, signal_rows = [], []
    for window in scr.load_windows().itertuples(index=False):
        window_inherent_rows, window_signal_rows = compute_window_rows(window, workloads)
        inherent_rows += window_inherent_rows
        signal_rows += window_signal_rows
        print(window.file, flush=True)

    write_rows(inherent_rows, T0_OUTPUT)
    write_rows(signal_rows, T1_OUTPUT)


if __name__ == "__main__":
    raise SystemExit(main())
