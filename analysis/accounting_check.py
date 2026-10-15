"""Recompute a run's footprint outside Batsim and compare it with the export.

The campaign analysis gets idle as a residual (platform total minus the jobs),
so a small inconsistency in either term becomes a large idle change. This
script recomputes both terms from out_jobs.csv, the trace and the platform,
assuming always-on nodes, PUE 1 and no embodied impact, as in the campaign.

Usage: python accounting_check.py <run_dir> <trace.csv> <platform.xml> [<other_run_dir>]
With two runs of the same window it also checks the conservation identity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import shift_ceiling_results as scr

SIGNALS = {"carbon": "total_carbon_operational", "water": "total_water_offsite"}


def load_trace(trace_path: Path):
    # load_trace/load_platform join their argument to a repository root; an absolute path wins the join.
    return scr.load_trace(SimpleNamespace(file=str(trace_path.resolve()), zone="AS0", season="-", year=0))


def load_platform(platform_path: Path):
    return scr.load_platform(str(platform_path.resolve().with_suffix("")))


def window_integral(values: np.ndarray, prefix: np.ndarray, begin, end) -> np.ndarray:
    """Integral of the intensity over [begin, end], elementwise for arrays of bounds."""
    return scr.integral_at(np.asarray(end, float), values, prefix) - scr.integral_at(
        np.asarray(begin, float), values, prefix
    )


def energy_report(jobs: pd.DataFrame, schedule: pd.Series, platform, makespan: float) -> dict:
    """Energy exported by Batsim versus recomputed from busy and idle node-seconds up to the makespan."""
    nodes = jobs["requested_number_of_resources"].to_numpy(float)
    runtimes = jobs["finish_time"].to_numpy(float) - jobs["starting_time"].to_numpy(float)
    busy_s = float(np.sum(nodes * runtimes))
    idle_s = int(schedule["nb_computing_machines"]) * makespan - busy_s
    offline_joules = platform.compute_power_w * busy_s + platform.idle_power_w * idle_s
    return {
        "energy_export_kwh": schedule["consumed_joules"] / 3.6e6,
        "energy_offline_to_makespan_kwh": offline_joules / 3.6e6,
    }


def signal_report(
    signal: str,
    values: np.ndarray,
    jobs: pd.DataFrame,
    schedule: pd.Series,
    platform,
    makespan: float,
    trace_end: float,
) -> dict:
    """Exported versus recomputed compute, idle and total footprint for one signal (carbon or water)."""
    nodes = jobs["requested_number_of_resources"].to_numpy(float)
    prefix = scr.prefix_integral(values)
    nb_nodes = int(schedule["nb_computing_machines"])
    exported_total = float(schedule[SIGNALS[signal]])
    exported_per_job = jobs[f"consumed_{signal}"]

    # Units: W x (intensity x h) / 1000 gives g for carbon and L for water.
    starts = jobs["starting_time"].to_numpy(float)
    finishes = jobs["finish_time"].to_numpy(float)
    job_node_integral = nodes * window_integral(values, prefix, starts, finishes)
    offline_per_job = platform.compute_power_w * job_node_integral / 1000
    offline_compute = offline_per_job.sum()

    def offline_idle(horizon: float) -> float:
        platform_node_integral = nb_nodes * float(window_integral(values, prefix, 0.0, horizon))
        return platform.idle_power_w * (platform_node_integral - job_node_integral.sum()) / 1000

    return {
        f"{signal}_job_max_abs_error": float(np.max(np.abs(exported_per_job.to_numpy() - offline_per_job))),
        f"{signal}_compute_export": exported_per_job.sum(),
        f"{signal}_compute_offline": offline_compute,
        f"{signal}_total_export": exported_total,
        f"{signal}_total_offline_to_trace_end": offline_compute + offline_idle(trace_end),
        f"{signal}_total_offline_to_makespan": offline_compute + offline_idle(makespan),
        f"{signal}_idle_residual_export": exported_total - exported_per_job.sum(),
        f"{signal}_idle_offline_to_makespan": offline_idle(makespan),
        f"{signal}_idle_offline_to_trace_end": offline_idle(trace_end),
    }


def check_run(run_dir: Path, trace_path: Path, platform_path: Path) -> dict:
    """Exported versus recomputed energy and footprint for one run."""
    trace = load_trace(trace_path)
    platform = load_platform(platform_path)
    jobs = pd.read_csv(run_dir / "out_jobs.csv")
    schedule = pd.read_csv(run_dir / "out_schedule.csv").iloc[0]

    makespan = float(schedule["makespan"])
    # The trace's update actors keep Batsim alive until the last timestamp, so host counters run to it.
    trace_end = (len(trace.carbon) - 1) * scr.DELTA_SECONDS

    report = {"run": run_dir.name, "makespan_h": makespan / 3600, "trace_end_h": trace_end / 3600}
    report |= energy_report(jobs, schedule, platform, makespan)
    for signal in SIGNALS:
        report |= signal_report(signal, getattr(trace, signal), jobs, schedule, platform, makespan, trace_end)
    report["idle_power_ratio"] = platform.idle_power_w / platform.compute_power_w
    return report


def conservation(first: dict, second: dict) -> dict:
    """On a common horizon, idle must change by -(P_idle/P_compute) times the compute change."""
    ratio = first["idle_power_ratio"]
    result = {}
    for signal in SIGNALS:
        delta_compute = second[f"{signal}_compute_export"] - first[f"{signal}_compute_export"]
        result[f"{signal}_delta_compute"] = delta_compute
        result[f"{signal}_delta_idle_expected"] = -ratio * delta_compute
        result[f"{signal}_delta_idle_export_residual"] = (
            second[f"{signal}_idle_residual_export"] - first[f"{signal}_idle_residual_export"]
        )
    return result


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    trace_path, platform_path = Path(argv[1]), Path(argv[2])
    reports = [check_run(Path(run), trace_path, platform_path) for run in [argv[0], *argv[3:]]]
    pd.set_option("display.float_format", "{:.6g}".format)
    print(pd.DataFrame(reports).set_index("run").T.to_string())
    if len(reports) == 2:
        print(pd.Series(conservation(*reports)).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
