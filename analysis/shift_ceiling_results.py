"""Generate the job-level temporal-shifting ceiling results.

The analysis crosses the four raw replay workloads with the 36 sampled
intensity windows. For every non-context job it finds an independent optimum
for carbon and water at each displacement horizon. Capacity and queue
constraints are intentionally ignored, so the result is an optimistic ceiling.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import TextIO
import uuid
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


DELTA_SECONDS = 900
BOUNDED_SLOWDOWN_FLOOR_SECONDS = 10.0
SHIFT_HOURS = (0, 1, 3, 6, 12, 24)
WORKLOAD_NAMES = (
    "mustang_stress",
    "mustang_slack",
    "trinity_stress",
    "trinity_slack",
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_ROOT = REPOSITORY_ROOT / "workloads and platforms"
TRACE_ROOT = REPOSITORY_ROOT / "intensities" / "traces"
WINDOWS_PATH = REPOSITORY_ROOT / "intensities" / "windows.csv"
DEFAULT_OUTPUT = Path(__file__).with_name("shift_ceiling_results.csv.gz")

OUTPUT_COLUMNS = [
    "workload",
    "platform",
    "zone",
    "intensity_trace",
    "season",
    "year",
    "job_id",
    "requested_time_s",
    "execution_time_s",
    "nodes",
    "submission_time_s",
    "d_m_label",
    "d_m_seconds",
    "signal",
    "t_star_s",
    "carbon_at_r_kg",
    "water_at_r_l",
    "carbon_at_tstar_kg",
    "water_at_tstar_l",
    "bsld",
]


@dataclass(frozen=True)
class Platform:
    name: str
    speed_flops: float
    idle_power_w: float
    compute_power_w: float


@dataclass(frozen=True)
class Workload:
    name: str
    platform: Platform
    job_ids: np.ndarray
    requested_time_s: np.ndarray
    execution_time_s: np.ndarray
    nodes: np.ndarray
    submission_time_s: np.ndarray


@dataclass(frozen=True)
class IntensityTrace:
    name: str
    zone: str
    season: str
    year: int
    carbon: np.ndarray
    water: np.ndarray

    @property
    def coverage_end_s(self) -> float:
        return float(len(self.carbon) * DELTA_SECONDS)


def parse_speed(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([kMGT]?)[fF]", value)
    if match is None:
        raise ValueError(f"Unsupported SimGrid speed: {value}")
    magnitude, prefix = match.groups()
    scale = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[prefix]
    return float(magnitude) * scale


def load_platform(name: str) -> Platform:
    path = WORKLOAD_ROOT / f"{name}.xml"
    root = ET.parse(path).getroot()
    compute_host = next(
        host
        for host in root.iter("host")
        if any(
            prop.attrib.get("id") == "role"
            and prop.attrib.get("value") == "compute_node"
            for prop in host.findall("prop")
        )
    )
    wattage = next(
        prop.attrib["value"]
        for prop in compute_host.findall("prop")
        if prop.attrib.get("id") == "wattage_per_state"
    )
    idle, _epsilon, all_cores = (float(part) for part in wattage.split(":"))
    platform = Platform(
        name=name,
        speed_flops=parse_speed(compute_host.attrib["speed"]),
        idle_power_w=idle,
        compute_power_w=all_cores,
    )
    values = np.asarray(
        [platform.speed_flops, platform.idle_power_w, platform.compute_power_w]
    )
    if not np.all(np.isfinite(values)) or platform.speed_flops <= 0:
        raise ValueError(f"Invalid platform values in {path}")
    if platform.idle_power_w < 0 or platform.compute_power_w < 0:
        raise ValueError(f"Negative platform power in {path}")
    return platform


def load_workload(name: str, max_jobs: int | None = None) -> Workload:
    platform_name, _regime = name.split("_", maxsplit=1)
    platform = load_platform(platform_name)
    path = WORKLOAD_ROOT / f"{name}.json"
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)

    jobs = [job for job in raw["jobs"] if not job["id"].startswith("ctx_")]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
    if not jobs:
        raise ValueError(f"No shiftable jobs found in {path}")

    execution_times = []
    for job in jobs:
        profile = raw["profiles"][job["profile"]]
        if profile.get("type") != "parallel_homogeneous":
            raise ValueError(f"Unsupported profile type for {job['id']}: {profile}")
        if float(profile.get("com", 0.0)) != 0.0:
            raise ValueError(f"Communication work is not supported for {job['id']}")
        execution_times.append(float(profile["cpu"]) / platform.speed_flops)

    workload = Workload(
        name=name,
        platform=platform,
        job_ids=np.asarray([job["id"] for job in jobs], dtype=object),
        requested_time_s=np.asarray([job["walltime"] for job in jobs], dtype=float),
        execution_time_s=np.asarray(execution_times, dtype=float),
        nodes=np.asarray([job["res"] for job in jobs], dtype=np.int32),
        submission_time_s=np.asarray([job["subtime"] for job in jobs], dtype=float),
    )
    numeric_fields = {
        "requested time": workload.requested_time_s,
        "execution time": workload.execution_time_s,
        "resource request": workload.nodes,
        "submission time": workload.submission_time_s,
    }
    for label, values in numeric_fields.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite {label} in {path}")
    if np.any(workload.execution_time_s <= 0):
        raise ValueError(f"Non-positive execution time in {path}")
    if np.any(workload.requested_time_s <= 0):
        raise ValueError(f"Non-positive requested time in {path}")
    if np.any(workload.execution_time_s > workload.requested_time_s + 1e-6):
        raise ValueError(f"Execution time exceeds requested time in {path}")
    if np.any(workload.nodes <= 0):
        raise ValueError(f"Non-positive resource request in {path}")
    if np.any(workload.submission_time_s < 0):
        raise ValueError(f"Negative submission time in {path}")
    return workload


def load_windows() -> pd.DataFrame:
    windows = pd.read_csv(WINDOWS_PATH)
    required = {"zone", "file", "start_date", "season", "year"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Missing windows.csv columns: {sorted(missing)}")
    if windows.duplicated(["zone", "start_date"]).any():
        raise ValueError("Duplicate intensity windows in windows.csv")
    return windows.sort_values(["zone", "start_date"], kind="stable").reset_index(drop=True)


def load_trace(row) -> IntensityTrace:
    path = REPOSITORY_ROOT / "intensities" / row.file
    long = pd.read_csv(path)
    required = {"timestamp", "zone", "property", "value"}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Missing trace columns in {path}: {sorted(missing)}")
    if set(long["zone"].unique()) != {"AS0"}:
        raise ValueError(f"Unexpected internal zones in {path}")

    wide = long.pivot(index="timestamp", columns="property", values="value").sort_index()
    expected_properties = {"carbon_intensity", "water_intensity"}
    if set(wide.columns) != expected_properties:
        raise ValueError(f"Unexpected intensity properties in {path}: {list(wide.columns)}")
    expected_timestamps = np.arange(len(wide), dtype=np.int64) * DELTA_SECONDS
    actual_timestamps = wide.index.to_numpy(dtype=np.int64)
    if not np.array_equal(actual_timestamps, expected_timestamps):
        raise ValueError(f"Trace timestamps are not a contiguous 15-minute grid in {path}")
    if wide.isna().any().any():
        raise ValueError(f"Missing intensity values in {path}")

    carbon = wide["carbon_intensity"].to_numpy(dtype=float)
    water = wide["water_intensity"].to_numpy(dtype=float)
    if not np.all(np.isfinite(carbon)) or not np.all(np.isfinite(water)):
        raise ValueError(f"Non-finite intensity values in {path}")
    if np.any(carbon < 0) or np.any(water < 0):
        raise ValueError(f"Negative intensity values in {path}")

    return IntensityTrace(
        name=path.stem,
        zone=str(row.zone),
        season=str(row.season).lower(),
        year=int(row.year),
        carbon=carbon,
        water=water,
    )


def prefix_integral(values: np.ndarray) -> np.ndarray:
    block_hours = DELTA_SECONDS / 3600.0
    return np.concatenate(([0.0], np.cumsum(values * block_hours, dtype=float)))


def integral_at(
    times_s: np.ndarray,
    values: np.ndarray,
    prefix: np.ndarray,
) -> np.ndarray:
    times = np.asarray(times_s, dtype=float)
    coverage_end = len(values) * DELTA_SECONDS
    if np.any(times < 0.0) or np.any(times > coverage_end + 1e-7):
        lo = float(np.min(times))
        hi = float(np.max(times))
        raise ValueError(
            f"Integral endpoint range [{lo}, {hi}] exceeds [0, {coverage_end}]"
        )

    times = np.clip(times, 0.0, float(coverage_end))
    block = np.floor_divide(times, DELTA_SECONDS).astype(np.int64)
    at_end = block == len(values)
    safe_block = np.minimum(block, len(values) - 1)
    partial_hours = (times - safe_block * DELTA_SECONDS) / 3600.0
    result = prefix[safe_block] + values[safe_block] * partial_hours
    if np.any(at_end):
        result = np.asarray(result)
        result[at_end] = prefix[-1]
    return result


def cost_matrices(
    workload: Workload,
    trace: IntensityTrace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets_s = np.arange(
        0,
        max(SHIFT_HOURS) * 3600 + DELTA_SECONDS,
        DELTA_SECONDS,
        dtype=float,
    )
    starts = workload.submission_time_s[:, None] + offsets_s[None, :]
    finishes = starts + workload.execution_time_s[:, None]
    required_end = float(np.max(finishes))
    if required_end > trace.coverage_end_s + 1e-7:
        raise ValueError(
            f"{workload.name} needs trace coverage through {required_end:.3f}s, "
            f"but {trace.name} ends at {trace.coverage_end_s:.3f}s"
        )

    carbon_prefix = prefix_integral(trace.carbon)
    water_prefix = prefix_integral(trace.water)

    carbon_at_start = integral_at(starts, trace.carbon, carbon_prefix)
    carbon_at_finish = integral_at(finishes, trace.carbon, carbon_prefix)
    carbon_at_submission = integral_at(
        workload.submission_time_s, trace.carbon, carbon_prefix
    )[:, None]

    water_at_start = integral_at(starts, trace.water, water_prefix)
    water_at_finish = integral_at(finishes, trace.water, water_prefix)
    water_at_submission = integral_at(
        workload.submission_time_s, trace.water, water_prefix
    )[:, None]

    compute_carbon = carbon_at_finish - carbon_at_start
    idle_carbon = carbon_at_start - carbon_at_submission
    compute_water = water_at_finish - water_at_start
    idle_water = water_at_start - water_at_submission

    node_factor = workload.nodes[:, None]
    carbon_kg = node_factor * (
        workload.platform.compute_power_w * compute_carbon
        + workload.platform.idle_power_w * idle_carbon
    ) / 1_000_000.0
    water_l = node_factor * (
        workload.platform.compute_power_w * compute_water
        + workload.platform.idle_power_w * idle_water
    ) / 1_000.0
    return offsets_s, carbon_kg, water_l


def result_frame(
    workload: Workload,
    trace: IntensityTrace,
    d_m_hours: int,
    signal: str,
    offsets_s: np.ndarray,
    carbon_costs: np.ndarray,
    water_costs: np.ndarray,
) -> pd.DataFrame:
    candidate_count = d_m_hours * 3600 // DELTA_SECONDS + 1
    objective_costs = carbon_costs if signal == "carbon" else water_costs
    best_index = np.argmin(objective_costs[:, :candidate_count], axis=1)
    row_index = np.arange(len(workload.job_ids))
    displacement_s = offsets_s[best_index]
    execution_denominator = np.maximum(
        workload.execution_time_s, BOUNDED_SLOWDOWN_FLOOR_SECONDS
    )
    bounded_slowdown = np.maximum(
        1.0,
        (displacement_s + workload.execution_time_s) / execution_denominator,
    )

    return pd.DataFrame(
        {
            "workload": workload.name,
            "platform": workload.platform.name,
            "zone": trace.zone,
            "intensity_trace": trace.name,
            "season": trace.season,
            "year": trace.year,
            "job_id": workload.job_ids,
            "requested_time_s": workload.requested_time_s,
            "execution_time_s": workload.execution_time_s,
            "nodes": workload.nodes,
            "submission_time_s": workload.submission_time_s,
            "d_m_label": f"{d_m_hours}h",
            "d_m_seconds": d_m_hours * 3600,
            "signal": signal,
            "t_star_s": workload.submission_time_s + displacement_s,
            "carbon_at_r_kg": carbon_costs[:, 0],
            "water_at_r_l": water_costs[:, 0],
            "carbon_at_tstar_kg": carbon_costs[row_index, best_index],
            "water_at_tstar_l": water_costs[row_index, best_index],
            "bsld": bounded_slowdown,
        },
        columns=OUTPUT_COLUMNS,
    )


def open_text_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="wt", encoding="utf-8", newline="")
    return path.open(mode="w", encoding="utf-8", newline="")


def generate(
    output: Path,
    workload_names: tuple[str, ...],
    trace_names: set[str] | None,
    max_jobs: int | None,
    overwrite: bool,
) -> int:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    if len(workload_names) != len(set(workload_names)):
        raise ValueError("Duplicate workload selections are not allowed")
    unknown_workloads = set(workload_names).difference(WORKLOAD_NAMES)
    if unknown_workloads:
        raise ValueError(f"Unknown workloads: {sorted(unknown_workloads)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_suffix = ".tmp.gz" if output.suffix == ".gz" else ".tmp"
    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}{temporary_suffix}"
    )

    workloads = [load_workload(name, max_jobs=max_jobs) for name in workload_names]
    windows = load_windows()
    if trace_names is not None:
        available_traces = set(windows["file"].map(lambda value: Path(value).stem))
        missing_traces = trace_names.difference(available_traces)
        if missing_traces:
            raise ValueError(f"Unknown intensity traces: {sorted(missing_traces)}")
        windows = windows[
            windows["file"].map(lambda value: Path(value).stem).isin(trace_names)
        ].reset_index(drop=True)
    if windows.empty:
        raise ValueError("No intensity windows selected")

    expected_rows = sum(len(workload.job_ids) for workload in workloads)
    expected_rows *= len(windows) * len(SHIFT_HOURS) * 2
    print(
        f"Generating {expected_rows:,} rows from {len(workloads)} workloads "
        f"and {len(windows)} intensity windows"
    )

    rows_written = 0
    header = True
    try:
        with open_text_output(temporary) as sink:
            pair_count = len(workloads) * len(windows)
            pair_index = 0
            for workload in workloads:
                for window in windows.itertuples(index=False):
                    pair_index += 1
                    trace = load_trace(window)
                    offsets_s, carbon_costs, water_costs = cost_matrices(workload, trace)
                    for d_m_hours in SHIFT_HOURS:
                        for signal in ("carbon", "water"):
                            frame = result_frame(
                                workload,
                                trace,
                                d_m_hours,
                                signal,
                                offsets_s,
                                carbon_costs,
                                water_costs,
                            )
                            frame.to_csv(
                                sink,
                                index=False,
                                header=header,
                                float_format="%.10g",
                            )
                            header = False
                            rows_written += len(frame)
                    print(
                        f"[{pair_index:>3}/{pair_count}] {workload.name} x {trace.name}"
                    )
        if rows_written != expected_rows:
            raise RuntimeError(
                f"Wrote {rows_written:,} rows, expected {expected_rows:,}"
            )
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    print(f"Wrote {rows_written:,} rows to {output}")
    return rows_written


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV or CSV.GZ path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--workload",
        action="append",
        choices=WORKLOAD_NAMES,
        dest="workloads",
        help="Limit generation to one or more workloads",
    )
    parser.add_argument(
        "--trace",
        action="append",
        dest="traces",
        help="Limit generation to one or more trace stems, such as DE_2019-02-04",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Limit jobs per workload for a smoke test",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output after a successful complete run",
    )
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    workload_names = tuple(args.workloads or WORKLOAD_NAMES)
    trace_names = set(args.traces) if args.traces else None
    generate(
        output=args.output,
        workload_names=workload_names,
        trace_names=trace_names,
        max_jobs=args.max_jobs,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
