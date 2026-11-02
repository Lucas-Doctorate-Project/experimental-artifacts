"""Small numerical helpers for the analysis scripts."""

from pathlib import Path
import json
import tomllib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
J_PER_KWH = 3_600_000.0


def experiments(group):
    """Read the same input paths used by the simulation runner."""
    config = tomllib.loads((ROOT / "experiments" / f"{group}.toml").read_text())
    rows = []
    for exp in config["experiment"]:
        variant, dataset, regime, zone, date = exp["name"].rsplit("_", 4)
        options = (
            json.loads((ROOT / "experiments" / exp["variant_options"]).read_text())
            if exp["variant_options"]
            else {}
        )
        rows.append(
            {
                **exp,
                "group": group,
                "variant": variant,
                "workload": dataset + "_" + regime,
                "trace": zone + "_" + date,
                "zone": zone,
                "objective": options.get("signal", "none"),
                "alpha": options.get("smoothing_factor", np.nan),
                "planning_horizon_seconds": options.get("planning_horizon_seconds", np.nan),
            }
        )
    return pd.DataFrame(rows)


def platform(path):
    hosts = ET.parse(ROOT / "experiments" / path).getroot().findall(".//host")
    nodes = [
        h
        for h in hosts
        if any(
            p.get("id") == "role" and p.get("value") == "compute_node" for p in h.findall("prop")
        )
    ]
    power = next(
        p.get("value") for p in nodes[0].findall("prop") if p.get("id") == "wattage_per_state"
    )
    idle, _, busy = map(float, power.split(":"))
    return {"nodes": len(nodes), "idle_w": idle, "busy_w": busy}


def load_jobs(group, name, nodes):
    jobs = pd.read_csv(
        ROOT / "analysis/outputs" / group / name / "out_jobs.csv", dtype={"job_id": str}
    )
    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    jobs["is_context"] = jobs.job_id.str.startswith("ctx_")
    jobs["class"] = np.select(
        [
            (jobs.requested_number_of_resources <= 0.01 * nodes) & (jobs.execution_time <= 7200),
            jobs.requested_number_of_resources >= 0.10 * nodes,
        ],
        ["narrow-short", "wide"],
        default="other",
    )
    return jobs


def load_signals(path):
    trace = pd.read_csv(ROOT / "experiments" / path)
    wide = trace.pivot(index="timestamp", columns="property", values="value").sort_index()
    times = wide.index.to_numpy(dtype=float)
    edges = np.r_[times, times[-1] + times[1] - times[0]]
    return {
        objective: {
            "times": times,
            "values": wide[f"{objective}_intensity"].to_numpy(),
            "prefix": np.r_[
                0.0, np.cumsum(wide[f"{objective}_intensity"].to_numpy() * np.diff(edges))
            ],
        }
        for objective in ("carbon", "water")
    }


def integral(signal, starts, ends):
    """Integrate the stepwise intensity, including partial sample intervals."""

    def cumulative(dates):
        dates = np.asarray(dates, dtype=float)
        index = np.minimum(
            np.searchsorted(signal["times"], dates, side="right") - 1, len(signal["values"]) - 1
        )
        return signal["prefix"][index] + signal["values"][index] * (dates - signal["times"][index])

    return cumulative(ends) - cumulative(starts)


def performance(jobs):
    replay = jobs.loc[~jobs.is_context]
    duration = replay.execution_time.to_numpy()
    response = (replay.finish_time - replay.submission_time).to_numpy()
    weight = duration * replay.requested_number_of_resources.to_numpy()
    return {
        "replay_jobs": len(replay),
        "bounded_slowdown": np.maximum(response / np.maximum(duration, 10), 1).mean(),
        "weighted_response_seconds": np.average(response, weights=weight),
        "median_wait_seconds": replay.waiting_time.median()
        if "waiting_time" in replay
        else np.median(response - duration),
        "p95_wait_seconds": np.quantile(response - duration, 0.95),
        "makespan_seconds": jobs.finish_time.max(),
    }


def physical_totals(jobs, platform, signals, horizon):
    """Fixed platform idle power plus each running job's increment above idle.

    Footprints include context jobs and idle nodes. All jobs must finish within
    the horizon, and every signal must cover it. Exported totals are never
    substituted for these integrals.
    """
    last_finish = float(jobs.finish_time.max())
    if last_finish > horizon:
        raise ValueError(
            f"a job finishes at {last_finish / 86400:.2f} days, after the {horizon / 86400:g}-day horizon"
        )
    for objective, signal in signals.items():
        coverage = signal["times"][-1] + signal["times"][1] - signal["times"][0]
        if horizon > coverage:
            raise ValueError(
                f"{objective} signal covers {coverage / 86400:.2f} days, shorter than the"
                f" {horizon / 86400:g}-day horizon"
            )
    nodes = jobs.requested_number_of_resources.to_numpy()
    node_seconds = float(np.dot(nodes, jobs.execution_time))
    busy_energy = platform["busy_w"] * node_seconds / J_PER_KWH
    idle_energy = platform["idle_w"] * (platform["nodes"] * horizon - node_seconds) / J_PER_KWH
    result = {
        "horizon_seconds": horizon,
        "node_hours": node_seconds / 3600,
        "energy_kwh": busy_energy + idle_energy,
        "busy_energy_kwh": busy_energy,
        "idle_energy_kwh": idle_energy,
    }
    for objective, signal in signals.items():
        exposure = float(np.dot(nodes, integral(signal, jobs.starting_time, jobs.finish_time)))
        fixed_idle = (
            platform["nodes"] * platform["idle_w"] * float(integral(signal, 0, horizon)) / J_PER_KWH
        )
        busy = platform["busy_w"] * exposure / J_PER_KWH
        total = fixed_idle + (platform["busy_w"] - platform["idle_w"]) * exposure / J_PER_KWH
        unit = "g" if objective == "carbon" else "l"
        result.update(
            {
                f"{objective}_{unit}": total,
                f"busy_{objective}_{unit}": busy,
                f"idle_{objective}_{unit}": total - busy,
                f"energy_weighted_{objective}_intensity": total / result["energy_kwh"],
                f"idle_{objective}_intensity": (total - busy) / idle_energy,
            }
        )
    return result


def class_contributions(easy, green, platform):
    """Signed additive contributions; class rows sum to each total delta."""
    (a, b) = (easy.set_index("job_id"), green.set_index("job_id"))
    replay = ~a.is_context
    duration = a.execution_time
    work = duration * a.requested_number_of_resources
    response_a = a.finish_time - a.submission_time
    response_b = b.finish_time - b.submission_time
    slowdown_a = np.maximum(response_a / np.maximum(duration, 10), 1)
    slowdown_b = np.maximum(response_b / np.maximum(duration, 10), 1)
    rows = []
    for label in ("narrow-short", "other", "wide", "context"):
        mask = a.is_context if label == "context" else replay & a["class"].eq(label)
        row = {"class": label, "jobs": int(mask.sum()), "node_hours": work[mask].sum() / 3600}
        row["slowdown_contribution"] = (
            0.0
            if label == "context"
            else (slowdown_b[mask] - slowdown_a[mask]).sum() / replay.sum()
        )
        row["wrt_contribution_seconds"] = (
            0.0
            if label == "context"
            else ((response_b - response_a)[mask] * work[mask]).sum() / work[replay].sum()
        )
        for objective in ("carbon", "water"):
            row[f"{objective}_contribution"] = (1 - platform["idle_w"] / platform["busy_w"]) * (
                b.loc[mask, f"consumed_{objective}"].sum()
                - a.loc[mask, f"consumed_{objective}"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def divide(numerator, denominator):
    """Leave a ratio undefined when its denominator is zero."""
    (numerator, denominator) = np.broadcast_arrays(
        np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    )
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result.item() if result.ndim == 0 else result


def sample(signal, dates):
    dates = np.asarray(dates, dtype=float)
    return signal["values"][np.searchsorted(signal["times"], dates, side="right") - 1]


def reconstruct_gate(jobs, signal, end, smoothing_factor):
    """Estimate one EMA update per distinct submission/completion timestamp."""
    events = np.unique(np.r_[0.0, jobs.submission_time, jobs.finish_time])
    events = events[events < end]
    intensity = sample(signal, events)
    opened = np.empty(len(events), dtype=bool)
    average = intensity[0]
    for i, current in enumerate(intensity):
        if i:
            average = smoothing_factor * current + (1 - smoothing_factor) * average
        opened[i] = current <= average
    return (events, opened)


def relaxed_opportunity(
    jobs,
    signal,
    platform,
    horizon,
    objective,
    delay_hours=(0, 1, 3, 6, 12, 24),
    allow_advance=True,
):
    """Exact savings bound: independent job placement, fixed context and idle power.

    A replay job may start from submission through its EASY start plus the delay.
    Without advance, the earliest start is the EASY start, which bounds policies
    that can only defer. Candidate starts are interval endpoints and intensity
    boundaries at either end of the job. Between candidates its integrated
    footprint is linear. The chosen start is the earliest minimum, so
    mean_shift_hours does not grow on flat intensity.
    """
    delays = np.asarray(delay_hours, dtype=float)
    release = jobs.submission_time.to_numpy()
    starts = jobs.starting_time.to_numpy()
    duration = jobs.execution_time.to_numpy()
    nodes = jobs.requested_number_of_resources.to_numpy()
    context = jobs.is_context.to_numpy()
    times = signal["times"]
    multiplier = (platform["busy_w"] - platform["idle_w"]) / J_PER_KWH
    baseline_jobs = nodes * multiplier * integral(signal, starts, starts + duration)
    fixed_idle = platform["nodes"] * platform["idle_w"] * integral(signal, 0, horizon) / J_PER_KWH
    baseline_active = baseline_jobs.sum()
    baseline_total = fixed_idle + baseline_active
    optimum_active = np.zeros(len(delays))
    shift_seconds = np.zeros(len(delays))
    for i in range(len(jobs)):
        if context[i]:
            optimum_active += baseline_jobs[i]
            continue
        lower = release[i] if allow_advance else starts[i]
        endpoints = starts[i] + delays * 3600
        upper = endpoints[-1]
        boundaries = times[
            np.searchsorted(times, lower, side="left") : np.searchsorted(times, upper, side="right")
        ]
        end_boundaries = (
            times[
                np.searchsorted(times, lower + duration[i], side="left") : np.searchsorted(
                    times, upper + duration[i], side="right"
                )
            ]
            - duration[i]
        )
        candidates = np.unique(np.r_[lower, starts[i], endpoints, boundaries, end_boundaries])
        candidates = candidates[(candidates >= lower) & (candidates <= upper)]
        costs = nodes[i] * multiplier * integral(signal, candidates, candidates + duration[i])
        minimum = np.minimum.accumulate(costs)
        last = np.searchsorted(candidates, endpoints, side="right") - 1
        optimum_active += minimum[last]
        # a later candidate replaces the earliest minimum only beyond rounding noise
        improves = np.r_[True, costs[1:] < minimum[:-1] - 1e-9 * np.abs(minimum[:-1])]
        earliest = np.maximum.accumulate(np.where(improves, np.arange(len(costs)), 0))
        shift_seconds += candidates[earliest[last]] - starts[i]
    # Clip floating-point accumulation noise at the included EASY baseline.
    optimum_active = np.minimum(optimum_active, baseline_active)
    totals = fixed_idle + optimum_active
    savings = baseline_total - totals
    return pd.DataFrame(
        {
            "delay_hours": delays,
            "horizon_s": horizon,
            "footprint_unit": "g" if objective == "carbon" else "L",
            "easy_footprint": baseline_total,
            "relaxed_footprint": totals,
            "fixed_idle_footprint": fixed_idle,
            "easy_active_increment": baseline_active,
            "relaxed_active_increment": optimum_active,
            "savings_absolute": savings,
            "savings_pct": 100 * savings / baseline_total,
            "mean_shift_hours": divide(shift_seconds / 3600, int((~context).sum())),
            "job_count": len(jobs),
            "shiftable_job_count": int((~context).sum()),
            "context_job_count": int(context.sum()),
        }
    )
