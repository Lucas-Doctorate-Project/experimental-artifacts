"""Small numerical helpers for the analysis notebooks."""

from pathlib import Path
import json
import tomllib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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
    the horizon. Exported totals are never substituted for these integrals.
    """
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


def region_work(jobs, edges):
    """Executed node-hours within each interval, including partial jobs."""
    starts = jobs.starting_time.to_numpy()
    ends = jobs.finish_time.to_numpy()
    nodes = jobs.requested_number_of_resources.to_numpy()
    return np.array(
        [
            np.dot(nodes, np.maximum(0, np.minimum(ends, stop) - np.maximum(starts, start))) / 3600
            for (start, stop) in zip(edges[:-1], edges[1:])
        ]
    )


def regional_comparison(easy, green, signal):
    """Compare common 6-hour regions after context drain and before submissions end."""
    context_end = max(
        (
            float(j.loc[j.is_context, "finish_time"].max()) if j.is_context.any() else 0
            for j in (easy, green)
        )
    )
    end = float(easy.loc[~easy.is_context, "submission_time"].max())
    edges = np.unique(
        np.r_[context_end, np.arange(np.ceil(context_end / 21600) * 21600, end, 21600), end]
    )
    mean_intensity = integral(signal, edges[:-1], edges[1:]) / np.diff(edges)
    result = pd.DataFrame(
        {
            "start_seconds": edges[:-1],
            "end_seconds": edges[1:],
            "hours": np.diff(edges) / 3600,
            "intensity": mean_intensity,
            "easy_node_hours": region_work(easy, edges),
            "green_node_hours": region_work(green, edges),
        }
    )
    result["intensity_quartile"] = (
        pd.qcut(result.intensity.rank(method="first"), 4, labels=False) + 1
    )
    for name, jobs in [("easy", easy), ("green", green)]:
        finish = jobs.loc[~jobs.is_context, "finish_time"]
        finish = finish.loc[(finish >= context_end) & (finish < end)]
        result[f"{name}_completions"] = np.histogram(finish, bins=edges)[0]
    return result


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


def pool_regions(regions):
    """Pool counts and durations; partial regions keep their actual duration."""
    regions = regions.copy()
    regions["intensity_hours"] = regions.intensity * regions.hours
    mean_intensity = regions.intensity_hours.sum() / regions.hours.sum()
    grouped = (
        regions.groupby("intensity_quartile")
        .agg(
            hours=("hours", "sum"),
            intensity_hours=("intensity_hours", "sum"),
            easy_node_hours=("easy_node_hours", "sum"),
            green_node_hours=("green_node_hours", "sum"),
            easy_completions=("easy_completions", "sum"),
            green_completions=("green_completions", "sum"),
        )
        .reset_index()
    )
    grouped["intensity"] = grouped.pop("intensity_hours") / grouped.hours
    grouped["relative_intensity"] = divide(grouped.intensity, mean_intensity)
    for name in ("easy", "green"):
        grouped[f"{name}_throughput"] = grouped[f"{name}_completions"] / grouped.hours
    grouped["throughput_ratio"] = divide(grouped.green_throughput, grouped.easy_throughput)
    for metric in ("node_hours", "completions"):
        grouped[f"{metric}_share_delta_pp"] = 100 * (
            divide(grouped[f"green_{metric}"], grouped[f"green_{metric}"].sum())
            - divide(grouped[f"easy_{metric}"], grouped[f"easy_{metric}"].sum())
        )
    return grouped


def regional_analysis(easy, green, signal, smoothing_factor):
    """Return interval rows, quartile rows, gate rows, and a pair summary."""
    regions = regional_comparison(easy, green, signal)
    (start, end) = (regions.start_seconds.iloc[0], regions.end_seconds.iloc[-1])
    mean_intensity = float(integral(signal, start, end) / (end - start))
    regions["relative_intensity"] = divide(regions.intensity, mean_intensity)
    for name in ("easy", "green"):
        regions[f"{name}_throughput"] = regions[f"{name}_completions"] / regions.hours
        regions[f"{name}_throughput_per_relative_intensity"] = divide(
            regions[f"{name}_throughput"], regions.relative_intensity
        )
    levels = pool_regions(regions)
    (events, opened) = reconstruct_gate(green, signal, end, smoothing_factor)

    def state_at(dates):
        return opened[np.searchsorted(events, dates, side="right") - 1]

    breaks = np.unique(
        np.r_[
            start,
            events[(events > start) & (events < end)],
            signal["times"][(signal["times"] > start) & (signal["times"] < end)],
            end,
        ]
    )
    durations = np.diff(breaks)
    segment_open = state_at(breaks[:-1])
    segment_intensity = sample(signal, breaks[:-1])
    finishes = {}
    summary = {
        "start_seconds": start,
        "end_seconds": end,
        "hours": (end - start) / 3600,
        "mean_intensity": mean_intensity,
        "smoothing_factor": smoothing_factor,
    }
    for name, jobs in (("easy", easy), ("green", green)):
        replay = jobs.loc[~jobs.is_context]
        finish = replay.finish_time.to_numpy()
        finishes[name] = finish[(finish >= start) & (finish < end)]
        count = len(finishes[name])
        summary[f"{name}_replay_jobs"] = len(replay)
        summary[f"{name}_completions"] = count
        summary[f"{name}_completed_pct"] = 100 * divide(count, len(replay))
        summary[f"{name}_completion_intensity"] = (
            float(sample(signal, finishes[name]).mean()) if count else np.nan
        )
    summary["completion_intensity_change_pct"] = 100 * (
        divide(summary["green_completion_intensity"], summary["easy_completion_intensity"]) - 1
    )
    gate_rows = []
    for state, is_open in (("open", True), ("closed", False)):
        selected = segment_open == is_open
        seconds = durations[selected].sum()
        row = {
            "gate_state": state,
            "hours": seconds / 3600,
            "time_share_pct": 100 * seconds / (end - start),
            "relative_intensity": divide(
                np.dot(segment_intensity[selected], durations[selected]), seconds * mean_intensity
            ),
        }
        for name, finish in finishes.items():
            count = int(np.count_nonzero(state_at(finish) == is_open))
            row[f"{name}_completions"] = count
            row[f"{name}_throughput"] = divide(count, row["hours"])
        row["throughput_ratio"] = divide(row["green_throughput"], row["easy_throughput"])
        gate_rows.append(row)
    gate = pd.DataFrame(gate_rows)
    return (regions, levels, gate, summary)


def relaxed_opportunity(
    jobs, signal, platform, horizon, objective, delay_hours=(0, 1, 3, 6, 12, 24)
):
    """Exact savings bound: independent job placement, fixed context and idle power.

    A replay job may start from submission through its EASY start plus the delay.
    Candidate starts are interval endpoints and intensity boundaries at either
    end of the job. Between candidates its integrated footprint is linear.
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
    for i in range(len(jobs)):
        if context[i]:
            optimum_active += baseline_jobs[i]
            continue
        lower = release[i]
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
        optimum_active += minimum[np.searchsorted(candidates, endpoints, side="right") - 1]
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
            "job_count": len(jobs),
            "shiftable_job_count": int((~context).sum()),
            "context_job_count": int(context.sum()),
        }
    )
