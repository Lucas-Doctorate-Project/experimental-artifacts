#!/usr/bin/env python3
"""Generate the full 4-week campaign.

Covers every 4-week intensity window under intensities/traces/ and both datasets
(Mustang and Trinity). Emits:

  - experiments.toml: schedulers x datasets x regimes x N windows, where the
    green_window_scheduling schedulers are further split by signal and planning
    horizon.
  - options/{carbon,water}_<horizon>_<dataset>_<CC>_<date>.json:
    green_window_scheduling variant options per window, signal, horizon, and
    dataset. Dataset-specific because computing_watts/idle_watts must match
    each platform's real wattage.

Experiment names carry every axis, so no two experiments share an output
directory: <variant>[_<signal>_<horizon>]_<dataset>_<regime>_<CC>_<date>.

Run from the experiments/ directory (where the campaign runner is launched), so
the relative paths it writes resolve correctly:

    python3 gen_campaign.py
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(HERE, "..", "intensities", "traces")
OPTIONS_DIR = os.path.join(HERE, "options")
TOML_OUT = os.path.join(HERE, "experiments.toml")

DATASETS = ["mustang", "trinity"]
REGIMES = ["slack", "stress"]
SIGNALS = ["carbon", "water"]

# How far ahead green_window_scheduling may look for a greener window to displace
# a job: the campaign uses a 24-hour planning horizon.
PLANNING_HORIZONS_SECONDS = [86400]

# Real per-node wattage from each platform's XML ("wattage_per_state" on
# compute_node hosts, idle:computing:computing). Must match the simulated
# platform so the algorithm's environmental-impact scoring is accurate.
DATASET_WATTAGE = {
    "mustang": {"idle_watts": 10, "computing_watts": 320},
    "trinity": {"idle_watts": 10, "computing_watts": 270},
}

# 4-week window CSVs are named <CC>_<YYYY-MM-DD>; whole-trace files like DE.csv
# do not match this pattern and are excluded.
WINDOW_RE = re.compile(r"^([A-Z]{2})_(\d{4}-\d{2}-\d{2})$")


def load_traces():
    """Return [(zone, start_date)] for every 4-week window CSV, in name order."""
    traces = []
    for path in sorted(glob.glob(os.path.join(TRACES_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        match = WINDOW_RE.match(stem)
        if match:
            traces.append((match.group(1), match.group(2)))
    return traces


def write_options(traces):
    os.makedirs(OPTIONS_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(OPTIONS_DIR, "*.json")):
        os.remove(stale)
    count = 0
    for cc, date in traces:
        for signal in SIGNALS:
            for horizon in PLANNING_HORIZONS_SECONDS:
                for dataset in DATASETS:
                    wattage = DATASET_WATTAGE[dataset]
                    opts = {
                        "intensity_trace": f"../intensities/traces/{cc}_{date}.csv",
                        "intensity_zone": "AS0",
                        "signal": signal,
                        "planning_horizon_seconds": horizon,
                        "computing_watts": wattage["computing_watts"],
                        "idle_watts": wattage["idle_watts"],
                        "green_window_scheduling_debug": False,
                    }
                    path = os.path.join(
                        OPTIONS_DIR, f"{signal}_{horizon}_{dataset}_{cc}_{date}.json"
                    )
                    with open(path, "w") as f:
                        json.dump(opts, f, indent=4)
                        f.write("\n")
                    count += 1
    return count


def experiment(name, dataset, regime, trace, variant, options):
    return (
        "[[experiment]]\n"
        f'name = "{name}"\n'
        f'workload = "../workloads and platforms/{dataset}_{regime}.json"\n'
        f'platform = "../workloads and platforms/{dataset}.xml"\n'
        f'environmental_trace = "../intensities/traces/{trace}.csv"\n'
        f'variant_name = "{variant}"\n'
        f'variant_options = "{options}"\n'
    )


def write_toml(traces):
    blocks = []
    # Baseline EASY backfilling: schedule is independent of the intensity trace,
    # but kept per-trace so every experiment has a matching environmental trace.
    for dataset in DATASETS:
        for regime in REGIMES:
            for cc, date in traces:
                blocks.append(experiment(
                    f"easy_bf_{dataset}_{regime}_{cc}_{date}",
                    dataset, regime, f"{cc}_{date}", "easy_bf", "",
                ))
    # green_window_scheduling: one variant per intensity signal and planning
    # horizon; options are dataset-specific (wattage differs per platform).
    for signal in SIGNALS:
        for horizon in PLANNING_HORIZONS_SECONDS:
            for dataset in DATASETS:
                for regime in REGIMES:
                    for cc, date in traces:
                        blocks.append(experiment(
                            f"green_window_scheduling_{signal}_{horizon}_{dataset}_{regime}_{cc}_{date}",
                            dataset, regime, f"{cc}_{date}",
                            "green_window_scheduling",
                            f"options/{signal}_{horizon}_{dataset}_{cc}_{date}.json",
                        ))
    with open(TOML_OUT, "w") as f:
        f.write("\n".join(blocks))
    return len(blocks)


def main():
    traces = load_traces()
    n_opts = write_options(traces)
    n_exp = write_toml(traces)
    print(f"{len(traces)} windows x {len(DATASETS)} datasets -> "
          f"{n_opts} option files in options/, {n_exp} experiments in experiments.toml")


if __name__ == "__main__":
    main()
