#!/usr/bin/env python3
"""Generate the one-week pilot campaign.

Reads the one-week intensity manifest at intensities/traces/small/windows.csv and
emits, repointed at the small/ artifacts:

  - experiments_small.toml: 3 schedulers x 2 workloads x 24 traces = 144 experiments
  - options/small/{carbon,water}_<CC>_<date>.json: greenfilling variant options

Run from the experiments/ directory (where the campaign runner is launched), so the
relative paths it writes resolve correctly:

    python3 gen_campaign.py
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.path.join(HERE, "..", "intensities", "traces", "small", "windows.csv")
OPTIONS_DIR = os.path.join(HERE, "options", "small")
TOML_OUT = os.path.join(HERE, "experiments_small.toml")

WORKLOADS = ["stress", "slack"]
SIGNALS = ["carbon", "water"]
SMOOTHING_FACTOR = 0.5


def load_traces():
    """Return [(zone, start_date)] from the manifest, in file order."""
    with open(WINDOWS, newline="") as f:
        return [(row["zone"], row["start_date"]) for row in csv.DictReader(f)]


def write_options(traces):
    os.makedirs(OPTIONS_DIR, exist_ok=True)
    for cc, date in traces:
        for signal in SIGNALS:
            opts = {
                "intensity_trace": f"../intensities/traces/small/{cc}_{date}.csv",
                "intensity_zone": "AS0",
                "signal": signal,
                "smoothing_factor": SMOOTHING_FACTOR,
                "greenfilling_debug": True,
            }
            path = os.path.join(OPTIONS_DIR, f"{signal}_{cc}_{date}.json")
            with open(path, "w") as f:
                json.dump(opts, f, indent=4)
                f.write("\n")
    return 2 * len(traces)


def experiment(name, workload, trace, variant, options):
    return (
        "[[experiment]]\n"
        f'name = "{name}"\n'
        f'workload = "../workloads and platforms/small/mustang_{workload}.json"\n'
        f'platform = "../workloads and platforms/mustang.xml"\n'
        f'environmental_trace = "../intensities/traces/small/{trace}.csv"\n'
        f'variant_name = "{variant}"\n'
        f'variant_options = "{options}"\n'
    )


def write_toml(traces):
    blocks = []
    # Baseline: schedule is independent of the trace, but kept per-trace for the
    # pilot (the offline-scorer dedup is deferred to the 4-week campaign).
    for wl in WORKLOADS:
        for cc, date in traces:
            blocks.append(experiment(
                f"easy_bf_{wl}_{cc}_{date}", wl, f"{cc}_{date}",
                "easy_bf", "",
            ))
    # Greenfilling: one variant per intensity signal.
    for signal in SIGNALS:
        for wl in WORKLOADS:
            for cc, date in traces:
                blocks.append(experiment(
                    f"greenfilling_{signal}_{wl}_{cc}_{date}", wl, f"{cc}_{date}",
                    "greenfilling", f"options/small/{signal}_{cc}_{date}.json",
                ))
    with open(TOML_OUT, "w") as f:
        f.write("\n".join(blocks))
    return len(blocks)


def main():
    traces = load_traces()
    n_opts = write_options(traces)
    n_exp = write_toml(traces)
    print(f"{len(traces)} traces -> {n_opts} option files in options/small/, "
          f"{n_exp} experiments in experiments_small.toml")


if __name__ == "__main__":
    main()
