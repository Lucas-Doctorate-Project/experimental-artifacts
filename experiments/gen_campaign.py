#!/usr/bin/env python3
"""Generate the Greenfilling campaign on the user-resampled Mustang workloads.

Covers every 4-week intensity window under intensities/traces/ and the two
resampled Mustang workloads (one per regime) exported by
analysis/mustang_user_resampling.ipynb. Emits:

  - experiments.toml: (EASY + greenfilling x signals) x regimes x N windows.
  - options/greenfilling_{carbon,water}_<CC>_<date>.json: greenfilling variant
    options per window and signal. Greenfilling reads no platform wattage, so
    the options are not dataset-specific.

Experiment names carry every axis, so no two experiments share an output
directory: <variant>[_<signal>]_mustang_<regime>_<CC>_<date>. The dataset token
stays `mustang` (the platform) so analysis/campaign_analysis.py parses the
names; the resampled workload behind each regime is recorded in
RESAMPLED_WORKLOADS below.

Run from the experiments/ directory (where the campaign runner is launched), so
the relative paths it writes resolve correctly:

    python3 gen_campaign.py
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRACES_DIR = os.path.join(HERE, "..", "intensities", "traces")
OPTIONS_DIR = os.path.join(HERE, "options")
TOML_OUT = os.path.join(HERE, "experiments.toml")

DATASET = "mustang"
SIGNALS = ["carbon", "water"]

# One resampled workload per regime: the IDs selected and exported by the last
# cell of analysis/mustang_user_resampling.ipynb.
RESAMPLED_DIR = "../workloads and platforms/generated/mustang_user_resampling"
RESAMPLED_WORKLOADS = {
    "slack": "mustang_slack_heavy_4x_seed20260930",
    "stress": "mustang_stress_heavy_8x_seed20260930",
}

# Greenfilling's alpha: the smoothing coefficient of the exponential moving
# average of the intensity signal. Backfilling is allowed only while the
# current intensity is at or below that average.
SMOOTHING_FACTOR = 0.5

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


def workload_path(regime):
    workload_id = RESAMPLED_WORKLOADS[regime]
    return f"{RESAMPLED_DIR}/{workload_id}/{workload_id}.json"


def missing_workloads():
    """Return the resampled workload paths that are not on disk yet."""
    paths = [workload_path(regime) for regime in RESAMPLED_WORKLOADS]
    return [p for p in paths if not os.path.isfile(os.path.join(HERE, p))]


def write_options(traces):
    os.makedirs(OPTIONS_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(OPTIONS_DIR, "*.json")):
        os.remove(stale)
    count = 0
    for cc, date in traces:
        for signal in SIGNALS:
            opts = {
                "intensity_trace": f"../intensities/traces/{cc}_{date}.csv",
                "intensity_zone": "AS0",
                "signal": signal,
                "smoothing_factor": SMOOTHING_FACTOR,
                "greenfilling_debug": False,
            }
            path = os.path.join(OPTIONS_DIR, f"greenfilling_{signal}_{cc}_{date}.json")
            with open(path, "w") as f:
                json.dump(opts, f, indent=4)
                f.write("\n")
            count += 1
    return count


def experiment(name, regime, trace, variant, options):
    return (
        "[[experiment]]\n"
        f'name = "{name}"\n'
        f'workload = "{workload_path(regime)}"\n'
        f'platform = "../workloads and platforms/{DATASET}.xml"\n'
        f'environmental_trace = "../intensities/traces/{trace}.csv"\n'
        f'variant_name = "{variant}"\n'
        f'variant_options = "{options}"\n'
    )


def write_toml(traces):
    blocks = []
    # Baseline EASY backfilling: schedule is independent of the intensity trace,
    # but kept per-trace so every experiment has a matching environmental trace.
    for regime in RESAMPLED_WORKLOADS:
        for cc, date in traces:
            blocks.append(experiment(
                f"easy_bf_{DATASET}_{regime}_{cc}_{date}",
                regime, f"{cc}_{date}", "easy_bf", "",
            ))
    # greenfilling: one variant per intensity signal.
    for signal in SIGNALS:
        for regime in RESAMPLED_WORKLOADS:
            for cc, date in traces:
                blocks.append(experiment(
                    f"greenfilling_{signal}_{DATASET}_{regime}_{cc}_{date}",
                    regime, f"{cc}_{date}", "greenfilling",
                    f"options/greenfilling_{signal}_{cc}_{date}.json",
                ))
    with open(TOML_OUT, "w") as f:
        f.write("\n".join(blocks))
    return len(blocks)


def main():
    traces = load_traces()
    n_opts = write_options(traces)
    n_exp = write_toml(traces)
    print(f"{len(traces)} windows x {len(RESAMPLED_WORKLOADS)} resampled workloads -> "
          f"{n_opts} option files in options/, {n_exp} experiments in experiments.toml")
    for path in missing_workloads():
        print(f"warning: workload not found, run analysis/mustang_user_resampling.ipynb: {path}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
