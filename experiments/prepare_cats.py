"""Prepare the CATS carbon runs from the original campaign. Run from any directory."""

import csv
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EXPERIMENTS = Path(__file__).resolve().parent
INTENSITIES = EXPERIMENTS.parent / "intensities"
OUTPUT = Path("experimental-campaigns/cats-carbon")
ZONES = {"DE": "Europe/Berlin", "FR": "Europe/Paris", "PL": "Europe/Warsaw"}


def read_carbon(path):
    with path.open() as stream:
        return {
            int(row["timestamp"]): row["value"]
            for row in csv.DictReader(stream)
            if row["property"] == "carbon_intensity" and row["zone"] == "AS0"
        }


def write_campaign(name, rows):
    lines = [f'output_dir = "../analysis/outputs/cats-carbon-{name}"']
    for row in rows:
        lines.extend(["", "[[experiment]]"])
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in row.items())
    (EXPERIMENTS / OUTPUT / f"{name}.toml").write_text("\n".join(lines) + "\n")


def prepare():
    destination = EXPERIMENTS / OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    with (INTENSITIES / "windows.csv").open() as stream:
        windows = list(csv.DictReader(stream))
    for zone, tz in ZONES.items():
        full = read_carbon(INTENSITIES / "traces" / f"{zone}.csv")
        tz = ZoneInfo(tz)
        origin = datetime(2018, 1, 1, tzinfo=tz).timestamp()
        for row in (w for w in windows if w["zone"] == zone):
            key = f"{zone}_{row['start_date']}"
            offset = int(
                datetime.fromisoformat(row["start_date"]).replace(tzinfo=tz).timestamp()
                - origin
            )
            actual = read_carbon(INTENSITIES / row["file"])
            history = {t: full[offset + t] for t in range(-86400, 0, 900)}
            for t, value in actual.items():
                if not math.isclose(
                    float(value), float(full[offset + t]), abs_tol=1e-8
                ):
                    raise ValueError(f"{key}: regional trace mismatch at {t}")
            with (destination / f"{key}.csv").open("w") as stream:
                stream.write("timestamp,zone,property,value\n")
                for t, v in (history | actual).items():
                    stream.write(f"{t},AS0,carbon_intensity,{v}\n")
            for delay in (0, 21600):
                options = {
                    "history_file": str(OUTPUT / f"{key}.csv"),
                    "max_delay_seconds": delay,
                }
                (destination / f"{key}-{delay}.json").write_text(
                    json.dumps(options) + "\n"
                )

    baseline = tomllib.loads((EXPERIMENTS / "original.toml").read_text())["experiment"]
    full_runs, zero_delay_runs, pilot_runs = [], [], []
    for experiment in baseline:
        if experiment["variant_name"] != "easy_bf":
            continue
        key = Path(experiment["environmental_trace"]).stem
        cats = experiment | {
            "name": experiment["name"].replace("easy_bf_", "cats_carbon_", 1),
            "variant_name": "cats",
            "variant_options": str(OUTPUT / f"{key}-21600.json"),
        }
        full_runs.append(cats)
        if key == "DE_2019-02-04":
            zero_delay_runs.append(
                cats | {"variant_options": str(OUTPUT / f"{key}-0.json")}
            )
            pilot_runs.append(cats)
    if (len(full_runs), len(zero_delay_runs), len(pilot_runs)) != (144, 4, 4):
        raise ValueError("unexpected number of original workload/window pairs")
    for name, rows in (
        ("full", full_runs),
        ("zero", zero_delay_runs),
        ("pilot", pilot_runs),
    ):
        write_campaign(name, rows)

    # an isolated arrival and an inverted future signal distinguish lagged scoring from lookahead
    with (destination / "callback.csv").open("w") as stream:
        stream.write("timestamp,zone,property,value\n")
        for t in range(-86400, 86401, 900):
            clean = t in (-82800, -81900) if t < 0 else t < 1800
            stream.write(f"{t},AS0,carbon_intensity,{10 if clean else 100}\n")
    (destination / "callback.json").write_text(
        json.dumps(
            {"history_file": str(OUTPUT / "callback.csv"), "max_delay_seconds": 21600}
        )
        + "\n"
    )
    (destination / "single.json").write_text(
        json.dumps(
            {
                "nb_res": 1,
                "jobs": [
                    {
                        "id": "single",
                        "subtime": 1,
                        "res": 1,
                        "walltime": 900,
                        "profile": "one",
                    }
                ],
                "profiles": {"one": {"type": "delay", "delay": 900}},
            }
        )
        + "\n"
    )
    write_campaign(
        "callback",
        [
            pilot_runs[0]
            | {
                "name": "single",
                "workload": str(OUTPUT / "single.json"),
                "variant_options": str(OUTPUT / "callback.json"),
            }
        ],
    )


if __name__ == "__main__":
    prepare()
