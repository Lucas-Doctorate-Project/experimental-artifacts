"""Prepare the two missing policy/forecast combinations from the six-hour carbon campaigns."""

import json
import tomllib
from pathlib import Path

from prepare_cats import EXPERIMENTS
from prepare_cats import OUTPUT as CATS_INPUTS

OUTPUT = Path("experimental-campaigns/forecast-cross")


def prepare():
    destination = EXPERIMENTS / OUTPUT
    destination.mkdir(parents=True, exist_ok=False)
    cats_runs = tomllib.loads((EXPERIMENTS / CATS_INPUTS / "full.toml").read_text())[
        "experiment"
    ]
    head_runs = tomllib.loads(
        (EXPERIMENTS / "green-window-scheduling.toml").read_text()
    )["experiment"]
    heads_by_case = {
        row["name"].removeprefix("green_window_scheduling_carbon_h21600_"): row
        for row in head_runs
        if row["name"].startswith("green_window_scheduling_carbon_h21600_")
    }
    full, controls = [], []
    for cats_experiment in cats_runs:
        case = cats_experiment["name"].removeprefix("cats_carbon_")
        head_experiment = heads_by_case.pop(case)
        for key in ("workload", "platform", "environmental_trace"):
            if cats_experiment[key] != head_experiment[key]:
                raise ValueError(f"{case}: {key} differs")
        for row, policy, new_mode, old_mode in (
            (cats_experiment, "cats", "known_future", "previous_day"),
            (head_experiment, "head", "previous_day", "known_future"),
        ):
            options_path = EXPERIMENTS / row["variant_options"]
            options = json.loads(options_path.read_text())
            if policy == "head":
                options["intensity_trace"] = str(
                    CATS_INPUTS / Path(row["environmental_trace"]).name
                )
                options["green_window_scheduling_debug"] = True
            option_file = OUTPUT / f"{policy}_{new_mode}_{options_path.name}"
            (EXPERIMENTS / option_file).write_text(
                json.dumps(options | {"forecast": new_mode}) + "\n"
            )
            full.append(
                row
                | {
                    "name": f"{policy}_{new_mode}_{case}",
                    "variant_options": str(option_file),
                }
            )
            if case.endswith("DE_2019-02-04"):
                controls.append(row | {"name": f"{policy}_{old_mode}_{case}"})
    if heads_by_case or len(full) != 288 or len(controls) != 8:
        raise ValueError("incomplete comparison")
    for phase, rows in (("controls", controls), ("full", full)):
        lines = [f'output_dir = "../analysis/outputs/forecast-cross-{phase}"']
        for row in rows:
            lines.extend(["", "[[experiment]]"])
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in row.items())
        (destination / f"{phase}.toml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    prepare()
