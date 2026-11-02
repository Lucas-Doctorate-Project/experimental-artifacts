"""Compare Greenfilling with the green-window oracle: python analysis/scripts/oracle.py."""

import argparse
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from opportunity import BASELINE_GROUP, bound_curves, policy_label, realized_points
from oracle_plots import save_oracle_figures

ORACLE_GROUP = "green-window-scheduling"
# the oracle campaign was interrupted and resumed into a second output directory
ORACLE_OUTPUTS = ("green-window-scheduling", "green-window-scheduling-resume")


def oracle_output(name: str) -> str:
    """Output directory holding one oracle run. Example: oracle_output('green_window_scheduling_carbon_h21600_mustang_slack_DE_2019-02-04')."""
    found = [d for d in ORACLE_OUTPUTS if (helpers.ROOT / "analysis/outputs" / d / name / "out_jobs.csv").exists()]
    if len(found) != 1:
        raise ValueError(f"oracle run {name!r} found in {found}, expected exactly one of {ORACLE_OUTPUTS}")
    return found[0]


class OracleSource(ModuleType):
    """helpers, with oracle runs read from whichever output directory holds them."""

    def __init__(self) -> None:
        super().__init__("oracle_source")
        # instance attributes shadow methods, so load_jobs is left to the class
        self.__dict__.update(
            {k: v for k, v in vars(helpers).items() if not k.startswith("__") and k != "load_jobs"}
        )

    def load_jobs(self, group: str, name: str, nodes: int) -> pd.DataFrame:
        return helpers.load_jobs(oracle_output(name) if group == ORACLE_GROUP else group, name, nodes)


def moved_intensity_change(easy: pd.DataFrame, policy: pd.DataFrame, signal: dict) -> float:
    """Mean intensity change of replay jobs that start at a different time than under EASY, in %."""
    replay = ~easy.is_context.to_numpy()
    duration = easy.execution_time.to_numpy()
    start_easy, start_policy = easy.starting_time.to_numpy(), policy.starting_time.to_numpy()
    moved = replay & (np.abs(start_policy - start_easy) > 1) & (duration > 0)
    if not moved.any():
        return np.nan
    (start_easy, start_policy, duration) = (start_easy[moved], start_policy[moved], duration[moved])
    before = helpers.integral(signal, start_easy, start_easy + duration) / duration
    after = helpers.integral(signal, start_policy, start_policy + duration) / duration
    return float(100 * (after.mean() / before.mean() - 1))


def paired_diagnostics(source: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Moved-job intensity per pair and class contributions for every Greenfilling and oracle run."""
    baseline = source.experiments(BASELINE_GROUP)
    easy_runs = baseline.loc[baseline.variant == "easy_bf"].set_index(["workload", "trace"])
    platforms = {path: source.platform(path) for path in baseline.platform.unique()}
    signals = {path: source.load_signals(path) for path in baseline.environmental_trace.unique()}
    groups = [
        (BASELINE_GROUP, baseline.loc[baseline.variant != "easy_bf"]),
        (ORACLE_GROUP, source.experiments(ORACLE_GROUP)),
    ]
    moved, classes, easy_jobs = [], [], {}
    for group, runs in groups:
        for _, run in runs.iterrows():
            machine = platforms[run.platform]
            easy_name = easy_runs.loc[(run.workload, run.trace), "name"]
            if easy_name not in easy_jobs:
                easy_jobs[easy_name] = source.load_jobs(BASELINE_GROUP, easy_name, machine["nodes"])
            easy = easy_jobs[easy_name]
            policy = source.load_jobs(group, run["name"], machine["nodes"])
            if not easy.job_id.equals(policy.job_id):
                raise ValueError(f"run {run['name']!r} and its EASY baseline contain different jobs")
            labels = {"policy": policy_label(run), "workload": run.workload, "trace": run.trace, "objective": run.objective}
            signal = signals[run.environmental_trace][run.objective]
            moved.append(labels | {"intensity_change_pct": moved_intensity_change(easy, policy, signal)})
            classes.append(helpers.class_contributions(easy, policy, machine).assign(**labels))
    return pd.DataFrame(moved), pd.concat(classes, ignore_index=True)


def main() -> None:
    """Render the oracle figures. Example: python analysis/scripts/oracle.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    source = OracleSource()
    savings = pd.concat(
        [
            realized_points(BASELINE_GROUP, BASELINE_GROUP, source),
            realized_points(ORACLE_GROUP, BASELINE_GROUP, source),
        ],
        ignore_index=True,
    )
    moved, classes = paired_diagnostics(source)
    save_oracle_figures(savings, bound_curves(BASELINE_GROUP), moved, classes, arguments.output_dir)
    print(f"Oracle figures saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
