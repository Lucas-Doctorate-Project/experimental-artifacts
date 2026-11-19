"""Jobs that job shifting and CATS delayed on purpose, at the 6 h limit with known future intensity."""

import json

import numpy as np
import pandas as pd

import helpers
from opportunity import BASELINE_GROUP, HORIZON

POLICIES = ("Job shifting", "CATS")
# job shifting at 6 h, rerun with decision logs; its job exports equal the original 6 h runs
JOB_SHIFTING_GROUP = "green-window-scheduling-debug-6h"
CATS_GROUP = "forecast-cross-20261104/full"
LIMIT_SECONDS = 21600


def unrelated_traces(config: pd.DataFrame) -> dict[str, str]:
    """For each window, the next window of the same zone, cyclically. Its moves were not chosen for it."""
    traces = config[["zone", "trace", "environmental_trace"]].drop_duplicates().sort_values("trace")
    unrelated = {}
    for _, zone in traces.groupby("zone"):
        paths = zone.environmental_trace.tolist()
        for i, trace in enumerate(zone.trace):
            unrelated[trace] = paths[(i + 1) % len(paths)]
    return unrelated


def policy_tasks() -> list[dict]:
    """One task per policy and carbon case of the original campaign, with its EASY run and control window."""
    baseline = helpers.experiments(BASELINE_GROUP)
    easy = baseline.loc[baseline.variant == "easy_bf"]
    unrelated = unrelated_traces(easy)
    tasks = []
    for run in easy.to_dict("records"):
        case = f"{run['workload']}_{run['trace']}"
        common = {
            "workload": run["workload"],
            "trace": run["trace"],
            "platform": run["platform"],
            "environmental_trace": run["environmental_trace"],
            "unrelated_trace": unrelated[run["trace"]],
            "easy_name": run["name"],
        }
        tasks.append(
            common
            | {
                "policy": "Job shifting",
                "group": JOB_SHIFTING_GROUP,
                "name": f"green_window_scheduling_carbon_h{LIMIT_SECONDS}_{case}",
            }
        )
        tasks.append(common | {"policy": "CATS", "group": CATS_GROUP, "name": f"cats_known_future_{case}"})
    if len(tasks) != 2 * 144:
        raise ValueError(f"expected 144 original cases per policy, got {len(tasks) // 2}")
    return tasks


def decisions(task: dict) -> pd.DataFrame:
    """Each decided job's decision date and chosen start, indexed by job ID. Example: decisions(task).

    Job shifting: the last window choice in decisions.txt. Its date is the last
    `now` line before it, the date the job could have started without the delay.
    CATS: admissions.tsv, decided at submission and released at eligible_time.
    """
    folder = helpers.ROOT / "analysis/outputs" / task["group"] / task["name"]
    if task["policy"] == "CATS":
        admitted = pd.read_csv(folder / "admissions.tsv", sep="\t", dtype={"job_id": str})
        return admitted.set_index("job_id").rename(
            columns={"submission_time": "decided", "eligible_time": "begin"}
        )[["decided", "begin"]]
    date, last = 0.0, {}
    with (folder / "decisions.txt").open() as lines:
        for line in lines:
            if line.startswith("now "):
                date = float(line[4:])
                continue
            choice = json.loads(line)
            last[choice["job"].split("!", 1)[1]] = (date, choice["begin"])
    return pd.DataFrame.from_dict(last, orient="index", columns=["decided", "begin"])


def delayed_on_purpose(task: dict) -> pd.DataFrame:
    """Decided jobs whose chosen start is later than the decision date. Example: delayed_on_purpose(task)."""
    decided = decisions(task)
    return decided.loc[decided.begin > decided.decided + 1e-6]


def load_pair(task: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Platform, EASY jobs, and policy jobs of one task, both sorted by job ID."""
    machine = helpers.platform(task["platform"])
    easy = helpers.load_jobs(BASELINE_GROUP, task["easy_name"], machine["nodes"])
    policy = helpers.load_jobs(task["group"], task["name"], machine["nodes"])
    if not easy.job_id.equals(policy.job_id):
        raise ValueError(f"run {task['name']!r} and its EASY baseline contain different jobs")
    return machine, easy, policy


def easy_emissions(easy: pd.DataFrame, machine: dict, signal: dict) -> float:
    """Total emissions of the EASY run over the accounting horizon, idle nodes included."""
    return helpers.physical_totals(easy, machine, {"scored": signal}, HORIZON)["scored_l"]


def emission_change(easy: pd.DataFrame, policy: pd.DataFrame, machine: dict, signal: dict) -> np.ndarray:
    """Each job's emission change from EASY to the policy. Example: emission_change(easy, policy, machine, signal).

    A job's change is its nodes times the busy power above idle times the change
    in its intensity integral. Idle power is fixed, so the changes sum to the pair's
    change in total emissions.
    """
    increment = (machine["busy_w"] - machine["idle_w"]) / helpers.J_PER_KWH
    return (
        increment
        * easy.requested_number_of_resources.to_numpy()
        * (
            helpers.integral(signal, policy.starting_time.to_numpy(), policy.finish_time.to_numpy())
            - helpers.integral(signal, easy.starting_time.to_numpy(), easy.finish_time.to_numpy())
        )
    )
