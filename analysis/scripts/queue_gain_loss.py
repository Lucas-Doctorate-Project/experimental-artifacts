"""Split each policy's saving into jobs that ran cleaner and dirtier: python analysis/scripts/queue_gain_loss.py."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from delayed_jobs import easy_emissions, emission_change, load_pair, policy_tasks
from queue_gain_loss_plots import SIZES, save_queue_gain_loss_figure

WORKERS = 8
# a workload's large jobs are the biggest ones that make up this share of its node-hours,
# and its medium jobs the next ones up to the second share; the rest are small
SIZE_SHARES = (0.5, 0.9)


def size_thresholds(replay_node_hours: np.ndarray) -> tuple[float, float]:
    """Smallest node-hours of a large and of a medium job. Example: size_thresholds(np.array([1.0, 5.0, 100.0]))."""
    ordered = np.sort(replay_node_hours)[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    large, medium = (ordered[np.searchsorted(cumulative, share)] for share in SIZE_SHARES)
    return float(large), float(medium)


def job_sizes(easy: pd.DataFrame) -> np.ndarray:
    """Size class of each job, from the node-hours of the workload's replay jobs."""
    node_hours = (easy.requested_number_of_resources * easy.execution_time).to_numpy() / 3600
    large, medium = size_thresholds(node_hours[~easy.is_context.to_numpy()])
    return np.select([node_hours >= large, node_hours >= medium], ["large", "medium"], default="small")


def gain_and_loss(change: np.ndarray, total: float) -> dict:
    """Gain from jobs that ran cleaner and loss from jobs that ran dirtier, in % of total emissions."""
    return {
        "gain_pct": 100 * -change[change < 0].sum() / total,
        "loss_pct": 100 * change[change > 0].sum() / total,
        "net_pct": 100 * -change.sum() / total,
    }


def score_pair(task: dict) -> list[dict]:
    """Gain and loss of one run on real and unrelated intensity data, and by job size on real data."""
    machine, easy, policy = load_pair(task)
    sizes = job_sizes(easy)
    labels = {"policy": task["policy"], "workload": task["workload"], "trace": task["trace"]}
    rows = []
    for scoring, path in (("real", task["environmental_trace"]), ("unrelated", task["unrelated_trace"])):
        signal = helpers.load_signals(path)["carbon"]
        change = emission_change(easy, policy, machine, signal)
        total = easy_emissions(easy, machine, signal)
        rows.append(labels | {"scoring": scoring, "size": "all"} | gain_and_loss(change, total))
        if scoring == "real":
            for size in SIZES:
                rows.append(labels | {"scoring": scoring, "size": size} | gain_and_loss(change[sizes == size], total))
    return rows


def main() -> None:
    """Render the gain and loss figure. Example: python analysis/scripts/queue_gain_loss.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    with ProcessPoolExecutor(WORKERS) as pool:
        scores = pd.DataFrame([row for rows in pool.map(score_pair, policy_tasks()) for row in rows])
    by_size = scores.loc[scores["size"] != "all"].groupby(["policy", "workload", "trace"])["net_pct"].sum()
    whole = scores.loc[(scores["size"] == "all") & (scores.scoring == "real")].set_index(["policy", "workload", "trace"])
    gap = (by_size - whole.net_pct).abs().max()
    if gap > 1e-9:
        raise ValueError(f"size classes do not add up to the saving (gap {gap:g} points)")
    save_queue_gain_loss_figure(scores, arguments.output_dir)
    medians = scores.groupby(["policy", "scoring", "size"])[["gain_pct", "loss_pct", "net_pct"]].median()
    print(f"Medians in % of total emissions:\n{medians.round(2).to_string()}")
    print(f"Gain and loss figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
