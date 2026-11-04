"""Paired campaign metrics for the results analysis."""

from types import ModuleType

import numpy as np
import pandas as pd

import helpers

# original also accounts the oracle campaign, whose latest run ends at 36.76 days
HORIZONS = {
    "original": 37 * 86400,
    "resampled": 70 * 86400,
    "alpha-sensitivity": 35 * 86400,
}
METRICS = [
    "carbon_g",
    "water_l",
    "energy_kwh",
    "bounded_slowdown",
    "weighted_response_seconds",
    "median_wait_seconds",
    "p95_wait_seconds",
    "makespan_seconds",
]
RUN_LABELS = ["group", "workload", "trace", "variant", "objective", "alpha"]
COMPARISONS = [
    "target_delta_pct",
    "bounded_slowdown_delta_pct",
    "weighted_response_seconds_delta_pct",
]


def campaign_metrics(
    group: str, horizon: int, source: ModuleType = helpers
) -> pd.DataFrame:
    """Recalculate a campaign using common-interval accounting. Example: campaign_metrics('original', 3024000)."""
    config = source.experiments(group)
    # Read each shared trace and platform once within this calculation.
    signals = {
        path: source.load_signals(path) for path in config.environmental_trace.unique()
    }
    platforms = {path: source.platform(path) for path in config.platform.unique()}
    rows = []
    for _, experiment in config.iterrows():
        machine = platforms[experiment.platform]
        jobs = source.load_jobs(group, experiment["name"], machine["nodes"])
        metadata = experiment[RUN_LABELS].to_dict()
        totals = source.physical_totals(
            jobs, machine, signals[experiment.environmental_trace], horizon
        )
        rows.append(metadata | source.performance(jobs) | totals)
    return pd.DataFrame(rows)


def pair_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Compare each Greenfilling run with its campaign EASY. Example: pair_metrics(runs)."""
    case = ["group", "workload", "trace"]
    easy = runs.loc[runs.variant == "easy_bf", case + METRICS]
    pairs = runs.loc[runs.variant != "easy_bf"].merge(
        easy, on=case, suffixes=("_green", "_easy"), validate="many_to_one"
    )
    for metric in METRICS:
        pairs[metric + "_delta"] = pairs[metric + "_green"] - pairs[metric + "_easy"]
        pairs[metric + "_delta_pct"] = 100 * (
            pairs[metric + "_green"] / pairs[metric + "_easy"].replace(0, np.nan) - 1
        )
    pairs["target_delta_pct"] = np.where(
        pairs.objective == "carbon", pairs.carbon_g_delta_pct, pairs.water_l_delta_pct
    )
    return pairs


def _pareto_flags(cases: pd.DataFrame) -> pd.Series:
    values = cases[COMPARISONS[:2]].to_numpy()
    alternatives = np.vstack([values, [0.0, 0.0]])
    flags = [
        not np.any(
            np.all(alternatives <= value, axis=1) & np.any(alternatives < value, axis=1)
        )
        for value in values
    ]
    return pd.Series(flags, index=cases.index)


def alpha_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    """Compare smoothing factors against alpha 0.5 and EASY. Example: alpha_metrics(pairs)."""
    sensitivity = pairs.loc[pairs.group == "alpha-sensitivity"].copy()
    case = ["workload", "trace", "objective"]
    reference = sensitivity.loc[sensitivity.alpha == 0.5, case + COMPARISONS]
    sensitivity = sensitivity.merge(
        reference, on=case, suffixes=("", "_a050"), validate="many_to_one"
    )
    for column in COMPARISONS:
        sensitivity[column.replace("_delta_pct", "_difference_vs_a050_pp")] = (
            sensitivity[column] - sensitivity[column + "_a050"]
        )
    sensitivity["competitive"] = (sensitivity.target_delta_pct < 0) & (
        sensitivity.bounded_slowdown_delta_pct <= 0
    )
    sensitivity["pareto_among_alphas_and_easy"] = False
    for _, cases in sensitivity.groupby(case):
        flags = _pareto_flags(cases)
        sensitivity.loc[cases.index, "pareto_among_alphas_and_easy"] = flags
    return sensitivity
