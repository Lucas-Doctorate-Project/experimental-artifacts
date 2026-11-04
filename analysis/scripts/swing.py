"""Relate each method's saving to the daily swing of its window: python analysis/scripts/swing.py."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import helpers
from opportunity import BASELINE_GROUP, realized_points
from oracle import ORACLE_GROUP, OracleSource
from swing_plots import save_swing_figure

OFFLINE_RESULTS = helpers.ROOT / "analysis/outputs/offline-shifting/results.csv"
BUDGET_HOURS = 24
ORACLE_POLICY = "Oracle head shift 24 h"
KEYS = ["workload", "trace", "objective"]


def window_swings() -> pd.DataFrame:
    """Daily swing and zone of each window and objective, from the window-selection manifest.

    Swing is the mean daily peak-to-trough range over the 28-day window divided by
    the window mean, as computed in intensities/window_selection.ipynb.
    """
    windows = pd.read_csv(helpers.ROOT / "intensities/windows.csv")
    windows["trace"] = windows.zone + "_" + windows.start_date
    return pd.concat(
        [windows[["trace", "zone"]].assign(objective=objective, swing=windows[f"swing_{objective}"]) for objective in ("carbon", "water")],
        ignore_index=True,
    )


def method_savings(offline_results: Path) -> pd.DataFrame:
    """Saving against EASY of the delay-only bound, offline shifting, the oracle, and Greenfilling, with each window's swing."""
    offline = pd.read_csv(offline_results)
    offline = offline.loc[(offline.group == BASELINE_GROUP) & (offline.budget_hours == BUDGET_HOURS)]
    source = OracleSource()
    policies = pd.concat(
        [realized_points(BASELINE_GROUP, BASELINE_GROUP, source), realized_points(ORACLE_GROUP, BASELINE_GROUP, source)],
        ignore_index=True,
    )
    oracle = policies.loc[policies.policy == ORACLE_POLICY]
    greenfilling = policies.loc[policies.policy == "Greenfilling"]
    savings = pd.concat(
        [
            offline[KEYS].assign(method="bound", saving_pct=offline.bound_saving_pct),
            offline[KEYS].assign(method="offline", saving_pct=offline.saving_pct),
            oracle[KEYS].assign(method="oracle", saving_pct=oracle.saving_pct),
            greenfilling[KEYS].assign(method="greenfilling", saving_pct=greenfilling.saving_pct),
        ],
        ignore_index=True,
    )
    counts = savings.groupby("method").size()
    if (counts != len(offline)).any():
        raise ValueError(f"expected {len(offline)} pairs per method, got {counts.to_dict()}")
    return savings.merge(window_swings(), on=["trace", "objective"], validate="many_to_one")


def spearman(rows: pd.DataFrame) -> float:
    """Rank correlation between swing and saving. Example: spearman(savings.loc[savings.method == 'oracle'])."""
    return rows.saving_pct.rank().corr(rows.swing.rank())


def main() -> None:
    """Render the swing figure. Example: python analysis/scripts/swing.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    parser.add_argument(
        "--offline-results",
        type=Path,
        default=OFFLINE_RESULTS,
        help="results.csv from offline_shifting.py (default: analysis/outputs/offline-shifting/results.csv)",
    )
    arguments = parser.parse_args()
    savings = method_savings(arguments.offline_results)
    save_swing_figure(savings, arguments.output_dir)
    print("Spearman correlation of saving with swing, all pairs:")
    print(savings.groupby("method").apply(spearman).round(2).to_string())
    print("By zone:")
    print(savings.groupby(["method", "zone"]).apply(spearman).unstack().round(2).to_string())
    print(f"Swing figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
