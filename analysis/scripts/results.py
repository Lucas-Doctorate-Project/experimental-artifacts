"""Reproduce footprint and service results: python analysis/results.py."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from results_metrics import (
    HORIZONS,
    alpha_metrics,
    campaign_metrics,
    pair_metrics,
)
from results_plots import save_results_figures


def write_results(runs: pd.DataFrame, destination: Path) -> None:
    """Export figures, replacing previous versions in the destination. Example: write_results(runs, Path('/tmp/results'))."""
    destination.mkdir(parents=True, exist_ok=True)
    pairs = pair_metrics(runs)
    sensitivity = alpha_metrics(pairs)
    save_results_figures(pairs, sensitivity, destination)


def main() -> None:
    """Run all three campaigns. Example: python analysis/results.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    campaigns = []
    for group, horizon in HORIZONS.items():
        campaigns.append(campaign_metrics(group, horizon))
        print(f"{group}: {len(campaigns[-1])} runs")
    write_results(pd.concat(campaigns, ignore_index=True), arguments.output_dir)
    print(f"Results saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
