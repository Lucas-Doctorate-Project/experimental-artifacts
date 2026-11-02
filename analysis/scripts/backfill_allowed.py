"""Show when Greenfilling allows backfilling: python analysis/backfill_allowed.py."""

import argparse
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

import helpers
from backfill_allowed_plots import save_backfill_allowed_figure

GROUP = "original"
STEP_SECONDS = 900
LEVELS = ["Q1", "Q2", "Q3", "Q4"]
DIRECTIONS = ["falling", "flat", "rising"]


def allowed_shares(jobs: pd.DataFrame, signal: dict, smoothing_factor: float) -> pd.DataFrame:
    """Time-weighted share with backfilling allowed, by intensity quartile and last trace step. Example: allowed_shares(jobs, signal, 0.5).

    The estimated permission and intensity are those at each scheduler event and
    hold until the next event. The span ends at the last replay submission.
    Quartiles use the trace samples within that span.
    """
    end = jobs.submission_time[~jobs.is_context].max()
    events, opened = helpers.reconstruct_gate(jobs, signal, end, smoothing_factor)
    durations = np.diff(np.r_[events, end])
    current = helpers.sample(signal, events)
    previous = helpers.sample(signal, np.maximum(events - STEP_SECONDS, 0))
    within = signal["values"][signal["times"] < end]
    level = np.searchsorted(np.quantile(within, [0.25, 0.5, 0.75]), current, side="right")
    direction = np.sign(current - previous).astype(int) + 1
    rows = []
    for kind, codes, names in (("level", level, LEVELS), ("direction", direction, DIRECTIONS)):
        for code, name in enumerate(names):
            selected = codes == code
            rows.append(
                {
                    "kind": kind,
                    "category": name,
                    "time_share_pct": 100 * durations[selected].sum() / durations.sum(),
                    "open_share_pct": 100
                    * helpers.divide(durations[selected & opened].sum(), durations[selected].sum()),
                }
            )
    return pd.DataFrame(rows)


def campaign_allowed_shares(group: str = GROUP, source: ModuleType = helpers) -> pd.DataFrame:
    """Allowed shares for every Greenfilling run in a campaign. Example: campaign_allowed_shares('original')."""
    config = source.experiments(group)
    signals = {path: source.load_signals(path) for path in config.environmental_trace.unique()}
    platforms = {path: source.platform(path) for path in config.platform.unique()}
    rows = []
    for _, run in config.loc[config.variant.str.startswith("greenfilling")].iterrows():
        jobs = source.load_jobs(group, run["name"], platforms[run.platform]["nodes"])
        signal = signals[run.environmental_trace][run.objective]
        shares = allowed_shares(jobs, signal, run.alpha)
        rows.append(shares.assign(workload=run.workload, trace=run.trace, objective=run.objective))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    """Render the backfilling figure. Example: python analysis/backfill_allowed.py --output-dir /tmp/results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="PNG/PDF destination (default: analysis/figures), existing figures are replaced",
    )
    arguments = parser.parse_args()
    save_backfill_allowed_figure(campaign_allowed_shares(), arguments.output_dir)
    print(f"Backfilling figure saved to {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
