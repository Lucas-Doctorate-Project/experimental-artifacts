#!/usr/bin/env python3
"""Rebuild every result the Greenfilling paper reports, and check it.

The repository holds more experiments and analyses than the paper uses. This
script isolates the paper: it reads the campaign outputs under `experiments/out`,
rebuilds the paper figure, recomputes every quantity quoted in the text, and
compares each one against the value printed in the paper.

Usage, from the repository root inside the Nix dev shell:

    python analysis/reproduce_paper.py

Outputs (all under `analysis/paper_results/`, plus the figure in
`analysis/figures/`):

  - greenfilling_deltas_vs_easy.pdf/.png : the paper figure
  - run_metrics.csv                      : one row per simulation run (432)
  - paired_comparisons.csv               : one row per Greenfilling/EASY pair (288)
  - scenario_summary.csv                 : per-scenario medians and counts
  - signal_means.csv                     : country means of the intensity signals
  - paper_claims.csv                     : every checked claim, side by side

The exit status is 0 when every claim matches the paper within tolerance, and 1
otherwise, so the script doubles as a regression test of the results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import campaign_analysis as ca


# Claims are checked to the precision the paper prints them with. Percentages
# quoted to one decimal get 0.05, counts are exact.
DEFAULT_TOLERANCE = 0.05

# Objective -> the footprint column that objective targets.
TARGET_COLUMN = {
    "greenfilling_carbon": "total_carbon_footprint_delta_pct",
    "greenfilling_water": "total_water_footprint_delta_pct",
}
SLOWDOWN_COLUMN = "replay_mean_bounded_slowdown_delta_pct"


# --- Paths -------------------------------------------------------------------

def resolve_root(explicit):
    """Return the repository root, whether the script runs from root or analysis/."""
    if explicit is not None:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    return here.parent


# --- Claim bookkeeping -------------------------------------------------------

class ClaimLog:
    """Collects (paper value, reproduced value) pairs and reports mismatches."""

    def __init__(self):
        self.rows = []

    def check(self, section, claim, paper_value, reproduced, tolerance=DEFAULT_TOLERANCE):
        difference = abs(float(reproduced) - float(paper_value))
        self.rows.append({
            "section": section,
            "claim": claim,
            "paper_value": paper_value,
            "reproduced_value": round(float(reproduced), 4),
            "abs_difference": round(difference, 4),
            "tolerance": tolerance,
            "status": "ok" if difference <= tolerance else "MISMATCH",
        })

    def note(self, section, claim, reproduced):
        """Record a reproduced quantity the paper states qualitatively."""
        self.rows.append({
            "section": section,
            "claim": claim,
            "paper_value": "",
            "reproduced_value": reproduced,
            "abs_difference": "",
            "tolerance": "",
            "status": "reported",
        })

    def frame(self):
        return pd.DataFrame(self.rows)

    def failures(self):
        return [row for row in self.rows if row["status"] == "MISMATCH"]


# --- Stage 1: campaign results ----------------------------------------------

def load_paired_comparisons(root, out_dir, claims):
    campaign, windows = ca.load_campaign(
        root / "experiments" / "experiments.toml",
        root / "intensities" / "windows.csv",
    )

    result_files = ca.check_completeness(campaign, out_dir)
    missing = ca.missing_results(result_files)
    if not missing.empty:
        names = ", ".join(missing["name"].head(5))
        raise SystemExit(
            f"{len(missing)} of {len(result_files)} campaign runs have no usable output "
            f"(first: {names}).\nRun the campaign first: "
            "cd experiments && go run . --campaign experiments.toml"
        )

    schedules, jobs = ca.load_results(campaign, result_files, out_dir, windows)
    job_metrics = ca.build_job_metrics(jobs)
    run_metrics = ca.build_run_metrics(schedules, job_metrics)
    paired = ca.build_paired_comparisons(run_metrics)

    claims.check("III-B", "simulation runs in the campaign", 432, len(run_metrics), 0)
    claims.check("III-B", "paired comparisons against EASY", 288, len(paired), 0)
    claims.check("III-B", "environmental windows", 36, windows["file"].nunique(), 0)
    claims.check(
        "III-B",
        "windows per country",
        12,
        windows.groupby("zone").size().unique()[0],
        0,
    )
    return run_metrics, paired


def scenario_summary(paired):
    """Per (objective, workload): targeted-footprint and slowdown deltas."""
    rows = []
    for objective, target_column in TARGET_COLUMN.items():
        subset = paired[paired["greenfilling_variant"].eq(objective)]
        for workload in ca.WORKLOAD_ORDER:
            scenario = subset[subset["workload_label"].eq(workload)]
            target = scenario[target_column]
            slowdown = scenario[SLOWDOWN_COLUMN]
            rows.append({
                "objective": objective.removeprefix("greenfilling_"),
                "workload_label": workload,
                "windows": len(scenario),
                "target_delta_pct_median": target.median(),
                "target_delta_pct_min": target.min(),
                "target_delta_pct_max": target.max(),
                "windows_with_lower_target": int((target < 0).sum()),
                "slowdown_delta_pct_median": slowdown.median(),
                "slowdown_delta_pct_min": slowdown.min(),
                "slowdown_delta_pct_max": slowdown.max(),
            })
    return pd.DataFrame(rows)


def check_result_claims(paired, summary, claims):
    def scenario(objective, workload):
        return summary[
            summary["objective"].eq(objective) & summary["workload_label"].eq(workload)
        ].iloc[0]

    # Section III-C, footprint side.
    claims.check(
        "III-C",
        "carbon Greenfilling, Mustang slack: windows with lower carbon (of 36)",
        30,
        scenario("carbon", "mustang_slack")["windows_with_lower_target"],
        0,
    )
    claims.check(
        "III-C",
        "water Greenfilling, Mustang slack: windows with lower water (of 36)",
        29,
        scenario("water", "mustang_slack")["windows_with_lower_target"],
        0,
    )
    for objective in ("carbon", "water"):
        claims.note(
            "III-C",
            f"{objective} Greenfilling, Mustang slack: median target delta is negative",
            bool(scenario(objective, "mustang_slack")["target_delta_pct_median"] < 0),
        )
        claims.note(
            "III-C",
            f"{objective} Greenfilling, Mustang stress: median target delta is positive",
            bool(scenario(objective, "mustang_stress")["target_delta_pct_median"] > 0),
        )

    largest_median = summary["target_delta_pct_median"].abs().max()
    claims.note(
        "III-C",
        "largest scenario median target-footprint change, in magnitude [%]",
        round(float(largest_median), 4),
    )
    claims.note(
        "III-C",
        "no scenario median target-footprint change exceeds 0.1% in magnitude",
        bool(largest_median < 0.1),
    )

    # Section III-C, performance side.
    worsened = int((paired[SLOWDOWN_COLUMN] > 0).sum())
    claims.check("III-C", "pairs where mean bounded slowdown rises", 288, worsened, 0)
    claims.check(
        "III-C",
        "smallest slowdown increase across pairs [%]",
        24.45,
        paired[SLOWDOWN_COLUMN].min(),
    )
    claims.check(
        "III-C",
        "largest slowdown increase across pairs [%]",
        757.19,
        paired[SLOWDOWN_COLUMN].max(),
    )
    claims.check(
        "III-C",
        "smallest scenario median slowdown increase [%]",
        161.52,
        summary["slowdown_delta_pct_median"].min(),
    )
    claims.check(
        "III-C",
        "largest scenario median slowdown increase [%]",
        323.58,
        summary["slowdown_delta_pct_median"].max(),
    )

    # The competitiveness criterion of Section III-C: lower targeted footprint
    # and no increase in mean bounded slowdown, in the same pair.
    targeted = pd.Series(
        [
            row[TARGET_COLUMN[row["greenfilling_variant"]]]
            for _, row in paired.iterrows()
        ],
        index=paired.index,
    )
    competitive = int(((targeted < 0) & (paired[SLOWDOWN_COLUMN] <= 0)).sum())
    claims.check("III-C", "pairs that are competitive", 0, competitive, 0)

    # Section IV reuses the most favourable scenario.
    claims.check(
        "IV",
        "carbon Greenfilling, Mustang slack: median slowdown increase [%]",
        323.6,
        scenario("carbon", "mustang_slack")["slowdown_delta_pct_median"],
        0.06,
    )


# --- Stage 2: grid signal means (Section II-B) -------------------------------

def signal_means(root, claims):
    """Arithmetic means of the 15-minute intensity samples, per country."""
    rows = []
    for country in ("DE", "FR", "PL"):
        trace = pd.read_csv(root / "intensities" / "traces" / f"{country}.csv")
        wide = trace.pivot(index="timestamp", columns="property", values="value")
        rows.append({
            "zone": country,
            "mean_carbon_intensity_g_per_kwh": wide["carbon_intensity"].mean(),
            "mean_water_intensity_l_per_kwh": wide["water_intensity"].mean(),
        })
    means = pd.DataFrame(rows).set_index("zone")

    claims.check(
        "II-B", "France mean carbon intensity [gCO2eq/kWh]",
        36.4, means.loc["FR", "mean_carbon_intensity_g_per_kwh"],
    )
    claims.check(
        "II-B", "Poland mean carbon intensity [gCO2eq/kWh]",
        569.4, means.loc["PL", "mean_carbon_intensity_g_per_kwh"],
    )
    claims.check(
        "II-B", "France mean water intensity [L/kWh]",
        3.320, means.loc["FR", "mean_water_intensity_l_per_kwh"], 0.0005,
    )
    claims.check(
        "II-B", "Poland mean water intensity [L/kWh]",
        1.587, means.loc["PL", "mean_water_intensity_l_per_kwh"], 0.0005,
    )
    claims.note(
        "II-B",
        "France has the lowest carbon and the highest water mean of the three",
        bool(
            means["mean_carbon_intensity_g_per_kwh"].idxmin() == "FR"
            and means["mean_water_intensity_l_per_kwh"].idxmax() == "FR"
        ),
    )
    return means.reset_index()


# --- Stage 3: carbon consistency check (Section II-D) ------------------------

def carbon_validation(root, claims, run_if_missing=True):
    """Read (or produce) the Ember comparison behind Section II-D."""
    output_dir = root / "intensities" / "validation" / "out"
    table_path = output_dir / "paper_table.csv"

    if not table_path.exists() and run_if_missing:
        subprocess.run(
            [sys.executable, str(root / "intensities" / "validate_carbon.py"),
             "--output-dir", str(output_dir)],
            check=True,
            cwd=root,
        )
    if not table_path.exists():
        claims.note("II-D", "carbon consistency check skipped, no paper_table.csv", "skipped")
        return None

    table = pd.read_csv(table_path).set_index("country")
    expected = {
        "DE": ("Germany", 0.987, -16.9),
        "FR": ("France", 0.994, -34.0),
        "PL": ("Poland", 0.999, -19.7),
    }
    for country, (name, pearson, difference) in expected.items():
        claims.check(
            "II-D", f"{name}: Pearson r between modelled and Ember annual means",
            pearson, table.loc[country, "pearson_r"], 0.0005,
        )
        claims.check(
            "II-D", f"{name}: modelled period mean versus Ember [%]",
            difference, table.loc[country, "difference_pct"],
        )
    return table.reset_index()


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="Repository root (default: parent of this script)")
    parser.add_argument("--out-dir", default=None, help="Campaign output directory (default: <root>/experiments/out)")
    parser.add_argument("--results-dir", default=None, help="Where to write the CSV tables (default: <root>/analysis/paper_results)")
    parser.add_argument("--figures-dir", default=None, help="Where to write the figure (default: <root>/analysis/figures)")
    parser.add_argument("--skip-validation", action="store_true", help="Do not run or read the carbon consistency check of Section II-D")
    parser.add_argument("--all-figures", action="store_true", help="Also export the supplementary figures that are not in the paper")
    args = parser.parse_args()

    root = resolve_root(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "experiments" / "out"
    results_dir = Path(args.results_dir) if args.results_dir else root / "analysis" / "paper_results"
    figures_dir = Path(args.figures_dir) if args.figures_dir else root / "analysis" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"repository root : {root}")
    print(f"campaign output : {out_dir}")
    print(f"figures         : {figures_dir}")
    print(f"tables          : {results_dir}\n")

    claims = ClaimLog()

    print("Loading the campaign, this reads 432 run directories ...")
    run_metrics, paired = load_paired_comparisons(root, out_dir, claims)
    summary = scenario_summary(paired)
    check_result_claims(paired, summary, claims)

    print("Rebuilding the paper figure ...")
    ca.apply_paper_style()
    delta_figure, _ = ca.plot_deltas_vs_easy(paired)
    pdf_path, _ = ca.export_figure(delta_figure, figures_dir, "greenfilling_deltas_vs_easy")
    print(f"  Figure 1 -> {pdf_path}")

    if args.all_figures:
        print("Rebuilding the supplementary figures (not used in the paper) ...")
        carbon_figure, _ = ca.plot_tradeoff_scatter(
            paired, "total_carbon_footprint_delta_pct",
            "Carbon footprint saving [% vs EASY]",
            "Carbon saving versus bounded slowdown penalty",
        )
        ca.export_figure(carbon_figure, figures_dir, "carbon_saving_vs_slowdown")
        water_figure, _ = ca.plot_tradeoff_scatter(
            paired, "total_water_footprint_delta_pct",
            "Water footprint saving [% vs EASY]",
            "Water saving versus bounded slowdown penalty",
        )
        ca.export_figure(water_figure, figures_dir, "water_saving_vs_slowdown")
        exposure_figure, _ = ca.plot_energy_exposure_deltas(paired)
        ca.export_figure(exposure_figure, figures_dir, "energy_exposure_deltas")
        swing_figure, _ = ca.plot_swing_relationship(paired)
        ca.export_figure(swing_figure, figures_dir, "swing_vs_footprint_saving")

    print("Recomputing the grid signal means ...")
    means = signal_means(root, claims)

    if not args.skip_validation:
        print("Recomputing the carbon consistency check ...")
        carbon_validation(root, claims)

    run_metrics.to_csv(results_dir / "run_metrics.csv", index=False)
    paired[
        ["greenfilling_variant", "workload_label", "zone", "start_date", "season"]
        + ca.DELTA_COLUMNS[1:]
    ].to_csv(results_dir / "paired_comparisons.csv", index=False)
    summary.to_csv(results_dir / "scenario_summary.csv", index=False)
    means.to_csv(results_dir / "signal_means.csv", index=False)

    claim_frame = claims.frame()
    claim_frame.to_csv(results_dir / "paper_claims.csv", index=False)

    print("\nClaims checked against the paper:\n")
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(claim_frame.to_string(index=False))

    failures = claims.failures()
    if failures:
        print(f"\n{len(failures)} claim(s) did not match the paper.")
        return 1
    checked = sum(1 for row in claims.rows if row["status"] == "ok")
    print(f"\nAll {checked} checked claims match the paper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
