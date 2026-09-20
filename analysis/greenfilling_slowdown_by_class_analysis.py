"""Test which job classes drive Greenfilling's mean-slowdown penalty.

Use the complete four-workload Greenfilling campaign, whose experiment names
contain the dataset. Every replay job is checked against its source workload
before comparing the same job under Greenfilling and EASY.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import shift_ceiling_results as scr
import workload_characterization_analysis as wca


OUT_DIR = scr.REPOSITORY_ROOT / "experiments" / "out"
FIGURE_DIR = Path(__file__).with_name("figures")
EFFECT_FIGURE_STEM = "greenfilling_slowdown_by_class"
LATER_START_FIGURE_STEM = "greenfilling_later_starts_by_class"
OBJECTIVES = ("carbon", "water")
JOB_CLASSES = ("narrow-short", "other", "wide")
BOUNDED_SLOWDOWN_FLOOR_SECONDS = 10.0
JOB_COLUMNS = [
    "job_id",
    "requested_number_of_resources",
    "submission_time",
    "success",
    "final_state",
    "starting_time",
    "execution_time",
    "waiting_time",
    "turnaround_time",
]


def expected_manifest(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Inventory the 36 × 4 × 3 complete Greenfilling result files."""
    windows = scr.load_windows()
    if len(windows) != 36:
        raise ValueError(f"Expected 36 intensity windows, found {len(windows)}")
    rows = []
    for workload in scr.WORKLOAD_NAMES:
        dataset, regime = workload.split("_", maxsplit=1)
        for window in windows.itertuples(index=False):
            suffix = f"{dataset}_{regime}_{window.zone}_{window.start_date}"
            for variant in ("easy_bf", "greenfilling_carbon", "greenfilling_water"):
                name = f"{variant}_{suffix}"
                path = out_dir / name / "out_jobs.csv"
                rows.append(
                    {
                        "workload": workload,
                        "dataset": dataset,
                        "regime": regime,
                        "zone": window.zone,
                        "start_date": window.start_date,
                        "variant": variant,
                        "name": name,
                        "has_jobs": path.is_file(),
                        "jobs_path": path,
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest["name"].duplicated().any():
        raise ValueError("Duplicate expected experiment names")
    return manifest


def completeness_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Show the number of available job CSVs for each workload and variant."""
    return manifest.groupby(["workload", "variant"], sort=False).agg(
        expected=("has_jobs", "size"), available=("has_jobs", "sum")
    )


def _read_replay(path: Path, expected: pd.DataFrame) -> pd.DataFrame:
    """Load one output and verify its replay jobs match the raw workload."""
    jobs = pd.read_csv(path, usecols=JOB_COLUMNS, dtype={"job_id": "string"})
    jobs = jobs.loc[~jobs["job_id"].str.startswith("ctx_")].copy()
    if jobs["job_id"].duplicated().any() or len(jobs) != len(expected):
        raise ValueError(f"Duplicate or missing replay jobs in {path}")
    if set(jobs["job_id"]) != set(expected.index):
        raise ValueError(f"Replay job IDs do not match source workload in {path}")
    jobs = jobs.set_index("job_id").loc[expected.index]
    if not np.array_equal(
        jobs["requested_number_of_resources"].to_numpy(),
        expected["nodes"].to_numpy(),
    ):
        raise ValueError(f"Job node requests do not match source workload in {path}")
    if not np.allclose(
        jobs["execution_time"], expected["runtime_seconds"], rtol=0, atol=1e-4
    ):
        raise ValueError(f"Job runtimes do not match source workload in {path}")
    valid_terminal_state = (
        (jobs["success"].eq(1) & jobs["final_state"].eq("COMPLETED_SUCCESSFULLY"))
        | (jobs["success"].eq(0) & jobs["final_state"].eq("COMPLETED_WALLTIME_REACHED"))
    )
    if not valid_terminal_state.all():
        raise ValueError(f"Unexpected replay-job terminal state in {path}")
    timing_columns = [
        "submission_time", "starting_time", "execution_time",
        "waiting_time", "turnaround_time",
    ]
    if not np.isfinite(jobs[timing_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite job timings in {path}")
    if not np.allclose(
        jobs["turnaround_time"], jobs["execution_time"] + jobs["waiting_time"],
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError(f"Turnaround does not equal execution plus waiting in {path}")
    jobs["bounded_slowdown"] = np.maximum(
        jobs["turnaround_time"]
        / jobs["execution_time"].clip(lower=BOUNDED_SLOWDOWN_FLOOR_SECONDS),
        1.0,
    )
    return jobs


def load_paired_results(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-window totals and exhaustive class contributions.

    The contribution for a class is sum(job-level slowdown deltas) / all replay
    jobs in that window. Its three class values add exactly to the change in
    overall mean bounded slowdown; no weighting or pooled-window shortcut is
    involved.
    """
    missing = manifest.loc[~manifest["has_jobs"], "name"]
    if not missing.empty:
        raise FileNotFoundError(
            f"Missing {len(missing)} campaign job outputs; first: {missing.iloc[0]}"
        )
    path_by_name = manifest.set_index("name")["jobs_path"]

    classified, _thresholds = wca.load_jobs()
    totals = []
    class_rows = []
    for workload in scr.WORKLOAD_NAMES:
        dataset, regime = workload.split("_", maxsplit=1)
        expected = classified.loc[
            classified["workload"].eq(workload),
            ["job_id", "runtime_seconds", "nodes", "narrow_short", "wide"],
        ].set_index("job_id")
        groups = np.select(
            [expected["narrow_short"], expected["wide"]],
            ["narrow-short", "wide"],
            default="other",
        )
        for window in scr.load_windows().itertuples(index=False):
            suffix = f"{dataset}_{regime}_{window.zone}_{window.start_date}"
            easy = _read_replay(path_by_name[f"easy_bf_{suffix}"], expected)
            for objective in OBJECTIVES:
                green = _read_replay(
                    path_by_name[f"greenfilling_{objective}_{suffix}"],
                    expected,
                )
                if not np.allclose(
                    easy["submission_time"], green["submission_time"],
                    rtol=0, atol=1e-6,
                ) or not np.allclose(
                    easy["execution_time"], green["execution_time"],
                    rtol=0, atol=1e-6,
                ):
                    raise ValueError(
                        f"Unmatched EASY/Greenfilling jobs: {suffix}, {objective}"
                    )

                easy_bsld = easy["bounded_slowdown"].to_numpy()
                green_bsld = green["bounded_slowdown"].to_numpy()
                delta = green_bsld - easy_bsld
                wait_delta = (
                    green["waiting_time"].to_numpy()
                    - easy["waiting_time"].to_numpy()
                )
                start_delta = (
                    green["starting_time"].to_numpy()
                    - easy["starting_time"].to_numpy()
                )
                easy_success = easy["success"].to_numpy() == 1
                green_success = green["success"].to_numpy() == 1
                both_success = easy_success & green_success
                total_delta = float(delta.mean())
                keys = {
                    "workload": workload,
                    "dataset": dataset,
                    "regime": regime,
                    "zone": window.zone,
                    "start_date": window.start_date,
                    "objective": objective,
                }
                easy_mean = float(easy["bounded_slowdown"].mean())
                totals.append(
                    {
                        **keys,
                        "n_jobs": len(expected),
                        "easy_mean_bsld": easy_mean,
                        "green_mean_bsld": float(green["bounded_slowdown"].mean()),
                        "delta_mean_bsld": total_delta,
                        "delta_mean_bsld_pct": 100 * total_delta / easy_mean,
                        "n_both_success": int(both_success.sum()),
                        "delta_mean_bsld_both_success": float(delta[both_success].mean()),
                        "easy_walltime_jobs": int((~easy_success).sum()),
                        "green_walltime_jobs": int((~green_success).sum()),
                        "changed_terminal_state_jobs": int(
                            (easy_success != green_success).sum()
                        ),
                        "later_start_jobs": int((start_delta > 1e-6).sum()),
                        "later_start_pct": 100 * float((start_delta > 1e-6).mean()),
                        "easy_median_wait_s": float(easy["waiting_time"].median()),
                        "green_median_wait_s": float(green["waiting_time"].median()),
                    }
                )
                for job_class in JOB_CLASSES:
                    mask = groups == job_class
                    later = mask & (start_delta > 1e-6)
                    contribution = float(delta[mask].sum() / len(delta))
                    class_rows.append(
                        {
                            **keys,
                            "job_class": job_class,
                            "n_jobs": int(mask.sum()),
                            "job_share_pct": 100 * float(mask.mean()),
                            "easy_mean_bsld": float(easy_bsld[mask].mean()),
                            "green_mean_bsld": float(green_bsld[mask].mean()),
                            "mean_delta_bsld": float(delta[mask].mean()),
                            "mean_runtime_hours": float(
                                expected["runtime_seconds"].to_numpy()[mask].mean()
                                / 3600
                            ),
                            "mean_wait_delta_hours": float(wait_delta[mask].mean() / 3600),
                            "contribution_bsld": contribution,
                            "contribution_share_pct": (
                                100 * contribution / total_delta
                                if not np.isclose(total_delta, 0, atol=1e-12)
                                else np.nan
                            ),
                            "worsened_jobs_pct": 100 * float(
                                (delta[mask] > 1e-9).mean()
                            ),
                            "later_start_pct": 100 * float(
                                later.sum() / mask.sum()
                            ),
                            "mean_later_delay_hours": (
                                float(start_delta[later].mean() / 3600)
                                if later.any() else np.nan
                            ),
                            "mean_later_runtime_hours": (
                                float(
                                    expected["runtime_seconds"].to_numpy()[later].mean()
                                    / 3600
                                )
                                if later.any() else np.nan
                            ),
                            "earlier_start_pct": 100 * float(
                                (start_delta[mask] < -1e-6).mean()
                            ),
                            "median_wait_delta_s": float(np.median(wait_delta[mask])),
                        }
                    )

    totals = pd.DataFrame(totals)
    by_class = pd.DataFrame(class_rows)
    summed = by_class.groupby(
        ["workload", "zone", "start_date", "objective"], sort=False
    )["contribution_bsld"].sum()
    observed = totals.set_index(["workload", "zone", "start_date", "objective"])[
        "delta_mean_bsld"
    ]
    if not np.allclose(summed.loc[observed.index], observed, rtol=1e-10, atol=1e-10):
        raise ValueError("Class contributions do not add up to overall mean slowdown")
    return totals, by_class


def summarize_totals(totals: pd.DataFrame) -> pd.DataFrame:
    """Median paired slowdown penalties over the sampled intensity windows."""
    return totals.groupby(["workload", "objective"], sort=False).agg(
        windows=("delta_mean_bsld", "size"),
        median_easy_bsld=("easy_mean_bsld", "median"),
        median_green_bsld=("green_mean_bsld", "median"),
        median_delta_bsld=("delta_mean_bsld", "median"),
        median_delta_pct=("delta_mean_bsld_pct", "median"),
        windows_with_penalty=("delta_mean_bsld", lambda x: int((x > 0).sum())),
        median_later_start_pct=("later_start_pct", "median"),
        median_both_success_delta=("delta_mean_bsld_both_success", "median"),
        both_success_penalty_windows=(
            "delta_mean_bsld_both_success", lambda x: int((x > 0).sum())
        ),
        median_state_changes=("changed_terminal_state_jobs", "median"),
    ).round(2)


def summarize_classes(by_class: pd.DataFrame) -> pd.DataFrame:
    """Class-level within-group effects and contributions, window by window."""
    return by_class.groupby(["workload", "objective", "job_class"], sort=False).agg(
        jobs=("n_jobs", "first"),
        job_share_pct=("job_share_pct", "first"),
        median_class_delta=("mean_delta_bsld", "median"),
        mean_runtime_hours=("mean_runtime_hours", "first"),
        median_mean_wait_delta_hours=("mean_wait_delta_hours", "median"),
        median_contribution=("contribution_bsld", "median"),
        median_share_of_penalty_pct=("contribution_share_pct", "median"),
        median_worsened_jobs_pct=("worsened_jobs_pct", "median"),
        median_later_start_pct=("later_start_pct", "median"),
        median_mean_later_delay_hours=("mean_later_delay_hours", "median"),
        median_mean_later_runtime_hours=("mean_later_runtime_hours", "median"),
        median_earlier_start_pct=("earlier_start_pct", "median"),
    ).round(2)


def narrow_short_majority(by_class: pd.DataFrame) -> pd.DataFrame:
    """Count windows where narrow-short contributes >50% of the total change."""
    selected = by_class.loc[by_class["job_class"].eq("narrow-short")]
    return selected.groupby(["workload", "objective"], sort=False).agg(
        windows=("contribution_share_pct", "size"),
        majority_windows=(
            "contribution_share_pct", lambda x: int((x > 50).sum())
        ),
        median_share_pct=("contribution_share_pct", "median"),
    ).round(2)


def _class_boxes(ax: plt.Axes, selected: pd.DataFrame, column: str) -> None:
    """Draw the same three-class grouping at both optimization objectives."""
    hatches = {"narrow-short": "///", "other": "", "wide": "xxx"}
    offsets = {"narrow-short": -0.23, "other": 0, "wide": 0.23}
    for objective_idx, objective in enumerate(OBJECTIVES):
        for job_class in JOB_CLASSES:
            values = selected.loc[
                selected["objective"].eq(objective)
                & selected["job_class"].eq(job_class),
                column,
            ].to_numpy()
            artists = ax.boxplot(
                values,
                positions=[objective_idx + offsets[job_class]],
                widths=0.19,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                boxprops={"edgecolor": "black", "linewidth": 0.7},
                whiskerprops={"color": "black", "linewidth": 0.7},
                capprops={"color": "black", "linewidth": 0.7},
            )
            artists["boxes"][0].set(facecolor="white", hatch=hatches[job_class])
    ax.set_xlim(-0.48, 1.48)
    ax.set_xticks([0, 1], ["Carbon", "Water"])


def _add_class_legend(fig: plt.Figure) -> None:
    hatches = {"narrow-short": "///", "other": "", "wide": "xxx"}
    fig.legend(
        handles=[
            Patch(facecolor="white", edgecolor="black", hatch=hatches[c], label=c)
            for c in JOB_CLASSES
        ],
        loc="upper center", ncol=3, frameon=False,
    )


def plot_class_effects(by_class: pd.DataFrame) -> plt.Figure:
    """Per-window contributions and within-class effects, without pooling jobs."""
    wca.configure_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 8.5), sharex=True)
    for row, workload in enumerate(scr.WORKLOAD_NAMES):
        selected = by_class.loc[by_class["workload"].eq(workload)]
        for col, column in enumerate(("contribution_bsld", "mean_delta_bsld")):
            ax = axes[row, col]
            _class_boxes(ax, selected, column)
            ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
            ax.set_title(
                f"{workload.replace('_', ' ').title()}: "
                + ("contribution" if col == 0 else "within-class change")
            )
            ax.set_ylabel("Bounded slowdown points")
    _add_class_legend(fig)
    fig.subplots_adjust(
        left=0.10, right=0.99, bottom=0.07, top=0.92, wspace=0.30, hspace=0.62
    )
    return fig


def plot_later_starts(by_class: pd.DataFrame) -> plt.Figure:
    """Fraction of jobs started later than the same job under EASY, per window."""
    wca.configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.2), sharex=True, sharey=True)
    for ax, workload in zip(axes.flat, scr.WORKLOAD_NAMES, strict=True):
        selected = by_class.loc[by_class["workload"].eq(workload)]
        _class_boxes(ax, selected, "later_start_pct")
        ax.set_title(workload.replace("_", " ").title())
        ax.set_ylabel("Jobs starting later than EASY [%]")
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
    _add_class_legend(fig)
    fig.subplots_adjust(
        left=0.11, right=0.99, bottom=0.12, top=0.86, wspace=0.25, hspace=0.53
    )
    return fig


def export_figure(
    fig: plt.Figure, stem: str, output_dir: Path = FIGURE_DIR
) -> list[Path]:
    """Save the result in the same PDF/PNG style as the other analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{suffix}" for suffix in ("pdf", "png")]
    fig.savefig(paths[0], format="pdf", dpi=600)
    fig.savefig(paths[1], format="png", dpi=600)
    return paths
