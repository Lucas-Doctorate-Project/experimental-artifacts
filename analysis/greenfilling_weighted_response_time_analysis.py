"""Compare area-weighted response time and mean bounded slowdown.

The paper's area is cores × runtime. In these node-based experiments we use
nodes × actual runtime; a constant number of cores per node would cancel from
the weighted mean. All calculations use matched replay jobs in one window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import greenfilling_slowdown_by_class_analysis as slowdown
import shift_ceiling_results as scr
import workload_characterization_analysis as wca


OBJECTIVES = slowdown.OBJECTIVES
JOB_CLASSES = slowdown.JOB_CLASSES
FIGURE_DIR = Path(__file__).with_name("figures")
SECONDS_PER_HOUR = 3600.0


def weighted_response_time_seconds(
    response_seconds: np.ndarray, area_node_seconds: np.ndarray
) -> float:
    """Return Σ(area × response time) / Σ(area), in seconds."""
    if len(response_seconds) != len(area_node_seconds) or len(response_seconds) == 0:
        raise ValueError("Response times and areas must have the same nonzero length")
    if not np.isfinite(response_seconds).all() or not np.isfinite(area_node_seconds).all():
        raise ValueError("Response times and areas must be finite")
    if np.any(response_seconds < 0) or np.any(area_node_seconds <= 0):
        raise ValueError("Response times must be nonnegative and areas positive")
    return float(np.average(response_seconds, weights=area_node_seconds))


def load_paired_metrics(
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute both metrics for every EASY/Greenfilling pair and job class."""
    missing = manifest.loc[~manifest["has_jobs"], "name"]
    if not missing.empty:
        raise FileNotFoundError(
            f"Missing {len(missing)} campaign job outputs; first: {missing.iloc[0]}"
        )
    paths = manifest.set_index("name")["jobs_path"]
    classified, _ = wca.load_jobs()
    totals: list[dict] = []
    class_rows: list[dict] = []

    for workload in scr.WORKLOAD_NAMES:
        dataset, regime = workload.split("_", maxsplit=1)
        expected = classified.loc[
            classified["workload"].eq(workload),
            ["job_id", "runtime_seconds", "nodes", "node_seconds", "narrow_short", "wide"],
        ].set_index("job_id")
        area = expected["node_seconds"].to_numpy(dtype=float)
        class_names = np.select(
            [expected["narrow_short"], expected["wide"]],
            ["narrow-short", "wide"], default="other",
        )
        for window in scr.load_windows().itertuples(index=False):
            suffix = f"{dataset}_{regime}_{window.zone}_{window.start_date}"
            easy = slowdown._read_replay(paths[f"easy_bf_{suffix}"], expected)
            easy_rt = easy["turnaround_time"].to_numpy(dtype=float)
            easy_bsld = easy["bounded_slowdown"].to_numpy(dtype=float)
            easy_wrt = weighted_response_time_seconds(easy_rt, area)
            easy_mean_bsld = float(easy_bsld.mean())

            for objective in OBJECTIVES:
                green = slowdown._read_replay(
                    paths[f"greenfilling_{objective}_{suffix}"], expected
                )
                if not np.allclose(
                    easy["submission_time"], green["submission_time"], rtol=0, atol=1e-6
                ) or not np.allclose(
                    easy["execution_time"], green["execution_time"], rtol=0, atol=1e-6
                ):
                    raise ValueError(f"Unmatched jobs: {suffix}, {objective}")

                green_rt = green["turnaround_time"].to_numpy(dtype=float)
                green_bsld = green["bounded_slowdown"].to_numpy(dtype=float)
                delta_rt = green_rt - easy_rt
                delta_wait = (
                    green["waiting_time"].to_numpy(dtype=float)
                    - easy["waiting_time"].to_numpy(dtype=float)
                )
                if not np.allclose(delta_rt, delta_wait, rtol=0, atol=2e-4):
                    raise ValueError(f"Response-time change differs from wait change: {suffix}")
                delta_bsld = green_bsld - easy_bsld
                green_wrt = weighted_response_time_seconds(green_rt, area)
                delta_wrt = green_wrt - easy_wrt
                if not np.isclose(
                    delta_wrt, np.dot(area, delta_rt) / area.sum(), rtol=1e-9, atol=1e-5
                ):
                    raise ValueError(f"Weighted response-time identity failed: {suffix}")

                both_success = (
                    easy["success"].to_numpy() == 1
                ) & (green["success"].to_numpy() == 1)
                keys = {
                    "workload": workload,
                    "zone": window.zone,
                    "start_date": window.start_date,
                    "objective": objective,
                }
                totals.append(
                    {
                        **keys,
                        "n_jobs": len(area),
                        "easy_mean_bsld": easy_mean_bsld,
                        "green_mean_bsld": float(green_bsld.mean()),
                        "delta_mean_bsld": float(delta_bsld.mean()),
                        "delta_mean_bsld_pct": 100 * float(delta_bsld.mean()) / easy_mean_bsld,
                        "easy_wrt_hours": easy_wrt / SECONDS_PER_HOUR,
                        "green_wrt_hours": green_wrt / SECONDS_PER_HOUR,
                        "delta_wrt_hours": delta_wrt / SECONDS_PER_HOUR,
                        "delta_wrt_pct": 100 * delta_wrt / easy_wrt,
                        "n_both_success": int(both_success.sum()),
                        "both_success_delta_bsld": float(delta_bsld[both_success].mean()),
                        "both_success_delta_wrt_hours": (
                            weighted_response_time_seconds(
                                green_rt[both_success], area[both_success]
                            )
                            - weighted_response_time_seconds(
                                easy_rt[both_success], area[both_success]
                            )
                        ) / SECONDS_PER_HOUR,
                    }
                )
                for job_class in JOB_CLASSES:
                    mask = class_names == job_class
                    class_area = area[mask]
                    class_delta_wrt = float(np.dot(class_area, delta_rt[mask]) / class_area.sum())
                    easy_class_wrt = weighted_response_time_seconds(easy_rt[mask], class_area)
                    class_rows.append(
                        {
                            **keys,
                            "job_class": job_class,
                            "n_jobs": int(mask.sum()),
                            "job_share_pct": 100 * float(mask.mean()),
                            "area_share_pct": 100 * float(class_area.sum() / area.sum()),
                            "easy_class_wrt_hours": easy_class_wrt / SECONDS_PER_HOUR,
                            "green_class_wrt_hours": (
                                easy_class_wrt + class_delta_wrt
                            ) / SECONDS_PER_HOUR,
                            "class_delta_wrt_hours": class_delta_wrt / SECONDS_PER_HOUR,
                            "class_delta_wrt_pct": 100 * class_delta_wrt / easy_class_wrt,
                            "contribution_wrt_hours": (
                                np.dot(class_area, delta_rt[mask]) / area.sum()
                            ) / SECONDS_PER_HOUR,
                            "easy_class_mean_bsld": float(easy_bsld[mask].mean()),
                            "green_class_mean_bsld": float(green_bsld[mask].mean()),
                            "class_delta_bsld": float(delta_bsld[mask].mean()),
                            "contribution_bsld": float(delta_bsld[mask].sum() / len(area)),
                        }
                    )

    totals_df = pd.DataFrame(totals)
    classes_df = pd.DataFrame(class_rows)
    keys = ["workload", "zone", "start_date", "objective"]
    by_window = classes_df.groupby(keys, sort=False)[
        ["contribution_wrt_hours", "contribution_bsld"]
    ].sum()
    matched = totals_df.set_index(keys)
    for contribution, delta in (
        ("contribution_wrt_hours", "delta_wrt_hours"),
        ("contribution_bsld", "delta_mean_bsld"),
    ):
        if not np.allclose(
            by_window.loc[matched.index, contribution], matched[delta],
            rtol=1e-9, atol=1e-8,
        ):
            raise ValueError(f"Class contributions do not sum to {delta}")
    return totals_df, classes_df


def summarize_totals(totals: pd.DataFrame) -> pd.DataFrame:
    """Summarize 36 paired windows per workload and objective."""
    with_seconds = totals.assign(
        delta_wrt_seconds=totals["delta_wrt_hours"] * SECONDS_PER_HOUR,
    )
    return with_seconds.groupby(["workload", "objective"], sort=False).agg(
        windows=("delta_wrt_hours", "size"),
        median_bsld_change_pct=("delta_mean_bsld_pct", "median"),
        median_easy_wrt_hours=("easy_wrt_hours", "median"),
        median_green_wrt_hours=("green_wrt_hours", "median"),
        median_wrt_change_seconds=("delta_wrt_seconds", "median"),
        median_wrt_change_pct=("delta_wrt_pct", "median"),
        bsld_worse_windows=("delta_mean_bsld", lambda x: int((x > 0).sum())),
        wrt_worse_windows=("delta_wrt_hours", lambda x: int((x > 0).sum())),
        both_success_wrt_worse_windows=(
            "both_success_delta_wrt_hours", lambda x: int((x > 0).sum())
        ),
    ).round(3)


def summarize_classes(classes: pd.DataFrame) -> pd.DataFrame:
    """Compare the two metrics' class weights and per-window contributions."""
    with_seconds = classes.assign(
        contribution_wrt_seconds=classes["contribution_wrt_hours"] * SECONDS_PER_HOUR,
    )
    return with_seconds.groupby(["workload", "objective", "job_class"], sort=False).agg(
        jobs=("n_jobs", "first"),
        job_share_pct=("job_share_pct", "first"),
        area_share_pct=("area_share_pct", "first"),
        median_class_bsld_change=("class_delta_bsld", "median"),
        median_bsld_contribution=("contribution_bsld", "median"),
        median_class_wrt_change_hours=("class_delta_wrt_hours", "median"),
        median_wrt_contribution_seconds=("contribution_wrt_seconds", "median"),
    ).round(3)


def compare_metric_signs(totals: pd.DataFrame) -> pd.DataFrame:
    """Count windows where slowdown and WRT prefer different schedulers."""
    frame = totals.assign(
        bsld_worse=totals["delta_mean_bsld"].gt(0),
        wrt_worse=totals["delta_wrt_hours"].gt(0),
    )
    return frame.groupby(["workload", "objective"], sort=False).agg(
        both_worse=("wrt_worse", lambda x: int((x & frame.loc[x.index, "bsld_worse"]).sum())),
        slowdown_worse_wrt_better=(
            "wrt_worse", lambda x: int((~x & frame.loc[x.index, "bsld_worse"]).sum())
        ),
        slowdown_better_wrt_worse=(
            "wrt_worse", lambda x: int((x & ~frame.loc[x.index, "bsld_worse"]).sum())
        ),
        both_better=(
            "wrt_worse", lambda x: int((~x & ~frame.loc[x.index, "bsld_worse"]).sum())
        ),
    )


def plot_total_changes(totals: pd.DataFrame) -> plt.Figure:
    """Show the paired relative changes on a common percentage scale."""
    wca.configure_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.7))
    symbols = {"carbon": "o", "water": "^"}
    for ax, workload in zip(axes.flat, scr.WORKLOAD_NAMES, strict=True):
        selected = totals.loc[totals["workload"].eq(workload)]
        for objective in OBJECTIVES:
            rows = selected.loc[selected["objective"].eq(objective)]
            ax.scatter(
                rows["delta_mean_bsld_pct"], rows["delta_wrt_pct"],
                marker=symbols[objective], s=22, facecolors="white",
                edgecolors="black", linewidths=0.65, alpha=0.8,
            )
        ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="0.4", linewidth=0.7, linestyle="--")
        ax.set_title(wca.WORKLOAD_LABELS[workload])
        ax.set_xlabel("Mean bounded slowdown change [%]")
        ax.set_ylabel("Weighted response time change [%]")
    fig.legend(
        handles=[Line2D([], [], color="black", marker=symbols[o], linestyle="None",
                        markerfacecolor="white", label=o.title()) for o in OBJECTIVES],
        loc="upper center", ncol=2, frameon=False,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.87,
                        wspace=0.25, hspace=0.46)
    return fig


def plot_class_contributions(classes: pd.DataFrame) -> plt.Figure:
    """Compare exact class contributions to each metric's total change."""
    wca.configure_plot_style()
    fig, axes = plt.subplots(4, 2, figsize=(7.16, 8.3), sharex=True)
    hatches = {"narrow-short": "///", "other": "", "wide": "xxx"}
    offsets = {"narrow-short": -0.23, "other": 0.0, "wide": 0.23}
    for row, workload in enumerate(scr.WORKLOAD_NAMES):
        selected = classes.loc[classes["workload"].eq(workload)]
        for col, value_col in enumerate(
            ("contribution_bsld", "contribution_wrt_hours")
        ):
            ax = axes[row, col]
            for objective_idx, objective in enumerate(OBJECTIVES):
                for job_class in JOB_CLASSES:
                    values = selected.loc[
                        selected["objective"].eq(objective)
                        & selected["job_class"].eq(job_class), value_col,
                    ].to_numpy()
                    artist = ax.boxplot(
                        values, positions=[objective_idx + offsets[job_class]],
                        widths=0.19, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 1.1},
                        boxprops={"edgecolor": "black", "linewidth": 0.7},
                        whiskerprops={"color": "black", "linewidth": 0.7},
                        capprops={"color": "black", "linewidth": 0.7},
                    )
                    artist["boxes"][0].set(facecolor="white", hatch=hatches[job_class])
            ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
            ax.set_xlim(-0.48, 1.48)
            ax.set_xticks([0, 1], ["Carbon", "Water"])
            ax.set_title(wca.WORKLOAD_LABELS[workload])
    fig.text(0.025, 0.5, "Slowdown contribution [points]", rotation=90,
             va="center", ha="center")
    fig.text(0.515, 0.5, "Weighted response contribution [h]", rotation=90,
             va="center", ha="center")
    fig.legend(
        handles=[Patch(facecolor="white", edgecolor="black", hatch=hatches[c], label=c)
                 for c in JOB_CLASSES],
        loc="upper center", ncol=3, frameon=False,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.07, top=0.91,
                        wspace=0.40, hspace=0.60)
    return fig


def export_figure(
    fig: plt.Figure, stem: str, output_dir: Path = FIGURE_DIR
) -> list[Path]:
    """Save matching PDF and high-resolution PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.{extension}" for extension in ("pdf", "png")]
    fig.savefig(paths[0], format="pdf", dpi=600)
    fig.savefig(paths[1], format="png", dpi=600)
    return paths
