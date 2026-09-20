"""Loading, classification, summaries, and plotting for workload characterization.

The narrow-short and wide predicates reproduce the criteria used to select
the four workload windows. They are not an exhaustive job-size taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import shift_ceiling_analysis as sa
import shift_ceiling_results as scr


NARROW_PLATFORM_FRACTION = 0.01
WIDE_PLATFORM_FRACTION = 0.10
SHORT_RUNTIME_SECONDS = 2 * 3_600
FIGURE_STEM = "workload_narrow_short_and_wide_shares"

WORKLOAD_LABELS = {
    "mustang_stress": "Mustang — Stress",
    "mustang_slack": "Mustang — Slack",
    "trinity_stress": "Trinity — Stress",
    "trinity_slack": "Trinity — Slack",
}
EXPECTED_REPLAY_JOBS = {
    "mustang_stress": 5_089,
    "mustang_slack": 10_683,
    "trinity_stress": 3_537,
    "trinity_slack": 4_622,
}
EXPECTED_CLASS_COUNTS = {
    "mustang_stress": (3_199, 235),
    "mustang_slack": (4_803, 272),
    "trinity_stress": (2_300, 524),
    "trinity_slack": (3_329, 563),
}


def configure_plot_style() -> None:
    """Use the analysis figure style, with a fallback for missing fonts."""
    sa.configure_plot_style()
    try:
        font_manager.findfont("Roboto Condensed", fallback_to_default=False)
    except ValueError:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["DejaVu Sans"],
                "mathtext.fontset": "dejavusans",
            }
        )


def load_jobs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and classify replay jobs, returning jobs and platform thresholds."""
    frames = []
    threshold_rows = []
    for workload_name in scr.WORKLOAD_NAMES:
        workload_path = scr.WORKLOAD_ROOT / f"{workload_name}.json"
        with workload_path.open(encoding="utf-8") as source:
            capacity_nodes = int(json.load(source)["nb_res"])

        workload = scr.load_workload(workload_name)
        frame = pd.DataFrame(
            {
                "workload": workload_name,
                "job_id": workload.job_ids,
                "runtime_seconds": workload.execution_time_s,
                "nodes": workload.nodes,
            }
        )
        frame["node_seconds"] = frame["nodes"] * frame["runtime_seconds"]
        frame["narrow_short"] = (
            frame["nodes"].le(NARROW_PLATFORM_FRACTION * capacity_nodes)
            & frame["runtime_seconds"].le(SHORT_RUNTIME_SECONDS)
        )
        frame["wide"] = frame["nodes"].ge(
            WIDE_PLATFORM_FRACTION * capacity_nodes
        )

        if (frame["narrow_short"] & frame["wide"]).any():
            raise ValueError(f"Classification overlap in {workload_name}")
        if frame["nodes"].gt(capacity_nodes).any():
            raise ValueError(f"Job exceeds platform capacity in {workload_name}")
        frames.append(frame)
        threshold_rows.append(
            {
                "workload": workload_name,
                "platform_nodes": capacity_nodes,
                "narrow_max_nodes": int(
                    np.floor(NARROW_PLATFORM_FRACTION * capacity_nodes)
                ),
                "short_max_runtime_hours": SHORT_RUNTIME_SECONDS / 3_600,
                "wide_min_nodes": int(
                    np.ceil(WIDE_PLATFORM_FRACTION * capacity_nodes)
                ),
            }
        )

    jobs = pd.concat(frames, ignore_index=True)
    jobs["workload"] = pd.Categorical(
        jobs["workload"], categories=scr.WORKLOAD_NAMES, ordered=True
    )
    thresholds = pd.DataFrame(threshold_rows).set_index("workload")
    validate_jobs(jobs)
    return jobs, thresholds


def validate_jobs(jobs: pd.DataFrame) -> None:
    """Check that classification covers the expected replay jobs only."""
    observed_counts = jobs.groupby("workload", observed=True).size().to_dict()
    if observed_counts != EXPECTED_REPLAY_JOBS:
        raise ValueError(f"Unexpected replay-job counts: {observed_counts}")
    if jobs["job_id"].astype(str).str.startswith("ctx_").any():
        raise ValueError("Context jobs must not enter the analysis")
    if jobs.duplicated(["workload", "job_id"]).any():
        raise ValueError("Job IDs must be unique within each workload")
    numeric = jobs[["runtime_seconds", "nodes", "node_seconds"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all() or np.any(numeric <= 0):
        raise ValueError("Runtime, nodes, and node-seconds must be finite and positive")


def summarize_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """Count selected jobs and their shares of jobs and node-seconds."""
    summary_rows = []
    for workload_name in scr.WORKLOAD_NAMES:
        selected = jobs[jobs["workload"].eq(workload_name)]
        total_node_seconds = selected["node_seconds"].sum()
        narrow_short = selected["narrow_short"]
        wide = selected["wide"]
        summary_rows.append(
            {
                "workload": workload_name,
                "replay_jobs": len(selected),
                "narrow_short_jobs": int(narrow_short.sum()),
                "narrow_short_job_share_pct": 100 * narrow_short.mean(),
                "narrow_short_node_seconds_share_pct": (
                    100 * selected.loc[narrow_short, "node_seconds"].sum()
                    / total_node_seconds
                ),
                "wide_jobs": int(wide.sum()),
                "wide_job_share_pct": 100 * wide.mean(),
                "wide_node_seconds_share_pct": (
                    100 * selected.loc[wide, "node_seconds"].sum()
                    / total_node_seconds
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).set_index("workload")
    observed_class_counts = {
        name: (int(row.narrow_short_jobs), int(row.wide_jobs))
        for name, row in summary.iterrows()
    }
    if observed_class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"Unexpected class counts: {observed_class_counts}")
    return summary


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Produce the readable table displayed by the notebook."""
    report = summary.rename(
        columns={
            "replay_jobs": "Replay jobs",
            "narrow_short_jobs": "Narrow-short jobs",
            "narrow_short_job_share_pct": "Narrow-short jobs [%]",
            "narrow_short_node_seconds_share_pct": "Narrow-short node-seconds [%]",
            "wide_jobs": "Wide jobs",
            "wide_job_share_pct": "Wide jobs [%]",
            "wide_node_seconds_share_pct": "Wide node-seconds [%]",
        }
    ).rename(index=WORKLOAD_LABELS)
    percentage_columns = [
        column for column in report.columns if column.endswith("[%]")
    ]
    report[percentage_columns] = report[percentage_columns].round(2)
    return report


def plot_shares(summary: pd.DataFrame) -> tuple[plt.Figure, np.ndarray]:
    """Compare equal-job and node-seconds-weighted shares for both predicates."""
    labels = [name.replace("_", "\n") for name in scr.WORKLOAD_NAMES]
    positions = np.arange(len(summary))
    bar_width = 0.36

    fig, axes = plt.subplots(
        1, 2, figsize=(sa.IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.75)
    )
    panels = [
        (
            "narrow_short_job_share_pct",
            "wide_job_share_pct",
            "Share of replay jobs [%]",
            "Equal weight per job",
        ),
        (
            "narrow_short_node_seconds_share_pct",
            "wide_node_seconds_share_pct",
            "Share of node-seconds [%]",
            "Weighted by nodes × runtime",
        ),
    ]
    for axis, (narrow_column, wide_column, ylabel, title) in zip(
        axes, panels, strict=True
    ):
        narrow_bars = axis.bar(
            positions - bar_width / 2,
            summary[narrow_column],
            width=bar_width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            label="Narrow-short",
        )
        wide_bars = axis.bar(
            positions + bar_width / 2,
            summary[wide_column],
            width=bar_width,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            hatch="///",
            label="Wide",
        )
        narrow_labels = [
            f"{value:.2f}" if value < 1 else f"{value:.1f}"
            for value in summary[narrow_column]
        ]
        axis.bar_label(narrow_bars, labels=narrow_labels, padding=2, fontsize=7)
        axis.bar_label(wide_bars, fmt="%.1f", padding=2, fontsize=7)
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 100)
        axis.set_yticks(np.arange(0, 101, 20))
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="x", visible=False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.20, top=0.78, wspace=0.34)
    return fig, axes


def export_figure(fig: plt.Figure, output_dir: Path) -> list[Path]:
    """Export the workload figure as vector PDF and 600 dpi PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    png_path = output_dir / f"{FIGURE_STEM}.png"
    fig.savefig(pdf_path, format="pdf", dpi=sa.IEEE_LINE_ART_DPI)
    fig.savefig(png_path, format="png", dpi=sa.IEEE_LINE_ART_DPI)
    return [pdf_path, png_path]
