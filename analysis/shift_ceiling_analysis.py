"""Loading, metrics, and plotting for the temporal-shifting ceiling notebook."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


IEEE_DOUBLE_COLUMN_WIDTH_INCHES = 7.16
IEEE_SINGLE_COLUMN_WIDTH_INCHES = 3.58
IEEE_LINE_ART_DPI = 600
HORIZON_LABELS = ["0h", "1h", "3h", "6h", "12h", "24h"]
NONZERO_HORIZON_LABELS = HORIZON_LABELS[1:]
SIGNALS = ["carbon", "water"]
WORKLOADS = [
    "mustang_stress",
    "mustang_slack",
    "trinity_stress",
    "trinity_slack",
]
WORKLOAD_LABELS = {
    "mustang_stress": "Mustang\nStress",
    "mustang_slack": "Mustang\nSlack",
    "trinity_stress": "Trinity\nStress",
    "trinity_slack": "Trinity\nSlack",
}
SEASONS = ["winter", "spring", "summer", "autumn"]
EXPECTED_FULL_ROWS = 10_338_192
EXPECTED_PAIR_COUNT = 144

PAIR_KEYS = [
    "workload",
    "platform",
    "zone",
    "intensity_trace",
    "season",
    "year",
    "d_m_label",
    "d_m_seconds",
    "signal",
]

RESULT_COLUMNS = PAIR_KEYS + [
    "submission_time_s",
    "t_star_s",
    "carbon_at_r_kg",
    "water_at_r_l",
    "carbon_at_tstar_kg",
    "water_at_tstar_l",
    "bsld",
]


@dataclass(frozen=True)
class CeilingAnalysis:
    pair_summary: pd.DataFrame
    at_24h: pd.DataFrame
    displaced_carbon: pd.DataFrame
    row_count: int


def configure_plot_style() -> None:
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "Roboto Condensed",
            "font.sans-serif": ["Roboto Condensed"],
            "font.size": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Roboto Condensed",
            "mathtext.it": "Roboto Condensed:italic",
            "mathtext.bf": "Roboto Condensed:bold",
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 10,
            "figure.dpi": 120,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.bottom": False,
            "axes.spines.left": False,
            "grid.color": "0.86",
            "grid.alpha": 0.8,
            "grid.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "lines.markersize": 4.5,
            "hatch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": IEEE_LINE_ART_DPI,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _summarize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    is_carbon = chunk["signal"].eq("carbon").to_numpy()
    baseline = np.where(
        is_carbon,
        chunk["carbon_at_r_kg"].to_numpy(),
        chunk["water_at_r_l"].to_numpy(),
    )
    shifted = np.where(
        is_carbon,
        chunk["carbon_at_tstar_kg"].to_numpy(),
        chunk["water_at_tstar_l"].to_numpy(),
    )
    moved = chunk["t_star_s"].to_numpy() > chunk["submission_time_s"].to_numpy()
    summary_source = chunk[PAIR_KEYS].copy()
    summary_source["baseline"] = baseline
    summary_source["shifted"] = shifted
    summary_source["jobs"] = 1
    summary_source["displaced_jobs"] = moved.astype(np.int64)
    summary_source["bsld_sum"] = chunk["bsld"].to_numpy()
    return (
        summary_source.groupby(PAIR_KEYS, sort=False, observed=True)
        .agg(
            baseline=("baseline", "sum"),
            shifted=("shifted", "sum"),
            jobs=("jobs", "sum"),
            displaced_jobs=("displaced_jobs", "sum"),
            bsld_sum=("bsld_sum", "sum"),
        )
        .reset_index()
    )


def _finish_summary(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["saving_pct"] = 100.0 * (result["baseline"] - result["shifted"]) / result[
        "baseline"
    ]
    result["displaced_pct"] = 100.0 * result["displaced_jobs"] / result["jobs"]
    result["mean_bsld"] = result["bsld_sum"] / result["jobs"]
    return result


def load_results(
    path: Path,
    chunksize: int = 500_000,
    expected_rows: int | None = EXPECTED_FULL_ROWS,
) -> CeilingAnalysis:
    path = Path(path)
    dtypes = {
        "workload": "category",
        "platform": "category",
        "zone": "category",
        "intensity_trace": "category",
        "season": "category",
        "year": "int16",
        "d_m_label": "category",
        "d_m_seconds": "int32",
        "signal": "category",
        "submission_time_s": "float64",
        "t_star_s": "float64",
        "carbon_at_r_kg": "float64",
        "water_at_r_l": "float64",
        "carbon_at_tstar_kg": "float64",
        "water_at_tstar_l": "float64",
        "bsld": "float64",
    }

    partial_summaries = []
    at_24h_parts = []
    displaced_carbon_parts = []
    row_count = 0
    reader = pd.read_csv(
        path,
        usecols=RESULT_COLUMNS,
        dtype=dtypes,
        chunksize=chunksize,
    )
    for chunk in reader:
        row_count += len(chunk)
        partial_summaries.append(_summarize_chunk(chunk))

        at_24h = chunk[chunk["d_m_label"].eq("24h")]
        if not at_24h.empty:
            at_24h_parts.append(
                at_24h[
                    [
                        "workload",
                        "signal",
                        "submission_time_s",
                        "t_star_s",
                        "bsld",
                    ]
                ].copy()
            )

        displaced_carbon = chunk[
            chunk["signal"].eq("carbon")
            & ~chunk["d_m_label"].eq("0h")
            & chunk["t_star_s"].gt(chunk["submission_time_s"])
        ]
        if not displaced_carbon.empty:
            displaced_carbon_parts.append(
                displaced_carbon[["d_m_label", "bsld"]].copy()
            )

    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"Read {row_count:,} rows, expected {expected_rows:,}")

    pair_summary = (
        pd.concat(partial_summaries, ignore_index=True)
        .groupby(PAIR_KEYS, sort=False, observed=True)
        .agg(
            baseline=("baseline", "sum"),
            shifted=("shifted", "sum"),
            jobs=("jobs", "sum"),
            displaced_jobs=("displaced_jobs", "sum"),
            bsld_sum=("bsld_sum", "sum"),
        )
        .reset_index()
    )
    pair_summary = _finish_summary(pair_summary)
    pair_summary["d_m_label"] = pd.Categorical(
        pair_summary["d_m_label"], categories=HORIZON_LABELS, ordered=True
    )

    pair_count = pair_summary[["workload", "intensity_trace"]].drop_duplicates().shape[0]
    if expected_rows is not None and pair_count != EXPECTED_PAIR_COUNT:
        raise ValueError(f"Found {pair_count} workload-window pairs, expected 144")

    at_24h = pd.concat(at_24h_parts, ignore_index=True)
    at_24h["displacement_h"] = (
        at_24h["t_star_s"] - at_24h["submission_time_s"]
    ) / 3600.0
    displaced_carbon = pd.concat(displaced_carbon_parts, ignore_index=True)
    displaced_carbon["d_m_label"] = pd.Categorical(
        displaced_carbon["d_m_label"],
        categories=NONZERO_HORIZON_LABELS,
        ordered=True,
    )
    return CeilingAnalysis(
        pair_summary=pair_summary,
        at_24h=at_24h,
        displaced_carbon=displaced_carbon,
        row_count=row_count,
    )


def aggregate_summary(pair_summary: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    summary = (
        pair_summary.groupby(groups, sort=False, observed=True)
        .agg(
            baseline=("baseline", "sum"),
            shifted=("shifted", "sum"),
            jobs=("jobs", "sum"),
            displaced_jobs=("displaced_jobs", "sum"),
            bsld_sum=("bsld_sum", "sum"),
        )
        .reset_index()
    )
    return _finish_summary(summary)


def headline_metrics(analysis: CeilingAnalysis) -> dict[str, object]:
    pair = analysis.pair_summary
    global_summary = aggregate_summary(pair, ["d_m_label", "signal"])
    at_24h_global = global_summary[global_summary["d_m_label"].eq("24h")]
    savings = at_24h_global.set_index("signal")["saving_pct"].to_dict()
    moved = at_24h_global.set_index("signal")["displaced_pct"].to_dict()

    at_24h_carbon = analysis.at_24h[analysis.at_24h["signal"].eq("carbon")]
    displaced = at_24h_carbon[at_24h_carbon["displacement_h"].gt(0.0)]
    carbon_pairs = pair[
        pair["d_m_label"].eq("24h") & pair["signal"].eq("carbon")
    ]
    correlation = carbon_pairs["saving_pct"].corr(carbon_pairs["mean_bsld"])

    zone = aggregate_summary(
        pair[pair["d_m_label"].eq("24h")], ["zone", "signal"]
    )
    season = aggregate_summary(
        pair[pair["d_m_label"].eq("24h")], ["season", "signal"]
    )
    return {
        "aggregate_saving_pct": savings,
        "displaced_jobs_pct": moved,
        "median_displacement_h_carbon": float(displaced["displacement_h"].median()),
        "mean_bsld_all_carbon": float(at_24h_carbon["bsld"].mean()),
        "mean_bsld_displaced_carbon": float(displaced["bsld"].mean()),
        "p99_bsld_displaced_carbon": float(displaced["bsld"].quantile(0.99)),
        "saving_bsld_correlation_carbon": float(correlation),
        "saving_by_zone_pct": zone.pivot(
            index="zone", columns="signal", values="saving_pct"
        ),
        "saving_by_season_pct": season.pivot(
            index="season", columns="signal", values="saving_pct"
        ),
    }


def _save_figure(fig: plt.Figure, output_dir: Path, name: str) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{name}.pdf"
    png_path = output_dir / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", dpi=IEEE_LINE_ART_DPI)
    fig.savefig(png_path, format="png", dpi=IEEE_LINE_ART_DPI)
    return [pdf_path, png_path]


def _boxplot(
    ax: plt.Axes,
    values: list[np.ndarray],
    positions: np.ndarray,
    widths: float,
    hatch: str | None = None,
) -> None:
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.1},
    )
    if hatch is not None:
        for box in boxes["boxes"]:
            box.set_hatch(hatch)


def _jitter_points(
    ax: plt.Axes,
    values: list[np.ndarray],
    positions: np.ndarray,
    width: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for position, group in zip(positions, values, strict=True):
        jitter = rng.uniform(-width, width, size=len(group))
        ax.scatter(
            position + jitter,
            group,
            s=4.5,
            facecolors="white",
            edgecolors="0.45",
            linewidths=0.35,
            zorder=3,
        )


def plot_ceiling_by_horizon(
    analysis: CeilingAnalysis,
    output_dir: Path,
) -> list[Path]:
    pair = analysis.pair_summary
    global_summary = aggregate_summary(pair, ["d_m_label", "signal"])
    x = np.arange(len(HORIZON_LABELS), dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.75),
        gridspec_kw={"wspace": 0.28},
    )

    for signal, marker, linestyle, label in [
        ("carbon", "o", "-", "Carbon"),
        ("water", "^", "--", "Water"),
    ]:
        series = (
            global_summary[global_summary["signal"].eq(signal)]
            .set_index("d_m_label")
            .reindex(HORIZON_LABELS)
        )
        axes[0].plot(
            x,
            series["saving_pct"],
            color="black",
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_title("Aggregate saving")
    axes[0].set_ylabel("Saving [% wrt. no shifting]")
    axes[0].set_xlabel(r"Maximum displacement $d_m$")
    axes[0].set_xticks(x, HORIZON_LABELS)
    axes[0].legend(loc="upper left", frameon=False)

    carbon = pair[pair["signal"].eq("carbon")]
    distributions = [
        carbon[carbon["d_m_label"].eq(label)]["saving_pct"].to_numpy()
        for label in HORIZON_LABELS
    ]
    _boxplot(axes[1], distributions, x, widths=0.48)
    _jitter_points(axes[1], distributions, x, width=0.11, seed=40)
    axes[1].set_title("Spread over the 144 workload-window pairs")
    axes[1].set_ylabel("Carbon saving [%]")
    axes[1].set_xlabel(r"Maximum displacement $d_m$")
    axes[1].set_xticks(x, HORIZON_LABELS)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.91)
    paths = _save_figure(fig, output_dir, "ceiling_by_horizon")
    plt.close(fig)
    return paths


def _paired_boxes(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    categories: list[str],
    labels: list[str],
    seed: int,
) -> None:
    x = np.arange(len(categories), dtype=float)
    carbon_positions = x - 0.17
    water_positions = x + 0.17
    carbon_values = [
        data[data[column].eq(category) & data["signal"].eq("carbon")][
            "saving_pct"
        ].to_numpy()
        for category in categories
    ]
    water_values = [
        data[data[column].eq(category) & data["signal"].eq("water")][
            "saving_pct"
        ].to_numpy()
        for category in categories
    ]
    _boxplot(ax, carbon_values, carbon_positions, widths=0.29)
    _boxplot(ax, water_values, water_positions, widths=0.29, hatch="//")
    _jitter_points(ax, carbon_values, carbon_positions, width=0.055, seed=seed)
    _jitter_points(ax, water_values, water_positions, width=0.055, seed=seed + 1)
    ax.set_xticks(x, labels)


def plot_ceiling_at_24h(
    analysis: CeilingAnalysis,
    output_dir: Path,
) -> list[Path]:
    data = analysis.pair_summary[analysis.pair_summary["d_m_label"].eq("24h")]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.9),
        sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    _paired_boxes(axes[0], data, "zone", ["DE", "FR", "PL"], ["DE", "FR", "PL"], 10)
    axes[0].set_title("By grid zone")
    axes[0].set_ylabel(r"Saving at $d_m = 24$h [%]")

    _paired_boxes(
        axes[1],
        data,
        "workload",
        WORKLOADS,
        [WORKLOAD_LABELS[name] for name in WORKLOADS],
        20,
    )
    axes[1].set_title("By workload")

    _paired_boxes(
        axes[2],
        data,
        "season",
        SEASONS,
        [season.title() for season in SEASONS],
        30,
    )
    axes[2].set_title("By season")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)

    legend = [
        Patch(facecolor="white", edgecolor="black", label="Carbon"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="Water"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.16, top=0.82)
    paths = _save_figure(fig, output_dir, "ceiling_at_24h")
    plt.close(fig)
    return paths


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0:
        raise ValueError("Cannot plot an empty empirical distribution")
    y = 100.0 * np.arange(1, len(x) + 1) / len(x)
    return x, y


def plot_displaced_jobs(
    analysis: CeilingAnalysis,
    output_dir: Path,
) -> list[Path]:
    global_summary = aggregate_summary(
        analysis.pair_summary, ["d_m_label", "signal"]
    )
    x = np.arange(len(HORIZON_LABELS), dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.75),
        gridspec_kw={"wspace": 0.28},
    )
    for signal, marker, linestyle, label in [
        ("carbon", "o", "-", "Carbon"),
        ("water", "^", "--", "Water"),
    ]:
        series = (
            global_summary[global_summary["signal"].eq(signal)]
            .set_index("d_m_label")
            .reindex(HORIZON_LABELS)
        )
        axes[0].plot(
            x,
            series["displaced_pct"],
            color="black",
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_title("Share of jobs that move")
    axes[0].set_ylabel("Jobs displaced [%]")
    axes[0].set_xlabel(r"Maximum displacement $d_m$")
    axes[0].set_xticks(x, HORIZON_LABELS)
    axes[0].legend(loc="upper left", frameon=False)

    for signal, marker, linestyle, label in [
        ("carbon", "o", "-", "Carbon"),
        ("water", "^", "--", "Water"),
    ]:
        displaced = analysis.at_24h[
            analysis.at_24h["signal"].eq(signal)
            & analysis.at_24h["displacement_h"].gt(0.0)
        ]["displacement_h"].to_numpy()
        cdf_x, cdf_y = _ecdf(displaced)
        axes[1].step(cdf_x, cdf_y, color="black", linestyle=linestyle)
        marker_indices = np.linspace(0, len(cdf_x) - 1, 11, dtype=int)
        axes[1].plot(
            cdf_x[marker_indices],
            cdf_y[marker_indices],
            color="black",
            linestyle="none",
            marker=marker,
            label=label,
        )
    axes[1].set_title(r"How far they move ($d_m = 24$h)")
    axes[1].set_ylabel("Displaced jobs [%]")
    axes[1].set_xlabel(r"Displacement $t^* - r_j$ [h]")
    axes[1].set_xlim(-0.3, 24.3)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.91)
    paths = _save_figure(fig, output_dir, "displaced_jobs")
    plt.close(fig)
    return paths


def plot_bsld_distribution(
    analysis: CeilingAnalysis,
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_COLUMN_WIDTH_INCHES, 2.75),
        gridspec_kw={"wspace": 0.28},
    )
    positions = np.arange(len(NONZERO_HORIZON_LABELS), dtype=float)
    values = [
        analysis.displaced_carbon[
            analysis.displaced_carbon["d_m_label"].eq(label)
        ]["bsld"].to_numpy()
        for label in NONZERO_HORIZON_LABELS
    ]
    _boxplot(axes[0], values, positions, widths=0.55)
    axes[0].set_title("Displaced jobs only (carbon objective)")
    axes[0].set_ylabel("Bounded slowdown")
    axes[0].set_xlabel(r"Maximum displacement $d_m$")
    axes[0].set_xticks(positions, NONZERO_HORIZON_LABELS)

    carbon = analysis.at_24h[analysis.at_24h["signal"].eq("carbon")]
    distributions = [
        (carbon["bsld"].to_numpy(), "All jobs", "o", "-"),
        (
            carbon[carbon["displacement_h"].gt(0.0)]["bsld"].to_numpy(),
            "Displaced only",
            "^",
            "--",
        ),
    ]
    for distribution, label, marker, linestyle in distributions:
        cdf_x, cdf_y = _ecdf(distribution)
        axes[1].step(cdf_x, cdf_y, color="black", linestyle=linestyle)
        marker_indices = np.linspace(0, len(cdf_x) - 1, 7, dtype=int)
        axes[1].plot(
            cdf_x[marker_indices],
            cdf_y[marker_indices],
            color="black",
            linestyle="none",
            marker=marker,
            label=label,
        )
    axes[1].set_title(r"Distribution at $d_m = 24$h")
    axes[1].set_ylabel("Jobs [%]")
    axes[1].set_xlabel("Bounded slowdown")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 103.0)
    axes[1].legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.91)
    paths = _save_figure(fig, output_dir, "bsld_distribution")
    plt.close(fig)
    return paths


def plot_saving_vs_bsld(
    analysis: CeilingAnalysis,
    output_dir: Path,
) -> list[Path]:
    data = analysis.pair_summary[
        analysis.pair_summary["d_m_label"].eq("24h")
        & analysis.pair_summary["signal"].eq("carbon")
    ]
    markers = {
        "mustang_stress": "o",
        "mustang_slack": "^",
        "trinity_stress": "s",
        "trinity_slack": "D",
    }
    fig, ax = plt.subplots(
        figsize=(IEEE_SINGLE_COLUMN_WIDTH_INCHES, 2.9),
    )
    legend_handles = []
    for workload in WORKLOADS:
        subset = data[data["workload"].eq(workload)]
        marker = markers[workload]
        ax.scatter(
            subset["saving_pct"],
            subset["mean_bsld"],
            marker=marker,
            s=24,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color="black",
                marker=marker,
                linestyle="none",
                markerfacecolor="white",
                label=WORKLOAD_LABELS[workload].replace("\n", " "),
            )
        )
    ax.set_title(r"Saving versus delay ($d_m = 24$h)")
    ax.set_xlabel("Carbon saving [%]")
    ax.set_ylabel("Mean bounded slowdown")
    ax.legend(handles=legend_handles, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.93)
    paths = _save_figure(fig, output_dir, "saving_vs_bsld")
    plt.close(fig)
    return paths


def render_all(analysis: CeilingAnalysis, output_dir: Path) -> list[Path]:
    configure_plot_style()
    paths = []
    paths.extend(plot_ceiling_by_horizon(analysis, output_dir))
    paths.extend(plot_ceiling_at_24h(analysis, output_dir))
    paths.extend(plot_displaced_jobs(analysis, output_dir))
    paths.extend(plot_bsld_distribution(analysis, output_dir))
    paths.extend(plot_saving_vs_bsld(analysis, output_dir))
    return paths
