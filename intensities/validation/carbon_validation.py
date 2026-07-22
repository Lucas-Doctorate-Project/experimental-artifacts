"""Reproducible external checks for the modeled carbon-intensity traces.

The standard run is deliberately offline: it reads the checked-in ENTSO-E
caches and the pinned, filtered Ember and EEA snapshots in ``reference/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


VALIDATION_DIR = Path(__file__).resolve().parent
INTENSITIES_DIR = VALIDATION_DIR.parent
DEFAULT_REFERENCE_DIR = VALIDATION_DIR / "reference"
DEFAULT_FACTOR_PATH = VALIDATION_DIR / "factors.csv"

COUNTRIES = {"DE": "Germany", "FR": "France", "PL": "Poland"}
COUNTRY_TIMEZONES = {
    "DE": "Europe/Berlin",
    "FR": "Europe/Paris",
    "PL": "Europe/Warsaw",
}
PRIMARY_YEARS = tuple(range(2018, 2025))
SUPPLEMENT_YEARS = tuple(range(2018, 2026))
STORAGE_SOURCES = {"Energy storage", "Hydro Pumped Storage"}
STEP_HOURS = 0.25

# This follows Ember's nine published fuel groups. The less obvious mappings
# preserve the assumptions already documented for the study traces: ENTSO-E's
# "Other renewable" is mostly biogas/small biomass; manufactured gases, oil,
# waste, and unclassified fossil output belong to Ember's Other Fossil group.
EMBER_FUEL_CROSSWALK = {
    "Biomass": "Bioenergy",
    "Other renewable": "Bioenergy",
    "Fossil Brown coal/Lignite": "Coal",
    "Fossil Hard coal": "Coal",
    "Fossil Coal-derived gas": "Other Fossil",
    "Fossil Gas": "Gas",
    "Fossil Oil": "Other Fossil",
    "Other": "Other Fossil",
    "Waste": "Other Fossil",
    "Geothermal": "Other Renewables",
    "Hydro Run-of-river and poundage": "Hydro",
    "Hydro Water Reservoir": "Hydro",
    "Nuclear": "Nuclear",
    "Solar": "Solar",
    "Wind Offshore": "Wind",
    "Wind Onshore": "Wind",
}


def load_factor_profiles(path: str | Path | None = None) -> pd.DataFrame:
    """Load the study and non-CCS factor profiles, indexed by ENTSO-E source."""

    factor_path = Path(path) if path is not None else DEFAULT_FACTOR_PATH
    factors = pd.read_csv(factor_path).set_index("source")
    expected_profiles = {"study", "non_ccs"}
    if set(factors.columns) != expected_profiles:
        raise ValueError(
            f"{factor_path} must contain exactly {sorted(expected_profiles)} profiles"
        )
    if factors.index.has_duplicates:
        raise ValueError(f"{factor_path} contains duplicate source names")
    if factors.isna().any().any():
        raise ValueError(f"{factor_path} contains missing factor values")
    return factors.astype(float)


def _factor_series(
    factors: pd.Series | Mapping[str, float] | pd.DataFrame,
) -> pd.Series:
    if isinstance(factors, pd.DataFrame):
        if factors.shape[1] != 1:
            raise ValueError("A factor DataFrame must contain exactly one profile column")
        factors = factors.iloc[:, 0]
    return pd.Series(factors, dtype=float)


def generation_weighted_intensity(
    generation: pd.DataFrame,
    factors: pd.Series | Mapping[str, float] | pd.DataFrame,
) -> float:
    """Return total modeled emissions divided by total modeled generation."""

    factor_series = _factor_series(factors)
    if (generation < 0).any().any():
        raise ValueError("Generation values must be non-negative")
    source_generation = generation.sum(axis="rows").astype(float)
    positive_sources = source_generation.loc[source_generation > 0].index
    missing = sorted(set(positive_sources) - set(factor_series.index))
    if missing:
        raise ValueError(f"Positive-generation sources lack carbon factors: {missing}")
    selected_factors = factor_series.reindex(positive_sources)
    undefined = sorted(selected_factors.index[selected_factors.isna()])
    if undefined:
        raise ValueError(
            f"Positive-generation sources have undefined carbon factors: {undefined}"
        )
    denominator = float(source_generation.sum())
    if denominator <= 0:
        raise ValueError("Generation must have a positive total")
    numerator = float(
        source_generation.reindex(positive_sources).dot(selected_factors)
    )
    return numerator / denominator


def _load_clean_generation(raw_path: Path, country: str) -> pd.DataFrame:
    """Reproduce the cleaning and 15-minute interpolation used by the notebook."""

    raw = pd.read_csv(raw_path, index_col=0)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(
        COUNTRY_TIMEZONES[country]
    )

    source_columns = [
        source
        for source in raw.columns
        if source not in STORAGE_SOURCES and (raw[source] > 0).any()
    ]
    generation = raw[source_columns].fillna(0).clip(lower=0)
    total = generation.sum(axis=1)
    artifacts = total < 0.2 * total.median()
    generation = generation.loc[~artifacts]
    generation = (
        generation.resample("15min").asfreq().interpolate(method="linear")
    )
    generation = generation.loc[generation.sum(axis=1) > 0]
    return generation


def _calendar_year_generation(
    generation: pd.DataFrame,
    country: str,
    year: int,
) -> pd.DataFrame:
    """Select and require a complete local-calendar 15-minute year."""

    annual = generation.loc[generation.index.year == year]
    timezone = COUNTRY_TIMEZONES[country]
    start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
    expected_index = pd.date_range(
        start=start,
        end=end,
        inclusive="left",
        freq="15min",
    )
    if not annual.index.equals(expected_index):
        missing = expected_index.difference(annual.index)
        unexpected = annual.index.difference(expected_index)
        raise ValueError(
            f"{country} does not provide a complete 15-minute calendar year "
            f"for {year}: {len(missing)} missing and {len(unexpected)} "
            "unexpected timestamps"
        )
    return annual


def annual_model_intensities(
    raw_dir: str | Path,
    profile: str = "study",
    years: Iterable[int] = SUPPLEMENT_YEARS,
    factors_path: str | Path | None = None,
) -> pd.DataFrame:
    """Compute generation-weighted country-year model intensities."""

    factors = load_factor_profiles(factors_path)
    if profile not in factors.columns:
        raise ValueError(f"Unknown factor profile {profile!r}")
    selected_years = tuple(int(year) for year in years)
    raw_root = Path(raw_dir)
    rows: list[dict[str, object]] = []

    for country, country_name in COUNTRIES.items():
        generation = _load_clean_generation(raw_root / f"{country}.csv", country)
        missing = sorted(set(generation.columns) - set(factors.index))
        if missing:
            raise ValueError(f"{country} has sources missing from factors.csv: {missing}")

        for year in selected_years:
            annual = _calendar_year_generation(generation, country, year)
            generation_twh = float(annual.to_numpy().sum()) * STEP_HOURS / 1_000_000
            rows.append(
                {
                    "country": country,
                    "country_name": country_name,
                    "year": year,
                    "profile": profile,
                    "intensity": generation_weighted_intensity(
                        annual, factors[profile]
                    ),
                    "generation_twh": generation_twh,
                }
            )

    return pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)


def load_reference_snapshot(
    source: str,
    reference_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load a pinned reference snapshot with normalized common columns."""

    root = Path(reference_dir) if reference_dir is not None else DEFAULT_REFERENCE_DIR
    if source == "ember":
        frame = pd.read_csv(root / "ember_2026_de_fr_pl.csv")
        frame = frame.rename(
            columns={"intensity_gco2e_per_kwh": "intensity"}
        )
    elif source == "eea":
        frame = pd.read_csv(root / "eea_ener038_de_fr_pl.csv")
        frame = frame.rename(
            columns={"intensity_gco2e_per_kwh": "intensity"}
        )
        frame["record_type"] = "total"
        frame["fuel"] = "Total"
        frame["generation_twh"] = pd.NA
        frame["emissions_mtco2e"] = pd.NA
    else:
        raise ValueError("source must be 'ember' or 'eea'")

    required = {"country", "country_name", "year", "intensity", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Pinned {source} snapshot lacks columns: {missing}")
    frame["source"] = source
    return frame.sort_values(["country", "year", "record_type", "fuel"]).reset_index(
        drop=True
    )


def _total_reference_rows(reference: pd.DataFrame) -> pd.DataFrame:
    if "record_type" in reference.columns:
        reference = reference.loc[reference["record_type"] == "total"]
    duplicates = reference.duplicated(["country", "year"])
    if duplicates.any():
        raise ValueError("Reference snapshot has duplicate country-year totals")
    return reference


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("Weights must have a positive total")
    return float((values * weights).sum() / denominator)


def _reference_period_intensity(reference: pd.DataFrame) -> float:
    """Return total reference emissions divided by total generation."""

    required = {"emissions_mtco2e", "generation_twh"}
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"Reference rows lack period-total fields: {missing}")
    generation_twh = float(reference["generation_twh"].sum())
    if generation_twh <= 0:
        raise ValueError("Reference generation must have a positive period total")
    return 1000 * float(reference["emissions_mtco2e"].sum()) / generation_twh


def build_paper_table(
    model: pd.DataFrame,
    reference: pd.DataFrame,
    years: Iterable[int] = PRIMARY_YEARS,
) -> pd.DataFrame:
    """Build the three-row manuscript table from annual model/reference values."""

    selected_years = tuple(int(year) for year in years)
    model = model.loc[
        (model["profile"] == "study")
        & model["year"].isin(selected_years)
    ]
    ember = _total_reference_rows(reference)
    ember = ember.loc[ember["year"].isin(selected_years)]

    rows: list[dict[str, object]] = []
    for country, country_name in COUNTRIES.items():
        modeled = model.loc[model["country"] == country].sort_values("year")
        reference = ember.loc[ember["country"] == country].sort_values("year")
        if tuple(modeled["year"]) != selected_years:
            raise ValueError(f"Model does not cover {country} in every primary year")
        if tuple(reference["year"]) != selected_years:
            raise ValueError(f"Ember does not cover {country} in every primary year")

        paired = modeled[["year", "intensity"]].merge(
            reference[["year", "intensity"]],
            on="year",
            suffixes=("_model", "_reference"),
            validate="one_to_one",
        )
        model_mean = _weighted_mean(
            modeled["intensity"], modeled["generation_twh"]
        )
        reference_mean = _reference_period_intensity(reference)
        rows.append(
            {
                "country": country,
                "country_name": country_name,
                "model_mean": model_mean,
                "reference_mean": reference_mean,
                "difference_pct": 100 * (model_mean / reference_mean - 1),
                "pearson_r": paired["intensity_model"].corr(
                    paired["intensity_reference"]
                ),
            }
        )

    return pd.DataFrame(rows)


def _comparison_summary(
    model_annual: pd.DataFrame,
    reference: pd.DataFrame,
    source: str,
    years: Iterable[int],
    period: str,
) -> pd.DataFrame:
    selected_years = tuple(int(year) for year in years)
    totals = _total_reference_rows(reference)
    totals = totals.loc[totals["year"].isin(selected_years)]
    rows: list[dict[str, object]] = []

    for profile in ("study", "non_ccs"):
        modeled_profile = model_annual.loc[
            (model_annual["profile"] == profile)
            & model_annual["year"].isin(selected_years)
        ]
        for country, country_name in COUNTRIES.items():
            modeled = modeled_profile.loc[
                modeled_profile["country"] == country
            ].sort_values("year")
            external = totals.loc[totals["country"] == country].sort_values("year")
            paired = modeled[["year", "intensity", "generation_twh"]].merge(
                external[["year", "intensity", "generation_twh"]],
                on="year",
                suffixes=("_model", "_reference"),
                validate="one_to_one",
            )
            if tuple(paired["year"]) != selected_years:
                raise ValueError(
                    f"{source} comparison lacks {country} coverage for {period}"
                )

            if source == "ember":
                model_mean = _weighted_mean(
                    paired["intensity_model"], paired["generation_twh_model"]
                )
                reference_mean = _reference_period_intensity(
                    totals.loc[totals["country"] == country]
                )
                mean_method = "total emissions divided by total generation"
                difference_pct = 100 * (model_mean / reference_mean - 1)
            else:
                # The pinned EEA chart provides annual intensities but not the
                # annual gross-generation denominators needed to construct a
                # period national mean. Keep EEA as an annual trend check
                # rather than silently substituting an arithmetic mean.
                model_mean = float("nan")
                reference_mean = float("nan")
                difference_pct = float("nan")
                mean_method = "not reported; EEA generation weights unavailable"

            relative = paired["intensity_model"] / paired["intensity_reference"] - 1
            rows.append(
                {
                    "country": country,
                    "country_name": country_name,
                    "profile": profile,
                    "source": source,
                    "period": period,
                    "years": len(selected_years),
                    "model_mean": model_mean,
                    "reference_mean": reference_mean,
                    "difference_pct": difference_pct,
                    "mean_annual_difference_pct": 100 * float(relative.mean()),
                    "mape_pct": 100 * float(relative.abs().mean()),
                    "pearson_r": paired["intensity_model"].corr(
                        paired["intensity_reference"]
                    ),
                    "spearman_rho": paired["intensity_model"].rank().corr(
                        paired["intensity_reference"].rank()
                    ),
                    "mean_method": mean_method,
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    model_annual: pd.DataFrame,
    ember_reference: pd.DataFrame,
    eea_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build primary and explicitly qualified supplementary summaries."""

    frames = [
        _comparison_summary(
            model_annual,
            ember_reference,
            "ember",
            PRIMARY_YEARS,
            "2018-2024 primary",
        ),
        _comparison_summary(
            model_annual,
            ember_reference,
            "ember",
            SUPPLEMENT_YEARS,
            "2018-2025 incl. provisional 2025",
        ),
        _comparison_summary(
            model_annual,
            eea_reference,
            "eea",
            range(2018, 2024),
            "2018-2023 finalized",
        ),
        _comparison_summary(
            model_annual,
            eea_reference,
            "eea",
            PRIMARY_YEARS,
            "2018-2024 incl. early estimate 2024",
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def build_country_year(
    model_annual: pd.DataFrame,
    ember_reference: pd.DataFrame,
    eea_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Return one auditable row per model profile, country, and year."""

    ember = _total_reference_rows(ember_reference)[
        ["country", "year", "intensity", "generation_twh", "status"]
    ].rename(
        columns={
            "intensity": "ember_intensity",
            "generation_twh": "ember_generation_twh",
            "status": "ember_status",
        }
    )
    eea = _total_reference_rows(eea_reference)[
        ["country", "year", "intensity", "status"]
    ].rename(columns={"intensity": "eea_intensity", "status": "eea_status"})
    result = model_annual.rename(
        columns={
            "intensity": "model_intensity",
            "generation_twh": "model_generation_twh",
        }
    ).merge(ember, on=["country", "year"], how="left", validate="many_to_one")
    return result.merge(
        eea, on=["country", "year"], how="left", validate="many_to_one"
    ).sort_values(["country", "year", "profile"])


def validate_trace_reproduction(
    raw_dir: str | Path,
    trace_dir: str | Path,
    profile: str = "study",
    factors_path: str | Path | None = None,
) -> pd.DataFrame:
    """Check that the validation pipeline reproduces tracked trace values."""

    factors = load_factor_profiles(factors_path)
    raw_root, trace_root = Path(raw_dir), Path(trace_dir)
    rows: list[dict[str, object]] = []

    for country in COUNTRIES:
        generation = _load_clean_generation(raw_root / f"{country}.csv", country)
        source_factors = factors[profile].reindex(generation.columns)
        intensity = generation.mul(source_factors, axis="columns").sum(axis=1)
        intensity = intensity / generation.sum(axis=1)
        expected_timestamps = (
            (generation.index - generation.index[0]).total_seconds().astype(int)
        )

        trace = pd.read_csv(trace_root / f"{country}.csv")
        trace = trace.loc[trace["property"] == "carbon_intensity"]
        length_matches = len(trace) == len(intensity)
        timestamps_match = length_matches and (
            trace["timestamp"].to_numpy() == expected_timestamps
        ).all()
        if length_matches:
            errors = (
                trace["value"].to_numpy() - intensity.round(4).to_numpy()
            )
            max_abs_error = float(abs(errors).max())
        else:
            max_abs_error = float("inf")
        rows.append(
            {
                "country": country,
                "max_abs_error": max_abs_error,
                "length_matches": length_matches,
                "timestamps_match": timestamps_match,
                "matched": length_matches
                and timestamps_match
                and max_abs_error <= 0.00005,
            }
        )

    return pd.DataFrame(rows)


def _aggregate_to_ember_fuels(generation: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(generation.columns) - set(EMBER_FUEL_CROSSWALK))
    if missing:
        raise ValueError(f"ENTSO-E sources lack an Ember fuel crosswalk: {missing}")
    fuels: dict[str, pd.Series] = {}
    for source, fuel in EMBER_FUEL_CROSSWALK.items():
        if source not in generation.columns:
            continue
        fuels[fuel] = fuels.get(fuel, 0) + generation[source]
    return pd.DataFrame(fuels)


def build_decomposition(
    raw_dir: str | Path,
    ember_reference: pd.DataFrame,
    factors_path: str | Path | None = None,
) -> pd.DataFrame:
    """Separate factor sensitivity from the residual activity-data mismatch.

    The residual after applying Ember's effective fuel factors is descriptive:
    it combines source coverage, the nine-fuel crosswalk, and net/gross basis.
    It must not be interpreted as a uniquely identified causal contribution.
    """

    factors = load_factor_profiles(factors_path)
    raw_root = Path(raw_dir)
    rows: list[dict[str, object]] = []

    for country, country_name in COUNTRIES.items():
        generation = _load_clean_generation(raw_root / f"{country}.csv", country)
        # Keep factor harmonization on the complete 2018-2024 benchmark
        # period. In Ember's provisional 2025 rows, some country/fuel pairs
        # have 0 TWh and 0 MtCO2e, so their effective factor is undefined.
        for year in PRIMARY_YEARS:
            annual = _calendar_year_generation(generation, country, year)
            fuel_generation = _aggregate_to_ember_fuels(annual)
            ember_year = ember_reference.loc[
                (ember_reference["country"] == country)
                & (ember_reference["year"] == year)
            ]
            ember_fuels = ember_year.loc[ember_year["record_type"] == "fuel"].copy()
            effective_factors = (
                1000
                * ember_fuels.set_index("fuel")["emissions_mtco2e"]
                / ember_fuels.set_index("fuel")["generation_twh"]
            )
            positive_fuels = fuel_generation.columns[
                fuel_generation.sum(axis=0) > 0
            ]
            missing = [
                fuel
                for fuel in positive_fuels
                if fuel not in effective_factors.index
                or pd.isna(effective_factors.loc[fuel])
            ]
            if missing:
                raise ValueError(
                    f"Ember lacks effective factors for {country} {year}: {missing}"
                )
            harmonized = generation_weighted_intensity(
                fuel_generation, effective_factors
            )
            ember_total = _total_reference_rows(ember_year).iloc[0]
            study = generation_weighted_intensity(annual, factors["study"])
            non_ccs = generation_weighted_intensity(annual, factors["non_ccs"])
            model_generation_twh = (
                float(annual.to_numpy().sum()) * STEP_HOURS / 1_000_000
            )
            ember_intensity = float(ember_total["intensity"])
            rows.append(
                {
                    "country": country,
                    "country_name": country_name,
                    "year": year,
                    "study_intensity": study,
                    "non_ccs_intensity": non_ccs,
                    "ember_factor_harmonized_intensity": harmonized,
                    "ember_intensity": ember_intensity,
                    "model_generation_twh": model_generation_twh,
                    "ember_generation_twh": float(ember_total["generation_twh"]),
                    "generation_coverage_pct": 100
                    * model_generation_twh
                    / float(ember_total["generation_twh"]),
                    "study_difference_pct": 100 * (study / ember_intensity - 1),
                    "non_ccs_difference_pct": 100
                    * (non_ccs / ember_intensity - 1),
                    "harmonized_difference_pct": 100
                    * (harmonized / ember_intensity - 1),
                    "status": ember_total["status"],
                }
            )
    return pd.DataFrame(rows).sort_values(["country", "year"])


def run_validation(
    raw_dir: str | Path,
    trace_dir: str | Path,
    reference_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Run every offline validation and write deterministic tabular outputs."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    model = pd.concat(
        [
            annual_model_intensities(raw_dir, profile="study"),
            annual_model_intensities(raw_dir, profile="non_ccs"),
        ],
        ignore_index=True,
    )
    ember = load_reference_snapshot("ember", reference_dir)
    eea = load_reference_snapshot("eea", reference_dir)

    country_year = build_country_year(model, ember, eea)
    summary = build_summary(model, ember, eea)
    decomposition = build_decomposition(raw_dir, ember)
    trace_check = validate_trace_reproduction(raw_dir, trace_dir)
    paper_table = build_paper_table(model, ember)

    outputs = {
        "country_year": output_root / "country_year.csv",
        "summary": output_root / "summary.csv",
        "decomposition": output_root / "decomposition.csv",
        "trace_reproduction": output_root / "trace_reproduction.csv",
        "paper_table_csv": output_root / "paper_table.csv",
    }
    country_year.to_csv(outputs["country_year"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    decomposition.to_csv(outputs["decomposition"], index=False)
    trace_check.to_csv(outputs["trace_reproduction"], index=False)
    paper_table.to_csv(outputs["paper_table_csv"], index=False)

    if not trace_check["matched"].all():
        failed = trace_check.loc[~trace_check["matched"], "country"].tolist()
        raise RuntimeError(f"Tracked trace reproduction failed for: {failed}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate modeled DE/FR/PL carbon intensities offline."
    )
    parser.add_argument("--raw-dir", type=Path, default=INTENSITIES_DIR / "raw")
    parser.add_argument(
        "--trace-dir", type=Path, default=INTENSITIES_DIR / "traces"
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR
    )
    parser.add_argument(
        "--output-dir", type=Path, default=VALIDATION_DIR / "out"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run_validation(
        args.raw_dir, args.trace_dir, args.reference_dir, args.output_dir
    )
    print(f"Validated carbon intensities; wrote {len(outputs)} files:")
    for path in outputs.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
