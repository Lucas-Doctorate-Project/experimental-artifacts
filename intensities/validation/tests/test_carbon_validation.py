"""Regression tests for the offline national carbon-intensity validation.

Run from the ``experimental-artifacts`` repository root with::

    python -m unittest discover -s intensities/validation/tests -v

The integration tests intentionally use the pinned raw ENTSO-E caches and
reference snapshots.  No network access is needed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


VALIDATION_DIR = Path(__file__).resolve().parents[1]
INTENSITIES_DIR = VALIDATION_DIR.parent
RAW_DIR = INTENSITIES_DIR / "raw"
TRACE_DIR = INTENSITIES_DIR / "traces"
REFERENCE_DIR = VALIDATION_DIR / "reference"

# ``intensities`` is not an importable package (and the repository itself is
# not installed), so load the validation module from its documented location.
sys.path.insert(0, str(VALIDATION_DIR))
import carbon_validation as validation  # noqa: E402


EXPECTED_STUDY_FACTORS = {
    "Biomass": 230.0,
    "Fossil Brown coal/Lignite": 1000.0,
    "Fossil Coal-derived gas": 630.0,
    "Fossil Gas": 280.0,
    "Fossil Hard coal": 630.0,
    "Fossil Oil": 280.0,
    "Geothermal": 38.0,
    "Hydro Run-of-river and poundage": 81.0,
    "Hydro Water Reservoir": 81.0,
    "Nuclear": 5.1,
    "Other": 280.0,
    "Other renewable": 230.0,
    "Solar": 21.0,
    "Waste": 230.0,
    "Wind Offshore": 13.0,
    "Wind Onshore": 12.0,
}

EXPECTED_NON_CCS_OVERRIDES = {
    "Fossil Coal-derived gas": 933.3333333333,
    "Fossil Gas": 430.0,
    "Fossil Hard coal": 933.3333333333,
    "Fossil Oil": 430.0,
    "Other": 430.0,
}

EXPECTED_PAPER_ROWS = {
    "DE": {
        "country_name": "Germany",
        "model_mean": 324.6,
        "reference_mean": 390.7,
        "difference_pct": -16.9,
        "pearson_r": 0.987,
    },
    "FR": {
        "country_name": "France",
        "model_mean": 38.5,
        "reference_mean": 58.3,
        "difference_pct": -34.0,
        "pearson_r": 0.994,
    },
    "PL": {
        "country_name": "Poland",
        "model_mean": 581.1,
        "reference_mean": 723.8,
        "difference_pct": -19.7,
        "pearson_r": 0.999,
    },
}

EXPECTED_ANNUAL_STUDY_INTENSITIES = {
    ("DE", 2018): 380.275331,
    ("DE", 2019): 323.041962,
    ("DE", 2020): 286.574724,
    ("DE", 2021): 326.013218,
    ("DE", 2022): 350.560924,
    ("DE", 2023): 306.913369,
    ("DE", 2024): 287.510315,
    ("FR", 2018): 39.970224,
    ("FR", 2019): 38.533639,
    ("FR", 2020): 39.581794,
    ("FR", 2021): 40.061058,
    ("FR", 2022): 49.373514,
    ("FR", 2023): 34.953052,
    ("FR", 2024): 28.527414,
    ("PL", 2018): 654.821306,
    ("PL", 2019): 623.350424,
    ("PL", 2020): 599.084260,
    ("PL", 2021): 607.079402,
    ("PL", 2022): 587.307123,
    ("PL", 2023): 516.316388,
    ("PL", 2024): 485.382251,
}


class AggregationAndFactorTests(unittest.TestCase):
    def test_generation_weighted_intensity_is_not_temporal_arithmetic_mean(self) -> None:
        # One dirty MWh followed by nine clean MWh.  The mean of the two
        # instantaneous intensities is 50, whereas total emissions divided by
        # total generation is 10; the national statistic must be the latter.
        generation = pd.DataFrame(
            {
                "dirty": [1.0, 0.0],
                "clean": [0.0, 9.0],
            }
        )
        factors = pd.Series({"dirty": 100.0, "clean": 0.0})

        actual = validation.generation_weighted_intensity(generation, factors)

        self.assertAlmostEqual(actual, 10.0, places=12)
        self.assertNotAlmostEqual(actual, 50.0, places=12)

    def test_factor_profiles_preserve_study_values_and_declared_sensitivity(self) -> None:
        factors = validation.load_factor_profiles()

        self.assertEqual(set(factors.columns), {"study", "non_ccs"})
        self.assertEqual(set(factors.index), set(EXPECTED_STUDY_FACTORS))
        self.assertFalse(factors.isna().any().any())

        for source, expected in EXPECTED_STUDY_FACTORS.items():
            with self.subTest(profile="study", source=source):
                self.assertAlmostEqual(float(factors.loc[source, "study"]), expected)

        for source, study_value in EXPECTED_STUDY_FACTORS.items():
            expected = EXPECTED_NON_CCS_OVERRIDES.get(source, study_value)
            with self.subTest(profile="non_ccs", source=source):
                self.assertAlmostEqual(
                    float(factors.loc[source, "non_ccs"]), expected, places=9
                )


class ReferenceSnapshotTests(unittest.TestCase):
    def test_committed_reference_files_have_the_pinned_schemas(self) -> None:
        ember = pd.read_csv(REFERENCE_DIR / "ember_2026_de_fr_pl.csv")
        eea = pd.read_csv(REFERENCE_DIR / "eea_ener038_de_fr_pl.csv")

        self.assertEqual(
            list(ember.columns),
            [
                "country",
                "country_name",
                "year",
                "record_type",
                "fuel",
                "generation_twh",
                "emissions_mtco2e",
                "intensity_gco2e_per_kwh",
                "status",
            ],
        )
        self.assertEqual(
            list(eea.columns),
            [
                "country",
                "country_name",
                "year",
                "intensity_gco2e_per_kwh",
                "status",
            ],
        )

    def test_ember_snapshot_has_one_total_and_nine_fuels_per_country_year(self) -> None:
        ember = pd.read_csv(REFERENCE_DIR / "ember_2026_de_fr_pl.csv")
        expected_fuels = {
            "Bioenergy",
            "Coal",
            "Gas",
            "Hydro",
            "Nuclear",
            "Other Fossil",
            "Other Renewables",
            "Solar",
            "Wind",
        }

        for (country, year), rows in ember.groupby(["country", "year"]):
            with self.subTest(country=country, year=year):
                totals = rows.loc[rows["record_type"] == "total"]
                fuels = rows.loc[rows["record_type"] == "fuel"]
                self.assertEqual(len(totals), 1)
                self.assertEqual(totals.iloc[0]["fuel"], "Total")
                self.assertEqual(set(fuels["fuel"]), expected_fuels)

                total = totals.iloc[0]
                calculated_intensity = (
                    1000.0
                    * float(total["emissions_mtco2e"])
                    / float(total["generation_twh"])
                )
                self.assertAlmostEqual(
                    float(total["intensity_gco2e_per_kwh"]),
                    calculated_intensity,
                    delta=0.1,
                )

    def test_normalized_references_cover_the_declared_country_years(self) -> None:
        ember = validation.load_reference_snapshot("ember", REFERENCE_DIR)
        eea = validation.load_reference_snapshot("eea", REFERENCE_DIR)

        required = {"country", "country_name", "year", "intensity", "source", "status"}
        self.assertTrue(required.issubset(ember.columns))
        self.assertTrue(required.issubset(eea.columns))

        self.assertEqual(set(ember["country"]), {"DE", "FR", "PL"})
        self.assertEqual(set(eea["country"]), {"DE", "FR", "PL"})

        ember_totals = ember
        if "record_type" in ember.columns:
            ember_totals = ember[ember["record_type"] == "total"]

        self.assertEqual(
            set(zip(ember_totals["country"], ember_totals["year"])),
            {(country, year) for country in ("DE", "FR", "PL") for year in range(2018, 2026)},
        )
        self.assertEqual(
            set(zip(eea["country"], eea["year"])),
            {(country, year) for country in ("DE", "FR", "PL") for year in range(2018, 2025)},
        )
        self.assertFalse(ember_totals["intensity"].isna().any())
        self.assertFalse(eea["intensity"].isna().any())

        self.assertTrue(
            (ember_totals.loc[ember_totals["year"] <= 2024, "status"] == "finalized").all()
        )
        self.assertTrue(
            (ember_totals.loc[ember_totals["year"] == 2025, "status"] == "provisional").all()
        )

        # EEA 2024 is deliberately kept separate from its finalized years.
        self.assertTrue((eea.loc[eea["year"] <= 2023, "status"] == "finalized").all())
        self.assertTrue((eea.loc[eea["year"] == 2024, "status"] == "early_estimate").all())

    def test_source_metadata_records_reproducibility_fields(self) -> None:
        metadata = json.loads((REFERENCE_DIR / "sources.json").read_text())

        self.assertIn("sources", metadata)
        self.assertGreaterEqual(len(metadata["sources"]), 2)
        for source in metadata["sources"]:
            with self.subTest(source=source.get("name")):
                self.assertTrue(source.get("url"))
                self.assertRegex(source.get("retrieved", ""), r"^\d{4}-\d{2}-\d{2}$")
                self.assertTrue(source.get("methodology_version"))
                self.assertTrue(source.get("license"))
                self.assertRegex(source.get("sha256", ""), r"^[0-9a-f]{64}$")
                self.assertRegex(
                    source.get("snapshot_sha256", ""), r"^[0-9a-f]{64}$"
                )
                snapshot_path = REFERENCE_DIR / source["snapshot"]
                snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                self.assertEqual(snapshot_hash, source["snapshot_sha256"])


class FullArtifactRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = validation.annual_model_intensities(
            RAW_DIR, profile="study", years=range(2018, 2025)
        )
        cls.ember = validation.load_reference_snapshot("ember", REFERENCE_DIR)
        cls.paper = validation.build_paper_table(
            model=cls.model,
            reference=cls.ember,
            years=range(2018, 2025),
        )

    def test_model_has_complete_2018_through_2024_calendar_years(self) -> None:
        self.assertTrue(
            {
                "country",
                "country_name",
                "year",
                "profile",
                "intensity",
                "generation_twh",
            }.issubset(self.model.columns)
        )
        self.assertEqual(len(self.model), 21)
        self.assertEqual(
            set(zip(self.model["country"], self.model["year"])),
            {(country, year) for country in ("DE", "FR", "PL") for year in range(2018, 2025)},
        )
        self.assertEqual(set(self.model["profile"]), {"study"})
        self.assertTrue((self.model["generation_twh"] > 0).all())
        self.assertFalse(self.model[["intensity", "generation_twh"]].isna().any().any())

    def test_all_annual_study_intensities_match_the_regression_baseline(self) -> None:
        actual = self.model.set_index(["country", "year"])["intensity"]
        self.assertEqual(set(actual.index), set(EXPECTED_ANNUAL_STUDY_INTENSITIES))
        for key, expected in EXPECTED_ANNUAL_STUDY_INTENSITIES.items():
            with self.subTest(country=key[0], year=key[1]):
                self.assertEqual(round(float(actual.loc[key]), 6), expected)

    def test_paper_table_is_recomputed_and_matches_rounded_regression_values(self) -> None:
        self.assertEqual(set(self.paper["country"]), set(EXPECTED_PAPER_ROWS))
        paper = self.paper.set_index("country")

        for country, expected in EXPECTED_PAPER_ROWS.items():
            with self.subTest(country=country):
                row = paper.loc[country]
                self.assertEqual(row["country_name"], expected["country_name"])
                self.assertEqual(round(float(row["model_mean"]), 1), expected["model_mean"])
                self.assertEqual(
                    round(float(row["reference_mean"]), 1), expected["reference_mean"]
                )
                self.assertEqual(
                    round(float(row["difference_pct"]), 1), expected["difference_pct"]
                )
                self.assertEqual(round(float(row["pearson_r"]), 3), expected["pearson_r"])

        self.assertGreater(paper.loc["PL", "model_mean"], paper.loc["DE", "model_mean"])
        self.assertGreater(paper.loc["DE", "model_mean"], paper.loc["FR", "model_mean"])

    def test_paper_metrics_are_derived_from_annual_inputs(self) -> None:
        paper = self.paper.set_index("country")
        reference = self.ember.loc[
            (self.ember["record_type"] == "total")
            & self.ember["year"].isin(range(2018, 2025))
        ]
        for country, annual in self.model.groupby("country"):
            expected_model = (
                annual["intensity"] * annual["generation_twh"]
            ).sum() / annual["generation_twh"].sum()
            external = reference.loc[reference["country"] == country]
            expected_reference = (
                1000.0
                * external["emissions_mtco2e"].sum()
                / external["generation_twh"].sum()
            )
            paired = annual[["year", "intensity"]].merge(
                external[["year", "intensity"]],
                on="year",
                suffixes=("_model", "_reference"),
                validate="one_to_one",
            )
            expected_r = paired["intensity_model"].corr(paired["intensity_reference"])
            with self.subTest(country=country):
                self.assertAlmostEqual(
                    float(paper.loc[country, "model_mean"]),
                    float(expected_model),
                    places=10,
                )
                self.assertAlmostEqual(
                    float(paper.loc[country, "reference_mean"]),
                    float(expected_reference),
                    places=10,
                )
                self.assertAlmostEqual(
                    float(paper.loc[country, "difference_pct"]),
                    100.0 * (float(expected_model) / float(expected_reference) - 1.0),
                    places=10,
                )
                self.assertAlmostEqual(
                    float(paper.loc[country, "pearson_r"]),
                    float(expected_r),
                    places=12,
                )

    def test_study_profile_reproduces_tracked_four_decimal_traces(self) -> None:
        result = validation.validate_trace_reproduction(
            raw_dir=RAW_DIR,
            trace_dir=TRACE_DIR,
            profile="study",
        )

        self.assertEqual(set(result["country"]), {"DE", "FR", "PL"})
        self.assertTrue(result["matched"].all())
        self.assertTrue((result["max_abs_error"] <= 0.0000501).all())


class EndToEndValidationTests(unittest.TestCase):
    def test_run_validation_writes_all_outputs_and_approved_paper_rows(self) -> None:
        expected_outputs = {
            "country_year",
            "summary",
            "decomposition",
            "trace_reproduction",
            "paper_table_csv",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = validation.run_validation(
                raw_dir=RAW_DIR,
                trace_dir=TRACE_DIR,
                reference_dir=REFERENCE_DIR,
                output_dir=temporary_directory,
            )

            self.assertEqual(set(outputs), expected_outputs)
            for name, output_path in outputs.items():
                with self.subTest(output=name):
                    self.assertTrue(output_path.is_file())
                    self.assertGreater(output_path.stat().st_size, 0)
            self.assertEqual(
                {path.name for path in Path(temporary_directory).iterdir()},
                {path.name for path in outputs.values()},
            )

            decomposition = pd.read_csv(outputs["decomposition"])
            harmonized = decomposition["ember_factor_harmonized_intensity"]
            self.assertFalse(harmonized.empty)
            self.assertTrue(harmonized.map(math.isfinite).all())

            paper = pd.read_csv(outputs["paper_table_csv"]).set_index("country")
            for country, expected in EXPECTED_PAPER_ROWS.items():
                with self.subTest(country=country):
                    self.assertEqual(
                        round(float(paper.loc[country, "model_mean"]), 1),
                        expected["model_mean"],
                    )
                    self.assertEqual(
                        round(float(paper.loc[country, "reference_mean"]), 1),
                        expected["reference_mean"],
                    )
                    self.assertEqual(
                        round(float(paper.loc[country, "difference_pct"]), 1),
                        expected["difference_pct"],
                    )
                    self.assertEqual(
                        round(float(paper.loc[country, "pearson_r"]), 3),
                        expected["pearson_r"],
                    )


if __name__ == "__main__":
    unittest.main()
