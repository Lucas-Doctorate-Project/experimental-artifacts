"""Tests for the ceiling ladder.

Run from the ``experimental-artifacts`` repository root with::

    python -m unittest discover -s analysis/ceiling -v

They read the tracked intensity and facility extracts. No network access is
needed.
"""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np
import pandas as pd

import ladder_results as ladder
from ladder_results import scr


class FacilityScenarioTest(unittest.TestCase):
    """Step 0: the intensities per kWh of IT energy, as in Batsim."""

    def test_scenarios_follow_the_footprint_model(self) -> None:
        facilities = pd.read_csv(scr.FACILITY_ROOT / "facility.csv").set_index("zone")
        for row in scr.load_windows().itertuples(index=False):
            with self.subTest(trace=row.file):
                site = facilities.loc[row.zone]
                grid = scr.load_trace(row)
                dry = scr.load_trace(row, facility="dry")
                continuous = scr.load_trace(row, facility="continuous")
                economizer = scr.load_trace(row, facility="economizer")

                np.testing.assert_allclose(dry.carbon, site.pue * grid.carbon)
                np.testing.assert_allclose(dry.water, site.pue * grid.water)
                np.testing.assert_array_equal(continuous.carbon, dry.carbon)
                np.testing.assert_array_equal(economizer.carbon, dry.carbon)
                np.testing.assert_allclose(continuous.water - dry.water, site.wue)
                # The economizer either evaporates at full rate or not at all.
                wue = economizer.water - dry.water
                self.assertTrue(np.all(np.isclose(wue, 0.0) | np.isclose(wue, site.wue_evap)))

    def test_unknown_scenario_is_rejected(self) -> None:
        row = next(scr.load_windows().itertuples(index=False))
        with self.assertRaises(ValueError):
            scr.load_trace(row, facility="wet")


class InherentFootprintTest(unittest.TestCase):
    """T0: the grid means are the window descriptors of windows.csv."""

    def test_grid_means_match_the_window_descriptors(self) -> None:
        workload = scr.load_workload("mustang_slack")
        for row in scr.load_windows().itertuples(index=False):
            with self.subTest(trace=row.file):
                inherent = ladder.compute_inherent_footprint(scr.load_trace(row), [workload])
                by_signal = {inherent_row["signal"]: inherent_row for inherent_row in inherent}
                self.assertAlmostEqual(by_signal["carbon"]["mean_intensity"], row.mean_carbon, delta=1e-4)
                self.assertAlmostEqual(by_signal["water"]["mean_intensity"], row.mean_water, delta=1e-4)


class SignalCeilingTest(unittest.TestCase):
    """T1: a toy signal and the invariants of a real window."""

    def test_toy_signal(self) -> None:
        intensity_by_instant = np.array([10, 10, 2, 10, 10, 2, 10, 10], dtype=float)
        first_arrival = np.array([0])
        # A 2-instant job may wait 3 instants: starting at instant 1 or 2 averages 10 and 2.
        mean_at_arrival, best_mean = ladder.find_best_deferrable_placement(
            intensity_by_instant, first_arrival, job_length=2, slack=3
        )
        self.assertEqual((mean_at_arrival[0], best_mean[0]), (10.0, 6.0))
        # Interrupted, the same job runs in instants 2 and 5 and skips the 10s between them.
        _, best_mean = ladder.find_best_interruptible_placement(
            intensity_by_instant, first_arrival, job_length=2, slack=4
        )
        self.assertEqual(best_mean[0], 2.0)

    def test_real_window_invariants(self) -> None:
        row = next(scr.load_windows().itertuples(index=False))
        dry = pd.DataFrame(ladder.compute_signal_ceiling(scr.load_trace(row, facility="dry"), "water"))
        continuous = pd.DataFrame(ladder.compute_signal_ceiling(scr.load_trace(row, facility="continuous"), "water"))
        keys = ["length_h", "slack_h"]
        modes = dry.pivot_table(index=keys, columns="mode", values="reduction")

        self.assertTrue((modes["deferrable"] >= 0).all())
        # Interruption can always pick the best contiguous block, so it never does worse.
        self.assertTrue((modes["interruptible"] >= modes["deferrable"] - 1e-12).all())
        # More slack only adds options.
        for _, by_slack in dry.groupby(["length_h", "mode"]):
            self.assertTrue(np.all(np.diff(by_slack.sort_values("slack_h").reduction) >= -1e-12))
        # A constant WUE shifts every placement equally: same absolute ceiling, lower percentage.
        np.testing.assert_allclose(
            continuous.at_arrival - continuous.best, dry.at_arrival - dry.best, atol=1e-9
        )
        self.assertTrue((continuous.reduction <= dry.reduction + 1e-12).all())


class WorkloadCeilingTest(unittest.TestCase):
    """T2: consistent with T1, close to the exact integrals, and its invariants on a real window."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.trace = scr.load_trace(next(scr.load_windows().itertuples(index=False)))
        cls.workload = scr.load_workload("mustang_stress")

    def test_uniform_arrivals_of_one_length_are_the_signal_ceiling(self) -> None:
        six_hour_jobs = dataclasses.replace(
            self.workload, execution_time_s=np.full(len(self.workload.job_ids), 6 * 3600.0)
        )
        workload_rows = pd.DataFrame(ladder.compute_workload_ceiling(self.trace, "carbon", [six_hour_jobs]))
        signal_rows = pd.DataFrame(ladder.compute_signal_ceiling(self.trace, "carbon"))
        uniform = workload_rows[workload_rows.arrivals == "uniform"].set_index(["slack_h", "mode"])
        six_hours = signal_rows[signal_rows.length_h == 6].set_index(["slack_h", "mode"]).loc[uniform.index]
        np.testing.assert_allclose(uniform[["at_arrival", "best"]], six_hours[["at_arrival", "best"]])

    def test_real_deferrable_is_close_to_the_exact_integrals(self) -> None:
        # Without idle power, cost_matrices is the exact footprint of each job at each start.
        no_idle = dataclasses.replace(
            self.workload, platform=dataclasses.replace(self.workload.platform, idle_power_w=0.0)
        )
        _, carbon_by_start, _ = scr.cost_matrices(no_idle, self.trace)
        rows = pd.DataFrame(ladder.compute_workload_ceiling(self.trace, "carbon", [self.workload]))
        for slack_h in ladder.SLACK_HOURS:
            with self.subTest(slack_h=slack_h):
                candidates = ladder.convert_hours_to_instants(slack_h) + 1
                exact = 1 - carbon_by_start[:, :candidates].min(axis=1).sum() / carbon_by_start[:, 0].sum()
                rounded = rows.query("slack_h == @slack_h and mode == 'deferrable' and arrivals == 'real'")
                self.assertAlmostEqual(rounded.reduction.item(), exact, delta=0.002)

    def test_real_window_invariants(self) -> None:
        rows = pd.DataFrame(ladder.compute_workload_ceiling(self.trace, "water", [self.workload]))
        modes = rows.pivot_table(index=["slack_h", "arrivals"], columns="mode", values="reduction")

        self.assertTrue((modes["deferrable"] >= 0).all())
        self.assertTrue((modes["interruptible"] >= modes["deferrable"] - 1e-12).all())
        for _, by_slack in rows.groupby(["mode", "arrivals"]):
            self.assertTrue(np.all(np.diff(by_slack.sort_values("slack_h").reduction) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
