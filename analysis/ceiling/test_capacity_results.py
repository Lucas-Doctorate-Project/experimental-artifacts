"""Tests for T3, the capacity step of the ceiling ladder.

Run from the ``experimental-artifacts`` repository root with::

    python -m unittest discover -s analysis/ceiling -v

Besides the tracked extracts, the real-window tests read the EASY and pricing
runs of the campaign in experiments/out.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import capacity_results as capacity
from capacity_results import scr


def make_toy_schedule() -> pd.DataFrame:
    """EASY on 2 nodes: A and B (2 h, 1 node) arrive at 0 and start at once, C (1 h) arrives at 1 and waits to 2."""
    return pd.DataFrame(dict(
        job_id=["A", "B", "C"],
        submission_time=[0.0, 0.0, 3600.0],
        starting_time=[0.0, 0.0, 7200.0],
        execution_time=[7200.0, 7200.0, 3600.0],
        finish_time=[7200.0, 7200.0, 10800.0],
        requested_number_of_resources=[1, 1, 1],
    ))


def compute_weighted_response_time_h(schedule: pd.DataFrame) -> float:
    """Area-weighted response time, as in greenfilling_weighted_response_time_analysis.py."""
    response = schedule.finish_time - schedule.submission_time
    return float(np.average(response, weights=schedule.requested_number_of_resources * schedule.execution_time)) / 3600


class ToyFrontierTest(unittest.TestCase):
    """The worked example: hours 3 and 4 are clean, every other hour costs 10."""

    INTENSITY = np.array([10, 10, 10, 4, 4, 10, 10, 10, 10], dtype=float)

    def setUp(self) -> None:
        self.lp = capacity.build_capacity_lp(make_toy_schedule(), machine_nodes=2, max_lateness_h=6)

    def test_easy_is_the_reference(self) -> None:
        self.assertEqual(self.lp.slots, len(self.INTENSITY))
        np.testing.assert_allclose(self.lp.easy_busy[:3], [7200, 7200, 3600])
        self.assertAlmostEqual(capacity.compute_delay_h(self.lp, self.lp.easy_busy), 0.0)

    def test_each_price_gives_its_frontier_point(self) -> None:
        # Filling the k-th clean node-hour shifts the busy profile by k node-hours of delay and saves 6,
        # so a price p per hour fills it when 6 > p x k. The delay is the total shift over the 5 node-hours.
        for price, delay_h, reduction in ((100.0, 0.0, 0.0), (4.0, 0.2, 0.12), (2.5, 0.6, 0.24),
                                          (1.6, 1.2, 0.36), (1e-3, 2.0, 0.48)):
            with self.subTest(price=price):
                busy = capacity.solve_busy_profile(self.lp, self.INTENSITY, price)
                self.assertAlmostEqual(capacity.compute_delay_h(self.lp, busy), delay_h)
                self.assertAlmostEqual(1 - self.INTENSITY @ busy / (self.INTENSITY @ self.lp.easy_busy), reduction)

    def test_delay_of_a_rigid_schedule_is_its_response_time_change(self) -> None:
        easy = make_toy_schedule()
        shifted = easy.assign(starting_time=[10800.0, 10800.0, 3600.0])
        shifted = shifted.assign(finish_time=shifted.starting_time + shifted.execution_time)
        busy = capacity.compute_busy_by_slot(shifted, self.lp.slots, capacity.SLOT_SECONDS)

        self.assertAlmostEqual(capacity.compute_delay_h(self.lp, busy), 2.2)
        self.assertAlmostEqual(
            capacity.compute_delay_h(self.lp, busy),
            compute_weighted_response_time_h(shifted) - compute_weighted_response_time_h(easy),
        )


class RealWindowTest(unittest.TestCase):
    """One Mustang window, on the jobs submitted in its first two days to keep the LPs small."""

    TRACE_NAME = "PL_2020-07-20"
    WORKLOAD_NAME = "mustang_stress"

    @classmethod
    def setUpClass(cls) -> None:
        window = next(row for _, row in scr.load_windows().iterrows() if row.file.endswith(f"{cls.TRACE_NAME}.csv"))
        cls.carbon_by_instant = scr.load_trace(window).carbon
        cls.machine_nodes = capacity.load_machine_nodes(cls.WORKLOAD_NAME)
        cls.easy = capacity.load_easy_schedule(cls.WORKLOAD_NAME, cls.TRACE_NAME)
        cls.early_easy = cls.easy[cls.easy.submission_time < 2 * 86400]

    def test_delay_matches_the_response_time_change_of_a_campaign_run(self) -> None:
        pricing = pd.read_csv(
            capacity.EASY_RUN_ROOT / f"green_window_scheduling_carbon_86400_{self.WORKLOAD_NAME}_{self.TRACE_NAME}" / "out_jobs.csv"
        )
        lp = capacity.build_capacity_lp(self.easy, self.machine_nodes)
        replay = ~pricing.job_id.str.startswith("ctx_")
        busy = capacity.compute_busy_by_slot(pricing[replay], lp.slots, capacity.SLOT_SECONDS)

        is_easy_replay = ~self.easy.job_id.str.startswith("ctx_")
        expected = (compute_weighted_response_time_h(pricing[replay])
                    - compute_weighted_response_time_h(self.easy[is_easy_replay]))
        np.testing.assert_allclose(busy.sum(), lp.work.sum())
        # The slot centers stand for the exact start and finish times inside each hour; the gap is about 0.001 h.
        self.assertAlmostEqual(capacity.compute_delay_h(lp, busy), expected, delta=0.01)

    def test_curve_falls_with_the_price(self) -> None:
        lp = capacity.build_capacity_lp(self.early_easy, self.machine_nodes)
        intensity = capacity.average_by_slot(self.carbon_by_instant)
        standardized = capacity.standardize(intensity)

        points = []
        for price in (1e-4, 1e-2, 1.0):
            busy = capacity.solve_busy_profile(lp, standardized, price)
            np.testing.assert_allclose(busy.sum(), lp.work.sum())
            self.assertTrue(np.all(busy <= lp.free_capacity + 1e-6))
            points.append((capacity.compute_delay_h(lp, busy), 1 - intensity[: lp.slots] @ busy / (intensity[: lp.slots] @ lp.easy_busy)))

        delays, reductions = np.array(points).T
        self.assertGreater(reductions[0], 0.0)
        self.assertTrue(np.all(np.diff(delays) <= 1e-6), delays)
        self.assertTrue(np.all(np.diff(reductions) <= 1e-9), reductions)

    def test_hourly_slots_are_close_to_the_trace_instants(self) -> None:
        reductions = {}
        for slot_seconds, intensity in ((900, self.carbon_by_instant),
                                        (3600, capacity.average_by_slot(self.carbon_by_instant))):
            lp = capacity.build_capacity_lp(self.early_easy, self.machine_nodes, max_lateness_h=48, slot_seconds=slot_seconds)
            busy = capacity.solve_busy_profile(lp, intensity, price_per_hour=1e-6)
            reductions[slot_seconds] = 1 - intensity[: lp.slots] @ busy / (intensity[: lp.slots] @ lp.easy_busy)

        self.assertAlmostEqual(reductions[3600], reductions[900], delta=0.005)


if __name__ == "__main__":
    unittest.main()
