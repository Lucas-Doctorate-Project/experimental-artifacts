# Workloads and platforms

Batsim workloads and SimGrid platforms built from production cluster traces. Each trace gets its own notebook implementing the same pipeline. The Mustang pipeline lives in [mustang.ipynb](mustang.ipynb) and produces `mustang_slack.json`, `mustang_stress.json`, and `mustang.xml`.

## Datasets

| Dataset | Settings | Main Batsim-conversion caveats |
|---|---|---|
| Mustang | LANL capacity cluster, 1600 homogeneous nodes (24 cores each). 61 months (2011-2016), 2.1M jobs, 565 users. | Done in [mustang.ipynb](mustang.ipynb). Invalid node counts, TIMEOUT jobs overran their walltime, 365-day default walltimes. All handled by hygiene and clipping. |
| Trinity | LANL capability machine (Cray XC40), 9408 homogeneous nodes (32 cores each). About 3 months (Feb-Apr 2016), 25k jobs. | First implementation in [trinity.ipynb](trinity.ipynb). Same formatted schema as Mustang, but the trace is short and leaves few 4-week candidate windows. Covers the pre-production open-science period, so the job mix is not steady production. Failed jobs have empty `start_time`. |

## Extract selection

The goal is to replay 4-week windows in Batsim to evaluate environmental-aware scheduling (carbon and water intensity signals) against an EASY backfilling baseline. The windows must span multiple weeks because intensity signals vary on diurnal and weather timescales, and they must exercise both a relaxed and a saturated regime.

Candidate windows cover the whole trace, anchored at Monday 00:00 UTC and slid by 1 week, so the weekly phase is identical across extracts. Each window is described by metrics normalized by machine size: mean and standard deviation of node utilization, fraction of saturated hours (>= 95% capacity), fraction and longest streak of low-utilization hours (< 20%), normalized queue depth, top-user share, node-seconds share of wide jobs (>= 10% of the machine), and the share of narrow short jobs that EASY can backfill.

Feasibility constraints shared by both regimes exclude pathological periods (ramp-up, drains, single-user bursts):

- `frac_low <= 0.05` and `max_low_streak_h <= 6`
- `top_user_share <= 0.5`
- `wide_ns_share >= 0.2` and `frac_bf_candidates >= 0.3`, so backfilling has a real effect

Two extracts are then selected, one per regime:

- **slack**: busy but not saturated (`0.5 <= mean_util <= 0.9`, `frac_saturated <= 0.3`). Ranked by `z(std_util) + z(mean_queue_norm)`. The friendly regime for time-shifting jobs.
- **stress**: heavily saturated (`mean_util <= 0.95`, `0.3 < frac_saturated <= 0.5`). Ranked by `z(frac_saturated) + z(mean_queue_norm)`. The regime where disabling backfilling has real consequences.

## Workload generation

Each job gets its own `parallel_homogeneous` profile with `cpu = runtime * node_speed` and `com = 0` (network not modelled). Walltimes come from the trace's `wallclock_limit`, clipped so that `runtime <= walltime` always holds, since walltime violations are not modelled. EASY needs these estimates to build its reservations.

The simulation starts with an empty machine, so warm-up jobs reconstruct the system state at the extract start `T0`. Jobs running at `T0` are submitted at `t = 0` with their remaining runtime and walltime, ordered by original start time. Jobs queued at `T0` are submitted at `t = 0` with their full runtime, ordered by original submit time. The steady state begins when the first replay job (submitted inside the 4-week window) arrives. Evaluation metrics should be computed on the steady state only.

## Platform generation

`mustang.xml` is a homogeneous SimGrid platform modelled after Mustang. Hosts are full nodes (no `core` attribute) computing at full-node speed. Power states use the SimGrid `wattage_per_state` format `idle:epsilon:all_cores`.

## References

- Mustang and Trinity: Amvrosiadis et al., *On the Diversity of Cluster Workloads and its Impact on Research Results*, USENIX ATC 2018. <https://www.usenix.org/conference/atc18/presentation/amvrosiadis>
