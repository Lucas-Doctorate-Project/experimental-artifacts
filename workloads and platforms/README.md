# Workloads and platforms

Batsim workloads and SimGrid platforms built from production cluster traces. Each trace gets its own notebook implementing the same pipeline. The Mustang pipeline lives in [mustang.ipynb](mustang.ipynb) and produces `mustang_slack.json`, `mustang_stress.json`, and `mustang.xml`.

## Datasets

| Dataset | Settings | Main Batsim-conversion caveats |
|---|---|---|
| Mustang | LANL capacity cluster, 1600 homogeneous nodes (24 cores each). 61 months (2011-2016), 2.1M jobs, 565 users. | Done in [mustang.ipynb](mustang.ipynb). Invalid node counts, TIMEOUT jobs overran their walltime, 365-day default walltimes. All handled by hygiene and clipping. |
| Trinity | LANL capability machine (Cray XC40), 9408 homogeneous nodes (32 cores each). About 3 months (Feb-Apr 2016), 25k jobs. | Same schema as Mustang. Short trace leaves few 4-week candidate windows. Covers the pre-production open-science period, so the job mix is not steady production. Failed jobs have empty `start_time`. |
| MetaCentrum | Czech national grid, 47 heterogeneous clusters, 34,400 cores, 42 queues (GPU, interactive, backfill). Year 2023, 10.1M jobs. | SWF format, core-level (not node-level) allocations on a heterogeneous grid, which breaks the single homogeneous platform assumption. GPU and memory dimensions have no place in our schema. Focus on cluster 17 (see to-do). |
| CC-IN2P3 | French HEP computing centre (HTC), up to 312 machines, 46k concurrent threads, 105 TB RAM. Year 2024, 44M jobs, about 1000 users. | Slurm accounting TSVs in `datasets/CC-IN2P3/`, one file per month (5.6 GB total). Thread-level HTC jobs (`alloccpus`), not node-level. Memory is the binding resource (95% allocated vs 69% CPU), which our schema does not model. A 4-week extract still holds millions of jobs, which stresses Batsim and the schedulers. Focus on the `htc` partition (see to-do). |

## Getting the datasets

The `datasets/` directory is gitignored (several GB) and the notebooks expect it next to them, at `workloads and platforms/datasets/`. To recreate it:

- **Mustang** and **Trinity**: in the ATLAS repository, <https://ftp.pdl.cmu.edu/pub/datasets/ATLAS/>. For Trinity take the *formatted* release. Decompress and save as `datasets/mustang.csv` and `datasets/trinity.csv`.
- **MetaCentrum**: in the JSSPP Workloads Archive (see references). Save the 2023 SWF file as `datasets/metacentrum.swf`.
- **CC-IN2P3**: on Zenodo, linked from the dataset paper (see references). Place the monthly TSVs as `datasets/CC-IN2P3/01.tsv` through `12.tsv`.

## To do

- [ ] **Trinity**: `trinity.ipynb` reusing the Mustang pipeline. Adjust the constants (9408 nodes, 32 cores, Haswell power draw). Expect the window search to need relaxed constraints, since the trace barely fits ten 4-week windows.
- [ ] **MetaCentrum**: `metacentrum.ipynb` restricted to **cluster 17** (SWF `partition` field). It is the only cluster where backfilling has material work: 0.17% of jobs are wider than 10% of capacity but they hold 4.7% of core-seconds (jobs up to 1,024 cores on a ~3,000-core pool). It is also large and active all year (410k jobs in 2023, 61% mean utilization vs peak). Runner-up cluster 2 has more load (85% mean utilization) but zero wide jobs, so EASY would collapse to FCFS. Model resources as cores (`nb_res` = peak concurrent cores), drop the 1% GPU jobs, and use the trace's soft walltimes as the runtime estimates, which are more refined than the usual user estimates.
- [ ] **CC-IN2P3**: `ccin2p3.ipynb` restricted to the **`htc` partition** (99.5% of jobs, 96.6% of CPU time, 273 of the ~310 machines, 21,952 cores). The `hpc` partition is only 16 nodes and `gpu` would need GPU modelling. Caveats: 89.9% of jobs are single-core, so the backfill toggle will likely be silent and this trace mainly tests time-shifting at HTC scale. Memory, not CPU, is the binding resource (95% allocated vs 69% CPU). One account holds 35.7% of jobs (user0137 alone 27.9%), near our `top_user_share` cap. Four maintenance days (03-12, 06-25, 09-17, 12-03) killed and requeued running jobs, the drain-streak constraint should exclude them. Volume remains the open question: a 4-week extract holds about 3.4M jobs, so job aggregation may be needed before Batsim replay is practical.

## Extract selection

The goal is to replay 4-week windows in Batsim to evaluate environmental-aware scheduling (carbon and water intensity signals) against FCFS and EASY backfilling baselines. The windows must span multiple weeks because intensity signals vary on diurnal and weather timescales, and they must exercise both a relaxed and a saturated regime.

Candidate windows cover the whole trace, anchored at Monday 00:00 UTC and slid by 1 week, so the weekly phase is identical across extracts. Each window is described by metrics normalized by machine size: mean and standard deviation of node utilization, fraction of saturated hours (>= 95% capacity), fraction and longest streak of low-utilization hours (< 20%), normalized queue depth, top-user share, node-seconds share of wide jobs (>= 10% of the machine), and the share of narrow short jobs that EASY can backfill.

Feasibility constraints shared by both regimes exclude pathological periods (ramp-up, drains, single-user bursts):

- `frac_low <= 0.05` and `max_low_streak_h <= 6`
- `top_user_share <= 0.5`
- `wide_ns_share >= 0.2` and `frac_bf_candidates >= 0.3`, so EASY actually differs from FCFS

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
- MetaCentrum: Klusáček and Chlumský, MetaCentrum 2023 workload trace, JSSPP Workloads Archive. <https://jsspp.org/workload/index.php?page=meta23>
- CC-IN2P3: CC-IN2P3 2024 workload dataset, arXiv:2606.05914. <https://arxiv.org/abs/2606.05914>
