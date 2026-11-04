# Scheduling policies for a second comparison

Research date: 15 October 2026. This note identifies a comparator; it does not report a new experiment or change the paper.

The most practical candidate is **CATS**, a published carbon-aware submission tool with a small, inspectable implementation. Its temporal selection rule can be combined with EASY to produce capacity-feasible schedules. This would be a **CATS-based admission policy evaluated with EASY**, rather than a reproduction of its Slurm deployment. **Interval Switch (IS)** is closer to a published HPC scheduling policy, but reproducing its energy prediction and priority rules requires more decisions. Neither candidate justifies calling a comparison a benchmark against the best current scheduler.

## Scope and ranking

The current simulator has homogeneous nodes, constant idle and busy power, rigid nonpreemptive jobs, and time-varying carbon and water intensities. Greenfilling changes whether an EASY backfill job may start; it still admits the priority job. The relevant external scheduler is pinned in `nix/flake.lock` at `ffe1b9f634ee3b1bfa70bcc7917d0b3d9465b4af`. See the pinned [Greenfilling implementation](https://github.com/Lucas-Doctorate-Project/batsched/blob/ffe1b9f634ee3b1bfa70bcc7917d0b3d9465b4af/src/algo/greenfilling.cpp) and [EASY implementation](https://github.com/Lucas-Doctorate-Project/batsched/blob/ffe1b9f634ee3b1bfa70bcc7917d0b3d9465b4af/src/algo/easy_bf.cpp).

| Rank | Candidate | Publication and source status | Fit and implementation effort |
|---|---|---|---|
| 1 | CATS | JOSS, July 2025; MIT implementation available | Small temporal-selection rule. Requires forecast input, a duration estimate, and an admission adapter before EASY. |
| 2 | Interval Switch / Ecological Weight | JSSPP 2025 proceedings, published January 2026; scheduler source not located | Actual HPC queue policies with EASY backfilling. Priority port is modest; prediction and several policy details need clarification. Prefer IS over EW. |
| 3 | COAST | 2026 preprint; implementation not located | Explicit delay and congestion costs, but recommendations do not enforce capacity or reservations. Adding those changes the evaluated policy. |
| 4 | Flex, previously CarbonFlex | SoCC 2026 technical report; MIT repository currently has no implementation | Changes cluster capacity and elastic job allocations. These mechanisms are outside the present simulator. |

The ranking is an assessment of suitability for this study, not a ranking of published performance. “Source not located” means that the inspected paper and author repositories did not provide an implementation; it does not establish that none exists.

## 1. CATS: recommended practical comparator

Bartholomew et al., *CATS: The Climate Aware Task Scheduler*, was published in JOSS 10(111), 8251 on 8 July 2025. Given a duration and a carbon forecast, it identifies a lower-carbon execution window and can submit a job for later execution. It is particularly relevant to flexible work on infrastructure with spare capacity. It is a submission tool, rather than a replacement for the cluster resource scheduler. [Paper](https://joss.theoj.org/papers/10.21105/joss.08251), [full text](https://joss.theoj.org/papers/10.21105/joss.08251.pdf).

The author repository has an MIT license. Version `v1.2.1`, commit `67461c6dc3bc8ec923cea8ba4001a2e4109311fb`, is a suitable fixed reference. `WindowedForecast` computes the mean forecast intensity over each candidate execution window using linear interpolation and trapezoidal integration. The CLI chooses the minimum window. The Slurm adapter uses `sbatch --begin`, leaving resource allocation to Slurm. [Forecast implementation](https://github.com/GreenScheduler/cats/blob/67461c6dc3bc8ec923cea8ba4001a2e4109311fb/cats/forecast.py), [selection](https://github.com/GreenScheduler/cats/blob/67461c6dc3bc8ec923cea8ba4001a2e4109311fb/cats/cli.py), [scheduler adapters](https://github.com/GreenScheduler/cats/blob/67461c6dc3bc8ec923cea8ba4001a2e4109311fb/cats/schedulers.py), [license](https://github.com/GreenScheduler/cats/blob/67461c6dc3bc8ec923cea8ba4001a2e4109311fb/LICENSE).

### Proposed implementation, not an existing result

For job `j`, let `r_j` be its original submission, `w_j` its requested walltime, and `D` an allowed admission delay. At submission, select a not-before time:

`a_j = earliest argmin_t integral(t, t + w_j, forecast_carbon), for r_j <= t <= r_j + D`.

Use candidates spaced by the trace cadence, including `r_j`. Admit the job to EASY at `a_j`. EASY then enforces capacity and its reservation among eligible jobs. Preserve `r_j` in the analysis: intentional admission delay is `a_j - r_j`, and subsequent queue delay is `actual_start_j - a_j`. Scheduling only once at submission avoids repeated postponement. Context jobs should retain their existing behavior.

This recipe preserves the minimum-integrated-intensity decision, but replaces Slurm with EASY. Its delay budget bounds admission deferral; contention can delay the actual start further. It therefore provides no completion-deadline guarantee. It can also defer a job that would otherwise become the queue head, whereas Greenfilling exempts its priority job. Restricting this rule to backfill jobs would be a further adaptation and should be named separately.

For a close implementation of the published software, retain the pinned CATS interpolation and window selection. Physical accounting should still use this repository's measured trace convention. If selection is changed to piecewise-constant integration to match that convention, document the change and call the method CATS-based.

Distinguish the 2025 publication from later software features. The current main-branch README at `666e73f4fdc7bb5ec7f08c3071e8be491b886cef` already documents water footprint, water stress footprint, and an environmental score through wattnet.eu, as well as European providers. A water objective is therefore not a novel extension to current CATS. Using this study's saved country-season water traces with EASY would still require a trace adapter and a declared experimental configuration; it would not reproduce the 2025 carbon evaluation. [Current README at a fixed commit](https://github.com/GreenScheduler/cats/blob/666e73f4fdc7bb5ec7f08c3071e8be491b886cef/README.md).

Use requested walltime in the main comparison because it is already available to EASY. Actual execution time would give the policy additional information and belongs in a separately labeled oracle-duration diagnostic. All four original workloads have maximum requested walltimes below or equal to 36 hours. Their latest submissions are before day 28. A 24-hour admission window therefore requires forecast coverage beyond day 30, which fits within the original 35-day trace coverage. This check concerns forecast coverage; it does not establish that the resulting queue drains before day 35.

## 2. Adaptive carbon-aware HPC policies: strongest domain match

Benhari and Trystram's *Adaptive Carbon-Aware Scheduling Policies for HPC Systems* appeared at JSSPP 2025; Springer published the revised chapter on 2 January 2026. The publisher full text is restricted. The author's open thesis identifies Chapter 3 as the JSSPP work and supplies its algorithms. The thesis should be cited for those details rather than treated as a verified copy of the final chapter. [Publisher record](https://link.springer.com/chapter/10.1007/978-3-032-10507-3_4), [workshop program](https://jsspp.org/index.php?page=program25), [author thesis](https://theses.hal.science/tel-05724808/file/BENHARI_2025_archivage.pdf).

In thesis Chapter 3, Interval Switch (IS) uses a default queue policy outside critical carbon intervals and an ecological priority inside them. Ecological Weight (EW) mixes default and ecological scores according to predicted carbon intensity. The ecological score is predicted job energy divided by requested resources and walltime. The study uses energy prediction, carbon prediction, and EASY backfilling. See printed pages 37–39, PDF pages 46–48. EW's displayed linear weight decreases with intensity although the surrounding explanation describes increasing ecological weight; this needs clarification before faithful reproduction. No scheduler implementation was located in the author's public repositories. [Thesis](https://theses.hal.science/tel-05724808/file/BENHARI_2025_archivage.pdf), [author GitHub account](https://github.com/aBenhari).

### The ecological score is not necessarily constant here

With fixed busy power per node, estimated energy is proportional to allocated nodes times estimated runtime. Dividing by requested nodes and walltime leaves a score proportional to `estimated_runtime / walltime`. The score is constant only if energy is estimated from walltime itself. A predictor of actual runtime can produce different priorities.

This was checked on the four original JSON inputs and their XML platform speeds. Replay jobs exclude IDs starting with `ctx_`. Runtime is `min(profile cpu / node speed, walltime)`.

| Original workload | Replay jobs | Median runtime / walltime | Interquartile range | Jobs reaching walltime |
|---|---:|---:|---:|---:|
| Mustang slack | 10,683 | 0.27754 | 0.00052–0.96900 | 24.15% |
| Mustang stress | 5,089 | 0.09333 | 0.01000–0.75035 | 18.73% |
| Trinity slack | 4,622 | 0.05111 | 0.00769–0.44362 | 10.19% |
| Trinity stress | 3,537 | 0.05000 | 0.00642–0.47396 | 13.37% |

The inputs are `workloads/original/{mustang,trinity}_{slack,stress}.json`, with node speeds of 110.4 Gflop/s and 1177.6 Gflop/s respectively. Thus, dismissing IS or EW as inherently degenerate would be wrong. Under this power model, however, their ecological ordering reflects predicted runtime relative to requested walltime, rather than differences in power per node.

A useful IS implementation would need a causal runtime predictor or an explicit oracle-energy assumption, a fixed critical-interval rule, and confirmed tie and backfill ordering. An always-ecological-priority control would distinguish the benefit of reordering from the benefit of switching with carbon intensity. This is feasible research work, but less direct than the CATS adapter. It should not be described as a faithful reproduction until those decisions are resolved.

## 3. COAST: relevant recent idea, insufficient feasibility guarantees

*COAST: Congestion-Aware Start-Time Recommendations for Carbon-Aware HPC Jobs* is a 2026 preprint. It selects recommendations using carbon, delay, and a quadratic congestion cost. It operates in five-minute batches with 30-minute candidate slots over 48 hours, using runtime, resources, energy, and deadlines. Its evaluation uses actual carbon trajectories and runtimes. The delay cap is four times runtime, capped at 24 hours. The congestion term is soft: the model does not enforce cluster capacity, queue behavior, or reservations, and evaluates ideal realization of recommendations. [Preprint](https://arxiv.org/abs/2609.05443), [full text](https://arxiv.org/html/2609.05443v1).

My assessment is that adding EASY would create a new combined policy whose realized starts can differ from the recommended starts. This could be informative, but the preprint cannot supply the capacity-feasible comparator directly. No implementation was located. It is therefore a lower priority than CATS or IS.

## 4. Flex / CarbonFlex: current related work with different mechanisms

The 13 October 2026 revision of *CarbonFlex* is titled *Flex: Grid-Responsive Provisioning and Scheduling for Elastic Cloud Clusters*. It is the technical report for a SoCC 2026 paper. Flex jointly controls provisioned cluster capacity and scheduling of elastic batch jobs, learning decisions from recent history. It is implemented over AWS ParallelCluster. [Current paper record](https://arxiv.org/abs/2505.18357), [publication DOI](https://doi.org/10.1145/3828161.3857880).

The author's `CarbonFlex` repository redirects to `Flex`. On the research date it contained a README, license, and ignore file, with implementation availability still promised. The MIT license does not mean the scheduler implementation is already available. [Author repository](https://github.com/umassos/Flex).

My assessment is that the elasticity and provisioning controls make this a poor first comparator for rigid jobs on a fixed platform. Removing those controls would remove central parts of the method. It is useful related work when explaining why conclusions about simple nonpreemptive policies do not extend to systems with more scheduling freedom.

## A bounded comparison protocol

These are proposed decisions for a separate campaign, not changes to the running α campaign.

1. **Preserve pairing.** Use all four original workloads and the same country-season windows for EASY, Greenfilling, and the new policy. Keep context, arrivals, resource counts, execution profiles, and accounting identical. A six-window pilot can check implementation; use all 36 windows for the eventual main comparison if that comparison is pursued.
2. **Fix parameters before inspecting savings.** For CATS-based admission, use `D = 0` as an EASY-equivalence check, `D = 6 hours` as a proposed main budget, and `D = 24 hours` as a sensitivity case. These are study choices, not CATS defaults or validated optima. Do not select them on the final evaluation outcomes.
3. **Separate information assumptions.** A future measured carbon trace is an oracle forecast. It is useful for testing a favorable setting, but results must be labeled accordingly. A deployment claim needs causal forecasts constructed from earlier data and evaluation of their error. A simple day-lag forecast could be a separate baseline, provided all required historical coverage exists. Do not describe synthetic perturbations as evidence about an operational forecast service.
4. **Measure actual schedules.** Validate resource capacity and reservation behavior; evaluate realized job starts and completions. Retain original submission times in waiting and slowdown metrics. Report admission delay, subsequent queue delay, mean and tail waiting, bounded slowdown, makespan, and completion count. Treat each paired workload/window as the comparison unit rather than treating jobs as independent experimental replications.
5. **Use the corrected footprint boundary.** Count the full platform over the same interval for every policy, including idle power once. Also report the contribution above idle power to explain small platform differences. If a policy has unfinished jobs or needs trace extrapolation at the boundary, do not report a completed-work saving from that interval. Extend coverage and recompute all comparators under a new, explicit common horizon.
6. **Keep configurations separate.** Carbon CATS is the first literature comparison. A water objective uses a feature documented in current software, but extends beyond the 2025 carbon evaluation. A backfill-only restriction, oracle runtime, or modified integration rule must be identified as an adaptation. Record the source commit and all departures in the campaign manifest. Preserve original results and add outputs under a new campaign name.

Two environmentally aware policies can support a claim about the observed tradeoffs of those policies under the tested workload, information, power, and trace assumptions. They cannot establish that environmentally aware scheduling generally fails, that forecast-based methods cannot work, or that Greenfilling is representative of all recent methods. A CATS comparison also changes both forecast access and the set of jobs that may be delayed. It can test a useful alternative, but it does not isolate the EMA threshold as the cause of a difference.

**Recommended next decision:** choose a small CATS-based admission comparator if the aim is a transparent, source-backed second policy. Prefer IS only if the aim specifically requires a published HPC queue policy and there is time to resolve its prediction and priority details. No comparator was implemented or launched for this note.
