# Reproducing the Greenfilling paper

This branch pins the state of the testbed used by the Greenfilling workshop paper. The `main` branch keeps evolving, so start here to rebuild the published results.

The repository holds more than the paper reports. This README separates the two. It lists every element of the paper, the command that rebuilds it, and the output file to compare against. It also lists what lives here that the paper does not use, so you can tell which files belong to the results without reading the code to find out.

## Repository layout

- [intensities/](intensities/): carbon and water intensity traces derived from ENTSO-E generation data (PL, FR, DE), the window sampling, and the offline consistency check against pinned Ember and EEA snapshots.
- [workloads and platforms/](workloads%20and%20platforms/): Batsim workloads and SimGrid platforms built from the Mustang and Trinity production traces.
- [experiments/](experiments/): Go campaign runner that combines the intensity traces with the workloads and platforms to run batches of Batsim and Batsched simulations.
- [analysis/](analysis/): `reproduce_paper.py`, which rebuilds and checks the published results, and `greenfilling_tradeoffs.ipynb`, the wider exploratory analysis.

Each directory has its own README with the details.

The short version, on a machine with the toolchain and about 60 GB of free disk:

```sh
nix develop ./nix
cd experiments && go run . --campaign experiments.toml --concurrency 16 && cd ..
python analysis/reproduce_paper.py
```

The first command builds the pinned simulator stack, the second runs the 432 simulations, and the third rebuilds the paper figure and checks every number the paper quotes. `reproduce_paper.py` exits `0` only when all of them match.

## What the paper uses

| Paper element | Stage | Command | Output to compare |
| --- | --- | --- | --- |
| Section II-B, country mean intensities (36.4 and 569.4 gCO2eq/kWh, 3.320 and 1.587 L/kWh) | 5 | `python analysis/reproduce_paper.py` | `analysis/paper_results/signal_means.csv` |
| Section II-D, Ember consistency check (Pearson 0.987, 0.994, 0.999 and the -16.9%, -34.0%, -19.7% offsets) | 6 | `python intensities/validate_carbon.py --output-dir intensities/validation/out` | `intensities/validation/out/paper_table.csv` |
| Table I, campaign design (432 runs, 288 pairs, 36 windows, alpha = 0.5, seed 20180101) | 2 and 4 | `python experiments/gen_campaign.py` then the campaign run | `experiments/experiments.toml`, `intensities/windows.csv` |
| Figure 1, footprint and slowdown deltas versus EASY | 5 | `python analysis/reproduce_paper.py` | `analysis/figures/greenfilling_deltas_vs_easy.pdf` |
| Section III-C, all reported medians, counts, and the +24.5% to +757.2% slowdown range | 5 | `python analysis/reproduce_paper.py` | `analysis/paper_results/scenario_summary.csv`, `analysis/paper_results/paper_claims.csv` |
| Section IV, the +323.6% median and the 30 of 36 windows for Mustang slack | 5 | `python analysis/reproduce_paper.py` | `analysis/paper_results/paper_claims.csv` |

`analysis/paper_results/paper_claims.csv` is the single place to look. It puts each value printed in the paper next to the value recomputed from the outputs, with the difference and a pass or fail status.

## What is in the repository but not in the paper

None of the following is needed to reproduce the paper. It is kept because the testbed is meant to be reused.

- `analysis/greenfilling_tradeoffs.ipynb` is the working notebook, and only its delta boxplot became Figure 1. The trade-off scatters, the energy and intensity exposure decomposition, and the swing correlations stayed out, along with the figures `carbon_saving_vs_slowdown`, `water_saving_vs_slowdown`, `energy_exposure_deltas`, and `swing_vs_footprint_saving`. `reproduce_paper.py --all-figures` regenerates them anyway if you want them.
- Section II-D reports only the Ember comparison. The other five validation checks, covering the EEA cross-check, the non-CCS factor sensitivity, the 2025 supplement, the fuel harmonization decomposition, and generation coverage, are described in `intensities/validation/README.md` and written to `intensities/validation/out/`, but the paper does not use them.
- `intensities/traces/{DE,FR,PL}.csv` span 2018 to 2025, while only the 36 sampled 4-week windows feed the campaign. The full traces are still read for the Section II-B means and by the validation.
- `swing_carbon` and `swing_water` in `intensities/windows.csv` are window descriptors used only by the exploratory analysis.
- Each run writes more than the analysis reads: `out_schedule.trace`, `out_machine_states.csv`, `out_consumed_energy.csv`, `out_environmental_footprint.csv`, and the two simulator logs. The paper's results come from `out_schedule.csv` and `out_jobs.csv` only.
- `experiments/out/` may contain stale directories from earlier campaigns, for example runs whose names lack the dataset field. The analysis enumerates runs from `experiments/experiments.toml`, so anything not named there is ignored.
- `run.sh` runs a campaign on a host that has podman but no Nix. It is a convenience, not a separate result.

## Stages

Stages 1 to 3 are optional. Their outputs are committed, so a reproduction can start at stage 4. They are documented because the paper describes how those inputs were built.

### Stage 0: environment

```sh
nix develop ./nix
```

The flake pins the whole toolchain, including the three simulator forks that carry the environmental footprint model. `nix/flake.lock` records the exact commits. The build needs network access on first use and takes a while, since SimGrid and Batsim are compiled from source.

### Stage 1 (optional): intensity traces

```sh
jupyter nbconvert --to notebook --execute --inplace intensities/generate_traces.ipynb
```

The notebook runs offline from the raw ENTSO-E caches in `intensities/raw/` and rewrites `intensities/traces/{DE,FR,PL}.csv`. It contacts the ENTSO-E API, and needs `ENTSOE_API_KEY`, only when a cache file is missing.

### Stage 2 (optional): window sampling

```sh
jupyter nbconvert --to notebook --execute --inplace intensities/window_selection.ipynb
```

This draws the 36 windows under the fixed seed 20180101 and rewrites `intensities/traces/<CC>_<date>.csv` and `intensities/windows.csv`. It is deterministic, so the same 36 windows come back.

### Stage 3 (optional): workloads and platforms

```sh
jupyter nbconvert --to notebook --execute --inplace "workloads and platforms/mustang.ipynb"
jupyter nbconvert --to notebook --execute --inplace "workloads and platforms/trinity.ipynb"
```

These need the LANL job traces in `workloads and platforms/datasets/`, which are not redistributed here. They come from the ATLAS repository at CMU PDL (<https://www.pdl.cmu.edu/ATLAS/>), released with Amvrosiadis et al., USENIX ATC 2018. The generated `mustang_*.json`, `trinity_*.json`, `mustang.xml`, and `trinity.xml` are committed, so this stage can be skipped.

### Stage 4: the campaign

```sh
cd experiments
python3 gen_campaign.py        # optional, experiments.toml is committed
go run . --campaign experiments.toml --concurrency 16
```

432 simulations: 3 schedulers, 2 machines, 2 regimes, 36 windows. The reference run took about one hour of wall-clock time at `--concurrency 16` on a dual AMD EPYC Bergamo 9734 node, 224 cores, 1.2 TB of DDR5. Lower the concurrency on smaller machines. Each experiment is a Batsim and Batsched pair in lock step, so oversubscribing them costs throughput rather than gaining it.

Budget about 50 GB of disk for `experiments/out/`. The runner skips any experiment that already has `out_schedule.csv`, so an interrupted campaign resumes, and deleting a run directory forces it to run again. The campaign exits `0` only when all 432 runs succeed.

### Stage 5: figure and reported numbers

```sh
python analysis/reproduce_paper.py
```

The script reads the 432 run directories, rebuilds Figure 1 into `analysis/figures/`, writes the tables into `analysis/paper_results/`, prints every checked claim next to its value in the paper, and exits non-zero if any of them disagree. It takes about a minute and needs no simulator, so it can run outside the Nix shell given Python with pandas, numpy, and matplotlib.

Useful flags: `--out-dir` to point at a campaign output tree elsewhere, `--all-figures` to also export the supplementary figures, `--skip-validation` to leave stage 6 out.

### Stage 6: carbon consistency check

```sh
python intensities/validate_carbon.py --output-dir intensities/validation/out
python -m unittest discover -s intensities/validation/tests -v
```

Both commands run offline, from the pinned Ember and EEA snapshots in `intensities/validation/reference/`. Stage 5 runs the first one itself when `paper_table.csv` is absent. `intensities/validation/README.md` explains all six checks and their limits.

## If a number does not match

- If runs are missing, `reproduce_paper.py` stops and names the incomplete ones. Rerun the campaign, which skips what already succeeded.
- Results depend on the pinned SimGrid, Batsim, and Batsched commits in `nix/flake.lock`. Building outside the flake, or updating the lock, can move the numbers.
- The window sample has to match, so `intensities/windows.csv` should list the same 36 windows. Rerunning stage 1 against refreshed ENTSO-E data can change the traces, and therefore the sample, because ENTSO-E revises published generation.
- Differences in the last printed digit are expected. The claim checks allow half of that digit, and anything larger is a real difference worth reporting.

## Citation

If you use this repository, cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

This repository is licensed under the [Apache License 2.0](LICENSE), except where otherwise noted. Third-party datasets and artifacts derived from them remain subject to their original source terms.
