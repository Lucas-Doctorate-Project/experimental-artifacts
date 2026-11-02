# Experimental artifacts

Batsim experiments comparing Greenfilling with EASY backfilling.

```text
analysis/
  outputs/
    original/                         # 432 simulation runs
    resampled/                        # 216 simulation runs
    alpha-sensitivity/                # 168 simulation runs
    green-window-scheduling/          # 608 of the 864 oracle runs
    green-window-scheduling-resume/   # the remaining 256 oracle runs
  figures/
    pdf/                  # PDF figures
    png/                  # PNG figures
  fonts/                  # existing results-figure font
  scripts/
    results.py            # footprint, service metrics and alpha sensitivity
    results_metrics.py    # paired comparisons
    results_plots.py      # results figures
    opportunity.py        # available saving versus Greenfilling
    backfill_allowed.py   # when Greenfilling allows backfilling
    oracle.py             # Greenfilling compared with the green-window oracle
    helpers.py            # shared numerical calculations
experiments/
  main.go                 # simulation runner
  original.toml
  resampled.toml
  alpha-sensitivity.toml
  green-window-scheduling.toml
  green-window-scheduling-resume.toml
  options/
workloads/                # workload inputs, platforms and preparation
intensities/              # intensity inputs and preparation
nix/                      # simulation and notebook environment
```

## Analyze results

See the [analysis README](analysis/README.md) for results and figures.

## Run simulations

From the repository root:

```sh
nix develop ./nix
cd experiments
go run . --campaign original.toml --out-dir ../analysis/outputs/original-rerun
```

Choose `resampled.toml` or `alpha-sensitivity.toml` for the other inputs. Paths inside TOML and scheduler option files are relative to `experiments/`. Use `--concurrency 4` to control simultaneous runs and `--timeout 30s` for the grace period after one simulator process exits. The runner requires a new output directory, writes one directory per run, and exits unsuccessfully if a simulation fails.

On a host with Podman, run from the repository root:

```sh
./run.sh experiments/original.toml --out-dir ../analysis/outputs/original-rerun
```

The container runs detached; the script prints commands to follow logs or stop it. To copy results from another host, use ordinary `rsync` or `scp`; analysis only needs each run's `out_jobs.csv` and the input traces and platform files.

Workload preparation is documented in [workloads](workloads/README.md); intensity preparation is documented in [intensities](intensities/README.md).

## Citation and license

Use [CITATION.cff](CITATION.cff) to cite this repository. Code uses the [Apache License 2.0](LICENSE); third-party data retain their source terms.
