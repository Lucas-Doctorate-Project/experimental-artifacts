# Experimental artifacts

Batsim experiments comparing Greenfilling with EASY backfilling.

```text
analysis/
  outputs/
    original/             # 432 simulation runs
    resampled/            # 216 simulation runs
    alpha-sensitivity/    # 168 simulation runs
  figures/                # all analysis figures
  fonts/                  # existing results-figure font
  01_results.ipynb
  02_mechanism.ipynb
  03_opportunity.ipynb
  helpers.py              # shared numerical calculations
experiments/
  main.go                 # simulation runner
  original.toml
  resampled.toml
  alpha-sensitivity.toml
  options/
workloads/                # workload inputs, platforms and preparation
intensities/              # intensity inputs and preparation
nix/                      # simulation and notebook environment
```

## Analyze results

Enter `nix develop ./nix` and open the notebooks in `analysis/`. Each notebook runs independently, reads the simulation CSV files directly, and saves PNG/PDF figures into `analysis/figures/`. Original and resampled figures use filename prefixes, keeping their existing layouts. Saved notebook outputs can be read without executing them. Simulation outputs are local and excluded from Git.

The results notebook compares footprint and service metrics, including alpha sensitivity. The mechanism notebook examines job classes, regional throughput and an estimated backfilling gate. The opportunity notebook computes a savings bound that ignores capacity conflicts.

Corrected energy and footprint include idle nodes and context jobs over a common interval: 35 days for original and alpha results, 70 days for resampled results. Service metrics exclude context jobs. The resamples change composition and offered load together, with one selected seed per source window. Notebook text explains these calculations and their limits.

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
