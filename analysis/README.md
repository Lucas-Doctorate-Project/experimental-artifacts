# Analysis

From the repository root, enter `nix develop ./nix`. Run scripts with `python analysis/scripts/<script>.py`.

## Figures

Figures go to `analysis/figures/{png,pdf}/`, replacing previous versions.

| Script | Figure content |
| --- | --- |
| `results.py` | Footprint, service metrics, and alpha sensitivity |
| `opportunity.py` | Delay-only savings bound and Greenfilling |
| `backfill_allowed.py` | When Greenfilling allows backfilling |
| `policy_comparison.py` | Carbon saving, area-weighted response time, shifted-job intensity, and horizon sensitivity |
| `oracle.py` | Shared policy figures plus delay by job class |
| `queue_gain_loss.py` | Gain from jobs that ran cleaner and loss from jobs that ran dirtier, for job shifting and CATS, and by job size |
| `held_jobs.py` | Savings from jobs delayed on purpose and all other jobs, for job shifting and CATS |
| `held_walltime.py` | Expected, real, and best possible saving of delayed jobs by share of requested time used |
| `swing.py` | Savings versus daily intensity variation |

Use `--output-dir PATH` for another destination and `--data-dir PATH` to export per-case CSVs to a new directory.

To compute offline shifting, use a new output directory, then render the shared comparison from its CSV.

```sh
python analysis/scripts/offline_shifting.py --output-dir /tmp/offline-shifting
python analysis/scripts/offline_shifting.py --output-dir /tmp/offline-shifting --figure-dir analysis/figures
```

## Inputs and accounting

Simulation archives live in `analysis/outputs/` and are excluded from Git. Oracle analysis reads both `green-window-scheduling/` and `green-window-scheduling-resume/`. `held_jobs.py`, `held_walltime.py`, and `queue_gain_loss.py` compare job shifting and CATS at the 6 h limit with known future intensity, through `delayed_jobs.py`. They read the job-shifting decision logs in `green-window-scheduling-debug-6h/` and the CATS admissions in `forecast-cross-20261104/full/cats_known_future_*`. The 24 h decision logs in `green-window-scheduling-debug/` are no longer read by any figure. `swing.py` reads `offline-shifting/results.csv`, or a file selected with `--offline-results PATH`.

Footprints include idle nodes and context jobs over 37 days for original campaigns, 35 for alpha sensitivity, and 70 for resampled campaigns. Service metrics exclude context jobs.

## CATS and matched-forecast analysis

Recompute metrics using fresh output paths.

```sh
python analysis/scripts/cats.py \
  experiments/experimental-campaigns/cats-carbon/full.toml \
  analysis/outputs/cats-carbon-20261104/full \
  /tmp/cats-metrics.csv

python analysis/scripts/forecast_comparison.py \
  experiments/experimental-campaigns/forecast-cross/full.toml \
  analysis/outputs/forecast-cross-20261104/full \
  /tmp/forecast-comparison
```

The first command writes run metrics. The second writes `metrics.csv` and `contrasts.csv`, using both campaign archives and the original EASY and six-hour head-job schedules.

Generate campaign inputs with `experiments/prepare_cats.py` and `experiments/prepare_forecast_comparison.py`. For simulation reruns, build the scheduler using the patch and build record in each archive's `provenance/` directory. Use a new output directory and place that binary first in PATH. The flake lock selects the original scheduler.
