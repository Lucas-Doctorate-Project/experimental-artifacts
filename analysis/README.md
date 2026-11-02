# Analysis

From the repository root, enter `nix develop ./nix`, then run:

```sh
python analysis/scripts/results.py
```

The results script reads all three campaigns directly and saves the paired-results
and alpha-sensitivity figures as PNG/PDF in `analysis/figures/pdf/` and `analysis/figures/png/`, replacing the
previous versions. Use `--output-dir PATH` to select a different destination.
Input and default output paths are relative to the repository, so the script
also works from other working directories.

The opportunity script saves `opportunity_capture` to the same folders:

```sh
python analysis/scripts/opportunity.py
```

It compares a delay-only savings bound with the savings Greenfilling achieved
in the original campaign. In the bound, each replay job may start at its
cleanest time between its EASY start and that start plus a delay budget of up
to 24 hours. Jobs are placed independently, without capacity conflicts, so the
bound is not a feasible schedule. Both the bound and Greenfilling are placed at
their mean added wait per replay job, with carbon and water targets in one
panel. Two zoom insets show the Greenfilling results for Mustang and Trinity.

The backfilling script saves `backfill_allowed` to the same folders:

```sh
python analysis/scripts/backfill_allowed.py
```

It shows when Greenfilling allows backfilling in every original Greenfilling
run. The permission is estimated with one moving-average update per distinct
submission or completion time. The actual scheduler can update at other calls
too, so the result is an estimate. Each event's state holds until the next
event, from time zero to the last replay submission. The figure shows the share
of time with backfilling allowed, by intensity quartile within the window and
by the direction of the last 15-minute trace step. Flat steps are computed but
not drawn.

The oracle script saves `oracle_saving_ladder`, `moved_job_intensity`, and
`delay_by_job_class` to the same folders:

```sh
python analysis/scripts/oracle.py
```

It compares Greenfilling with the `green-window-scheduling` oracle at planning
horizons of 6, 12, and 24 hours, each paired with the original EASY runs. The
oracle outputs are split between `analysis/outputs/green-window-scheduling` and
`analysis/outputs/green-window-scheduling-resume`, because the campaign was
interrupted and resumed. The script reads each run from the one directory that
holds it and stops if a run is missing or duplicated. The ladder shows median
targeted saving with interquartile ranges, beside the available saving for a
24-hour budget. The intensity figure compares, for replay jobs that start at a
different time than under EASY, the mean intensity over their execution with
the mean over their EASY slot. The class figure splits the increase in mean
bounded slowdown and in work-weighted response time by job class, summed over
all pairs, for Greenfilling and the oracle with a 6-hour horizon.

Simulation outputs are local and excluded from Git.

The results script compares footprint and service metrics, including alpha sensitivity.

Corrected energy and footprint include idle nodes and context jobs over a common interval: 37 days for original results, 35 days for alpha results, and 70 days for resampled results. The 37-day interval covers the latest finish of the `green-window-scheduling` runs paired with the original EASY runs. The scripts stop with an error if a job finishes after the interval or a trace ends before it. Service metrics exclude context jobs. The resamples change composition and offered load together, with one selected seed per source window.
