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

The offline shifting script saves `results.csv` to a new directory, by default
`analysis/outputs/offline-shifting`:

```sh
python analysis/scripts/offline_shifting.py
```

It measures how much of the delay-only saving fits on the machine. Starting from
each EASY run in the original and resampled campaigns, it moves replay jobs to
cleaner start times without simulating them again, which is valid because every
job profile has `com = 0`, so a start time changes neither duration nor energy.
Each job may start up to 3, 6, 12, or 24 hours after its EASY start, never
earlier. Jobs with the largest remaining gain move first, and a move needs enough
free nodes on a one-minute grid that counts every touched minute as occupied.
Passes repeat until no job moves. Context jobs stay fixed, and the script stops
if the exact peak allocation exceeds the machine. The schedules assume perfect
knowledge of runtimes and intensity and ignore queue order, and the heuristic can
miss feasible savings, so `saving_pct` shows what is reachable offline, not a
maximum and not what an online policy could do. `bound_saving_pct` is the
delay-only bound for the same run and budget.

To save `offline_shifting`, which compares savings and added wait with Greenfilling
and the oracle in the original campaign, render it from that `results.csv`:

```sh
python analysis/scripts/offline_shifting.py --figure-dir analysis/figures
```

The queue script saves `queue_gain_loss` to the same folders as the other figures:

```sh
python analysis/scripts/queue_gain_loss.py
```

For each oracle run with a 24-hour horizon, it compares every job with the
paired EASY run. Jobs that ran in cleaner hours add to the gain and jobs that
ran in dirtier hours add to the loss. Gain minus loss is the pair's saving. Each
schedule is also scored on the next window of the same zone, which its moves
were not chosen for, to show the gain and loss that moving jobs produces by
chance. The outputs do not record which jobs the oracle held on purpose, so the
split is by outcome, not by cause.

The held-jobs script saves `held_jobs` to the same folders:

```sh
python analysis/scripts/held_jobs.py
```

It reads the decision logs of the oracle rerun in
`analysis/outputs/green-window-scheduling-debug`, which ran the 288 runs with a
24-hour horizon again with `green_window_scheduling_debug` enabled. Each run
holds `out_jobs.csv` and `decisions.txt`, made from `batsched.log` by keeping
`now <date>` for each event the scheduler received and the JSON of each
`green_window_selected` line. A job counts as held when its last chosen start is
later than the date of that decision. The script stops if a rerun schedule
differs from the analyzed one. It splits each pair's saving into held jobs and
all other jobs, scored on the run's own window and on the unrelated window used
by `queue_gain_loss.py`.

The held-walltime script saves `held_walltime` to the same folders:

```sh
python analysis/scripts/held_walltime.py
```

It uses the same decision logs to ask whether held jobs gain little because the
oracle scores the requested walltime while jobs run shorter. Each held replay
job is compared with itself started at the decision date, which is when it
would have started without the hold. The target drop is the relative drop in
mean intensity over the requested walltime, which the oracle compares, and the
got drop is the same over the real runtime. The best drop takes the cleanest
start for the real runtime on the oracle's 15-minute candidate grid within 24
hours, ignoring capacity, so it is optimistic. Idle power during the wait is
left out. The figure groups jobs by runtime as a share of walltime and weights
each job by nodes times runtime.

The swing script saves `swing_savings` to the same folders:

```sh
python analysis/scripts/swing.py
```

It relates each method's saving against EASY in the original campaign to the
daily swing of the window's targeted signal: the delay-only bound and offline
shifting with a 24-hour budget, read from the offline shifting `results.csv`
(select another with `--offline-results PATH`), the oracle with a 24-hour
horizon, and Greenfilling. Swing comes from `intensities/windows.csv`: the mean
daily peak-to-trough range over the 28-day window divided by the window mean.
Each panel shows one point per pair, colored by country, with a least-squares
line per country. Each panel title gives the Spearman rank correlation over all
pairs. The script also prints the correlation per country.

Simulation outputs are local and excluded from Git.

The results script compares footprint and service metrics, including alpha sensitivity.

Corrected energy and footprint include idle nodes and context jobs over a common interval: 37 days for original results, 35 days for alpha results, and 70 days for resampled results. The 37-day interval covers the latest finish of the `green-window-scheduling` runs paired with the original EASY runs. The scripts stop with an error if a job finishes after the interval or a trace ends before it. Service metrics exclude context jobs. The resamples change composition and offered load together, with one selected seed per source window.
