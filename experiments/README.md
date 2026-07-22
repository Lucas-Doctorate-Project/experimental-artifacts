# Experiments

Campaign runner for Batsim simulations driven by Batsched. It takes the artifacts produced in this repository (intensity traces from `intensities/`, workloads and platforms from `workloads and platforms/`) and runs them as reproducible experimental campaigns.

Each experiment launches Batsim and Batsched as co-running processes wired over a ZMQ socket. The runner captures their logs and writes output into a per-experiment directory.

## Set up the dev shell

The flake in [../nix/](../nix/) provides `go`, `batsim`, and `batsched`.

```sh
nix develop ../nix
```

All commands below assume you are inside this shell.

## Declare a campaign

A campaign is a TOML file with one or more `[[experiment]]` tables.

```toml
[[experiment]]

name = "example"
workload = "../workloads and platforms/mustang_slack.json"
platform = "../workloads and platforms/mustang.xml"
environmental_trace = "../intensities/traces/FR_2020-01-06.csv"
variant_name = "greenfilling"
variant_options = "options.json"
```

Fields:

- `name`: output directory name under `out/`, created if absent.
- `workload`: path to the Batsim workload JSON.
- `platform`: path to the SimGrid platform XML.
- `environmental_trace`: path passed to Batsim as `--environmental-footprint-dynamic`.
- `variant_name`: Batsched variant, passed as `-v`.
- `variant_options`: path passed to Batsched as `--variant_options_filepath`.

Paths are resolved by the OS. Use paths relative to the directory you run the binary from. Experiment names must be unique within one campaign because the runner uses them as output directory names.

## Declare variant options

The file referenced by `variant_options` is JSON handed to Batsched untouched. For the `greenfilling` variant:

```json
{
    "intensity_trace": "../intensities/traces/FR_2020-01-06.csv",
    "intensity_zone": "AS0",
    "signal": "carbon",
    "smoothing_factor": 0.3,
    "greenfilling_debug": true
}
```

`signal` selects the intensity signal to optimize for (`carbon` or `water`, defaults to `carbon`).

See the Batsched documentation for the keys accepted by each variant.

## Generate the campaign

`gen_campaign.py` writes the full campaign instead of hand-authoring it. It scans the 4-week intensity windows under `../intensities/traces/` (files named `<CC>_<date>.csv`) and emits, over both datasets (Mustang and Trinity):

- `experiments.toml`: EASY and greenfilling (carbon and water) over every window, both datasets, and both regimes (slack and stress), so 3 x 2 x 2 x 36 = 432 experiments.
- `options/{carbon,water}_<CC>_<date>.json`: the greenfilling variant options per window and signal (72 files). These are shared across datasets, since the signal depends only on the intensity window, not the workload.

Experiment names carry the dataset and regime (`<variant>[_<signal>]_<dataset>_<regime>_<CC>_<date>`), so Mustang and Trinity never share an output directory.

Run it from this directory, then run the generated campaign:

```sh
python3 gen_campaign.py
go run . --campaign experiments.toml
```

## Run a campaign

```sh
cd experiments
go run . --campaign experiments.toml
```

Flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--campaign` | `experiments.toml` | Path to the campaign TOML file. |
| `--failure-timeout` | `30s` | Grace period after the other process fails. |
| `--success-timeout` | `30s` | Grace period after the other process exits cleanly. |

The runner launches experiments in declaration order, with up to 4 experiments running at once. The cap is fixed and deliberately well below the core count: each experiment is a batsim+batsched pair locked in ZMQ step, so the live process count is twice the cap, and oversubscribing these lock-step pairs collapses throughput. Each experiment gets its own temporary IPC socket endpoint. There is no cap on per-experiment runtime; simulations run to completion. A failed experiment does not abort the campaign. The program exits `0` only when every experiment succeeds, `1` otherwise.

The runner skips any experiment whose `out/<name>/out_schedule.csv` already exists, so an interrupted campaign resumes where it left off. Delete an experiment's output directory to force it to run again.

## Read the console output

Because up to 4 experiments run at once, their output interleaves. Each line is self-contained and names its experiment, so it stays attributable. A line is printed when an experiment starts and again when it reaches a terminal state:

```
RUN   greenfilling_carbon_stress_FR_2024-07-08
[1/216] SKIP greenfilling_water_stress_DE_2019-02-04  (already complete)
[2/216] OK   greenfilling_carbon_stress_FR_2024-07-08  (3m12s)
[3/216] FAIL easy_bf_stress_PL_2020-07-27  (47s): batsched=<nil> batsim=exit status 1
```

The `[k/N]` counter on terminal lines tracks how many of the `N` campaign experiments have finished, in completion order. `OK` and `FAIL` carry the experiment's wall-clock duration. `RUN` and `OK`/`SKIP` go to stdout, `FAIL` to stderr.

At the end the runner prints a one-line tally and, if any experiment failed, the list of failed names:

```
Summary: 216 total, 208 succeeded, 5 skipped, 3 failed, elapsed 47m12s
Failed:
  - easy_bf_stress_PL_2020-07-27
  - ...
```

## Inspect the output

For an experiment named `example`:

- `out/example/batsched.log`: Batsched stdout and stderr, combined into one file.
- `out/example/batsim.log`: Batsim stdout and stderr, combined into one file.
- `out/example/out_*.csv`: Batsim exports. The main ones are `out_jobs.csv` (per-job metrics) and `out_schedule.csv` (run aggregates).

Log files are opened in append mode. Since the runner skips experiments that already have an `out_schedule.csv`, delete an experiment's directory to give it a clean slate and force a re-run.
