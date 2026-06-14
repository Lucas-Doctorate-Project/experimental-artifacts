# Experiments

Campaign runner for Batsim simulations driven by Batsched. It takes the artifacts produced in this repository (intensity traces from `intensities/`, workloads and platforms from `workloads and platforms/`) and runs them as reproducible experimental campaigns.

Each experiment launches Batsim and Batsched as co-running processes wired over a ZMQ socket. The runner captures their logs, enforces timeouts, and writes output into a per-experiment directory.

## Set up the dev shell

The flake at the repo root provides `go`, `batsim`, and `batsched`.

```sh
nix develop
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
    "smoothing_factor": 0.3,
    "ema_threshold": 1.0,
    "backfilling_combinator": "and",
    "greenfilling_debug": true
}
```

See the Batsched documentation for the keys accepted by each variant.

## Run a campaign

```sh
cd experiments
go run . --campaign experiments.toml
```

Flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--campaign` | `experiments.toml` | Path to the campaign TOML file. |
| `--simulation-timeout` | `1h` | Max wall-clock time per experiment. |
| `--failure-timeout` | `30s` | Grace period after the other process fails. |
| `--success-timeout` | `30s` | Grace period after the other process exits cleanly. |

The runner launches experiments in declaration order, with up to `runtime.NumCPU()` experiments running at once. Each experiment gets its own temporary IPC socket endpoint. A failed experiment does not abort the campaign. The program exits `0` only when every experiment succeeds, `1` otherwise.

## Inspect the output

For an experiment named `example`:

- `out/example/batsched.log`, `out/example/batsched.err`: Batsched stdout and stderr.
- `out/example/batsim.log`, `out/example/batsim.err`: Batsim stdout and stderr.
- `out/example/out_*.csv`: Batsim exports. The main ones are `out_jobs.csv` (per-job metrics) and `out_schedule.csv` (run aggregates).

Log files are opened in append mode. Delete the directory between runs for a clean slate.