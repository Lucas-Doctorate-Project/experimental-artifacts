# Experimental Artifacts

Artifacts and runner for reproducible Batsim scheduling experiments evaluating environmental-aware heuristics. The repository is organized in three parts:

- [intensities/](intensities/): carbon and water intensity traces derived from ENTSO-E generation data (PL, FR, DE). The `window_selection` notebook samples 4-week windows for the experiments.
- [workloads and platforms/](workloads%20and%20platforms/): Batsim workloads and SimGrid platforms built from production cluster traces (Mustang, Trinity, MetaCentrum, CC-IN2P3).
- [experiments/](experiments/): Go-based campaign runner that combines the intensity traces with the workloads and platforms to run batches of Batsim/Batsched simulations.

## Quick start

```sh
nix develop
cd experiments
go run . --campaign experiments.toml
```

See each directory's README for details.