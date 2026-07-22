# Experimental Artifacts

Artifacts and runner for reproducible Batsim scheduling experiments evaluating environmental-aware heuristics. The repository is organized in three parts:

- [intensities/](intensities/): carbon and water intensity traces derived from ENTSO-E generation data (PL, FR, DE). The `window_selection` notebook samples 4-week windows for the experiments, and the [offline validation artifact](intensities/validation/README.md) checks national carbon intensities against pinned Ember and EEA snapshots.
- [workloads and platforms/](workloads%20and%20platforms/): Batsim workloads and SimGrid platforms built from production cluster traces (Mustang, Trinity).
- [experiments/](experiments/): Go-based campaign runner that combines the intensity traces with the workloads and platforms to run batches of Batsim/Batsched simulations.

## Quick start

```sh
nix develop ./nix
cd experiments
go run . --campaign experiments.toml
```

The flake lives in [nix/](nix/) so that nix only ever copies the toolchain files into its store, not the multi-gigabyte data traces at the repo root.

### Running without Nix

On a host that has podman but no Nix, use [run.sh](run.sh). It runs the campaign inside the `nixos/nix` image, so the container provides the toolchain and the host needs only podman.

```sh
./run.sh experiments/experiments.toml
```

The campaign path is relative to the repo root. The run is detached so it survives an SSH disconnect, and the container is named after the campaign file (`campaign-<name>`) so its logs stay readable after it exits. The script prints the commands to follow the logs (`podman logs -f <container>`) and to stop the run. Re-running the same campaign reuses the container name, and the script refuses to start if that container is still running.

See each directory's README for details.

## Citation

If you use this repository, cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

This repository is licensed under the [Apache License 2.0](LICENSE), except where otherwise noted. Third-party datasets and artifacts derived from them remain subject to their original source terms.
