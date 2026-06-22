#!/usr/bin/env bash
# Run a campaign on a host that has podman but not nix.
#
# Builds and runs the toolchain inside the nixos/nix image. The nix store and
# the Go build cache live in named volumes so batsim/batsched/simgrid and the
# runner are not rebuilt from scratch on every invocation: nix rebuilds only
# the derivations whose inputs changed, and Go reuses its compiled objects.
#
# Named volumes (not bind mounts) are used for the store so podman seeds the
# volume from the image's own /nix on first run, keeping the nix binary intact.
#
# The campaign runs detached so it survives an SSH disconnect. The container is
# named after the campaign file and is not auto-removed, so its logs remain
# readable after it exits; the next run for the same campaign reuses the name.
#
# Usage: ./run.sh <campaign.toml>
#
# The campaign path is relative to the repo root, e.g. experiments/experiments.toml.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <campaign.toml>" >&2
  exit 2
fi

campaign="$1"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container="campaign-$(basename "$campaign" .toml)"

podman volume inspect nix-volume >/dev/null 2>&1 || podman volume create nix-volume
podman volume inspect gocache-volume >/dev/null 2>&1 || podman volume create gocache-volume

if podman container exists "$container"; then
  if [[ "$(podman inspect -f '{{.State.Running}}' "$container")" == "true" ]]; then
    echo "container '$container' is already running; stop it first with 'podman stop $container'" >&2
    exit 1
  fi
  podman rm "$container" >/dev/null
fi

podman run -d --name "$container" \
  -e NIX_CONFIG="experimental-features = nix-command flakes" \
  -e GOCACHE="/gocache" \
  -v nix-volume:/nix \
  -v gocache-volume:/gocache \
  -v "$repo:/work" -w /work/experiments \
  nixos/nix \
  nix develop ../nix --command bash -lc "go run . --campaign '/work/$campaign' --concurrency 16"

echo "campaign running detached as container '$container'"
echo "follow logs: podman logs -f $container"
echo "stop:        podman stop $container"
