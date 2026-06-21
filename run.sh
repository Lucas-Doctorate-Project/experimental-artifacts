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
# Usage: ./run.sh <campaign.toml>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <campaign.toml>" >&2
  exit 2
fi

campaign="$1"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

podman volume inspect nix-volume >/dev/null 2>&1 || podman volume create nix-volume
podman volume inspect gocache-volume >/dev/null 2>&1 || podman volume create gocache-volume

exec podman run --rm -it \
  -e NIX_CONFIG="experimental-features = nix-command flakes" \
  -e GOCACHE="/gocache" \
  -v nix-volume:/nix \
  -v gocache-volume:/gocache \
  -v "$repo:/work" -w /work/experiments \
  nixos/nix \
  nix develop ../nix --command bash -lc "go run . --campaign '$campaign'"
