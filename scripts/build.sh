#!/usr/bin/env bash
# Build the FREUID inference image. Network IS allowed here (build time only).
# Weights must be present under weights/ first (scripts/download_weights.sh);
# the Dockerfile COPYs all 11 checkpoints into the image.
set -euo pipefail

# Run from the repo root regardless of where this is invoked from.
cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:-freuid-repro:local}"

# Verify weights are staged + match the shipped manifest before building.
if [ -f weights_sha256.txt ]; then
  if ! sha256sum -c weights_sha256.txt --quiet 2>/dev/null; then
    echo "ERROR: weights/ missing or checksum mismatch." >&2
    echo "  Populate first:  git lfs pull   OR   bash scripts/download_weights.sh" >&2
    exit 1
  fi
  echo ">> weights sha256 OK (11/11)"
fi

echo ">> Building ${IMAGE_TAG} ..."
docker build -t "${IMAGE_TAG}" .
echo ">> Built ${IMAGE_TAG}"
