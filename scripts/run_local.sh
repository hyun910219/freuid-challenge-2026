#!/usr/bin/env bash
# Reproduce the organizers' run: NO network, /data read-only, output to ./out.
# Single final pick (fb5) — the entrypoint takes no variant flags.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:-freuid-repro:local}"

# TODO: point this at a FLAT directory of test images (pass as arg 1 or edit the default).
IMAGES_DIR="${1:-/path/to/flat/test/images}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "ERROR: images dir '${IMAGES_DIR}' not found." >&2
  echo "Usage: scripts/run_local.sh /absolute/path/to/flat/test/images" >&2
  exit 1
fi

mkdir -p out

echo ">> Running ${IMAGE_TAG} with --network none ..."
docker run --rm --network none \
  --gpus all \
  -v "${IMAGES_DIR}:/data:ro" \
  -v "$(pwd)/out:/submissions" \
  "${IMAGE_TAG}"

echo ">> Wrote out/submission.csv"
echo ">> Validate with: python3 scripts/validate_submission.py --submission out/submission.csv --data \"${IMAGES_DIR}\""
