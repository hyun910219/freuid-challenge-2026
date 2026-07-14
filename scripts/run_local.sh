#!/usr/bin/env bash
# Reproduce the organizers' run: NO network, /data read-only, output to ./out.
#   VARIANT=ens3 (default, Pick 1: FB5+FC3+FD3) | fd (Pick 2: FB5+FD3)
# Needs --shm-size: worker->main IPC uses /dev/shm even with the file_system strategy.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_TAG="${IMAGE_TAG:-freuid-repro:local}"
VARIANT="${VARIANT:-ens3}"        # ens3 = Pick 1 (FB5+FC3+FD3) | fd = Pick 2 (FB5+FD3)

# TODO: point this at a FLAT directory of test images (pass as arg 1 or edit the default).
IMAGES_DIR="${1:-/path/to/flat/test/images}"

if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "ERROR: images dir '${IMAGES_DIR}' not found." >&2
  echo "Usage: scripts/run_local.sh /absolute/path/to/flat/test/images" >&2
  exit 1
fi

mkdir -p out

echo ">> Running ${IMAGE_TAG} (VARIANT=${VARIANT}) with --network none ..."
docker run --rm --network none \
  --gpus all \
  --shm-size=16g \
  -e VARIANT="${VARIANT}" \
  -v "${IMAGES_DIR}:/data:ro" \
  -v "$(pwd)/out:/submissions" \
  "${IMAGE_TAG}"

echo ">> Wrote out/submission.csv"
echo ">> Validate with: python3 scripts/validate_submission.py --submission out/submission.csv --data \"${IMAGES_DIR}\""
