#!/bin/bash
# Download the 11 released checkpoints (~13 GB) into weights/ for docker build.
# Run from the repo root:  bash scripts/download_weights.sh
# Then:                    docker build -t freuid-repro:local -f docker/Dockerfile .
set -euo pipefail

# GitHub Releases download base (override with WEIGHTS_BASE_URL if mirroring).
# Release assets are FLAT (no directories): one "<member>.ckpt" per checkpoint;
# this script maps each asset back to weights/final/<member>/best.ckpt.
BASE_URL="${WEIGHTS_BASE_URL:-https://github.com/hyun910219/freuid-challenge-2026/releases/download/weights-v1}"

# FB5 + FC3 + FD3 — must match prepare_submission.py and weights_sha256.txt
# (main pick uses the FB4 trim ff_b_fold{0,2,3_v2,4}; fb5 pick uses all 5 FB folds)
FINAL_MEMBERS=(ff_b_fold0 ff_b_fold1 ff_b_fold2 ff_b_fold3_v2 ff_b_fold4
               ff_c_fold0 ff_c_fold1 ff_c_fold2
               ff_d_fold1 ff_d_fold2 ff_d_fold3)

for m in "${FINAL_MEMBERS[@]}"; do
  mkdir -p "weights/final/$m"
  [ -f "weights/final/$m/best.ckpt" ] || curl -fL --retry 3 -o "weights/final/$m/best.ckpt" "$BASE_URL/$m.ckpt"
done

# integrity check against the shipped manifest (sha256)
if [ -f weights_sha256.txt ]; then
  sha256sum -c weights_sha256.txt
fi
echo "OK: all 11 checkpoints present under weights/"
