#!/bin/bash
# Materialize the 11 submission checkpoints into weights/ for `docker build`.
# Run from the repo root:  bash scripts/stage_weights.sh <KAGGLE_OUTPUTS_DIR>
# (weights/ is gitignored — it exists only for the docker build context)
set -euo pipefail
SRC=${1:?usage: stage_weights.sh <KAGGLE_OUTPUTS_DIR>}

# FB5 + FC3 + FD3 — must match prepare_submission.py and weights_sha256.txt
FINAL="ff_b_fold0 ff_b_fold1 ff_b_fold2 ff_b_fold3_v2 ff_b_fold4 \
       ff_c_fold0 ff_c_fold1 ff_c_fold2 \
       ff_d_fold1 ff_d_fold2 ff_d_fold3"

for m in $FINAL; do
  mkdir -p "weights/final/$m"
  cp -v "$SRC/$m/best.ckpt" "weights/final/$m/best.ckpt"
done
echo "staged: $(find weights -name best.ckpt | wc -l)/11 ckpts, $(du -sh weights | cut -f1)"
