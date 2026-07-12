#!/bin/bash
# Prepare the 11 submission checkpoints as FLAT GitHub Release assets (<member>.ckpt),
# ready for `gh release upload`. Uses SYMLINKS (no 13 GB copy): gh uploads the resolved
# content and names the asset after the symlink's basename (<member>.ckpt), which matches
# the URL scheme that scripts/download_weights.sh expects ($WEIGHTS_BASE_URL/<member>.ckpt).
#
# Run from the repo root:
#   bash scripts/stage_release_assets.sh <KAGGLE_OUTPUTS_DIR> [OUT_DIR]
#   e.g. bash scripts/stage_release_assets.sh \
#          /home/ec2-user/workspace/efs/ml_workspace/kaggle/outputs  dist/release_assets
set -euo pipefail
SRC=${1:?usage: stage_release_assets.sh <KAGGLE_OUTPUTS_DIR> [OUT_DIR]}
OUT=${2:-dist/release_assets}

# Must match prepare_submission.py member lists and weights_sha256.txt.
MEMBERS="ff_b_fold0 ff_b_fold1 ff_b_fold2 ff_b_fold3_v2 ff_b_fold4 \
         ff_c_fold0 ff_c_fold1 ff_c_fold2 \
         ff_d_fold1 ff_d_fold2 ff_d_fold3"

mkdir -p "$OUT"
n=0
for m in $MEMBERS; do
  src="$SRC/$m/best.ckpt"
  [ -f "$src" ] || { echo "MISSING: $src" >&2; exit 1; }
  ln -sf "$(readlink -f "$src")" "$OUT/$m.ckpt"
  n=$((n + 1))
done
echo ">> staged $n/11 symlinked release assets in $OUT/"

# Asset-name-keyed sha256 manifest, DERIVED from the already-verified weights_sha256.txt
# (no re-hash of 13 GB): "weights/final/<m>/best.ckpt" -> "<m>.ckpt".
if [ -f weights_sha256.txt ]; then
  awk '{ nseg = split($2, a, "/"); m = a[nseg - 1]; print $1 "  " m ".ckpt" }' \
      weights_sha256.txt > "$OUT/SHA256SUMS.txt"
  echo ">> wrote $OUT/SHA256SUMS.txt ($(wc -l < "$OUT/SHA256SUMS.txt") entries) — upload this as a release asset too"
fi
echo ">> next:  gh release upload <TAG> $OUT/*.ckpt $OUT/SHA256SUMS.txt"
