# FREUID 2026 reproducibility container (frozen-backbone ensemble, FB/FC/FD).
# Built with network; RUN with NO network:
#   docker run --rm --network none --gpus all --shm-size=8g \
#       -v <IMAGES_DIR>:/data:ro -v <OUT_DIR>:/submissions freuid-repro:local
# Two final picks from the SAME image/weights (inference-orchestration only):
#   Pick 1 (main): docker run ... freuid-repro:local                # default VARIANT=main
#   Pick 2 (fb5) : docker run ... -e VARIANT=fb5 freuid-repro:local
# (VARIANT=plain also available: 3-way core, captured lever off — diagnostic.)
# (--shm-size: DataLoader workers need shared memory; the 64MB docker default
#  crashes mid-inference. Without the flag the entrypoint degrades to
#  num_workers=0 — correct but much slower.)
# Input : /data          — flat dir of images (.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff)
# Output: /submissions/submission.csv  — id,label (finite float fraud score, higher=fraud)
#
# Base: torch 2.12 matches requirements.txt; cuda12.6 chosen for evaluator driver compat.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# ---- Dependencies FIRST for layer caching (network is build-time only) ----
COPY requirements.txt /app/
# --break-system-packages: base image python is PEP668 externally-managed; this
# container is single-purpose so a system install is fine.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# ---- Code ----
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY prepare_submission.py /app/prepare_submission.py

# ---- Weights baked into the image (11 ckpts; no runtime download) ----
#   /weights/final/<member>/best.ckpt — ff_b_fold{0,1,2,3_v2,4}, ff_c_fold{0,1,2},
#       ff_d_fold{1,2,3}
COPY weights/ /weights/
# The codebase resolves ckpts under <root>/outputs/<member>/best.ckpt.
RUN ln -s /weights/final /app/outputs

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
# VARIANT=main|fb5|plain — which final pick to reproduce (see prepare_submission.py)
# TTA=0|1 — core hflip-TTA override (default: main/plain 0, fb5 1)
ENV VARIANT=main
ENTRYPOINT ["python", "-u", "/app/prepare_submission.py", \
            "--images-dir", "/data", "--out", "/submissions/submission.csv"]
