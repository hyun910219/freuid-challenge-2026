# FREUID 2026 reproducibility container (frozen-backbone ensemble, FB/FC/FD).
# Built with network; RUN with NO network:
#   docker run --rm --network none --gpus all \
#       -v <IMAGES_DIR>:/data:ro -v <OUT_DIR>:/submissions freuid-repro:local
# Single final pick (fb5) — no variant flags (see prepare_submission.py):
#   docker run ... freuid-repro:local
# (No --shm-size needed: DataLoader workers use the file_system sharing strategy
#  (/tmp-backed IPC) instead of /dev/shm, so the 64MB docker default is fine.)
# Input : /data          — flat dir of images (.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff)
# Output: /submissions/submission.csv  — id,label (finite float fraud score, higher=fraud)
#
# Base: torch 2.12 matches requirements.txt; cuda12.6 chosen for evaluator driver compat.
FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# ---- Build toolchain: torch.compile (Inductor) JIT-compiles at the first forward
#      pass and needs a host C/C++ compiler. The *-runtime base image ships without
#      one, so install g++ at build time (network is build-time only). Without it,
#      torch.compile would raise under --network none at inference with no way to
#      recover; prepare_submission.py additionally wraps compile in a suppress_errors
#      + try/except eager fallback (belt-and-suspenders). ----
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

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
ENTRYPOINT ["python", "-u", "/app/prepare_submission.py", \
            "--images-dir", "/data", "--out", "/submissions/submission.csv"]
