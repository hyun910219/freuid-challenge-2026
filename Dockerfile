# FREUID 2026 reproducibility container (frozen-backbone ensemble, FB/FC/FD).
# Built with network; RUN with NO network (pass --shm-size=16g or --ipc=host —
# worker->main IPC uses /dev/shm even with the file_system strategy on torch 2.12,
# so the 64MB docker default is exhausted and the run crashes without the flag):
#   docker run --rm --network none --gpus all --shm-size=16g \
#       -v <IMAGES_DIR>:/data:ro -v <OUT_DIR>:/submissions freuid-repro:local
# Two final picks from THIS image via a documented flag (weights frozen):
#   Pick 1 (default): docker run ... freuid-repro:local                # VARIANT=ens3 (FB5+FC3+FD3)
#   Pick 2:           docker run ... -e VARIANT=fd freuid-repro:local  # (FB5+FD3)
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
