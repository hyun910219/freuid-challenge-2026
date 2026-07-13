# FREUID Challenge 2026 — Reproducible Solution

Document fraud detection (Microblink / IJCAI-ECAI). Verified by organizers via
**`git clone` → `docker build` → `docker run --network none`**: runs offline with all
model weights baked into the image, reads a flat directory of test images, and writes a
single `submission.csv`.

## Method (summary)

Three frozen-backbone LoRA classifiers, ensembled by rank, with an inference-side lever
for recaptured (screen/print-recapture) images:

| Component | Detail |
|---|---|
| **FB** | frozen DINOv2-L (`vit_large_patch14_dinov2.lvd142m`) + merged LoRA (q/k/v last-8, r16), 5 folds @448×714 |
| **FC** | frozen OpenCLIP ViT-L/336 (`vit_large_patch14_clip_336.openai_ft_in12k_in1k`) + merged LoRA, 3 folds, CLIP norm |
| **FD** | frozen SigLIP2-L (`vit_large_patch16_siglip_384.v2_webli`) + merged LoRA, 3 folds @448×720, 0.5/0.5 norm |
| Fold aggregation | plain mean of fold scores (per backbone) |
| Backbone combine | equal rank-mean of the 3 backbones (cross-domain rule) |
| Captured detection | resolution-size: native = width >= 1000 px AND (freq >= 0.5% or known cluster); everything smaller (e.g. 840x530 recaptures) = captured |
| capShift | score shift δ=0.75 in logit space, per backbone, captured rows only |
| Captured reorder | within-captured equal mean-rank of the 3 backbones (**ens3** = FB5 + FC3 + FD3) |
| Output | strict total order → `(pos+0.5)/n` rank values |

Frozen backbones preserve the pretraining generalization prior; fine-tuned variants
collapsed on blind held-out countries (leave-type-out screening). The pick runs all 5 FB
folds with hflip-TTA as the core and scores FC3/FD3 on the captured subset only, so the
captured ens3 reorder stays within the 6h/A100 cap. Details in `report/report.md`.

## Contract (must always hold)

| Aspect | Guarantee |
|---|---|
| Network | Inference runs with `--network none`. No runtime downloads. |
| Weights | 11 checkpoints COPY'd into the image at build time. |
| Runtime | Fits **< 6 h on a single A100**: FB5 bf16 hflip-TTA core (~5.4h @ conservative A10G/2.0) + FC3/FD3 on captured rows only, budget-tiered (TTA / noTTA / off). |
| Input | Flat, read-only dir at `/data`. `id` = filename **without** extension. |
| Extensions | `.jpeg .jpg .png .webp .bmp .tif .tiff`. No CSV/manifest/subfolders assumed. |
| Output | Exactly `/submissions/submission.csv`, header `id,label`. One row per image; no missing/extra/duplicate ids. Nothing written outside `/submissions/`. |
| Label | Finite float fraud score, **higher = more fraudulent**. |
| License | Apache-2.0 (`LICENSE`). |

## Prerequisites — weights

The 11 checkpoints (~13 GB) are baked into the image. Populate `weights/` before building:

```bash
git clone https://github.com/hyun910219/freuid-challenge-2026 && cd freuid-challenge-2026
bash scripts/download_weights.sh      # fetches the 11 ckpts from the GitHub Release
                                      # (weights-v1) and verifies weights_sha256.txt
```

Expected layout (checked against `weights_sha256.txt`):
`weights/final/<member>/best.ckpt` — ff_b_fold{0,1,2,3_v2,4}, ff_c_fold{0,1,2}, ff_d_fold{1,2,3}.

## Build

```bash
scripts/build.sh          # verifies weights_sha256.txt, then: docker build -t freuid-repro:local .
```

## Run (offline, exactly as evaluated)

```bash
scripts/run_local.sh /absolute/path/to/flat/test/images

# equivalently:
docker run --rm --network none --gpus all \
  -v /absolute/path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Output → `out/submission.csv`. DataLoader workers use the file_system sharing
strategy (/tmp-backed IPC) instead of `/dev/shm`, so the run needs only
`--network none` + GPU access — no `--shm-size` flag required.

### The final pick (fb5)

A single pick — no variant flags. `docker run` reproduces it directly:

- **core**: FB5 (all 5 FB folds of DINOv2-L), hflip-TTA → FB score order.
- **captured lever**: capShift(δ=0.75) + within-captured **ens3** mean-rank reorder
  (**FB5 + FC3 + FD3**), with FC3 (OpenCLIP ViT-L/336) + FD3 (SigLIP2-L/16-384) scored
  on the captured rows only, tiered (TTA/noTTA/off) by the 6h/A100 budget.

```bash
docker run ... freuid-repro:local && sha256sum out/submission.csv
```

## Validate

```bash
python3 scripts/validate_submission.py \
  --submission out/submission.csv --data /absolute/path/to/flat/test/images
```

Checks: header `id,label`, all labels finite, no duplicate ids, ids exactly match `/data`.

## Layout

```
freuid-solution/
├── Dockerfile              # pytorch 2.12 / cuda12.6 base; COPY deps -> code -> weights
├── prepare_submission.py   # entrypoint: inventory -> infer -> combine -> captured -> write
├── src/                    # FB/FC/FD codebase (models/, data/, infer.py, train.py, utils/)
├── configs/                # member configs: ff_b_fold*, ff_c_fold*, ff_d_fold*
├── weights/                # 11 ckpts (download_weights.sh) -> COPY'd into image
├── weights_sha256.txt      # integrity manifest
├── scripts/                # build.sh, run_local.sh, validate_submission.py, download_weights.sh, private_day.py
├── report/                 # technical report (report.md + make_pdf.py)
└── tests/sample_data/      # tiny local smoke-test images (not in image)
```

## Training

Each member = one config; LoRA is merged into a plain checkpoint on save.

```bash
PYTHONPATH=. python src/train.py --config configs/ff_b_fold0.yaml   # etc.
```

Training data = the official FREUID competition dataset only (no external data used for any
submitted weight).

## External data / pretrained backbones

- **No external datasets** used for any submitted weight (competition dataset only).
- Pretrained backbones (all license-compatible): DINOv2 (Apache-2.0), timm OpenCLIP ViT-L
  (OpenAI CLIP weights via timm, MIT), SigLIP2 `v2_webli` (via timm, Apache-2.0). Cited in
  the technical report (`report/report.md`, §External Resources).

## Reference hardware & runtime

- **Hardware:** 1× NVIDIA A100 (evaluator). **Budget:** < 6 h inference on that single A100.
- **Base image:** `pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime`.
- GPU nondeterminism may perturb raw scores at ~1e-4; the pipeline is rank-based end-to-end.

See [`SUBMISSION_CHECKLIST.md`](SUBMISSION_CHECKLIST.md) before submitting.
