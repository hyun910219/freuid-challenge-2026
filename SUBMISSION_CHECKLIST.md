# FREUID 2026 — Submission Checklist

Run through this before every submission and before the final code freeze.

## Runtime & hardware
- [ ] Full inference on the hidden test set finishes **< 6 h on a single A100**.
- [ ] Ensemble/TTA justified by budget: FB5 bf16 hflip-TTA core fits the cap, with the
      FC3/FD3 captured-subset pass budget-tiered (TTA / noTTA / off) at the observed
      captured fraction.
- [ ] bf16 autocast verified rank-safe and within budget.

## No-network contract
- [ ] Image builds, then runs successfully with `docker run --network none`.
- [ ] **All** weights are COPY'd into the image at build time — nothing downloaded at runtime.
- [ ] No `wget`/`curl`/`pip install`/hub download in any code path hit during inference.
- [ ] `--gpus all` used for the A100; container never assumes internet.

## Input handling (flat `/data`)
- [ ] Reads a **flat**, read-only dir at `/data`; no CSV/manifest/subfolders assumed.
- [ ] All 7 extensions handled: `.jpeg .jpg .png .webp .bmp .tif .tiff` (case-insensitive).
- [ ] `id` = filename **without** extension.
- [ ] Every image produces exactly one row (unreadable images still get a row).

## Output schema (`/submissions/submission.csv`)
- [ ] Exactly one file written: `/submissions/submission.csv`. Nothing written outside `/submissions/`.
- [ ] Header is exactly `id,label`.
- [ ] One row per input image — no missing, extra, or duplicate ids.
- [ ] Every `label` is a **finite float**; higher = more fraudulent.
- [ ] `scripts/validate_submission.py --submission out/submission.csv --data <images>` passes.

## Weights (11 ckpts)
- [ ] Weights staged under `weights/final/` (11), matching
      `weights_sha256.txt` (`sha256sum -c weights_sha256.txt` passes).
- [ ] Populated via `scripts/download_weights.sh` — real checkpoint files.
- [ ] Member lists in `prepare_submission.py` (FB/FB5/FC/FD) match
      the staged checkpoints and `weights_sha256.txt`.
- [ ] `docker build` COPYs `weights/` into `/weights/`; entrypoint resolves via
      `/app/outputs` → `/weights/final`.

## Licensing & external data
- [ ] `LICENSE` is OSI-approved (Apache-2.0).
- [ ] All external data is public and license-compatible; shipped-weight training data is
      Apache-2.0 / MIT / CC-BY only (no ShareAlike / NonCommercial baked into weights).
- [ ] External data + licenses declared in `README.md` and the technical report.

## Reproducibility deliverables
- [ ] `git clone → docker build → docker run --network none` reproduces the submission.
- [ ] Pinned deps (`requirements.txt`) build against `pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime`.
- [ ] Technical report PDF (`report/`, built from `report.md` via `make_pdf.py`) covers
      method, data, inference, results, repro commands. `[D-DAY]` results filled post private-day.
- [ ] Final pick documented: `fb5` (single pick, no variant flags), with the
      submission ↔ command ↔ output-checksum mapping recorded in `README.md`.

## Code freeze & final steps
- [ ] Whole inference pipeline (pre/post-processing included) is in the frozen commit; after
      the freeze only inference/doc/packaging changes (no weights/architecture/training edits).
- [ ] Public repo + 40-char commit SHA; weights hosted (LFS or release/bucket) and reachable.
- [ ] Final commit tagged; LFS objects pushed / weights hosting URL live.
- [ ] **Single pinned-thread reply** posted with the required reproducibility info
      (repo/commit, docker image, hardware) — post exactly once (edit, never re-post).
