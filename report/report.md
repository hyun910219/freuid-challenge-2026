# FREUID Challenge 2026 — Technical Report

<!-- Convert to PDF before 07-15 AoE. Toolchain verified 2026-07-10:
     system python3 has weasyprint 66.0 (markdown -> HTML -> PDF); no pandoc.
     [D-DAY] markers = fill after private-day execution (07-13/14). -->

## 1. Introduction

We address fraud detection in identity documents under a cross-domain private
test regime (two undisclosed document types + recaptured images). Our guiding
finding: **fine-tuned backbones overfit the seen countries and collapse on
blind domains, while frozen pretrained backbones with lightweight LoRA
adapters preserve the pretraining generalization prior.** The final system is
a rank-mean ensemble of three frozen-backbone classifiers with an
inference-side lever for recaptured images.

## 2. Method

### 2.1 Models

- **FB** — frozen DINOv2-L (`vit_large_patch14_dinov2.lvd142m`), LoRA on
  q/k/v of the last 8 blocks (r=16, alpha=32), MLP head (hidden 256),
  input 448x714 (aspect resize), 5 folds. LoRA merged into plain weights
  after training.
- **FC** — frozen OpenCLIP ViT-L/336
  (`vit_large_patch14_clip_336.openai_ft_in12k_in1k`), identical recipe,
  CLIP normalization. Decorrelated pretraining prior (image-text contrastive
  vs. self-supervised) is the source of the ensemble gain.
- **FD** — frozen SigLIP2-L/16-384 (`vit_large_patch16_siglip_384.v2_webli`),
  identical recipe, input 448x720, 0.5/0.5 normalization. A third,
  independently-pretrained prior (sigmoid image-text, WebLI).

All backbones are trained as 5-fold CV. The final pick uses all **5 FB folds**
with hflip-TTA as its core; **FC3** (3 folds) and **FD3** (3 folds) score the
captured subset only, so the captured ens3 reorder stays within the 6h/A100
inference cap (Sec. 4).

Blind leave-country-out (LTO) evidence for the frozen-vs-finetuned choice:
fine-tuned FA collapses to 0.18 on held-out GUINEA; frozen FB reaches 0.030.
Adding FC improves blind rank-mean by -35% (GUINEA) / -27% (EGYPT) and pooled
OOF by -70%. Adding FD improves blind rank-mean further (-4.6% GUINEA /
-54% EGYPT vs. the FB+FC pair; FD is the strongest single backbone on
held-out EGYPT) and pooled OOF by another -57%.

### 2.2 Training

- Data: official FREUID training set ONLY (no external datasets for any
  submitted weight; pretrained backbones are Apache-2.0, see Sec. 7).
- 5-fold CV. Loss: BCE. AMP. Epoch budget 4 with early best-selection
  (frozen-backbone LoRA converges at epoch 0-1).
- Augmentation: standard photometric/geometric domain aug + tamper-synth
  (p=0.3) + G1 recapture-style aug. Heavier domain-randomization
  (FACT amplitude-spectrum mix, Shades-of-Gray color constancy) DESTROYED
  the fraud signal in ablation (train loss stuck at random plateau) and was
  rejected.

### 2.3 Inference pipeline

1. Per-backbone fold plain mean -> per-backbone score.
2. **Cross-backbone equal rank-mean over FB/FC/FD** (score-mean dilutes the
   weaker-calibrated backbone on out-of-domain data).
3. **Recaptured-row detection** by resolution: a resolution is native only if
   it is high-resolution (width >= 1000 px) AND either frequent (>= 0.5% of
   rows) or a known native cluster; everything else — including frequent but
   low-resolution formats such as 840x530 — is captured. In the training data
   digital captures are always >= 1000 px wide while recaptured/downscaled
   images are smaller, so width cleanly separates the two acquisition types
   (a frequent 840x530 recapture format would otherwise be misread as native).
4. **capShift**: captured rows are shifted by delta = 0.75 in logit space
   (per backbone) — recaptured bona fide images otherwise inflate the fraud
   tail (APCER at 1% BPCER dominates the metric). Public-LB verified:
   0.01870 -> 0.01524.
5. **Captured-internal reorder (ens3)**: within captured rows, order by the
   equal mean-rank of the three backbones (FB5 + FC3 + FD3). FC3/FD3 are
   scored on the captured subset only, in a TTA / noTTA / off tier chosen by
   the remaining inference budget at the observed captured fraction (captured
   ordering is robust to the TTA reduction, spearman 0.9958); capShift is
   CPU-side and always applies. The reorder mechanic is public-LB verified
   (0.01524 -> 0.01304 -> 0.01252 as the captured-reorder ensemble grew); the
   shipped ens3 orders captured rows by the three released backbones only
   (FB5 + FC3 + FD3). The intermediate 0.01304 / 0.01252 points additionally
   used recapture-specialist checkpoints that are not part of this release;
   the released package reproduces the ens3 composition, which keeps the
   captured lever within the inference budget.
6. Output = strict total order -> (pos+0.5)/n rank values (metric is
   rank-only; ties score as constants).

The single final pick (**fb5**) runs this pipeline with an FB5 hflip-TTA core:
step 1 aggregates the 5 FB folds, step 2 degenerates to the FB score order,
and FC3/FD3 enter only through the captured reorder in step 5.

Precision: bf16 (rank-safety spearman vs fp32: FB 0.99992, FC 0.999799,
FD 0.999627).
Ensemble-level / cross-backbone TTA on the core was rejected by measurement —
blind leave-country-out showed degradation (+27.8% on held-out EGYPT; hflip
severely harms the CLIP backbone on documents). The FB5 core keeps hflip-TTA
(benign for the DINOv2 backbone; part of the pick's frozen public
composition), while FC3/FD3 touch only the captured subset.

## 3. Data

- Training: FREUID official training set (69,352 images), 5-fold splits.
  **No external dataset was used at any stage — not for training, and not for
  model selection.** The submitted weights and every selection decision rely on
  the competition data (and public-leaderboard feedback) only.
- External model weights (pretrained backbones only): Sec. 7.
- Validation protocol: blind leave-country-out (GUINEA fold, EGYPT fold) —
  the only offline signal that correlated with cross-domain behavior; pooled
  OOF does NOT predict LB (measured: best-OOF model was worst on LB).

## 4. Inference / Budget

- Evaluator (organizer-confirmed): 1x NVIDIA A100 40 GB, 24 CPU cores, <= 6 h
  on the hidden test set.
- End-to-end offline (`--network none`) measurement on a single A10G (a ~2-2.7x
  slower proxy for the A100): 7,821 images in ~33 min for the full pipeline
  (~0.25 s/img). Projected to the full 142,818-image test set: ~10 h on A10G ->
  **~5 h on A100 at a conservative /2.0 factor** (~3.5 h at a realistic /2.7),
  within the 6 h budget. Peak GPU memory < 23 GB (fits A100 40 GB).
- torch.compile is applied to the inference model (rank-preserving; raw scores
  shift ~1e-2 but the pipeline is rank-based end-to-end). It amortizes over the
  many batches of the full test set (~+17% throughput measured on A10G).
- The FC3/FD3 captured pass (25.8 / 27.7 ms/img/ckpt on A10G) touches captured
  rows only, in the budget tier of Sec. 2.3 step 5 (TTA up to c<=9.8%, noTTA
  up to c<=19.6%, off beyond — capShift still applies). Core cost (FB5
  hflip-TTA) dominates and scales linearly with the image count, so the run
  stays within the cap regardless of the private captured fraction.

## 5. Results

| Stage | Public LB |
|---|---|
| FB5 (frozen DINOv2-L + LoRA) | 0.01870 |
| + capShift (delta 0.75) | 0.01524 |
| + captured reorder, early ensemble (¹) | 0.01304 |
| + captured reorder, all backbones (¹) | 0.01252 |
| + captured ens3 reorder (released FB5+FC3+FD3, size-based routing) | 0.01238 |
| Container-native output (`docker run`, rank values, reproduces the above) | 0.01245 |
| Private fb5 pick (FB5 hflip-TTA + captured lever) | hidden until 2026-07-24 |

(¹) These two intermediate public-LB points additionally used
recapture-specialist checkpoints that are not part of this release. They are
shown only to document the reorder mechanic's progression; the released
package reproduces the ens3 (FB5+FC3+FD3) composition.

Blind LTO (held-out country, freuid score, lower=better):

| Model | GUINEA | EGYPT |
|---|---|---|
| FA (fine-tuned) | 0.18 | — |
| FB (frozen) | 0.0297 | 0.0647 |
| FC (frozen) | 0.0325 | 0.0656 |
| FD (frozen) | 0.0813 | 0.0143 |
| FB+FC rank-mean | 0.0194 | 0.0474 |
| FB+FC+FD rank-mean | 0.0185 | 0.0218 |

FD's country profile is the mirror of FB/FC (weakest on GUINEA, 4.5x
strongest on EGYPT) — direct evidence of prior decorrelation; the 3-way
rank-mean improves both held-out countries.

The public leaderboard rows above keep the FB-based ordering: the public
main block is in-domain, where cross-backbone rank-mean measurably loses
to the single-backbone/score-mean orderings (consistent with pooled OOF);
the 3-backbone rank-mean targets the cross-domain private blocks, per the
blind LTO table.

## 6. Reproducibility

- Repo: https://github.com/hyun910219/freuid-challenge-2026, commit SHA:
  __NEW_SHA__. Model weights are frozen throughout; every post-freeze commit is
  inference-orchestration / documentation only (size-based captured routing; the
  `--shm-size` runtime requirement + a `num_workers=0` fallback; the `VARIANT`
  flag selecting the captured-reorder backbones). No weight/architecture/training
  change at any point.
- Runtime: run with `--shm-size=16g` (or `--ipc=host`). DataLoader worker->main
  tensor IPC uses `/dev/shm` even with the file_system sharing strategy on
  torch 2.12; the 64 MB docker default is exhausted otherwise. The entrypoint
  falls back to `num_workers=0` when the flag is absent (correct but slower).
- Two final picks from ONE image via a documented inference-time flag
  (host-approved: same commit + weights, flag only). Submission <-> command <->
  output checksum:

  | Final pick (Kaggle id / timestamp) | Command (all `--network none --shm-size=16g`) | sha256(container `submission.csv`) |
  |---|---|---|
  | **Pick 1** — `submission_fb5_fc3fd3_SUBMIT.csv` / 2026-07-14 02:55 UTC | `docker run ... freuid-repro:local` (`VARIANT=ens3`, captured = FB5+FC3+FD3) | `c09feef...908fc77e` |
  | **Pick 2** — `submission_fb5_fd3_SUBMIT.csv` / 2026-07-14 03:14 UTC | `docker run ... -e VARIANT=fd freuid-repro:local` (captured = FB5+FD3) | `ac7eb76...eab33efe` |

  (Reference sha256 from our A10G run; a fresh run matches at rank-Spearman
  >= 0.9999 — bf16/compile perturbs raw scores ~1e-2, the pipeline is rank-based.)

  Per the organizers, only the ranking submission (the better of the two selected
  picks) must reproduce; both are documented. The container output's public block
  and row order differ from the uploaded CSV (public rows were frozen mid-
  competition, see the note on the submitted file); the private-block ranking —
  which decides the score — is what the container reproduces.
- Docker: no-network container, weights baked in; flat `/data` in,
  `/submissions/submission.csv` out. See README. Verified end-to-end
  (`git clone` -> `docker build` -> `docker run --network none`) on public images
  -> valid, contract-compliant `submission.csv` offline.
- Training: one config per member, `src/train.py --config configs/<m>.yaml`.
- Deterministic postprocessing; GPU/compile nondeterminism shifts raw scores at
  ~1e-2 but the pipeline is rank-based (the offline container reproduces our
  submitted ranking at Spearman >= 0.9999).

## 7. External Resources

| Resource | License | Use |
|---|---|---|
| DINOv2 ViT-L `lvd142m` (timm/Meta) | Apache-2.0 (repo LICENSE verified) | FB backbone |
| OpenCLIP ViT-L/336 `openai_ft_in12k_in1k` (timm) | Apache-2.0 (OpenAI CLIP origin: MIT) | FC backbone |
| SigLIP2 ViT-L/16-384 `v2_webli` (timm/big_vision) | Apache-2.0 (big_vision repo LICENSE verified) | FD backbone |
| PyTorch 2.12 / timm 1.0.27 / albumentations 2.0.8 etc. | BSD/Apache/MIT | framework |

Only the pretrained backbones above (all license-compatible) and the listed
frameworks are used. No external dataset was used at any stage — not to train any
submitted weight, and not for model selection.
