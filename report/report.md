# FREUID Challenge 2026 — Technical Report (DRAFT)

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

All backbones are trained as 5-fold CV. The **main pick** uses a **4/3/3
fold subset (FB/FC/FD)** — the weakest pooled-OOF folds per backbone were
dropped so the three-backbone core fits the 6h/A100 inference cap (Sec. 4);
the **fb5 pick** runs all 5 FB folds with hflip-TTA as its core.

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
3. **Recaptured-row detection** by resolution frequency: native resolution
   clusters cover >= 0.5% of rows each; everything else is captured.
4. **capShift**: captured rows are shifted by delta = 0.75 in logit space
   (per backbone) — recaptured bona fide images otherwise inflate the fraud
   tail (APCER at 1% BPCER dominates the metric). Public-LB verified:
   0.01870 -> 0.01524.
5. **Captured-internal reorder (ens3)**: within captured rows, order by the
   equal mean-rank of the three backbones (FB + FC + FD). In the main pick
   the FC/FD ranks reuse the core scores from step 1 at zero extra GPU cost;
   in the fb5 pick FC3/FD3 are scored on the captured subset only, in a
   TTA / noTTA / off tier chosen by the remaining inference budget at the
   observed captured fraction (captured ordering is robust to the TTA
   reduction, spearman 0.9958); capShift is CPU-side and always applies.
   The reorder mechanic is public-LB verified (0.01524 -> 0.01304 -> 0.01252
   as the captured-reorder ensemble grew); the shipped ens3 orders captured
   rows by the three released backbones only (FB + FC + FD). The intermediate
   0.01304 / 0.01252 points additionally used recapture-specialist
   checkpoints that are not part of this release; the released package
   reproduces the ens3 composition, which keeps the captured lever within
   the inference budget.
6. Output = strict total order -> (pos+0.5)/n rank values (metric is
   rank-only; ties score as constants).

Two final picks share this pipeline: **main** = the 3-way core (steps 1-6,
noTTA); **fb5** = FB5 hflip-TTA core, where step 2 degenerates to the FB
score order and FC/FD enter only through step 5.

Precision: bf16 (rank-safety spearman vs fp32: FB 0.99992, FC 0.999799,
FD 0.999627).
Ensemble-level TTA: rejected by measurement — blind leave-country-out showed
degradation (+27.8% on held-out EGYPT; hflip severely harms the CLIP backbone
on documents), so the 3-way core runs noTTA. The FB-only fb5 pick keeps
hflip-TTA (benign for the DINOv2 backbone; part of that pick's frozen
public composition).

## 3. Data

- Training: FREUID official training set (69,352 images), 5-fold splits.
  **No external dataset was used to train any submitted weight.**
- External public datasets were used for offline evaluation only
  (regression-guard battery; never seen by any training pipeline): FantasyID,
  IDNet (eval holdout), MIDV-Holo, DLC-2021. Full list + licenses: Sec. 7.
- External model weights: Sec. 7.
- Validation protocol: blind leave-country-out (GUINEA fold, EGYPT fold) —
  the only offline signal that correlated with cross-domain behavior; pooled
  OOF does NOT predict LB (measured: best-OOF model was worst on LB).

## 4. Inference / Budget

- Full test (142,818 images) through FB4+FC3+FD3 bf16 noTTA:
  ~4.9h A100-equivalent at /2.2, ~5.3h at a conservative /2.0 (measured
  27.1 / 25.8 / 27.7 ms/img/ckpt for FB / FC / FD on A10G), within the
  6h/A100 budget. All 12+-fold member sets exceed the cap — hence the
  4/3/3 weakest-OOF-first fold trim. torch.compile measured slower than
  eager (x0.73-0.79) and is not used.
- fb5 pick: FB5 bf16 hflip-TTA core ~5.4h at /2.0; its FC3/FD3 pass touches
  captured rows only, in the budget tier of Sec. 2.3 step 5 (TTA up to
  c<=9.8%, noTTA up to c<=19.6%, off beyond — capShift still applies).
- [D-DAY: actual private captured fraction c, captured-lever GO/NO-GO,
  fb5 FC/FD tier, wall time]

## 5. Results

| Stage | Public LB |
|---|---|
| FB5 (frozen DINOv2-L + LoRA) | 0.01870 |
| + capShift (delta 0.75) | 0.01524 |
| + captured reorder, early ensemble (¹) | 0.01304 |
| + captured reorder, all backbones (¹) | 0.01252 |
| + captured ens3 reorder (released FB+FC+FD) | [07-13] |
| Private main pick (FB4+FC3+FD3 rank-mean + captured lever) | [D-DAY] |
| Private fb5 pick (FB5 hflip-TTA + captured lever) | [D-DAY] |

(¹) These two intermediate public-LB points additionally used
recapture-specialist checkpoints that are not part of this release. They are
shown only to document the reorder mechanic's progression; the released
package reproduces the ens3 (FB+FC+FD) composition.

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

- Repo: https://github.com/hyun910219/freuid-challenge-2026, frozen commit SHA: [SHA].
- Docker: no-network container, weights baked in; flat `/data` in,
  `/submissions/submission.csv` out. See README.
- Training: one config per member, `src/train.py --config configs/<m>.yaml`.
- Deterministic postprocessing; GPU nondeterminism affects raw scores at
  ~1e-4 (pipeline is rank-based).

## 7. External Resources

| Resource | License | Use |
|---|---|---|
| DINOv2 ViT-L `lvd142m` (timm/Meta) | Apache-2.0 (repo LICENSE verified) | FB backbone |
| OpenCLIP ViT-L/336 `openai_ft_in12k_in1k` (timm) | Apache-2.0 (OpenAI CLIP origin: MIT) | FC backbone |
| SigLIP2 ViT-L/16-384 `v2_webli` (timm/big_vision) | Apache-2.0 (big_vision repo LICENSE verified) | FD backbone |
| PyTorch 2.12 / timm 1.0.27 / albumentations 2.0.8 etc. | BSD/Apache/MIT | framework |
| FantasyID (Idiap, zenodo 17063366) | CC-BY 4.0 + CC0 | **eval-only** (offline battery) |
| IDNet (eval holdout subset) | CC-BY 4.0 | **eval-only** (offline battery) |
| MIDV-Holo | CC-BY-SA 2.5 | **eval-only** (offline battery) |
| DLC-2021 (zenodo 6466768 / 6792396) | CC-BY-SA 2.5 | **eval-only** (offline battery) |

No external dataset was used to train any submitted weight; the four datasets
above were used exclusively as an offline evaluation battery and are cited
per the competition's external-resource rules (Rules Sec. 6).
