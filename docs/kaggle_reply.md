# Reproducibility reply — pinned Kaggle discussion thread

Post **EXACTLY ONE** top-level reply on the pinned Kaggle discussion thread by
**2026-07-15 23:59 AoE**. No follow-ups, no separate threads. Edit this one reply if you
must correct it — never re-post. The README already covers Docker build/run, weights,
external resources, and hardware, so keep the reply short.

Paste the block below (plain text) with every `<...>` / `[...]` field filled in.

---
FREUID Challenge 2026 - Reproducibility Package
---

Team name: Hyeonwoo Jeon
Kaggle usernames: hyeonwoojeon
Final Kaggle submission: 2026-07-13_ffb5_container_native_public.csv (2026-07-13 10:17 UTC, public 0.01245)

Repository (public git): https://github.com/hyun910219/freuid-challenge-2026
Commit SHA: 88c6686e899fbc07d0117b4be37b48b1fb00b520
Technical report (PDF): https://github.com/hyun910219/freuid-challenge-2026/blob/main/report/report.pdf
Model weights: hosted as GitHub Release assets (tag weights-v1, 11 checkpoints). SHA-256 in
  weights_sha256.txt; fetched at build time and baked into the image (no runtime download).
Docker (offline, exactly as evaluated):
  docker build -t freuid-repro .
  docker run --rm --network none --gpus all \
    -v <flat_test_images>:/data:ro -v <out>:/submissions freuid-repro
Hardware: 1x NVIDIA A100 40GB, 24 CPU cores (inference < 6 h on a single A100).

We confirm this repository at the stated commit reproduces our selected final submission
and complies with the competition rules.

Signed (team captain): Hyeonwoo Jeon
Date (UTC): 2026-07-13

---

## Fill-in notes (do NOT post these)
- **Repository / Commit SHA**: after `git push`, freeze the 40-char SHA of the commit that
  contains code + Dockerfile + weights_sha256.txt + the finalized report. The rules require
  the solution/weights public by 2026-07-13 and forbid weight/architecture/training changes
  after; documentation/packaging commits after are allowed if the weights are unchanged.
- **Weights (tag `weights-v1`)**: the 11 Release assets `<member>.ckpt` + `SHA256SUMS.txt`
  (see scripts/stage_release_assets.sh + scripts/download_weights.sh). The build verifies
  every checkpoint against weights_sha256.txt and fails loudly on any mismatch.
- **Technical report URL**: swap to a `/<SHA>/` permalink (as above) to pin it to the exact
  frozen commit; the `/master/` (or default-branch) blob URL also renders in-browser.
- **Team name / Kaggle usernames / captain**: use your Kaggle competition registration.
- Reply = this block only; organizers follow the repo README, not a second copy on the forum.
