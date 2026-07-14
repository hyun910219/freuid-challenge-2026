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
Final Kaggle submission (two selected final picks — same commit + weights, inference-time VARIANT flag only):
  Pick 1: submission_fb5_fc3fd3_SUBMIT.csv (2026-07-14 02:55 UTC, public 0.01238)
          reproduce: docker run ... freuid-repro                 (default; VARIANT=ens3, captured FB5+FC3+FD3)
  Pick 2: submission_fb5_fd3_SUBMIT.csv    (2026-07-14 03:14 UTC, public 0.01238)
          reproduce: docker run ... -e VARIANT=fd freuid-repro   (captured FB5+FD3)

Repository (public git): https://github.com/hyun910219/freuid-challenge-2026
Commit SHA: b21e0a8108a9298c18e00694d8c282e993ebdf4f
Technical report (PDF): https://github.com/hyun910219/freuid-challenge-2026/blob/b21e0a8108a9298c18e00694d8c282e993ebdf4f/report/report.pdf
Model weights: hosted as GitHub Release assets (tag weights-v1, 11 checkpoints). SHA-256 in
  weights_sha256.txt; fetched at build time and baked into the image (no runtime download).
Docker (offline, exactly as evaluated) — pass --shm-size=16g (worker->main IPC uses /dev/shm):
  docker build -t freuid-repro .
  # Pick 1 (default, VARIANT=ens3):
  docker run --rm --network none --gpus all --shm-size=16g \
    -v <flat_test_images>:/data:ro -v <out>:/submissions freuid-repro
  # Pick 2 (VARIANT=fd):
  docker run --rm --network none --gpus all --shm-size=16g -e VARIANT=fd \
    -v <flat_test_images>:/data:ro -v <out>:/submissions freuid-repro
Hardware: 1x NVIDIA A100 40GB, 24 CPU cores (inference < 6 h on a single A100).

We confirm this repository at the stated commit reproduces our selected final submissions
and complies with the competition rules.

Signed (team captain): Hyeonwoo Jeon
Date (UTC): 2026-07-14

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
