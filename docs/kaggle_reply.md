# Reproducibility reply — pinned Kaggle discussion thread

Post **EXACTLY ONE** top-level reply on the pinned Kaggle discussion thread by
**2026-07-15 23:59 AoE**. No follow-ups, no separate threads. Edit this one reply if you
must correct it — never re-post. The README already covers Docker build/run, weights,
external resources, and hardware, so keep the reply short.

Copy the fenced block below verbatim. On Kaggle, paste it inside a fenced code block
(triple backticks) so it renders exactly as written — otherwise Markdown will eat the
`<...>`/`#` characters. Update team name / usernames / date only if they differ.

```text
FREUID Challenge 2026 - Reproducibility Package

Team name: Hyeonwoo Jeon
Kaggle usernames: hyeonwoojeon

Final Kaggle submission (two selected final picks; same commit + weights, inference-time VARIANT flag only):
- Pick 1: submission_fb5_fc3fd3_SUBMIT.csv (2026-07-14 02:55 UTC, public 0.01238) - VARIANT=ens3 (default), captured reorder FB5+FC3+FD3
- Pick 2: submission_fb5_fd3_SUBMIT.csv (2026-07-14 03:14 UTC, public 0.01238) - VARIANT=fd, captured reorder FB5+FD3

Repository (public git): https://github.com/hyun910219/freuid-challenge-2026
Commit SHA: b21e0a8108a9298c18e00694d8c282e993ebdf4f
Technical report (PDF): https://github.com/hyun910219/freuid-challenge-2026/blob/b21e0a8108a9298c18e00694d8c282e993ebdf4f/report/report.pdf
Model weights: GitHub Release assets (tag weights-v1, 11 checkpoints); SHA-256 in weights_sha256.txt, baked into the image at build time (no runtime download).

Docker (offline; requires --shm-size=16g since worker-to-main IPC uses /dev/shm):
  docker build -t freuid-repro .
  Pick 1: docker run --rm --network none --gpus all --shm-size=16g -v /PATH/TO/IMAGES:/data:ro -v /PATH/TO/OUT:/submissions freuid-repro
  Pick 2: docker run --rm --network none --gpus all --shm-size=16g -e VARIANT=fd -v /PATH/TO/IMAGES:/data:ro -v /PATH/TO/OUT:/submissions freuid-repro

Hardware: 1x NVIDIA A100 40GB, 24 CPU cores (inference < 6 h on a single A100).

We confirm this repository at the stated commit reproduces our selected final submissions and complies with the competition rules.

Signed (team captain): Hyeonwoo Jeon
Date (UTC): 2026-07-14
```

## Fill-in notes (do NOT post these)
- Team name / Kaggle usernames / captain: use your Kaggle competition registration.
- Both final picks reproduce from ONE image via `-e VARIANT` (ens3 default / fd). The README and
  report Sec. 6 hold the submission <-> command <-> output-checksum mapping.
- Commit SHA `b21e0a8...` is the frozen package (code + Dockerfile + weights_sha256 + report + PDF).
  Post-freeze commits are inference-orchestration / documentation only (weights unchanged).
- If you post on 2026-07-15, update the Date line accordingly.
