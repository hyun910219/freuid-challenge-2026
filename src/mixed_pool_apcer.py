"""Mixed-pool APCER stress test (04_external_eval.md §3) — private failure sim.

Private test = unseen countries. Its failure mechanism (per the FREUID metric
analysis): unseen-domain BONA whose scores sit high inflate the 1%-BPCER
threshold, which collapses APCER@1%BPCER (the winning axis). This harness
reproduces that mechanism WITHOUT retraining, by re-scoring a fixed attack set
against an ENLARGED bona pool = (FREUID val bona ∪ external bona):

    attacks(FREUID val)  fixed
    bona pool            = FREUID val bona ∪ external bona  (T1+T2 battery)
    mixed pool           = attacks ∪ bona pool  ->  official metric

Outputs (all via the official scorer, src.eval_freuid):
  - mixed_freuid / mixed_apcer1 / mixed_audet   (headline: mixed-pool score)
  - base_freuid  / base_apcer1                  (FREUID-val-only baseline)
  - apcer_inflation = mixed_apcer1 - base_apcer1
  - thr_val / thr_mixed                          (score at 1% BPCER on each bona pool)
  - ext_bona_false_alarm = mean(external bona >= thr_val)  (how many unseen-domain
    bona the in-dist 1%-BPCER threshold would wrongly reject / push the tail)

Interpretation: a model that scores external (unseen-domain) bona LOW keeps
thr_mixed ≈ thr_val and apcer_inflation ≈ 0 -> robust for private. This is a
"directional signal" (regression guard / tiebreaker), NOT a gate (04 §5).

No retraining, no GPU, no cv2 — consumes score csvs only. Runnable self-test:
    python3 kaggle/final/src/mixed_pool_apcer.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_freuid import freuid_eval  # noqa: E402


def _threshold_at_bpcer(bona_scores: np.ndarray, target: float) -> float:
    """Smallest tau with BPCER = mean(bona >= tau) <= target (official convention)."""
    b = np.sort(np.asarray(bona_scores, dtype=float))[::-1]  # desc
    n = len(b)
    k = int(np.floor(target * n + 1e-12))  # allow up to k bona above tau
    if k <= 0:
        return float(b[0]) + 1e-12  # reject none -> tau just above max bona
    if k >= n:
        return float(b[-1])
    # tau just above the (k+1)-th largest score so exactly k bona are >= tau
    return float(b[k])


@dataclass(frozen=True)
class MixedPoolResult:
    base_freuid: float
    base_apcer1: float
    base_audet: float
    mixed_freuid: float
    mixed_apcer1: float
    mixed_audet: float
    apcer_inflation: float
    thr_val: float
    thr_mixed: float
    thr_shift: float
    ext_bona_false_alarm: float
    n_attack: int
    n_val_bona: int
    n_ext_bona: int


def mixed_pool_report(
    val_scores: pd.DataFrame,
    ext_bona_scores: pd.DataFrame,
    label_col: str = "label",
    score_col: str = "score",
    bpcer_target: float = 0.01,
) -> MixedPoolResult:
    """val_scores: FREUID val with both classes (label 0/1). ext_bona_scores:
    external bona ONLY (label ignored, treated as bona=0)."""
    va = val_scores
    attacks = va[va[label_col] == 1][score_col].to_numpy(dtype=float)
    val_bona = va[va[label_col] == 0][score_col].to_numpy(dtype=float)
    ext_bona = ext_bona_scores[score_col].to_numpy(dtype=float)
    if len(attacks) == 0 or len(val_bona) == 0:
        raise ValueError("val_scores must contain both attacks and bona")

    # baseline (FREUID val only)
    base_y = np.r_[np.ones(len(attacks)), np.zeros(len(val_bona))].astype(int)
    base_s = np.r_[attacks, val_bona]
    base = freuid_eval(base_y, base_s, bpcer_target)

    # mixed pool: attacks + (val bona ∪ ext bona)
    mixed_bona = np.r_[val_bona, ext_bona]
    mix_y = np.r_[np.ones(len(attacks)), np.zeros(len(mixed_bona))].astype(int)
    mix_s = np.r_[attacks, mixed_bona]
    mixed = freuid_eval(mix_y, mix_s, bpcer_target)

    thr_val = _threshold_at_bpcer(val_bona, bpcer_target)
    thr_mixed = _threshold_at_bpcer(mixed_bona, bpcer_target)
    ext_fa = float(np.mean(ext_bona >= thr_val)) if len(ext_bona) else float("nan")

    return MixedPoolResult(
        base_freuid=base.score, base_apcer1=base.apcer_at_bpcer, base_audet=base.audet,
        mixed_freuid=mixed.score, mixed_apcer1=mixed.apcer_at_bpcer, mixed_audet=mixed.audet,
        apcer_inflation=mixed.apcer_at_bpcer - base.apcer_at_bpcer,
        thr_val=thr_val, thr_mixed=thr_mixed, thr_shift=thr_mixed - thr_val,
        ext_bona_false_alarm=ext_fa,
        n_attack=len(attacks), n_val_bona=len(val_bona), n_ext_bona=len(ext_bona),
    )


def _selftest() -> None:
    rng = np.random.default_rng(0)
    # well-separated in-dist: attacks high, val bona low
    attacks = rng.uniform(0.6, 1.0, 500)
    val_bona = rng.uniform(0.0, 0.3, 2000)
    val = pd.DataFrame({
        "label": np.r_[np.ones(500), np.zeros(2000)].astype(int),
        "score": np.r_[attacks, val_bona],
    })

    # Case A: external bona behaves like in-dist (low) -> little inflation
    ext_good = pd.DataFrame({"score": rng.uniform(0.0, 0.3, 1500)})
    ra = mixed_pool_report(val, ext_good)
    # Case B: external bona scores HIGH (unseen-domain tail) -> big inflation
    ext_bad = pd.DataFrame({"score": rng.uniform(0.5, 0.95, 1500)})
    rb = mixed_pool_report(val, ext_bad)

    print("Case A (benign external bona):")
    for k, v in asdict(ra).items():
        print(f"  {k:>20} = {v}")
    print("Case B (adversarial external bona, high scores):")
    for k, v in asdict(rb).items():
        print(f"  {k:>20} = {v}")

    assert rb.apcer_inflation >= ra.apcer_inflation, "adversarial bona must inflate APCER more"
    assert rb.ext_bona_false_alarm > ra.ext_bona_false_alarm, "adversarial bona must false-alarm more"
    assert rb.thr_shift >= ra.thr_shift - 1e-9, "adversarial bona must push threshold up"
    print("\nOK — mixed-pool harness self-test passed "
          "(adversarial unseen-domain bona correctly inflates APCER & threshold)")


if __name__ == "__main__":
    _selftest()
