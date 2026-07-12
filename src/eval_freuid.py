"""FREUID official-metric evaluation wrapper — single source of truth for scoring.

The scorer is the EXACT code Kaggle runs: ``kaggle/metric/metric.py`` (official,
published 2026-07-07). We *import* its internal helpers rather than copy them, so
every offline number is byte-for-byte consistent with the leaderboard (03_ops.md:
"official metric.py imported not copied").

DEPRECATED harnesses (do NOT use for gates):
  - fable/src/metrics.py        — APCER@BPCER via np.quantile(bona, .99)
  - src/metrics/freuid.py       — APCER@BPCER via np.interp on DET curve
Both differ from the official step-function APCER@BPCER and mis-rank near the 1%
operating point (the winning axis). Everything here routes through the official
module only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# numpy<2.0 back-compat: official metric.py calls np.trapezoid (added in numpy 2.0,
# present on Kaggle). np.trapz is the identical pre-2.0 name — shim it so offline
# scoring works on this box (numpy 1.26) without touching the official metric.py.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz  # type: ignore[attr-defined]

# --- import the official scorer module ------------------------------------
# kaggle/final/src/eval_freuid.py -> parents[2] == kaggle/ ; metric/ lives there.
_METRIC_DIR = Path(__file__).resolve().parents[2] / "metric"
if str(_METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(_METRIC_DIR))
import metric as _official  # noqa: E402  (official kaggle/metric/metric.py)

# Re-export the official Kaggle entry point verbatim.
score = _official.score
ParticipantVisibleError = _official.ParticipantVisibleError
DEFAULT_BPCER_TARGET = _official.DEFAULT_BPCER_TARGET


@dataclass(frozen=True)
class FreuidResult:
    """Combined FREUID score plus its two components (lower is better)."""

    score: float
    audet: float
    apcer_at_bpcer: float
    n: int
    n_pos: int
    n_neg: int


def freuid_eval(y_true, y_score, bpcer_target: float = DEFAULT_BPCER_TARGET) -> FreuidResult:
    """Score arrays via the official internals, returning the components too.

    Uses ``metric._det_curve`` / ``_audet_from_curve`` / ``_apcer_at_bpcer_from_curve``
    / ``_combine_det_f1`` — the same functions the public ``score()`` calls — so the
    combined value equals ``score()`` exactly while also exposing AuDET and
    APCER@BPCER for analysis (component breakdown that ``score()`` hides).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    bpcer, apcer = _official._det_curve(y_true, y_score)
    audet = _official._audet_from_curve(bpcer, apcer)
    ap = _official._apcer_at_bpcer_from_curve(bpcer, apcer, bpcer_target)
    combined = _official._combine_det_f1(audet, ap)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return FreuidResult(
        score=float(combined),
        audet=float(audet),
        apcer_at_bpcer=float(ap),
        n=int(len(y_true)),
        n_pos=n_pos,
        n_neg=n_neg,
    )


def score_arrays(y_true, y_score, bpcer_target: float = DEFAULT_BPCER_TARGET) -> float:
    """Combined score via the official *public* DataFrame entry point.

    Builds (id, label) / (id, score) frames and calls ``metric.score`` — the true
    Kaggle code path (validation included). Used in tests to prove ``freuid_eval``
    matches ``score()`` on the same inputs.
    """
    n = len(y_true)
    sol = pd.DataFrame({"id": range(n), "label": np.asarray(y_true).astype(int)})
    sub = pd.DataFrame({"id": range(n), "score": np.asarray(y_score, dtype=float)})
    return float(_official.score(sol, sub, "id", bpcer_target=bpcer_target))
