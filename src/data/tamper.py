"""Synthetic localized field-tamper augmentation (FA/FB core aug).

PORT of efs/ml_workspace/kaggle/src/data/tamper.py (verified v4 + g1 recipe).
Faithful copy of the numeric recipe with two deliberate changes:
  1. The g1 switch is an explicit ``g1: bool`` parameter (was a process-global
     env var FREUID_GUINEA_AUG) — so it is unit-testable and config-driven.
     The env var is still honored as the default when g1 is left unspecified.
  2. The unused ``seamless`` branch (copy_move_seamless / splice_seamless,
     an abandoned experiment) is dropped — FA/FB use only _MODES / _MODES_G1.

g1 = the ONLY verified cross-domain lever (E0.2 official re-score: 3 held-out
countries each 2-3x better, matched-mean 0.0425 -> 0.0182). It (a) weights
char_edit 3x and (b) applies LOCALIZED JPEG recompress + tone/gamma jitter to
the char_edit patch ONLY. The photometric jitter is confined to the tampered
patch, so it is NOT a global signal a model could shortcut on (no label leak) —
this locality, not any symmetric bona-side application, is what makes it safe.

test error analysis (2026-06-12): synthesize "subtle single-field text tamper"
attacks onto bona images and train with the attack label -> learn sensitivity
to localized patch inconsistency. Applied at native resolution.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# v2 (verified) + v4 additions. char_edit is weighted 2x (most common ID forgery type).
_MODES = [
    "copy_move", "blur", "bg_fill",          # v2 (verified)
    "char_edit", "char_edit",                # v4: character-level edit (e.g. change one digit of a date) — most common
    "strikethrough",                          # v4: strikethrough
    "print_mismatch",                         # v4: reprint texture overlay
]

# g1: char_edit 3x (verified on cross-domain probe). Applies JPEG+tone locally to the char_edit patch.
_MODES_G1 = [
    "copy_move", "blur", "bg_fill",
    "char_edit", "char_edit", "char_edit",
    "strikethrough", "print_mismatch",
]

# env fallback: referenced only when no explicit g1 argument is given (for reproducing old runs). config passes g1 explicitly.
_ENV_G1 = os.environ.get("FREUID_GUINEA_AUG") == "1"


def _encode_jpeg(patch: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", patch.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    if not ok:
        return patch
    return cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)


def apply_field_tamper(
    img: np.ndarray,
    rng: np.random.Generator,
    g1: bool | None = None,
) -> np.ndarray:
    """Synthesize 1-2 localized tamper patches in the document text area. For generating attack (label=1).

    Modes (mimicking artifacts observed in test):
      copy_move      — copy an adjacent region (background patch inconsistency)
      blur           — blur the patch (inpainting over-smoothing)
      bg_fill        — fill with border median color (erasing a field)
      char_edit      — overwrite with an adjacent same-row char patch at char size (single character/digit change)
      strikethrough  — thin line crossing the field (strikethrough)
      print_mismatch — reprint texture inconsistency of the patch (low JPEG/sharpen + tone shift)

    If g1=True, applies char_edit 3x + localized JPEG (QF30-92, p0.7) + tone/gamma
    jitter (x0.85~1.15, +/-12) to the char_edit patch. If None, decided by env FREUID_GUINEA_AUG.
    """
    use_g1 = _ENV_G1 if g1 is None else bool(g1)
    h, w = img.shape[:2]
    out = img.copy()
    modes = _MODES_G1 if use_g1 else _MODES
    for _ in range(int(rng.integers(1, 3))):
        mode = str(rng.choice(modes))
        # char_edit is one-character size (small, aligned to horizontally long text rows)
        if mode == "char_edit":
            rw = int(w * rng.uniform(0.02, 0.05))
            rh = int(h * rng.uniform(0.025, 0.06))
        else:
            rw = int(w * rng.uniform(0.08, 0.25))
            rh = int(h * rng.uniform(0.03, 0.08))
        if rw < 6 or rh < 5:
            continue
        x0 = int(np.clip(int(w * rng.uniform(0.22, 0.85) - rw / 2), 0, w - rw))
        y0 = int(np.clip(int(h * rng.uniform(0.18, 0.88) - rh / 2), 0, h - rh))
        patch = out[y0 : y0 + rh, x0 : x0 + rw].astype(np.float32)

        if mode == "copy_move":
            dx = int(rng.uniform(-0.06, 0.06) * w)
            dy = int(rng.uniform(-0.04, 0.04) * h)
            sx = int(np.clip(x0 + dx, 0, w - rw))
            sy = int(np.clip(y0 + dy, 0, h - rh))
            new = out[sy : sy + rh, sx : sx + rw].astype(np.float32)
        elif mode == "char_edit":
            # take a patch 1~3 char-widths to the side within the same text row (dy≈0) and overwrite → replace with a different character
            dx = int(np.sign(rng.uniform(-1, 1)) * rng.uniform(1.2, 3.5) * rw)
            dy = int(rng.uniform(-0.3, 0.3) * rh)
            sx = int(np.clip(x0 + dx, 0, w - rw))
            sy = int(np.clip(y0 + dy, 0, h - rh))
            new = out[sy : sy + rh, sx : sx + rw].astype(np.float32)
            if use_g1:
                # g1: apply localized JPEG recompress + tone/gamma jitter to the synthetic char patch → block
                # synthesis artifacts (edge sharpness/pixel-identical copy) as a shortcut, forcing a semantic tamper signal.
                if rng.random() < 0.7:
                    new = _encode_jpeg(new, int(rng.integers(30, 92)))
                new = np.clip(new * float(rng.uniform(0.85, 1.15))
                              + float(rng.uniform(-12, 12)), 0, 255)
        elif mode == "blur":
            new = cv2.GaussianBlur(patch, (0, 0), float(rng.uniform(1.5, 4.0)))
        elif mode == "strikethrough":
            new = patch.copy()
            ink = float(rng.uniform(20, 80))  # dark pen/strikethrough
            color = np.array([ink, ink, ink], np.float32) + rng.normal(0, 8, 3).astype(np.float32)
            thick = max(1, int(round(rh * rng.uniform(0.06, 0.16))))
            yc = rng.uniform(0.35, 0.65)
            slope = float(rng.uniform(-0.10, 0.10))
            for xi in range(rw):
                yy = int(np.clip(rh * yc + slope * (xi - rw / 2), 0, rh - 1))
                new[max(0, yy - thick) : min(rh, yy + thick + 1), xi] = color
        elif mode == "print_mismatch":
            # reprint texture inconsistency: low JPEG or sharpen/blur + tone shift
            if rng.random() < 0.5:
                new = _encode_jpeg(patch, rng.integers(18, 45))
            else:
                k = float(rng.uniform(0.6, 1.6))
                blurred = cv2.GaussianBlur(patch, (0, 0), 1.0)
                new = np.clip(patch + k * (patch - blurred), 0, 255)  # unsharp
            new = np.clip(new * float(rng.uniform(0.90, 1.10)) + float(rng.uniform(-10, 10)), 0, 255)
        else:  # bg_fill
            border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
            color = np.median(border, axis=0)
            new = np.full_like(patch, color) + rng.normal(0, 2.0, patch.shape).astype(np.float32)

        alpha = float(rng.uniform(0.75, 1.0))
        blended = alpha * new + (1 - alpha) * patch
        # edge feather (1~3px) — prevents a too-easy hard edge
        f = int(rng.integers(1, 4))
        mask = np.zeros((rh, rw), np.float32)
        if rh > 2 * f and rw > 2 * f:
            mask[f : rh - f, f : rw - f] = 1.0
        else:
            mask[:] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), max(f, 1))[..., None]
        out[y0 : y0 + rh, x0 : x0 + rw] = np.clip(
            mask * blended + (1 - mask) * patch, 0, 255
        ).astype(np.uint8)
    return out
