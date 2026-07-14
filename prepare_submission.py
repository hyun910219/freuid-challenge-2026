"""FREUID 2026 reproducibility container entrypoint (two final picks, one image).

End-to-end pipeline:
  1. inventory  — scan flat /data dir, PIL header resolution scan,
                  captured detection = resolution-size (native >= 0.5% & >=1000px).
  2. infer      — frozen-backbone + merged-LoRA members, bf16:
                    core — FB5 (DINOv2-L, all 5 folds), hflip-TTA.
  3. combine    — per-fold plain-mean -> FB score order (single backbone).
  4. captured   — capShift (delta=0.75, sigmoid space) + captured-internal
                  mean-rank reorder over the VARIANT's backbones, with the extra
                  backbones scored on the captured subset only, tiered
                  (TTA/noTTA/off) by the 6h/A100 budget at the observed c.
  5. write      — strict total order -> (pos+0.5)/n -> id,label CSV.

Two documented final picks from THIS frozen commit + THESE frozen weights
(host-approved: inference-time flag only, no training/checkpoint change):
  VARIANT=ens3 (default)  captured reorder = FB5 + FC3 + FD3   -> Pick 1
  VARIANT=fd              captured reorder = FB5 + FD3          -> Pick 2
The two picks are byte-identical everywhere except the captured rows' internal
order (FC included vs excluded from the captured mean-rank).

Note: the submitted file's public-block rows were frozen mid-competition
(whole-file public metric); this container reproduces the ranking pipeline
that produced them and the private-block ranking that decides the final score.

Runtime: pass --shm-size=16g (or --ipc=host). Worker->main tensor IPC uses
/dev/shm even with the file_system sharing strategy on torch 2.12; the 64MB
docker default is exhausted otherwise (a num_workers=0 fallback keeps a no-flag
run alive but much slower — the documented commands include the flag).

Entrypoint sits at the repo/image root (/app). Weights are baked in and
resolved via /app/outputs -> /weights/final.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent            # /app (repo root)
OUT = APP / "outputs"                            # -> /weights/final (symlink)
IMG_EXT = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
# fb5 pick: full 5-fold FB core (hflip-TTA); FC3/FD3 only touch captured rows
FB5_MEMBERS = ["ff_b_fold0", "ff_b_fold1", "ff_b_fold2", "ff_b_fold3_v2", "ff_b_fold4"]
FC_MEMBERS = ["ff_c_fold0", "ff_c_fold1", "ff_c_fold2"]
FD_MEMBERS = ["ff_d_fold1", "ff_d_fold2", "ff_d_fold3"]
CORE_BB = {"fb": FB5_MEMBERS}   # single-backbone core (FB5); FC/FD = captured lever
EXT_MEMBERS = {"fc": FC_MEMBERS, "fd": FD_MEMBERS}
# VARIANT selects which backbones enter the captured-internal reorder (see module
# docstring). Same image, same weights — inference-time flag only.
CAP_ENS = {"ens3": ["fb", "fc", "fd"], "fd": ["fb", "fd"]}
VARIANT = os.environ.get("VARIANT", "ens3").strip().lower()
assert VARIANT in CAP_ENS, f"unknown VARIANT={VARIANT!r} (use one of {list(CAP_ENS)})"
CAP_DELTA = 0.75          # capShift, sigmoid space (exp06d sweep + LB verified)
NATIVE_FREQ = 0.005       # resolution cluster >= 0.5% of rows -> native (non-captured)
MIN_NATIVE_W = 1000       # digital acquisitions are always >=1000px wide; a smaller
#                           cluster (e.g. 840x530) is recapture/downscale -> captured,
#                           even when frequent. Matches the LB-adopted size-based split
#                           (public 158 captured, LB 0.01238) vs freq-only (37, 0.01679).
# budget tier (mirrors scripts/private_day.py — decisions must match):
# A10G bench ms/img/ckpt (noTTA; hflip-TTA = 2x); cap normalized to full-test size.
MS_CORE = {"fb": 27.1, "fc": 25.8, "fd": 27.7}
A100_FACTOR_CONSERVATIVE, CAP_H, N_TEST_FULL = 2.0, 6.0, 142_818
PUBLIC_MAIN_RES = {(1585, 1000), (1584, 1000), (1000, 630), (1387, 875)}
BATCH = 48
# shm-exhaustion error substrings (torch surfaces the worker collate failure)
_SHM_ERR = ("shared memory", "No space left", "/dev/shm", "bus error")


def log(m):  # noqa: ANN001
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def n_workers(default: int) -> int:
    """Worker count. Worker->main tensor IPC uses /dev/shm even with the
    file_system sharing strategy on torch 2.12, so the container must be run
    with --shm-size=16g (or --ipc=host). stage_infer falls back to num_workers=0
    if /dev/shm is exhausted, so a no-flag run still produces output (slower)."""
    return default


def rank01(v) -> np.ndarray:  # noqa: ANN001
    r = pd.Series(v).rank(method="average").to_numpy()
    n = len(r)
    if n <= 1:                       # single-row edge (e.g. 1-image smoke test): avoid /0
        return np.zeros(n, dtype=float)
    return (r - 1) / (n - 1)         # match scripts/private_day.py (identical for n>1)


def logit_shift(p: np.ndarray, delta: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) - delta)))


def _imsize(path: str):  # noqa: ANN001
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:  # noqa: BLE001
        return None


def stage_inventory(images_dir: Path, work: Path) -> pd.DataFrame:
    files = sorted(p for p in images_dir.iterdir()
                   if p.suffix.lower() in IMG_EXT)
    assert files, f"no images under {images_dir}"
    # id = filename stem, must be unique. If two files share a stem (e.g.
    # a.jpg + a.png), keep the first and warn -> guarantees unique ids and
    # avoids a cross-product blow-up in stage_combine (graceful fallback).
    seen, uniq = set(), []
    for p in files:
        if p.stem not in seen:
            seen.add(p.stem)
            uniq.append(p)
    if len(uniq) != len(files):
        log(f"WARNING: {len(files) - len(uniq)} duplicate-stem file(s) dropped "
            f"(id must be unique); kept first per stem")
    files = uniq
    log(f"inventory: {len(files)} images")
    with ProcessPoolExecutor(max_workers=16) as ex:
        sizes = list(ex.map(_imsize, (str(p) for p in files), chunksize=256))
    inv = pd.DataFrame({"id": [p.stem for p in files],
                        "path": [p.name for p in files],
                        "size": sizes})
    bad = int(inv["size"].isna().sum())
    if bad:  # contract: one row per image — keep them (datasets decode on a
        #      neutral canvas); size=NaN never joins a native cluster -> captured
        log(f"WARNING: {bad} unreadable image(s) — kept, treated as captured")
    freq = inv["size"].value_counts()
    native = {s for s, n in freq.items()
              if s[0] >= MIN_NATIVE_W and (n >= NATIVE_FREQ * len(inv) or s in PUBLIC_MAIN_RES)}
    inv["captured"] = ~inv["size"].isin(native)
    c = inv.captured.mean()
    log(f"resolution clusters: {len(freq)}, captured fraction c = {c:.3%}")
    inv.drop(columns=["size"]).to_csv(work / "inventory.csv", index=False)
    return inv


def stage_infer(images_dir: Path, work: Path, members: list, tta: bool,
                test_csv: Path | None = None, prefix: str = "scores") -> None:
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from src.data import FreuidDataset, build_transforms
    from src.infer import _load_model

    # file_system sharing strategy (named shm objects). Worker->main IPC still
    # uses /dev/shm on torch 2.12, so the container needs --shm-size=16g /
    # --ipc=host; the num_workers=0 fallback below degrades gracefully if missing.
    torch.multiprocessing.set_sharing_strategy("file_system")
    # Auto-detect device: use the GPU when exposed (docker --gpus), else fall
    # back to CPU (correct, far slower) instead of hard-crashing on "cuda".
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_csv = test_csv or work / "test_infer.csv"
    for name in members:
        out_csv = work / f"{prefix}_{name}.csv"
        done = work / f"{prefix}_{name}.DONE"
        if done.exists():
            log(f"skip {name} (.DONE)")
            continue
        cfg = OmegaConf.load(APP / "configs" / f"{name}.yaml")
        isz = tuple(cfg.data.image_size)
        ds = FreuidDataset(
            test_csv, str(images_dir),
            transforms=build_transforms(
                "test", image_size=isz,
                resize_strategy=cfg.train.get("resize_strategy", "aspect_resize"),
                mean=cfg.data.get("norm_mean", None),
                std=cfg.data.get("norm_std", None)),
            mode="test",
        )
        model = _load_model(cfg, OUT / name / "best.ckpt", torch.device(device))
        # torch.compile: inference-orchestration speedup (~+17% on A10G, rank-preserving;
        # raw scores shift ~1e-2 but the pipeline is rank-based end-to-end). Freeze-legal:
        # no weight/fold/resolution change. Runs offline (the Dockerfile installs g++ for
        # the Inductor backend). torch.compile is LAZY (compiles at first forward), so
        # suppress_errors degrades a first-batch compile failure to eager; the try/except
        # only covers an unavailable backend at call time.
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        try:
            model = torch.compile(model)
        except Exception as e:  # noqa: BLE001
            log(f"WARNING: torch.compile unavailable, eager fallback ({type(e).__name__}: {e})")

        def _run(nw):  # noqa: ANN001,ANN202
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                                num_workers=nw, pin_memory=True)
            ids, scores = [], []
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                for batch in loader:
                    x = batch["image"].to(device, non_blocking=True)
                    n = x.shape[0]
                    if n < BATCH:  # pad final partial batch -> single compile per model
                        x = torch.cat([x, x[-1:].expand(BATCH - n, *x.shape[1:])], dim=0)
                    p = torch.sigmoid(model(x).float())[:n]
                    if tta:
                        p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[-1])).float())[:n])
                    # guard: a non-finite score would poison ranking and break the
                    # "finite float" contract -> clamp to neutral 0.5 / the bounds.
                    p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
                    scores.extend(p.cpu().numpy().tolist())
                    ids.extend(batch["id"])
            return ids, scores

        t0 = time.time()
        log(f"infer {name} (bf16 {'TTA' if tta else 'noTTA'}, n={len(ds)}, compiled) ...")
        try:
            ids, scores = _run(n_workers(16))
        except RuntimeError as e:  # noqa: BLE001
            if any(k in str(e) for k in _SHM_ERR):
                log("WARNING: /dev/shm exhausted — run with --shm-size=16g (or "
                    "--ipc=host). Retrying this member with num_workers=0 (slower).")
                if device == "cuda":
                    torch.cuda.empty_cache()
                ids, scores = _run(0)
            else:
                raise
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame({"id": ids, "score": scores}).to_csv(
            out_csv, index=False, float_format="%.10f")
        done.touch()
        log(f"    done {name} in {(time.time() - t0) / 60:.1f} min")


def stage_combine(work: Path, bbs: dict) -> pd.DataFrame:
    inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
    merged = inv[["id", "captured"]].copy()
    members = [m for mem in bbs.values() for m in mem]
    for name in members:
        df = pd.read_csv(work / f"scores_{name}.csv", dtype={"id": str})
        # left join keeps exactly one row per inventory id (no drops/blow-up)
        merged = merged.merge(df.rename(columns={"score": name}), on="id", how="left")
    assert len(merged) == len(inv), "row count changed after left-merge"
    miss = int(merged[members].isna().any(axis=1).sum())
    if miss:  # fallback: fill a missing member score with that member's mean
        log(f"WARNING: {miss} row(s) missing a member score — filled with column mean")
        for name in members:
            merged[name] = merged[name].fillna(merged[name].mean())
    for b, mem in bbs.items():
        merged[b] = merged[mem].mean(axis=1)   # fold-agg = plain mean
    # cross-backbone = equal rank-mean (single backbone -> its own score order)
    merged["combined"] = sum(rank01(merged[b]) for b in bbs) / len(bbs)
    log(f"combine: n={len(merged)} (FB 5-fold plain-mean, score order)")
    return merged[["id", "captured", *bbs, "combined"]].copy()


def fb5_extras_tier(c: float, ext_ms: float) -> str:
    """Captured-lever budget tier at the conservative A100 factor (mirrors
    scripts/private_day.py). fb5 core is ALWAYS hflip-TTA (cost hardcoded at x2);
    ext_ms = per-image ms of the VARIANT's captured extra backbones (noTTA)."""
    core_ms = len(FB5_MEMBERS) * MS_CORE["fb"] * 2
    headroom = CAP_H - N_TEST_FULL * core_ms / 1000 / 3600 / A100_FACTOR_CONSERVATIVE

    def h(tta):  # noqa: ANN001
        return (N_TEST_FULL * c * ext_ms * (2 if tta else 1)
                / 1000 / 3600 / A100_FACTOR_CONSERVATIVE)

    if h(True) <= headroom:
        return "TTA"
    if h(False) <= headroom:
        return "noTTA"
    return "OFF"


def stage_captured(sc: pd.DataFrame, images_dir: Path, work: Path,
                   bbs: dict) -> pd.DataFrame:
    cap = sc[sc.captured].copy()
    c = len(cap) / len(sc)
    for b in bbs:
        sc[f"{b}_s"] = sc[b]
    sc["cap_ens"] = np.nan
    if len(cap) == 0:
        log("no captured rows — captured lever inactive")
        return sc
    m = sc.captured
    for b in bbs:
        sc.loc[m, f"{b}_s"] = logit_shift(sc.loc[m, b].to_numpy(), CAP_DELTA)
    sc["combined"] = sum(rank01(sc[f"{b}_s"]) for b in bbs) / len(bbs)
    log(f"capShift delta={CAP_DELTA} applied to {len(cap)} rows (c={c:.2%})")

    # captured-internal reorder members for this VARIANT: 'fb' is the core score
    # (already in sc); the extras (fc and/or fd) are scored on captured rows only.
    ens_cols = CAP_ENS[VARIANT]
    extra = [b for b in ens_cols if b != "fb"]
    members = [mm for b in extra for mm in EXT_MEMBERS[b]]
    ext_ms = sum(len(EXT_MEMBERS[b]) * MS_CORE[b] for b in extra)
    tier = fb5_extras_tier(c, ext_ms)
    log(f"captured reorder = {'+'.join(ens_cols).upper()} (VARIANT={VARIANT}); "
        f"extras {'+'.join(extra).upper()} tier={tier} (c={c:.2%} @x{A100_FACTOR_CONSERVATIVE})")
    if tier == "OFF":
        log("captured extras over budget -> capShift only, no reorder")
        return sc
    cap_csv = work / "captured_infer.csv"
    inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
    inv[inv.captured].rename(columns={"path": "image_path"})[
        ["id", "image_path"]].to_csv(cap_csv, index=False)
    stage_infer(images_dir, work, members, tta=(tier == "TTA"),
                test_csv=cap_csv, prefix="scores_cap")
    for name in members:
        df = pd.read_csv(work / f"scores_cap_{name}.csv", dtype={"id": str})
        cap = cap.merge(df.rename(columns={"score": name}), on="id", how="inner")
    if len(cap) != int(sc.captured.sum()):  # fallback: capShift only, no reorder
        log("WARNING: captured extras coverage mismatch -> capShift only, no reorder")
        return sc
    for b in extra:
        cap[b] = cap[EXT_MEMBERS[b]].mean(axis=1)
    # equal mean-rank over the VARIANT's backbones, captured subset only (exp16).
    for col in ens_cols:
        cap[f"rk_{col}"] = cap[col].rank(method="average")
    cap["cap_ens"] = cap[[f"rk_{col}" for col in ens_cols]].mean(axis=1)
    sc = sc.drop(columns=["cap_ens"]).merge(cap[["id", "cap_ens"]], on="id", how="left")
    log(f"captured-internal ordering computed ({'+'.join(ens_cols).upper()})")
    return sc


def stage_write(sc: pd.DataFrame, out_csv: Path) -> None:
    # capShift'd fb_s is the tiebreak; cap_ens (if present) reorders captured rows
    fb_col = "fb_s" if "fb_s" in sc.columns else "fb"
    sc = sc.sort_values(["combined", fb_col, "id"], kind="mergesort").reset_index(drop=True)
    n = len(sc)
    sc["value"] = (np.arange(n) + 0.5) / n
    if "cap_ens" in sc.columns and sc.cap_ens.notna().any():
        m = sc.cap_ens.notna()
        vals = np.sort(sc.loc[m, "value"].to_numpy())
        order = sc.loc[m].sort_values(["cap_ens", "id"], kind="mergesort").index
        sc.loc[order, "value"] = vals
        log(f"captured-internal reorder applied to {int(m.sum())} rows")
    vals_fmt = ["%.10f" % v for v in sc.value]
    if len(set(vals_fmt)) != n:  # metric is rank-only; near-ties are harmless
        log(f"WARNING: {n - len(set(vals_fmt))} formatted-value collision(s) "
            f"(rank-only metric; writing anyway)")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write("id,label\n")
        for rid, v in zip(sc.id, vals_fmt):
            f.write(f"{rid},{v}\n")
    log(f"wrote {out_csv} (n={n})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="/data")
    ap.add_argument("--out", default="/submissions/submission.csv")
    # under /submissions to honour the sandbox "no writes outside /submissions/" rule
    ap.add_argument("--work", default="/submissions/_work")
    args = ap.parse_args()
    images_dir, work = Path(args.images_dir), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    bbs = CORE_BB
    tta = True                       # fb5 core is always hflip-TTA
    log(f"pick=fb5 VARIANT={VARIANT} (FB5 core hflip-TTA + capShift + "
        f"{'+'.join(CAP_ENS[VARIANT]).upper()} captured reorder)")

    inv = stage_inventory(images_dir, work)
    inv.rename(columns={"path": "image_path"})[["id", "image_path"]].to_csv(
        work / "test_infer.csv", index=False)
    stage_infer(images_dir, work, [m for mem in bbs.values() for m in mem], tta)
    sc = stage_combine(work, bbs)
    sc = stage_captured(sc, images_dir, work, bbs)   # capShift + captured reorder
    stage_write(sc, Path(args.out))
    log("done")


if __name__ == "__main__":
    main()
