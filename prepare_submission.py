"""FREUID 2026 reproducibility container entrypoint.

Uniform end-to-end pipeline (the method behind the two final picks):
  1. inventory  — scan flat /data dir, PIL header resolution scan,
                  captured detection = resolution-frequency (native >= 0.5%).
  2. infer      — frozen-backbone + merged-LoRA members, bf16:
                    main/plain — FB4 (DINOv2-L) + FC3 (OpenCLIP ViT-L/336) +
                                 FD3 (SigLIP2-L/16-384), hflip-TTA per $TTA.
                    fb5        — FB5 (DINOv2-L, all 5 folds), hflip-TTA default.
  3. combine    — per-backbone fold plain-mean -> cross-backbone equal rank-mean
                  (fb5: single backbone -> FB score order).
  4. captured   — capShift (delta=0.75, sigmoid space, per backbone) +
                  captured-internal ens3 mean-rank reorder (FB + FC + FD):
                    main — FC/FD ranks reuse the core scores (zero extra GPU).
                    fb5  — FC3/FD3 scored on the captured subset only, tiered
                           (TTA/noTTA/off) by the 6h/A100 budget at observed c.
  5. write      — strict total order -> (pos+0.5)/n -> id,label CSV.

Note: the submitted files' public-block rows were frozen mid-competition
(whole-file public metric); this container reproduces the ranking pipeline
that produced them and the private-block ranking that decides the final score.
The captured reorder is ens3 = FB + FC + FD everywhere; earlier ensembles that
added recapture-specialist checkpoints (not part of this release) were retired
2026-07-12 to keep the captured lever within the inference budget.

Env:
  VARIANT=main|fb5|plain — selects which Kaggle final pick to reproduce
    (weights identical; inference-orchestration only, allowed under code freeze):
      main (default) — 3-way core; captured lever = capShift + ens3 reorder.
      fb5            — FB5+TTA core; captured lever = capShift(FB) + ens3 reorder
                       with FC3/FD3 scored on captured rows (budget-tiered).
      plain          — 3-way core, captured lever OFF (diagnostic).
  TTA=0|1 — core hflip-TTA override (default: main/plain 0, fb5 1).

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
# main pick: trimmed to fit the 6h cap (weakest pooled-OOF folds dropped)
FB_MEMBERS = ["ff_b_fold0", "ff_b_fold2", "ff_b_fold3_v2", "ff_b_fold4"]
FC_MEMBERS = ["ff_c_fold0", "ff_c_fold1", "ff_c_fold2"]
FD_MEMBERS = ["ff_d_fold1", "ff_d_fold2", "ff_d_fold3"]
BACKBONES = {"fb": FB_MEMBERS, "fc": FC_MEMBERS, "fd": FD_MEMBERS}
# fb5 pick: full 5-fold FB core (hflip-TTA); FC/FD only touch captured rows
FB5_MEMBERS = ["ff_b_fold0", "ff_b_fold1", "ff_b_fold2", "ff_b_fold3_v2", "ff_b_fold4"]
CAP_DELTA = 0.75          # capShift, sigmoid space (exp06d sweep + LB verified)
NATIVE_FREQ = 0.005       # resolution cluster >= 0.5% of rows -> native (non-captured)
# budget tier (mirrors scripts/private_day.py — decisions must match):
# A10G bench ms/img/ckpt (noTTA; hflip-TTA = 2x); cap normalized to full-test size.
MS_CORE = {"fb": 27.1, "fc": 25.8, "fd": 27.7}
A100_FACTOR_CONSERVATIVE, CAP_H, N_TEST_FULL = 2.0, 6.0, 142_818
PUBLIC_MAIN_RES = {(1585, 1000), (1584, 1000), (1000, 630), (1387, 875)}


def log(m):  # noqa: ANN001
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def active_backbones(variant: str) -> dict:
    return {"fb": FB5_MEMBERS} if variant == "fb5" else dict(BACKBONES)


def n_workers(default: int) -> int:
    """DataLoader workers need shared memory; docker's default /dev/shm is 64MB
    and crashes mid-inference (measured 2026-07-11 smoke). Degrade to 0 workers
    (slow but correct) instead of crashing when the flag was forgotten."""
    import shutil
    shm_gib = shutil.disk_usage("/dev/shm").total / 2**30
    if shm_gib >= 2:
        return default
    log(f"WARNING: /dev/shm is {shm_gib:.2f} GiB — falling back to "
        f"num_workers=0 (much slower). Re-run with --shm-size=8g.")
    return 0


def rank01(v) -> np.ndarray:  # noqa: ANN001
    r = pd.Series(v).rank(method="average").to_numpy()
    return (r - 1) / (len(r) - 1)   # match scripts/private_day.py exactly


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
              if n >= NATIVE_FREQ * len(inv) or s in PUBLIC_MAIN_RES}
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
        loader = DataLoader(ds, batch_size=48, shuffle=False,
                            num_workers=n_workers(16), pin_memory=True)
        model = _load_model(cfg, OUT / name / "best.ckpt", torch.device("cuda"))
        ids, scores = [], []
        t0 = time.time()
        log(f"infer {name} (bf16 {'TTA' if tta else 'noTTA'}, n={len(ds)}) ...")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader:
                x = batch["image"].to("cuda", non_blocking=True)
                p = torch.sigmoid(model(x).float())
                if tta:
                    p = 0.5 * (p + torch.sigmoid(model(torch.flip(x, dims=[-1])).float()))
                # guard: a non-finite score would poison ranking and break the
                # "finite float" contract -> clamp to neutral 0.5 / the bounds.
                p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
                scores.extend(p.cpu().numpy().tolist())
                ids.extend(batch["id"])
        del model
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
    if len(bbs) > 1:
        log(f"combine: n={len(merged)} "
            f"spearman(fb,fc)={merged.fb.corr(merged.fc, method='spearman'):.4f} "
            f"(fb,fd)={merged.fb.corr(merged.fd, method='spearman'):.4f}")
    else:
        log(f"combine: n={len(merged)} (FB 5-fold plain-mean, score order)")
    return merged[["id", "captured", *bbs, "combined"]].copy()


def fb5_extras_tier(c: float) -> str:
    """fb5 captured-lever budget tier at the conservative A100 factor
    (mirrors scripts/private_day.py — decisions must match).
    fb5 core is ALWAYS hflip-TTA, so the core cost is hardcoded at x2."""
    core_ms = len(FB5_MEMBERS) * MS_CORE["fb"] * 2
    headroom = CAP_H - N_TEST_FULL * core_ms / 1000 / 3600 / A100_FACTOR_CONSERVATIVE
    ext_ms = len(FC_MEMBERS) * MS_CORE["fc"] + len(FD_MEMBERS) * MS_CORE["fd"]

    def h(tta):  # noqa: ANN001
        return (N_TEST_FULL * c * ext_ms * (2 if tta else 1)
                / 1000 / 3600 / A100_FACTOR_CONSERVATIVE)

    if h(True) <= headroom:
        return "TTA"
    if h(False) <= headroom:
        return "noTTA"
    return "OFF"


def stage_captured(sc: pd.DataFrame, images_dir: Path, work: Path,
                   variant: str) -> pd.DataFrame:
    bbs = active_backbones(variant)
    cap = sc[sc.captured].copy()
    c = len(cap) / len(sc)
    for b in bbs:
        sc[f"{b}_s"] = sc[b]
    sc["ens3"] = np.nan
    if len(cap) == 0:
        log("no captured rows — captured lever inactive")
        return sc
    m = sc.captured
    for b in bbs:
        sc.loc[m, f"{b}_s"] = logit_shift(sc.loc[m, b].to_numpy(), CAP_DELTA)
    sc["combined"] = sum(rank01(sc[f"{b}_s"]) for b in bbs) / len(bbs)
    log(f"capShift delta={CAP_DELTA} applied to {len(cap)} rows (c={c:.2%})")

    if variant == "fb5":
        tier = fb5_extras_tier(c)
        log(f"fb5 captured FC/FD tier = {tier} "
            f"(c={c:.2%} @x{A100_FACTOR_CONSERVATIVE})")
        if tier == "OFF":
            log("captured FC/FD over budget -> capShift only, no reorder")
            return sc
        cap_csv = work / "captured_infer.csv"
        inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
        inv[inv.captured].rename(columns={"path": "image_path"})[
            ["id", "image_path"]].to_csv(cap_csv, index=False)
        stage_infer(images_dir, work, FC_MEMBERS + FD_MEMBERS,
                    tta=(tier == "TTA"), test_csv=cap_csv, prefix="scores_cap")
        for name in FC_MEMBERS + FD_MEMBERS:
            df = pd.read_csv(work / f"scores_cap_{name}.csv", dtype={"id": str})
            cap = cap.merge(df.rename(columns={"score": name}), on="id", how="inner")
        if len(cap) != int(sc.captured.sum()):  # fallback: capShift only, no ens3
            log("WARNING: captured FC/FD coverage mismatch -> capShift only, no reorder")
            return sc
        cap["fc"] = cap[FC_MEMBERS].mean(axis=1)
        cap["fd"] = cap[FD_MEMBERS].mean(axis=1)
    # ens3 = FB + FC + FD equal mean-rank (exp16).
    # main: fc/fd ranks reuse the 3-way core scores already in sc = zero extra GPU.
    cols = ["fb", "fc", "fd"]
    for col in cols:
        cap[f"rk_{col}"] = cap[col].rank(method="average")
    cap["ens3"] = cap[[f"rk_{col}" for col in cols]].mean(axis=1)
    sc = sc.drop(columns=["ens3"]).merge(cap[["id", "ens3"]], on="id", how="left")
    log("ens3 captured-internal ordering computed")
    return sc


def stage_write(sc: pd.DataFrame, out_csv: Path) -> None:
    # main/fb5 variants carry capShift'd fb_s; plain has neither fb_s nor ens3
    fb_col = "fb_s" if "fb_s" in sc.columns else "fb"
    sc = sc.sort_values(["combined", fb_col, "id"], kind="mergesort").reset_index(drop=True)
    n = len(sc)
    sc["value"] = (np.arange(n) + 0.5) / n
    if "ens3" in sc.columns and sc.ens3.notna().any():
        m = sc.ens3.notna()
        vals = np.sort(sc.loc[m, "value"].to_numpy())
        order = sc.loc[m].sort_values(["ens3", "id"], kind="mergesort").index
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
    variant = os.environ.get("VARIANT", "main")
    assert variant in ("main", "fb5", "plain"), f"bad VARIANT={variant}"
    tta_env = os.environ.get("TTA", "")
    tta = (variant == "fb5") if tta_env == "" else tta_env == "1"
    log(f"VARIANT={variant} TTA={int(tta)}")
    bbs = active_backbones(variant)

    inv = stage_inventory(images_dir, work)
    inv.rename(columns={"path": "image_path"})[["id", "image_path"]].to_csv(
        work / "test_infer.csv", index=False)
    stage_infer(images_dir, work, [m for mem in bbs.values() for m in mem], tta)
    sc = stage_combine(work, bbs)
    if variant == "plain":
        log("plain variant — captured lever skipped (raw 3-way rank-mean)")
    else:
        sc = stage_captured(sc, images_dir, work, variant)  # capShift + ens3
    stage_write(sc, Path(args.out))
    log("done")


if __name__ == "__main__":
    main()
