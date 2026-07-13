"""PRIVATE DAY one-shot pipeline (FREUID 2026, private test release ~07-13).

Single final pick (fb5):
  base submissions/2026-07-13_ffb5_capReorder_ens3.csv.
  Private rows = FB5 (all 5 folds) hflip-TTA, score order;
  captured lever = capShift(FB) + ens3 reorder with FC3/FD3 scored on the
  captured subset only (budget tier TTA/noTTA/OFF at the CONSERVATIVE A100
  factor — mirrors prepare_submission.py).

Frozen decisions encoded here:
  - The captured-internal reorder is ens3 = FB5 + FC3 + FD3 (exp16).
    Earlier ensembles that added recapture-specialist checkpoints (not part of
    this release) were retired 2026-07-12 to keep the lever within the GPU budget.
  - Public rows byte-identical, row order preserved (whole-file-metric bug mitigation).
  - RAW scores only — NO per-type/domain normalization (p2_norm_sim: all negative).
  - fold-agg = plain mean (rank/median/trimmed probes all negative);
    cross-backbone = equal RANK-MEAN (single backbone -> FB score order).
  - capShift delta=0.75 sigmoid space (exp06d sweep, LB 0.01870 -> 0.01524).
  - Inference budget: 6h/A100 cap. bench (A10G bs48 bf16 noTTA): FB 27.1 /
    FC 25.8 / FD 27.7 ms/img/ckpt; hflip-TTA = 2x. FB5+TTA core 5.38h@2.0.

Stages (resumable; per-member .DONE sentinels):
  inventory -> infer -> combine -> [captured] -> assemble

Run on g5 (GPU box, images local):
  cd /home/ec2-user/workspace/efs/ml_workspace/kaggle/final && \
  PYTHONPATH=. PYTHONUNBUFFERED=1 /home/ec2-user/extenv/bin/python -u \
    scripts/private_day.py --stage inventory --images-dir <PRIVATE_IMG_DIR>

Dry-run (mechanics check on public_test, small N, throwaway assemble):
  ... scripts/private_day.py --stage all --dry-run 240 \
      --images-dir /home/ec2-user/data/freuid/public_test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINAL))

DATA = Path("/home/ec2-user/data/freuid")
OUT = Path("/home/ec2-user/workspace/efs/ml_workspace/kaggle/outputs")
# EFS copies (== ml-workspace originals) — visible from g5
SUB_DIR = Path("/home/ec2-user/workspace/efs/ml_workspace/kaggle/submissions")
BASE_SUB = SUB_DIR / "2026-07-13_ffb5_capReorder_ens3.csv"

# members: ckpt at OUT/<config>/best.ckpt
# fb5 core: full 5-fold FB, hflip-TTA (the composition behind the base file).
# FC3/FD3 (captured lever) score the captured subset only.
FB5_MEMBERS = ["ff_b_fold0", "ff_b_fold1", "ff_b_fold2", "ff_b_fold3_v2", "ff_b_fold4"]
FC_MEMBERS = ["ff_c_fold0", "ff_c_fold1", "ff_c_fold2"]
FD_MEMBERS = ["ff_d_fold1", "ff_d_fold2", "ff_d_fold3"]
CORE_BB = {"fb": FB5_MEMBERS}

# bench constants (outputs/_bench_infer.log + bench_fd_throughput.py, A10G bs48 bf16)
MS_CORE = {"fb": 27.1, "fc": 25.8, "fd": 27.7}   # noTTA ms/img/ckpt; TTA = 2x
A100_FACTOR, A100_FACTOR_CONSERVATIVE, CAP_H = 2.2, 2.0, 6.0
N_TEST_FULL = 142_818
CAP_DELTA = 0.75          # exp06d sweep optimum, LB-verified 0.01870 -> 0.01524
NATIVE_FREQ = 0.005       # resolution seen in >=0.5% of scanned images = native
PUBLIC_MAIN_RES = {(1585, 1000), (1584, 1000), (1000, 630), (1387, 875)}  # cross-check


# ---------------------------------------------------------------- utilities
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def public_ids() -> set[str]:
    return {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(str(DATA / "public_test/public_test/*.jpeg"))}


def all_ids() -> list[str]:
    return pd.read_csv(DATA / "sample_submission.csv")["id"].astype(str).tolist()


def _imsize(path: str) -> tuple[int, int]:
    from PIL import Image
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return (-1, -1)


def rank01(s: pd.Series) -> pd.Series:
    r = s.rank(method="average")
    return (r - 1) / (len(r) - 1)


def logit_shift(p: np.ndarray, delta: float) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    z = np.log(p / (1 - p)) - delta
    return 1.0 / (1.0 + np.exp(-z))


def fb5_extras_tier(c: float) -> str:
    """fb5 captured-lever budget tier at the conservative A100 factor
    (mirrors prepare_submission.py — decisions must match)."""
    core_ms = len(FB5_MEMBERS) * MS_CORE["fb"] * 2   # fb5 core = hflip-TTA
    headroom = CAP_H - N_TEST_FULL * core_ms / 1000 / 3600 / A100_FACTOR_CONSERVATIVE
    ext_ms = len(FC_MEMBERS) * MS_CORE["fc"] + len(FD_MEMBERS) * MS_CORE["fd"]

    def h(tta: bool) -> float:
        return (N_TEST_FULL * c * ext_ms * (2 if tta else 1)
                / 1000 / 3600 / A100_FACTOR_CONSERVATIVE)

    if h(True) <= headroom:
        return "TTA"
    if h(False) <= headroom:
        return "noTTA"
    return "OFF"


# ---------------------------------------------------------------- stage: inventory
def stage_inventory(args, work: Path) -> None:
    ids_all = all_ids()
    pub = public_ids()
    target = sorted(pub) if args.dry_run else [i for i in ids_all if i not in pub]
    if args.dry_run:
        target = target[:: max(1, len(target) // args.dry_run)][: args.dry_run]
    log(f"target rows: {len(target)} ({'DRY public subset' if args.dry_run else 'private'})")

    files = {}
    for ext in ("jpeg", "jpg", "png"):
        for p in glob.glob(f"{args.images_dir}/**/*.{ext}", recursive=True):
            files[os.path.splitext(os.path.basename(p))[0]] = p
    missing = [i for i in target if i not in files]
    log(f"images found for {len(target) - len(missing)}/{len(target)} target ids "
        f"(missing {len(missing)})")
    if missing and not args.dry_run:
        (work / "missing_ids.txt").write_text("\n".join(missing))
        log(f"WARNING: missing ids -> {work/'missing_ids.txt'} — investigate before infer")
    target = [i for i in target if i in files]

    with ProcessPoolExecutor(max_workers=16) as ex:
        sizes = list(ex.map(_imsize, [files[i] for i in target], chunksize=256))
    inv = pd.DataFrame({"id": target,
                        "path": [os.path.relpath(files[i], args.images_dir) for i in target],
                        "w": [s[0] for s in sizes], "h": [s[1] for s in sizes]})
    freq = inv.groupby(["w", "h"]).size().sort_values(ascending=False)
    native = {wh for wh, n in freq.items() if n >= NATIVE_FREQ * len(inv)}
    inv["captured"] = [wh not in native for wh in zip(inv.w, inv.h)]
    c = inv.captured.mean()
    inv.to_csv(work / "inventory.csv", index=False)
    freq.rename("n").reset_index().to_csv(work / "resolutions.csv", index=False)

    log(f"resolution clusters: {len(freq)} distinct, native={len(native)}")
    for (w, h), n in list(freq.items())[:15]:
        tag = "native" if (w, h) in native else "CAPTURED?"
        star = " (=public main)" if (w, h) in PUBLIC_MAIN_RES else ""
        log(f"    {w}x{h}: {n} ({n/len(inv):.2%}) {tag}{star}")
    log(f"captured candidate fraction c = {c:.3%} ({int(inv.captured.sum())} rows)")

    # budget table (docker reproducibility = FULL test through every used member)
    def a100h(n_img, ms, factor):  # noqa: E306
        return n_img * ms / 1000 / 3600 / factor
    bbs = CORE_BB
    core_ms = sum(len(mem) * MS_CORE[b] for b, mem in bbs.items()) * 2   # fb5 core = TTA
    core_h = a100h(N_TEST_FULL, core_ms, A100_FACTOR)
    core_h_cons = a100h(N_TEST_FULL, core_ms, A100_FACTOR_CONSERVATIVE)
    log("budget (A100eq, FULL-test docker run, fb5):")
    log(f"    core {'+'.join(f'{b.upper()}{len(mem)}' for b, mem in bbs.items())} "
        f"TTA = {core_h:.2f}h @x{A100_FACTOR} / "
        f"{core_h_cons:.2f}h @x{A100_FACTOR_CONSERVATIVE} (cap {CAP_H}h)")
    ext_ms = len(FC_MEMBERS) * MS_CORE["fc"] + len(FD_MEMBERS) * MS_CORE["fd"]
    ext_tta = a100h(int(N_TEST_FULL * c), ext_ms * 2, A100_FACTOR_CONSERVATIVE)
    ext_nt = a100h(int(N_TEST_FULL * c), ext_ms, A100_FACTOR_CONSERVATIVE)
    log(f"    captured FC3+FD3 @c={c:.1%} @x{A100_FACTOR_CONSERVATIVE}: "
        f"TTA {ext_tta:.2f}h / noTTA {ext_nt:.2f}h -> tier {fb5_extras_tier(c)} "
        f"(decided again in captured stage; capShift always applies)")
    wall = (core_h + ext_tta) * A100_FACTOR * len(inv) / N_TEST_FULL
    log(f"    today's g5 wall (target rows only): ~{wall:.1f}h")


# ---------------------------------------------------------------- stage: infer
def _predict_bf16(cfg, ckpt, test_csv, image_root, tta: bool = False) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader
    from src.data import FreuidDataset, build_transforms
    from src.infer import _load_model
    device = torch.device("cuda")
    isz = tuple(cfg.data.image_size) if not isinstance(cfg.data.image_size, int) \
        else cfg.data.image_size
    ds = FreuidDataset(
        test_csv, str(image_root),
        transforms=build_transforms("test", image_size=isz,
                                    resize_strategy=cfg.train.get("resize_strategy", "aspect_resize"),
                                    mean=cfg.data.get("norm_mean", None),
                                    std=cfg.data.get("norm_std", None)),
        mode="test",
    )
    loader = DataLoader(ds, batch_size=48, shuffle=False, num_workers=16,
                        pin_memory=True)
    model = _load_model(cfg, ckpt, device)
    # torch.compile: inference-orchestration speedup (~+17% on A10G, rank-preserving;
    # raw scores shift ~1e-2 but the pipeline is rank-based end-to-end). Freeze-legal:
    # no weight/fold/resolution change. Runs offline (gcc suffices; no g++/network).
    model = torch.compile(model)
    ids, scores = [], []
    t0, n_done = time.time(), 0
    bs = loader.batch_size
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            n = imgs.shape[0]
            if n < bs:  # pad final partial batch to fixed bs -> single compile per model
                imgs = torch.cat([imgs, imgs[-1:].expand(bs - n, *imgs.shape[1:])], dim=0)
            p = torch.sigmoid(model(imgs).float())[:n]
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(torch.flip(imgs, dims=[-1])).float())[:n])
            scores.extend(p.cpu().numpy().tolist())
            ids.extend(batch["id"])
            n_done += len(batch["id"])
            if n_done % 9600 < 48:
                log(f"        {n_done}/{len(ds)} ({n_done/(time.time()-t0):.1f} img/s)")
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame({"id": ids, "score": scores})


def stage_infer(args, work: Path) -> None:
    from omegaconf import OmegaConf
    bbs = CORE_BB
    tta = True                         # fb5 core = hflip-TTA (base-file composition)
    inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
    test_csv = work / "test_infer.csv"
    inv.rename(columns={"path": "image_path"})[["id", "image_path"]].to_csv(
        test_csv, index=False)
    for name in [m for mem in bbs.values() for m in mem]:
        out_csv, done = work / f"scores_{name}.csv", work / f"scores_{name}.DONE"
        if done.exists():
            log(f"skip {name} (.DONE)")
            continue
        cfg = OmegaConf.load(FINAL / "configs" / f"{name}.yaml")
        ckpt = OUT / name / "best.ckpt"
        assert ckpt.exists(), f"missing ckpt {ckpt}"
        log(f"infer {name} (bf16 {'TTA' if tta else 'noTTA'} bs48) ...")
        t0 = time.time()
        df = _predict_bf16(cfg, ckpt, test_csv, args.images_dir, tta=tta)
        assert len(df) == len(inv), f"{name}: {len(df)} != {len(inv)}"
        df.to_csv(out_csv, index=False, float_format="%.10f")
        done.touch()
        log(f"    done {name} in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------- stage: combine
def stage_combine(args, work: Path) -> pd.DataFrame:
    bbs = CORE_BB
    inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
    merged = inv[["id", "captured"]].copy()
    for name in [m for mem in bbs.values() for m in mem]:
        df = pd.read_csv(work / f"scores_{name}.csv", dtype={"id": str})
        merged = merged.merge(df.rename(columns={"score": name}), on="id", how="inner")
    assert len(merged) == len(inv), "id mismatch across member dumps"
    for b, mem in bbs.items():
        merged[b] = merged[mem].mean(axis=1)   # fold-agg = plain mean (probes 3/3 negative)
    # cross-backbone = equal rank-mean (single backbone -> its own score order)
    merged["combined"] = sum(rank01(merged[b]) for b in bbs) / len(bbs)
    out = merged[["id", "captured", *bbs, "combined"]]
    out.to_csv(work / "private_scores.csv", index=False, float_format="%.12g")
    if len(bbs) > 1:
        log(f"combine: n={len(out)} "
            f"spearman(fb,fc)={merged.fb.corr(merged.fc, method='spearman'):.4f} "
            f"(fb,fd)={merged.fb.corr(merged.fd, method='spearman'):.4f} "
            f"(fc,fd)={merged.fc.corr(merged.fd, method='spearman'):.4f}")
    else:
        log(f"combine: n={len(out)} (FB 5-fold plain-mean, score order)")
    return out


# ---------------------------------------------------------------- stage: captured
def stage_captured(args, work: Path) -> None:
    """capShift(delta in sigmoid space, per backbone) + captured-internal ens3
    reorder (FB + FC + FD equal mean-rank; exp16).

    Requires human GO (--go-captured) after reviewing inventory resolution table:
    detection on UNSEEN private types is frequency-based and must look sane first.
    """
    if not args.go_captured:
        log("captured stage requires --go-captured (human gate). Skipping.")
        return
    bbs = CORE_BB
    sc = pd.read_csv(work / "private_scores.csv", dtype={"id": str})
    cap = sc[sc.captured].copy()
    c = len(cap) / len(sc)
    log(f"captured rows: {len(cap)} ({c:.2%})")
    if len(cap) == 0:
        log("no captured rows — nothing to do")
        return

    # 1) capShift per backbone in sigmoid space (delta tuned there; monotone in-group)
    #    CPU-side -> always applies regardless of any budget tier below.
    m = sc.captured
    for b in bbs:
        sc[f"{b}_s"] = sc[b]
        sc.loc[m, f"{b}_s"] = logit_shift(sc.loc[m, b].to_numpy(), CAP_DELTA)
    sc["combined"] = sum(rank01(sc[f"{b}_s"]) for b in bbs) / len(bbs)

    # 2) ens3 member scores: FC3/FD3 scored on the captured subset only,
    #    budget-tiered at the CONSERVATIVE A100 factor (exp09: captured order
    #    TTA-robust, sp 0.9958).
    from omegaconf import OmegaConf
    tier = fb5_extras_tier(c)
    log(f"fb5 captured FC/FD tier = {tier} (c={c:.2%} "
        f"@x{A100_FACTOR_CONSERVATIVE})")
    if tier == "OFF":
        sc["ens3"] = np.nan
        sc.to_csv(work / "private_scores_captured.csv", index=False,
                  float_format="%.12g")
        log("captured FC/FD over budget -> capShift only, no reorder")
        return
    cap_csv = work / "captured_infer.csv"
    inv = pd.read_csv(work / "inventory.csv", dtype={"id": str})
    inv[inv.captured].rename(columns={"path": "image_path"})[
        ["id", "image_path"]].to_csv(cap_csv, index=False)
    for name in FC_MEMBERS + FD_MEMBERS:
        out_csv = work / f"scores_cap_{name}.csv"
        done = work / f"scores_cap_{name}.DONE"
        if done.exists():
            log(f"skip cap {name} (.DONE)")
        else:
            cfg = OmegaConf.load(FINAL / "configs" / f"{name}.yaml")
            ckpt = OUT / name / "best.ckpt"
            assert ckpt.exists(), f"missing ckpt {ckpt}"
            log(f"infer cap {name} (bf16 {tier} bs48, n={len(cap)}) ...")
            df = _predict_bf16(cfg, ckpt, cap_csv, args.images_dir,
                               tta=(tier == "TTA"))
            assert len(df) == len(cap), f"{name}: {len(df)} != {len(cap)}"
            df.to_csv(out_csv, index=False, float_format="%.10f")
            done.touch()
        df = pd.read_csv(out_csv, dtype={"id": str})
        cap = cap.merge(df.rename(columns={"score": name}), on="id", how="inner")
    assert len(cap) == sc.captured.sum(), "captured FC/FD coverage mismatch"
    cap["fc"] = cap[FC_MEMBERS].mean(axis=1)
    cap["fd"] = cap[FD_MEMBERS].mean(axis=1)

    # 3) ens3 mean-rank within captured (exp16; the reorder mechanic itself is
    #    public-LB verified: 0.01524 -> 0.01252 as the reorder ensemble grew)
    cols = ["fb", "fc", "fd"]
    for col in cols:
        cap[f"rk_{col}"] = cap[col].rank(method="average")
    cap["ens3"] = cap[[f"rk_{col}" for col in cols]].mean(axis=1)
    sp = cap["rk_fb"].corr(cap["ens3"], method="spearman")
    log(f"spearman(FB order, ens3 order) on captured = {sp:.4f}")
    sc = sc.merge(cap[["id", "ens3"]], on="id", how="left")
    sc.to_csv(work / "private_scores_captured.csv", index=False, float_format="%.12g")
    log(f"captured stage done -> private_scores_captured.csv")


# ---------------------------------------------------------------- stage: assemble
def stage_assemble(args, work: Path) -> None:
    cap_file = work / "private_scores_captured.csv"
    use_cap = cap_file.exists()
    sc = pd.read_csv(cap_file if use_cap else work / "private_scores.csv", dtype={"id": str})
    log(f"assemble from {'CAPTURED-adjusted' if use_cap else 'plain combined'} scores")

    # strict total order: combined, fb tiebreak, id tiebreak -> unique values
    fb_col = "fb_s" if use_cap else "fb"
    sc = sc.sort_values(["combined", fb_col, "id"], kind="mergesort").reset_index(drop=True)
    n = len(sc)
    sc["value"] = (np.arange(n) + 0.5) / n

    if use_cap and "ens3" in sc.columns and sc.ens3.notna().any():
        # captured-internal reorder: value multiset fixed, order := ens3 ascending
        m = sc.ens3.notna()
        vals = np.sort(sc.loc[m, "value"].to_numpy())
        order = sc.loc[m].sort_values(["ens3", "id"], kind="mergesort").index
        sc.loc[order, "value"] = vals
        log(f"captured-internal reorder applied to {int(m.sum())} rows")

    new_vals = dict(zip(sc.id, ("%.10f" % v for v in sc.value)))
    assert len(set(new_vals.values())) == n, "tie collapse in formatted values"

    base_lines = Path(args.base).read_text().splitlines()
    assert base_lines[0] == "id,label", "unexpected header"
    out_lines, replaced = [base_lines[0]], 0
    for line in base_lines[1:]:
        rid = line.split(",", 1)[0]
        if rid in new_vals:
            out_lines.append(f"{rid},{new_vals[rid]}")
            replaced += 1
        else:
            out_lines.append(line)
    assert replaced == n, f"replaced {replaced} != scored {n}"
    assert len(out_lines) == len(base_lines) == N_TEST_FULL + 1

    if not args.dry_run:
        pub = public_ids()
        assert len(pub) == 7821, f"public id listing broken ({len(pub)})"
        assert not (set(new_vals) & pub), "would overwrite public rows — abort"
        exp_priv = N_TEST_FULL - len(pub)
        assert replaced == exp_priv, f"private coverage {replaced}/{exp_priv} — check missing_ids"

    out = Path(args.out) if args.out else work / ("dryrun_submission.csv" if args.dry_run
                                                  else "private_day_submission.csv")
    out.write_text("\n".join(out_lines) + "\n")
    log(f"wrote {out} (rows={len(out_lines)-1}, replaced={replaced}, "
        f"public byte-identical, base={Path(args.base).name})")
    log("NEXT: eyeball resolutions.csv + this log, then submit via kaggle CLI "
        "(manual, user confirm) and write .meta.md")


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["inventory", "infer", "combine", "captured", "assemble", "all"])
    p.add_argument("--images-dir", required=True)
    p.add_argument("--work", default=None)
    p.add_argument("--base", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", type=int, default=0, metavar="N",
                   help="mechanics check: N public images stand in for private")
    p.add_argument("--go-captured", action="store_true",
                   help="human GO for captured lever (after reviewing inventory)")
    args = p.parse_args()
    if args.work is None:
        args.work = str(OUT / "private_day_fb5")
    if args.base is None:
        args.base = str(BASE_SUB)
    work = Path(args.work)
    if args.dry_run:
        work = work / "dryrun"
    work.mkdir(parents=True, exist_ok=True)
    log(f"pick: fb5 | work dir: {work} | base: {Path(args.base).name}")

    stages = ([args.stage] if args.stage != "all"
              else ["inventory", "infer", "combine", "captured", "assemble"])
    for s in stages:
        log(f"===== stage: {s} =====")
        {"inventory": stage_inventory, "infer": stage_infer, "combine": stage_combine,
         "captured": stage_captured, "assemble": stage_assemble}[s](args, work)
    log("ALL REQUESTED STAGES DONE")


if __name__ == "__main__":
    main()
