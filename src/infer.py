"""Inference + submission writer (FA/FB, 5-fold equal-weight ensemble + Hflip TTA).

PORT/rewrite of efs/ml_workspace/kaggle/src/inference.py +
src/scripts/ensemble_5fold_tta_submit.py, trimmed to the verified path.

- Hflip TTA (2-view mean) — verified small stabilizer (0.3181->0.3179).
- Equal-weight ensemble only (OOF-optimal weights DO NOT transfer to LB).
- Submission = (id, label) with FULL float precision (no rounding — tie collapse
  makes the rank-only metric score constants at 1.0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import FreuidDataset, build_transforms
from src.models import BinaryClassifier


def _prepare_test_csv(test_csv: Path) -> Path:
    df = pd.read_csv(test_csv)
    if "image_path" not in df.columns:
        df["image_path"] = df["id"].astype(str).apply(lambda i: f"test/{i}.jpeg")
        tmp = test_csv.with_suffix(".inference.csv")
        df.to_csv(tmp, index=False)
        return tmp
    return test_csv


def _load_model(cfg, ckpt_path, device):
    model = BinaryClassifier(
        backbone=cfg.model.backbone, pretrained=False,
        drop_rate=cfg.model.drop_rate, drop_path_rate=cfg.model.drop_path_rate,
        img_size=(tuple(cfg.data.image_size) if not isinstance(cfg.data.image_size, int)
                  else cfg.data.image_size),
        head_hidden_dim=cfg.model.get("head_hidden_dim", None),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def predict_ckpt(cfg, ckpt_path, test_csv, image_root=None, tta_hflip: bool = True) -> pd.DataFrame:
    """Return DataFrame(id, score) for one checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = _prepare_test_csv(Path(test_csv))
    root = str(image_root) if image_root is not None else cfg.data.image_root
    image_size = tuple(cfg.data.image_size) if not isinstance(cfg.data.image_size, int) \
        else cfg.data.image_size
    ds = FreuidDataset(
        test_csv, root,
        transforms=build_transforms("test", image_size=image_size,
                                    resize_strategy=cfg.train.get("resize_strategy", "aspect_resize"),
                                    mean=cfg.data.get("norm_mean", None),  # None -> ImageNet; CLIP needs its own
                                    std=cfg.data.get("norm_std", None)),
        mode="test",
    )
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.train.num_workers)
    model = _load_model(cfg, ckpt_path, device)
    ids: list[str] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            p = torch.sigmoid(model(imgs))
            if tta_hflip:
                p2 = torch.sigmoid(model(torch.flip(imgs, dims=[-1])))
                p = 0.5 * (p + p2)
            scores.extend(p.cpu().numpy().tolist())
            ids.extend(batch["id"])
    return pd.DataFrame({"id": ids, "score": scores})


def ensemble_predict(members, test_csv, image_root=None, tta_hflip: bool = True) -> pd.DataFrame:
    """members = list of (cfg, ckpt_path). Equal-weight mean of per-member scores.

    Members are aligned on id (merged), so fold order / row order can differ.
    """
    merged = None
    for i, (cfg, ckpt) in enumerate(members):
        df = predict_ckpt(cfg, ckpt, test_csv, image_root, tta_hflip).rename(
            columns={"score": f"s{i}"})
        merged = df if merged is None else merged.merge(df, on="id", how="inner")
    scols = [c for c in merged.columns if c.startswith("s")]
    merged["score"] = merged[scols].mean(axis=1)
    return merged[["id", "score"]]


def write_submission(df: pd.DataFrame, out_csv: Path) -> Path:
    """Write (id, label) with full float precision."""
    out = df.rename(columns={"score": "label"})[["id", "label"]]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.17g")  # full precision, no tie collapse
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="reference config (backbone/data settings)")
    p.add_argument("--ckpts", nargs="+", required=True, help="one or more fold checkpoints")
    p.add_argument("--test-csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-root", default=None)
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    members = [(cfg, c) for c in args.ckpts]
    df = ensemble_predict(members, args.test_csv, args.image_root, tta_hflip=not args.no_tta)
    out = write_submission(df, Path(args.out))
    print(f"[ok] wrote {out}  (n={len(df)}, members={len(members)})")


if __name__ == "__main__":
    main()
