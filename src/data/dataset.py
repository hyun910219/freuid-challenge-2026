"""Dataset for FREUID Challenge 2026 (FA/FB core).

PORT of efs/ml_workspace/kaggle/src/data/dataset.py, trimmed to the FA/FB path:
bona -> synthetic field-tamper attack injection (train only), with the g1 switch
threaded through explicitly. Dropped experimental inputs: srm(6ch) / seamless /
pc_aug(print-capture) / recap_inv.

CSV schema (verified 2026-05-28): id, image_path, label, is_digital, type
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FreuidDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        transforms: Callable | None = None,
        mode: str = "train",
        id_col: str = "id",
        path_col: str = "image_path",
        label_col: str = "label",
        tamper_synth_p: float = 0.0,
        g1: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.transforms = transforms
        self.mode = mode
        self.id_col = id_col
        self.path_col = path_col
        self.label_col = label_col
        # bona -> 합성 국소 필드조작 attack (train 전용)
        self.tamper_synth_p = float(tamper_synth_p) if mode == "train" else 0.0
        self.g1 = bool(g1)
        if mode not in ("train", "valid", "test"):
            raise ValueError(f"unknown mode: {mode}")
        if mode != "test" and label_col not in self.df.columns:
            raise ValueError(f"label column '{label_col}' not in csv")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path: str) -> Path:
        primary = self.image_root / rel_path
        if primary.exists():
            return primary
        parts = Path(rel_path).parts
        if parts and parts[0] in {"train_sample", "train", "test"}:
            nested = self.image_root / parts[0] / Path(*parts)
            if nested.exists():
                return nested
        raise FileNotFoundError(f"image not found: {primary} (rel={rel_path})")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = self._resolve_image_path(str(row[self.path_col]))
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception:
            if self.mode != "test":
                raise
            # sandbox contract: one row per image — score undecodable files
            # on a neutral canvas instead of killing the whole run
            img = np.full((512, 512, 3), 128, dtype=np.uint8)
        label = float(row[self.label_col]) if self.mode != "test" else None
        if self.tamper_synth_p > 0 and label == 0.0:
            import torch

            from .tamper import apply_field_tamper

            # deterministic per-(seed, idx) rng — 재현성 + worker 간 독립
            rng = np.random.default_rng((torch.initial_seed() + idx * 2654435761) % 2**63)
            if rng.random() < self.tamper_synth_p:
                img = apply_field_tamper(img, rng, g1=self.g1)
                label = 1.0
        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        item: dict[str, Any] = {"image": img, "id": str(row[self.id_col])}
        if self.mode != "test":
            item["label"] = label
        if "is_digital" in self.df.columns:
            item["is_digital"] = bool(row["is_digital"])
        if "type" in self.df.columns:
            item["type"] = str(row["type"])
        return item
