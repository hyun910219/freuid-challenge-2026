"""Augmentation pipelines (FA/FB core).

PORT of efs/ml_workspace/kaggle/src/data/transforms.py, trimmed to the verified
FA/FB path: base geometric/color aug + ``domain_aug`` (JPEG/blur/tone/gamma/
downscale/perspective) + ``aspect_resize`` (fable v1: 전 type aspect≈1.585 동일 →
직사각 resize, crop 손실 0 · padding leak 0). Dropped experimental branches:
color_norm(ShadesOfGray, RAW>CAL 음성결과) / recapture_aug / short_side_crop.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def shades_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy (Finlayson & Trezzi 2004). Minkowski p-norm
    illuminant estimate -> per-channel gain to equalize channel means."""
    img_f = image.astype(np.float32)
    e = (np.mean(img_f ** p, axis=(0, 1))) ** (1.0 / p)  # (C,)
    gain = float(e.mean()) / (e + 1e-8)
    return np.clip(img_f * gain, 0, 255).astype(np.uint8)


class ShadesOfGray(ImageOnlyTransform):
    """C-shot V2: STOCHASTIC SoG aug (p<1). Differs from the dropped color_norm
    branch, which applied SoG deterministically (p=1, train+eval, RAW>CAL 음성) —
    here it randomizes the color-cast axis train-only, eval untouched."""

    def __init__(self, p_norm: int = 6, p: float = 0.5):
        super().__init__(p=p)
        self.p_norm = p_norm

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return shades_of_gray(img, p=self.p_norm)

    def get_transform_init_args_names(self):
        return ("p_norm",)


def _resize_ops(image_size, resize_strategy: str) -> list:
    if resize_strategy == "aspect_resize":
        # image_size = (H, W) 직사각 resize
        h, w = image_size
        return [A.Resize(height=h, width=w)]
    # letterbox fallback (int image_size)
    return [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0),
    ]


def _train(image_size, *, domain_aug, resize_strategy, mean, std, sog_p=0.0) -> A.Compose:
    ops: list = list(_resize_ops(image_size, resize_strategy))
    if sog_p > 0:  # C-shot V2: stochastic color-constancy aug (before ColorJitter)
        ops.append(ShadesOfGray(p_norm=6, p=sog_p))
    ops.extend([
        A.HorizontalFlip(p=0.5),
        A.Affine(rotate=(-7, 7), scale=(0.95, 1.05), translate_percent=0.03, p=0.7),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03, p=0.5),
    ])
    if domain_aug:
        ops.extend([
            A.ImageCompression(quality_range=(30, 95), p=0.5),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.3),
            A.RandomToneCurve(scale=0.1, p=0.3),
            A.RandomGamma(gamma_limit=(85, 115), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.2),
            A.Downscale(
                scale_range=(0.5, 0.9),
                interpolation_pair={"downscale": 1, "upscale": 1},
                p=0.2,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
        ])
    ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(ops)


def _eval(image_size, *, resize_strategy, mean, std) -> A.Compose:
    ops: list = list(_resize_ops(image_size, resize_strategy))
    ops.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(ops)


def build_transforms(
    mode: str,
    image_size=448,
    *,
    domain_aug: bool = False,
    resize_strategy: str = "aspect_resize",
    mean: tuple | None = None,
    std: tuple | None = None,
    sog_p: float = 0.0,
) -> A.Compose:
    """``mode`` ∈ {train, valid, test}. valid/test 는 동일 변환(aug 없음)."""
    m = tuple(mean) if mean is not None else IMAGENET_MEAN
    s = tuple(std) if std is not None else IMAGENET_STD
    if mode == "train":
        return _train(image_size, domain_aug=domain_aug, resize_strategy=resize_strategy,
                      mean=m, std=s, sog_p=sog_p)
    if mode in ("valid", "test"):
        return _eval(image_size, resize_strategy=resize_strategy, mean=m, std=s)
    raise ValueError(f"unknown mode: {mode}")
