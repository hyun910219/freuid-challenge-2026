"""FA/FB training entry point (FREUID final).

PORT of efs/ml_workspace/kaggle/src/train.py, trimmed to the verified FA/FB path
and re-wired to the OFFICIAL metric for model selection (eval_freuid.freuid_eval,
not the deprecated np.interp harness). Dropped branches: SSDG / dual_stream /
EMA / color_norm / srm / recap_inv / print_capture.

Supports:
  - curriculum: init_ckpt (stage A -> stage B) with pos_embed interpolation
  - tamper-synth v4 set + g1 (config: train.tamper_synth_p, train.g1)
  - LoRA (config: model.lora) with post-train merge -> standard ckpt
  - stop_after_epochs (verified epoch-0 recipe reproduction) + lr warmup

Usage:
    python -m src.train --config configs/ff_a_stageB_fold0.yaml
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import FreuidDataset, build_transforms
from src.eval_freuid import freuid_eval
from src.models import BinaryClassifier
from src.utils.logger import JsonlLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split(df, fold_csv, fold):
    if fold_csv is None:
        return df, df
    folds = pd.read_csv(fold_csv)
    merged = df.merge(folds[["id", "fold"]], on="id", how="left")
    train_df = merged[merged.fold != fold].drop(columns=["fold"]).reset_index(drop=True)
    valid_df = merged[merged.fold == fold].drop(columns=["fold"]).reset_index(drop=True)
    return train_df, valid_df


def _write_split_csv(df, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return dst


def _fact_mix(images: torch.Tensor, p: float, alpha_max: float) -> torch.Tensor:
    """FACT (Xu et al. 2021): batch-level Fourier amplitude interpolation.

    Amplitude spectrum ~ style/domain, phase ~ semantics -> each image keeps
    its own phase (and label), amplitude is lerped toward a shuffled batch
    partner with lam ~ U(0, alpha_max). Train-only, config-gated (fact_p).
    """
    b = images.shape[0]
    if b < 2 or p <= 0:
        return images
    mask = torch.rand(b, device=images.device) < p
    if not mask.any():
        return images
    # random rotation (not randperm): guarantees partner != self for every image
    shift = int(torch.randint(1, b, (1,)))
    perm = (torch.arange(b, device=images.device) + shift) % b
    lam = torch.rand(b, 1, 1, 1, device=images.device) * alpha_max
    fft = torch.fft.fft2(images.float(), dim=(-2, -1))
    amp, pha = fft.abs(), fft.angle()
    amp_new = (1.0 - lam) * amp + lam * amp[perm]
    mixed = torch.fft.ifft2(torch.polar(amp_new, pha), dim=(-2, -1)).real
    return torch.where(mask.view(b, 1, 1, 1), mixed, images.float()).to(images.dtype)


def _load_init_ckpt(model: BinaryClassifier, ckpt_path: str) -> None:
    """Initialize high-res fine-tune from low-res ckpt (pos_embed interpolation)."""
    from timm.layers import resample_abs_pos_embed

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"]
    key = "backbone.pos_embed"
    tgt_shape = model.state_dict()[key].shape
    if state[key].shape != tgt_shape:
        old_hw = ck["cfg"]["data"]["image_size"]
        patch = model.backbone.patch_embed.patch_size[0]
        old_size = [old_hw[0] // patch, old_hw[1] // patch]
        new_size = list(model.backbone.patch_embed.grid_size)
        state[key] = resample_abs_pos_embed(
            state[key], new_size=new_size, old_size=old_size,
            num_prefix_tokens=getattr(model.backbone, "num_prefix_tokens", 1),
        )
        assert state[key].shape == tgt_shape, \
            f"pos_embed resample mismatch: {state[key].shape} vs {tgt_shape}"
    model.load_state_dict(state, strict=True)


def _build_loader(csv, root, image_size, cfg, *, train: bool):
    ds = FreuidDataset(
        csv, root,
        transforms=build_transforms(
            "train" if train else "valid",
            image_size=image_size,
            domain_aug=cfg.train.get("domain_aug", False) if train else False,
            resize_strategy=cfg.train.get("resize_strategy", "aspect_resize"),
            mean=cfg.data.get("norm_mean", None),  # None -> ImageNet (DINOv2); CLIP needs its own
            std=cfg.data.get("norm_std", None),
            sog_p=cfg.train.get("sog_p", 0.0) if train else 0.0,  # C-shot V2: stochastic SoG
        ),
        mode="train" if train else "valid",
        tamper_synth_p=cfg.train.get("tamper_synth_p", 0.0) if train else 0.0,
        g1=cfg.train.get("g1", False),
    )
    device_cuda = torch.cuda.is_available()
    return DataLoader(
        ds, batch_size=cfg.train.batch_size, shuffle=train,
        num_workers=cfg.train.num_workers, drop_last=False, pin_memory=device_cuda,
    )


def train(cfg) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(out_dir, run_name=cfg.run_name)
    logger.log(stage="setup", device=str(device), cfg=OmegaConf.to_container(cfg))

    df = pd.read_csv(cfg.data.csv_path)
    image_size = cfg.data.image_size
    if not isinstance(image_size, int):
        image_size = tuple(image_size)
    train_df, valid_df = _split(df, cfg.data.fold_csv, cfg.data.fold)
    train_csv = _write_split_csv(train_df, out_dir / "train_split.csv")
    valid_csv = _write_split_csv(valid_df, out_dir / "valid_split.csv")

    train_loader = _build_loader(train_csv, cfg.data.image_root, image_size, cfg, train=True)
    valid_loader = _build_loader(valid_csv, cfg.data.image_root, image_size, cfg, train=False)

    model = BinaryClassifier(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.get("pretrained", True),
        drop_rate=cfg.model.drop_rate,
        drop_path_rate=cfg.model.drop_path_rate,
        freeze_backbone=cfg.model.get("freeze_backbone", False),
        img_size=image_size,
        unfreeze_last_blocks=cfg.model.get("unfreeze_last_blocks", 0),
        head_hidden_dim=cfg.model.get("head_hidden_dim", None),
    ).to(device)

    init_ckpt = cfg.train.get("init_ckpt", None)
    if init_ckpt:
        m = model.cpu()
        _load_init_ckpt(m, init_ckpt)
        model = m.to(device)
        logger.log(stage="init_ckpt", path=str(init_ckpt))

    lora_cfg = cfg.model.get("lora", None)
    if lora_cfg:
        from src.models.lora import inject_lora, lora_param_count, merge_lora
        model = inject_lora(
            model, r=lora_cfg.get("r", 16), alpha=lora_cfg.get("alpha", 32),
            targets=tuple(lora_cfg.get("targets", ["qkv"])),
            dropout=lora_cfg.get("dropout", 0.0), last_n=lora_cfg.get("last_n", None),
        ).to(device)
        logger.log(stage="lora_inject", n_params=lora_param_count(model))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.train.amp and device.type == "cuda"))

    warmup_steps = int(cfg.train.get("warmup_steps", 0))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    global_step = 0
    best_freuid = float("inf")
    best_path = out_dir / "best.ckpt"

    val_every = int(cfg.train.get("val_every_steps", 0))
    mid_loader = None
    if val_every:
        import torch.utils.data as _tud
        _vds = valid_loader.dataset
        _stride = max(1, len(_vds) // 3000)
        _sub = _tud.Subset(_vds, list(range(0, len(_vds), _stride)))
        mid_loader = _tud.DataLoader(_sub, batch_size=valid_loader.batch_size,
                                     shuffle=False, num_workers=0)

    all_scores: list[float] = []
    all_labels: list[float] = []
    all_ids: list[str] = []
    log_every = int(cfg.train.get("log_every_steps", 100))  # throughput/loss (E0.3 timing)
    fact_p = float(cfg.train.get("fact_p", 0.0))            # C-shot V1: FACT amplitude mix
    fact_alpha = float(cfg.train.get("fact_alpha", 0.5))
    n_train_steps = len(train_loader)
    for epoch in range(cfg.train.epochs):
        model.train()
        train_losses: list[float] = []
        _t_win = time.time()
        for step_in_epoch, batch in enumerate(train_loader):
            if warmup_steps and global_step < warmup_steps:
                scale = float(global_step + 1) / float(warmup_steps)
                for g, b in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = b * scale
            global_step += 1
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).float()
            if fact_p > 0:
                images = _fact_mix(images, fact_p, fact_alpha)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

            if log_every and global_step % log_every == 0:
                dt = time.time() - _t_win
                its = log_every / dt if dt > 0 else 0.0
                eta_ep_min = (n_train_steps - step_in_epoch - 1) / its / 60 if its > 0 else -1
                logger.log(stage="step", epoch=epoch, step=global_step,
                           step_in_epoch=step_in_epoch + 1, n_steps=n_train_steps,
                           it_per_s=round(its, 3), recent_loss=round(float(np.mean(train_losses[-log_every:])), 4),
                           epoch_eta_min=round(eta_ep_min, 1))
                _t_win = time.time()

            if mid_loader is not None and global_step % val_every == 0:
                model.eval()
                vs, vl = [], []
                with torch.no_grad():
                    for vb in mid_loader:
                        vlog = model(vb["image"].to(device, non_blocking=True))
                        vs.extend(torch.sigmoid(vlog).cpu().numpy().tolist())
                        vl.extend(vb["label"].numpy().tolist())
                vr = freuid_eval(np.array(vl), np.array(vs))
                logger.log(stage="mid_epoch", epoch=epoch, step=global_step, n=len(vl),
                           valid_freuid=vr.score, valid_apcer_at_1pct=vr.apcer_at_bpcer)
                model.train()
        scheduler.step()

        model.eval()
        all_scores, all_labels, all_ids = [], [], []
        with torch.no_grad():
            for batch in valid_loader:
                logits = model(batch["image"].to(device, non_blocking=True))
                all_scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                all_labels.extend(batch["label"].numpy().tolist())
                all_ids.extend(batch["id"])
        rep = freuid_eval(np.array(all_labels), np.array(all_scores))
        logger.log(stage="epoch", epoch=epoch,
                   train_loss=float(np.mean(train_losses)) if train_losses else None,
                   lr=optimizer.param_groups[0]["lr"],
                   valid_freuid=rep.score, valid_audet=rep.audet,
                   valid_apcer_at_1pct=rep.apcer_at_bpcer)

        if rep.score < best_freuid:
            best_freuid = rep.score
            torch.save({"model": model.state_dict(),
                        "cfg": OmegaConf.to_container(cfg),
                        "epoch": epoch, "freuid": best_freuid}, best_path)

        stop_after = cfg.train.get("stop_after_epochs", None)
        if stop_after is not None and (epoch + 1) >= int(stop_after):
            logger.log(stage="early_stop", epoch=epoch, stop_after_epochs=int(stop_after))
            break

    if lora_cfg:
        from src.models.lora import merge_lora
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        merge_lora(model)
        ck["model"] = model.state_dict()
        ck["lora_merged"] = True
        torch.save(ck, best_path)
        logger.log(stage="lora_merge", path=str(best_path))

    oof_df = pd.DataFrame({"id": all_ids, "score": all_scores, "label": all_labels})
    oof_path = out_dir / f"oof_fold{cfg.data.fold}.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.log(stage="done", best_freuid=best_freuid, oof=str(oof_path))
    logger.close()
    return {"best_freuid": best_freuid, "best_ckpt": str(best_path), "oof": str(oof_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    result = train(cfg)
    print("=" * 60)
    print(f"BEST FREUID (official) = {result['best_freuid']:.4f}")
    print(f"OOF csv: {result['oof']}")


if __name__ == "__main__":
    main()
