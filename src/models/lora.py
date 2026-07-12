"""Minimal LoRA for timm ViT attention (q/k/v), with merge-back to plain Linear.

PORT of efs/ml_workspace/kaggle/src/models/lora.py (verbatim; clean already).

근거(deep-research 2026-06-27): frozen backbone + LoRA on q/k/v 는 pretrained
일반화 prior 를 보존하고 digital-forgery 편향 overfit 을 줄이는 저비용 cross-domain
레시피(C2P-CLIP AAAI2025). 학습 후 merge 하면 표준 BinaryClassifier state_dict 가
되어 추론/앙상블 스크립트가 그대로 로드한다(구조 불변).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """base(frozen) Linear + low-rank delta. B=0 초기화라 초기 delta=0(학습 시작 시 항등)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        delta = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
        merged = nn.Linear(
            self.base.in_features, self.base.out_features,
            bias=self.base.bias is not None,
        )
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def _target_attns(model):
    bb = getattr(model, "backbone", None)
    if bb is None or not hasattr(bb, "blocks"):
        raise ValueError("model has no backbone.blocks (timm ViT expected)")
    for blk in bb.blocks:
        if hasattr(blk, "attn"):
            yield blk.attn


def inject_lora(model, r: int = 16, alpha: float = 32, targets=("qkv",), dropout: float = 0.0,
                last_n: int | None = None):
    """transformer block 의 attn.{targets} Linear 를 LoRALinear 로 교체.

    last_n 지정 시 마지막 N block 에만 주입 → backward 깊이 제한(속도↑, early 일반특징 보존).
    """
    bb = getattr(model, "backbone", None)
    if bb is None or not hasattr(bb, "blocks"):
        raise ValueError("model has no backbone.blocks (timm ViT expected)")
    blocks = bb.blocks if last_n is None else bb.blocks[-int(last_n):]
    n = 0
    for blk in blocks:
        if not hasattr(blk, "attn"):
            continue
        attn = blk.attn
        for t in targets:
            mod = getattr(attn, t, None)
            if isinstance(mod, nn.Linear):
                setattr(attn, t, LoRALinear(mod, r=r, alpha=alpha, dropout=dropout))
                n += 1
    if n == 0:
        raise ValueError(f"inject_lora matched 0 modules for targets={targets}")
    return model


def merge_lora(model):
    """LoRALinear → merged 표준 Linear 로 in-place 교체(추론 호환)."""
    for attn in _target_attns(model):
        for name, child in list(attn.named_children()):
            if isinstance(child, LoRALinear):
                dev = child.lora_A.device
                setattr(attn, name, child.merged_linear().to(dev))
    return model


def lora_param_count(model) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if "lora_" in n and p.requires_grad)
