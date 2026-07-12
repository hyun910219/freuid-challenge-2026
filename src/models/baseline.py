"""timm-backed binary classifier (FA/FB core).

PORT of efs/ml_workspace/kaggle/src/models/baseline.py, trimmed to the FA/FB path:
DINOv2-L backbone + MLP head. Dropped SRM 6ch input and the SSDG
forward_with_feats path (unused by FA/FB).

Backbone: vit_large_patch14_dinov2.lvd142m (timm, Apache 2.0).
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class BinaryClassifier(nn.Module):
    """timm backbone + single-logit head (sigmoid 은 loss/추론에서 처리).

    freeze_backbone=True + unfreeze_last_blocks=N: 마지막 N transformer block +
    norm + head 만 학습 (large 모델 over-specialization 방지). LoRA(FB)는 이 위에
    inject_lora 로 q/k/v adapter 를 얹고 backbone 을 완전 freeze(unfreeze_last_blocks=0).
    """

    def __init__(
        self,
        backbone: str = "vit_large_patch14_dinov2.lvd142m",
        pretrained: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False,
        img_size=None,
        unfreeze_last_blocks: int = 0,
        head_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        num_classes = 0 if head_hidden_dim else 1
        kwargs: dict = dict(
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        if img_size is not None:
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(backbone, **kwargs)

        if head_hidden_dim is not None:
            feat_dim = self.backbone.num_features
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            self.head = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last_blocks > 0 and hasattr(self.backbone, "blocks"):
                for blk in self.backbone.blocks[-unfreeze_last_blocks:]:
                    for p in blk.parameters():
                        p.requires_grad = True
                if hasattr(self.backbone, "norm"):
                    for p in self.backbone.norm.parameters():
                        p.requires_grad = True
            tm_head = self.backbone.get_classifier()
            if tm_head is not None and hasattr(tm_head, "parameters"):
                for p in tm_head.parameters():
                    p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits.squeeze(-1)
