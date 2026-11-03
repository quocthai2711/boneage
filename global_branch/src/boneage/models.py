"""Image and sex-fusion model."""

from __future__ import annotations

import timm
import torch
from torch import nn


class BoneAgeModel(nn.Module):
    def __init__(self, backbone_name: str, *, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feature_dim = int(
            getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        )
        self.sex_mlp = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16))
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 16, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def freeze_backbone(self, frozen: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, image: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        image_features = self.backbone(image)
        sex_features = self.sex_mlp(sex)
        return self.head(torch.cat([image_features, sex_features], dim=1)).squeeze(1)
