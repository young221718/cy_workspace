"""by lyuwenyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from src.core import register


__all__ = [
    "RTDETR_auxloss",
]


@register
class RTDETR_auxloss(nn.Module):
    __inject__ = [
        "backbone",
        "encoder",
        "decoder",
    ]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        multi_scale=None,
        num_classes=80,
        backbone_aux_loss=False,
        encoder_aux_loss=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale

        self.backbone_aux_loss = backbone_aux_loss
        if backbone_aux_loss:
            self.backbone_logits = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(encoder.in_channels[-1], num_classes*2),
            )

        self.encoder_aux_loss = encoder_aux_loss
        if encoder_aux_loss:
            raise NotImplementedError

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        out = {}
        x = self.backbone(x)
        if self.backbone_aux_loss and targets is not None:
            out["backbone_logits"] = self.backbone_logits(x[-1])

        x = self.encoder(x)
        if self.encoder_aux_loss and targets is not None:
            pass
            # TODO
            # out["encoder_logits"] =

        x = self.decoder(x, targets)
        x.update(out)

        return x

    def deploy(
        self,
    ):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
