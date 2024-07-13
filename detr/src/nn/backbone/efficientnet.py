import torch
from torch import nn, Tensor
from torchvision.models import efficientnet

from src.core import register
from .common import FrozenBatchNorm2d

__all__ = [
    "EfficientNet",
]

call_eff = {
    "b0": efficientnet.efficientnet_b0,
    "b1": efficientnet.efficientnet_b1,
    "b2": efficientnet.efficientnet_b2,
    "b3": efficientnet.efficientnet_b3,
    "b4": efficientnet.efficientnet_b4,
    "b5": efficientnet.efficientnet_b5,
    "b6": efficientnet.efficientnet_b6,
    "b7": efficientnet.efficientnet_b7,
}


@register
class EfficientNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        return_strides: list = [8, 16, 32],
        freeze_norm: bool = True,
    ):
        super().__init__()

        model = call_eff[model_name](pretrained=pretrained)
        self.features = nn.ModuleList([module for module in model.features])

        del model

        self.return_idx = []
        x = torch.randn(1, 3, 640, 640)
        last_shape = 640

        if freeze_norm:
            self._freeze_norm(self)

        for i, module in enumerate(self.features):
            x = module(x)
            print(x.shape)

            if 640 // x.shape[2] in return_strides:
                if last_shape == x.shape[2]:
                    self.return_idx[-1] = i
                else:
                    last_shape = x.shape[2]
                    self.return_idx.append(i)
        print(self.return_idx)

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x: Tensor) -> Tensor:
        outs = []

        for i, module in enumerate(self.features):
            x = module(x)

            if i in self.return_idx:
                outs.append(x)

        return outs
