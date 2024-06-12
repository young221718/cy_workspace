"""by lyuwenyu
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import get_activation

from src.core import register


__all__ = ["EfficientEncoder"]


# bifpn ==============================================================================
class MConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False, act=None):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=bias
        )

        # Pointwise Convolution
        self.pointwise_conv = nn.Conv2d(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=bias
        )

        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return self.act(self.norm(x))


class FastNormalizedFusion(nn.Module):
    """Combines 2 or 3 feature maps into one with weights.
    Args:
        input_num (int): 2 for intermediate features, 3 for output features
    """

    def __init__(self, in_nodes, activation="relu"):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(in_nodes, dtype=torch.float32))
        self.eps = 1e-4
        self.act = get_activation(activation)

    def forward(self, *features):
        # Assure that weights are positive (see paper)
        weights = F.relu(self.weights)
        # Normalize weights
        weights = weights / (weights.sum() + self.eps)
        fused_features = sum([p * w for p, w in zip(features, weights)])
        return self.act(fused_features)


# need to create weights to allow loading anyway. So inherit from FastNormalizedFusion for simplicity
class SumFusion(FastNormalizedFusion):
    def forward(self, *features):
        return self.act(sum(features))


class BiFPNLayer(nn.Module):
    """Builds one layer of Bi-directional Feature Pyramid Network
    Args:
        channels (int): Number of channels in each feature map after BiFPN. Defaults to 64.

    Input:
        features (List): 5 feature maps from encoder with resolution from 1/128 to 1/8

    Returns:
        p_out: features processed by 1 layer of BiFPN
    """

    def __init__(self, num_features=5, channels=256, norm_act="relu"):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.down = nn.MaxPool2d(3, stride=2, padding=1)
        self.num_features = num_features

        # disable attention for large models. This is a very dirty way to check that it's B6 & B7. But i don't care
        Fusion = SumFusion if channels > 288 else FastNormalizedFusion

        # There is no activation in SeparableConvs, instead activation is in fusion layer
        # fusions for p6, p5, p4, p3. (no fusion for first feature map)
        self.fuse_up = nn.ModuleList(
            [Fusion(in_nodes=2, activation=norm_act) for _ in range(num_features - 1)]
        )

        # fusions for p4, p5, p6, p7. last is different because there is no bottop up tensor for it
        self.fuse_out = nn.ModuleList(
            [
                *(
                    Fusion(in_nodes=3, activation=norm_act)
                    for _ in range(num_features - 2)
                ),
                Fusion(in_nodes=2, activation=norm_act),
            ]
        )

        # Top-down pathway, no block for first and last features. P3 and P7 by default
        self.p_up_convs = nn.ModuleList(
            [MConvNormLayer(channels, channels) for _ in range(num_features - 1)]
        )

        # Bottom-up pathway
        self.p_out_convs = nn.ModuleList(
            [MConvNormLayer(channels, channels) for _ in range(num_features - 1)]
        )

    def forward(self, features):
        # p7_in, p6_in, p5_in, p4_in, p3_in = features
        # Top-down pathway (from low res to high res). High res features depend on upsampled low res
        features.reverse()
        p_up = [features[0]]  # from p7 to p3
        for idx in range(self.num_features - 1):
            p_up.append(
                self.p_up_convs[idx](
                    self.fuse_up[idx](  # fuse: input and upsampled previous feature map
                        features[idx + 1], self.up(p_up[-1])
                    )
                )
            )

        # Bottom-Up Pathway (from high res to low res). Low res depends on downscaled high res
        p_out = [p_up[-1]]  # p3 is final and ready to be returned. from p3 to p7
        for idx in range(1, self.num_features - 1):
            p_out.append(
                self.p_out_convs[
                    idx - 1
                ](  # fuse: input, output from top-bottom path and downscaled high res
                    self.fuse_out[idx - 1](
                        features[-(idx + 1)], p_up[-(idx + 1)], self.down(p_out[-1])
                    )
                )
            )
        # fuse for p7: input, downscaled high res
        p_out.append(
            self.p_out_convs[-1](self.fuse_out[-1](features[0], self.down(p_out[-1])))
        )

        return p_out  # want to return in the same order as input


class BiFPN(nn.Sequential):
    """
    Implementation of Bi-directional Feature Pyramid Network

    Args:
        num_features (int): Number of channels for each feature map from low res to high res.
        hidden_dim (int): Number of channels in each feature map after BiFPN. Defaults to 64.
        num_layers (int): Number or repeats for BiFPN block. Default is 2

    Input:
        features (List): 5 feature maps from encoder [low_res, ... , high_res]

    https://arxiv.org/pdf/1911.09070.pdf
    """

    def __init__(self, num_features, hidden_dim=256, num_layers=1):
        # First layer preprocesses raw encoder features
        bifpns = []
        # Apply BiFPN block `num_layers` times
        for _ in range(num_layers):
            bifpns.append(BiFPNLayer(num_features, hidden_dim))
        super().__init__(*bifpns)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class EfficientEncoder(nn.Module):
    def __init__(
        self,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        encoder_feat_strides=[64, 128],
        num_encoder_layers=[1, 1],
        num_bifpn_layers=1,
        pe_temperature=10000,
        act="silu",
        eval_spatial_size=None,
    ):
        super().__init__()
        assert len(encoder_feat_strides) == len(num_encoder_layers)
        print("EfficientEncoder")

        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.encoder_feat_strides = encoder_feat_strides
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        self.encoder_proj = nn.ModuleList()
        for _ in encoder_feat_strides:
            self.encoder_proj.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )

        self.encoders = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_layer)
                for num_layer in num_encoder_layers
            ]
        )

        self.bifpn = BiFPN(
            len(feat_strides) + len(encoder_feat_strides), hidden_dim, num_bifpn_layers
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx, stride in enumerate(self.encoder_feat_strides):
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """ """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        for i, encoder in enumerate(self.encoders):
            proj_feats.append(self.encoder_proj[i](proj_feats[-1]))
            h, w = proj_feats[-1].shape[2:]

            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = proj_feats[-1].flatten(2).permute(0, 2, 1)
            pos_embed = self.build_2d_sincos_position_embedding(
                w, h, self.hidden_dim, self.pe_temperature
            ).to(src_flatten.device)

            memory = encoder(src_flatten, pos_embed=pos_embed)
            proj_feats[-1] = (
                memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
            )

        # broadcasting and fusion
        outs = self.bifpn(proj_feats)

        return outs
