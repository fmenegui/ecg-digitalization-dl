"""A compact residual U-Net with an optional ``timm`` encoder.

The built-in encoder has no external dependency and no pretrained weights, so the
package always works offline. For large-scale training, any ``timm`` backbone with
``features_only`` support can replace it (``encoder="timm:convnext_small"`` for example).
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1, act: bool = True):
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(ConvNormAct(cin, cout, 3, stride), ConvNormAct(cout, cout, 3, act=False))
        self.skip = nn.Identity() if cin == cout and stride == 1 else ConvNormAct(cin, cout, 1, stride, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class ResEncoder(nn.Module):
    """Residual encoder. Feature ``k`` has stride ``stem_stride * 2**k``."""

    def __init__(self, in_ch: int, widths: Sequence[int], depths: Sequence[int], stem_stride: int = 1):
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        self.channels = list(widths)
        self.strides = [stem_stride * 2**k for k in range(len(widths))]
        stages = []
        cin = in_ch
        for k, (w, d) in enumerate(zip(widths, depths)):
            stride = stem_stride if k == 0 else 2
            blocks: list[nn.Module] = [ConvNormAct(cin, w, 3, stride)] if k == 0 else [ResBlock(cin, w, stride)]
            blocks += [ResBlock(w, w) for _ in range(d)]
            stages.append(nn.Sequential(*blocks))
            cin = w
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class TimmEncoder(nn.Module):
    """``timm`` backbone plus a light full-resolution stem for fine detail.

    Deep pretrained backbones start at stride 2 or 4; thin ECG traces and sub-pixel grid
    intersections need earlier features, so a small convolutional stem supplies them
    (the "fine encoder" idea of dual-encoder competition models).
    """

    def __init__(self, name: str, in_ch: int, pretrained: bool = False, stem_width: int = 32, stem_stride: int = 1):
        super().__init__()
        try:
            import timm
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("install the 'timm' extra to use timm encoders") from e
        self.backbone = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_ch)
        info = self.backbone.feature_info
        self.fine = nn.Sequential(ConvNormAct(in_ch, stem_width, 3, stem_stride), ResBlock(stem_width, stem_width))
        self.channels = [stem_width, *info.channels()]
        self.strides = [stem_stride, *info.reduction()]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.fine(x), *self.backbone(x)]


class UNet(nn.Module):
    """Encoder-decoder returning the finest decoder feature and the deepest encoder feature."""

    def __init__(
        self,
        in_ch: int = 3,
        widths: Sequence[int] = (32, 48, 96, 160, 256),
        depths: Sequence[int] = (1, 1, 2, 2, 2),
        stem_stride: int = 1,
        encoder: str = "res",
        pretrained: bool = False,
        decoder_widths: Sequence[int] | None = None,
    ):
        super().__init__()
        if encoder == "res":
            self.encoder: nn.Module = ResEncoder(in_ch, widths, depths, stem_stride)
        elif encoder.startswith("timm:"):
            self.encoder = TimmEncoder(encoder[5:], in_ch, pretrained, widths[0], stem_stride)
        else:
            raise ValueError(f"unknown encoder: {encoder}")
        ch = self.encoder.channels
        dec = list(decoder_widths) if decoder_widths is not None else [min(c, 256) for c in ch[:-1]]
        if len(dec) != len(ch) - 1:
            raise ValueError("decoder_widths must have one entry per skip connection")
        self.fuse = nn.ModuleList()
        cin = ch[-1]
        for k in range(len(ch) - 2, -1, -1):
            self.fuse.append(nn.Sequential(ConvNormAct(cin + ch[k], dec[k], 3), ResBlock(dec[k], dec[k])))
            cin = dec[k]
        self.out_channels = dec[0]
        self.bottleneck_channels = ch[-1]
        self.out_stride = self.encoder.strides[0]
        self.total_stride = self.encoder.strides[-1]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        y = feats[-1]
        for k, fuse in zip(range(len(feats) - 2, -1, -1), self.fuse):
            y = F.interpolate(y, size=feats[k].shape[-2:], mode="bilinear", align_corners=False)
            y = fuse(torch.cat([y, feats[k]], 1))
        return y, feats[-1]


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Reflect-pad ``[B, C, H, W]`` on the bottom/right to a multiple of ``multiple``."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        mode = "reflect" if ph < h and pw < w else "replicate"
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (h, w)
