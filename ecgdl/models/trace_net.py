"""TraceNet: a metric crop of one printed lead in, one vertical position per column out.

The crop is centred on the estimated baseline of the *target* lead; neighbouring leads,
grid, text and handwriting may all be inside it. The network outputs, for every column,
logits over rows (decoded by :mod:`ecgdl.signal.codec`) and a per-column visibility logit
(the lead is absent, occluded or outside the paper in that column).

A normalised row-coordinate channel is appended to the input (CoordConv): "the target is
the trace whose baseline sits at the centre" is a positional statement, and plain
convolutions cannot see absolute position.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .unet import ConvNormAct, UNet, pad_to_multiple


@dataclass(frozen=True)
class TraceNetConfig:
    widths: Sequence[int] = (32, 64, 96, 160, 224)
    depths: Sequence[int] = (1, 1, 2, 2, 2)
    encoder: str = "res"
    pretrained: bool = False
    #: crop geometry the network was trained with (stored with the weights)
    crop_height: int = 160
    px_per_mm_x: float = 6.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["widths"], d["depths"] = list(self.widths), list(self.depths)
        return d


class TraceNet(nn.Module):
    def __init__(self, config: TraceNetConfig | None = None):
        super().__init__()
        self.config = config or TraceNetConfig()
        c = self.config
        self.unet = UNet(4, c.widths, c.depths, 1, c.encoder, c.pretrained)
        width = self.unet.out_channels
        self.head = nn.Sequential(ConvNormAct(width, width, 3), nn.Conv2d(width, 2, 1))
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))

    def forward(self, crop: torch.Tensor) -> dict[str, torch.Tensor]:
        """``crop`` is ``[B, 3, H, W]`` in ``[0, 1]``. Returns ``logits[B, H, W]`` and ``visible[B, W]``."""
        b, _, h, w = crop.shape
        rows = torch.linspace(-1, 1, h, device=crop.device, dtype=crop.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        x = torch.cat([(crop - self.mean) / self.std, rows], 1)
        x, _ = pad_to_multiple(x, self.unet.total_stride)
        feat, _ = self.unet(x)
        out = self.head(feat)[..., :h, :w]
        return {"logits": out[:, 0], "visible": out[:, 1].mean(1)}
