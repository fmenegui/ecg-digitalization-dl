"""PageNet: one pass over the photo, every dense cue the geometry needs.

Outputs (all at input resolution):

``grid``       1 channel   logit of a 5 mm grid intersection (Dotter-style keypoints)
``offset``     2 channels  sub-pixel (dx, dy) from the pixel centre to the intersection
``trace``      1 channel   logit of ECG trace ink (labels, pulses and grid excluded)
``names``      12 channels logit of the centre of a printed lead name, one channel per lead
``direction``  2 values    unit vector (cos, sin) of the reading/time direction on the page

The direction is regressed as a vector rather than classified into four rotations:
it has no decision boundary at 45 degrees, and the lattice later snaps it to a grid axis.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import N_LEADS
from .unet import ConvNormAct, UNet, pad_to_multiple


@dataclass(frozen=True)
class PageNetConfig:
    widths: Sequence[int] = (32, 48, 96, 160, 256)
    depths: Sequence[int] = (1, 1, 2, 2, 2)
    #: stride of the finest feature map; heads are upsampled back to full resolution
    stem_stride: int = 2
    encoder: str = "res"
    pretrained: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["widths"], d["depths"] = list(self.widths), list(self.depths)
        return d


class PageNet(nn.Module):
    N_DENSE = 1 + 2 + 1 + N_LEADS

    def __init__(self, config: PageNetConfig | None = None):
        super().__init__()
        self.config = config or PageNetConfig()
        c = self.config
        self.unet = UNet(3, c.widths, c.depths, c.stem_stride, c.encoder, c.pretrained)
        s = self.unet.out_stride
        width = self.unet.out_channels
        self.head = nn.Sequential(
            ConvNormAct(width, width, 3),
            nn.Conv2d(width, self.N_DENSE * s * s, 1),
            nn.PixelShuffle(s) if s > 1 else nn.Identity(),
        )
        self.direction = nn.Sequential(
            nn.Linear(self.unet.bottleneck_channels, 128), nn.SiLU(inplace=True), nn.Linear(128, 2)
        )
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1))
        # Sparse targets: start the heatmap logits strongly negative so the loss is stable.
        # (PixelShuffle maps conv channel ``c * s*s + k`` to output channel ``c``.)
        bias = torch.full((self.N_DENSE, s * s), -4.0)
        bias[1:3] = 0.0  # the two offset channels are regressions, not logits
        with torch.no_grad():
            self.head[1].bias.copy_(bias.flatten())

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """``image`` is ``[B, 3, H, W]`` in ``[0, 1]``."""
        x, (h, w) = pad_to_multiple((image - self.mean) / self.std, self.unet.total_stride)
        feat, deep = self.unet(x)
        dense = self.head(feat)[..., :h, :w]
        direction = F.normalize(self.direction(deep.mean((-2, -1))), dim=-1)
        return {
            "grid": dense[:, 0:1],
            "offset": dense[:, 1:3],
            "trace": dense[:, 3:4],
            "names": dense[:, 4:],
            "direction": direction,
        }
