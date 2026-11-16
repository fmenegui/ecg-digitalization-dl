"""Sub-pixel peak extraction from keypoint heatmaps."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def decode_peaks(
    heat: torch.Tensor,
    offsets: torch.Tensor | None = None,
    threshold: float = 0.3,
    nms_radius: int = 2,
    max_points: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of a probability map ``heat[H, W]`` with sub-pixel refinement.

    If ``offsets[2, H, W]`` (dx, dy from the peak pixel centre to the true centre, each in
    ``[-0.5, 0.5]``) is given and plausible it is used; otherwise the position is refined
    with a three-point log-quadratic fit, exact for an isolated Gaussian.

    Returns ``xy[N, 2]`` (x, y at pixel centres) and ``score[N]``, sorted by score.
    """
    if heat.ndim != 2:
        raise ValueError("heat must be [H, W]")
    h, w = heat.shape
    heat = heat.float()
    k = 2 * nms_radius + 1
    # A centre halfway between two pixels gives two equal maxima: break the tie with a tiny ramp.
    ramp = torch.arange(h * w, device=heat.device, dtype=torch.float32).view(h, w) * (1e-6 / (h * w))
    ranked = heat + ramp
    # separable max filter: O(k) per pixel instead of O(k^2), which matters for wide suppression radii
    pooled = F.max_pool2d(ranked[None, None], (k, 1), 1, (nms_radius, 0))
    pooled = F.max_pool2d(pooled, (1, k), 1, (0, nms_radius))[0, 0]
    peak = (ranked >= pooled) & (heat >= threshold)
    ys, xs = torch.nonzero(peak, as_tuple=True)
    if ys.numel() == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)
    score = heat[ys, xs]
    if score.numel() > max_points:
        score, order = torch.topk(score, max_points)
        ys, xs = ys[order], xs[order]

    def at(dy: int, dx: int) -> torch.Tensor:
        return heat[(ys + dy).clamp(0, h - 1), (xs + dx).clamp(0, w - 1)].clamp_min(1e-6).log()

    c = at(0, 0)
    def quad(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        denom = lo - 2 * c + hi
        d = 0.5 * (lo - hi) / torch.where(denom.abs() < 1e-6, torch.full_like(denom, -1e-6), denom)
        return d.clamp(-0.5, 0.5)

    dx, dy = quad(at(0, -1), at(0, 1)), quad(at(-1, 0), at(1, 0))
    dx = torch.where((xs == 0) | (xs == w - 1), torch.zeros_like(dx), dx)
    dy = torch.where((ys == 0) | (ys == h - 1), torch.zeros_like(dy), dy)
    if offsets is not None:
        ox, oy = offsets[0, ys, xs].float(), offsets[1, ys, xs].float()
        good = (ox.abs() <= 0.75) & (oy.abs() <= 0.75)
        dx = torch.where(good, ox.clamp(-0.5, 0.5), dx)
        dy = torch.where(good, oy.clamp(-0.5, 0.5), dy)
    xy = torch.stack((xs.float() + dx, ys.float() + dy), -1)
    order = torch.argsort(score, descending=True)
    return xy[order].cpu().numpy().astype(np.float32), score[order].cpu().numpy().astype(np.float32)
