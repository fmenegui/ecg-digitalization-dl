"""Sampling of metric regions of the paper straight from the original photo.

Every crop the networks see after the page stage is resampled **once**, from the
full-resolution input, through the lattice map. There is no intermediate "rectified
page" image whose interpolation blur would be paid twice.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .lattice import Lattice


def region_coords(
    lattice: Lattice,
    x0_mm: float,
    y0_mm: float,
    width_mm: float,
    height_mm: float,
    out_w: int,
    out_h: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Photo pixel coordinates ``[out_h, out_w, 2]`` of the centres of a metric region.

    The region is the upright rectangle ``[x0, x0 + width] x [y0, y0 + height]`` in paper
    millimetres. Horizontal and vertical resolutions are independent, which lets the trace
    network spend its pixels on time resolution.
    """
    xs = x0_mm + (torch.arange(out_w, device=device, dtype=torch.float32) + 0.5) * (width_mm / out_w)
    ys = y0_mm + (torch.arange(out_h, device=device, dtype=torch.float32) + 0.5) * (height_mm / out_h)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)  # [out_h, out_w, 2]
    return lattice.mm_to_image(grid)


def sample_image(image: torch.Tensor, coords: torch.Tensor, antialias: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear lookup of ``image[C, H, W]`` at pixel-centre ``coords[h, w, 2]``.

    When the photo is much sharper than the requested region (a 12 MP photo sampled into a
    small crop) the image is box-filtered first so that thin traces are not aliased away.

    Returns ``(crop[C, h, w], inside[h, w])``; ``inside`` is false where the region leaves
    the photo.
    """
    if image.ndim != 3:
        raise ValueError("image must be [C, H, W]")
    _, h, w = image.shape
    coords = coords.to(image.device)
    inside = (coords[..., 0] >= -0.5) & (coords[..., 0] <= w - 0.5) & (coords[..., 1] >= -0.5) & (coords[..., 1] <= h - 0.5)
    src = image[None]
    if antialias and coords.shape[0] > 1 and coords.shape[1] > 1:
        step_x = (coords[:, 1:] - coords[:, :-1]).norm(dim=-1).median()
        step_y = (coords[1:, :] - coords[:-1, :]).norm(dim=-1).median()
        factor = int(torch.minimum(step_x, step_y).clamp(1, 8).floor().item())
        if factor >= 2:
            src = F.avg_pool2d(src, factor, ceil_mode=True)
            coords = (coords + 0.5) / factor - 0.5
            h, w = src.shape[-2:]
    gx = (coords[..., 0] + 0.5) / w * 2 - 1
    gy = (coords[..., 1] + 0.5) / h * 2 - 1
    out = F.grid_sample(src, torch.stack((gx, gy), -1)[None], mode="bilinear", padding_mode="border", align_corners=False)
    return out[0], inside


def sample_region(
    image: torch.Tensor,
    lattice: Lattice,
    x0_mm: float,
    y0_mm: float,
    width_mm: float,
    height_mm: float,
    out_w: int,
    out_h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: :func:`region_coords` followed by :func:`sample_image`."""
    coords = region_coords(lattice, x0_mm, y0_mm, width_mm, height_mm, out_w, out_h, image.device)
    return sample_image(image, coords)
