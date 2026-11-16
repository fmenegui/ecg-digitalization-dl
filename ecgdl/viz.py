"""Debug overlays drawn with Pillow only."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from .constants import LEADS
from .pipeline import DigitizationResult, load_image

_COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
            (240, 50, 230), (170, 110, 40), (0, 128, 128), (128, 0, 0), (0, 0, 128), (128, 128, 0)]


def overlay(source, result: DigitizationResult, max_side: int = 1800) -> Image.Image:
    """Photo with the recovered lattice, the segment boxes and the decoded traces drawn on top.

    Requires a result produced with ``debug=True``.
    """
    image = load_image(source)
    h, w = image.shape[-2:]
    scale = min(1.0, max_side / max(h, w))
    canvas = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    if scale < 1:
        canvas = canvas.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    if not result.ok or result.lattice is None or not result.debug:
        draw.text((10, 10), "; ".join(result.notes) or "no result", fill=(255, 0, 0))
        return canvas
    page_scale = result.debug["scale"]

    def to_canvas(mm: np.ndarray) -> list[tuple[float, float]]:
        px = result.lattice.mm_to_image(torch.as_tensor(mm, dtype=torch.float32)).numpy()
        px = ((px + 0.5) / page_scale - 0.5) * scale
        return [tuple(p) for p in px.tolist()]

    for x, y in result.debug["grid_xy"] * scale:
        draw.point((float(x), float(y)), fill=(0, 200, 255))
    for seg, dec in zip(result.segments, result.debug["decoded"]):
        colour = _COLOURS[seg.lead % len(_COLOURS)]
        half = dec["span"] / 2
        box = np.array([[seg.x0, seg.baseline - half], [seg.x1, seg.baseline - half],
                        [seg.x1, seg.baseline + half], [seg.x0, seg.baseline + half], [seg.x0, seg.baseline - half]])
        draw.line(to_canvas(box), fill=colour, width=1)
        ok = (dec["x_mm"] >= seg.x0) & (dec["x_mm"] <= seg.x1) & (dec["visible"] >= 0.5)
        pts = np.stack([dec["x_mm"][ok], dec["y_mm"][ok]], 1)
        if len(pts) > 1:
            draw.line(to_canvas(pts), fill=colour, width=2)
        draw.text(to_canvas(np.array([[seg.x0 + 1, seg.baseline - half + 1]]))[0], LEADS[seg.lead], fill=colour)
    return canvas
