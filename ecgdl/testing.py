"""Oracles for testing the pipeline without trained weights.

``OraclePageNet`` answers with the renderer's ground truth, and ``InkTraceNet`` is a
classical darkest-ink tracer. Plugged into :class:`ecgdl.pipeline.Digitizer` they exercise
every non-learned stage (lattice, layout, metric crops, decoding, calibration, fusion)
end to end, which separates geometry bugs from model errors and gives an upper bound for
what better networks can deliver.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import N_LEADS
from .data.synth import PageSample
from .data.targets import page_targets


def _logit(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


class OraclePageNet(nn.Module):
    """Returns the ground-truth maps of one :class:`PageSample`, resized to the request."""

    def __init__(self, sample: PageSample, name_dropout: float = 0.0, seed: int = 0):
        super().__init__()
        t = page_targets(sample, grid_sigma=1.2)
        names = t["names"].clone()
        if name_dropout > 0:
            rng = np.random.default_rng(seed)
            for lead in range(N_LEADS):
                if rng.random() < name_dropout:
                    names[lead] = 0
        self.maps = torch.cat([t["grid"], t["trace"], names])
        self.direction = t["direction"]

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        h, w = image.shape[-2:]
        maps = F.interpolate(self.maps[None].to(image.device), size=(h, w), mode="bilinear", align_corners=False)
        return {
            "grid": _logit(maps[:, 0:1]),
            "offset": torch.full((1, 2, h, w), 9.0, device=image.device),  # implausible -> quadratic fit is used
            "trace": _logit(maps[:, 1:2]),
            "names": _logit(maps[:, 2:]),
            "direction": self.direction[None].to(image.device),
        }


class InkTraceNet(nn.Module):
    """Classical stand-in for TraceNet: column logits proportional to ink darkness.

    Only adequate for clean prints with a light grid; its purpose is testing the plumbing.
    """

    def __init__(self, crop_height: int = 128, px_per_mm_x: float = 8.0, sharpness: float = 40.0):
        super().__init__()
        self.config = SimpleNamespace(crop_height=crop_height, px_per_mm_x=px_per_mm_x)
        self.unet = SimpleNamespace(total_stride=1)
        self.sharpness = sharpness

    def forward(self, crop: torch.Tensor) -> dict[str, torch.Tensor]:
        gray = crop.amax(1)  # coloured grid lines are bright in at least one channel
        # ink is a thin stroke darker than what lies just above and below it (a dark table is not)
        k = max(3, crop.shape[-2] // 12) | 1
        above = F.max_pool2d(gray[:, None], (k, 1), 1, (k // 2, 0))[:, 0]
        dark = (above - gray).clamp_min(0)
        # prefer ink close to the centre row when several strokes compete
        rows = torch.linspace(-1, 1, crop.shape[-2], device=crop.device).view(1, -1, 1)
        logits = self.sharpness * dark - 12.0 * rows**2
        visible = torch.full((crop.shape[0], crop.shape[-1]), 6.0, device=crop.device)
        return {"logits": logits, "visible": visible}


def paper_time_ground_truth(sample: PageSample, fs: float = 500.0) -> np.ndarray:
    """Reference ``[12, n]`` in *paper time*: the instant of a sample is its x position.

    NaN where a lead is not printed. For sequential layouts paper time equals recording time.
    Values are the true lead (an inverted ``-aVR`` print is expected to be restored).
    """
    seg = sample.segments
    origin = float(np.median([s[2] for s in seg if s[2] <= seg[:, 2].min() + 5.0]))
    duration = max((s[3] - origin) / sample.speed for s in seg)
    n = int(round(duration * fs))
    out = np.full((N_LEADS, n), np.nan)
    t = np.arange(n) / fs
    for lead, _sign, x0, x1, _, t0, t1 in seg:
        lo, hi = (x0 - origin) / sample.speed, (x1 - origin) / sample.speed
        m = (t >= lo) & (t <= hi)
        src = t0 + (t[m] - lo)
        ok = src <= min(t1, (sample.signals.shape[1] - 1) / sample.fs)
        vals = np.interp(src, np.arange(sample.signals.shape[1]) / sample.fs, sample.signals[int(lead)])
        idx = np.nonzero(m)[0][ok]
        out[int(lead), idx] = vals[ok]
    return out


def draw_page(
    signals: np.ndarray,
    fs: float = 500.0,
    layout: tuple[tuple[str, ...], ...] = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6")),
    rhythm: tuple[str, ...] = ("II",),
    px_per_mm: float = 6.0,
    speed: float = 25.0,
    gain: float = 10.0,
    segment_s: float = 2.5,
    row_mm: float = 32.0,
    tilt: float = 0.06,
    seed: int = 0,
) -> PageSample:
    """Minimal procedural ECG page with exact ground truth, for tests without the renderer.

    A flat page (red 5 mm / 1 mm grid, black traces, lead names) is drawn upright and then
    viewed through a mild random perspective. It is deliberately simple: realism is the
    job of ``ecg-torch-renderer``; this only needs to exercise the geometry end to end.
    """
    from PIL import Image, ImageDraw, ImageFont

    from .constants import LEAD_INDEX

    rng = np.random.default_rng(seed)
    cols = len(layout[0])
    margin = 15.0
    width_mm = 2 * margin + cols * segment_s * speed
    n_rows = len(layout) + len(rhythm)
    height_mm = 2 * margin + n_rows * row_mm
    w, h = int(round(width_mm * px_per_mm)), int(round(height_mm * px_per_mm))
    img = Image.new("RGB", (w, h), (252, 250, 246))
    ink = Image.new("L", (w, h), 0)
    d, di = ImageDraw.Draw(img), ImageDraw.Draw(ink)
    for k in range(int(width_mm) + 1):
        x = k * px_per_mm
        d.line([(x, 0), (x, h)], fill=(240, 150, 150) if k % 5 == 0 else (250, 222, 222), width=2 if k % 5 == 0 else 1)
    for k in range(int(height_mm) + 1):
        y = k * px_per_mm
        d.line([(0, y), (w, y)], fill=(240, 150, 150) if k % 5 == 0 else (250, 222, 222), width=2 if k % 5 == 0 else 1)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(3.2 * px_per_mm))
    except OSError:  # pragma: no cover - depends on the system fonts
        font = ImageFont.load_default()
    segments, name_xy, name_size, name_lead = [], [], [], []

    def trace(lead: str, x0: float, x1: float, baseline: float, t0: float) -> None:
        li = LEAD_INDEX[lead]
        xs = np.arange(x0 * px_per_mm, x1 * px_per_mm, 0.5)
        t = t0 + (xs / px_per_mm - x0) / speed
        mv = np.interp(t * fs, np.arange(signals.shape[1]), signals[li])
        pts = list(zip(xs.tolist(), ((baseline - mv * gain) * px_per_mm).tolist()))
        d.line(pts, fill=(20, 20, 24), width=2, joint="curve")
        di.line(pts, fill=255, width=2)
        box = d.textbbox((0, 0), lead, font=font)
        tx, ty = (x0 + 2.0) * px_per_mm, (baseline - 9.0) * px_per_mm
        d.text((tx, ty), lead, fill=(20, 20, 24), font=font)
        name_xy.append([tx + (box[0] + box[2]) / 2, ty + (box[1] + box[3]) / 2])
        name_size.append([box[2] - box[0], box[3] - box[1]])
        name_lead.append(li)
        segments.append([li, 1.0, x0, x1, baseline, t0, t0 + (x1 - x0) / speed])

    for r, row in enumerate(layout):
        for c, lead in enumerate(row):
            x0 = margin + c * segment_s * speed
            trace(lead, x0, x0 + segment_s * speed, margin + (r + 0.6) * row_mm, c * segment_s)
    for r, lead in enumerate(rhythm):
        trace(lead, margin, margin + cols * segment_s * speed, margin + (len(layout) + r + 0.6) * row_mm, 0.0)

    # view through a random mild perspective: output pixel -> source pixel homography
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    moved = corners + rng.uniform(-tilt, tilt, (4, 2)) * np.array([w, h])
    from .geometry.lattice import apply_homography, fit_homography

    fwd = fit_homography(corners, moved)  # upright -> photo
    inv = np.linalg.inv(fwd)
    coeffs = (inv / inv[2, 2]).flatten()[:8]
    photo = img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(70, 60, 55))
    ink_photo = ink.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    valid = Image.new("L", (w, h), 255).transform((w, h), Image.PERSPECTIVE, coeffs, Image.NEAREST)
    gi, gj = np.meshgrid(np.arange(int(width_mm // 5) + 1), np.arange(int(height_mm // 5) + 1))
    grid_ij = np.stack([gi.ravel(), gj.ravel()], 1)
    grid_xy = apply_homography(fwd, grid_ij * 5.0 * px_per_mm)
    keep = (grid_xy[:, 0] > 1) & (grid_xy[:, 0] < w - 2) & (grid_xy[:, 1] > 1) & (grid_xy[:, 1] < h - 2)
    direction = apply_homography(fwd, np.array([[w / 2 + 10.0, h / 2]]))[0] - apply_homography(fwd, np.array([[w / 2, h / 2]]))[0]
    return PageSample(
        image=np.array(photo), valid=np.array(valid), trace=np.array(ink_photo),
        grid_xy=grid_xy[keep].astype(np.float32), grid_ij=grid_ij[keep].astype(np.int32),
        name_xy=apply_homography(fwd, np.array(name_xy)).astype(np.float32),
        name_size=np.array(name_size, np.float32), name_lead=np.array(name_lead, np.int64),
        direction=(direction / np.linalg.norm(direction)).astype(np.float32),
        segments=np.array(segments, np.float32), signals=signals.astype(np.float32), fs=fs, speed=speed, gain=gain,
        px_per_mm=px_per_mm,
    )
