"""End-to-end digitization: photo or scan in, 12-lead signals in millivolts out.

Stages (see ``docs/ARCHITECTURE.md``):

1. **PageNet** finds 5 mm grid intersections, trace ink, lead names and the reading direction.
2. **Lattice** (classical) indexes the intersections and fits a smooth paper-to-photo map.
3. **Layout** (classical) reads rows, columns and lead identities in paper millimetres.
4. **TraceNet** decodes each printed segment from a metric crop resampled once from the
   original image; a dynamic-programming decoder turns column distributions into a path.
5. **Fusion** (classical) aligns duplicates and projects limb leads onto the
   Einthoven/Goldberger subspace with confidence weights.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F

from .constants import DEFAULT_GAIN_MM_MV, DEFAULT_SPEED_MM_S, LEADS, LIMB_BASIS, N_LEADS, OUTPUT_FS
from .geometry.lattice import Lattice, LatticeConfig, LatticeError, build_lattice
from .geometry.peaks import decode_peaks
from .geometry.rectify import region_coords, sample_image
from .layout.assemble import Layout, LayoutConfig, NameDetection, Segment, assemble_layout, row_hypotheses
from .layout.verify import (
    chest_half_from_polarity,
    choose_layout,
    ecg_likeness,
    identify_by_correlation,
    shift_columns,
)
from .models.io import load_checkpoint
from .signal.codec import decode_columns
from .signal.constraints import Observation, align_duplicates, einthoven_residual, fuse_leads
from .signal.resample import columns_to_signal


@dataclass
class DigitizerConfig:
    speed_mm_s: float = DEFAULT_SPEED_MM_S
    gain_mm_mv: float = DEFAULT_GAIN_MM_MV
    output_fs: float = OUTPUT_FS
    #: long side of the first PageNet pass (pixels)
    page_long_side: int = 1152
    #: grid spacing (pixels per 5 mm) PageNet works best at; a second pass rescales towards it
    target_spacing_px: float = 20.0
    spacing_range_px: tuple[float, float] = (13.0, 32.0)
    page_long_side_limits: tuple[int, int] = (640, 2304)
    grid_threshold: float = 0.3
    name_threshold: float = 0.3
    #: resolution of the rectified maps used for layout analysis (pixels per mm)
    layout_px_per_mm: float = 2.0
    #: how far beyond the detected grid the page is searched for traces (mm, extrapolated lattice)
    layout_margin_mm: float = 30.0
    #: cap of the photo resolution used for trace crops (pixels per mm); ``None`` = no cap
    max_source_px_per_mm: float | None = 8.0
    #: vertical extent of trace crops (mm): minimum, maximum and the "tall" retry cap
    span_mm: tuple[float, float, float] = (32.0, 64.0, 96.0)
    segment_margin_mm: float = 2.0
    decode: str = "hybrid"
    flip_tta: bool = False
    use_constraints: bool = True
    reconstruct_limb: bool = False
    #: maximum time shift searched when aligning repeated leads (seconds)
    duplicate_align_s: float = 0.08
    visibility_threshold: float = 0.5
    #: closure residual below which an upside-down reading is believed (see ``ecgdl.layout.verify``)
    turn_accept: float = 0.4
    trace_batch: int = 8
    lattice: LatticeConfig = field(default_factory=lambda: LatticeConfig(margin_cells=7))
    layout: LayoutConfig = field(default_factory=LayoutConfig)


@dataclass
class DigitizationResult:
    """Output contract: ``signals[12, n]`` in mV at ``fs``; NaN where a lead is not printed."""

    ok: bool
    signals: np.ndarray
    confidence: np.ndarray
    fs: float
    leads: tuple[str, ...] = LEADS
    segments: list[Segment] = field(default_factory=list)
    #: per lead name: ``(first, last)`` output sample of every printed occurrence
    segment_samples: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    layout: str | None = None
    quality: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    lattice: Lattice | None = None
    debug: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-friendly summary (signals excluded)."""
        return {
            "ok": self.ok, "fs": self.fs, "leads": list(self.leads), "layout": self.layout,
            "segments": [asdict(s) | {"lead_name": LEADS[s.lead] if s.lead >= 0 else None} for s in self.segments],
            "segment_samples": self.segment_samples, "quality": self.quality, "notes": self.notes,
            "timings": self.timings,
        }


def load_image(source: str | Path | Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Any image input as ``[3, H, W]`` float in ``[0, 1]`` (EXIF rotation applied)."""
    if isinstance(source, torch.Tensor):
        t = source.float()
        return t / 255 if t.max() > 1.5 else t
    if isinstance(source, np.ndarray):
        arr = source
    else:
        img = source if isinstance(source, Image.Image) else Image.open(source)
        arr = np.array(ImageOps.exif_transpose(img).convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    t = torch.from_numpy(np.ascontiguousarray(arr[..., :3])).permute(2, 0, 1).float()
    return t / 255 if arr.dtype == np.uint8 or t.max() > 1.5 else t


class Digitizer:
    """Reusable digitizer. Build once (``from_checkpoints``) and call on many images."""

    def __init__(self, page_net: torch.nn.Module, trace_net: torch.nn.Module, config: DigitizerConfig | None = None,
                 device: str = "cpu"):
        self.page_net = page_net.to(device).eval()
        self.trace_net = trace_net.to(device).eval()
        self.config = config or DigitizerConfig()
        self.device = device

    @classmethod
    def from_checkpoints(cls, page: str | Path, trace: str | Path, config: DigitizerConfig | None = None,
                         device: str | None = None) -> Digitizer:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(load_checkpoint(page, device, "page"), load_checkpoint(trace, device, "trace"), config, device)

    # ------------------------------------------------------------------ stage 1
    @torch.no_grad()
    def _page_pass(self, image: torch.Tensor, long_side: int) -> tuple[dict[str, torch.Tensor], float]:
        h, w = image.shape[-2:]
        scale = long_side / max(h, w)
        size = (max(32, int(round(h * scale))), max(32, int(round(w * scale))))
        small = F.interpolate(image[None], size=size, mode="bilinear", antialias=scale < 1, align_corners=False)
        out = self.page_net(small.to(self.device))
        return out, size[1] / w

    def _detect_page(self, image: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, np.ndarray, Lattice | None, str]:
        """PageNet pass(es) and lattice. Returns ``(outputs, scale, grid points, lattice, error)``.

        The grid spacing is measured on the fitted lattice, not on raw detections: false
        positives (minor-grid crossings, text) shorten nearest-neighbour distances but do
        not form a consistent lattice. If the spacing is outside the range PageNet was
        trained on, or no lattice is found, the page is processed once more at a better scale.
        """
        cfg = self.config
        lo, hi = cfg.page_long_side_limits
        long_side = int(np.clip(min(cfg.page_long_side, 1.5 * max(image.shape[-2:])), lo, hi))

        def attempt(side: int):
            out, scale = self._page_pass(image, side)
            xy, score = decode_peaks(out["grid"][0, 0].sigmoid(), out["offset"][0], cfg.grid_threshold)
            try:
                return out, scale, xy, build_lattice(xy, score, cfg.lattice), ""
            except LatticeError as e:
                return out, scale, xy, None, str(e)

        first = attempt(long_side)
        lattice = first[3]
        if lattice is not None and cfg.spacing_range_px[0] <= lattice.spacing_px <= cfg.spacing_range_px[1]:
            return first
        if lattice is not None:
            sides = [long_side * cfg.target_spacing_px / lattice.spacing_px]
        else:
            sides = [long_side * 1.6, long_side * 0.65]
        best = first
        for side in sides:
            side = int(np.clip(side, lo, hi))
            if abs(side - long_side) <= 0.12 * long_side:
                continue
            second = attempt(side)
            if second[3] is not None and (best[3] is None or second[3].quality["nodes"] >= 0.8 * best[3].quality["nodes"]):
                best = second
                break
        return best

    # ------------------------------------------------------------------ stage 2
    def _row_peakiness(self, out: dict[str, torch.Tensor], lattice: Lattice) -> float:
        """How strongly trace ink concentrates on horizontal lines in the frame of ``lattice``."""
        width_mm, height_mm = lattice.extent_mm
        w, h = max(8, int(width_mm)), max(8, int(height_mm))
        coords = region_coords(lattice, 0.0, 0.0, float(w), float(h), w, h, self.device)
        prob = out["trace"][0].sigmoid()
        k = int(round(lattice.spacing_px / lattice.mm_per_cell)) | 1
        if k >= 3:
            prob = F.max_pool2d(prob[None], k, 1, k // 2)[0]
        ink, inside = sample_image(prob, coords, antialias=False)
        profile = (ink[0] * inside).sum(1)
        total = profile.sum()
        return float((profile**2).sum() / (total**2 + 1e-9) * h) if total > 0 else 0.0

    def _orient(self, out: dict[str, torch.Tensor], lattice: Lattice, direction: np.ndarray) -> tuple[Lattice, np.ndarray]:
        """Choose the reading direction: learned vector first, classical projection profile as referee.

        ECG traces run along the time axis, so in the correct frame their ink piles up on a
        few rows. If the frame turned by 90 degrees is clearly "peakier", the predicted
        direction was wrong by a quarter turn. (A half turn is caught later by the layout:
        lead names sit at the start of their segments.)
        """
        primary = lattice.oriented(direction)
        turned_dir = np.array([-direction[1], direction[0]])
        turned = lattice.oriented(turned_dir)
        if self._row_peakiness(out, turned) > 1.4 * self._row_peakiness(out, primary):
            return turned, turned_dir
        return primary, direction

    # ------------------------------------------------------------------ stage 3
    def _layout_maps(self, out: dict[str, torch.Tensor], lattice: Lattice, margin_mm: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        ppmm = self.config.layout_px_per_mm
        width_mm, height_mm = lattice.extent_mm
        w = int(round((width_mm + 2 * margin_mm) * ppmm))
        h = int(round((height_mm + 2 * margin_mm) * ppmm))
        coords = region_coords(lattice, -margin_mm, -margin_mm, w / ppmm, h / ppmm, w, h, self.device)
        maps = torch.cat([out["trace"][0], out["names"][0]]).sigmoid()
        # Thin strokes and point-like peaks must survive the reduction to layout resolution:
        # dilate by the reduction factor (a max filter) instead of averaging them away.
        k = int(round(lattice.spacing_px / lattice.mm_per_cell / ppmm))
        if k >= 2:
            k |= 1
            maps = F.max_pool2d(maps[None], k, 1, k // 2)[0]
        rect, inside = sample_image(maps, coords, antialias=False)
        rect = (rect * inside[None]).cpu().numpy()
        return rect[0], rect[1:], (-margin_mm, -margin_mm)

    def _read_layout(self, out: dict[str, torch.Tensor], lattice: Lattice, speed: float) -> tuple[Layout, np.ndarray, tuple[float, float]]:
        cfg = self.config
        ppmm = cfg.layout_px_per_mm
        ink, names_map, origin = self._layout_maps(out, lattice, margin_mm=cfg.layout_margin_mm)
        detections = []
        for lead in range(N_LEADS):
            xy, score = decode_peaks(torch.from_numpy(names_map[lead]), None, cfg.name_threshold,
                                     nms_radius=max(2, int(round(5 * ppmm))), max_points=6)
            detections += [NameDetection(lead, (x + 0.5) / ppmm, (y + 0.5) / ppmm, float(s)) for (x, y), s in zip(xy, score)]
        layout = assemble_layout(ink, ppmm, detections, speed, cfg.layout)
        # layout coordinates are relative to the map corner; move them to the lattice frame
        for seg in layout.segments:
            seg.x0, seg.x1, seg.baseline = seg.x0 + origin[0], seg.x1 + origin[0], seg.baseline + origin[1]
        layout.rows = [r + origin[1] for r in layout.rows]
        layout.row_extents = [(b + origin[1], x0 + origin[0], x1 + origin[0]) for b, x0, x1 in layout.row_extents]
        layout.time_origin_mm += origin[0]
        for extra in layout.extra_hypotheses:
            for seg in extra.segments:
                seg.x0, seg.x1, seg.baseline = seg.x0 + origin[0], seg.x1 + origin[0], seg.baseline + origin[1]
            extra.rows = [r + origin[1] for r in extra.rows]
            extra.row_extents = [(b + origin[1], x0 + origin[0], x1 + origin[0]) for b, x0, x1 in extra.row_extents]
            extra.time_origin_mm += origin[0]
        return layout, ink, origin

    # ------------------------------------------------------------------ stage 4
    def _span_for(self, gain: float) -> float:
        """Vertical extent of a trace crop (mm): +-2 mV at the page gain, within the trained range.

        A crop that turns out too small (the decoded trace touches its border) is decoded
        again with a taller one, see ``__call__``.
        """
        cfg = self.config
        return float(np.clip(4.0 * gain, cfg.span_mm[0], cfg.span_mm[1]))

    @torch.no_grad()
    def _decode_segments(self, image: torch.Tensor, lattice: Lattice, scale: float, segments: list[Segment],
                         spans: list[float]) -> list[dict]:
        cfg = self.config
        tc = self.trace_net.config
        h, ppmm_x = tc.crop_height, tc.px_per_mm_x
        stride = self.trace_net.unet.total_stride
        # optionally limit the source sharpness to what the trace network was trained on
        src, src_scale = image, 1.0
        native = lattice.spacing_px / lattice.mm_per_cell / scale  # px/mm of the original photo
        if cfg.max_source_px_per_mm and native > cfg.max_source_px_per_mm:
            src_scale = cfg.max_source_px_per_mm / native
            size = (int(round(image.shape[-2] * src_scale)), int(round(image.shape[-1] * src_scale)))
            src = F.interpolate(image[None], size=size, mode="bilinear", antialias=True, align_corners=False)[0]
        src = src.to(self.device)
        results: list[dict | None] = [None] * len(segments)
        jobs = []
        for k, (seg, span) in enumerate(zip(segments, spans)):
            x_start = seg.x0 - cfg.segment_margin_mm
            width_mm = (seg.x1 - seg.x0) + 2 * cfg.segment_margin_mm
            w = int(np.ceil(width_mm * ppmm_x / stride) * stride)
            jobs.append((w, k, x_start, w / ppmm_x, span))
        jobs.sort()
        i = 0
        while i < len(jobs):
            group = [j for j in jobs[i : i + cfg.trace_batch] if j[0] == jobs[i][0]]
            i += len(group)
            crops = []
            for w, k, x_start, width_mm, span in group:
                seg = segments[k]
                coords = region_coords(lattice, x_start, seg.baseline - span / 2, width_mm, span, w, h, self.device)
                coords = ((coords + 0.5) / scale) * src_scale - 0.5  # page-pass pixels -> source pixels
                crops.append(sample_image(src, coords)[0])
            batch = torch.stack(crops)
            pred = self.trace_net(batch)
            logits, visible = pred["logits"].float(), pred["visible"].float()
            if cfg.flip_tta:
                flipped = self.trace_net(batch.flip(-1))
                logits = 0.5 * (logits + flipped["logits"].float().flip(-1))
                visible = 0.5 * (visible + flipped["visible"].float().flip(-1))
            dec = decode_columns(logits, cfg.decode)
            y = dec["y"].cpu().numpy()
            conf = dec["confidence"].cpu().numpy()
            vis = torch.sigmoid(visible).cpu().numpy()
            for n, (w, k, x_start, width_mm, span) in enumerate(group):
                x_mm = x_start + (np.arange(w) + 0.5) * (width_mm / w)
                y_mm = segments[k].baseline - span / 2 + (y[n] + 0.5) * (span / h)
                clipped = float(np.mean((y[n] < 1.0) | (y[n] > h - 2.0)))
                results[k] = {"x_mm": x_mm, "y_mm": y_mm, "confidence": conf[n], "visible": vis[n],
                              "clipped": clipped, "span": span}
        return results  # type: ignore[return-value]

    def _row_strips(self, image: torch.Tensor, lattice: Lattice, scale: float, layout: Layout, speed: float,
                    gain: float) -> dict[int, np.ndarray]:
        """Every row decoded as one strip, in mV on the output grid that starts at the page time origin."""
        fs = self.config.output_fs
        rows: dict[int, Segment] = {}
        for seg in layout.segments:
            if seg.row not in rows:
                rows[seg.row] = Segment(0, seg.x0, seg.x1, seg.baseline, row=seg.row)
            rows[seg.row].x0, rows[seg.row].x1 = min(rows[seg.row].x0, seg.x0), max(rows[seg.row].x1, seg.x1)
        strips = list(rows.values())
        decoded = self._decode_segments(image, lattice, scale, strips, [self._span_for(gain)] * len(strips))
        out = {}
        for seg, d in zip(strips, decoded):
            t = (d["x_mm"] - layout.time_origin_mm) / speed
            n = int(round((seg.x1 - layout.time_origin_mm) / speed * fs))
            good = (d["x_mm"] >= seg.x0) & (d["x_mm"] <= seg.x1) & (d["visible"] >= self.config.visibility_threshold)
            out[seg.row] = columns_to_signal(-(d["y_mm"] - seg.baseline) / gain, t, fs, max(n, 1), good)
        return out

    def _referee(self, image: torch.Tensor, lattice: Lattice, scale: float, layout: Layout, speed: float,
                 gain: float, allow_turn: bool = True) -> tuple[Layout, bool]:
        """Settle what the names left open with the signals themselves.

        1. Every candidate row is decoded as one strip; rows that do not look like an ECG
           (page borders, lines of text) are dropped.
        2. The standard arrangements that fit the remaining rows are compared by how well
           their limb leads satisfy Einthoven and Goldberger.
        3. The same strips answer whether the page is upside down: a half turn reverses
           the row order, reverses time and inverts the voltage.

        Returns the chosen layout and whether the page must be turned by 180 degrees.
        """
        cfg = self.config
        fs = cfg.output_fs
        strips = self._row_strips(image, lattice, scale, layout, speed, gain)
        extents = layout.row_extents
        keep = [r for r in sorted(strips) if ecg_likeness(strips[r], fs) >= 0.25]
        notes = list(layout.notes)
        if len(keep) < len(strips):
            notes.append(f"{len(strips) - len(keep)} of {len(strips)} rows do not look like an ECG and were ignored")
        blank = np.zeros((1, 1), np.float32)

        def evaluate(subset: list[int]):
            """Best arrangement for a run of rows, upright and after a half turn."""
            rows = [extents[r] for r in subset]
            candidates = [h for h, _, _ in row_hypotheses(blank, 1.0, speed, cfg.layout, None, rows)]
            if not candidates:
                return None
            # strips start at the old time origin; move them to the origin of these rows
            shift = int(round((candidates[0].time_origin_mm - layout.time_origin_mm) / speed * fs))
            kept = {k: (strips[r][shift:] if shift >= 0 else np.concatenate([np.full(-shift, np.nan), strips[r]]))
                    for k, r in enumerate(subset)}
            chosen, residuals, offsets = choose_layout(candidates, kept, fs, speed)
            # the same rows read after a half turn: last row first, time reversed, voltage inverted
            length = int(round((max(r[2] for r in rows) - candidates[0].time_origin_mm) / speed * fs))
            turned = {}
            for k, v in kept.items():
                padded = np.full(max(length, len(v)), np.nan)
                padded[: len(v)] = v
                turned[len(kept) - 1 - k] = -padded[::-1]
            chosen_turned, residuals_turned, offsets_turned = choose_layout(candidates, turned, fs, speed)
            return {"rows": rows, "candidates": candidates, "kept": kept, "chosen": chosen,
                    "upright": float(residuals[candidates.index(chosen)]), "chosen_turned": chosen_turned,
                    "offset": float(offsets[candidates.index(chosen)]), "offsets": offsets,
                    "flipped": float(residuals_turned[candidates.index(chosen_turned)]),
                    "offset_turned": float(offsets_turned[candidates.index(chosen_turned)])}

        # A page border or a line of text can pass for a row at the top or bottom of the block.
        # Try the row block without up to two outer rows: the right block is the one that closes.
        best = None
        for drop_top in range(3):
            for drop_bottom in range(3 - drop_top):
                subset = keep[drop_top : len(keep) - drop_bottom]
                if len(subset) < 3 and (drop_top or drop_bottom):
                    continue
                result = evaluate(subset) if subset else None
                if result is None:
                    continue
                result["dropped"] = drop_top + drop_bottom
                result["score"] = np.nanmin([result["upright"], result["flipped"]]) if np.isfinite(
                    [result["upright"], result["flipped"]]).any() else np.inf
                closes = result["score"] <= 0.35
                if best is None or (closes and result["score"] < best["score"] - 0.05 * (result["dropped"] - best["dropped"])):
                    best = result
        # two-block pages bring their own rows: decode those and let the same test decide
        if layout.extra_hypotheses and (best is None or not best["score"] <= 0.25):
            block_strips = self._row_strips(image, lattice, scale, layout.extra_hypotheses[0], speed, gain)
            chosen_block, residuals, offsets = choose_layout(layout.extra_hypotheses, block_strips, fs, speed)
            k = layout.extra_hypotheses.index(chosen_block)
            if np.isfinite(residuals[k]) and residuals[k] <= 0.5 and (best is None or residuals[k] < 0.8 * best["score"]):
                for seg in chosen_block.segments:  # slide both blocks by the lead-in (calibration pulse)
                    seg.x0, seg.x1 = seg.x0 + chosen_block.lead_in_mm, seg.x1 + chosen_block.lead_in_mm
                chosen_block.time_origin_mm += chosen_block.lead_in_mm
                chosen_block.notes = notes + chosen_block.notes + [
                    f"limb-lead closure residual {residuals[k]:.2f} ({chosen_block.template}, two blocks)"]
                return chosen_block, False
        if best is None:
            layout.segments = []
            layout.notes = notes + ["no standard layout fits the rows that look like an ECG"]
            return layout, False
        if best["dropped"]:
            notes.append(f"{best['dropped']} outer rows fit no arrangement that satisfies Einthoven and were ignored")
        chosen, upright, flipped = best["chosen"], best["upright"], best["flipped"]
        note = f"limb-lead closure residual {upright:.2f} ({chosen.template}); upside down {flipped:.2f}"
        # turn only on positive evidence: the limb leads close upside down and clearly not upright
        # (chest-only pages have nothing to close and are left as predicted)
        if allow_turn and np.isfinite(flipped) and flipped < cfg.turn_accept and (
                not np.isfinite(upright) or flipped < 0.6 * upright):
            # express the rows in the frame of the turned lattice and rebuild the chosen arrangement there
            width_mm, height_mm = lattice.extent_mm
            rows_turned = [(height_mm - b, width_mm - x1, width_mm - x0) for b, x0, x1 in reversed(best["rows"])]
            again = [h for h, _, _ in row_hypotheses(blank, 1.0, speed, cfg.layout, None, rows_turned)]
            match = [h for h in again if h.template == best["chosen_turned"].template]
            result = shift_columns(match[0] if match else again[0], best["offset_turned"])
            result.notes = notes + [note]
            return result, True
        if chosen.template in ("3x1-V1-V2-V3", "3x1-V4-V5-V6"):
            # chest leads only: nothing to close; tell the two halves apart by the R-wave progression
            half = chest_half_from_polarity(best["kept"])
            chosen = next((c for c in best["candidates"] if c.template == half), chosen)
            note += f"; chest half {half[4:]} assumed from the QRS polarity"
        shift_columns(chosen, best["offsets"][best["candidates"].index(chosen)])
        chosen.notes = notes + [n for n in chosen.notes if n not in notes] + [note]
        return chosen, False

    # ------------------------------------------------------------------ public
    def __call__(self, source, speed_mm_s: float | None = None, gain_mm_mv: float | None = None,
                 debug: bool = False) -> DigitizationResult:
        cfg = self.config
        speed = speed_mm_s or cfg.speed_mm_s
        gain = gain_mm_mv or cfg.gain_mm_mv
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        image = load_image(source)
        empty = np.full((N_LEADS, 0), np.nan, np.float32)

        def fail(reason: str, **extra) -> DigitizationResult:
            timings["total"] = time.perf_counter() - t0
            return DigitizationResult(False, empty, empty.copy(), cfg.output_fs, notes=[reason], timings=timings,
                                      quality=extra)

        out, scale, xy, lattice, error = self._detect_page(image)
        timings["page"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        if lattice is None:
            return fail(f"grid not recovered: {error}", grid_points=int(len(xy)))
        direction = out["direction"][0].cpu().numpy()
        lattice, direction = self._orient(out, lattice, direction)
        timings["lattice"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        layout, ink, origin = self._read_layout(out, lattice, speed)
        if not layout.segments or (layout.names_attached >= 4 and layout.name_offset_mm > 20.0):
            # names sit at the *start* of their segments; if they do not, try reading the page the other way
            flipped = lattice.oriented(-direction)
            layout2, ink2, origin2 = self._read_layout(out, flipped, speed)
            better = layout2.segments and (
                not layout.segments
                or (layout2.name_offset_mm < layout.name_offset_mm
                    and (layout2.reading == "names" or layout2.confirmed > layout.confirmed)))
            if better:
                lattice, layout, ink, origin = flipped, layout2, ink2, origin2
                layout.notes.append("page read in the direction opposite to the predicted one")
        if layout.segments and layout.reading == "rows" and (
                layout.alternatives or layout.extra_hypotheses or layout.confirmed == 0):
            # the names did not settle the arrangement: let physiology choose it, and the way up
            layout, upside_down = self._referee(image, lattice, scale, layout, speed, gain,
                                                allow_turn=layout.confirmed == 0)
            if upside_down:
                lattice = lattice.oriented(-direction)
                layout.notes.append("page turned by 180 degrees: only then do the limb leads satisfy Einthoven")
            layout.alternatives = []
        timings["layout"] = time.perf_counter() - t1
        if not layout.segments:
            return fail("no printed lead found", lattice=lattice.quality, notes=layout.notes)
        segments = [s for s in layout.segments if s.lead >= 0]
        if not segments:
            return fail("no segment could be named", lattice=lattice.quality)

        t1 = time.perf_counter()
        spans = [self._span_for(gain)] * len(segments)
        decoded = self._decode_segments(image, lattice, scale, segments, spans)
        retry = [k for k, d in enumerate(decoded) if d["clipped"] > 0.004 and d["span"] < cfg.span_mm[2]]
        if retry:
            tall = self._decode_segments(image, lattice, scale, [segments[k] for k in retry],
                                         [min(cfg.span_mm[2], 1.5 * decoded[k]["span"]) for k in retry])
            for k, d in zip(retry, tall):
                decoded[k] = d
        timings["trace"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        fs = cfg.output_fs
        duration = max(s.t0 + (s.x1 - s.x0) / speed for s in segments)
        n = int(round(duration * fs))
        observations: list[Observation] = []
        seg_samples: dict[str, list[tuple[int, int]]] = {}
        for seg, d in zip(segments, decoded):
            inside = (d["x_mm"] >= seg.x0) & (d["x_mm"] <= seg.x1)
            good = inside & (d["visible"] >= cfg.visibility_threshold)
            mv = -(d["y_mm"] - seg.baseline) / gain
            t = seg.t0 + (d["x_mm"] - seg.x0) / speed
            values = columns_to_signal(mv, t, fs, n, good)
            weight = columns_to_signal(d["confidence"] * d["visible"], t, fs, n, inside)
            weight = np.clip(np.nan_to_num(weight), 0.0, 1.0) * np.isfinite(values)
            observations.append(Observation(seg.lead, values, weight))
        # a rhythm strip without a readable name repeats one of the other leads: find which
        for k, seg in enumerate(segments):
            if seg.source != "assumed":
                continue
            known: dict[int, np.ndarray] = {}
            for other, ob in zip(segments, observations):
                if other.source != "assumed" and ob.lead not in known:
                    known[ob.lead] = ob.values
            found = identify_by_correlation(observations[k].values, known)
            if found is not None:
                seg.lead, seg.source = found[0], "correlation"
                observations[k] = Observation(found[0], observations[k].values, observations[k].weights)
            else:
                layout.notes.append(f"rhythm strip on row {seg.row} matches no lead; assumed {LEADS[seg.lead]}")
        seg_samples = {}
        for seg, ob in zip(segments, observations):
            fin = np.nonzero(np.isfinite(ob.values))[0]
            if len(fin):
                seg_samples.setdefault(LEADS[seg.lead], []).append((int(fin[0]), int(fin[-1])))
        observations = align_duplicates(observations, int(round(cfg.duplicate_align_s * fs)))
        observations, flipped_avr = resolve_avr_polarity(observations)
        signals, confidence = fuse_leads(observations, n, use_constraints=cfg.use_constraints,
                                         reconstruct_limb=cfg.reconstruct_limb)
        timings["fusion"] = time.perf_counter() - t1
        timings["total"] = time.perf_counter() - t0

        notes = list(layout.notes)
        if flipped_avr:
            notes.append("aVR is printed inverted (-aVR, Cabrera); polarity restored from the other limb leads")
        lead_conf = {LEADS[l]: float(np.mean(confidence[l][np.isfinite(signals[l])])) if np.isfinite(signals[l]).any() else 0.0
                     for l in range(N_LEADS)}
        quality = {
            "lattice": lattice.quality,
            "px_per_mm": float(lattice.spacing_px / lattice.mm_per_cell / scale),
            "leads_found": int(sum(np.isfinite(signals[l]).any() for l in range(N_LEADS))),
            "lead_confidence": lead_conf,
            "einthoven_residual": einthoven_residual(signals),
            "clipped_fraction": float(np.mean([d["clipped"] for d in decoded])),
            "speed_mm_s": speed, "gain_mm_mv": gain,
        }
        result = DigitizationResult(True, signals.astype(np.float32), confidence.astype(np.float32), fs,
                                    segments=segments, segment_samples=seg_samples,
                                    layout=layout.template or f"{len(layout.rows)} rows", quality=quality,
                                    notes=notes, timings=timings, lattice=lattice)
        if debug:
            result.debug = {"grid_xy": xy / scale, "ink": ink, "origin": origin, "decoded": decoded, "scale": scale,
                            "direction": direction}
        return result


def resolve_avr_polarity(observations: list[Observation]) -> tuple[list[Observation], bool]:
    """Detect a ``-aVR`` print (Cabrera) from its disagreement with the other limb leads.

    aVR equals ``-(I + II) / 2``; whenever two other limb leads overlap an aVR segment
    they predict it. If the prediction correlates strongly *negatively* with what was
    read, the page prints ``-aVR`` and the observation is flipped.
    """
    avr = [i for i, o in enumerate(observations) if o.lead == 3]
    others = [o for o in observations if o.lead < 6 and o.lead != 3]
    if not avr or len({o.lead for o in others}) < 2:
        return observations, False
    n = len(observations[avr[0]].values)
    a = LIMB_BASIS[[o.lead for o in others]]
    y = np.stack([np.nan_to_num(o.values) - np.nanmean(o.values) for o in others])
    w = np.stack([np.where(np.isfinite(o.values), o.weights, 0.0) for o in others])
    normal = np.einsum("kn,ki,kj->nij", w, a, a)
    det = normal[:, 0, 0] * normal[:, 1, 1] - normal[:, 0, 1] ** 2
    ok = det > 1e-3 * np.maximum(normal[:, 0, 0] * normal[:, 1, 1], 1e-12)
    if ok.sum() < 50:
        return observations, False
    rhs = np.einsum("kn,ki->ni", w * y, a)
    s = np.zeros((n, 2))
    s[ok, 0] = (normal[ok, 1, 1] * rhs[ok, 0] - normal[ok, 0, 1] * rhs[ok, 1]) / det[ok]
    s[ok, 1] = (normal[ok, 0, 0] * rhs[ok, 1] - normal[ok, 0, 1] * rhs[ok, 0]) / det[ok]
    predicted = s @ LIMB_BASIS[3]
    votes = []
    for i in avr:
        m = ok & np.isfinite(observations[i].values)
        if m.sum() >= 50 and predicted[m].std() > 1e-6 and observations[i].values[m].std() > 1e-6:
            votes.append(float(np.corrcoef(predicted[m], observations[i].values[m])[0, 1]))
    if not votes or np.mean(votes) > -0.5:
        return observations, False
    out = [Observation(o.lead, -o.values, o.weights) if o.lead == 3 else o for o in observations]
    return out, True
