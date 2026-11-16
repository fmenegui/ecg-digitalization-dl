"""Layout assembly: from detected lead names and trace ink to printed segments.

Competition pipelines register the page to one known template. Here the layout is read
from the page itself, in metric paper coordinates:

1. every detected lead name is attached to the trace row it labels (the mode of the ink
   distribution next to it is the row's baseline);
2. names on the same baseline form a row; the row's ink run gives its extent;
3. a segment starts at its name and ends at the next name of the row (or the row's end);
4. starts are pooled across rows into page columns, which makes boundaries consistent
   and reveals a name that was missed; a small library of standard arrangements then
   names the orphan segment.

Any number of rows, columns and rhythm strips is handled, in standard or Cabrera order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..constants import LEAD_INDEX


@dataclass
class NameDetection:
    lead: int
    x: float  #: mm
    y: float  #: mm
    score: float = 1.0


@dataclass
class Segment:
    lead: int  #: lead index, or -1 when the segment could not be named
    x0: float  #: mm, start of the printed window
    x1: float  #: mm, end of the printed window
    baseline: float  #: mm, estimated 0 mV line
    t0: float = 0.0  #: seconds from the page time origin
    row: int = 0
    score: float = 1.0
    #: how the lead identity was obtained: "name" (read from the page), "template" (position in a
    #: standard arrangement), "assumed" (default rhythm lead) or "correlation" (matched to another lead)
    source: str = "name"


@dataclass(frozen=True)
class LayoutConfig:
    ink_threshold: float = 0.3
    #: vertical search range for the baseline around a lead name (mm)
    name_reach_mm: float = 26.0
    #: horizontal window to the right of a name used to find its row (mm)
    name_window_mm: float = 45.0
    row_tolerance_mm: float = 4.0
    #: half height of the band used to measure where a row has ink (mm)
    band_mm: float = 7.0
    #: ink gaps shorter than this do not end a row (mm)
    gap_mm: float = 9.0
    column_tolerance_mm: float = 7.0
    min_segment_mm: float = 18.0
    #: assumed distance from the start of a segment to the centre of its name (mm), when unmeasurable
    default_name_offset_mm: float = 4.0
    #: seconds of signal per row on a standard page
    row_seconds: float = 10.0
    #: shortest stretch of ink that counts as a block of a row (mm)
    min_block_mm: float = 40.0


@dataclass
class Layout:
    segments: list[Segment]
    rows: list[float] = field(default_factory=list)  #: baselines (mm), top to bottom
    time_origin_mm: float = 0.0
    template: str | None = None
    #: median distance from the start of a row's ink to its first name (mm); large values
    #: mean the page is being read backwards
    name_offset_mm: float = 0.0
    notes: list[str] = field(default_factory=list)
    #: "names" when the arrangement was read from lead names, "rows" when it is a standard
    #: arrangement inferred from the row structure
    reading: str = "names"
    #: number of detected names that agree with a row-based arrangement
    confirmed: int = 0
    #: number of detected names that could be attached to a trace row
    names_attached: int = 0
    #: by how much the ink runs exceed the standard row duration (mm): room for a calibration pulse
    lead_in_mm: float = 0.0
    #: complete arrangements with their own rows (two-block pages), tested by the referee as well
    extra_hypotheses: list[Layout] = field(default_factory=list)
    #: ``(baseline, x0, x1)`` of the ink rows behind a row-based arrangement
    row_extents: list[tuple[float, float, float]] = field(default_factory=list)
    #: other standard arrangements that the names could not rule out (see ``ecgdl.layout.verify``)
    alternatives: list[Layout] = field(default_factory=list)


def _L(*names: str) -> list[int]:
    return [LEAD_INDEX[n] for n in names]


#: Standard arrangements (matrix rows only; rhythm strips are separate rows).
TEMPLATES: dict[str, list[list[int]]] = {
    "3x4": [_L("I", "aVR", "V1", "V4"), _L("II", "aVL", "V2", "V5"), _L("III", "aVF", "V3", "V6")],
    "6x2": [_L("I", "V1"), _L("II", "V2"), _L("III", "V3"), _L("aVR", "V4"), _L("aVL", "V5"), _L("aVF", "V6")],
    "12x1": [[i] for i in range(12)],
    "3x4-cabrera": [_L("aVL", "II", "V1", "V4"), _L("I", "aVF", "V2", "V5"), _L("aVR", "III", "V3", "V6")],
    "6x2-cabrera": [_L("aVL", "V1"), _L("I", "V2"), _L("aVR", "V3"), _L("II", "V4"), _L("aVF", "V5"), _L("III", "V6")],
    "12x1-cabrera": [[i] for i in _L("aVL", "I", "aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6")],
    # one page of an ECG printed over several pages (full-width strips)
    "6x1-limb": [[i] for i in _L("I", "II", "III", "aVR", "aVL", "aVF")],
    "6x1-chest": [[i] for i in _L("V1", "V2", "V3", "V4", "V5", "V6")],
    "3x1-I-II-III": [[i] for i in _L("I", "II", "III")],
    "3x1-aVR-aVL-aVF": [[i] for i in _L("aVR", "aVL", "aVF")],
    "3x1-V1-V2-V3": [[i] for i in _L("V1", "V2", "V3")],
    "3x1-V4-V5-V6": [[i] for i in _L("V4", "V5", "V6")],
}
#: arrangements that show only part of the 12 leads (one page of several)
PARTIAL = frozenset(name for name in TEMPLATES if name.startswith(("6x1", "3x1")))


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Runs of true values ``[start, end)`` after closing gaps of at most ``max_gap``."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return []
    breaks = np.nonzero(np.diff(idx) > max_gap)[0]
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def _baseline_candidates(ink: np.ndarray, ppmm: float, det: NameDetection, cfg: LayoutConfig) -> list[tuple[float, float]]:
    """Candidate baselines ``(y_mm, strength)`` of the row labelled by ``det``.

    The trace spends most of its time on the isoelectric line, so a row shows up as a peak
    of the ink profile next to the name. Several rows may be within reach; the page-level
    vote in :func:`_attach_names` picks the right one.
    """
    h, w = ink.shape
    x_lo = int(max(0, (det.x - 3) * ppmm))
    x_hi = int(min(w, (det.x + cfg.name_window_mm) * ppmm))
    y_lo = int(max(0, (det.y - cfg.name_reach_mm) * ppmm))
    y_hi = int(min(h, (det.y + cfg.name_reach_mm) * ppmm))
    if x_hi - x_lo < 4 or y_hi - y_lo < 4:
        return []
    profile = gaussian_filter1d(ink[y_lo:y_hi, x_lo:x_hi].sum(1), 0.8 * ppmm, mode="nearest")
    if profile.max() < 0.05 * (x_hi - x_lo):
        return []
    peak = (profile >= np.roll(profile, 1)) & (profile > np.roll(profile, -1)) & (profile >= 0.35 * profile.max())
    peak[0] = peak[-1] = False
    return [((k + y_lo + 0.5) / ppmm, float(profile[k] / (x_hi - x_lo))) for k in np.nonzero(peak)[0]]


def _attach_names(ink: np.ndarray, ppmm: float, names: list[NameDetection], cfg: LayoutConfig,
                  notes: list[str]) -> list[tuple[NameDetection, float]]:
    """Attach each name to a baseline with a page-level vote on the name-to-trace offset.

    A device prints its lead names at a fixed place relative to the trace (a little above,
    or below). A name between two rows is ambiguous on its own, but the correct offset is
    the one shared by the most names on the page.
    """
    cands = {i: _baseline_candidates(ink, ppmm, d, cfg) for i, d in enumerate(names)}
    remaining = {i for i, c in cands.items() if c}
    for i, c in cands.items():
        if not c:
            notes.append(f"name {names[i].lead} at ({names[i].x:.0f}, {names[i].y:.0f}) mm has no trace next to it")
    attached: list[tuple[NameDetection, float]] = []
    tol = 2.5
    for _ in range(4):
        if not remaining:
            break
        offsets = [(y - names[i].y, i) for i in remaining for y, _ in cands[i]]
        best_score, best_mode = -1.0, 0.0
        for d, _ in offsets:
            support = {i for o, i in offsets if abs(o - d) <= tol}
            score = len(support) - 0.01 * abs(d)
            if score > best_score:
                best_score, best_mode = score, d
        done = set()
        for i in remaining:
            near = [(abs(y - names[i].y - best_mode), y) for y, _ in cands[i] if abs(y - names[i].y - best_mode) <= tol]
            if near:
                attached.append((names[i], min(near)[1]))
                done.add(i)
        remaining -= done
    return attached


def _match_template(matrix: list[list[int]]) -> tuple[str, list[list[int]]] | None:
    """Best standard arrangement compatible with a partially known lead matrix."""
    best: tuple[int, str] | None = None
    for name, tpl in TEMPLATES.items():
        if len(tpl) != len(matrix) or any(len(a) != len(b) for a, b in zip(tpl, matrix)):
            continue
        known = [(a, b) for ra, rb in zip(tpl, matrix) for a, b in zip(ra, rb) if b >= 0]
        if not known or any(a != b for a, b in known):
            continue
        if best is None or len(known) > best[0]:
            best = (len(known), name)
    if best is None:
        return None
    return best[1], TEMPLATES[best[1]]


def find_rows(ink: np.ndarray, px_per_mm: float, cfg: LayoutConfig,
              x_range_mm: tuple[float, float] | None = None) -> list[tuple[float, float, float]]:
    """Trace rows found from ink alone: ``(baseline, x0, x1)`` in mm, top to bottom.

    ``x_range_mm`` restricts the search to a vertical band of the page (one column).
    """
    binary = ink >= cfg.ink_threshold
    shift = 0.0
    if x_range_mm is not None:
        a = int(max(0, round(x_range_mm[0] * px_per_mm)))
        b = int(min(binary.shape[1], round(x_range_mm[1] * px_per_mm)))
        binary, shift = binary[:, a:b], a / px_per_mm
        if binary.shape[1] < 8:
            return []
    profile = gaussian_filter1d(binary.sum(1).astype(float), 1.0 * px_per_mm, mode="nearest")
    if profile.max() < min(10 * px_per_mm, 0.1 * binary.shape[1]):
        return []
    sep = max(1, int(round(3.0 * px_per_mm)))
    peaks = [k for k in range(1, len(profile) - 1)
             if profile[k] >= 0.25 * profile.max() and profile[k] == profile[max(0, k - sep) : k + sep + 1].max()]
    band = max(1, int(round(cfg.band_mm * px_per_mm)))
    rows = []
    for k in peaks:
        occupied = binary[max(0, k - band) : k + band + 1].any(0)
        runs = _runs(occupied, int(round(cfg.gap_mm * px_per_mm)))
        if runs:
            # a row may be printed as several blocks separated by a gutter (and a calibration
            # pulse): its extent runs from the first to the last long stretch of ink
            longest = max(b - a for a, b in runs)
            long_runs = [r for r in runs if r[1] - r[0] >= min(longest, cfg.min_block_mm * px_per_mm)]
            a, b = long_runs[0][0], long_runs[-1][1]
            rows.append(((k + 0.5) / px_per_mm, a / px_per_mm + shift, b / px_per_mm + shift))
    return regular_rows(rows)


def split_column_hypotheses(ink: np.ndarray, px_per_mm: float, speed_mm_s: float, cfg: LayoutConfig) -> list[Layout]:
    """6x2 pages whose two columns are independent blocks with their own row spacing.

    Some devices print the limb leads and the chest leads as two side-by-side blocks, the
    chest block with wider rows. Row finding over the full width then sees twelve
    interleaved rows. Here each half of the page is searched on its own; if both show six
    regular rows, a 6x2 arrangement (standard and Cabrera) with per-column baselines is
    proposed. Each block is described as six full-width rows of its own column.
    """
    every = find_rows(ink, px_per_mm, cfg)
    if len(every) < 6:
        return []
    x0 = float(np.percentile([r[1] for r in every], 10))
    x1 = float(np.percentile([r[2] for r in every], 90))
    span = cfg.row_seconds * speed_mm_s
    if not 0.8 * span <= x1 - x0 <= 1.3 * span:
        return []
    lead_in = float(np.clip(x1 - x0 - span, 0.0, 30.0))
    half = span / 2
    margin = 0.12 * half
    left = find_rows(ink, px_per_mm, cfg, (x0 + lead_in + margin, x0 + lead_in + half - margin))
    right = find_rows(ink, px_per_mm, cfg, (x0 + lead_in + half + margin, x0 + lead_in + span - margin))
    if len(left) != 6 or len(right) != 6:
        return []
    if max(abs(a[0] - b[0]) for a, b in zip(left, right)) < cfg.row_tolerance_mm:
        return []  # the two halves share their rows: an ordinary 6x2 page
    out = []
    for name in ("6x2", "6x2-cabrera"):
        tpl = TEMPLATES[name]
        layout = Layout([])
        for r in range(6):
            layout.segments.append(Segment(tpl[r][0], x0, x0 + half, left[r][0], t0=0.0, row=r, score=0.0,
                                           source="template"))
            layout.segments.append(Segment(tpl[r][1], x0 + half, x0 + span, right[r][0], t0=half / speed_mm_s,
                                           row=6 + r, score=0.0, source="template"))
        layout.rows = [r[0] for r in left] + [r[0] for r in right]
        layout.row_extents = [(r[0], x0, x0 + half) for r in left] + [(r[0], x0 + half, x0 + span) for r in right]
        layout.time_origin_mm, layout.lead_in_mm, layout.reading = x0, lead_in, "rows"
        layout.template = name
        layout.notes.append("two-block page: each column has its own row spacing")
        out.append(layout)
    return out


def regular_rows(rows: list[tuple[float, float, float]], tolerance: float = 0.18) -> list[tuple[float, float, float]]:
    """Longest run of consecutive rows with a constant pitch.

    The leads of an ECG page are printed at equal vertical distances. A page border, a
    line of text or a ruler that was mistaken for a trace breaks that rhythm and is dropped.
    """
    if len(rows) < 4:
        return rows
    base = np.array([r[0] for r in rows])
    gaps = np.diff(base)
    support = [(np.abs(gaps - g) <= tolerance * g).sum() for g in gaps]
    pitch = float(np.median(gaps[np.abs(gaps - gaps[int(np.argmax(support))]) <= tolerance * gaps[int(np.argmax(support))]]))
    best, start = (0, 1), 0
    for k in range(len(gaps) + 1):
        if k == len(gaps) or abs(gaps[k] - pitch) > tolerance * pitch:
            if k + 1 - start > best[1] - best[0]:
                best = (start, k + 1)
            start = k + 1
    return rows[best[0] : best[1]] if best[1] - best[0] >= 3 else rows


def _candidates(n_rows: int) -> list[tuple[str, int]]:
    """Standard arrangements compatible with a row count: ``(template, rhythm strips)``."""
    out = []
    for name, tpl in TEMPLATES.items():
        extra = n_rows - len(tpl)
        if 0 <= extra <= (0 if name in PARTIAL else 3):
            out.append((name, extra))
    return out


def row_hypotheses(
    ink: np.ndarray,
    px_per_mm: float,
    speed_mm_s: float,
    cfg: LayoutConfig,
    attached: list[tuple[NameDetection, float]] | None = None,
    rows: list[tuple[float, float, float]] | None = None,
) -> list[tuple[Layout, int, int]]:
    """Standard layouts compatible with the row structure, best first.

    Rows are found from the ink alone (or given as ``(baseline, x0, x1)`` in mm). Every
    standard arrangement with that many rows (3x4, 6x2, 12x1, standard or Cabrera, plus up
    to three rhythm strips) predicts which lead is printed where; the detected names
    vote. Each hypothesis comes with its number of confirmed and contradicted names.
    """
    rows = find_rows(ink, px_per_mm, cfg) if rows is None else rows
    if not rows:
        return []
    attached = attached or []
    baselines = np.array([r[0] for r in rows])
    placed: list[tuple[int, float, int]] = []  # (row, x, lead)
    for det, base in attached:
        r = int(np.abs(baselines - base).argmin())
        if abs(baselines[r] - base) <= cfg.row_tolerance_mm:
            placed.append((r, det.x, det.lead))
    x0 = float(np.median([r[1] for r in rows]))
    x1 = float(np.median([r[2] for r in rows]))
    # A row shows cfg.row_seconds of signal. Its ink run is usually a little longer (calibration
    # pulse in front, sometimes a frame line behind), so columns are laid out from the standard
    # duration, not by dividing the run; the referee then slides them into place.
    span = cfg.row_seconds * speed_mm_s
    if not 0.8 * span <= x1 - x0 <= 1.3 * span:
        span = x1 - x0  # not a standard-duration page: fall back to the ink run
    lead_in = float(np.clip(x1 - x0 - span, 0.0, 30.0))
    default_rhythm = _L("II", "V1", "V5")
    out: list[tuple[float, Layout, int, int]] = []
    for name, extra in _candidates(len(rows)):
        tpl = TEMPLATES[name]
        cols = len(tpl[0])
        width = span / cols
        good = bad = 0
        for r, x, lead in placed:
            if r >= len(tpl):
                continue  # rhythm strips may show any lead
            c = int(np.clip((x - x0 - lead_in + 0.25 * width) // width, 0, cols - 1))
            good, bad = good + (tpl[r][c] == lead), bad + (tpl[r][c] != lead)
        layout = Layout([])
        for r, (baseline, _, _) in enumerate(rows):
            if r < len(tpl):
                leads, source = tpl[r], "template"
            else:
                seen = [lead for rr, _, lead in placed if rr == r]
                leads = [seen[0] if seen else default_rhythm[r - len(tpl)]]
                source = "name" if seen else "assumed"
            w = span / len(leads)
            for c, lead in enumerate(leads):
                layout.segments.append(Segment(lead, x0 + c * w, x0 + (c + 1) * w, baseline, t0=c * w / speed_mm_s,
                                               row=r, score=0.0, source=source))
        layout.rows = [r[0] for r in rows]
        layout.row_extents = list(rows)
        layout.time_origin_mm = x0
        layout.lead_in_mm = lead_in
        layout.reading = "rows"
        layout.template = name + (f"+{extra}R" if extra else "")
        if good == 0:
            layout.notes.append("lead names not found: standard layout assumed from the row structure")
        else:
            layout.notes.append(f"layout inferred from the row structure, confirmed by {good} lead names"
                                + (f" (contradicted by {bad})" if bad else ""))
        # names decide; without them prefer the standard order and the fewest rhythm strips
        # then complete pages before single pages of a multi-page print
        prior = 0.01 * ("cabrera" in name) + 0.001 * extra + 0.005 * (name in PARTIAL)
        out.append((good - bad - prior, layout, good, bad))
    out.sort(key=lambda t: -t[0])
    return [(layout, good, bad) for _, layout, good, bad in out]


def layout_from_rows(
    ink: np.ndarray,
    px_per_mm: float,
    speed_mm_s: float,
    cfg: LayoutConfig,
    attached: list[tuple[NameDetection, float]] | None = None,
) -> tuple[Layout, int, int]:
    """Best hypothesis of :func:`row_hypotheses`; an empty layout when no standard one fits."""
    found = row_hypotheses(ink, px_per_mm, speed_mm_s, cfg, attached)
    if not found:
        # rows exist but their number fits no standard arrangement (often a page border or a
        # line of text mistaken for a trace): hand the bare rows to the referee, which
        # discards the ones that do not look like an ECG once they are decoded
        rows = find_rows(ink, px_per_mm, cfg)
        bare = Layout([Segment(-1, x0, x1, b, row=r, score=0.0, source="template") for r, (b, x0, x1) in enumerate(rows)])
        bare.rows, bare.row_extents, bare.reading = [r[0] for r in rows], rows, "rows"
        if rows:
            bare.time_origin_mm = float(np.median([r[1] for r in rows]))
        bare.notes.append(f"{len(rows)} rows match no standard layout")
        return bare, 0, 0
    layout, good, bad = found[0]
    # hypotheses the names cannot tell apart are kept for a physiological check
    layout.alternatives = [alt for alt, g, b in found[1:] if g - b == good - bad]
    return layout, good, bad


def assemble_layout(
    ink: np.ndarray,
    px_per_mm: float,
    names: list[NameDetection],
    speed_mm_s: float,
    config: LayoutConfig | None = None,
) -> Layout:
    """Build the list of printed segments from two complementary readings.

    * **names first** (:func:`layout_from_names`): any arrangement, as long as the names are read;
    * **rows first** (:func:`layout_from_rows`): a standard arrangement that the row count
      allows and the detected names confirm, which survives missed or unreadable names.

    A complete name reading wins. Otherwise the row hypothesis is used when names confirm
    it without contradiction (or when there are no names at all).
    """
    cfg = config or LayoutConfig()
    notes: list[str] = []
    attached = _attach_names(ink, px_per_mm, names, cfg, notes)
    by_names = layout_from_names(ink, px_per_mm, attached, speed_mm_s, cfg) if attached else Layout([])
    by_names.notes = notes + by_names.notes
    by_names.names_attached = len(attached)
    complete = bool(by_names.segments) and all(s.source == "name" for s in by_names.segments)
    if complete and (by_names.template is not None or len({s.lead for s in by_names.segments}) >= 12):
        return by_names  # every segment was read and the result is a coherent page
    by_rows, good, bad = layout_from_rows(ink, px_per_mm, speed_mm_s, cfg, attached)
    if not by_rows.segments:
        return by_names
    by_rows.name_offset_mm = by_names.name_offset_mm
    by_rows.names_attached = len(attached)
    for alt in by_rows.alternatives:
        alt.name_offset_mm, alt.names_attached = by_names.name_offset_mm, len(attached)
    if attached and not ((good >= 1 and bad == 0) or (good >= 4 and good >= 3 * bad)):
        # the names that were read contradict every standard arrangement: do not let them vote.
        # All hypotheses stay open for the physiological referee (ecgdl.layout.verify).
        hypotheses = [h for h, _, _ in row_hypotheses(ink, px_per_mm, speed_mm_s, cfg, None)]
        if hypotheses:
            by_rows = hypotheses[0]
            by_rows.alternatives = hypotheses[1:]
            by_rows.name_offset_mm, by_rows.names_attached = by_names.name_offset_mm, len(attached)
        if any(s.lead < 0 for s in by_rows.segments) and by_names.segments and all(s.lead >= 0 for s in by_names.segments):
            return by_names  # no standard arrangement fits the rows either: keep what the names say
        by_rows.notes = notes + ["lead names unreliable: standard layouts tested against Einthoven instead"]
        by_rows.confirmed = 0
        by_rows.extra_hypotheses = split_column_hypotheses(ink, px_per_mm, speed_mm_s, cfg)
        return by_rows
    by_rows.notes = notes + by_rows.notes
    by_rows.confirmed = good
    if good == 0:
        by_rows.extra_hypotheses = split_column_hypotheses(ink, px_per_mm, speed_mm_s, cfg)
    return by_rows


def layout_from_names(
    ink: np.ndarray,
    px_per_mm: float,
    attached: list[tuple[NameDetection, float]],
    speed_mm_s: float,
    cfg: LayoutConfig,
) -> Layout:
    """Layout read from the lead names (see the module docstring for the steps).

    Parameters
    ----------
    ink : ``[H, W]`` trace-ink probability resampled on the upright paper frame
    px_per_mm : resolution of ``ink``
    attached : lead-name detections in paper millimetres with the baseline each one labels
    speed_mm_s : paper speed, only used to convert positions to seconds
    """
    layout = Layout([])
    h, w = ink.shape
    binary = ink >= cfg.ink_threshold

    # 2. rows = clusters of baselines
    attached.sort(key=lambda t: t[1])
    rows: list[list[tuple[NameDetection, float]]] = [[attached[0]]]
    for item in attached[1:]:
        if item[1] - np.median([b for _, b in rows[-1]]) <= cfg.row_tolerance_mm:
            rows[-1].append(item)
        else:
            rows.append([item])

    # 3. per row: ink runs and raw segment starts
    band = max(1, int(round(cfg.band_mm * px_per_mm)))
    raw: list[dict] = []
    offsets: list[float] = []
    for r, members in enumerate(rows):
        baseline = float(np.median([b for _, b in members]))
        yc = int(round(baseline * px_per_mm))
        occupied = binary[max(0, yc - band) : min(h, yc + band + 1)].any(0)
        runs = _runs(occupied, int(round(cfg.gap_mm * px_per_mm)))
        members = sorted(members, key=lambda t: t[0].x)
        # one name per position: drop duplicates of the same place (keep the strongest)
        dedup: list[tuple[NameDetection, float]] = []
        for m in members:
            if dedup and abs(m[0].x - dedup[-1][0].x) < cfg.min_segment_mm / 2:
                if m[0].score > dedup[-1][0].score:
                    dedup[-1] = m
                continue
            dedup.append(m)
        for a, b in runs:
            xa, xb = a / px_per_mm, b / px_per_mm
            inside = [m for m in dedup if xa - 12 <= m[0].x <= xb]
            if not inside:
                continue
            offsets.append(inside[0][0].x - xa)
            raw.append({"row": r, "baseline": baseline, "x0": xa, "x1": xb, "names": inside})
    if not raw:
        layout.notes.append("lead names found, but no trace ink on their rows")
        return layout
    layout.name_offset_mm = float(np.median(offsets)) if offsets else 0.0
    offset = float(np.clip(np.median(offsets), 0.0, 12.0)) if offsets else cfg.default_name_offset_mm

    segments: list[Segment] = []
    for part in raw:
        dets = [m[0] for m in part["names"]]
        starts = [max(part["x0"], d.x - offset) for d in dets]
        if starts[0] - part["x0"] >= cfg.min_segment_mm:
            # ink before the first name of the run: a segment whose name was not detected
            segments.append(Segment(-1, part["x0"], starts[0], part["baseline"], row=part["row"], score=0.0,
                                    source="template"))
        else:
            starts[0] = part["x0"]
        ends = starts[1:] + [part["x1"]]
        for d, s, e in zip(dets, starts, ends):
            if e - s >= cfg.min_segment_mm:
                segments.append(Segment(d.lead, s, e, part["baseline"], row=part["row"], score=d.score))
    if not segments:
        layout.notes.append("no printed segment long enough to digitize")
        return layout

    # 4. page columns: pool starts across rows, snap, and split where a name was missed
    starts_all = np.array(sorted(s.x0 for s in segments))
    clusters: list[list[float]] = [[starts_all[0]]]
    for x in starts_all[1:]:
        if x - np.mean(clusters[-1]) <= cfg.column_tolerance_mm:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    columns = [(float(np.median(c)), len(c)) for c in clusters]
    for seg in segments:
        centre, _ = min(columns, key=lambda c: abs(c[0] - seg.x0))
        if abs(centre - seg.x0) <= cfg.column_tolerance_mm:
            seg.x0 = centre
    by_row: dict[int, list[Segment]] = {}
    for seg in segments:
        by_row.setdefault(seg.row, []).append(seg)
    widest = max(len(v) for v in by_row.values())
    for r, segs in by_row.items():
        segs.sort(key=lambda s: s.x0)
        if len(segs) >= widest:
            continue
        elsewhere = {o.lead for q, other in by_row.items() if q != r for o in other}
        if len(segs) == 1 and segs[0].lead in elsewhere:
            continue  # a lone lead that is also printed elsewhere is a rhythm strip, not a row with missed names
        added: list[Segment] = []
        for centre, support in columns:
            if support < 2:
                continue
            for seg in segs:
                inside = seg.x0 + cfg.min_segment_mm < centre < seg.x1 - cfg.min_segment_mm
                taken = any(abs(o.x0 - centre) <= cfg.column_tolerance_mm for o in segs + added)
                if inside and not taken:
                    added.append(Segment(-1, centre, seg.x1, seg.baseline, row=r, score=0.0, source="template"))
        if added and len(segs) + len(added) <= widest:
            segs.extend(added)
            segs.sort(key=lambda s: s.x0)
    for segs in by_row.values():
        for a, b in zip(segs[:-1], segs[1:]):
            a.x1 = min(a.x1, b.x0)

    # 5. recognise a standard arrangement; it names the segments whose name was missed
    ordered_rows = sorted(by_row)
    matrix_rows = [r for r in ordered_rows if len(by_row[r]) == widest]
    matrix = [[s.lead for s in by_row[r]] for r in matrix_rows]
    match = _match_template(matrix)
    if match is not None:
        strips = len(ordered_rows) - len(matrix_rows)
        layout.template = match[0] + (f"+{strips}R" if strips else "")
        for r, leads in zip(matrix_rows, match[1]):
            for seg, lead in zip(by_row[r], leads):
                if seg.lead < 0:
                    seg.lead = lead
    if any(s.lead < 0 for segs in by_row.values() for s in segs):
        layout.notes.append("a segment without a name could not be identified from a standard layout")

    # 6. time origin: the same paper x is the same instant on every row
    segments = [s for r in ordered_rows for s in by_row[r]]
    origin = float(np.median([by_row[r][0].x0 for r in ordered_rows]))
    for seg in segments:
        seg.t0 = max(0.0, (seg.x0 - origin) / speed_mm_s)
    layout.segments = segments
    layout.rows = [float(by_row[r][0].baseline) for r in ordered_rows]
    layout.time_origin_mm = origin
    return layout
