"""Physiological referees for layout hypotheses.

When lead names cannot be read, several standard arrangements may fit the same rows (six
rows are a 6x2 page or a 3x4 page with three rhythm strips). The signals themselves can
tell: the six limb leads span only two dimensions (Einthoven, ``I - II + III = 0``, and
Goldberger, ``aVR + aVL + aVF = 0``), so any three of them printed over the same time
window determine each other. Only the right assignment of leads to rows and columns
makes that true.
"""
from __future__ import annotations

import numpy as np

from ..constants import LEAD_INDEX, LIMB_BASIS
from .assemble import Layout


def _predictability(leads: dict[int, np.ndarray], flip_avr: bool) -> list[float]:
    """Leave-one-out test of the limb leads of one time window.

    The six limb leads span a two-dimensional space, so with three or more of them each
    one is determined by the others. Every lead is predicted from the least-squares fit of
    the remaining ones and compared with what was read: ``1 - correlation`` averaged with
    the relative RMS difference (the correlation alone is fooled by neighbouring chest
    leads, which resemble each other up to a factor).
    """
    ids = sorted(lead for lead in leads if lead < 6)
    if len(ids) < 3:
        return []
    n = min(len(leads[i]) for i in ids)
    y = np.stack([leads[i][:n] for i in ids])
    ok = np.all(np.isfinite(y), axis=0)
    if ok.sum() < 100:
        return []
    y = y[:, ok] - y[:, ok].mean(1, keepdims=True)
    if flip_avr and LEAD_INDEX["aVR"] in ids:
        y[ids.index(LEAD_INDEX["aVR"])] *= -1
    a = LIMB_BASIS[ids]
    scores = []
    for k in range(len(ids)):
        rest = [j for j in range(len(ids)) if j != k]
        if np.linalg.matrix_rank(a[rest]) < 2:
            continue
        s, *_ = np.linalg.lstsq(a[rest], y[rest], rcond=None)
        pred, target = a[k] @ s, y[k]
        if pred.std() < 1e-9 or target.std() < 1e-9:
            scores.append(1.0)
            continue
        corr = float(np.corrcoef(pred, target)[0, 1])
        relative = float(np.sqrt(np.mean((pred - target) ** 2) / (pred.std() * target.std())))
        scores.append(0.5 * (1.0 - corr) + 0.5 * min(relative, 2.0))
    return scores


def closure_residual(layout: Layout, row_signals: dict[int, np.ndarray], fs: float, speed_mm_s: float,
                     offset_mm: float = 0.0) -> float:
    """How badly the limb-lead identities fail under a layout hypothesis.

    ``row_signals[r]`` is row ``r`` decoded as one strip, in mV, on a grid that starts at
    the page time origin. Each time window that shows at least three limb leads is tested
    (see :func:`_predictability`): 0 when the identities hold, about 1 or more for
    unrelated signals. The result is the mean over windows and leads, ``nan`` when the
    layout offers nothing to test (a page with chest leads only). An inverted aVR
    (Cabrera prints ``-aVR``) is accepted. ``offset_mm`` slides every column to the right.
    """
    windows: dict[int, dict[int, np.ndarray]] = {}
    for seg in layout.segments:
        strip = row_signals.get(seg.row)
        if strip is None:
            continue
        a = int(round((seg.t0 + offset_mm / speed_mm_s) * fs))
        b = int(round((seg.t0 + (offset_mm + seg.x1 - seg.x0) / speed_mm_s) * fs))
        cut = strip[max(a, 0) : b]
        if len(cut) > 100:
            windows.setdefault(a, {})[seg.lead] = cut
    scores: list[float] = []
    for leads in windows.values():
        plain = _predictability(leads, False)
        if plain and LEAD_INDEX["aVR"] in leads:
            inverted = _predictability(leads, True)
            if inverted and np.mean(inverted) < np.mean(plain):
                plain = inverted
        scores += plain
    return float(np.mean(scores)) if scores else float("nan")


#: column offsets tried by the referee (mm): the ink run of a row starts with its calibration pulse
OFFSETS_MM = tuple(np.arange(0.0, 30.1, 2.5))


def best_offset(layout: Layout, row_signals: dict[int, np.ndarray], fs: float, speed_mm_s: float) -> tuple[float, float]:
    """Closure residual of a hypothesis at its best column offset: ``(residual, offset_mm)``.

    The limb leads only close when the column boundaries are where the lead changes are, so
    the same test that verifies an arrangement also registers it in time. Arrangements with
    a single column are invariant to the offset; they get the measured lead-in instead.
    """
    per_row: dict[int, int] = {}
    for seg in layout.segments:
        per_row[seg.row] = per_row.get(seg.row, 0) + 1
    if max(per_row.values(), default=1) <= 1:  # full-width rows only: nothing to slide
        return closure_residual(layout, row_signals, fs, speed_mm_s, layout.lead_in_mm), layout.lead_in_mm
    scored = [(closure_residual(layout, row_signals, fs, speed_mm_s, d), d) for d in OFFSETS_MM]
    finite = [t for t in scored if np.isfinite(t[0])]
    return min(finite) if finite else (float("nan"), layout.lead_in_mm)


def choose_layout(candidates: list[Layout], row_signals: dict[int, np.ndarray], fs: float, speed_mm_s: float,
                  margin: float = 0.85, accept: float = 0.5,
                  cabrera_accept: float = 0.2) -> tuple[Layout, list[float], list[float]]:
    """Pick the hypothesis whose limb-lead identities close best.

    ``candidates`` are in order of prior preference; a later one replaces an earlier one
    only if its residual is clearly smaller (``< margin *`` the incumbent's). If no
    testable hypothesis closes (residual above ``accept``) and some hypothesis cannot be
    tested at all, the page probably shows chest leads only and the first untestable
    hypothesis is returned. Also returns the residual and the column offset of every candidate.
    """
    scored = [best_offset(c, row_signals, fs, speed_mm_s) for c in candidates]
    residuals, offsets = [r for r, _ in scored], [d for _, d in scored]
    # Cabrera order is rare and its columns mix leads in a way that neighbouring chest leads
    # can satisfy by chance: it must close convincingly to be believed without a name.
    eligible = [not ("cabrera" in (c.template or "") and r > cabrera_accept) for c, r in zip(candidates, residuals)]
    testable = [k for k, r in enumerate(residuals) if np.isfinite(r) and eligible[k]]
    untestable = [k for k, r in enumerate(residuals) if not np.isfinite(r)]
    if not testable:
        return candidates[0], residuals, offsets
    best = testable[0]
    for k in testable[1:]:
        if residuals[k] < margin * residuals[best]:
            best = k
    if residuals[best] > accept and untestable:
        return candidates[untestable[0]], residuals, offsets
    return candidates[best], residuals, offsets


def shift_columns(layout: Layout, offset_mm: float) -> Layout:
    """Slide every segment of a row-based layout to the right by ``offset_mm`` (times unchanged)."""
    for seg in layout.segments:
        seg.x0, seg.x1 = seg.x0 + offset_mm, seg.x1 + offset_mm
    layout.time_origin_mm += offset_mm
    return layout


def chest_half_from_polarity(row_signals: dict[int, np.ndarray]) -> str:
    """Guess whether three chest leads are V1-V3 or V4-V6 from the R-wave progression.

    The QRS is predominantly negative in V1-V2 and positive in V5-V6. This is a
    physiological prior, not a reading: it fails for some pathologies and is reported as
    an assumption.
    """
    negative = 0
    for strip in row_signals.values():
        ok = np.isfinite(strip)
        if ok.sum() < 100:
            continue
        x = strip[ok] - np.median(strip[ok])
        negative += abs(np.percentile(x, 0.5)) > 1.3 * np.percentile(x, 99.5)
    return "3x1-V1-V2-V3" if negative >= 2 else "3x1-V4-V5-V6"


def identify_by_correlation(values: np.ndarray, others: dict[int, np.ndarray], threshold: float = 0.8,
                            min_overlap: int = 250) -> tuple[int, float] | None:
    """Identify an unnamed strip as the lead it correlates with (rhythm strips repeat a lead).

    ``others[lead]`` are signals of known leads on the same time grid. Returns
    ``(lead, correlation)`` when the best match is above ``threshold``.
    """
    best: tuple[int, float] | None = None
    for lead, ref in others.items():
        m = np.isfinite(values) & np.isfinite(ref)
        if m.sum() < min_overlap or values[m].std() < 1e-6 or ref[m].std() < 1e-6:
            continue
        c = float(np.corrcoef(values[m], ref[m])[0, 1])
        if best is None or c > best[1]:
            best = (lead, c)
    return best if best is not None and best[1] >= threshold else None


def ecg_likeness(strip: np.ndarray, fs: float) -> float:
    """Score in ``[0, 1]``: does a decoded row look like an ECG rather than a border or text line?

    Three cues, all needed: enough of the row was read, it has millivolt-scale excursions,
    and those excursions are either spiky (QRS complexes give the derivative heavy tails)
    or periodic.
    """
    ok = np.isfinite(strip)
    if ok.mean() < 0.5 or ok.sum() < fs:
        return 0.0
    x = strip[ok] - np.median(strip[ok])
    swing = float(np.percentile(x, 99.5) - np.percentile(x, 0.5))
    if swing < 0.12:
        return 0.0
    d = np.diff(x)
    spread = d.std()
    kurt = float(np.mean((d - d.mean()) ** 4) / (spread**4 + 1e-12)) if spread > 0 else 0.0
    lo, hi = int(0.3 * fs), min(int(2.0 * fs), len(x) // 2)
    period = 0.0
    if hi > lo + 2:
        y = x - x.mean()
        full = np.correlate(y[::4], y[::4], "full")[len(y[::4]) - 1 :]
        period = float(full[lo // 4 : hi // 4].max() / (full[0] + 1e-12))
    spiky = np.clip((kurt - 4.0) / 8.0, 0.0, 1.0)
    return float(max(spiky, np.clip(period / 0.5, 0.0, 1.0)))
