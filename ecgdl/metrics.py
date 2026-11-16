"""Evaluation metrics used by the public benchmarks.

Both protocols are *reimplementations* from the public descriptions. They are meant for
development and for comparing versions under identical rules; an official leaderboard
may differ in details.
"""
from __future__ import annotations

import numpy as np

from .constants import LEADS


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """``10 log10(sum ref^2 / sum (ref - est)^2)`` over samples finite in both."""
    m = np.isfinite(reference) & np.isfinite(estimate)
    if not m.any():
        return float("nan")
    e = reference[m] - estimate[m]
    return float(10 * np.log10(max(float(reference[m] @ reference[m]), 1e-12) / max(float(e @ e), 1e-12)))


def _shift_windows(pred: np.ndarray, idx: np.ndarray, max_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """``pred[idx + s]`` for every ``s`` in ``[-max_shift, max_shift]``: ``[S, len(idx)]`` + validity."""
    shifts = np.arange(-max_shift, max_shift + 1)
    j = idx[None, :] + shifts[:, None]
    ok = (j >= 0) & (j < len(pred))
    vals = pred[np.clip(j, 0, len(pred) - 1)]
    ok &= np.isfinite(vals)
    return np.where(ok, vals, 0.0), ok


def aligned_lead_error(
    gt: np.ndarray,
    pred: np.ndarray,
    max_shift: int,
    min_coverage: float = 0.8,
) -> tuple[float, float, int] | None:
    """Best ``(noise_power, signal_power, shift)`` over integer shifts with the mean offset removed.

    Samples of ``gt`` that are NaN are not part of the comparison (lead not printed).
    Prediction samples that are missing count as zero, as in the competition.
    """
    idx = np.nonzero(np.isfinite(gt))[0]
    if len(idx) < 10:
        return None
    g = gt[idx]
    pred0 = np.nan_to_num(np.asarray(pred, dtype=np.float64), nan=0.0)
    vals, inside = _shift_windows(pred0, idx, max_shift)
    n_in = inside.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        offset = ((g[None] - vals) * inside).sum(1) / np.maximum(n_in, 1)
    err = (g[None] - vals - offset[:, None]) * inside
    noise = (err**2).sum(1)
    signal = ((g[None] ** 2) * inside).sum(1)
    noise = np.where(n_in >= min_coverage * len(idx), noise, np.inf)
    k = int(noise.argmin())
    if not np.isfinite(noise[k]):
        return None
    return float(noise[k]), float(signal[k]), int(k - max_shift)


def kaggle_snr(
    gt: np.ndarray,
    pred: np.ndarray,
    fs: float,
    max_shift_s: float = 0.2,
    align: bool = True,
) -> tuple[float, dict[str, float]]:
    """SNR of one record in the style of the PhysioNet/Kaggle 2025 competition.

    Per lead: time alignment by up to ``max_shift_s`` and removal of the constant offset;
    then the signal and noise powers of the 12 leads are summed and converted to dB.

    ``gt`` and ``pred`` are ``[12, n]`` at the ground-truth rate; ``gt`` is NaN where the
    paper does not show the lead. Returns ``(snr_db, per_lead_snr_db)``.
    """
    ms = int(round(max_shift_s * fs)) if align else 0
    sig = noi = 0.0
    per: dict[str, float] = {}
    for i, name in enumerate(LEADS):
        if align:
            r = aligned_lead_error(np.asarray(gt[i], float), np.asarray(pred[i], float), ms)
        else:
            m = np.isfinite(gt[i])
            if m.sum() < 10:
                r = None
            else:
                p = np.nan_to_num(np.asarray(pred[i], float))[m]
                r = (float(((gt[i][m] - p) ** 2).sum()), float((gt[i][m] ** 2).sum()), 0)
        if r is None:
            continue
        noi += r[0]
        sig += r[1]
        per[name] = float(10 * np.log10(max(r[1], 1e-12) / max(r[0], 1e-12)))
    if sig == 0:
        return float("nan"), per
    return float(10 * np.log10(sig / max(noi, 1e-12))), per


def aggregate_snr(per_record_db: list[float], mode: str = "mean_db") -> float:
    """Aggregate per-record SNRs. ``mean_db`` averages decibels; ``db_of_mean`` averages ratios."""
    v = np.asarray([x for x in per_record_db if np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return float("nan")
    if mode == "mean_db":
        return float(v.mean())
    if mode == "db_of_mean":
        return float(10 * np.log10(np.mean(10 ** (v / 10))))
    raise ValueError(f"unknown aggregation: {mode}")


def pm_lead_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float] | None:
    """PCC, RMSE (mV) and SNR (dB) of one aligned lead, PM-ECG-ID style.

    The vertical offset is removed with the median of the difference. A lead with less
    than half of the reference covered is a failure (``None``).
    """
    m = np.isfinite(pred) & np.isfinite(gt)
    if m.sum() < 0.5 * np.isfinite(gt).sum() or m.sum() < 8:
        return None
    g, p = gt[m].astype(np.float64), pred[m].astype(np.float64)
    p = p + np.median(g - p)
    e = g - p
    snr = 10 * np.log10(max(float(g @ g), 1e-12) / max(float(e @ e), 1e-12))
    pcc = float(np.corrcoef(g, p)[0, 1]) if g.std() > 1e-9 and p.std() > 1e-9 else 0.0
    return {"snr": float(snr), "pcc": pcc, "rmse": float(np.sqrt(np.mean(e * e)))}


def pm_best_shift(gt: np.ndarray, pred: np.ndarray, start: int, max_shift: int, span: int = 0) -> np.ndarray | None:
    """Cut of ``pred`` best correlated with ``gt`` within ``start ± max_shift`` (plus ``span``)."""
    n = len(gt)
    best: tuple[float, np.ndarray] | None = None

    def cut(s: int) -> np.ndarray | None:
        i0 = start + s
        lo, hi = max(0, i0), min(len(pred), i0 + n)
        if hi - lo < n // 2:
            return None
        p = np.full(n, np.nan)
        p[lo - i0 : hi - i0] = pred[lo:hi]
        return p

    def score(p: np.ndarray) -> float | None:
        m = np.isfinite(p) & np.isfinite(gt)
        if m.sum() < n // 2 or p[m].std() < 1e-9 or gt[m].std() < 1e-9:
            return None
        return float(np.corrcoef(gt[m], p[m])[0, 1])

    coarse = list(range(-max_shift, max_shift + max(0, span) + 1, 2))
    best_s = None
    for s in coarse:
        p = cut(s)
        c = None if p is None else score(p)
        if c is not None and (best is None or c > best[0]):
            best, best_s = (c, p), s
    if best is None:
        return None
    for s in (best_s - 1, best_s + 1):
        p = cut(s)
        c = None if p is None else score(p)
        if c is not None and c > best[0]:
            best = (c, p)
    return best[1]
