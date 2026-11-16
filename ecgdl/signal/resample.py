"""Conversions between sampled signals (mV at ``fs``) and image columns (pixels)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def signal_to_columns(
    signal: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
    n_cols: int,
    oversample: int = 4,
) -> np.ndarray:
    """Average a sampled signal over the time span of each of ``n_cols`` image columns.

    Column ``j`` covers ``[t_start + j*dt, t_start + (j+1)*dt)`` with
    ``dt = (t_end - t_start) / n_cols``. The signal is linearly interpolated and
    box-averaged, which is the anti-aliased value a printer/camera integrates.
    Columns outside the recording are NaN.
    """
    signal = np.asarray(signal, dtype=np.float64)
    dt = (t_end - t_start) / n_cols
    sub = (np.arange(oversample) + 0.5) / oversample
    t = t_start + (np.arange(n_cols)[:, None] + sub[None, :]) * dt  # [W, K]
    pos = t * fs
    inside = (pos >= 0) & (pos <= len(signal) - 1)
    values = np.interp(pos.ravel(), np.arange(len(signal)), signal).reshape(pos.shape)
    count = inside.sum(1)
    out = np.where(inside, values, 0.0).sum(1) / np.maximum(count, 1)
    out[count == 0] = np.nan
    return out


def columns_to_signal(
    values: np.ndarray,
    t_cols: np.ndarray,
    fs: float,
    n: int,
    valid: np.ndarray | None = None,
    max_gap_s: float = 0.06,
) -> np.ndarray:
    """Resample a per-column series onto the grid ``arange(n) / fs`` with a cubic spline.

    Samples outside the observed time span, or inside a gap of invalid columns longer
    than ``max_gap_s``, are NaN: unknown stays unknown rather than being invented.
    """
    values = np.asarray(values, dtype=np.float64)
    t_cols = np.asarray(t_cols, dtype=np.float64)
    ok = np.isfinite(values) & np.isfinite(t_cols)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    out = np.full(n, np.nan)
    if ok.sum() < 4:
        return out
    tc, vc = t_cols[ok], values[ok]
    order = np.argsort(tc)
    tc, vc = tc[order], vc[order]
    keep = np.concatenate(([True], np.diff(tc) > 1e-9))
    tc, vc = tc[keep], vc[keep]
    if len(tc) < 4:
        return out
    t = np.arange(n) / fs
    half = 0.5 * float(np.median(np.diff(tc)))
    inside = (t >= tc[0] - half) & (t <= tc[-1] + half)
    spline = CubicSpline(tc, vc, bc_type="natural", extrapolate=True)
    out[inside] = spline(t[inside])
    # Blank the samples that fall inside long runs of invalid columns.
    gaps = np.nonzero(np.diff(tc) > max_gap_s)[0]
    for g in gaps:
        out[(t > tc[g] + half) & (t < tc[g + 1] - half)] = np.nan
    return out


def best_lag(reference: np.ndarray, moving: np.ndarray, max_lag: int) -> int:
    """Integer lag ``s`` maximising the correlation of ``reference[t]`` with ``moving[t + s]``."""
    reference = np.asarray(reference, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    best, best_score = 0, -np.inf
    n = len(reference)
    for s in range(-max_lag, max_lag + 1):
        lo, hi = max(0, -s), min(n, len(moving) - s)
        if hi - lo < 16:
            continue
        a, b = reference[lo:hi], moving[lo + s : hi + s]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 16:
            continue
        a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
        denom = np.sqrt((a @ a) * (b @ b))
        score = (a @ b) / denom if denom > 1e-12 else -np.inf
        if score > best_score:
            best, best_score = s, score
    return best


def shift_signal(x: np.ndarray, lag: int) -> np.ndarray:
    """Return ``y`` with ``y[t] = x[t + lag]``, NaN-padded."""
    y = np.full_like(np.asarray(x, dtype=np.float64), np.nan)
    n = len(y)
    if lag >= 0:
        y[: n - lag] = x[lag:]
    else:
        y[-lag:] = x[: n + lag]
    return y
