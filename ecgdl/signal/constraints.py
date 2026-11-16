"""Physiological consistency by weighted least squares.

The six limb leads live in a two-dimensional space: with ``s(t) = (I(t), II(t))``,
every limb lead is ``a_l . s(t)`` (Einthoven and Goldberger relations). A page usually
prints more observations than unknowns: I, II, III share a time window, and so do aVR,
aVL, aVF, often together with a rhythm strip. Each printed segment also carries an
unknown vertical offset, because the 0 mV line is not marked on paper.

Instead of the common "average, then nudge" heuristics, this module solves the model

    y_k(t) = a_{l(k)} . s(t) + o_k + noise,    noise variance = 1 / w_k(t)

for all samples ``s(t)`` and all segment offsets ``o_k`` by alternating exact
least-squares steps. The per-sample confidence of the trace decoder becomes the weight,
so an occluded or ambiguous stretch of one lead is repaired by the other leads instead
of contaminating them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import LEADS, LIMB_BASIS, N_LEADS
from .resample import best_lag, shift_signal


@dataclass
class Observation:
    """One printed occurrence of a lead placed on the common output time grid."""

    lead: int  #: index in :data:`ecgdl.constants.LEADS`
    values: np.ndarray  #: ``[N]`` mV, NaN where not observed
    weights: np.ndarray  #: ``[N]`` non-negative confidence, 0 where not observed


def align_duplicates(observations: list[Observation], max_lag: int) -> list[Observation]:
    """Align repeated occurrences of the same lead to the longest one (integer lag).

    A short segment and the rhythm strip of the same lead show the same samples; a small
    error in the estimated start of a segment appears as a pure time shift, which the
    overlap measures directly.
    """
    by_lead: dict[int, list[int]] = {}
    for i, ob in enumerate(observations):
        by_lead.setdefault(ob.lead, []).append(i)
    out = list(observations)
    for idx in by_lead.values():
        if len(idx) < 2:
            continue
        ref = max(idx, key=lambda i: np.isfinite(observations[i].values).sum())
        for i in idx:
            if i == ref:
                continue
            lag = best_lag(observations[ref].values, observations[i].values, max_lag)
            if lag:
                w = shift_signal(observations[i].weights, lag)
                out[i] = Observation(observations[i].lead, shift_signal(observations[i].values, lag),
                                     np.nan_to_num(w, nan=0.0))
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.where(np.isfinite(values), weights, 0.0)
    total = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.nan_to_num(values) * w).sum(0) / total
    return np.where(total > 0, mean, np.nan), total


def fuse_leads(
    observations: list[Observation],
    n_samples: int,
    *,
    use_constraints: bool = True,
    reconstruct_limb: bool = False,
    iterations: int = 4,
    min_weight: float = 1e-3,
    conditioning: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse all printed occurrences into ``signals[12, N]`` and ``confidence[12, N]``.

    Parameters
    ----------
    use_constraints : project limb leads onto the Einthoven/Goldberger subspace.
    reconstruct_limb : also output limb leads at times where they are *not* printed but
        at least two other limb leads are (they are then fully determined).
    conditioning : minimum ratio between the smallest and largest eigenvalue of the
        per-sample 2x2 normal matrix for a sample to count as fully determined.
    """
    signals = np.full((N_LEADS, n_samples), np.nan)
    confidence = np.zeros((N_LEADS, n_samples))
    if not observations:
        return signals, confidence
    values = np.stack([np.asarray(o.values, dtype=np.float64) for o in observations])
    weights = np.stack([np.asarray(o.weights, dtype=np.float64) for o in observations])
    weights = np.where(np.isfinite(values) & (weights > min_weight), weights, 0.0)
    leads = np.array([o.lead for o in observations])

    # Plain confidence-weighted average of duplicates: the baseline for every lead.
    for l in range(N_LEADS):
        rows = np.nonzero(leads == l)[0]
        if len(rows):
            signals[l], confidence[l] = _weighted_mean(values[rows], weights[rows])

    limb = np.nonzero(leads < 6)[0]
    if not use_constraints or len({int(x) for x in leads[limb]}) < 3:
        return signals, confidence

    a = LIMB_BASIS[leads[limb]]  # [K, 2]
    y = np.nan_to_num(values[limb])  # [K, N]
    w = weights[limb]
    offsets = np.zeros(len(limb))
    outer = a[:, :, None] * a[:, None, :]  # [K, 2, 2]
    normal = np.einsum("kn,kij->nij", w, outer)  # [N, 2, 2]
    trace = normal[:, 0, 0] + normal[:, 1, 1]
    det = normal[:, 0, 0] * normal[:, 1, 1] - normal[:, 0, 1] * normal[:, 1, 0]
    disc = np.sqrt(np.maximum(trace**2 / 4 - det, 0.0))
    lam_min, lam_max = trace / 2 - disc, trace / 2 + disc
    determined = (lam_max > 0) & (lam_min > conditioning * lam_max)
    if determined.sum() < 8:
        return signals, confidence
    inv = np.zeros_like(normal)
    d = det[determined]
    inv[determined, 0, 0] = normal[determined, 1, 1] / d
    inv[determined, 1, 1] = normal[determined, 0, 0] / d
    inv[determined, 0, 1] = -normal[determined, 0, 1] / d
    inv[determined, 1, 0] = -normal[determined, 1, 0] / d

    s = np.zeros((n_samples, 2))
    for _ in range(max(1, iterations)):
        rhs = np.einsum("kn,ki->ni", w * (y - offsets[:, None]), a)  # [N, 2]
        s = np.einsum("nij,nj->ni", inv, rhs)
        model = a @ s.T  # [K, N]
        wd = w * determined[None, :]
        total = wd.sum(1)
        offsets = np.where(total > 0, (wd * (y - model)).sum(1) / np.maximum(total, 1e-12), 0.0)

    for l in range(6):
        rows = np.nonzero(leads[limb] == l)[0]
        fitted = LIMB_BASIS[l] @ s.T
        if len(rows):
            # Keep the lead on its own printed baseline: add back the mean offset of its segments.
            own_w = w[rows].sum(1)
            fitted = fitted + (offsets[rows] * own_w).sum() / max(own_w.sum(), 1e-12)
            printed = confidence[l] > 0
            use = determined & printed
        elif reconstruct_limb:
            use = determined
        else:
            continue
        signals[l] = np.where(use, fitted, signals[l])
        gain = np.einsum("i,nij,j->n", LIMB_BASIS[l], inv, LIMB_BASIS[l])  # variance of the fit
        with np.errstate(divide="ignore"):
            confidence[l] = np.where(use, 1.0 / np.maximum(gain, 1e-12), confidence[l])
    return signals, confidence


def einthoven_residual(signals: np.ndarray) -> float:
    """RMS of ``I - II + III`` relative to the RMS of the three leads (0 = consistent)."""
    i, ii, iii = (signals[LEADS.index(n)] for n in ("I", "II", "III"))
    m = np.isfinite(i) & np.isfinite(ii) & np.isfinite(iii)
    if m.sum() < 8:
        return float("nan")
    r = (i - ii + iii)[m]
    r = r - r.mean()
    ref = np.concatenate([x[m] - x[m].mean() for x in (i, ii, iii)])
    return float(np.sqrt(np.mean(r**2)) / max(np.sqrt(np.mean(ref**2)), 1e-9))
