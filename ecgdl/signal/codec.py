"""Column-wise trace codec.

A printed lead is a function ``y(x)``: one vertical position per image column. The
trace network therefore predicts, for every column, a categorical distribution over
rows instead of a binary mask. This is the formulation behind the strongest entries of
the PhysioNet/Kaggle 2025 digitization competition (column-wise 1D Gaussian heatmaps
decoded with a soft-argmax), because the codec itself is nearly lossless: encoding a
signal and decoding it back keeps sub-pixel positions.

Conventions
-----------
* ``y`` is measured in pixels from the top of the crop, at **pixel centres**
  (row 0 covers ``[-0.5, 0.5)``).
* Tensors are ``[B, H, W]`` for distributions/logits and ``[B, W]`` for positions.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F


def encode_columns(y: torch.Tensor, height: int, sigma: float = 1.0) -> torch.Tensor:
    """Encode positions ``y[B, W]`` as normalised per-column Gaussians ``[B, H, W]``.

    Positions outside the crop are clamped to the border row; mark those columns as
    invalid in the loss instead of trusting the clamped target.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rows = torch.arange(height, device=y.device, dtype=y.dtype)[None, :, None]
    centre = y.clamp(0, height - 1)[:, None, :]
    logits = -0.5 * ((rows - centre) / sigma) ** 2
    return torch.softmax(logits, dim=1)


def soft_argmax(logits: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable expectation of the row index. Returns ``(y[B, W], prob[B, H, W])``."""
    prob = torch.softmax(logits.float() / temperature, dim=1)
    rows = torch.arange(logits.shape[1], device=logits.device, dtype=prob.dtype)[None, :, None]
    return (prob * rows).sum(1), prob


def local_soft_argmax(prob: torch.Tensor, centre: torch.Tensor, radius: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Expectation restricted to ``centre ± radius`` rows.

    A global soft-argmax is the mean of the whole column distribution, so a second mode
    (another lead crossing the crop, a grid line, handwriting) drags it away from the
    trace. Restricting the expectation to a window around a robust mode keeps the
    sub-pixel precision without that bias.

    Returns ``(y[B, W], mass[B, W])`` where ``mass`` is the probability inside the window,
    a natural per-column confidence.
    """
    rows = torch.arange(prob.shape[1], device=prob.device, dtype=prob.dtype)[None, :, None]
    window = ((rows - centre[:, None, :].to(prob.dtype)).abs() <= radius).to(prob.dtype)
    weight = prob * window
    mass = weight.sum(1)
    y = (weight * rows).sum(1) / mass.clamp_min(1e-12)
    return y, mass


def viterbi_path(
    log_prob: np.ndarray,
    smooth: float = 0.35,
    jump: float = 6.0,
) -> np.ndarray:
    """Most probable smooth path through column distributions (classical dynamic programming).

    Minimises ``sum_x -log p(y_x | x) + min(smooth * |y_x - y_{x-1}|, jump)``.
    The truncated-linear transition lets the path follow a QRS complex (a large jump costs
    at most ``jump``) while refusing to hop between two parallel traces for a few columns.

    Parameters
    ----------
    log_prob : array ``[N, H, W]`` or ``[H, W]``
    smooth : cost per row of vertical movement between neighbouring columns (nats)
    jump : cap of the transition cost (nats)

    Returns
    -------
    Integer path ``[N, W]`` (or ``[W]``).
    """
    single = log_prob.ndim == 2
    cost = -np.asarray(log_prob, dtype=np.float32)
    if single:
        cost = cost[None]
    n, h, w = cost.shape
    # Only |dy| <= jump / smooth can beat the truncated cost, so a banded search is exact.
    band = int(min(h - 1, np.ceil(jump / max(smooth, 1e-6))))
    offsets = np.arange(-band, band + 1)
    penalty = (smooth * np.abs(offsets)).astype(np.float32)  # [K]
    index = np.arange(h)[:, None] + offsets[None, :]  # [H, K]
    inside = (index >= 0) & (index < h)
    index = np.clip(index, 0, h - 1)
    big = np.float32(1e9)
    penalty_hk = np.where(inside, penalty[None, :], big)  # [H, K]

    acc = cost[:, :, 0].copy()  # [N, H]
    back = np.zeros((n, h, w), dtype=np.int32)
    for x in range(1, w):
        cand = acc[:, index] + penalty_hk[None]  # [N, H, K]
        k = cand.argmin(-1)  # [N, H]
        local = np.take_along_axis(cand, k[..., None], -1)[..., 0]
        prev_local = np.take_along_axis(np.broadcast_to(index[None], cand.shape), k[..., None], -1)[..., 0]
        gmin_idx = acc.argmin(1)  # [N]
        gmin = acc[np.arange(n), gmin_idx] + jump
        use_jump = gmin[:, None] < local
        back[:, :, x] = np.where(use_jump, gmin_idx[:, None], prev_local)
        acc = np.where(use_jump, gmin[:, None], local) + cost[:, :, x]
    path = np.zeros((n, w), dtype=np.int64)
    path[:, -1] = acc.argmin(1)
    for x in range(w - 1, 0, -1):
        path[:, x - 1] = back[np.arange(n), path[:, x], x]
    return path[0] if single else path


def decode_columns(
    logits: torch.Tensor,
    method: str = "hybrid",
    radius: int = 8,
    smooth: float = 0.1,
    jump: float = 4.0,
    trust: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Decode trace-network logits ``[B, H, W]`` into sub-pixel positions.

    ``method``:
      * ``"softargmax"``: global expectation, exactly what the network is trained with;
      * ``"argmax"``: per-column mode followed by a local soft-argmax;
      * ``"viterbi"``: smooth dynamic-programming mode followed by a local soft-argmax;
      * ``"hybrid"`` (default): the global expectation wherever the column distribution is
        unimodal (at least ``trust`` of its mass lies within ``radius`` rows of the
        expectation), and the Viterbi answer in the remaining, ambiguous columns. On clean
        columns this is identical to the trained decoder; where two strokes compete, the
        expectation would land between them and the path through the neighbours decides.

    Returns ``y[B, W]`` (float, pixel centres) and ``confidence[B, W]`` in ``[0, 1]``.
    """
    if method not in ("softargmax", "argmax", "viterbi", "hybrid"):
        raise ValueError(f"unknown decode method: {method}")
    log_prob = F.log_softmax(logits.float(), dim=1)
    prob = log_prob.exp()
    rows = torch.arange(logits.shape[1], device=logits.device, dtype=prob.dtype)[None, :, None]
    y_soft = (prob * rows).sum(1)
    _, mass_soft = local_soft_argmax(prob, y_soft.round(), radius)
    if method == "softargmax":
        return {"y": y_soft, "confidence": mass_soft}
    if method == "hybrid" and bool((mass_soft >= trust).all()):
        return {"y": y_soft, "confidence": mass_soft}
    if method == "argmax":
        mode = prob.argmax(1)
    else:
        mode = torch.from_numpy(viterbi_path(log_prob.cpu().numpy(), smooth=smooth, jump=jump)).to(logits.device)
    y, mass = local_soft_argmax(prob, mode, radius)
    if method == "hybrid":
        clean = mass_soft >= trust
        return {"y": torch.where(clean, y_soft, y), "confidence": torch.where(clean, mass_soft, mass)}
    return {"y": y, "confidence": mass}
