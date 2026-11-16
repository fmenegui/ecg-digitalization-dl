"""Training objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .signal.codec import encode_columns


def quality_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for soft keypoint heatmaps (Li et al., *Generalized Focal Loss*, 2020).

    ``|target - p|^gamma * BCE(p, target)`` summed and normalised by the target mass, so a
    page with thousands of intersections and a crop with ten are weighted alike.
    """
    logits, target = logits.float(), target.float()
    p = torch.sigmoid(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none") * (target - p).abs().pow(gamma)
    return loss.sum() / target.sum().clamp_min(1.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits.float())
    target = target.float()
    inter = (p * target).sum((-2, -1))
    return (1 - (2 * inter + 1) / (p.sum((-2, -1)) + target.sum((-2, -1)) + 1)).mean()


def page_loss(pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
              weights: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
    """Multi-task PageNet loss. Returns each term and their weighted sum under ``"loss"``."""
    w = {"grid": 1.0, "offset": 1.0, "trace": 1.0, "names": 1.0, "direction": 0.5} | (weights or {})
    out = {
        "grid": quality_focal_loss(pred["grid"], batch["grid"]),
        "names": quality_focal_loss(pred["names"], batch["names"]),
        "trace": F.binary_cross_entropy_with_logits(pred["trace"].float(), batch["trace"].float())
        + dice_loss(pred["trace"], batch["trace"]),
        "direction": (1 - (pred["direction"].float() * batch["direction"].float()).sum(-1)).mean(),
    }
    ov = batch["offset_valid"].float()
    out["offset"] = (F.smooth_l1_loss(pred["offset"].float(), batch["offset"].float(), reduction="none", beta=0.1)
                     * ov).sum() / ov.sum().clamp_min(1.0)
    out["loss"] = sum(w[k] * out[k] for k in w)
    return out


def snr_loss(y: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, centre: float,
             remove_offset: bool = True) -> torch.Tensor:
    """Negative SNR (dB) of the decoded trace, the quantity the benchmarks measure.

    Positions are converted to amplitudes around the crop centre row. With
    ``remove_offset`` the mean error is discarded, as in the competition metric, so the
    network is not punished for the unknowable 0 mV level.
    """
    v = valid.float()
    n = v.sum(1).clamp_min(1.0)
    s = (centre - target) * v
    e = (y - target) * v
    if remove_offset:
        e = (e - e.sum(1, keepdim=True) / n[:, None]) * v
    signal = (s**2).sum(1) / n
    noise = (e**2).sum(1) / n
    snr = 10 * torch.log10((signal + 1e-2) / (noise + 1e-2))
    return -(snr * (v.sum(1) > 8)).sum() / (v.sum(1) > 8).sum().clamp_min(1)


def trace_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    sigma: float = 1.0,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """TraceNet loss: column cross-entropy + sub-pixel L1 + SNR + visibility.

    The cross-entropy against a narrow Gaussian shapes a sharp, unimodal column
    distribution; the L1 and SNR terms act on the soft-argmax and tie training to the
    evaluation metric (the recipe shared by the best competition entries).
    """
    w = {"ce": 1.0, "l1": 0.5, "snr": 0.02, "visible": 0.3} | (weights or {})
    logits = pred["logits"].float()
    h = logits.shape[1]
    valid = batch["valid"]
    v = valid.float()
    n = v.sum().clamp_min(1.0)
    log_prob = F.log_softmax(logits, dim=1)
    target = encode_columns(batch["y"].float(), h, sigma)
    ce = -(target * log_prob).sum(1)
    rows = torch.arange(h, device=logits.device, dtype=torch.float32)[None, :, None]
    y_hat = (log_prob.exp() * rows).sum(1)
    out = {
        "ce": (ce * v).sum() / n,
        "l1": (F.smooth_l1_loss(y_hat, batch["y"].float(), reduction="none", beta=0.5) * v).sum() / n,
        "snr": snr_loss(y_hat, batch["y"].float(), valid, (h - 1) / 2),
    }
    window = batch["window"].float()
    out["visible"] = (F.binary_cross_entropy_with_logits(pred["visible"].float(), v, reduction="none")
                      * window).sum() / window.sum().clamp_min(1.0)
    out["loss"] = sum(w[k] * out[k] for k in w)
    with torch.no_grad():
        out["mae_px"] = ((y_hat - batch["y"]).abs() * v).sum() / n
    return out
