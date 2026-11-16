"""Training entry points driven by a YAML configuration (see ``configs/``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from ..data.cache import OnlinePages, PageWindows, TraceCrops, load_bank
from ..data.synth import PageSample
from ..data.targets import CropConfig, page_targets
from ..geometry.peaks import decode_peaks
from ..losses import page_loss, trace_loss
from ..models.io import build_model
from .engine import TrainConfig, fit


def match_points(pred: np.ndarray, truth: np.ndarray, radius: float) -> tuple[int, np.ndarray]:
    """Greedy one-to-one matching within ``radius``. Returns ``(true positives, distances)``."""
    if len(pred) == 0 or len(truth) == 0:
        return 0, np.zeros(0)
    dist, idx = cKDTree(truth).query(pred, distance_upper_bound=radius)
    order = np.argsort(dist)
    used: set[int] = set()
    kept = []
    for k in order:
        if not np.isfinite(dist[k]):
            break
        if int(idx[k]) in used:
            continue
        used.add(int(idx[k]))
        kept.append(dist[k])
    return len(kept), np.array(kept)


@torch.no_grad()
def evaluate_page(model: torch.nn.Module, pages: list[Path], device: str, threshold: float = 0.3) -> dict[str, float]:
    """Keypoint F1 / localisation, name recall, trace Dice and direction error on full pages."""
    model.eval()
    tp = fp = fn = 0
    errs, dice, ang = [], [], []
    name_tp = name_total = name_pred = 0
    for p in pages:
        sample = PageSample.load(str(p))
        t = page_targets(sample)
        out = model(t["image"][None].to(device))
        xy, _ = decode_peaks(out["grid"][0, 0].sigmoid(), out["offset"][0], threshold)
        k, d = match_points(xy, sample.grid_xy, 1.5)
        tp, fp, fn = tp + k, fp + len(xy) - k, fn + len(sample.grid_xy) - k
        errs.append(d)
        prob = out["trace"][0, 0].sigmoid().cpu()
        target = t["trace"][0]
        dice.append(float((2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)))
        ang.append(float(np.degrees(np.arccos(np.clip(out["direction"][0].cpu().numpy() @ sample.direction, -1, 1)))))
        names = out["names"][0].sigmoid()
        for lead in range(names.shape[0]):
            pxy, _ = decode_peaks(names[lead], None, 0.3, nms_radius=4, max_points=8)
            truth = sample.name_xy[sample.name_lead == lead]
            k, _ = match_points(pxy, truth, 6.0)
            name_tp, name_total, name_pred = name_tp + k, name_total + len(truth), name_pred + len(pxy)
    errs_all = np.concatenate(errs) if errs else np.zeros(0)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "grid_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "grid_precision": round(prec, 4),
        "grid_recall": round(rec, 4),
        "grid_err_px": round(float(np.median(errs_all)) if len(errs_all) else float("nan"), 3),
        "trace_dice": round(float(np.mean(dice)), 4),
        "direction_deg": round(float(np.median(ang)), 2),
        "name_recall": round(name_tp / max(name_total, 1), 4),
        "name_precision": round(name_tp / max(name_pred, 1), 4),
    }


@torch.no_grad()
def evaluate_trace(model: torch.nn.Module, loader: DataLoader, device: str, max_batches: int = 40) -> dict[str, float]:
    """Mean absolute error (pixels) and offset-free SNR (dB) of the soft-argmax on held-out crops."""
    model.eval()
    mae_sum = n_sum = 0.0
    snrs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["crop"])["logits"].float()
        h = logits.shape[1]
        rows = torch.arange(h, device=device, dtype=torch.float32)[None, :, None]
        y = (torch.softmax(logits, 1) * rows).sum(1)
        v = batch["valid"].float()
        mae_sum += float(((y - batch["y"]).abs() * v).sum())
        n_sum += float(v.sum())
        n = v.sum(1).clamp_min(1)
        e = (y - batch["y"]) * v
        e = (e - e.sum(1, keepdim=True) / n[:, None]) * v
        s = ((h - 1) / 2 - batch["y"]) * v
        s = (s - s.sum(1, keepdim=True) / n[:, None]) * v
        ok = v.sum(1) > 32
        snr = 10 * torch.log10((s**2).sum(1).clamp_min(1e-6) / (e**2).sum(1).clamp_min(1e-6))
        snrs += snr[ok].cpu().tolist()
    return {"mae_px": round(mae_sum / max(n_sum, 1), 4), "snr_db": round(float(np.mean(snrs)) if snrs else float("nan"), 3),
            "snr_db_median": round(float(np.median(snrs)) if snrs else float("nan"), 3)}


def train_from_config(config: dict | str | Path, device: str | None = None) -> Path:
    """Train a PageNet (``kind: page``) or TraceNet (``kind: trace``) from a config dict or YAML path."""
    if not isinstance(config, dict):
        config = yaml.safe_load(Path(config).read_text())
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kind = config["kind"]
    data = config.get("data", {})
    model = build_model(kind, config.get("model"))
    tc = TrainConfig(**config.get("train", {}))
    batch_size, workers = int(data.get("batch_size", 8)), int(data.get("workers", 2))
    if kind == "page":
        if data.get("online"):
            signals = load_bank(data.get("bank"), "train")
            batches = OnlinePages(signals, "page", data.get("synth"), batch_size, tc.seed, device, data.get("window"))
        else:
            ds = PageWindows(data["pages"], int(data.get("window", 448)), length=max(tc.steps * batch_size, 1),
                             seed=tc.seed, augment=float(data.get("augment", 1.0)))
            batches = DataLoader(ds, batch_size, shuffle=True, num_workers=workers, drop_last=True,
                                 persistent_workers=workers > 0)
        val_pages = sorted(Path(data["val_pages"]).glob("page_*.npz"))[: int(data.get("val_count", 16))] if data.get("val_pages") else []
        evaluate = (lambda m: evaluate_page(m, val_pages, device)) if val_pages else None
        return fit(model, kind, model.config.to_dict(), batches, lambda pred, b: page_loss(pred, b, config.get("loss")),
                   tc, device, evaluate)
    if kind == "trace":
        if data.get("online"):
            signals = load_bank(data.get("bank"), "train")
            crop = CropConfig(**{k: (tuple(v) if k == "span_mm" else v) for k, v in (data.get("crop") or {}).items()})
            batches = OnlinePages(signals, "trace", data.get("synth"), int(data.get("pages_per_batch", 2)), tc.seed, device, crop=crop)
        else:
            ds = TraceCrops(data["crops"], float(data.get("augment", 1.0)), tc.seed)
            batches = DataLoader(ds, batch_size, shuffle=True, num_workers=workers, drop_last=True,
                                 persistent_workers=workers > 0)
        evaluate = None
        if data.get("val_crops"):
            val = DataLoader(TraceCrops(data["val_crops"], 0.0), batch_size, shuffle=False, num_workers=0)
            evaluate = lambda m: evaluate_trace(m, val, device)  # noqa: E731
        return fit(model, kind, model.config.to_dict(), batches, lambda pred, b: trace_loss(pred, b, weights=config.get("loss")),
                   tc, device, evaluate)
    raise ValueError(f"unknown kind: {kind}")


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides (values parsed as YAML) to a nested config."""
    for item in overrides:
        key, _, value = item.partition("=")
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return config


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train a PageNet or TraceNet from a YAML configuration.")
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads used by PyTorch")
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    config = apply_overrides(yaml.safe_load(Path(a.config).read_text()), a.set)
    print("checkpoint:", train_from_config(config, a.device))


if __name__ == "__main__":
    main()
