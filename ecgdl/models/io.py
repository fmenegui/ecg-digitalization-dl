"""Checkpoint loading."""
from __future__ import annotations

from pathlib import Path

import torch

from .page_net import PageNet, PageNetConfig
from .trace_net import TraceNet, TraceNetConfig


def build_model(kind: str, config: dict | None = None) -> PageNet | TraceNet:
    config = dict(config or {})
    for key in ("widths", "depths"):
        if key in config:
            config[key] = tuple(config[key])
    if kind == "page":
        return PageNet(PageNetConfig(**config))
    if kind == "trace":
        return TraceNet(TraceNetConfig(**config))
    raise ValueError(f"unknown model kind: {kind}")


def load_checkpoint(path: str | Path, device: str = "cpu", kind: str | None = None) -> PageNet | TraceNet:
    """Load the EMA weights of a training checkpoint for inference."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if kind is not None and ck["kind"] != kind:
        raise ValueError(f"{path} holds a {ck['kind']!r} model, expected {kind!r}")
    config = dict(ck["config"])
    config["pretrained"] = False  # weights come from the checkpoint
    model = build_model(ck["kind"], config)
    model.load_state_dict({k: (v.float() if v.dtype == torch.float16 else v) for k, v in ck["state_dict"].items()})
    return model.to(device).eval()


def export_inference_checkpoint(src: str | Path, dst: str | Path, half: bool = True) -> Path:
    """Strip a training checkpoint down to what inference needs (EMA weights, optionally float16)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    state = {k: (v.half() if half and v.dtype == torch.float32 else v) for k, v in ck["state_dict"].items()}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": ck["kind"], "config": ck["config"], "state_dict": state, "step": ck.get("step"),
                "extra": ck.get("extra", {})}, dst)
    return dst
