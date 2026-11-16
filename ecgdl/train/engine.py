"""A small, dependable training loop: AdamW, warmup + cosine, EMA, AMP, resumable."""
from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class TrainConfig:
    out_dir: str = "runs/exp"
    steps: int = 10_000
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup: int = 300
    ema: float = 0.999
    grad_clip: float = 2.0
    amp: bool = True
    log_every: int = 50
    eval_every: int = 1000
    save_every: int = 500
    #: stop early after this many minutes (the checkpoint stays resumable)
    max_minutes: float | None = None
    channels_last: bool = True
    seed: int = 0


class Ema:
    """Exponential moving average of the weights, with the usual warm-up of the decay."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def cycle(loader: Iterable) -> Iterator:
    while True:
        yield from loader


def save_checkpoint(path: Path, kind: str, model_config: dict, ema: Ema, model: nn.Module,
                    optimizer: torch.optim.Optimizer, step: int, extra: dict | None = None) -> None:
    """Atomic checkpoint. ``state_dict`` holds the EMA weights used for inference."""
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "kind": kind,
            "config": model_config,
            "state_dict": ema.model.state_dict(),
            "raw_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "ema_updates": ema.updates,
            "extra": extra or {},
        },
        tmp,
    )
    tmp.replace(path)


def fit(
    model: nn.Module,
    kind: str,
    model_config: dict,
    batches: Iterable[dict[str, torch.Tensor]],
    loss_fn: Callable[[dict, dict], dict[str, torch.Tensor]],
    config: TrainConfig,
    device: str = "cpu",
    evaluate: Callable[[nn.Module], dict[str, float]] | None = None,
    extra: dict | None = None,
) -> Path:
    """Train ``model`` and return the path of the last checkpoint.

    ``batches`` may be a DataLoader or any iterable of dict batches; it is cycled.
    ``loss_fn(prediction, batch)`` returns a dict with a ``"loss"`` entry.
    """
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    model = model.to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.lr,
    )

    def lr_at(step: int) -> float:
        if step < config.warmup:
            return (step + 1) / config.warmup
        t = (step - config.warmup) / max(1, config.steps - config.warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    ema = Ema(model, config.ema)
    use_amp = config.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    last = out / "last.pt"
    step = 0
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["raw_state_dict"])
        ema.model.load_state_dict(ck["state_dict"])
        ema.updates = ck.get("ema_updates", 0)
        optimizer.load_state_dict(ck["optimizer"])
        step = ck["step"]
        scheduler.last_epoch = step
        for g, base in zip(optimizer.param_groups, scheduler.base_lrs):
            g["lr"] = base * lr_at(step)
    (out / "train_config.json").write_text(json.dumps({"train": asdict(config), "model": model_config, "kind": kind}, indent=1))
    log = (out / "log.jsonl").open("a")
    start = time.time()
    running: dict[str, float] = {}
    count = 0
    model.train()
    for batch in cycle(batches):
        if step >= config.steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if config.channels_last:
            batch = {k: (v.contiguous(memory_format=torch.channels_last) if v.ndim == 4 else v) for k, v in batch.items()}
        with torch.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp):
            losses = loss_fn(model(batch[_input_key(batch)]), batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses["loss"]).backward()
        if config.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)
        step += 1
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.detach())
        count += 1
        elapsed = (time.time() - start) / 60
        stop = config.max_minutes is not None and elapsed >= config.max_minutes
        if step % config.log_every == 0 or stop or step == config.steps:
            rec = {"step": step, "minutes": round(elapsed, 2), "lr": optimizer.param_groups[0]["lr"],
                   **{k: round(v / count, 5) for k, v in running.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            running, count = {}, 0
        if evaluate is not None and (step % config.eval_every == 0 or stop or step == config.steps):
            metrics = evaluate(ema.model)
            rec = {"step": step, "eval": metrics}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
            model.train()
        if step % config.save_every == 0 or stop or step == config.steps:
            save_checkpoint(last, kind, model_config, ema, model, optimizer, step, extra)
        if stop:
            break
    log.close()
    save_checkpoint(last, kind, model_config, ema, model, optimizer, step, extra)
    return last


def _input_key(batch: dict) -> str:
    return "image" if "image" in batch else "crop"
