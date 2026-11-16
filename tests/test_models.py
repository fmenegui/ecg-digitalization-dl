import numpy as np
import pytest
import torch

from ecgdl.losses import page_loss, quality_focal_loss, snr_loss, trace_loss
from ecgdl.models.io import build_model, load_checkpoint
from ecgdl.models.unet import pad_to_multiple
from ecgdl.signal.codec import encode_columns
from ecgdl.train.engine import TrainConfig, fit

SMALL = {"widths": (8, 12, 16), "depths": (1, 1, 1)}


def test_pagenet_shapes_any_input_size():
    net = build_model("page", SMALL | {"stem_stride": 2}).eval()
    out = net(torch.rand(2, 3, 70, 101))
    assert out["grid"].shape == (2, 1, 70, 101)
    assert out["offset"].shape == (2, 2, 70, 101)
    assert out["names"].shape == (2, 12, 70, 101)
    assert torch.allclose(out["direction"].norm(dim=-1), torch.ones(2), atol=1e-5)
    # sparse heads start negative, regression heads at zero
    assert out["grid"].mean() < -2 and out["offset"].abs().mean() < 1.5


def test_tracenet_shapes():
    net = build_model("trace", SMALL).eval()
    out = net(torch.rand(2, 3, 48, 100))
    assert out["logits"].shape == (2, 48, 100) and out["visible"].shape == (2, 100)


def test_pad_to_multiple():
    x, (h, w) = pad_to_multiple(torch.rand(1, 3, 33, 50), 16)
    assert x.shape[-2:] == (48, 64) and (h, w) == (33, 50)


def test_focal_loss_prefers_the_right_heatmap():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 8, 8] = 1
    good = torch.full_like(target, -6.0)
    good[0, 0, 8, 8] = 6
    assert quality_focal_loss(good, target) < 0.01 < quality_focal_loss(-good, target)


def test_snr_loss_ignores_constant_offset():
    target = 20 + 5 * torch.sin(torch.arange(64) / 5)[None]
    valid = torch.ones(1, 64, dtype=torch.bool)
    assert snr_loss(target + 3.0, target, valid, 24.0) < -20
    assert snr_loss(target + 3.0, target, valid, 24.0, remove_offset=False) > snr_loss(target + 3.0, target, valid, 24.0) + 10


def _trace_batch(n=4, h=32, w=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = 16 + 8 * torch.sin(torch.arange(w)[None] / 6 + torch.rand(n, 1, generator=g) * 6)
    crop = torch.ones(n, 3, h, w)
    rows = torch.arange(h)[None, :, None]
    crop -= 0.9 * (encode_columns(y, h, 0.8) * 2).clamp(0, 1)[:, None] * (rows == rows)[:, None]
    return {"crop": crop, "y": y, "valid": torch.ones(n, w, dtype=torch.bool), "window": torch.ones(n, w, dtype=torch.bool)}


def test_trace_loss_is_finite_with_empty_validity():
    net = build_model("trace", SMALL)
    batch = _trace_batch()
    batch["valid"][:] = False
    batch["window"][:] = False
    loss = trace_loss(net(batch["crop"]), batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()


def test_training_loop_learns_resumes_and_reloads(tmp_path):
    torch.manual_seed(0)
    net = build_model("trace", SMALL)
    batches = [_trace_batch(seed=s) for s in range(4)]
    cfg = TrainConfig(out_dir=str(tmp_path), steps=60, lr=5e-3, warmup=5, ema=0.9, log_every=30, save_every=30, eval_every=10**9)
    ck = fit(net, "trace", net.config.to_dict(), batches, lambda p, b: trace_loss(p, b), cfg)
    log = [eval(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert log[-1]["mae_px"] < 0.6 * log[0]["mae_px"]
    # resume continues from the saved step instead of starting over
    cfg2 = TrainConfig(**{**cfg.__dict__, "steps": 70})
    fit(build_model("trace", SMALL), "trace", net.config.to_dict(), batches, lambda p, b: trace_loss(p, b), cfg2)
    assert torch.load(ck, weights_only=False)["step"] == 70
    model = load_checkpoint(ck, kind="trace")
    assert not model.training
    with pytest.raises(ValueError):
        load_checkpoint(ck, kind="page")


def test_page_loss_runs():
    net = build_model("page", SMALL | {"stem_stride": 2})
    b = {
        "image": torch.rand(2, 3, 64, 64), "grid": torch.zeros(2, 1, 64, 64), "offset": torch.zeros(2, 2, 64, 64),
        "offset_valid": torch.zeros(2, 1, 64, 64), "trace": torch.zeros(2, 1, 64, 64),
        "names": torch.zeros(2, 12, 64, 64), "direction": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    b["grid"][:, :, 20, 20] = 1
    out = page_loss(net(b["image"]), b)
    assert all(torch.isfinite(v) for v in out.values())
    out["loss"].backward()
    assert np.isfinite(float(out["loss"]))
