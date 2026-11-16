"""Checks of the renderer adapter. Skipped when ecg-torch-renderer is not installed."""
import numpy as np
import pytest
import torch

pytest.importorskip("ecg_gpu")

from ecgdl.data.cache import OnlinePages, build_crop_cache, build_page_cache, make_renderer  # noqa: E402
from ecgdl.data.signals import synthetic_ecg  # noqa: E402
from ecgdl.data.synth import PageSample  # noqa: E402
from ecgdl.data.targets import CropConfig, sample_lattice, trace_crops  # noqa: E402

pytestmark = pytest.mark.renderer

SYNTH = {"height": 384, "width": 512, "severity": 0.3,
         "parameters": {"grid_present": 1.0, "photo": ["bernoulli", 0.5], "speed_index": 1, "gain_index": 1}}


@pytest.fixture(scope="module")
def samples():
    return make_renderer(SYNTH, seed=1).render(synthetic_ecg(2, seed=0))


def test_page_sample_fields(samples, tmp_path):
    s = samples[0]
    assert s.image.shape == (384, 512, 3) and s.image.dtype == np.uint8
    assert len(s.grid_xy) == len(s.grid_ij) > 100
    assert s.speed == 25.0 and s.gain == 10.0
    assert abs(np.linalg.norm(s.direction) - 1) < 1e-4
    assert s.segments.shape[1] == 7 and (s.segments[:, 3] > s.segments[:, 2]).all()
    s.save(str(tmp_path / "p.npz"))
    back = PageSample.load(str(tmp_path / "p.npz"))
    assert np.array_equal(back.grid_ij, s.grid_ij) and back.speed == s.speed


def test_lattice_of_ground_truth_points_is_tight(samples):
    lattice = sample_lattice(samples[0], np.random.default_rng(0))
    assert lattice.quality["residual_px"] < 0.2
    assert abs(lattice.spacing_px - 5 * samples[0].px_per_mm) < 0.15 * 5 * samples[0].px_per_mm


def test_analytic_targets_lie_on_the_rendered_ink(samples):
    rng = np.random.default_rng(0)
    diffs = []
    for s in samples:
        out = trace_crops(s, CropConfig(height=96, width=256, px_per_mm_x=4.0, crops_per_page=6), rng)
        if out is None:
            continue
        gray = out["crop"].mean(1)
        for n in range(len(gray)):
            for c in torch.nonzero(out["valid"][n])[:, 0].tolist()[::9]:
                y = float(out["y"][n, c])
                lo, hi = max(0, int(y) - 5), min(96, int(y) + 7)
                col = gray[n, lo:hi, c]
                wts = (col.max() - col).clamp_min(0) ** 2
                if wts.sum() > 1e-3:
                    diffs.append(float((torch.arange(lo, hi) * wts).sum() / wts.sum()) - y)
    assert len(diffs) > 30 and abs(np.median(diffs)) < 0.5


def test_caches_and_online_streams(tmp_path):
    signals = synthetic_ecg(4, seed=2)
    assert build_page_cache(tmp_path / "pages", signals, 2, SYNTH, seed=3, batch=2) == 2
    config = CropConfig(height=64, width=128, px_per_mm_x=4.0, crops_per_page=4)
    assert build_crop_cache(tmp_path / "pages", tmp_path / "crops", config) > 0
    page_batch = next(iter(OnlinePages(signals, "page", SYNTH, batch=2, window=256)))
    assert page_batch["image"].shape == (2, 3, 256, 256) and page_batch["names"].shape == (2, 12, 256, 256)
    crop_batch = next(iter(OnlinePages(signals, "trace", SYNTH, batch=2, crop=config)))
    assert crop_batch["crop"].shape[1:] == (3, 64, 128) and crop_batch["y"].shape[1] == 128
