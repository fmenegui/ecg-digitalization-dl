"""End-to-end checks with oracles instead of trained weights (no renderer, no downloads)."""
import numpy as np
import pytest
import torch

from ecgdl.constants import LEADS
from ecgdl.data.signals import synthetic_ecg
from ecgdl.data.targets import CropConfig, page_targets, sample_lattice, trace_crops
from ecgdl.metrics import kaggle_snr
from ecgdl.pipeline import Digitizer, DigitizerConfig, load_image, resolve_avr_polarity
from ecgdl.signal.constraints import Observation
from ecgdl.testing import InkTraceNet, OraclePageNet, draw_page, paper_time_ground_truth


@pytest.fixture(scope="module")
def page():
    return draw_page(synthetic_ecg(1, seed=3)[0], seed=0)


def digitizer(sample, **kw):
    return Digitizer(OraclePageNet(sample, **kw), InkTraceNet(128, 8.0), DigitizerConfig(max_source_px_per_mm=None))


def test_page_targets_are_consistent(page):
    t = page_targets(page)
    h, w = page.image.shape[:2]
    assert t["grid"].shape == (1, h, w) and t["names"].shape == (12, h, w)
    assert abs(float(t["grid"].max()) - 1) < 0.2
    assert int(t["offset_valid"].sum()) == int(((page.grid_xy.round() >= 0).all(1)).sum())
    assert (t["offset"].abs() <= 0.5 + 1e-4).all()
    assert (t["names"].amax((1, 2)) > 0.5).sum() == 12


def test_trace_crops_targets_lie_on_the_ink(page):
    rng = np.random.default_rng(0)
    out = trace_crops(page, CropConfig(height=96, width=256, crops_per_page=6), rng, sample_lattice(page, rng))
    gray = out["crop"].mean(1)
    diffs = []
    for n in range(len(gray)):
        for c in torch.nonzero(out["valid"][n])[:, 0].tolist()[::7]:
            y = float(out["y"][n, c])
            lo, hi = max(0, int(y) - 5), min(96, int(y) + 7)
            col = gray[n, lo:hi, c]
            wts = (col.max() - col).clamp_min(0) ** 2
            if wts.sum() > 1e-3:
                diffs.append(float((torch.arange(lo, hi) * wts).sum() / wts.sum()) - y)
    assert len(diffs) > 50 and abs(np.median(diffs)) < 0.35


def test_end_to_end_layout_calibration_and_signal(page):
    result = digitizer(page)(page.image)
    assert result.ok and result.layout == "3x4+1R" and len(result.segments) == 13
    assert result.quality["lattice"]["residual_px"] < 0.2
    assert abs(result.quality["px_per_mm"] - page.px_per_mm) < 0.2
    gt = paper_time_ground_truth(page)
    n = min(gt.shape[1], result.signals.shape[1])
    assert abs(result.signals.shape[1] - 5000) <= 30
    snr, per = kaggle_snr(gt[:, :n], result.signals[:, :n].astype(float), 500)
    # the classical stand-in tracer is crude; the point is that geometry, time and gain are right
    assert snr > 5 and sorted(per.values())[-4] > 12
    for lead in ("I", "V1", "V6"):
        i = LEADS.index(lead)
        m = np.isfinite(gt[i, :n]) & np.isfinite(result.signals[i, :n])
        assert m.sum() > 1100
    assert set(result.segment_samples) == set(LEADS)
    assert len(result.segment_samples["II"]) == 2
    assert result.to_dict()["segments"][0]["lead_name"] in LEADS


@pytest.mark.parametrize("turn", [1, 2, 3])
def test_rotated_photo_is_read_upright(page, turn):
    rotated = draw_page(page.signals, seed=0)
    image = np.rot90(rotated.image, turn).copy()
    h, w = rotated.image.shape[:2]
    xy = rotated.grid_xy.copy()
    names = rotated.name_xy.copy()

    def rot(p):
        for _ in range(turn):  # np.rot90 (counter-clockwise): (x, y) -> (y, W - 1 - x)
            p = np.stack([p[:, 1], (w if _ % 2 == 0 else h) - 1 - p[:, 0]], 1)
        return p

    d = rotated.direction.copy()
    for _ in range(turn):
        d = np.array([d[1], -d[0]])
    rotated.image, rotated.grid_xy, rotated.name_xy, rotated.direction = image, rot(xy), rot(names), d.astype(np.float32)
    rotated.trace = np.rot90(rotated.trace, turn).copy()
    rotated.valid = np.rot90(rotated.valid, turn).copy()
    result = digitizer(rotated)(image)
    assert result.ok and result.layout == "3x4+1R" and len(result.segments) == 13


def test_wrong_direction_is_corrected_by_the_layout(page):
    net = OraclePageNet(page)
    net.direction = -net.direction  # the page is claimed to read right-to-left
    result = Digitizer(net, InkTraceNet(128, 8.0), DigitizerConfig(max_source_px_per_mm=None))(page.image)
    assert result.ok and result.layout == "3x4+1R"
    assert any("opposite" in note for note in result.notes)


def test_page_without_grid_fails_cleanly():
    blank = np.full((300, 400, 3), 255, np.uint8)
    sample = draw_page(synthetic_ecg(1)[0], seed=1)
    result = Digitizer(OraclePageNet(_without_grid(sample)), InkTraceNet(), DigitizerConfig())(blank)
    assert not result.ok and "grid" in result.notes[0]
    assert result.signals.shape == (12, 0)


def _without_grid(sample):
    sample.grid_xy = sample.grid_xy[:0]
    sample.grid_ij = sample.grid_ij[:0]
    return sample


def test_load_image_inputs(tmp_path):
    from PIL import Image

    arr = (np.random.default_rng(0).random((20, 30, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png")
    for src in (arr, tmp_path / "a.png", Image.fromarray(arr), torch.from_numpy(arr).permute(2, 0, 1)):
        t = load_image(src)
        assert t.shape == (3, 20, 30) and 0 <= float(t.min()) and float(t.max()) <= 1


def test_inverted_avr_is_detected():
    sig = synthetic_ecg(1, seed=9)[0].astype(float)
    n = sig.shape[1]
    ones = np.ones(n)
    obs = [Observation(0, sig[0], ones), Observation(1, sig[1], ones), Observation(3, -sig[3], ones)]
    fixed, flipped = resolve_avr_polarity(obs)
    assert flipped and np.allclose(fixed[2].values, sig[3])
    same, flipped = resolve_avr_polarity([obs[0], obs[1], Observation(3, sig[3], ones)])
    assert not flipped


def test_quarter_turn_error_is_corrected_by_the_projection_profile(page):
    net = OraclePageNet(page)
    net.direction = torch.tensor([-float(net.direction[1]), float(net.direction[0])])  # wrong by 90 degrees
    result = Digitizer(net, InkTraceNet(128, 8.0), DigitizerConfig(max_source_px_per_mm=None))(page.image)
    assert result.ok and result.layout == "3x4+1R" and len(result.segments) == 13


def test_unreadable_names_are_replaced_by_physiology(page):
    # no lead name is detected at all: rows -> standard hypotheses -> Einthoven referee,
    # and the rhythm strip is identified by its correlation with lead II
    result = digitizer(page, name_dropout=1.0)(page.image)
    assert result.ok and result.layout == "3x4+1R"
    assert len(result.segments) == 13
    strip = max(result.segments, key=lambda s: s.x1 - s.x0)
    assert LEADS[strip.lead] == "II" and strip.source in ("correlation", "assumed")
    assert any("closure residual" in note for note in result.notes)


def test_closure_referee_tells_6x2_from_3x4_plus_3_strips():
    from ecgdl.layout.assemble import LayoutConfig, row_hypotheses
    from ecgdl.layout.verify import choose_layout, closure_residual

    sig = synthetic_ecg(1, seed=11)[0].astype(float)
    # ink map of six full-width rows, and the six rows as they would be decoded from a 3x4+3R page
    ink = np.zeros((300, 560), np.float32)
    for r in range(6):
        ink[40 + 40 * r, 30:530] = 1.0
    hyps = [h for h, _, _ in row_hypotheses(ink, 2.0, 25.0, LayoutConfig())]
    by_name = {h.template: h for h in hyps}
    order = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    strips = {}
    for r, row in enumerate(order):
        strips[r] = np.concatenate([sig[LEADS.index(lead), c * 1250 : (c + 1) * 1250] for c, lead in enumerate(row)])
    for r, lead in enumerate(["II", "V1", "V5"]):
        strips[3 + r] = sig[LEADS.index(lead)]
    assert closure_residual(by_name["3x4+3R"], strips, 500, 25.0) < 0.05
    assert closure_residual(by_name["6x2"], strips, 500, 25.0) > 0.5
    chosen, _, _ = choose_layout([by_name["6x2"], by_name["3x4+3R"]], strips, 500, 25.0)
    assert chosen.template == "3x4+3R"


def test_half_turn_is_visible_in_the_closure_of_the_same_rows():
    """Reading rows bottom-up, right-to-left and inverted must close only for a turned page."""
    from ecgdl.layout.assemble import LayoutConfig, row_hypotheses
    from ecgdl.layout.verify import closure_residual

    sig = synthetic_ecg(1, seed=13)[0].astype(float)
    order = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    upright = {r: np.concatenate([sig[LEADS.index(lead), c * 1250 : (c + 1) * 1250] for c, lead in enumerate(row)])
               for r, row in enumerate(order)}
    rows = [(20.0 + 30 * r, 15.0, 265.0) for r in range(3)]
    hyp = next(h for h, _, _ in row_hypotheses(np.zeros((1, 1), np.float32), 1.0, 25.0, LayoutConfig(), None, rows)
               if h.template == "3x4")
    photographed_upside_down = {2 - r: -v[::-1] for r, v in upright.items()}  # what the decoder would return
    turned_back = {2 - r: -v[::-1] for r, v in photographed_upside_down.items()}
    assert closure_residual(hyp, upright, 500, 25.0) < 0.02
    assert closure_residual(hyp, photographed_upside_down, 500, 25.0) > 0.5
    assert closure_residual(hyp, turned_back, 500, 25.0) < 0.02


def test_ecg_likeness_separates_traces_from_borders():
    from ecgdl.layout.verify import ecg_likeness

    sig = synthetic_ecg(1, seed=4)[0, 1].astype(float)
    rng = np.random.default_rng(0)
    assert ecg_likeness(sig, 500) > 0.5
    assert ecg_likeness(np.zeros(5000) + rng.normal(0, 0.005, 5000), 500) == 0.0  # a flat border line
    assert ecg_likeness(rng.normal(0, 0.1, 5000), 500) < 0.25  # a line of text read as noise
    assert ecg_likeness(np.full(5000, np.nan), 500) == 0.0


def test_single_pages_of_a_multi_page_print_are_recognised_by_closure():
    from ecgdl.layout.assemble import LayoutConfig, row_hypotheses
    from ecgdl.layout.verify import chest_half_from_polarity, choose_layout

    sig = synthetic_ecg(1, seed=12)[0].astype(float)
    rows3 = [(20.0 + 30 * r, 15.0, 265.0) for r in range(3)]
    hyps = [h for h, _, _ in row_hypotheses(np.zeros((1, 1), np.float32), 1.0, 25.0, LayoutConfig(), None, rows3)]
    assert hyps[0].template == "3x4"  # complete pages are preferred a priori

    def pick(leads):
        strips = {r: sig[LEADS.index(lead)] for r, lead in enumerate(leads)}
        return choose_layout(hyps, strips, 500, 25.0)[0].template, strips

    assert pick(["I", "II", "III"])[0] == "3x1-I-II-III"
    assert pick(["aVR", "aVL", "aVF"])[0] == "3x1-aVR-aVL-aVF"
    name, strips = pick(["V1", "V2", "V3"])
    assert name.startswith("3x1-V")  # nothing closes: chest leads
    # the fixture has no R-wave progression; build one to test the polarity rule
    qrs = np.zeros(5000)
    for k in range(12):  # 24 ms wide complexes
        qrs[k::400] = 1.0
    assert chest_half_from_polarity({0: -qrs, 1: -qrs, 2: qrs}) == "3x1-V1-V2-V3"
    assert chest_half_from_polarity({0: qrs, 1: qrs, 2: qrs}) == "3x1-V4-V5-V6"
    # and a true 3x4 page is still read as 3x4
    order = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    strips = {r: np.concatenate([sig[LEADS.index(lead), c * 1250 : (c + 1) * 1250] for c, lead in enumerate(row)])
              for r, row in enumerate(order)}
    assert choose_layout(hyps, strips, 500, 25.0)[0].template == "3x4"
