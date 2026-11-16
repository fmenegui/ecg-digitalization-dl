import numpy as np
import torch

from ecgdl.constants import LEADS, LIMB_BASIS
from ecgdl.data.signals import synthetic_ecg
from ecgdl.signal.codec import decode_columns, encode_columns, local_soft_argmax, soft_argmax, viterbi_path
from ecgdl.signal.constraints import Observation, align_duplicates, einthoven_residual, fuse_leads
from ecgdl.signal.resample import best_lag, columns_to_signal, shift_signal, signal_to_columns


def test_codec_roundtrip_is_subpixel():
    y = torch.rand(3, 50) * 60 + 10
    prob = encode_columns(y, 96, sigma=1.0)
    assert torch.allclose(prob.sum(1), torch.ones(3, 50), atol=1e-5)
    decoded, _ = soft_argmax(prob.clamp_min(1e-12).log())
    assert (decoded - y).abs().max() < 0.02


def test_local_soft_argmax_ignores_second_mode():
    y = torch.full((1, 20), 30.3)
    prob = 0.6 * encode_columns(y, 96) + 0.4 * encode_columns(torch.full((1, 20), 70.0), 96)
    global_y, _ = soft_argmax(prob.log())
    local_y, mass = local_soft_argmax(prob, torch.full((1, 20), 30), radius=4)
    assert (global_y - 30.3).abs().min() > 10  # the plain expectation is dragged away
    assert (local_y - 30.3).abs().max() < 0.05
    assert torch.allclose(mass, torch.full_like(mass, 0.6), atol=0.01)


def test_viterbi_follows_the_continuous_trace():
    h, w = 64, 120
    true = (32 + 10 * np.sin(np.arange(w) / 9)).round().astype(int)
    logp = np.full((h, w), -8.0, np.float32)
    logp[true, np.arange(w)] = -0.5
    # a stronger distractor for a few isolated columns must not capture the path
    logp[5, 40:43] = -0.1
    path = viterbi_path(logp, smooth=0.35, jump=6.0)
    assert (path == true).mean() > 0.97
    # ...but a real jump (QRS) is followed
    step = np.where(np.arange(w) < 60, 20, 50)
    logp2 = np.full((h, w), -8.0, np.float32)
    logp2[step, np.arange(w)] = -0.2
    assert (viterbi_path(logp2) == step).all()


def test_decode_methods_agree_on_clean_logits():
    y = torch.rand(2, 64) * 40 + 12
    logits = encode_columns(y, 64, 1.0).clamp_min(1e-9).log()
    for method in ("softargmax", "argmax", "viterbi", "hybrid"):
        out = decode_columns(logits, method)
        assert (out["y"] - y).abs().max() < 0.1, method
        assert out["confidence"].min() > 0.95


def test_hybrid_decoder_resolves_ambiguous_columns_only():
    w = 80
    y = torch.full((1, w), 20.3)
    prob = encode_columns(y, 64, 1.0)
    prob[:, :, 30:40] = 0.55 * prob[:, :, 30:40] + 0.45 * encode_columns(torch.full((1, 10), 50.0), 64, 1.0)
    logits = prob.clamp_min(1e-9).log()
    soft = decode_columns(logits, "softargmax")["y"]
    hybrid = decode_columns(logits, "hybrid")
    assert (soft[0, 30:40] - 20.3).abs().min() > 5  # the expectation lands between the two strokes
    assert (hybrid["y"] - 20.3).abs().max() < 0.2  # the path through the neighbours picks the right one
    assert torch.equal(hybrid["y"][0, :30], soft[0, :30])  # clean columns are untouched
    assert hybrid["confidence"][0, 35] < 0.7 < hybrid["confidence"][0, 5]


def test_columns_roundtrip_reaches_codec_ceiling():
    sig = synthetic_ecg(1, seed=1)[0, 1].astype(float)
    n_cols = 1500  # 250 mm at 6 px/mm
    cols = signal_to_columns(sig, 500, 0.0, 10.0, n_cols)
    t = (np.arange(n_cols) + 0.5) * 10.0 / n_cols
    back = columns_to_signal(cols, t, 500, 5000)
    m = np.isfinite(back)
    err = sig[m] - back[m]
    assert m.mean() > 0.99
    assert 10 * np.log10(np.var(sig[m]) / np.var(err)) > 25


def test_columns_to_signal_keeps_gaps_unknown():
    t = np.arange(400) / 200.0
    v = np.sin(2 * np.pi * t)
    valid = np.ones(400, bool)
    valid[100:160] = False  # 0.3 s hole
    out = columns_to_signal(v, t, 500, 1000, valid)
    assert np.isnan(out[int(0.6 * 500)])
    assert np.isfinite(out[int(0.2 * 500)]) and np.isfinite(out[int(1.5 * 500)])


def test_best_lag_and_shift():
    x = np.random.default_rng(0).normal(size=600)
    moved = shift_signal(x, -7)  # moved[t] = x[t - 7]
    assert best_lag(x, moved, 20) == 7
    assert np.allclose(shift_signal(moved, 7)[:580], x[:580])


def _limb(seed=0):
    return synthetic_ecg(1, seed=seed)[0].astype(float)


def test_fusion_denoises_limb_leads_and_keeps_offsets_free():
    sig = _limb()
    rng = np.random.default_rng(0)
    n = sig.shape[1]
    obs = []
    for lead in range(6):
        noisy = sig[lead] + rng.normal(0, 0.05, n) + rng.uniform(-0.3, 0.3)  # unknown baseline per segment
        obs.append(Observation(lead, noisy, np.ones(n)))
    plain, _ = fuse_leads(obs, n, use_constraints=False)
    fused, conf = fuse_leads(obs, n, use_constraints=True)

    def err(a, lead):
        e = a[lead] - sig[lead]
        return np.std(e - e.mean())

    assert all(err(fused, lead) < 0.8 * err(plain, lead) for lead in range(6))
    assert einthoven_residual(fused) < 0.02 < einthoven_residual(plain)
    assert (conf[:6] > 0).all()


def test_fusion_repairs_an_occluded_stretch_from_other_leads():
    sig = _limb(2)
    n = sig.shape[1]
    obs = [Observation(lead, sig[lead].copy(), np.ones(n)) for lead in (0, 1, 2)]
    obs[2].values[1000:1500] = 9.0  # garbage in lead III ...
    obs[2].weights[1000:1500] = 1e-2  # ... that the decoder flagged as unreliable
    fused, _ = fuse_leads(obs, n)
    assert np.abs(fused[2, 1000:1500] - sig[2, 1000:1500]).max() < 0.25
    assert np.abs(fused[0] - sig[0]).max() < 0.25


def test_fusion_can_reconstruct_unprinted_limb_leads():
    sig = _limb(3)
    n = sig.shape[1]
    obs = [Observation(lead, sig[lead].copy(), np.ones(n)) for lead in (3, 4, 5)]  # aVR, aVL, aVF only
    silent, _ = fuse_leads(obs, n)
    assert np.isnan(silent[0]).all()
    fused, _ = fuse_leads(obs, n, reconstruct_limb=True)
    for lead in (0, 1, 2):
        assert np.corrcoef(fused[lead], sig[lead])[0, 1] > 0.999, LEADS[lead]


def test_align_duplicates_recovers_the_shift():
    sig = _limb(4)[1]
    n = len(sig)
    short = np.full(n, np.nan)
    short[:1250] = shift_signal(sig, 6)[:1250]
    obs = align_duplicates([Observation(1, sig, np.ones(n)), Observation(1, short, np.isfinite(short) * 1.0)], 20)
    m = np.isfinite(obs[1].values)
    assert np.abs(obs[1].values[m] - sig[m]).max() < 1e-9


def test_limb_basis_is_einthoven_and_goldberger():
    s = np.random.default_rng(0).normal(size=(2, 100))
    limb = LIMB_BASIS @ s
    assert np.allclose(limb[0] - limb[1] + limb[2], 0)
    assert np.allclose(limb[3] + limb[4] + limb[5], 0)
