import numpy as np

from ecgdl.data.signals import synthetic_ecg
from ecgdl.metrics import aggregate_snr, kaggle_snr, pm_best_shift, pm_lead_metrics, snr_db
from ecgdl.signal.resample import shift_signal


def test_kaggle_snr_is_invariant_to_offset_and_small_shift():
    gt = synthetic_ecg(1, seed=5)[0].astype(float)
    pred = np.stack([shift_signal(x, 12) + 0.4 for x in gt])  # 24 ms and 0.4 mV off
    aligned, per = kaggle_snr(gt, pred, 500)
    raw, _ = kaggle_snr(gt, pred, 500, align=False)
    assert aligned > 40 and raw < 5
    assert len(per) == 12


def test_kaggle_snr_only_scores_printed_samples_and_missing_prediction_is_zero():
    gt = synthetic_ecg(1, seed=6)[0].astype(float)
    gt[0, 1250:] = np.nan
    pred = gt.copy()
    pred[0, 1250:] = 123.0  # outside the printed window: ignored
    assert kaggle_snr(gt, np.nan_to_num(pred), 500)[0] > 60
    pred2 = gt.copy()
    pred2[5] = np.nan  # a lead that was not digitized counts as all-zero
    snr, per = kaggle_snr(gt, pred2, 500)
    assert abs(per["aVF"]) < 0.5 and snr < 20


def test_snr_value():
    x = np.sin(np.arange(1000) / 10)
    assert abs(snr_db(x, x + 0.1 * np.cos(np.arange(1000) / 10)) - 20) < 0.3


def test_aggregate_modes():
    assert aggregate_snr([10, 20]) == 15
    assert abs(aggregate_snr([10, 20], "db_of_mean") - 10 * np.log10(55)) < 1e-9


def test_pm_metrics_and_shift():
    g = synthetic_ecg(1, seed=7)[0, 1, :1250].astype(float)
    page = np.full(5000, np.nan)
    page[100:1350] = g + 0.2
    cut = pm_best_shift(g, page, 0, 250)
    m = pm_lead_metrics(g, cut)
    assert m["pcc"] > 0.999 and m["rmse"] < 1e-6 and m["snr"] > 60
    assert pm_lead_metrics(g, np.full(1250, np.nan)) is None
