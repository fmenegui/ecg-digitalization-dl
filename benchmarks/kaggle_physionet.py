"""Kaggle "PhysioNet - Digitization of ECG Images" benchmark on the public training set.

Expected folder (as distributed by the competition):

    train.csv                    id, fs, sig_len
    train/<id>/<id>.csv          12 columns (I..V6); empty where the lead is not printed
    train/<id>/<id>-NNNN.png     captures of the same record (0001 render, 0003/0004 scans,
                                 0005/0006 photos, 0009-0012 degraded)

    python benchmarks/kaggle_physionet.py --data /path/to/physionet-ecg-image-digitization \
        --page page.pt --trace trace.pt --out results/kaggle --records 20

Metric: per image, alignment of up to +-0.2 s and offset removal per lead, then
10 log10(sum signal^2 / sum noise^2) over the 12 leads (see ``ecgdl.metrics.kaggle_snr``).
A reimplementation: the official leaderboard may differ in details. The data needs a
Kaggle account and is not downloaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ecgdl.constants import LEADS
from ecgdl.metrics import aggregate_snr, kaggle_snr

TYPES = ("0001", "0003", "0004", "0005", "0006", "0009", "0010", "0011", "0012")


def load_truth(path: Path) -> np.ndarray:
    rows = list(csv.reader(path.open()))
    header = rows[0]
    out = np.full((12, len(rows) - 1), np.nan)
    for k, lead in enumerate(LEADS):
        if lead in header:
            c = header.index(lead)
            out[k] = [float(r[c]) if c < len(r) and r[c] != "" else np.nan for r in rows[1:]]
    return out


def to_rate(signals: np.ndarray, fs_in: float, fs_out: float, n: int) -> np.ndarray:
    """Resample ``[12, m]`` to ``n`` samples at ``fs_out``; unknown samples stay NaN."""
    out = np.full((12, n), np.nan)
    t_out, t_in = np.arange(n) / fs_out, np.arange(signals.shape[1]) / fs_in
    for i in range(12):
        ok = np.isfinite(signals[i])
        if ok.sum() < 2:
            continue
        v = np.interp(t_out, t_in[ok], signals[i][ok], left=np.nan, right=np.nan)
        v[~np.isfinite(np.interp(t_out, t_in, np.where(ok, 0.0, np.nan)))] = np.nan
        out[i] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--page", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    from ecgdl.pipeline import Digitizer, DigitizerConfig

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((data / "train.csv").open()))
    random.Random(a.seed).shuffle(rows)
    digitizer = Digitizer.from_checkpoints(a.page, a.trace, DigitizerConfig(flip_tta=a.tta), a.device)
    results = []
    for row in rows[: a.records]:
        rid, fs = row["id"], float(row["fs"])
        truth = load_truth(data / "train" / rid / f"{rid}.csv")
        for typ in TYPES:
            image = data / "train" / rid / f"{rid}-{typ}.png"
            if not image.exists():
                continue
            t0 = time.time()
            r = digitizer(image)
            rec = {"id": rid, "type": typ, "ok": r.ok, "layout": r.layout, "notes": r.notes, "seconds": time.time() - t0}
            if r.ok:
                pred = to_rate(r.signals.astype(float), r.fs, fs, truth.shape[1])
                rec["snr"], rec["per_lead"] = kaggle_snr(truth, pred, fs)
                rec["snr_raw"], _ = kaggle_snr(truth, pred, fs, align=False)
            else:
                rec["snr"], rec["per_lead"] = 0.0, {}  # nothing digitized: all-zero prediction
            results.append(rec)
            print(rid, typ, rec["layout"], round(rec["snr"], 2), rec["notes"][:1], flush=True)
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r["snr"])
    lines = ["| capture type | images | SNR, mean of dB | SNR, dB of mean ratio |", "|---|---|---|---|"]
    for t in sorted(by_type):
        lines.append(f"| {t} | {len(by_type[t])} | {aggregate_snr(by_type[t]):.2f} | {aggregate_snr(by_type[t], 'db_of_mean'):.2f} |")
    every = [r["snr"] for r in results]
    lines.append(f"| all | {len(every)} | {aggregate_snr(every):.2f} | {aggregate_snr(every, 'db_of_mean'):.2f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(results, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
