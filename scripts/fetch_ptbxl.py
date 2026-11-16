"""Download a patient-disjoint subset of PTB-XL (500 Hz) as a signal bank for training.

    python scripts/fetch_ptbxl.py --out data/ptbxl --train 1500 --val 100 \
        --exclude-pm data/pm/final_data/metadata.csv

* One recording per patient; PTB-XL folds 1-8 -> train, 9 -> val, 10 -> test.
* Every patient whose ECG appears in PM-ECG-ID (the external benchmark, also drawn from
  PTB-XL) is removed from *all* splits, so the benchmark stays unseen.
* WFDB format 16 is parsed directly (no extra dependency) and converted to mV.

PTB-XL: Wagner et al., Scientific Data 2020, CC BY 4.0, https://physionet.org/content/ptb-xl/1.0.3/
"""
from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _get(url: str, dst: Path, retries: int = 4) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            dst.write_bytes(data)
            return
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise


def read_wfdb16(hea: Path) -> tuple[np.ndarray, float]:
    """Read a 12-lead format-16 record. Returns ``(signals[12, T] in mV, fs)``."""
    lines = [ln for ln in hea.read_text().splitlines() if ln and not ln.startswith("#")]
    head = lines[0].split()
    n_sig, fs, n_samp = int(head[1]), float(head[2]), int(head[3])
    names, gains, baselines = [], [], []
    for ln in lines[1 : 1 + n_sig]:
        f = ln.split()
        if f[1] != "16":
            raise ValueError(f"{hea.name}: unsupported storage format {f[1]}")
        m = re.match(r"([0-9.eE+-]+)(?:\(([-0-9]+)\))?", f[2])
        gains.append(float(m.group(1)))
        baselines.append(float(m.group(2) or 0))
        names.append(f[-1].upper())
    raw = np.fromfile(hea.with_suffix(".dat"), dtype="<i2")
    if raw.size != n_sig * n_samp:
        raise ValueError(f"{hea.name}: expected {n_sig * n_samp} samples, found {raw.size}")
    raw = raw.reshape(n_samp, n_sig).T.astype(np.float32)
    sig = (raw - np.array(baselines, np.float32)[:, None]) / np.array(gains, np.float32)[:, None]
    order = [names.index(n) for n in LEADS]
    return sig[order], fs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--exclude-pm", type=Path, default=None, help="PM-ECG-ID metadata.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = a.out / "ptbxl_database.csv"
    _get(BASE + "ptbxl_database.csv", meta)
    rows = list(csv.DictReader(meta.open()))

    excluded_ecg: set[int] = set()
    if a.exclude_pm is not None:
        for r in csv.DictReader(a.exclude_pm.open()):
            m = re.search(r"(\d+)_hr", r["ECG ID"])
            if m:
                excluded_ecg.add(int(m.group(1)))
    excluded_patients = {r["patient_id"] for r in rows if int(r["ecg_id"]) in excluded_ecg}

    rng = np.random.default_rng(a.seed)
    by_patient: dict[str, dict] = {}
    for r in rows:
        if r["patient_id"] in excluded_patients or r["patient_id"] in by_patient:
            continue
        by_patient[r["patient_id"]] = r
    pool = {"train": [], "val": [], "test": []}
    for r in by_patient.values():
        fold = int(r["strat_fold"])
        pool["train" if fold <= 8 else "val" if fold == 9 else "test"].append(r)
    chosen = []
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        idx = rng.permutation(len(pool[split]))[:n]
        chosen += [(split, pool[split][i]) for i in idx]

    def fetch(item: tuple[str, dict]) -> tuple[str, dict] | None:
        _, r = item
        rel = r["filename_hr"]
        try:
            _get(BASE + rel + ".hea", a.out / (rel + ".hea"))
            _get(BASE + rel + ".dat", a.out / (rel + ".dat"))
        except Exception as e:  # noqa: BLE001
            print("skip", rel, e, flush=True)
            return None
        return item

    with ThreadPoolExecutor(a.threads) as ex:
        done = [x for x in ex.map(fetch, chosen) if x is not None]

    signals, splits, ecg_ids, patients = [], [], [], []
    for split, r in done:
        try:
            sig, fs = read_wfdb16(a.out / (r["filename_hr"] + ".hea"))
        except Exception as e:  # noqa: BLE001
            print("bad record", r["filename_hr"], e, flush=True)
            continue
        if fs != 500 or sig.shape != (12, 5000) or not np.isfinite(sig).all():
            continue
        signals.append(sig)
        splits.append(split)
        ecg_ids.append(int(r["ecg_id"]))
        patients.append(r["patient_id"])
    np.savez_compressed(
        a.out / "bank.npz",
        signals=np.stack(signals).astype(np.float32),
        fs=np.float32(500),
        splits=np.array(splits),
        ecg_ids=np.array(ecg_ids),
        patient_ids=np.array(patients),
        excluded_ecg_ids=np.array(sorted(excluded_ecg)),
    )
    print(f"saved {len(signals)} records ({ {s: splits.count(s) for s in set(splits)} }), "
          f"excluded {len(excluded_patients)} PM-ECG-ID patients", flush=True)


if __name__ == "__main__":
    main()
