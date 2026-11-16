"""PM-ECG-ID benchmark (PMcardio ECG Image Database, Zenodo 13617673, GPL-3.0-or-later).

6,600 images of 100 PTB-XL ECGs in 60 capture scenarios (phone photos, scans, blur,
perspective, compression ...), with the true signals. Protocol of Demolder et al.
(J Electrocardiol 2025): per lead, alignment of up to +-0.5 s, offset removed by the
median, then PCC, RMSE (mV) and SNR (dB). A lead that is missing or less than half
covered is a failure and counts as SNR 0 / PCC 0 in the whole-database means.

    python benchmarks/pm_ecg_id.py fetch --out data/pm --per-scenario 2
    python benchmarks/pm_ecg_id.py eval --data data/pm --page page.pt --trace trace.pt --out results/pm

The protocol code mirrors ``benchmarks/pmcardio`` of ecg-digitalization-classic so that
both digitizers are scored by identical rules. It is a reimplementation of the published
description, not the authors' evaluation code.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ecgdl.constants import LEADS
from ecgdl.metrics import pm_best_shift, pm_lead_metrics

URL = "https://zenodo.org/records/13617673/files/pmcardio_ecg_image_database.zip?download=1"
FS = 500
MAX_SHIFT = int(0.5 * FS)
#: published whole-database reference (PMcardio): for orientation, not a like-for-like run
REFERENCE = {"snr": 12.5, "pcc": 0.91, "fail": 0.0662}


def fetch(a: argparse.Namespace) -> None:
    """Download metadata, ground truth and a stratified sample of images by HTTP range requests."""
    import pandas as pd
    from remotezip import RemoteZip

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        for name in ("final_data/metadata.csv", "final_data/data/leads.npz", "final_data/data/rhythms.npz"):
            if not (out / name).exists():
                z.extract(name, out)
    m = pd.read_csv(out / "final_data/metadata.csv")
    m["scen"] = m["Image relative path"].str.split("/").str[0]
    if a.per_scenario:
        rng = np.random.default_rng(a.seed)
        keep = []
        for _, g in m.groupby("scen"):
            ids = rng.choice(sorted(g["Image ID"].unique()), min(a.per_scenario, g["Image ID"].nunique()), replace=False)
            keep += list(g[g["Image ID"].isin(ids)].index)
        m = m.loc[keep]
    paths = list(m["Image relative path"])

    def get(chunk: list[str]) -> None:
        with RemoteZip(URL) as z:
            for p in chunk:
                dst = out / "visual_data" / p
                if dst.exists():
                    continue
                for _ in range(5):
                    try:
                        z.extract("final_data/visual_data/" + p, out / "_tmp")
                        break
                    except Exception as e:  # noqa: BLE001 - network hiccups are retried
                        print("retry", p, e, flush=True)
                src = out / "_tmp/final_data/visual_data" / p
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(src, dst)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(get, [paths[i :: a.threads] for i in range(a.threads)]))
    pd.Series(paths, name="path").to_csv(out / "sel.csv", index=False)
    print("images", len(paths))


def evaluate_ecg(gt: np.ndarray, pages: list[dict]) -> dict[str, dict | None]:
    """``gt``: ``[n, 12]`` printed cut of each lead; ``pages``: digitizer outputs of the ECG's pages."""
    out: dict[str, dict | None] = {}
    n = gt.shape[0]
    for j, name in enumerate(LEADS):
        g = gt[:, j].astype(float)
        best = None
        for pg in pages:
            if not pg["ok"] or not pg["segments"].get(name):
                continue
            signal = pg["signals"][j].astype(float)
            first, last = min(pg["segments"][name], key=lambda s: abs((s[1] - s[0] + 1) - n))
            cut = pm_best_shift(g, signal, int(first), MAX_SHIFT, span=int(last - first + 1 - n))
            if cut is None:
                continue
            m = pm_lead_metrics(g, cut)
            if m and (best is None or m["pcc"] > best["pcc"]):
                best = m
        out[name] = best
    return out


def summarize(rows: list[dict]) -> tuple[str, dict]:
    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for name in LEADS:
            m = r["leads"].get(name)
            for key in ("ALL", r["scenario"]):
                agg[key]["snr"].append(m["snr"] if m else 0.0)
                agg[key]["pcc"].append(m["pcc"] if m else 0.0)
                agg[key]["fail"].append(m is None)
                if m:
                    agg[key]["snr_ok"].append(m["snr"])
                    agg[key]["rmse_ok"].append(m["rmse"])
    lines = ["| scenario | ECGs | SNR (dB) | PCC | failed leads | SNR without failures | RMSE without failures (mV) |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for k in ["ALL"] + sorted(x for x in agg if x != "ALL"):
        d = agg[k]
        s = {"n": len(d["snr"]) // 12, "snr": float(np.mean(d["snr"])), "pcc": float(np.mean(d["pcc"])),
             "fail": float(np.mean(d["fail"])), "snr_ok": float(np.mean(d["snr_ok"])) if d["snr_ok"] else 0.0,
             "rmse_ok": float(np.mean(d["rmse_ok"])) if d["rmse_ok"] else float("nan")}
        summary[k] = s
        lines.append(f"| {k} | {s['n']} | {s['snr']:.2f} | {s['pcc']:.3f} | {100 * s['fail']:.1f}% | "
                     f"{s['snr_ok']:.2f} | {s['rmse_ok']:.3f} |")
    return "\n".join(lines), summary


def evaluate(a: argparse.Namespace) -> None:
    import pandas as pd
    import torch

    from ecgdl.pipeline import Digitizer, DigitizerConfig

    if a.threads:
        torch.set_num_threads(a.threads)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(data / "final_data/metadata.csv")
    if (data / "sel.csv").exists():
        m = m[m["Image relative path"].isin(set(pd.read_csv(data / "sel.csv").iloc[:, 0]))]
    m = m[[(data / "visual_data" / p).exists() for p in m["Image relative path"]]]
    if a.limit:
        m = m.iloc[: a.limit]
    truth = np.load(data / "final_data/data/leads.npz")
    digitizer = Digitizer.from_checkpoints(a.page, a.trace, DigitizerConfig(flip_tta=a.tta), a.device)
    cache: dict[str, dict] = {}
    for p in m["Image relative path"]:
        t0 = time.time()
        try:
            r = digitizer(data / "visual_data" / p)
            cache[p] = {"ok": r.ok, "signals": r.signals, "segments": r.segment_samples, "layout": r.layout,
                        "notes": r.notes, "sec": time.time() - t0, "timings": r.timings}
        except Exception as e:  # noqa: BLE001 - a crash on one image is a failure of that image
            cache[p] = {"ok": False, "signals": None, "segments": {}, "layout": None,
                        "notes": [f"{type(e).__name__}: {e}"], "sec": time.time() - t0, "timings": {}}
        print(p, cache[p]["layout"], cache[p]["notes"][:1], f"{cache[p]['sec']:.1f}s", flush=True)
    m = m.assign(scen=m["Image relative path"].str.split("/").str[0])
    rows = []
    for (scen, ecg, iid), g in m.groupby(["scen", "ECG ID", "Image ID"]):
        pages = [cache[p] for p in g["Image relative path"]]
        rows.append({"scenario": scen, "ecg": ecg, "image_id": int(iid), "format": g["ECG format"].iloc[0],
                     "ok": all(p["ok"] for p in pages), "layout": [p["layout"] for p in pages],
                     "notes": [p["notes"] for p in pages], "sec": sum(p["sec"] for p in pages),
                     "leads": evaluate_ecg(truth[ecg], pages)})
    table, summary = summarize(rows)
    ref = (f"\nPublished PMcardio reference on the whole database: SNR > {REFERENCE['snr']} dB, "
           f"PCC > {REFERENCE['pcc']}, {100 * REFERENCE['fail']:.2f}% failed leads.\n")
    (out / "summary.md").write_text(table + "\n" + ref)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "results.json").write_text(json.dumps(rows, default=float))
    print(table + ref)
    print("median seconds per image:", round(float(np.median([c["sec"] for c in cache.values()])), 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--per-scenario", type=int, default=2, help="ECGs per scenario (0 = everything, 34 GB)")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threads", type=int, default=6)
    f.set_defaults(func=fetch)
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--page", required=True)
    e.add_argument("--trace", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--tta", action="store_true")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--threads", type=int, default=0)
    e.set_defaults(func=evaluate)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
