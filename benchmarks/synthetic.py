"""End-to-end score on held-out synthetic pages, with an error decomposition.

    python benchmarks/synthetic.py --pages data/cache/val --page page.pt --trace trace.pt

Each page is digitized three ways:

* ``full``          learned PageNet + learned TraceNet (the real system);
* ``oracle-page``   ground-truth grid points, names and ink, learned TraceNet
                    (what remains is trace decoding and layout assembly);
* the difference between the two is what page understanding costs.

The reference is in *paper time* (see ``ecgdl.testing.paper_time_ground_truth``) and uses
the true paper speed and gain of every page, so the score isolates digitization quality
from calibration reading. Metric: ``ecgdl.metrics.kaggle_snr``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ecgdl.data.synth import PageSample
from ecgdl.metrics import kaggle_snr
from ecgdl.models.io import load_checkpoint
from ecgdl.pipeline import Digitizer, DigitizerConfig
from ecgdl.testing import OraclePageNet, paper_time_ground_truth


def score(digitizer: Digitizer, sample: PageSample) -> dict:
    image = torch.from_numpy(sample.image.copy()).permute(2, 0, 1).float() / 255
    t0 = time.time()
    r = digitizer(image, speed_mm_s=sample.speed, gain_mm_mv=sample.gain)
    rec = {"ok": r.ok, "layout": r.layout, "notes": r.notes[:2], "seconds": time.time() - t0,
           "segments": len(r.segments), "expected_segments": int(len(sample.segments)), "timings": r.timings}
    if not r.ok:
        rec["snr"] = 0.0
        return rec
    gt = paper_time_ground_truth(sample)
    n = min(gt.shape[1], r.signals.shape[1])
    pred = np.full_like(gt, np.nan)
    pred[:, :n] = r.signals[:, :n]
    rec["snr"], rec["per_lead"] = kaggle_snr(gt, pred, 500)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", required=True, help="directory of cached PageSample files")
    ap.add_argument("--page", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--standard-only", action="store_true", help="only 25 mm/s, 10 mm/mV pages")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    page_net = load_checkpoint(a.page, kind="page")
    trace_net = load_checkpoint(a.trace, kind="trace")
    files = sorted(Path(a.pages).glob("*.npz"))
    rows = []
    for f in files[: a.limit or None]:
        sample = PageSample.load(str(f))
        if len(sample.segments) == 0 or len(sample.grid_xy) < 50:
            continue  # no grid printed: outside the scope of grid-based rectification
        if a.standard_only and (sample.speed != 25 or sample.gain != 10):
            continue
        # non-sequential pages repeat the same time window in every column: paper time is undefined
        if any(s[2] > sample.segments[:, 2].min() + 5 and s[5] == 0 for s in sample.segments):
            continue
        full = score(Digitizer(page_net, trace_net, DigitizerConfig()), sample)
        oracle = score(Digitizer(OraclePageNet(sample), trace_net, DigitizerConfig()), sample)
        rows.append({"file": f.name, "speed": sample.speed, "gain": sample.gain, "px_per_mm": sample.px_per_mm,
                     "full": full, "oracle_page": oracle})
        print(f.name, f"full {full['snr']:.1f} dB ({full['segments']}/{full['expected_segments']} segments)",
              f"| oracle-page {oracle['snr']:.1f} dB", full["notes"][:1], flush=True)
    if not rows:
        print("no eligible pages")
        return

    def stats(key: str) -> dict:
        v = np.array([r[key]["snr"] for r in rows])
        return {"pages": len(v), "mean_snr_db": float(v.mean()), "median_snr_db": float(np.median(v)),
                "failed": float(np.mean([not r[key]["ok"] for r in rows])),
                "all_segments": float(np.mean([r[key]["segments"] == r[key]["expected_segments"] for r in rows])),
                "median_seconds": float(np.median([r[key]["seconds"] for r in rows]))}

    summary = {"full": stats("full"), "oracle_page": stats("oracle_page")}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "synthetic.json").write_text(json.dumps({"summary": summary, "rows": rows}, default=float))


if __name__ == "__main__":
    main()
