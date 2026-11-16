"""Command line interface: ``ecgdl digitize | train | cache | export``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _digitize(a: argparse.Namespace) -> None:
    from .constants import LEADS
    from .pipeline import Digitizer, DigitizerConfig

    config = DigitizerConfig(speed_mm_s=a.speed, gain_mm_mv=a.gain, flip_tta=a.tta, use_constraints=not a.no_constraints,
                             reconstruct_limb=a.reconstruct_limb)
    digitizer = Digitizer.from_checkpoints(a.page, a.trace, config, a.device)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in a.images:
        result = digitizer(path, debug=a.debug)
        stem = out_dir / Path(path).stem
        summary = result.to_dict() | {"image": str(path)}
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
        if result.ok:
            t = np.arange(result.signals.shape[1]) / result.fs
            table = np.concatenate([t[:, None], result.signals.T], 1)
            np.savetxt(stem.with_suffix(".csv"), table, delimiter=",", header="time_s," + ",".join(LEADS), comments="",
                       fmt="%.5f")
        if a.debug:
            from .viz import overlay

            overlay(path, result).save(str(stem) + "_debug.jpg", quality=88)
        status = f"{result.layout}, {result.quality.get('leads_found', 0)} leads" if result.ok else "; ".join(result.notes)
        print(f"{path}: {status} ({result.timings.get('total', 0):.2f} s)")


def _train(a: argparse.Namespace) -> None:
    from .train.run import main as train_main

    argv = [a.config, "--set", *a.set]
    if a.device:
        argv += ["--device", a.device]
    if a.threads:
        argv += ["--threads", str(a.threads)]
    train_main(argv)


def _cache(a: argparse.Namespace) -> None:
    import yaml

    from .data.cache import build_crop_cache, build_page_cache, load_bank
    from .data.targets import CropConfig

    if a.what == "pages":
        synth = yaml.safe_load(Path(a.synth).read_text()) if a.synth else None
        n = build_page_cache(a.out, load_bank(a.bank, a.split), a.n, synth, a.seed, a.batch, a.device or "cpu", a.start)
        print(f"{n} pages written to {a.out}")
    else:
        kw = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
        if "span_mm" in kw:
            kw["span_mm"] = tuple(kw["span_mm"])
        print(f"{build_crop_cache(a.pages, a.out, CropConfig(**kw), a.seed)} crops written to {a.out}")


def _export(a: argparse.Namespace) -> None:
    from .models.io import export_inference_checkpoint

    print(export_inference_checkpoint(a.checkpoint, a.out, half=not a.float32))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ecgdl", description="Digitize paper ECG images (deep learning + classical geometry).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("digitize", help="images -> CSV (mV at 500 Hz) + JSON")
    d.add_argument("images", nargs="+")
    d.add_argument("--page", required=True, help="PageNet checkpoint")
    d.add_argument("--trace", required=True, help="TraceNet checkpoint")
    d.add_argument("-o", "--out", default="out")
    d.add_argument("--speed", type=float, default=25.0, help="paper speed, mm/s")
    d.add_argument("--gain", type=float, default=10.0, help="gain, mm/mV")
    d.add_argument("--tta", action="store_true", help="average with the time-reversed crop (slower, slightly better)")
    d.add_argument("--no-constraints", action="store_true", help="skip the Einthoven/Goldberger projection")
    d.add_argument("--reconstruct-limb", action="store_true", help="also output limb leads where they are not printed")
    d.add_argument("--debug", action="store_true", help="write an overlay image")
    d.add_argument("--device", default=None)
    d.set_defaults(func=_digitize)

    t = sub.add_parser("train", help="train PageNet or TraceNet from a YAML config")
    t.add_argument("config")
    t.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    t.add_argument("--device", default=None)
    t.add_argument("--threads", type=int, default=None)
    t.set_defaults(func=_train)

    c = sub.add_parser("cache", help="pre-render pages / pre-extract crops for CPU training")
    c.add_argument("what", choices=["pages", "crops"])
    c.add_argument("--out", required=True)
    c.add_argument("--bank", default=None, help="signal bank (.npz); a parametric fixture is used without it")
    c.add_argument("--split", default="train")
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--synth", default=None, help="YAML with renderer settings (pages)")
    c.add_argument("--pages", default=None, help="page cache directory (crops)")
    c.add_argument("--config", default=None, help="YAML with crop settings (crops)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default=None)
    c.set_defaults(func=_cache)

    e = sub.add_parser("export", help="strip a training checkpoint for inference")
    e.add_argument("checkpoint")
    e.add_argument("out")
    e.add_argument("--float32", action="store_true", help="keep float32 instead of float16 storage")
    e.set_defaults(func=_export)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
