# Training

Two networks are trained independently on synthetic pages rendered from real signals.

## 1. Signals

```bash
pip install -e ".[train,bench]"
python scripts/fetch_ptbxl.py --out data/ptbxl --train 1500 --val 100 --test 100 \
    --exclude-pm data/pm/final_data/metadata.csv      # optional but recommended, see below
```

One 500 Hz recording per patient from PTB-XL; folds 1-8 train, 9 validation, 10 test.
`--exclude-pm` removes every patient whose ECG is part of PM-ECG-ID, the external
benchmark, from all splits (97 patients). Get that metadata file with
`python benchmarks/pm_ecg_id.py fetch --out data/pm --per-scenario 1`.
Any `[N, 12, T]` array in mV with a `splits` vector works as a bank; use more records for
real training (the whole of PTB-XL, CODE-15, your own archive).

## 2a. GPU: synthesise online

```bash
ecgdl train configs/page_gpu.yaml
ecgdl train configs/trace_gpu.yaml
```

The renderer runs on the training device, so every batch is new: pages at 1536 x 2048,
ConvNeXt-S encoders (`pip install -e ".[timm]"`), trace crops of 256 x 1024 at 12 px/mm
along time. These configurations are the intended full-size models. **They have not been
trained yet**: the published baseline below was trained on a 2-core CPU. Override any
entry from the command line:

```bash
ecgdl train configs/trace_gpu.yaml --set train.steps=200000 data.pages_per_batch=8
```

Training is resumable: rerunning the command continues from `runs/<name>/last.pt`.
`train.max_minutes` stops cleanly after a time budget.

## 2b. CPU: cache, then train

```bash
make cache        # 600 pages take about 15 minutes on 2 cores; crops take 1-2 minutes
make train-cpu
```

`configs/page_cpu.yaml` trains on random 256 px windows of cached pages;
`configs/trace_cpu.yaml` trains on memory-mapped 96 x 320 crops (6 px/mm along time).

## 3. Use the weights

```bash
ecgdl export runs/trace_cpu/last.pt weights/trace.pt     # EMA weights only, float16
ecgdl export runs/page_cpu/last.pt weights/page.pt
ecgdl digitize photo.jpg --page weights/page.pt --trace weights/trace.pt --debug -o out
```

## What is monitored

* PageNet (full validation pages): F1, precision and recall of grid intersections within
  1.5 px, median localisation error, ink Dice, median direction error, recall and
  precision of lead names.
* TraceNet (held-out crops): mean absolute error in pixels and offset-free SNR of the
  decoded trace.
* End to end: `benchmarks/synthetic.py` (with an oracle-page ablation that separates page
  understanding from trace decoding), `benchmarks/pm_ecg_id.py`, `benchmarks/kaggle_physionet.py`.

## Recipe notes

* **Renderer settings** are in `configs/synth.yaml` (or inline under `data.synth`). Fixed
  values, `[choice, [...]]`, `[uniform, lo, hi]` and `[bernoulli, p]` are accepted for
  every renderer parameter. Standard speed and gain dominate; rarer ones are kept.
* **Trace crops** are built through the inference path (lattice fit, single resampling).
  `baseline_jitter_mm` simulates the error of the baseline estimate; `span_mm` the range
  of vertical scales; `point_noise_px` the error of detected intersections.
* **Augmentation on crops:** time reversal and polarity reversal (both exact for the
  target), brightness, contrast, gamma, colour cast, grayscale, noise, blur.
* **EMA** weights are the ones saved for inference.

## Next steps that need real data

Synthetic-only training leaves a domain gap. The competition winners trained on the
Kaggle images with ground truth; the same can be done here without manual labels:

1. run the digitizer on the Kaggle training images;
2. where the lattice and layout are confident, cut metric crops and align the true
   signal to each crop (time shift and offset, as the metric does);
3. fine-tune TraceNet on those crops mixed with synthetic ones.

This self-training loop is not implemented yet (see `ROADMAP.md`).
