# Evaluation

All numbers on this page were measured with the code in this repository on the weights
in `weights/` (the CPU baseline described in the README). Nothing here is an official
leaderboard score.

## The weights that were evaluated

`weights/page_cpu.pt` and `weights/trace_cpu.pt`, trained with `configs/page_cpu.yaml` and
`configs/trace_cpu.yaml` on one CPU thread each for about 90 minutes (75 minutes at full
learning rate, then a 14-minute cosine decay), on 560 cached synthetic pages rendered
from 1,400 PTB-XL recordings. No real ECG image was used for training.

| Network | Steps | Held-out result (synthetic) |
|---|---|---|
| PageNet (2.6 M parameters) | 3,700 x 6 windows of 256 px | grid intersections: F1 0.946 (precision 0.934, recall 0.959) within 1.5 px, median error 0.145 px; ink Dice 0.74; direction error 3.5 degrees (median); lead names: recall 0.74, precision 0.22 |
| TraceNet (3.8 M parameters) | 2,966 x 8 crops of 96 x 320 px | 0.48 px mean absolute error; SNR 17.1 dB (median 18.2) per crop, offset removed |

The validation pages come from PTB-XL patients that are not in the training split.

## PM-ECG-ID (real images)

[PM-ECG-ID](https://doi.org/10.5281/zenodo.13617673): 6,600 images of 100 PTB-XL ECGs in
60 capture scenarios, with ground truth. Protocol of the PMcardio paper, as implemented
in `benchmarks/pm_ecg_id.py`: per lead, alignment of up to +-0.5 s, offset removed by the
median, then PCC, RMSE and SNR; a lead that is missing or less than half covered is a
failure and counts as SNR 0 and PCC 0.

**Sample:** one ECG per scenario (60 ECGs, 66 images, seed 0; the list is in
`docs/results/pm_ecg_id_sample_images.csv`). Paper speed 25 mm/s and gain 10 mm/mV were
given, as in the PMcardio protocol. The 97 PTB-XL patients behind PM-ECG-ID were excluded
from every training split.

| | SNR (dB) | PCC | Failed leads | SNR without failures | RMSE without failures |
|---|---|---|---|---|---|
| `ecgdl` CPU baseline, 60-ECG sample | **7.30** | **0.581** | **26.5%** | 9.94 dB | 0.117 mV |
| PMcardio, whole database (published) | > 12.5 | > 0.91 | 6.62% | | |
| `ecgdig` (classical), whole database (its README) | 7.91 | 0.656 | 15.3% | | |

Median time per image: 5.5 s on two CPU cores (PageNet at up to 1152 px, lattice,
layout with the referee, TraceNet on every row twice).

By scenario family (same sample):

| Scenario family | ECGs | SNR (dB) | PCC | Failed leads |
|---|---|---|---|---|
| Perspective (20-40) | 3 | 13.3 | 0.96 | 0% |
| Padding | 4 | 11.8 | 0.83 | 12% |
| JPEG compression | 6 | 11.7 | 0.79 | 12% |
| Brightness | 8 | 10.4 | 0.78 | 12% |
| Photos and scans | 7 | 8.1 | 0.66 | 18% |
| Contrast (gamma) | 8 | 8.0 | 0.61 | 22% |
| Resolution (1/2 to 1/32) | 5 | 6.2 | 0.50 | 40% |
| High-frequency noise | 3 | 3.9 | 0.45 | 11% |
| Blur (kernel 11-41) | 4 | 3.2 | 0.25 | 71% |
| Rotation (5-35 degrees) | 4 | 2.1 | 0.34 | 48% |
| Low-frequency noise | 3 | 1.7 | 0.24 | 28% |
| Faint ink (opacity) | 5 | 1.6 | 0.25 | 55% |

The distribution is bimodal. 24 of the 60 ECGs are read with a mean PCC of at least 0.9,
and on those the mean is 14.5 dB and PCC 0.963; 47.9% of all leads reach PCC 0.9 and
37.8% reach 12.5 dB. The other pages are mostly lost before the signal is decoded:

- **Not recovered at all** (blur 31-41, brightness -160, gamma 4-8, opacity 0.15-0.2,
  resolution 1/16-1/32, rotation 25): the grid or the rows are not found. Rotations of
  15 degrees and more were never rendered for training (the renderer's default tilt is
  a few degrees plus quarter turns); the same holds for very dark and very faint pages.
- **Arrangement misread** (for example a 3x4 page taken for a chest-only page, or a 6x2
  page for 12x1): lead names are not read on these pages, so the arrangement depends on
  the Einthoven referee, which gives up when the decoded rows are too noisy to close.
- **Chest leads of 6x2 pages**: on the page examined in detail (JPEG 22%), the limb column
  is right (PCC 0.98-0.99, 14-17 dB) while the six chest leads are not (PCC 0.2-0.8).
  Six 6x2 pages show the same signature (mean PCC 0.6-0.8). The likely cause, not yet
  confirmed, is the second column being misplaced in time by more than the 0.5 s the
  protocol tolerates: columns are laid out from the standard duration and the gutter
  between the two blocks is not measured.

Three warnings about this table:

1. **Sixty ECGs are a small sample.** `ecgdig` documents that a 120-ECG sample misled by
   0.5 dB; one page moves this mean by 0.3 dB. Decisions need the whole database.
2. **The layout rules were developed on this sample.** The networks never saw these
   images, but the fallback logic (rows, referee, multi-page formats) was debugged while
   looking at them over four runs (6.97 -> 7.07 -> 7.07 -> 7.30 dB). The number is
   optimistic for the layout stage; a fresh sample or the whole database is the fair test.
3. **Whole-database references are not paired with the sample.** They are shown for
   orientation.

## Synthetic pages (held out)

`benchmarks/synthetic.py` on the 24 validation pages of the training distribution: all 30
renderer layouts, three paper speeds, three gains, missing names, photographs and flat
prints. 17 pages are eligible (a grid is printed and the layout is sequential). The true
speed and gain of each page are given; the metric is `kaggle_snr` against the reference
in paper time.

| | Mean SNR | Median SNR | Pages with every segment found |
|---|---|---|---|
| Full system | 2.6 dB | 0.1 dB | 59% |
| Ground-truth page maps (grid, ink, names) + learned TraceNet | 5.7 dB | 3.7 dB | 59% |

This set is harder than PM-ECG-ID for the layout stage: non-standard arrangements, rhythm
strips of arbitrary leads, 50 mm/s pages at 2 px/mm, names deliberately removed. The two
pages whose layout is read reach 13.8 and 17.0 dB. Even with perfect page maps the mean is
5.7 dB, so **layout assembly, not perception, is the first thing to fix**: the row-based
fallback assumes standard arrangements with 10 s rows, and the name-based reading gives
up as soon as one name is missing.

## Kaggle PhysioNet

Not measured: the images need a Kaggle account. `benchmarks/kaggle_physionet.py`
implements the metric (alignment up to 0.2 s, offset removed, powers summed over the 12
leads) for the public training set. Reference points: about 23 dB at the top of the
private leaderboard; 14.68 dB for `ecgdig` on 135 training images.

## What was verified without trained weights

The test suite replaces the networks by oracles (`ecgdl.testing`), which isolates the
classical stages:

| Check | Result |
|---|---|
| Lattice from true intersections + 0.4 px noise, 15% missing, 30 false positives (8 rendered pages) | median map error 0.13 px, 95th percentile <= 0.35 px, exact topology |
| Analytic trace targets versus rendered ink (through the inference sampling path) | median offset 0.03 px |
| Column codec round trip on PTB-XL | 23 / 28 / 31 / 34 dB at 100 / 150 / 200 / 250 px/s |
| Decoders on held-out crops (earlier checkpoint) | soft-argmax 15.6 dB, hybrid 15.6-15.7 dB, argmax + local window 15.0 dB, Viterbi everywhere 13.2-15.1 dB |
| End to end on a procedurally drawn 3x4 + rhythm page with a classical darkest-ink tracer | layout and calibration exact; 8.6 dB (limited by the stand-in tracer) |
| Layout without any lead name (same drawn page) | 3x4 + rhythm recovered through the Einthoven referee; the rhythm strip assigned to lead II |

The hybrid decoder is therefore not a measured improvement yet: on clean synthetic crops
it equals the soft-argmax within 0.1 dB. It is kept because it cannot do worse on
unimodal columns and is the principled answer to crossing traces, which the held-out
crops barely contain.

## Reproduce

```bash
python benchmarks/pm_ecg_id.py fetch --out data/pm --per-scenario 1 --seed 0
python benchmarks/pm_ecg_id.py eval --data data/pm --page weights/page_cpu.pt --trace weights/trace_cpu.pt --out results/pm
ecgdl cache pages --bank data/ptbxl/bank.npz --split val --out data/cache/val --n 24 --seed 101 --synth configs/synth.yaml
python benchmarks/synthetic.py --pages data/cache/val --page weights/page_cpu.pt --trace weights/trace_cpu.pt
```

