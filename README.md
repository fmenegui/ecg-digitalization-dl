# ecgdl: deep-learning ECG image digitizer

`ecgdl` turns a photo or scan of a paper ECG into 12-lead signals in millivolts at 500 Hz.
Two small networks do the perception; classical geometry, dynamic programming and
physiology do everything that follows exact rules.

```
photo -> PageNet (grid points, ink, lead names, direction)
      -> lattice: metric map of the paper, no page template
      -> layout: rows, columns, lead identities (names first, Einthoven as referee)
      -> TraceNet on metric crops sampled once from the original photo
      -> column distributions -> sub-pixel trace -> mV(t)
      -> confidence-weighted fusion on the Einthoven/Goldberger subspace
```

It is the deep-learning sibling of the classical digitizer
[`ecgdig`](https://github.com/fmenegui/ecg-digitalization-classic) and shares its output
contract. Design rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). What the best
public systems do and what was taken from each:
[`docs/STATE_OF_THE_ART.md`](docs/STATE_OF_THE_ART.md).

## Status

**This is a first baseline, not yet a state-of-the-art result.** The code, tests,
training recipe and benchmarks are complete. The weights in `weights/` were trained for
about 90 minutes on a 2-core CPU, on synthetic pages only (560 pages, 1,400 PTB-XL
recordings). The full-size configurations (`configs/*_gpu.yaml`) have not been trained.

Measured with these weights (details and caveats in [`docs/EVALUATION.md`](docs/EVALUATION.md)):

| Benchmark | `ecgdl` CPU baseline | Reference |
|---|---|---|
| [PM-ECG-ID](https://doi.org/10.5281/zenodo.13617673), stratified sample: 60 ECGs, one per capture scenario (66 real images) | SNR **7.30 dB**, PCC **0.581**, failed leads **26.5%**; 5.5 s per image on 2 CPU cores | PMcardio, whole database (published): SNR > 12.5 dB, PCC > 0.91, 6.62% failed. Classical `ecgdig` on the same 60 ECGs: 8.00 dB, PCC 0.688, 12.6% failed; 18 s per image. |
| Same sample, the 24 ECGs read with mean PCC >= 0.9 | SNR 14.5 dB, PCC 0.963 | PMcardio "easy" scenarios (published): 16.7 dB, PCC 0.977 |
| Same sample, phone photo (iPhone, 3x4 + 3 rhythm strips) | SNR 15.2 dB, PCC 0.982 | |
| Held-out trace crops (synthetic, 420 crops) | SNR 17.1 dB (median 18.2), 0.48 px | codec ceiling at this resolution: 28 dB |
| Held-out synthetic pages, all 30 layouts, 3 speeds, 3 gains (17 pages) | SNR 2.6 dB mean; 13.8 and 17.0 dB on the pages whose layout is read | 5.7 dB with ground-truth page maps: layout reading is the bottleneck |
| Kaggle PhysioNet | not measured (the data need a Kaggle account) | top of the private leaderboard: about 23 dB |

What the numbers say:

- **The signal path works on real images.** Where the page is understood, a network
  trained for 90 CPU-minutes on synthetic data alone reads real scans and phone photos
  at PCC 0.96-0.99 and 12-20 dB.
- **Page understanding is the weak link.** 26.5% of leads fail and the mean falls to
  7.3 dB because of pages that are not understood: heavy blur, darkness, faint ink,
  rotation beyond what the baseline was trained on, and layouts the fallback logic does
  not cover. Lead names are essentially not read on real pages yet (the baseline name
  detector has 22% precision even on synthetic pages), so the arrangement comes from
  the row structure plus the Einthoven referee.
- **It does not beat PMcardio, the Kaggle winners or our own classical digitizer today.**
  On the same 60 ECGs `ecgdig` is ahead overall (PCC 0.688 against 0.581, half as many
  failed leads). `ecgdl` is about three times faster and more accurate on the leads it
  does read (9.9 against 9.2 dB without failures); it is clearly better on 12 ECGs,
  `ecgdig` on 21, and taking the better of the two per ECG would give PCC 0.76, so the
  two fail on different pages. The path from here is training at scale and on real ink
  ([`docs/ROADMAP.md`](docs/ROADMAP.md)).


## Install and use

```bash
pip install -e .
ecgdl digitize ecg.jpg --page weights/page_cpu.pt --trace weights/trace_cpu.pt --debug -o out
# out/ecg.csv (time + 12 leads, mV), out/ecg.json (layout, quality, notes), out/ecg_debug.jpg
```

```python
from ecgdl import Digitizer

digitizer = Digitizer.from_checkpoints("weights/page_cpu.pt", "weights/trace_cpu.pt")
result = digitizer("ecg.jpg")            # speed_mm_s=25, gain_mm_mv=10 unless given
result.signals        # [12, n] float32, mV at 500 Hz, NaN where a lead is not printed
result.confidence     # [12, n]
result.layout         # e.g. "3x4+1R"
result.quality        # lattice residual, px/mm, Einthoven residual, per-lead confidence
result.notes          # every assumption and repair, in words
```

Paper speed and gain are inputs (25 mm/s and 10 mm/mV by default); they are not read
from the page yet.

## What is different

| | Competition pipelines | PMcardio / Open ECG Digitizer | `ecgdl` |
|---|---|---|---|
| Geometry | keypoints registered to one page template | grid matrix + per-cell undistortion / global Hough perspective | grid lattice grown from detections, homography + robust smooth residual field; no template |
| Layout | fixed by the template | format detector / name templates | read from lead names; standard arrangements as hypotheses, verified by Einthoven |
| Trace | column distribution, soft-argmax, SNR loss | segmentation mask, then tracing | column distribution on target-centred metric crops, soft-argmax + Viterbi in ambiguous columns, SNR loss |
| Resampling | page rectified, then cropped | page rectified, then segmented | every crop sampled once from the original photo |
| Lead redundancy | averaging / heuristics | not used | weighted least squares with free offsets; restores -aVR, repairs low-confidence stretches |
| Unreadable names | not needed (one layout) | failure or manual format | rows decoded first, then the arrangement, the column boundaries and the way up are chosen by which reading makes the limb leads consistent |

## Train

```bash
pip install -e ".[train,bench]"
python scripts/fetch_ptbxl.py --out data/ptbxl          # real signals, patient-disjoint splits
ecgdl train configs/page_gpu.yaml                       # GPU: pages synthesised online
ecgdl train configs/trace_gpu.yaml
make cache && make train-cpu                            # or the CPU baseline from cached pages
```

Details: [`docs/TRAINING.md`](docs/TRAINING.md). Synthetic pages come from
[`ecg-torch-renderer`](https://github.com/fmenegui/ecg-torch-renderer).

## Benchmarks

```bash
python benchmarks/pm_ecg_id.py fetch --out data/pm --per-scenario 2
python benchmarks/pm_ecg_id.py eval --data data/pm --page weights/page_cpu.pt --trace weights/trace_cpu.pt --out results/pm
python benchmarks/kaggle_physionet.py --data <kaggle folder> --page ... --trace ... --out results/kaggle
python benchmarks/synthetic.py --pages data/cache/val --page ... --trace ...
```

Protocols, numbers and how to read them: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Tests

```bash
pip install -e ".[dev]" && pytest -q && ruff check ecgdl tests scripts benchmarks
```

The suite needs neither the renderer nor trained weights: oracles stand in for the
networks (`ecgdl.testing`), so geometry, layout, calibration, decoding and fusion are
tested end to end on procedurally drawn pages.

## Known limitations

- The published weights are a CPU baseline. Lead names are not read reliably on real
  pages, so layouts are mostly recovered through the row structure and the Einthoven
  referee, which covers standard arrangements only.
- Pages with chest leads only (one page of a multi-page print) cannot be verified by
  physiology; V1-V3 versus V4-V6 is then an assumption from the QRS polarity.
- On 6x2 pages the second column is placed from the standard duration; the gutter between
  the two blocks is not measured, which appears to cost the chest leads on several
  PM-ECG-ID pages.
- Rotations of 15 degrees and more, very dark and very faint pages were not in the
  training distribution and mostly fail.
- Paper speed and gain are not read from the page.
- A page without a printed 5 mm grid cannot be rectified (no grid-free fallback yet).
- Non-sequential layouts (every column showing the same time window) are placed in time
  as if sequential.
- Pacing spikes were not studied.

Plan: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). No third-party data are redistributed.

## References

- Demolder A et al. High precision ECG digitization using artificial intelligence. *J Electrocardiol* 2025.
- Iring A et al. PMcardio ECG Image Database (PM-ECG-ID). Zenodo 2024. doi:10.5281/zenodo.13617673.
- Stenhede E, Bjørnstad AM, Ranjbar A. Digitizing paper ECGs at scale: an open-source algorithm for clinical research. *npj Digit Med* 2026.
- Reyna MA et al. Digitization and classification of ECG images: the George B. Moody PhysioNet Challenge 2024. *CinC* 2024.
- Krones F et al. Combining Hough transform and deep learning approaches to reconstruct ECG signals from printouts. 2024.
- Shivashankara KK et al. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. *Physiol Meas* 2024.
- Wagner P et al. PTB-XL, a large publicly available electrocardiography dataset. *Sci Data* 2020.
- Garcia D. Robust smoothing of gridded data in one and higher dimensions with missing values. *Comput Stat Data Anal* 2010.
- Li X et al. Generalized focal loss. *NeurIPS* 2020.
- Public write-ups of the Kaggle competition "PhysioNet - Digitization of ECG Images" (2nd, 4th, 5th, 7th, 11th, 12th places), linked in `docs/STATE_OF_THE_ART.md`.
