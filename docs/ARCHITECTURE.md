# Architecture

`ecgdl` splits the problem where the physics splits it. Perception, which depends on
appearance, is learned. Geometry, calibration and physiology, which follow exact rules,
are computed.

```
photo ──► PageNet ──► grid points ─► lattice ─► paper-mm map ─┐
             │                                                │
             ├─► lead names ─┐                                │
             └─► trace ink ──┴─► layout (rows, segments) ─────┤
                                                              ▼
                 original photo ──► metric crop per segment ─► TraceNet ─► column distributions
                                                                              │
                              signals [12, n] mV ◄─ fusion ◄─ mV(t) ◄─ decode ┘
```

| Stage | Kind | Module |
|---|---|---|
| 1. Page understanding | learned | `ecgdl.models.page_net` |
| 2. Lattice and metric map | classical | `ecgdl.geometry.lattice`, `ecgdl.geometry.rectify` |
| 3. Layout | classical | `ecgdl.layout.assemble` |
| 4. Trace decoding | learned + dynamic programming | `ecgdl.models.trace_net`, `ecgdl.signal.codec` |
| 5. Calibration and fusion | classical | `ecgdl.signal.resample`, `ecgdl.signal.constraints` |

## 1. PageNet

One U-Net pass over the downscaled photo returns, at input resolution:

* `grid` + `offset`: a heatmap of the **5 mm grid intersections** and the sub-pixel
  (dx, dy) to each centre. Only major intersections are positives, so the spacing of the
  detected lattice *is* 5 mm; the network, not a heuristic, decides between 1 mm and 5 mm lines.
* `trace`: ECG ink, excluding grid, text and calibration pulses.
* `names`: one heatmap per lead with the centre of each printed lead name.
* `direction`: a unit vector, the direction in which time increases. A vector has no
  decision boundary at 45 degrees, unlike a four-way rotation class.

The pass is scale-adaptive: if the detected grid spacing is outside the range the network
was trained on, the page is rescaled towards the target spacing and processed once more.

Losses: quality focal loss on the soft heatmaps (normalised by target mass), smooth-L1 on
offsets at their anchor pixels, BCE + Dice on ink, cosine loss on the direction.

## 2. Lattice: from points to a ruler

Input: an unordered, noisy, incomplete set of intersections with some false positives.

1. **Seeds.** Points whose nearest neighbours form a clean cross (two opposite pairs,
   roughly perpendicular) give a local basis.
2. **Best-first growth.** From a seed, the neighbour predicted with the smallest error is
   indexed next; its measured step updates the local basis. Perspective and curvature
   are followed gradually, and errors do not propagate because the most certain
   extension always goes first.
3. **Projective fit.** Homography `index -> pixel` by iteratively reweighted least
   squares (the flat-sheet, pinhole-camera part).
4. **Smooth residual field.** What remains (curl, folds, lens distortion) is fitted on
   the index grid by robust penalised least squares in the DCT domain. The same solve
   interpolates undetected nodes and extrapolates beyond the detected area. Outliers are
   re-weighted to zero, and detections that growth could not reach are re-attached.
5. **Orientation.** The grid axis closest to the predicted direction becomes time. Classical
   referees check the network: the *projection profile* (ink piles up on rows only in the
   right frame, which catches a quarter-turn error), the *name position* (lead names sit
   at the start of their segments) and, without names, the *limb-lead closure* (section 3),
   which both catch a half turn.

Measured on rendered pages with the true intersections perturbed by 0.4 px of noise per
axis, 15% dropped and 30 false positives added: median map error 0.13 px, 95th
percentile at most 0.35 px, exact topology on all eight pages, rotated and curled ones included.

The result is a function `paper mm -> photo pixel`. Every later crop is sampled **once**
from the original photo through it (with a box pre-filter when the photo is much sharper
than the crop). There is no intermediate rectified image, so interpolation blur is not
paid twice, and no template: the lattice assumes nothing but a 5 mm grid.

## 3. Layout

Two complementary readings in paper millimetres (`ecgdl.layout.assemble`), and a referee
that uses the signals themselves (`ecgdl.layout.verify`).

**Names first.** Each detected lead name is attached to a row: the row's baseline is a
peak of the ink profile next to the name, and when a name lies between two rows a
page-level vote on the name-to-trace offset decides (a device prints all its names at the
same relative place). Names on one baseline form a row, a segment runs from its name to
the next one, and segment starts are pooled across rows into page columns, which snaps
boundaries and reveals a missed name. This reading handles any arrangement, but it is
only trusted when it is complete and coherent.

**Rows first.** Rows are found from the ink alone (peaks of the row profile, then the
longest run of rows with a constant pitch, which drops page borders and lines of text).
Every standard arrangement with that many rows is a hypothesis: 3x4, 6x2, 12x1 with up to
three rhythm strips, in standard and Cabrera order, single pages of a multi-page print
(six or three full-width strips), and two-block pages whose columns have their own row
spacing. Names that were read vote; contradictory names are ignored.

**The referee.** When names do not settle the arrangement, every row is decoded as one
strip and physiology decides:

- rows whose decoded strip does not look like an ECG (flat, neither spiky nor periodic)
  are dropped, and up to two outer rows may be dropped if only then some arrangement closes;
- the six limb leads span two dimensions, so in any time window that shows three or more
  of them each one must be predictable from the others. The **closure residual**
  (leave-one-out, `1 - correlation` averaged with the relative RMS difference) is near 0
  for the right assignment of leads to rows and columns and near 1 otherwise;
- the same test **registers the columns**: boundaries are slid until the leads close,
  which removes the calibration pulse in front of the rows;
- and it tells **which way is up**: a half turn reverses the rows, reverses time and
  inverts the voltage, and only one of the two readings closes;
- a page where nothing can close is taken for chest leads only (V1-V3 versus V4-V6 is
  then assumed from the QRS polarity and reported as an assumption);
- a rhythm strip without a name is identified by its correlation with the other leads.

Time comes from position: the same x on paper is the same instant on every row.

## 4. TraceNet and the column codec

A printed lead is a function y(x). TraceNet therefore outputs, for each column of a crop,
a distribution over rows, and the position is its expectation (soft-argmax): sub-pixel by
construction, and trained directly on a loss tied to SNR.

* **Target-centred crops.** The crop is centred on the segment's baseline; a
  row-coordinate input channel tells the network where the centre is. Neighbouring
  leads, text and handwriting may cross the crop; "which stroke is mine" is learned
  from simulated baseline jitter, not resolved by post-hoc heuristics.
* **Anisotropic resolution.** The crop has more pixels per millimetre along time than
  along voltage, because time resolution is what limits SNR (next section).
* **One network for all amplitudes.** The vertical extent of the crop varies during
  training. At inference the crop spans +-2 mV at the page gain, and a segment whose
  decoded trace touches the border is decoded again with a taller crop.
* **Hybrid decoder.** Where a column distribution is unimodal, the answer is the global
  soft-argmax the network was trained with. Where two strokes compete, the expectation
  would land between them; a truncated-linear Viterbi path through the neighbouring
  columns picks the stroke, and a local expectation around it restores sub-pixel precision.
* **Visibility head.** A per-column logit marks occluded or missing stretches, which
  become NaN ("unknown stays unknown") and low weights in the fusion.

Loss: column cross-entropy against a narrow Gaussian + smooth-L1 on the soft-argmax +
negative SNR (offset removed, as in the benchmarks) + visibility BCE.

### The codec ceiling

Encoding a signal as one value per column and resampling it back loses information.
Measured on PTB-XL (60 records, three leads, 2.5 s windows, 25 mm/s, box-averaged columns,
cubic resampling to 500 Hz):

| Horizontal resolution | px/s | SNR of the codec alone |
|---|---|---|
| 4 px/mm | 100 | 23 dB |
| 6 px/mm | 150 | 28 dB |
| 8 px/mm | 200 | 31 dB |
| 10 px/mm | 250 | 34 dB |

Competition pipelines work at about 200 px/s, a ceiling of roughly 31 dB before any
model error. `ecgdl` decouples the two axes so the time axis can be pushed further at
little cost: the full-size configuration uses 12 px/mm along time and 256 rows.

## 5. Calibration and fusion

* Pixels to millivolts and seconds use the lattice scale and the paper speed and gain
  (25 mm/s and 10 mm/mV unless given).
* Columns are resampled to 500 Hz with a cubic spline; gaps stay NaN.
* **Duplicates** (a short segment and a rhythm strip of the same lead) are aligned by
  cross-correlation and averaged with the decoder's confidence.
* **Limb leads** live in a two-dimensional space (Einthoven, Goldberger). All printed
  limb segments are explained jointly by `y_k(t) = a_k . s(t) + o_k`, solved by weighted
  least squares for the two independent leads `s(t)` and one free offset `o_k` per
  segment (the 0 mV line is not printed). A low-confidence stretch of one lead is
  repaired by the others instead of contaminating them, and limb leads can optionally be
  reconstructed where they are not printed.
* **-aVR** (Cabrera) is detected from its negative correlation with the value predicted
  by the other limb leads, and restored.

## Training data

Pages are synthesised by [ecg-torch-renderer](https://github.com/fmenegui/ecg-torch-renderer)
from real PTB-XL recordings: 30 layouts, three speeds and gains, perspective, curl,
creases, lighting, blur, noise, JPEG, stains and occlusions, with exact targets from the
same render. Trace crops are produced *through the inference code path*: the lattice is
fitted to (perturbed) true intersections, the crop is sampled by the same function used
at test time, and the target is computed analytically from the source signal and the
print calibration, never from pixels. On rendered pages the analytic target sits on the
drawn ink within 0.03 px (median).

Patients are disjoint across splits, and every patient whose ECG appears in PM-ECG-ID is
removed from all training data.

## Output contract

`DigitizationResult.signals` is `[12, n]` float32 in mV at 500 Hz in the order
I, II, III, aVR, aVL, aVF, V1-V6, NaN where a lead is not printed, with a matching
`confidence` array, the list of printed segments, the recovered lattice, per-stage
timings, quality indicators (lattice residual, Einthoven residual, clipped fraction) and
notes about every assumption or repair. It matches the contract of `ecgdig`, so both
digitizers can be scored by the same benchmark code.
