# Roadmap

Ordered by expected gain per unit of work. The first block is what separates the current
CPU baseline from the targets (PMcardio on PM-ECG-ID, the Kaggle top five).

## 1. Train the full-size models (GPU)

- [ ] `configs/page_gpu.yaml` and `configs/trace_gpu.yaml`: ConvNeXt-S encoders, online
      synthesis at 1536 x 2048, trace crops at 12 px/mm along time. The CPU baseline saw
      about 20,000 crops and 18,000 page windows; the competition models saw millions.
- [ ] More signals: the whole of PTB-XL (minus the PM-ECG-ID patients), CODE-15.
- [ ] Renderer settings matched to real paper: strong 1 mm grids, names printed on the
      trace, calibration pulses, the multi-page formats of PM-ECG-ID.
- [ ] Whole-database PM-ECG-ID and Kaggle runs on CI, as in `ecg-digitalization-classic`
      (a 66-image sample moves by several dB with one page).

## 2. Close the sim-to-real gap

- [ ] Self-training on the Kaggle training images: digitize, keep confident lattices and
      layouts, align the true signal to each metric crop, fine-tune TraceNet on real ink.
- [ ] Same for PageNet with lattice-consistent pseudo-labels for grid intersections.
- [ ] Lead names: a head trained on real name styles, or a small recogniser on the crop at
      the start of each segment (the position is known once rows are found).

## 3. Layout

- [ ] Learn the name-to-baseline association (a vector from each name to the start of its
      segment) instead of the page-level vote.
- [ ] Columns with their own row spacing; pages with other row durations than 10 s.
- [ ] Decode each row once and slice it into segments (the referee already decodes rows;
      today the final pass decodes segments again).
- [ ] Non-sequential layouts: detect from the agreement between a rhythm strip and the
      short segments.

## 4. Calibration

- [ ] Paper speed and gain from the calibration pulse (height in mm = gain, width = 0.2 s)
      and from the printed text.
- [ ] Split gain (limb 10, chest 5 mm/mV).

## 5. Robustness

- [ ] Pages without a 5 mm grid: scale from the calibration pulse or the row length,
      deskew from the traces. `ecgdig` has a grid-free path that can serve as a fallback.
- [ ] Reject option: calibrate the quality indicators (lattice residual, closure
      residual, confidence) into a probability that the output is usable.
- [ ] Pacing spikes: targets that keep 1-2 ms spikes, higher time resolution around them.

## 6. Speed

- [ ] Vectorise the lattice growth (the Python loop dominates on pages with thousands of
      detections).
- [ ] ONNX / TorchScript export and half precision on GPU; batch pages.
- [ ] A lighter full-resolution stage in TraceNet (it is most of the CPU time).

## 7. Science

- [ ] Ablations on real data for every design choice claimed in `ARCHITECTURE.md`
      (hybrid decoder, anisotropic crops, fusion, single resampling).
- [ ] Clinical measurements (intervals, axes, ST level) of digitized versus original
      signals: SNR and PCC do not tell whether a P wave survived.
