# State of the art in ECG image digitization (November 2026)

What the strongest public systems do, what each leaves on the table, and what `ecgdl`
takes from them. Numbers are the ones their authors report; the benchmarks differ, so they
are not directly comparable with each other.

## 1. Kaggle "PhysioNet - Digitization of ECG Images" (2025-2026)

1,424 teams, hidden test set, one page format (3x4 plus a lead II rhythm strip) captured
nine ways (render, scans, phone and screen photos, stained and crumpled paper). Score:
per record, signal and noise power summed over the 12 leads after a time alignment of up
to 0.2 s and offset removal, in dB.

| Entry | Private score | Geometry | Signal |
|---|---|---|---|
| [2nd place](https://www.kaggle.com/takashisomeya/writeups) (weights reused in [this notebook](https://github.com/Yashjamine1207/physionet-ecg-image-digitization)) | 23.27 dB | public two-stage rectifier by hengck23: 9 reference points, then the printed grid, onto a fixed 1700 x 2200 frame | six EfficientNet U-Nets (whole page + lead windows), sparse masks of at most 2 px per column, flip averaging |
| [3rd place notebook](https://www.kaggle.com/code/hirotetsu/physionet-submission-3rd-place) | 22.76 dB | | |
| [4th place](https://github.com/uchiyama33/physionet_4th_place) | | grid intersections + layout (multi-task), per-pixel grid line index, map to a normalised frame | regression with an SNR loss; ConvNeXt V2 pretrained with FCMAE; PTB-XL and CODE-15 as extra data |
| [5th place](https://github.com/dangnh0611/kaggle_ecg_digitization) | 22.63 dB | rotation classifier; 2,422 keypoints (57 landmarks + a 43 x 55 grid) as 58 heatmap channels at 2048 px, pseudo-labelled by dense matching against the clean render; Hungarian registration; local piecewise homographies | column-wise 1D Gaussian heatmaps at 1024 x 1024, dual encoder (CoaT or ConvNeXt + early VGG19 blocks), a separate "tall" model for large amplitudes, lead fusion |
| [7th place](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/7th-place-solution) | | rotation, detection, segmentation | DINOv3 ViT-S encoder; SNR losses that first learn, then discount, the vertical shift |
| [11th place](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/11th-place-solution) | | hengck23's rectifier + iterative equal-spacing refinement of grid points | four long lead crops, SNR-aware loss |
| [12th place](https://www.kaggle.com/competitions/physionet-ecg-image-digitization/writeups/12th-place-solution) | | OCR for orientation, 17 keypoints, Delaunay piecewise warp to the template | |

An independent [re-implementation study](https://github.com/Anisimov-AA/cv-ecg-image-to-signal-digitization)
measured the ingredients: a column distribution decoded by soft-argmax with a
Jensen-Shannon + SNR loss was worth about 7 dB over a plain mask U-Net, and data
diversity was the largest single factor.

**What wins:** (1) rectify first, with learned keypoints on the grid; (2) predict a
distribution over rows per column, not a mask; (3) train on a loss tied to SNR; (4) give
the network one lead at a time; (5) ensembles and large backbones for the last dB.

**What is missing:** every pipeline registers the page to *the* competition template
(fixed lead positions, fixed frame). None reads the layout, so none works on a 6x2 or
12x1 page, a different vendor, or Cabrera order. Time resolution is about 200 px/s, which
caps the reachable SNR (see `ARCHITECTURE.md`). The top entries are ensembles of up to
six large models trained for 100+ GPU hours.

## 2. PMcardio (Powerful Medical)

[Demolder et al., *J Electrocardiol* 2025](https://www.sciencedirect.com/science/article/abs/pii/S0022073625000287)
([preprint](https://www.medrxiv.org/content/10.1101/2024.08.31.24312876v1.full)), the
method behind the PMcardio product:

* **Normalization.** A residual U-Net ("Dotter") marks the major grid intersections
  (about 1,900 per page on their annotated images); a "Gridder" orders them into a matrix
  and interpolates the gaps; each grid square is undistorted with its own 4-point
  transform, which handles crumpled paper.
* **Reconstruction.** A second U-Net segments the traces; post-processing isolates lead
  columns, estimates baselines, bridges gaps and resolves overlaps; a format detector
  reads rotation, layout and standard-vs-Cabrera order.
* Trained on a few hundred annotated photos cut into 256 px patches. Paper speed and gain
  are supplied by the user.
* **PM-ECG-ID** benchmark (6,000 images, 100 PTB-XL ECGs, 60 scenarios): PCC 0.977 /
  RMSE 0.041 mV / SNR 16.7 dB on the easy scenarios, 0.970 / 0.047 / 15.8 on the medium
  ones, 0.817 / 0.098 / 11.1 on the hard ones; 6.62% of leads fail overall; under 7 s per image.
* Stated limits: overlapping precordial leads (V3-V5), pacing spikes, extreme blur,
  scaling and darkness.

**What is right:** the grid is the ruler, and a *local* undistortion per cell copes with
non-flat paper. Any layout is supported. **What is missing:** a mask-then-trace design
(overlaps are resolved by heuristics after segmentation), no SNR-driven training, and no
use of the redundancy between leads.

## 3. Open ECG Digitizer (*npj Digital Medicine*, 2026)

[Stenhede, Bjørnstad and Ranjbar](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12891466/)
([code](https://github.com/Ahus-AIM/Open-ECG-Digitizer),
[preprint](https://arxiv.org/pdf/2510.19590)): a modular, fully open pipeline validated on
37,191 images.

* A residual U-Net separates signal, grid lines, text and background (trained on 8,658
  synthetic high-resolution pages, 1024 px crops).
* Perspective is found from the *grid-line* probability map with a Hough transform: the
  two dominant maxima give the horizontal and vertical grid directions.
* Grid size gives mm/pixel; lead names are located and matched against user-defined
  layout templates; traces are extracted from the dewarped signal map by connected
  components.
* Handles any subset of leads and up to about 45 degrees of rotation. Full dewarping of
  folded paper is experimental.

**What is right:** layout from lead names plus templates, any format, open data and
code. **What is missing:** a global projective model cannot follow curled or creased
paper, and trace extraction from a mask has no notion of which stroke belongs to which
lead when they cross.

## 4. Other reference points

* **PhysioNet Challenge 2024** (same task, earlier data): the winner,
  [ECG-Digitiser](https://github.com/felixkrones/ECG-Digitiser)
  ([paper](https://arxiv.org/html/2410.14185v1)), rotates with a Hough transform and
  segments each lead with nnU-Net; it scored
  [12.15 dB on the hidden set](https://www.cinc.org/archives/2024/pdf/CinC2024-011.pdf).
  Its released weights assume the 3x4 + rhythm layout.
* **Overlapping leads:** [Karbasi et al., 2025](https://arxiv.org/html/2506.10617v1) train
  a U-Net to isolate the *primary* trace of a crop with an "overlay another signal"
  augmentation, the same target-centred idea used here.
* **Data:** [ECG-Image-Kit](https://pubmed.ncbi.nlm.nih.gov/39150768/) (synthetic pages
  with artefacts), [ECG-Image-Database](https://www.kaggle.com/datasets/physionet/ecg-image-database)
  (1,977 records printed, scanned and photographed),
  [PTB-XL-Image-17K](https://arxiv.org/html/2602.07446v1).
* **Our own classical digitizer**, [`ecgdig`](https://github.com/fmenegui/ecg-digitalization-classic):
  7.91 dB / PCC 0.656 on the whole PM-ECG-ID and 14.68 dB on 135 Kaggle training images,
  with no trained model. Its error analysis says where learning should help most: pages
  whose format is misread (crossing traces, unreadable names) and heavy blur, low
  resolution, crumpled paper and faint ink.

## 5. Design consequences for `ecgdl`

| Lesson | Source | In `ecgdl` |
|---|---|---|
| Learned grid intersections are the most reliable geometric cue | PMcardio, Kaggle 4th/5th | PageNet heatmap + sub-pixel offsets |
| Order the points into a lattice, undistort locally | PMcardio Gridder, Kaggle piecewise warps | best-first lattice growth, homography + robust smooth residual field (handles curl, fills gaps, extrapolates) |
| Do not depend on one template | npj, PMcardio | layout read from lead names in paper millimetres; templates only repair missed names |
| One lead per crop, distribution per column, SNR loss | Kaggle 2nd/5th/7th, Anisimov | TraceNet on metric, target-centred crops; cross-entropy + L1 + SNR on the soft-argmax |
| Time resolution limits SNR | our codec study | anisotropic crops: more pixels per mm along time than along voltage |
| Large amplitudes need a taller view | Kaggle 5th ("tall" model) | one network trained over a range of vertical scales; automatic taller retry |
| Leads are redundant | Kaggle lead fusion, ecgdig | confidence-weighted least squares on the Einthoven/Goldberger subspace with free offsets |
| Real photos are scarce, physics is cheap | ECG-Image-Kit, Kaggle 4th | online synthesis with ecg-torch-renderer from real PTB-XL signals, exact targets |
| Resample once | (ours) | every crop is sampled from the original photo through the lattice, never from an intermediate rectified image |
