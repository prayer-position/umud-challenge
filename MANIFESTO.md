# Project log (manifesto)

A running record of what has been done on this repo, why, and what came out
of it. Newest entries at the bottom of each section. Kaggle notebooks are
private to the `charlesdidier` account.

## Status at a glance

| Item | State |
|---|---|
| Aponeurosis model | Trained (Kaggle run 2): val Dice 0.821 |
| Fascicle model | Run 3, 768x1152: val Dice 0.649 (baseline 0.318); angle error tied with baseline at the label-noise floor |
| Calibration | Per-image px/mm from on-screen rulers, all 309 test images (`src/umud/calibration.py`) |
| Submission | Run 4 `submission.csv` (calibrated, mm, in published ranges) is ready on the notebook's Output tab; **nothing uploaded to the competition yet** |
| Competition metric | Confirmed: mean of MAE/tolerance with tolerances PA 6 deg, FL 12 mm, MT 3 mm |

## Environment and access (2026-10-07)

- Kaggle API works from the cloud session through the proxy (key injected
  server-side). The `kaggle` CLI refuses to run without a local key file, so
  the API is called directly with `curl` (`kernels/push`, `kernels/status`,
  `kernels/output`, `competitions/data/download-all`).
- Competition data downloads fine: 2.75 GB zip, 7,930 files, ~6 GB unpacked.
- Blocked from the session: `huggingface.co` (pretrained weights), and
  `www.kaggleusercontent.com` (Kaggle notebook output files). Outputs must be
  fetched from each notebook's Output tab until that host is allowed.
  Inside Kaggle notebooks both work, so training uses ImageNet weights.
- The session has 4 CPUs and no GPU: local runs are code smoke tests only
  (random encoder weights, a few dozen images); real training runs on Kaggle.

## Data findings

- Two disjoint labelled pools: 1,048 aponeurosis image/mask pairs and 2,761
  fascicle pairs; 309 unlabelled test images.
- Mask polarity differs per pool (apo foreground = 0, fasc foreground = 255);
  `binarize_mask` picks the minority value (from `notebooks/01_eda.ipynb`).
- Mask canvases often differ from the image size (e.g. 800x1200 image with a
  556x996 mask); image and mask are each resized to the training size.
- **Fascicle labels are 1px-wide lines**, ~17 segments per image (5-35),
  covering ~0.24% of pixels; segment lengths are 4-25% of the image diagonal
  (5th-95th percentile); median angle ~13 deg from horizontal.
- **Fascicle labels are incomplete**: annotators marked a subset of visible
  fascicles (e.g. `image_0966`: 9 segments, dozens visible). Pixel metrics
  (Dice) therefore under-rate good models.
- Aponeurosis labels are ~12px-wide bands, 2-4 components per image.
- 58 of the 309 test images are PNGs (`IMG_00252.png`..`IMG_00309.png`; an
  earlier note here wrongly said they were named `.tif`). Submission ids
  keep the file extension, as in `sample_submission.csv`.
- **Training duplicates**: 75% of validation images have a near-identical
  training image (median thumbnail correlation 0.998), and the organizers
  confirm duplicates in the training set (competition discussion 740356).
  The random split is therefore leaky. Duplicate frames' labels differ by
  ~1.06 deg in median fascicle angle, i.e. ~1 deg is annotation noise.
- A constant fascicle-angle guess scores 3.37 deg on validation, so the
  models (~0.74 deg) do learn orientation.
- Some frames contain green scanner graphics near the bottom (part of the
  image, not the labels).

## Kaggle runs

| # | Notebook | What | Result |
|---|---|---|---|
| 1 | `charlesdidier/umud-unet-smoke` | Original pipeline, 1 epoch each, T4 | Pipeline works end to end on Kaggle (~5 min). Dice: apo 0.17, fasc 0.009 |
| 2 | `charlesdidier/umud-unet-train` | Original pipeline, 30 epochs each, 512x512, ~1 h 50 min | Best val Dice: **apo 0.821** (epoch 25, plateau from ~20), **fasc 0.318** (epoch 30, still creeping up). Outputs: `apo_best.pt`, `fasc_best.pt`, `submission.csv` |
| 3 | `charlesdidier/umud-fasc-exp1` | Reworked fascicle pipeline (`kaggle/runs/fasc_experiments.sh`): 512x768 vs 768x1152 in parallel on 2x T4; baseline scored on the same val split; submission + overlays from the best. ~2 h 50 min | Val angle MAE / p90 / Dice (flip TTA): **768x1152 0.73 / 1.48 deg / 0.649** (early-stopped at 37, best epoch 27), 512x768 0.75 / 1.50 / 0.634 (stopped at 39, best 29), baseline 0.74 / 1.65 / 0.318. Dice doubled; angle error is tied at the ~1 deg label-noise floor of a leaky split, so it cannot rank them. Submission still uncalibrated (pixels) |
| 5 | `charlesdidier/umud-grouped-v1` | `kaggle/runs/grouped_v1.sh`: both models retrained on the grouped split in parallel: apo 512x768 (GPU 0, selected by thickness error), fasc 768x1152 (GPU 1, selected by angle error); scored with flip TTA; run-2 apo scored on the same split for reference; calibrated submission from the two new models; apo and fasc overlays | Running |
| 4 | `charlesdidier/umud-predict-v1` | `kaggle/runs/predict.sh`: no training; run-2 apo + run-3 768x1152 fasc, per-image calibration, measurements in mm, clipped to published ranges. ~5 min | All 309 calibrated. Raw medians per layout: PA 10.5-19.4 deg, FL 66-129 mm, MT 18-28 mm; only 2% of Lumify FL fell outside the published ranges (clipped). Outputs: `submission.csv`, `diagnostics.csv`, `calibration.csv`, `viz/fasc_test.png` |

Training speed in run 2: ~22 img/s (fp32, one T4); validation (forward
only) ~44 img/s, so fp32 compute was the bottleneck, followed by decoding
the LZW TIFFs (~100 ms/image/core). In run 3, two trainings shared the 4
CPUs: 123 s/epoch at 512x768 and 270 s/epoch at 768x1152 (CPU-bound
augmentation), so parallel runs are slower than estimated.

## Competition facts (from the Kaggle pages, 2026-10-07)

- Score = mean over PA, FL, MT of MAE / tolerance, tolerances PA 6 deg,
  FL 12 mm, MT 3 mm (official `paulritsche/umud-score` notebook); MedAE and
  RMSE only break ties. Lower is better.
- Test ranges: PA 5-45 deg, FL 30-200 mm, MT 10-50 mm.
- Test labels: two raters, 3 fascicles / 3 PAs / 3 MTs per image, averaged;
  FL extrapolated linearly between aponeuroses when the fascicle leaves the
  frame; MT is the perpendicular distance at three locations across the width.
- Test devices: Siemens Acuson Juniper, Telemed ArtUS EXT-1H, Philips Lumify;
  test subjects are not in the training set; some test images are 5-frame
  video runs. Deadline 2026-11-14; 5 submissions per day.
- Prize eligibility needs an open-source (OSI licence), FAIR, reproducible
  repository.

## Calibration (pixels -> mm)

Every test layout draws a ruler; `src/umud/calibration.py` reads it per image
(`scripts/calibrate_images.py` prints the table):

| Layout | Images | px/mm | Source |
|---|---|---|---|
| Telemed 1200x800 screens | 90 | x 13.4, y 14.8 (stretched) | left depth ruler + bottom lateral ruler, 10 mm ticks |
| Telemed 1088x644 screens | 50 | 12.6 (square) | same rulers, unscaled |
| Telemed crops 853x1069 / 513x465 | 12 / 8 | 16.7 / 7.8 | bottom lateral ruler, 10 mm ticks (assumed) |
| Siemens Juniper 1200x800 | 91 | 8.7-16.0 by depth (3-7 cm) | right depth ruler; 2 mm ticks at 3 cm, 5 mm from 3.5 cm, resolved via the 12L3 probe's ~57 mm width |
| Philips Lumify 1200x800 PNG | 58 | 12.0-20.1 by depth (3-5 cm) | left depth ruler, 5 mm ticks |

All values fall on the scanners' discrete depth settings. Assumptions: square
pixels for Siemens, Lumify and crops (no lateral ruler), and 10 mm per tick on
crop rulers.

## Code changes

1. **Kaggle kernel builder** (`kaggle/build_kernel.py`, `kaggle/kernel_template.py`):
   embeds the committed repo in a private Kaggle script kernel with GPU and
   competition data attached. Later reworked to run a script from
   `kaggle/runs/` and attach previous notebooks' outputs (`--kernel-source`).
2. **Fascicle pipeline rework** (commit `7af6bcf`):
   - `resize_mask(line_px=...)`: line-preserving downsampling (area resize,
     any coverage = foreground) plus dilation to ~3-5px, replacing
     nearest-neighbour resizing that broke 1px lines into dots.
   - Training: mixed precision, cosine LR with warmup, early stopping,
     horizontal flips, in-memory cache of decoded images, per-run
     checkpoint names (`--run-name`) and `*_history.json` metric logs.
   - New validation metric (`src/umud/evaluation.py`, `scripts/eval_fasc.py`, later `scripts/eval_model.py`):
     per image, |median predicted - median ground-truth fascicle angle|,
     measured on the native label canvas so it is comparable across
     resolutions; misses count as 45 deg. Fascicle checkpoints are selected
     by this (`select_metric: angle_mae`) instead of Dice.
   - Geometry (`src/umud/geometry.py`): every fascicle segment >= 3% of the
     diagonal is fitted; each is extended to the superficial and deep
     aponeuroses for fascicle length, its angle is measured against the deep
     aponeurosis, and medians are reported. Previously only the largest
     blob's visible extent was used.
   - Inference (`src/umud/inference.py`): shared by predict/eval/viz;
     training-identical resizing, threshold after upsampling, flip TTA for
     models trained with flips; NaNs in the submission filled with medians.
   - `load_image` detects PNG content regardless of extension.
   - Configs `configs/fasc_seg_512x768.yaml`, `configs/fasc_seg_768x1152.yaml`.
3. **Run scripts** (`kaggle/runs/*.sh`): each Kaggle notebook runs one;
   `full.sh` reproduces run 2, `fasc_experiments.sh` is run 3. Before the
   push, every script path was smoke-tested locally (inference only) and the
   run script's control flow was dry-run with stubbed commands, which caught
   a bug that would have crashed the best-model selection.
4. **Mask overlays** (`scripts/visualize_fascicles.py`, later `scripts/visualize_masks.py`, `src/umud/viz.py`):
   grids of ground truth (green) and/or predictions (magenta) over images,
   for train/val/test, with per-image segment counts and median angles.
5. **Calibration and mm measurements**: `src/umud/calibration.py`,
   `scripts/calibrate_images.py`; geometry converts pixel coordinates to mm
   with separate x/y scales before fitting lines; muscle thickness is now the
   mean perpendicular distance at 25/50/75% of the shared aponeurosis width;
   `predict.py` calibrates each image, writes a diagnostics CSV and clips to
   the published test ranges; `kaggle/runs/predict.sh` runs prediction only.
6. **Grouped validation split** (`src/umud/data/splits.py`, config
   `split: grouped`): clusters near-duplicate frames (thumbnail correlation
   > 0.95, transitive) and moves whole clusters to validation until it holds
   `val_frac` of the images, skipping clusters larger than half that target
   (one fascicle cluster has 657 images). Result: fascicle 2346/415 and apo
   890/158 train/val (same sizes as the random split) with no validation
   image having a > 0.95 twin in training (random split: 75% fascicle, 22%
   apo). Clusters are cached in `outputs/cache/`. The default stays `random`
   so earlier results remain comparable.
7. **Aponeurosis metric and shared tooling**: `ApoThicknessEvaluator`
   (`src/umud/evaluation.py`) scores muscle thickness from predicted vs
   ground-truth masks as a relative error on the native label canvas (no
   calibration needed) plus the deep-aponeurosis angle error;
   `select_metric: mt_rel_err` picks apo checkpoints by it. `eval_fasc.py`
   became `scripts/eval_model.py --pool apo|fasc` and
   `visualize_fascicles.py` became `scripts/visualize_masks.py --pool
   apo|fasc` (apo captions show thickness and deep angle). Configs
   `configs/apo_seg_512x768.yaml` and `configs/fasc_seg_768x1152_grouped.yaml`.

## Decisions

- Select fascicle models by angle error, not Dice: the submission needs
  orientation and extent, and Dice punishes correct detections of
  unlabelled fascicles.
- Keep the 2:3 aspect ratio (512x768 / 768x1152) instead of squashing to a
  square, since most frames are 800x1200.
- Flip TTA only for models trained with flips (the run-2 apo model was not).
- Local CPU work is limited to smoke tests; no local training on random
  weights beyond that (user request).
- Measure in physical mm with per-axis scales (the stretched Telemed
  screens would otherwise skew angles by ~1 deg and lengths by up to 10%).
- Clip predictions to the published test ranges: it can only reduce error
  for truths inside those ranges.

## Open issues

- Calibration assumptions above are unverified against labels; the first
  leaderboard submission will show whether FL/MT are in the right range.
- Earlier validation numbers (runs 2-3) come from the leaky random split;
  new experiments should use `split: grouped`.
- The apo model (run 2) was trained with the original nearest-neighbour
  mask resizing and selected by Dice; MT has the tightest tolerance (3 mm),
  so it is the next model to revisit.
- Only the fascicle angle can be validated locally; fascicle length and
  muscle thickness have no ground truth in the training data.
- Hugging Face token added by the user is not visible in this session (it
  reaches new sessions only); not needed so far.
