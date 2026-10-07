# Project log (manifesto)

A running record of what has been done on this repo, why, and what came out
of it. Newest entries at the bottom of each section. Kaggle notebooks are
private to the `charlesdidier` account.

## Status at a glance

| Item | State |
|---|---|
| Aponeurosis model | Trained (Kaggle run 2): val Dice 0.821 |
| Fascicle model | Baseline (run 2): val Dice 0.318. Reworked pipeline, experiment run 3 in progress |
| Submission | Built by run 2; **not uploaded**. FL/MT are in pixels (`--pixel-to-mm` unknown) |
| Competition metric | Not confirmed; `src/umud/metrics.py` is a placeholder |

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
- 58 of the 309 test images are PNG files named `.tif`.
- Some frames contain green scanner graphics near the bottom (part of the
  image, not the labels).

## Kaggle runs

| # | Notebook | What | Result |
|---|---|---|---|
| 1 | `charlesdidier/umud-unet-smoke` | Original pipeline, 1 epoch each, T4 | Pipeline works end to end on Kaggle (~5 min). Dice: apo 0.17, fasc 0.009 |
| 2 | `charlesdidier/umud-unet-train` | Original pipeline, 30 epochs each, 512x512, ~1 h 50 min | Best val Dice: **apo 0.821** (epoch 25, plateau from ~20), **fasc 0.318** (epoch 30, still creeping up). Outputs: `apo_best.pt`, `fasc_best.pt`, `submission.csv` |
| 3 | `charlesdidier/umud-fasc-exp1` | Reworked fascicle pipeline: 512x768 vs 768x1152 in parallel on 2x T4; baseline (run 2 fasc model) scored on the same val split; submission from the best | In progress |

Training speed in run 2: ~22 img/s (fp32, one T4); validation (forward
only) ~44 img/s, so fp32 compute was the bottleneck, followed by decoding
the LZW TIFFs (~100 ms/image/core).

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
   - New validation metric (`src/umud/evaluation.py`, `scripts/eval_fasc.py`):
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
3. **Mask overlays** (`scripts/visualize_fascicles.py`, `src/umud/viz.py`):
   grids of ground truth (green) and/or predictions (magenta) over images,
   for train/val/test, with per-image segment counts and median angles.

## Decisions

- Select fascicle models by angle error, not Dice: the submission needs
  orientation and extent, and Dice punishes correct detections of
  unlabelled fascicles.
- Keep the 2:3 aspect ratio (512x768 / 768x1152) instead of squashing to a
  square, since most frames are 800x1200.
- Flip TTA only for models trained with flips (the run-2 apo model was not).
- Local CPU work is limited to smoke tests; no local training on random
  weights beyond that (user request).

## Open issues

- `pixel_to_mm` calibration is unknown, so `fl_mm`/`mt_mm` are in pixels.
- The competition's scoring formula is unconfirmed.
- Validation split is random per image; frames from the same subject or
  video may sit on both sides, which would make validation optimistic.
- Only the fascicle angle can be validated locally; fascicle length and
  muscle thickness have no ground truth in the training data.
- Hugging Face token added by the user is not visible in this session (it
  reaches new sessions only); not needed so far.
