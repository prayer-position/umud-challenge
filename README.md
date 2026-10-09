# UMUD Challenge — Muscle Architecture in Ultrasound Data

Kaggle: https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data

Predict, per test ultrasound image: Pennation Angle (`pa_deg`), Fascicle Length (`fl_mm`), Muscle Thickness (`mt_mm`).

## Data reality check

There is **no scalar-label CSV**. Ground truth is two disjoint sets of raster masks:

- `apo_imgs_v1/apo_images_new_model_v1/` + `apo_masks_v1/apo_masks_new_model_v1/` (1048 pairs) — aponeurosis masks.
- `fasc_imgs_v1/fasc_images_new_model_v1/` + `fasc_masks_v1/fasc_masks_new_model_v1/` (2761 pairs) — fascicle masks.
- `test_images_v2/test_set_v2/` (309 images, no masks) — full frames where both structures are presumably visible.

No single training image has both mask types. The approach here is therefore: train two segmentation models (aponeuroses, fascicles), run both on each test image, and derive PA/FL/MT geometrically from the two predicted masks (see `src/umud/geometry.py`). See `.claude` plan history / `notebooks/01_eda.ipynb` for how this was determined.

`sample_submission.csv` is **semicolon-delimited** with a UTF-8 BOM: `image_id;pa_deg;fl_mm;mt_mm`. Kaggle nevertheless parses submissions as plain comma-separated CSV and rejects the sample's format ("ID column image_id not found in submission"), so `scripts/predict.py` writes comma-separated UTF-8 without a BOM.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install torch separately (CPU build on a machine with no NVIDIA GPU):
# see https://pytorch.org/get-started/locally/ for the exact command for your machine
pip install -r requirements.txt
pip install -e .
```

Kaggle API (only needed if re-downloading data): place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json`, accept the competition rules on the Kaggle website, then `kaggle competitions download -c umud-challenge-muscle-architecture-in-ultrasound-data`.

## Workflow

1. `notebooks/01_eda.ipynb` — inspect real mask semantics (class values), image sizes/modes, visualize overlays.
2. `notebooks/02_geometry_sanity_check.ipynb` — validate `geometry.py` against real masks before training anything.
3. `python scripts/train_apo.py --config configs/apo_seg.yaml` and `python scripts/train_fasc.py --config configs/fasc_seg.yaml`.
4. `python scripts/predict.py` — runs both models + geometry over `test_images_v2/`, writes `submissions/submission_<timestamp>.csv`.
