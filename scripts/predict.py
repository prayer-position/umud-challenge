import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from umud import utils
from umud.calibration import calibrate
from umud.data import schema
from umud.data.dataset import is_png, load_image
from umud.geometry import fascicle_lines, measure_fascicle, measure_thickness
from umud.inference import load_model, predict_mask

# Ranges the organizers give for the test set (data description): clipping to
# them can only move a prediction closer to a truth that lies inside.
TEST_RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both segmenters + geometry over the test set.")
    parser.add_argument("--apo-checkpoint", default=None)
    parser.add_argument("--fasc-checkpoint", default=None)
    parser.add_argument("--apo-config", default="configs/apo_seg.yaml")
    parser.add_argument("--fasc-config", default="configs/fasc_seg.yaml")
    parser.add_argument("--no-clip", action="store_true", help="Don't clip predictions to the published test ranges.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N test images (smoke test).")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip test-time augmentation (used for models trained with flip: true).")
    args = parser.parse_args()

    device = utils.get_device()
    apo_config = utils.load_config(args.apo_config)
    fasc_config = utils.load_config(args.fasc_config)

    apo_checkpoint = Path(args.apo_checkpoint or schema.PROJECT_ROOT / "outputs" / "checkpoints" / "apo_best.pt")
    fasc_checkpoint = Path(args.fasc_checkpoint or schema.PROJECT_ROOT / "outputs" / "checkpoints" / "fasc_best.pt")

    apo_model = load_model(apo_checkpoint, apo_config, device)
    fasc_model = load_model(fasc_checkpoint, fasc_config, device)
    # Flip TTA only for models that saw flipped images during training.
    apo_tta = not args.no_tta and apo_config.get("flip", False)
    fasc_tta = not args.no_tta and fasc_config.get("flip", False)

    test_files = schema.list_image_files(schema.TEST_IMAGE_DIR)
    if args.limit is not None:
        test_files = test_files[: args.limit]

    rows, diagnostics = [], []
    for path in test_files:
        image = load_image(path)
        cal = calibrate(image, is_png=is_png(path))

        apo_mask = predict_mask(apo_model, image, apo_config, device, tta=apo_tta)
        fasc_mask = predict_mask(fasc_model, image, fasc_config, device, tta=fasc_tta)

        rows.append({"image_id": path.name, "cal": cal, "apo_mask": apo_mask, "fasc_mask": fasc_mask})

    # Images whose ruler wasn't recognised get the median calibration of the
    # images with the same size, else of all images.
    found = [r for r in rows if r["cal"] is not None]
    for row in rows:
        if row["cal"] is None:
            same = [r["cal"] for r in found if r["apo_mask"].shape == row["apo_mask"].shape] or [r["cal"] for r in found]
            row["scale"] = tuple(np.median([c.scale for c in same], axis=0)) if same else (1.0, 1.0)
            row["layout"] = "fallback"
            print(f"{row['image_id']}: no ruler found, using {row['scale']} px/mm")
        else:
            row["scale"], row["layout"] = row["cal"].scale, row["cal"].layout

    results = []
    for row in rows:
        mt_mm = measure_thickness(row["apo_mask"], px_per_mm=row["scale"])
        fl_mm, pa_deg = measure_fascicle(row["fasc_mask"], apo_mask=row["apo_mask"], px_per_mm=row["scale"])
        results.append({"image_id": row["image_id"], "pa_deg": pa_deg, "fl_mm": fl_mm, "mt_mm": mt_mm})
        diagnostics.append(
            {
                "image_id": row["image_id"],
                "layout": row["layout"],
                "px_per_mm_y": row["scale"][0],
                "px_per_mm_x": row["scale"][1],
                "n_fascicles": len(fascicle_lines(row["fasc_mask"], px_per_mm=row["scale"])),
                "pa_deg": pa_deg,
                "fl_mm": fl_mm,
                "mt_mm": mt_mm,
            }
        )

    submission = pd.DataFrame(results, columns=schema.SUBMISSION_COLUMNS)
    diag = pd.DataFrame(diagnostics)
    print("Raw predictions per layout (median, and share outside the published test ranges):")
    for column, (lo, hi) in TEST_RANGES.items():
        diag[f"{column}_out_of_range"] = ~diag[column].between(lo, hi)
    print(
        diag.groupby("layout")
        .agg(
            n=("image_id", "size"),
            fasc=("n_fascicles", "median"),
            pa=("pa_deg", "median"),
            fl=("fl_mm", "median"),
            mt=("mt_mm", "median"),
            pa_out=("pa_deg_out_of_range", "mean"),
            fl_out=("fl_mm_out_of_range", "mean"),
            mt_out=("mt_mm_out_of_range", "mean"),
        )
        .round(2)
        .to_string()
    )
    # Images where a structure wasn't found get the column median rather
    # than NaN, which the submission format doesn't allow.
    for column in schema.SUBMISSION_COLUMNS[1:]:
        n_missing = int(submission[column].isna().sum())
        if n_missing:
            print(f"{column}: filling {n_missing} missing value(s) with the median")
            submission[column] = submission[column].fillna(submission[column].median())
        if not args.no_clip:
            submission[column] = submission[column].clip(*TEST_RANGES[column])

    sample = pd.read_csv(
        schema.SAMPLE_SUBMISSION_PATH, sep=schema.SUBMISSION_SEP, encoding=schema.SUBMISSION_ENCODING
    )
    assert list(submission.columns) == list(sample.columns), "Submission columns don't match sample_submission.csv"
    assert len(submission) == len(test_files), "Row count doesn't match number of test images"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = schema.PROJECT_ROOT / "submissions" / f"submission_{timestamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, sep=schema.SUBMISSION_SEP, index=False, encoding=schema.SUBMISSION_ENCODING)
    diag.to_csv(out_path.with_name(f"diagnostics_{timestamp}.csv"), index=False)
    print(f"Wrote {out_path} ({len(submission)} rows) and its diagnostics CSV")


if __name__ == "__main__":
    main()
