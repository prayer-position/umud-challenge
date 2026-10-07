import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from umud import utils
from umud.data import schema
from umud.data.dataset import load_image
from umud.geometry import measure_fascicle, measure_thickness
from umud.inference import load_model, predict_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run both segmenters + geometry over the test set.")
    parser.add_argument("--apo-checkpoint", default=None)
    parser.add_argument("--fasc-checkpoint", default=None)
    parser.add_argument("--apo-config", default="configs/apo_seg.yaml")
    parser.add_argument("--fasc-config", default="configs/fasc_seg.yaml")
    parser.add_argument(
        "--pixel-to-mm",
        type=float,
        default=1.0,
        help="Calibration factor; unconfirmed for this dataset, defaults to pixel units.",
    )
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

    rows = []
    for path in test_files:
        image = load_image(path)

        apo_mask = predict_mask(apo_model, image, apo_config, device, tta=apo_tta)
        fasc_mask = predict_mask(fasc_model, image, fasc_config, device, tta=fasc_tta)

        mt_mm = measure_thickness(apo_mask, pixel_to_mm=args.pixel_to_mm)
        fl_mm, pa_deg = measure_fascicle(fasc_mask, apo_mask=apo_mask, pixel_to_mm=args.pixel_to_mm)

        rows.append({"image_id": path.name, "pa_deg": pa_deg, "fl_mm": fl_mm, "mt_mm": mt_mm})

    submission = pd.DataFrame(rows, columns=schema.SUBMISSION_COLUMNS)
    # Images where a structure wasn't found get the column median rather
    # than NaN, which the submission format doesn't allow.
    for column in schema.SUBMISSION_COLUMNS[1:]:
        n_missing = int(submission[column].isna().sum())
        if n_missing:
            print(f"{column}: filling {n_missing} missing value(s) with the median")
            submission[column] = submission[column].fillna(submission[column].median())

    sample = pd.read_csv(
        schema.SAMPLE_SUBMISSION_PATH, sep=schema.SUBMISSION_SEP, encoding=schema.SUBMISSION_ENCODING
    )
    assert list(submission.columns) == list(sample.columns), "Submission columns don't match sample_submission.csv"
    assert len(submission) == len(test_files), "Row count doesn't match number of test images"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = schema.PROJECT_ROOT / "submissions" / f"submission_{timestamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, sep=schema.SUBMISSION_SEP, index=False, encoding=schema.SUBMISSION_ENCODING)
    print(f"Wrote {out_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
