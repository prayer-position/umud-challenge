import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from umud import utils
from umud.data import schema
from umud.data.dataset import load_image
from umud.data.transforms import build_val_transform
from umud.geometry import measure_fascicle, measure_thickness
from umud.models.unet import build_unet


def load_model(checkpoint_path: Path, encoder: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_unet(encoder_name=encoder, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_mask(model: torch.nn.Module, image: np.ndarray, transform, device: torch.device) -> np.ndarray:
    dummy_mask = np.zeros(image.shape[:2], dtype=np.int64)
    augmented = transform(image=image, mask=dummy_mask)
    tensor = augmented["image"].unsqueeze(0).to(device)

    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = (prob > 0.5).astype(np.uint8)

    # Predictions come back at the model's input resolution; resize to the
    # original image size before running geometry so pixel measurements are
    # comparable across images of different native resolutions.
    return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


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
    args = parser.parse_args()

    device = utils.get_device()
    apo_config = utils.load_config(args.apo_config)
    fasc_config = utils.load_config(args.fasc_config)

    apo_checkpoint = Path(args.apo_checkpoint or schema.PROJECT_ROOT / "outputs" / "checkpoints" / "apo_best.pt")
    fasc_checkpoint = Path(args.fasc_checkpoint or schema.PROJECT_ROOT / "outputs" / "checkpoints" / "fasc_best.pt")

    apo_model = load_model(apo_checkpoint, apo_config["encoder"], apo_config["num_classes"], device)
    fasc_model = load_model(fasc_checkpoint, fasc_config["encoder"], fasc_config["num_classes"], device)

    apo_transform = build_val_transform(tuple(apo_config["image_size"]))
    fasc_transform = build_val_transform(tuple(fasc_config["image_size"]))

    test_files = schema.list_image_files(schema.TEST_IMAGE_DIR)
    if args.limit is not None:
        test_files = test_files[: args.limit]

    rows = []
    for path in test_files:
        image = load_image(path)

        apo_mask = predict_mask(apo_model, image, apo_transform, device)
        fasc_mask = predict_mask(fasc_model, image, fasc_transform, device)

        mt_mm = measure_thickness(apo_mask, pixel_to_mm=args.pixel_to_mm)
        fl_mm, pa_deg = measure_fascicle(fasc_mask, apo_mask=apo_mask, pixel_to_mm=args.pixel_to_mm)

        rows.append({"image_id": path.name, "pa_deg": pa_deg, "fl_mm": fl_mm, "mt_mm": mt_mm})

    submission = pd.DataFrame(rows, columns=schema.SUBMISSION_COLUMNS)

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
