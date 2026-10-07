"""Detect pixel->mm calibration for every image in a folder and summarize it
per layout (see umud/calibration.py).

    python scripts/calibrate_images.py                     # test images
    python scripts/calibrate_images.py --out outputs/calibration.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from umud.calibration import calibrate
from umud.data import schema
from umud.data.dataset import is_png, load_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default=str(schema.TEST_IMAGE_DIR))
    parser.add_argument("--out", default=str(schema.PROJECT_ROOT / "outputs" / "calibration.csv"))
    args = parser.parse_args()

    rows = []
    for path in schema.list_image_files(Path(args.dir)):
        image = load_image(path)
        cal = calibrate(image, is_png=is_png(path))
        rows.append(
            {
                "image_id": path.name,
                "height": image.shape[0],
                "width": image.shape[1],
                "layout": cal.layout if cal else None,
                "px_per_mm_x": cal.px_per_mm_x if cal else np.nan,
                "px_per_mm_y": cal.px_per_mm_y if cal else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    missing = df[df.layout.isna()]
    print(f"{len(df)} images, {len(missing)} without calibration {missing.image_id.tolist()[:10]}")
    summary = df.groupby(["layout", "height", "width"]).agg(
        n=("image_id", "size"),
        x_min=("px_per_mm_x", "min"),
        x_median=("px_per_mm_x", "median"),
        x_max=("px_per_mm_x", "max"),
        y_min=("px_per_mm_y", "min"),
        y_median=("px_per_mm_y", "median"),
        y_max=("px_per_mm_y", "max"),
    )
    print(summary.round(2).to_string())
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
