"""Rebuild predicted masks and the diagnostics table from a notebook log
written by predict.py --dump-log, and render overlays on the local test
images.

    python scripts/decode_log_dump.py --log log.txt --out-dir outputs/dump --overlays IMG_00001.tif IMG_00002.tif
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from umud.data import schema
from umud.data.dataset import load_image
from umud.maskdump import TAG, decode_line
from umud.viz import grid, overlay, save_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overlays", nargs="*", default=[], help="Test image ids to render (apo green, fasc magenta).")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masks: dict[str, dict[str, np.ndarray]] = {}
    diag_lines = []
    for line in Path(args.log).read_text().splitlines():
        line = line.strip()
        if line.startswith(TAG):
            image_id, kind, mask = decode_line(line)
            masks.setdefault(image_id, {})[kind] = mask
        elif line.startswith("DIAG "):
            diag_lines.append(line[5:])
    np.savez_compressed(out / "masks.npz", **{f"{i}|{k}": m for i, d in masks.items() for k, m in d.items()})
    if diag_lines:
        (out / "diagnostics.csv").write_text("\n".join(diag_lines) + "\n")
        print(pd.read_csv(out / "diagnostics.csv").shape, "diagnostics rows")
    print(f"{len(masks)} images with masks -> {out / 'masks.npz'}")

    tiles = [
        overlay(load_image(schema.TEST_IMAGE_DIR / i), masks[i].get("apo"), masks[i].get("fasc"), title=i, width=700)
        for i in args.overlays
        if i in masks
    ]
    if tiles:
        save_rgb(out / "overlays.png", grid(tiles, cols=2))
        print(f"wrote {out / 'overlays.png'}")


if __name__ == "__main__":
    main()
