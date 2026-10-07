"""Render fascicle or aponeurosis masks over their images as a PNG grid.

    # ground-truth labels only
    python scripts/visualize_masks.py --pool fasc --source train --out outputs/viz/fasc_gt.png
    # held-out validation images: ground truth (green) vs prediction (magenta)
    python scripts/visualize_masks.py --pool fasc --source val --checkpoint outputs/checkpoints/fasc_best.pt
    # test images: prediction only
    python scripts/visualize_masks.py --pool apo --source test --checkpoint outputs/checkpoints/apo_best.pt \
        --config configs/apo_seg.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np

from umud import utils
from umud.data import schema
from umud.data.dataset import binarize_mask, load_image, load_mask
from umud.geometry import deep_aponeurosis_angle, fascicle_angles, measure_thickness
from umud.inference import load_model, predict_mask
from umud.data.splits import split_indices
from umud.train_pipeline import POOL_DIRS
from umud.viz import grid, overlay, save_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=["apo", "fasc"], default="fasc")
    parser.add_argument("--source", choices=["train", "val", "test"], default="val")
    parser.add_argument("--checkpoint", default=None, help="Model for --pool; omit to draw ground truth only.")
    parser.add_argument("--config", default="configs/fasc_seg.yaml")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = utils.load_config(args.config)
    image_dir, mask_dir = POOL_DIRS[args.pool]
    if args.source == "test":
        paths = schema.list_image_files(schema.TEST_IMAGE_DIR)
    else:
        paths = schema.list_image_files(image_dir)
        train_idx, val_idx = split_indices(image_dir, config)
        paths = [paths[i] for i in (val_idx if args.source == "val" else train_idx)]
    paths = random.Random(args.seed).sample(paths, min(args.n, len(paths)))

    model = None
    if args.checkpoint:
        device = utils.get_device()
        model = load_model(Path(args.checkpoint), config, device)

    tiles = []
    for path in paths:
        image = load_image(path)
        gt = None
        if args.source != "test":
            gt = binarize_mask(load_mask(mask_dir / path.name))
        pred = None
        if model is not None:
            # Predict on the ground-truth canvas when there is one, so both
            # masks live in the same coordinate frame.
            out_shape = gt.shape if gt is not None else image.shape[:2]
            pred = predict_mask(model, image, config, device, out_shape=out_shape, tta=config.get("flip", False))

        parts = [path.name]
        for name, mask in (("gt", gt), ("pred", pred)):
            if mask is None:
                continue
            if args.pool == "fasc":
                angles = fascicle_angles(mask)
                median = f"{np.median(angles):.1f}" if angles else "n/a"
                parts.append(f"{name} {len(angles)} fasc, {median} deg")
            else:
                # Pixel units: overlays need no calibration.
                parts.append(f"{name} MT {measure_thickness(mask):.0f}px, deep {deep_aponeurosis_angle(mask):.1f} deg")
        tiles.append(overlay(image, gt, pred, title=" | ".join(parts)))

    out = Path(args.out or schema.PROJECT_ROOT / "outputs" / "viz" / f"{args.pool}_{args.source}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_rgb(out, grid(tiles, cols=args.cols))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
