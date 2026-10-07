"""Score a checkpoint on its pool's validation split: Dice plus the
submission-aligned metric (fascicles: angle error; aponeuroses: thickness
and deep-aponeurosis angle error; see umud/evaluation.py).

    python scripts/eval_model.py --pool fasc --config configs/fasc_seg.yaml --checkpoint outputs/checkpoints/fasc_best.pt
    python scripts/eval_model.py --pool apo --config configs/apo_seg.yaml --checkpoint outputs/checkpoints/apo_best.pt

The split comes from the config (`split`, `seed`, `val_frac`), so pass the
config the comparison should use.
"""

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from umud import utils
from umud.data import schema
from umud.data.dataset import MaskDataset
from umud.data.transforms import build_val_transform
from umud.engine import DiceBCELoss, evaluate
from umud.evaluation import ApoThicknessEvaluator, FascicleAngleEvaluator
from umud.inference import load_model
from umud.data.splits import split_indices
from umud.train_pipeline import POOL_DIRS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=["apo", "fasc"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["random", "grouped"], default=None, help="Override the config's split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--out", default=None, help="Also write the metrics to this JSON file.")
    args = parser.parse_args()

    config = utils.load_config(args.config)
    if args.split:
        config["split"] = args.split
    device = utils.get_device()
    image_size = tuple(config["image_size"])
    image_dir, mask_dir = POOL_DIRS[args.pool]
    _, val_idx = split_indices(image_dir, config)
    val_ds = MaskDataset(
        image_dir,
        mask_dir,
        image_size,
        build_val_transform(image_size),
        line_px=config.get("line_px", 0),
        indices=val_idx,
    )
    model = load_model(Path(args.checkpoint), config, device)

    loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=config["num_workers"])
    # Dice is against each config's own (possibly dilated) targets, so it is
    # only comparable between runs with the same image_size and line_px.
    metrics = evaluate(model, loader, DiceBCELoss(), device, amp=device.type == "cuda")
    metrics.update(
        (FascicleAngleEvaluator if args.pool == "fasc" else ApoThicknessEvaluator)(val_ds)(
            model, device, batch_size=config["batch_size"], num_workers=config["num_workers"], tta=not args.no_tta and config.get("flip", False),
        )
    )
    metrics.update(pool=args.pool, split=config.get("split", "random"), config=args.config, checkpoint=args.checkpoint, tta=not args.no_tta and config.get("flip", False))
    print(json.dumps(metrics, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
