"""Score one model or an ensemble on a pool's validation split, image by
image from the raw files, and list the worst validation images.

    python scripts/eval_ensemble.py --pool apo --split-config configs/apo_seg_grouped.yaml \
        --member outputs/checkpoints/a.pt configs/apo_seg_grouped.yaml \
        --member outputs/checkpoints/b.pt configs/apo_seg_512x768.yaml --details 15

Members' probabilities are averaged on each image's label canvas (see
inference.ensemble_prob). Aponeuroses: muscle-thickness relative error and
deep-aponeurosis angle error for both pairing rules ("largest", "topmost")
and both edge conventions ("center", "inner"), each scored against the
ground-truth mask measured the same way. Fascicles: median-angle error.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.measure import label

from umud import utils
from umud.data import schema
from umud.data.dataset import binarize_mask, load_image, load_mask
from umud.data.splits import split_indices
from umud.evaluation import MISS_PENALTY_DEG
from umud.geometry import (
    _aponeurosis_pair,
    angle_difference_deg,
    deep_aponeurosis_angle,
    fascicle_angles,
    measure_thickness,
)
from umud.inference import ensemble_prob, load_members
from umud.train_pipeline import POOL_DIRS

APO_VARIANTS = [("largest", "center"), ("topmost", "center"), ("topmost", "inner")]


def _summary(values: list[float]) -> dict:
    """Mean/median/p90 over the finite values (images whose ground truth
    can't be measured are NaN and skipped)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": int(len(v))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", choices=["apo", "fasc"], required=True)
    parser.add_argument("--split-config", required=True, help="Config whose split/seed/val_frac define validation.")
    parser.add_argument("--member", nargs=2, action="append", required=True, metavar=("CHECKPOINT", "CONFIG"))
    parser.add_argument("--details", type=int, default=0, help="Print the N worst validation images.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = utils.get_device()
    split_config = utils.load_config(args.split_config)
    image_dir, mask_dir = POOL_DIRS[args.pool]
    paths = schema.list_image_files(image_dir)
    _, val_idx = split_indices(image_dir, split_config)
    members = load_members(args.member, device)

    records = []
    for idx in val_idx:
        path = paths[idx]
        gt = binarize_mask(load_mask(mask_dir / path.name))
        prob = ensemble_prob(members, load_image(path), device, gt.shape)
        pred = (prob > 0.5).astype(np.uint8)
        rec = {"image": path.name}
        if args.pool == "apo":
            for rule, edge in APO_VARIANTS:
                g, p = measure_thickness(gt, rule=rule, edge=edge), measure_thickness(pred, rule=rule, edge=edge)
                # A missed prediction counts as 100% error; unmeasurable ground
                # truth leaves the image out (NaN).
                rec[f"mt_err_{rule}_{edge}"] = np.nan if np.isnan(g) else 1.0 if np.isnan(p) else abs(p - g) / g
                if (rule, edge) == ("topmost", "center"):
                    rec.update(gt_mt=g, pred_mt=p, n_pred_components=int(label(pred).max()))
                    rec["pred_rows"] = [round(float(c[:, 0].mean()), 1) for _, _, c in _aponeurosis_pair(pred, (1.0, 1.0))]
                    rec["gt_rows"] = [round(float(c[:, 0].mean()), 1) for _, _, c in _aponeurosis_pair(gt, (1.0, 1.0))]
            ga, pa = deep_aponeurosis_angle(gt), deep_aponeurosis_angle(pred)
            rec["deep_angle_err"] = angle_difference_deg(pa, ga) if not (np.isnan(ga) or np.isnan(pa)) else np.nan
        else:
            ga, pa = fascicle_angles(gt), fascicle_angles(pred)
            if not ga:
                continue
            rec["angle_err"] = MISS_PENALTY_DEG if not pa else angle_difference_deg(np.median(pa), np.median(ga))
        records.append(rec)

    metrics = {"name": args.name, "members": args.member, "n_images": len(records)}
    if args.pool == "apo":
        for rule, edge in APO_VARIANTS:
            metrics[f"mt_rel_err_{rule}_{edge}"] = _summary([r[f"mt_err_{rule}_{edge}"] for r in records])
        metrics["deep_angle_err"] = _summary([r["deep_angle_err"] for r in records if not np.isnan(r["deep_angle_err"])])
        key = "mt_err_topmost_center"
    else:
        metrics["angle_err"] = _summary([r["angle_err"] for r in records])
        key = "angle_err"

    print(json.dumps({k: v for k, v in metrics.items() if k != "members"}, indent=1))
    if args.details:
        print(f"worst {args.details} validation images by {key}:")
        for r in sorted(records, key=lambda r: -np.nan_to_num(r[key], nan=-1))[: args.details]:
            print("  " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    if args.out:
        Path(args.out).write_text(json.dumps({**metrics, "per_image": records}, indent=1, default=float))


if __name__ == "__main__":
    main()
