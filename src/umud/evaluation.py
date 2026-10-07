"""Fascicle-orientation validation metric.

Dice on 1px-wide fascicle lines mostly measures whether a predicted line
lands on exactly the right pixels; what the submission needs is fascicle
orientation (pennation angle) and the lines' extent. This scores each
validation image by the difference between the median predicted and median
ground-truth fascicle angle, measured on the native annotation canvas so the
number is comparable across training resolutions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from umud.data.dataset import MaskDataset, binarize_mask, load_mask
from umud.geometry import angle_difference_deg, fascicle_angles
from umud.inference import predict_prob, upsample_and_threshold

# Error charged for an image where no fascicle is predicted at all.
MISS_PENALTY_DEG = 45.0


class FascicleAngleEvaluator:
    def __init__(self, dataset: MaskDataset):
        self.dataset = dataset
        self.gt_shapes = []
        self.gt_angles = []
        for idx in range(len(dataset)):
            mask = binarize_mask(load_mask(dataset.mask_path(idx)))
            angles = fascicle_angles(mask)
            self.gt_shapes.append(mask.shape)
            self.gt_angles.append(float(np.median(angles)) if angles else float("nan"))

    @torch.no_grad()
    def __call__(self, model, device, batch_size: int = 8, num_workers: int = 2, tta: bool = False) -> dict:
        """`dataset` must yield normalized, un-augmented images in order."""
        model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        errors, misses, counts, idx = [], 0, [], 0
        for images, _ in loader:
            probs = predict_prob(model, images.to(device), tta=tta).cpu().numpy()
            for prob in probs:
                gt = self.gt_angles[idx]
                pred_mask = upsample_and_threshold(prob, self.gt_shapes[idx])
                idx += 1
                if np.isnan(gt):
                    continue
                angles = fascicle_angles(pred_mask)
                counts.append(len(angles))
                if not angles:
                    misses += 1
                    errors.append(MISS_PENALTY_DEG)
                else:
                    errors.append(angle_difference_deg(float(np.median(angles)), gt))
        errors = np.array(errors)
        return {
            # Misses count as MISS_PENALTY_DEG.
            "angle_mae": float(errors.mean()),
            "angle_median_err": float(np.median(errors)),
            "angle_p90_err": float(np.percentile(errors, 90)),
            "miss_rate": misses / len(errors),
            "segments_per_image": float(np.mean(counts)),
            "n_images": len(errors),
        }
