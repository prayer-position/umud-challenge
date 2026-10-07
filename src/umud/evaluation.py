"""Validation metrics aligned with the submission targets.

FascicleAngleEvaluator: fascicle orientation (pennation angle).
ApoThicknessEvaluator: muscle thickness and deep aponeurosis angle.

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
from umud.geometry import angle_difference_deg, deep_aponeurosis_angle, fascicle_angles, measure_thickness
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


class ApoThicknessEvaluator:
    """Aponeurosis validation metric aligned with the submission: muscle
    thickness (see geometry.measure_thickness) from the predicted mask vs the
    ground-truth mask, as a relative error on the native annotation canvas
    (scale-free, so it needs no calibration), plus the deep aponeurosis
    angle error, which feeds the pennation angle."""

    def __init__(self, dataset: MaskDataset):
        self.dataset = dataset
        self.gt_shapes, self.gt_thickness, self.gt_deep_angle = [], [], []
        for idx in range(len(dataset)):
            mask = binarize_mask(load_mask(dataset.mask_path(idx)))
            self.gt_shapes.append(mask.shape)
            self.gt_thickness.append(measure_thickness(mask))
            self.gt_deep_angle.append(deep_aponeurosis_angle(mask))

    @torch.no_grad()
    def __call__(self, model, device, batch_size: int = 8, num_workers: int = 2, tta: bool = False) -> dict:
        model.eval()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        rel_errors, angle_errors, misses, idx = [], [], 0, 0
        for images, _ in loader:
            probs = predict_prob(model, images.to(device), tta=tta).cpu().numpy()
            for prob in probs:
                gt_mt, gt_angle = self.gt_thickness[idx], self.gt_deep_angle[idx]
                pred_mask = upsample_and_threshold(prob, self.gt_shapes[idx])
                idx += 1
                if np.isnan(gt_mt):
                    continue
                mt = measure_thickness(pred_mask)
                if np.isnan(mt):
                    misses += 1
                    rel_errors.append(1.0)
                    continue
                rel_errors.append(abs(mt - gt_mt) / gt_mt)
                angle = deep_aponeurosis_angle(pred_mask)
                if not np.isnan(angle) and not np.isnan(gt_angle):
                    angle_errors.append(angle_difference_deg(angle, gt_angle))
        rel_errors = np.array(rel_errors)
        return {
            # Images without two predicted aponeuroses count as 100% error.
            "mt_rel_err": float(rel_errors.mean()),
            "mt_rel_median_err": float(np.median(rel_errors)),
            "mt_rel_p90_err": float(np.percentile(rel_errors, 90)),
            "deep_angle_err": float(np.mean(angle_errors)) if angle_errors else float("nan"),
            "miss_rate": misses / len(rel_errors),
            "n_images": len(rel_errors),
        }
