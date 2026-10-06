"""UMUD Score approximation: normalized MAE averaged across PA/FL/MT.

The competition's exact normalization formula wasn't confirmed from the
Kaggle page (JS-rendered, not fetchable) -- `scales` is a provisional
per-target normalizer (e.g. train-set std or range) for local validation
only. Calibrate against the real leaderboard score after the first
submission and adjust if they diverge.
"""

import numpy as np


def normalized_mae(preds: np.ndarray, targets: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(preds - targets) / scale))


def umud_score(
    preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    scales: dict[str, float],
) -> float:
    per_target = {
        key: normalized_mae(preds[key], targets[key], scales[key]) for key in targets
    }
    return float(np.mean(list(per_target.values())))
