"""Derive Pennation Angle / Fascicle Length / Muscle Thickness from predicted
segmentation masks.

pixel_to_mm calibration is unconfirmed (no scale bar / spacing metadata was
found in the extracted data during EDA) -- results are in pixel units unless
a calibration factor is supplied, and default to 1.0 (i.e. pixel units)
until this is resolved. Sanity-check against the real leaderboard score
after a first submission.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a line to (row, col) points via PCA.

    Returns (centroid, unit_direction).
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    return centroid, direction / np.linalg.norm(direction)


def _largest_components(mask: np.ndarray, n: int) -> list[np.ndarray]:
    """Return up to n largest connected components as (row, col) point
    arrays, largest first."""
    labeled = label(mask > 0)
    regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
    return [r.coords for r in regions[:n]]


def measure_thickness(apo_mask: np.ndarray, pixel_to_mm: float = 1.0) -> float:
    """Perpendicular distance between the two largest components of the
    aponeurosis mask (assumed to be the superficial and deep aponeurosis
    lines). Returns NaN if fewer than two components are found."""
    components = _largest_components(apo_mask, n=2)
    if len(components) < 2:
        return float("nan")

    centroid_a, direction_a = _fit_line(components[0])
    centroid_b, _ = _fit_line(components[1])

    normal = np.array([-direction_a[1], direction_a[0]])
    distance_px = abs(np.dot(centroid_b - centroid_a, normal))
    return float(distance_px * pixel_to_mm)


def measure_fascicle(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    pixel_to_mm: float = 1.0,
) -> tuple[float, float]:
    """Returns (fascicle_length_mm, pennation_angle_deg).

    Pennation angle is measured against the deepest aponeurosis component
    (largest mean row, i.e. furthest down the image) if apo_mask is given,
    otherwise against the image's horizontal axis.
    """
    components = _largest_components(fasc_mask, n=1)
    if not components:
        return float("nan"), float("nan")

    points = components[0]
    centroid, direction = _fit_line(points)

    projections = (points - centroid) @ direction
    length_px = projections.max() - projections.min()
    length_mm = float(length_px * pixel_to_mm)

    apo_direction = np.array([0.0, 1.0])
    if apo_mask is not None:
        apo_components = _largest_components(apo_mask, n=2)
        if apo_components:
            deep = max(apo_components, key=lambda c: c[:, 0].mean())
            _, apo_direction = _fit_line(deep)

    cos_angle = abs(np.dot(direction, apo_direction))
    cos_angle = min(1.0, max(-1.0, cos_angle))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    return length_mm, angle_deg
