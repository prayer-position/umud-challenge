"""Derive Pennation Angle / Fascicle Length / Muscle Thickness from predicted
segmentation masks.

pixel_to_mm calibration is unconfirmed (no scale bar / spacing metadata was
found in the extracted data during EDA) -- results are in pixel units unless
a calibration factor is supplied, and default to 1.0 (i.e. pixel units)
until this is resolved. Sanity-check against the real leaderboard score
after a first submission.

Fascicle masks hold many short line segments (~17 per training image, each
4-25% of the image diagonal), and fascicles usually run past the visible
segment. So each segment is fitted as a line and extrapolated to the
superficial and deep aponeuroses: fascicle length is the distance between
those two intersections, pennation angle is the angle against the deep
aponeurosis, and both are reported as the median over all segments.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

# Segments shorter than this fraction of the image diagonal are ignored:
# 3% sits at the 5th percentile of ground-truth segment lengths.
MIN_FASCICLE_LENGTH_FRAC = 0.03


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


def fascicle_lines(
    mask: np.ndarray, min_length_frac: float = MIN_FASCICLE_LENGTH_FRAC
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Fit a line to every fascicle segment in the mask.

    Returns (centroid, unit_direction, visible_length_px) per segment that is
    at least `min_length_frac` of the image diagonal long.
    """
    min_length = min_length_frac * float(np.hypot(*mask.shape[:2]))
    lines = []
    for region in regionprops(label(mask > 0, connectivity=2)):
        if region.area < 2:
            continue
        centroid, direction = _fit_line(region.coords.astype(float))
        projections = (region.coords - centroid) @ direction
        length = float(projections.max() - projections.min())
        if length >= min_length:
            lines.append((centroid, direction, length))
    return lines


def line_angle_deg(direction: np.ndarray) -> float:
    """Angle of a (row, col) direction against the image horizontal, in
    (-90, 90]; positive when the line rises to the right (rows grow down)."""
    angle = float(np.degrees(np.arctan2(-direction[0], direction[1])))
    if angle <= -90:
        angle += 180
    elif angle > 90:
        angle -= 180
    return angle


def fascicle_angles(mask: np.ndarray, min_length_frac: float = MIN_FASCICLE_LENGTH_FRAC) -> list[float]:
    """Per-segment fascicle angles against the image horizontal (see
    `line_angle_deg`)."""
    return [line_angle_deg(direction) for _, direction, _ in fascicle_lines(mask, min_length_frac)]


def angle_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two undirected line angles, in [0, 90]."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _acute_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    cos_angle = min(1.0, abs(float(np.dot(u, v))))
    return float(np.degrees(np.arccos(cos_angle)))


def _intersect(p: np.ndarray, d: np.ndarray, q: np.ndarray, e: np.ndarray) -> float | None:
    """Parameter t such that p + t*d lies on the line q + s*e, or None if the
    lines are (nearly) parallel."""
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(cross) < 1e-6:
        return None
    w = q - p
    return float((w[0] * e[1] - w[1] * e[0]) / cross)


def _aponeurosis_lines(apo_mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lines fitted to the two largest aponeurosis components, ordered
    superficial (smaller mean row) first."""
    components = sorted(_largest_components(apo_mask, n=2), key=lambda c: c[:, 0].mean())
    return [_fit_line(c.astype(float)) for c in components]


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
    min_length_frac: float = MIN_FASCICLE_LENGTH_FRAC,
) -> tuple[float, float]:
    """Returns (fascicle_length_mm, pennation_angle_deg), each the median over
    all fascicle segments.

    With both aponeuroses found, each segment's line is extended to the
    superficial and deep aponeurosis lines: length is the distance between
    the two intersections, and pennation angle is measured against the deep
    aponeurosis. Without them, length falls back to the segment's visible
    extent and the angle is measured against the image horizontal (or the
    single aponeurosis found).
    """
    lines = fascicle_lines(fasc_mask, min_length_frac)
    if not lines:
        return float("nan"), float("nan")

    apo_lines = _aponeurosis_lines(apo_mask) if apo_mask is not None else []
    reference = apo_lines[-1][1] if apo_lines else np.array([0.0, 1.0])

    angles = [_acute_angle_deg(direction, reference) for _, direction, _ in lines]

    lengths = []
    if len(apo_lines) == 2:
        (sup_c, sup_d), (deep_c, deep_d) = apo_lines
        max_length = 4 * float(np.hypot(*fasc_mask.shape[:2]))
        for centroid, direction, _ in lines:
            t_sup = _intersect(centroid, direction, sup_c, sup_d)
            t_deep = _intersect(centroid, direction, deep_c, deep_d)
            if t_sup is None or t_deep is None:
                continue
            length = abs(t_deep - t_sup)
            # Near-parallel segments extrapolate to absurd lengths; drop them.
            if length <= max_length:
                lengths.append(length)
    if not lengths:
        lengths = [visible for _, _, visible in lines]

    return float(np.median(lengths) * pixel_to_mm), float(np.median(angles))
