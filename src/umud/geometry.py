"""Derive Pennation Angle / Fascicle Length / Muscle Thickness from predicted
segmentation masks.

Measurements take `px_per_mm` as (rows, cols) pixels per millimetre (see
umud/calibration.py): pixel coordinates are converted to millimetres before
any line fitting, so lengths come out in mm and angles are physical even when
a screenshot was stretched unevenly. The default (1, 1) gives pixel units.

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


Scale = tuple[float, float]


def _largest_components(mask: np.ndarray, n: int, px_per_mm: Scale = (1.0, 1.0)) -> list[np.ndarray]:
    """Return up to n largest connected components as (row, col) point
    arrays in mm, largest first."""
    labeled = label(mask > 0)
    regions = sorted(regionprops(labeled), key=lambda r: r.area, reverse=True)
    return [r.coords / np.asarray(px_per_mm) for r in regions[:n]]


def fascicle_lines(
    mask: np.ndarray, min_length_frac: float = MIN_FASCICLE_LENGTH_FRAC, px_per_mm: Scale = (1.0, 1.0)
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Fit a line to every fascicle segment in the mask.

    Returns (centroid, unit_direction, visible_length) per segment that is at
    least `min_length_frac` of the image diagonal long, in mm.
    """
    scale = np.asarray(px_per_mm, dtype=float)
    min_length = min_length_frac * float(np.hypot(*(np.asarray(mask.shape[:2]) / scale)))
    lines = []
    for region in regionprops(label(mask > 0, connectivity=2)):
        if region.area < 2:
            continue
        coords = region.coords / scale
        centroid, direction = _fit_line(coords)
        projections = (coords - centroid) @ direction
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


def _aponeurosis_components(apo_mask: np.ndarray, px_per_mm: Scale) -> list[np.ndarray]:
    """The two largest aponeurosis components in mm, ordered superficial
    (smaller mean row) first."""
    return sorted(_largest_components(apo_mask, n=2, px_per_mm=px_per_mm), key=lambda c: c[:, 0].mean())


def _aponeurosis_lines(apo_mask: np.ndarray, px_per_mm: Scale) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_fit_line(c) for c in _aponeurosis_components(apo_mask, px_per_mm)]


def deep_aponeurosis_angle(apo_mask: np.ndarray, px_per_mm: Scale = (1.0, 1.0)) -> float:
    """Angle of the deep aponeurosis against the image horizontal (see
    `line_angle_deg`), or NaN without two aponeurosis components."""
    lines = _aponeurosis_lines(apo_mask, px_per_mm)
    return line_angle_deg(lines[-1][1]) if len(lines) == 2 else float("nan")


def measure_thickness(apo_mask: np.ndarray, px_per_mm: Scale = (1.0, 1.0)) -> float:
    """Muscle thickness: perpendicular distance from the deep aponeurosis to
    the superficial one, averaged over three points (25/50/75%) across the
    width both aponeuroses cover, as in the manual protocol. Returns NaN if
    fewer than two aponeurosis components are found."""
    components = _aponeurosis_components(apo_mask, px_per_mm)
    if len(components) < 2:
        return float("nan")
    (sup_c, sup_d), (deep_c, deep_d) = (_fit_line(c) for c in components)

    left = max(c[:, 1].min() for c in components)
    right = min(c[:, 1].max() for c in components)
    if right <= left:  # no horizontal overlap: use the deep line's centre
        left = right = deep_c[1]
    normal = np.array([-sup_d[1], sup_d[0]])
    distances = []
    for col in np.linspace(left, right, 5)[1:4]:
        if abs(deep_d[1]) < 1e-6:
            continue
        point = deep_c + deep_d * (col - deep_c[1]) / deep_d[1]
        distances.append(abs(float(np.dot(point - sup_c, normal))))
    return float(np.mean(distances)) if distances else float("nan")


def measure_fascicle(
    fasc_mask: np.ndarray,
    apo_mask: np.ndarray | None = None,
    px_per_mm: Scale = (1.0, 1.0),
    min_length_frac: float = MIN_FASCICLE_LENGTH_FRAC,
) -> tuple[float, float]:
    """Returns (fascicle_length, pennation_angle_deg), each the median over
    all fascicle segments; length is in mm given `px_per_mm`.

    With both aponeuroses found, each segment's line is extended to the
    superficial and deep aponeurosis lines: length is the distance between
    the two intersections, and pennation angle is measured against the deep
    aponeurosis. Without them, length falls back to the segment's visible
    extent and the angle is measured against the image horizontal (or the
    single aponeurosis found).
    """
    lines = fascicle_lines(fasc_mask, min_length_frac, px_per_mm)
    if not lines:
        return float("nan"), float("nan")

    apo_lines = _aponeurosis_lines(apo_mask, px_per_mm) if apo_mask is not None else []
    reference = apo_lines[-1][1] if apo_lines else np.array([0.0, 1.0])

    angles = [_acute_angle_deg(direction, reference) for _, direction, _ in lines]

    lengths = []
    if len(apo_lines) == 2:
        (sup_c, sup_d), (deep_c, deep_d) = apo_lines
        max_length = 4 * float(np.hypot(*(np.asarray(fasc_mask.shape[:2]) / np.asarray(px_per_mm))))
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

    return float(np.median(lengths)), float(np.median(angles))
