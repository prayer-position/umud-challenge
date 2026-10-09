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


# How the aponeurosis pair is chosen from a mask:
# - "topmost" (default): join broken fragments of one aponeurosis, keep
#   structures spanning >= APO_MIN_SPAN of the widest one and within
#   APO_MAX_TILT degrees of its angle (some apo masks also contain oblique
#   fascicle lines), take the topmost as superficial and the next one at
#   least APO_MIN_GAP of the image height below it as deep. ~10% of
#   ground-truth apo masks hold 3+ wide structures (two stacked muscles,
#   double lines); the target muscle is the top one.
# - "largest": the two largest components (the original rule).
APO_RULE = "topmost"
# Where an aponeurosis line sits: "center" of its band, or the "inner"
# (muscle-side) edge: the lower edge of the superficial aponeurosis and the
# upper edge of the deep one.
APO_EDGE = "center"
APO_MIN_AREA = 0.02  # of the largest component
APO_MERGE_GAP = 0.02  # of the image height: fragments this close are one line
APO_MIN_SPAN = 0.5  # of the widest structure's horizontal span
APO_MAX_TILT = 10.0  # degrees from the widest structure
APO_MIN_GAP = 0.08  # of the image height between superficial and deep


def _row_at(centroid: np.ndarray, direction: np.ndarray, col: float) -> float:
    if abs(direction[1]) < 1e-6:
        return float(centroid[0])
    return float(centroid[0] + direction[0] * (col - centroid[1]) / direction[1])


def _edge_points(coords_px: np.ndarray, lower: bool) -> np.ndarray:
    """Per image column, the lowest (lower=True) or highest pixel of a band."""
    order = np.lexsort((coords_px[:, 0] if lower else -coords_px[:, 0], coords_px[:, 1]))
    sorted_coords = coords_px[order]
    last_of_col = np.r_[np.flatnonzero(np.diff(sorted_coords[:, 1])), len(sorted_coords) - 1]
    return sorted_coords[last_of_col]


def _aponeurosis_pair(
    apo_mask: np.ndarray, px_per_mm: Scale, rule: str | None = None, edge: str | None = None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Superficial and deep aponeurosis as (centroid, direction, coords), all
    in mm, superficial first; fewer than two entries if not found."""
    rule, edge = rule or APO_RULE, edge or APO_EDGE
    scale = np.asarray(px_per_mm, dtype=float)
    regions = sorted(regionprops(label(apo_mask > 0)), key=lambda r: r.area, reverse=True)
    if rule == "largest":
        groups = [r.coords for r in regions[:2]]
    else:
        regions = [r for r in regions if regions and r.area >= APO_MIN_AREA * regions[0].area]
        height_mm = apo_mask.shape[0] / scale[0]
        lines = [_fit_line(r.coords / scale) for r in regions]
        spans = [(r.coords[:, 1].min() / scale[1], r.coords[:, 1].max() / scale[1]) for r in regions]
        parent = list(range(len(regions)))

        def find(i: int) -> int:
            while parent[i] != i:
                i = parent[i]
            return i

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                # Compare the two lines where they overlap, or midway across
                # the gap between them.
                lo, hi = max(spans[i][0], spans[j][0]), min(spans[i][1], spans[j][1])
                col = (lo + hi) / 2
                gap = abs(_row_at(*lines[i], col) - _row_at(*lines[j], col))
                if gap < APO_MERGE_GAP * height_mm:
                    parent[find(i)] = find(j)
        merged: dict[int, list[np.ndarray]] = {}
        for i, region in enumerate(regions):
            merged.setdefault(find(i), []).append(region.coords)
        groups = [np.concatenate(g) for g in merged.values()]
        widths = [(g[:, 1].max() - g[:, 1].min()) / scale[1] for g in groups]
        if groups:
            widest = _fit_line(groups[int(np.argmax(widths))] / scale)[1]
            groups = [
                g
                for g, w in zip(groups, widths)
                if w >= APO_MIN_SPAN * max(widths) and _acute_angle_deg(_fit_line(g / scale)[1], widest) <= APO_MAX_TILT
            ]
        centre = apo_mask.shape[1] / 2 / scale[1]
        rows = [_row_at(*_fit_line(g / scale), centre) for g in groups]
        order = np.argsort(rows)
        groups = [groups[i] for i in order]
        rows = [rows[i] for i in order]
        if len(groups) >= 2:
            deep = next((k for k in range(1, len(groups)) if rows[k] - rows[0] >= APO_MIN_GAP * height_mm), 1)
            groups = [groups[0], groups[deep]]

    groups = sorted(groups, key=lambda g: g[:, 0].mean())
    pair = []
    for k, coords in enumerate(groups):
        points = coords
        if edge == "inner" and len(groups) == 2:
            points = _edge_points(coords, lower=(k == 0))
        centroid, direction = _fit_line(points / scale)
        pair.append((centroid, direction, coords / scale))
    return pair


def _aponeurosis_lines(apo_mask: np.ndarray, px_per_mm: Scale, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(c, d) for c, d, _ in _aponeurosis_pair(apo_mask, px_per_mm, **kwargs)]


def deep_aponeurosis_angle(apo_mask: np.ndarray, px_per_mm: Scale = (1.0, 1.0), **kwargs) -> float:
    """Angle of the deep aponeurosis against the image horizontal (see
    `line_angle_deg`), or NaN without two aponeuroses. kwargs: rule, edge."""
    lines = _aponeurosis_lines(apo_mask, px_per_mm, **kwargs)
    return line_angle_deg(lines[-1][1]) if len(lines) == 2 else float("nan")


def measure_thickness(apo_mask: np.ndarray, px_per_mm: Scale = (1.0, 1.0), **kwargs) -> float:
    """Muscle thickness: perpendicular distance from the deep aponeurosis to
    the superficial one, averaged over three points (25/50/75%) across the
    width both aponeuroses cover, as in the manual protocol. Returns NaN if
    fewer than two aponeuroses are found. kwargs: rule, edge (see APO_RULE,
    APO_EDGE)."""
    pair = _aponeurosis_pair(apo_mask, px_per_mm, **kwargs)
    if len(pair) < 2:
        return float("nan")
    (sup_c, sup_d, sup_coords), (deep_c, deep_d, deep_coords) = pair
    components = [sup_coords, deep_coords]

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
    **apo_kwargs,
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

    apo_lines = _aponeurosis_lines(apo_mask, px_per_mm, **apo_kwargs) if apo_mask is not None else []
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
