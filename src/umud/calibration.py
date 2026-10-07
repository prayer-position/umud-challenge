"""Per-image pixel -> millimetre calibration from the scanner's on-screen rulers.

The test images come from three scanners with no spacing metadata, but every
layout draws a depth and/or lateral ruler. Each detector below recognises one
layout, finds the ruler's tick marks and converts their spacing to pixels per
millimetre, separately for x and y (the Telemed 1200x800 screenshots were
stretched unevenly: 10 mm is 148 px vertically but 134.5 px horizontally).

Layouts (test set sizes in brackets):
- telemed_crop   (~853x1069, ~513x465): image crops with a lateral ruler on
  the second-to-last row; ticks every 10 mm, drawn in the ruler's exact
  colour. Square pixels assumed (no depth ruler).
- telemed_screen (800x1200): EchoWave screen with a 0..De mm depth ruler on
  the left (10 mm ticks) and a lateral ruler along the bottom (10 mm ticks).
- telemed_native (644x1088): the same screen, unscaled; same rulers.
- siemens        (800x1200, black UI): depth ruler right of the image; the
  tick unit changes with depth (2 mm at 3 cm, 5 mm from 3.5 cm), so it is
  resolved from the 12L3 probe's fixed field of view (~57 mm wide).
  Square pixels assumed (no lateral ruler).
- lumify         (800x1200 PNG): depth ruler at the left edge, 5 mm ticks.
  Square pixels assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Width of the Siemens 12L3 image area in mm, measured on test images whose
# tick unit is unambiguous (4.5 cm depth: 777 px at 13.5 px/mm).
SIEMENS_FOV_MM = 57.3


@dataclass
class Calibration:
    px_per_mm_x: float
    px_per_mm_y: float
    layout: str

    @property
    def scale(self) -> tuple[float, float]:
        """(rows, cols) pixels per mm, matching (row, col) coordinates."""
        return self.px_per_mm_y, self.px_per_mm_x


def _runs(mask: np.ndarray, max_gap: int = 2) -> list[float]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > max_gap) + 1)
    return [float(g.mean()) for g in groups]


def _period(positions: list[float], lo: float, hi: float, min_ticks: int = 3) -> float | None:
    """Tick period from tick positions, tolerating missed or spurious ticks.

    Each gap inside [lo, hi] is a candidate period, scored by how many gaps
    are a whole multiple (1-3x) of it; ties go to the larger candidate (a
    regular series fits P, P/2 and P/3 equally well). The result is the
    median of the supporting gaps divided by their multiples."""
    if len(positions) < min_ticks:
        return None
    gaps = np.diff(sorted(positions))
    candidates = gaps[(gaps >= lo) & (gaps <= hi)]
    if len(candidates) == 0:
        return None

    def supporting(c: float) -> np.ndarray:
        k = np.round(gaps / c)
        return (k >= 1) & (k <= 3) & (np.abs(gaps / c - k) < 0.04)

    best = max(candidates, key=lambda c: (supporting(c).sum(), c))
    mask = supporting(best)
    if mask.sum() < min_ticks - 1:
        return None
    return float(np.median(gaps[mask] / np.round(gaps[mask] / best)))


def _strip_ticks(gray: np.ndarray, axis: str, lo: int, hi: int) -> list[float]:
    """Bright tick centres in a strip: columns lo:hi (axis 'v', ticks along
    rows) or rows lo:hi (axis 'h', ticks along columns)."""
    strip = gray[:, lo:hi] if axis == "v" else gray[lo:hi, :].T
    profile = strip.max(axis=1)
    threshold = (np.percentile(profile, 50) + np.percentile(profile, 99)) / 2
    if np.percentile(profile, 99) - np.percentile(profile, 50) < 40:
        return []
    return _runs(profile > threshold)


def _ruler_ticks(gray: np.ndarray, axis: str, search: range, tick_len: int = 3) -> list[float]:
    """Ticks of a Telemed-style ruler: a 1px bright line (axis 'h': a row,
    'v': a column, searched over `search`) with short ticks on one side.
    Resized screenshots blend the ruler's colour, so ticks are pixels at
    least 80% as bright as the line, in narrow runs, on all `tick_len`
    lines next to it. Returns the tick centres along the line."""
    arr = gray if axis == "h" else gray.T
    best: list[float] = []
    for line in search:
        if not 0 <= line < arr.shape[0]:
            continue
        level = np.median(arr[line])
        # The ruler line is bright and nearly uniform along most of its length.
        if level < 100 or np.mean(np.abs(arr[line] - level) < 15) < 0.4:
            continue
        for side in (-1, 1):
            neighbours = [line + side * k for k in range(1, tick_len + 1)]
            if not all(0 <= q < arr.shape[0] for q in neighbours):
                continue
            tick = np.all(arr[neighbours] >= 0.8 * level, axis=0)
            if tick.mean() > 0.2:
                # Bright tissue next to the ruler: fall back to the ruler's
                # exact grey level (works on crops that were not resized).
                tick = np.all(np.abs(arr[neighbours] - level) < 0.5, axis=0)
            if tick.mean() > 0.2:  # a parallel line, not ticks
                continue
            idx = np.flatnonzero(tick)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
            positions = [float(r.mean()) for r in runs if len(r) <= 6]
            if len(positions) > len(best):
                best = positions
    return best


def _telemed(image: np.ndarray, layout: str) -> Calibration | None:
    """Telemed rulers, 10 mm per tick: lateral along the bottom, depth (on
    full screens) along the left."""
    h, w = image.shape[:2]
    gray = image.astype(np.float32).mean(axis=-1)
    px = _period(_ruler_ticks(gray, "h", range(h - 15, h)), 40, 250)
    py = _period(_ruler_ticks(gray, "v", range(0, 90)), 40, 250) if layout != "telemed_crop" else None
    if px is None and py is None:
        return None
    px, py = px or py, py or px
    if layout == "telemed_native" and not 0.8 < px / py < 1.25:
        px = py  # unscaled screens have square pixels; trust the depth ruler
    return Calibration(px / 10, py / 10, layout)


def _siemens(gray: np.ndarray) -> Calibration | None:
    ticks = _strip_ticks(gray, "v", 1140, 1162)
    period = _period(ticks, 20, 100, min_ticks=4)
    if period is None:
        return None
    # Image-area width: columns with ultrasound content between the ruler's
    # first tick and the next one down.
    top = int(min(ticks)) + 10
    band = gray[top : top + int(period), 120:1120]
    cols = np.flatnonzero(band.mean(axis=0) > 15)
    if len(cols) < 100:
        return None
    # The longest run of bright columns; the settings text on the left is
    # separated from the image by a dark gap.
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 5) + 1)
    width = len(max(runs, key=len))
    unit = min((1.0, 2.0, 5.0, 10.0), key=lambda u: abs(width / (period / u) - SIEMENS_FOV_MM))
    return Calibration(period / unit, period / unit, "siemens")


def _lumify(gray: np.ndarray) -> Calibration | None:
    period = _period(_strip_ticks(gray, "v", 2, 14), 40, 120, min_ticks=4)
    if period is None:
        return None
    return Calibration(period / 5, period / 5, "lumify")


def calibrate(image: np.ndarray, is_png: bool = False) -> Calibration | None:
    """Pixels per mm for one RGB screen image, or None if no known layout's
    ruler is found."""
    gray = image.astype(np.float32).mean(axis=-1)
    h, w = gray.shape
    if (h, w) == (644, 1088):
        return _telemed(image, "telemed_native")
    if (h, w) == (800, 1200):
        if is_png:
            return _lumify(gray)
        # Telemed screens have a mid-grey UI; Siemens screens are black.
        if np.median(gray[:, :40]) > 30:
            return _telemed(image, "telemed_screen")
        return _siemens(gray)
    return _telemed(image, "telemed_crop")
