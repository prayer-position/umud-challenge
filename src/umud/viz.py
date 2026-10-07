"""Overlay rendering for eyeballing masks (ground truth and predictions)."""

from __future__ import annotations

import cv2
import numpy as np

GT_COLOR = (0, 255, 0)  # RGB green
PRED_COLOR = (255, 0, 255)  # RGB magenta


def _paint(canvas: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    # Float area-resize: any coverage counts, so 1px lines survive downscaling.
    mask = cv2.resize(mask.astype(np.float32), (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_AREA) > 0
    if thickness > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    canvas[mask] = (0.25 * canvas[mask] + 0.75 * np.array(color)).astype(np.uint8)


def overlay(
    image: np.ndarray,
    gt_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
    width: int = 600,
    title: str = "",
    thickness: int = 3,
) -> np.ndarray:
    """Render an RGB tile: the image (resized to `width`, aspect kept) with the
    ground-truth mask in green and the predicted mask in magenta. Masks may be
    at any resolution; they are resized onto the image (the masks' annotation
    canvas does not always match the image's native size)."""
    height = round(image.shape[0] * width / image.shape[1])
    canvas = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).copy()
    if gt_mask is not None:
        _paint(canvas, gt_mask, GT_COLOR, thickness)
    if pred_mask is not None:
        _paint(canvas, pred_mask, PRED_COLOR, thickness)
    if title:
        banner = np.zeros((30, width, 3), dtype=np.uint8)
        cv2.putText(banner, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas = np.vstack([banner, canvas])
    return canvas


def grid(tiles: list[np.ndarray], cols: int = 3, pad: int = 6) -> np.ndarray:
    """Stack tiles of equal width into a grid (shorter tiles are padded)."""
    width = tiles[0].shape[1]
    tile_h = max(t.shape[0] for t in tiles)
    rows = -(-len(tiles) // cols)
    out = np.full((rows * (tile_h + pad) + pad, cols * (width + pad) + pad, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = pad + r * (tile_h + pad), pad + c * (width + pad)
        out[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return out


def save_rgb(path, image: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
