import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset

from umud.data import schema


def load_image(path: Path) -> np.ndarray:
    # Some test "*.tif" files are really PNGs, so sniff the header rather
    # than trusting the extension.
    with open(path, "rb") as f:
        is_tiff = f.read(4) in (b"II*\x00", b"MM\x00*")
    if is_tiff:
        arr = tifffile.imread(str(path))
    else:
        arr = np.array(Image.open(path).convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    """Load a mask TIFF as an integer label array.

    ImageJ palette-style masks store pixel values as either a single-channel
    index array or, depending on how the writer flattened the palette, an
    RGB array of distinct colors. Both cases are handled; RGB masks are
    collapsed to integer labels by color frequency (most frequent color ->
    background/0).
    """
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3:
        flat = arr.reshape(-1, arr.shape[-1])
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        order = np.argsort(-counts)
        colors = colors[order]
        labels = np.zeros(arr.shape[:2], dtype=np.int64)
        for idx, color in enumerate(colors):
            if idx == 0:
                continue
            labels[np.all(arr == color, axis=-1)] = idx
        return labels
    raise ValueError(f"Unexpected mask shape {arr.shape} for {path}")


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a raw label mask to a foreground boolean array.

    Confirmed against real data (see notebooks/01_eda.ipynb): apo masks and
    fasc masks use OPPOSITE polarity -- apo foreground (the two thin
    aponeurosis lines) is labeled 0 against a 255 background, while fasc
    foreground (the thin fascicle line) is labeled 255 against a 0
    background. Both are strictly binary. Rather than hardcode either
    convention, treat whichever value is the minority (smaller pixel count)
    as foreground, since the anatomical structures of interest are always
    thin lines occupying a small fraction of the image.
    """
    values, counts = np.unique(mask, return_counts=True)
    if len(values) <= 1:
        return np.zeros_like(mask, dtype=bool)
    minority_value = values[np.argmin(counts)]
    return mask == minority_value


def resize_mask(mask: np.ndarray, size: tuple[int, int], line_px: int = 0) -> np.ndarray:
    """Resize a binary mask to (h, w).

    With line_px == 0 this is plain nearest-neighbour resizing. Fascicle
    labels are 1px-wide lines, which nearest-neighbour downsampling breaks
    into scattered dots; with line_px > 0, every target pixel that the line
    touches is kept (area resize, any coverage -> foreground) and the result
    is dilated to roughly line_px pixels wide.
    """
    h, w = size
    if line_px <= 0:
        return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    out = (cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    if line_px > 1:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (line_px, line_px)))
    return out


class MaskDataset(Dataset):
    """Binary foreground/background segmentation dataset for one image/mask
    pool (apo or fasc). See `binarize_mask` for how raw mask values are
    converted to foreground/background.

    Mask TIFFs are exported at a fixed annotation-canvas resolution that
    does not match each source image's native resolution (confirmed via
    notebooks/01_eda.ipynb and a visual overlay check -- squashing the image
    to the mask's resolution, despite the aspect-ratio change, produces
    correct anatomical alignment). Both image and mask are independently
    resized to `image_size` here, before Albumentations runs, since
    Albumentations requires matching input shapes and this keeps training
    and inference (where no mask exists to size against) consistent.

    `indices` restricts the dataset to those positions in the sorted file
    list. With `cache`, every resized image/mask pair is decoded once up
    front and kept in memory (decoding the LZW TIFFs dominates per-sample
    cost otherwise).
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: tuple[int, int],
        transform=None,
        line_px: int = 0,
        indices: list[int] | None = None,
        cache: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_paths = schema.list_image_files(self.image_dir)
        if indices is not None:
            self.image_paths = [self.image_paths[i] for i in indices]
        self.image_size = tuple(image_size)
        self.transform = transform
        self.line_px = line_px
        self._images = self._masks = None
        if cache:
            h, w = self.image_size
            self._images = np.empty((len(self), h, w, 3), dtype=np.uint8)
            self._masks = np.empty((len(self), h, w), dtype=np.uint8)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
                for idx, (image, mask) in enumerate(pool.map(self._load_resized, range(len(self)))):
                    self._images[idx], self._masks[idx] = image, mask

    def __len__(self) -> int:
        return len(self.image_paths)

    def mask_path(self, idx: int) -> Path:
        return self.mask_dir / self.image_paths[idx].name

    def _load_resized(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = load_image(self.image_paths[idx])
        mask = binarize_mask(load_mask(self.mask_path(idx)))
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        return image, resize_mask(mask, self.image_size, self.line_px)

    def __getitem__(self, idx: int):
        if self._images is not None:
            image, mask = self._images[idx], self._masks[idx]
        else:
            image, mask = self._load_resized(idx)
        mask = mask.astype(np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask
