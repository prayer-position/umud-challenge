"""Pass predicted masks through a text log.

Notebook output files can't always be downloaded, but notebook logs can, so
predict.py --dump-log prints each mask as one tagged line: downsampled with
area resizing (any coverage counts, so thin lines survive), bit-packed,
zlib-compressed and base64-encoded. decode_line() reverses it.
"""

from __future__ import annotations

import base64
import zlib

import cv2
import numpy as np

TAG = "MASKDUMP"


def encode_line(image_id: str, kind: str, mask: np.ndarray, factor: int = 4) -> str:
    h, w = mask.shape
    small = cv2.resize(mask.astype(np.float32), (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA) > 0
    payload = base64.b64encode(zlib.compress(np.packbits(small).tobytes(), 9)).decode()
    return f"{TAG} {image_id} {kind} {h} {w} {small.shape[0]} {small.shape[1]} {payload}"


def decode_line(line: str) -> tuple[str, str, np.ndarray]:
    """(image_id, kind, mask resized back to the original h x w)."""
    _, image_id, kind, h, w, sh, sw, payload = line.split()
    bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(payload)), dtype=np.uint8))
    small = bits[: int(sh) * int(sw)].reshape(int(sh), int(sw)).astype(np.uint8)
    return image_id, kind, cv2.resize(small, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
