import numpy as np

from umud.maskdump import decode_line, encode_line


def test_mask_roundtrip_keeps_thin_lines():
    mask = np.zeros((800, 1200), dtype=np.uint8)
    mask[400, 100:1100] = 1  # 1px line
    image_id, kind, back = decode_line(encode_line("IMG_00001.tif", "fasc", mask))
    assert (image_id, kind, back.shape) == ("IMG_00001.tif", "fasc", (800, 1200))
    rows = np.flatnonzero(back.any(axis=1))
    assert 396 <= rows.min() <= 400 <= rows.max() <= 404
