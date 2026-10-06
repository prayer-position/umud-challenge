import numpy as np

from umud.geometry import measure_fascicle, measure_thickness


def _line_mask(shape, row0, col0, row1, col1, thickness=1):
    mask = np.zeros(shape, dtype=np.uint8)
    n = max(abs(row1 - row0), abs(col1 - col0)) * 4
    rows = np.linspace(row0, row1, n).astype(int)
    cols = np.linspace(col0, col1, n).astype(int)
    for r, c in zip(rows, cols):
        mask[max(r - thickness, 0) : r + thickness + 1, max(c - thickness, 0) : c + thickness + 1] = 1
    return mask


def test_measure_thickness_parallel_horizontal_lines():
    shape = (200, 200)
    mask = _line_mask(shape, 50, 10, 50, 190) | _line_mask(shape, 100, 10, 100, 190)
    thickness = measure_thickness(mask)
    assert abs(thickness - 50) < 3


def test_measure_thickness_too_few_components_returns_nan():
    shape = (200, 200)
    mask = _line_mask(shape, 50, 10, 50, 190)
    assert np.isnan(measure_thickness(mask))


def test_measure_fascicle_length():
    shape = (200, 200)
    fasc_mask = _line_mask(shape, 20, 20, 120, 100)
    length, _ = measure_fascicle(fasc_mask)
    expected_length = np.hypot(120 - 20, 100 - 20)
    assert abs(length - expected_length) < 5


def test_measure_fascicle_angle_against_aponeurosis():
    shape = (200, 200)
    apo_mask = _line_mask(shape, 150, 0, 150, 199)
    fasc_mask = _line_mask(shape, 50, 50, 150, 150)
    _, angle = measure_fascicle(fasc_mask, apo_mask=apo_mask)
    assert abs(angle - 45) < 5
