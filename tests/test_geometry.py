import numpy as np
from skimage.measure import label

from umud.data.dataset import resize_mask
from umud.geometry import angle_difference_deg, fascicle_angles, measure_fascicle, measure_thickness


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


def test_measure_fascicle_extrapolates_between_aponeuroses():
    # Aponeuroses at rows 40 and 160; a short 45-degree segment in between
    # extends to 120 rows of travel, i.e. 120 * sqrt(2) along the fascicle.
    shape = (200, 300)
    apo_mask = _line_mask(shape, 40, 0, 40, 299) | _line_mask(shape, 160, 0, 160, 299)
    fasc_mask = _line_mask(shape, 80, 100, 110, 130)
    length, angle = measure_fascicle(fasc_mask, apo_mask=apo_mask)
    assert abs(length - 120 * np.sqrt(2)) < 5
    assert abs(angle - 45) < 3


def test_measure_fascicle_takes_median_over_segments():
    shape = (300, 300)
    apo_mask = _line_mask(shape, 20, 0, 20, 299) | _line_mask(shape, 280, 0, 280, 299)
    # Two 20-degree segments and one 40-degree outlier: median is 20.
    fasc_mask = np.zeros(shape, dtype=np.uint8)
    for row, col, deg in [(80, 30, 20), (150, 120, 20), (220, 60, 40)]:
        dr, dc = -40 * np.sin(np.radians(deg)), 40 * np.cos(np.radians(deg))
        fasc_mask |= _line_mask(shape, row, col, int(row + dr), int(col + dc), thickness=0)
    _, angle = measure_fascicle(fasc_mask, apo_mask=apo_mask)
    assert abs(angle - 20) < 3


def test_fascicle_angles_sign_and_min_length():
    shape = (200, 200)
    rising = _line_mask(shape, 150, 20, 100, 120, thickness=0)  # rises to the right
    speck = _line_mask(shape, 10, 10, 10, 12, thickness=0)  # below the length cutoff
    angles = fascicle_angles(rising | speck)
    assert len(angles) == 1
    assert abs(angles[0] - np.degrees(np.arctan2(50, 100))) < 2


def test_angle_difference_wraps():
    assert angle_difference_deg(89, -89) == 2
    assert angle_difference_deg(10, 25) == 15


def test_resize_mask_keeps_thin_lines_connected():
    mask = _line_mask((1080, 1640), 300, 100, 500, 1500, thickness=0)
    nearest = resize_mask(mask, (512, 768))
    preserved = resize_mask(mask, (512, 768), line_px=3)
    assert label(preserved).max() == 1
    assert label(nearest, connectivity=2).max() > 1


def test_measurements_use_anisotropic_scale():
    # Stretched screenshot: 2 px/mm vertically, 1 px/mm horizontally. A
    # 45-degree line in pixels is atan(1/2) = 26.6 degrees in mm, and
    # aponeuroses 120 px apart vertically are 60 mm apart.
    shape = (300, 300)
    apo_mask = _line_mask(shape, 40, 0, 40, 299) | _line_mask(shape, 160, 0, 160, 299)
    fasc_mask = _line_mask(shape, 80, 100, 110, 130)
    scale = (2.0, 1.0)
    assert abs(measure_thickness(apo_mask, px_per_mm=scale) - 60) < 2
    length, angle = measure_fascicle(fasc_mask, apo_mask=apo_mask, px_per_mm=scale)
    assert abs(angle - np.degrees(np.arctan(0.5))) < 2
    assert abs(length - np.hypot(60, 120)) < 5


def test_measure_thickness_averages_across_width():
    # Diverging aponeuroses: 40 px apart on the left, 80 px on the right;
    # the 25/50/75% average over the shared width is ~60.
    shape = (200, 400)
    apo_mask = _line_mask(shape, 40, 0, 40, 399) | _line_mask(shape, 80, 0, 120, 399)
    assert abs(measure_thickness(apo_mask) - 60) < 3


def _band(shape, row, col0, col1, half=4):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[row - half : row + half + 1, col0:col1] = 1
    return mask


def test_thickness_uses_top_two_of_three_stacked_aponeuroses():
    # Two stacked muscles: aponeuroses at rows 40, 140, 260; the deepest is
    # the largest band. The target muscle is the top one (40 -> 140).
    shape = (300, 400)
    mask = _band(shape, 40, 20, 380) | _band(shape, 140, 20, 380) | _band(shape, 260, 0, 400, half=8)
    assert abs(measure_thickness(mask) - 100) < 2
    assert abs(measure_thickness(mask, rule="largest") - 100) > 50  # the old rule pairs 40/260 or 140/260


def test_thickness_joins_a_broken_aponeurosis():
    # Superficial aponeurosis broken into two thick pieces; the deep one is
    # long but thin, so the old rule pairs the two pieces.
    shape = (300, 400)
    mask = _band(shape, 50, 0, 190, half=6) | _band(shape, 52, 210, 400, half=6) | _band(shape, 200, 0, 400, half=2)
    assert abs(measure_thickness(mask) - 149) < 3
    assert measure_thickness(mask, rule="largest") < 10


def test_thickness_skips_a_double_superficial_line():
    # A second line 12 px under the superficial one (< 8% of the height)
    # belongs to the superficial aponeurosis, not the deep one.
    shape = (300, 400)
    mask = _band(shape, 50, 0, 400) | _band(shape, 62, 0, 400, half=2) | _band(shape, 200, 0, 400)
    assert abs(measure_thickness(mask) - 150) < 2


def test_inner_edge_thickness_excludes_band_widths():
    shape = (300, 400)
    mask = _band(shape, 50, 0, 400, half=5) | _band(shape, 200, 0, 400, half=5)
    assert abs(measure_thickness(mask) - 150) < 1
    assert abs(measure_thickness(mask, edge="inner") - 140) < 1


def test_thickness_ignores_oblique_fascicle_lines_in_apo_mask():
    # Some apo label masks also hold fascicle lines: a long oblique line
    # between the aponeuroses must not be taken as the deep aponeurosis.
    shape = (300, 400)
    mask = _band(shape, 40, 0, 400) | _band(shape, 250, 0, 400)
    mask |= _line_mask(shape, 160, 60, 90, 330)  # ~15 degrees, 2/3 of the width
    assert abs(measure_thickness(mask) - 210) < 2


def test_thickness_falls_back_to_largest_when_one_aponeurosis_is_short():
    # The superficial aponeurosis only spans 30% of the width: the topmost
    # rule's span filter drops it, so the two largest components are used.
    shape = (300, 400)
    mask = _band(shape, 50, 0, 120) | _band(shape, 200, 0, 400)
    assert abs(measure_thickness(mask) - 150) < 3
