import numpy as np

from umud.calibration import _period, calibrate


def test_period_prefers_full_spacing_over_submultiples():
    ticks = [100.0 + 134.5 * i for i in range(7)]
    assert abs(_period(ticks, 40, 250) - 134.5) < 0.5


def test_period_tolerates_missed_and_spurious_ticks():
    ticks = [87.0, 155.0, 222.0, 289.5, 357.0, 426.0, 561.0, 629.0, 696.0, 400.0]
    assert abs(_period(ticks, 20, 100, min_ticks=4) - 67.6) < 1


def test_telemed_crop_ruler():
    # Crop with a lateral ruler: grey line on the second-to-last row and
    # 2px-wide ticks every 77.5 px (10 mm) rising 4 rows above it.
    image = np.full((513, 465, 3), 40, dtype=np.uint8)
    image[-1] = 56
    image[-2] = 168
    for x in np.arange(53.5, 465, 77.5):
        image[-6:-2, int(x) : int(x) + 2] = 168
    cal = calibrate(image)
    assert cal is not None and cal.layout == "telemed_crop"
    assert abs(cal.px_per_mm_x - 7.75) < 0.05
    assert cal.px_per_mm_x == cal.px_per_mm_y
