import numpy as np
import pytest
from packing_assets.meshy_views import choose_views, object_crop


def test_selection_spreads_views_in_camera_angle_not_time():
    rows = [
        {"image": str(i), "azimuth_rad": np.deg2rad(i), "score": 1 if i else 2}
        for i in range(0, 360, 10)
    ]
    assert [r["image"] for r in choose_views(rows)] == ["0", "90", "180", "270"]
    with pytest.raises(ValueError, match="coverage"):
        choose_views(rows[:5])


def test_crop_removes_background_preserves_holes_and_normalizes_occupancy():
    rgb = np.zeros((100, 150, 3), np.uint8)
    rgb[:] = [10, 80, 150]
    mask = np.zeros((100, 150), bool)
    mask[20:70, 30:90] = True
    mask[40:50, 50:60] = False
    result = np.asarray(object_crop(rgb, mask, size=300))
    selected = np.any(result < 200, axis=2)
    y, x = np.nonzero(selected)
    assert 250 <= np.ptp(x) <= 260
    assert (result[0] == 255).all()
    assert (result[145, 125] == 255).all()  # retained interior hole
    mask[0, 40] = True
    with pytest.raises(ValueError, match="border"):
        object_crop(rgb, mask)
