import numpy as np

from napari_sam3_assistant.core.coordinates import RoiBounds
from napari_sam3_assistant.widgets.advanced.advanced_mode_panel import AdvancedModePanel


def test_tile_seam_merge_reconnects_vertical_split_object():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[2:8, 2:5] = 1
    labels[2:8, 5:8] = 2
    labels[0:2, 0:2] = 3
    tiles = [
        RoiBounds(y0=0, x0=0, y1=10, x1=5),
        RoiBounds(y0=0, x0=5, y1=10, x1=10),
    ]

    merged, merge_count = panel._merge_tile_seam_labels(
        labels,
        tiles,
        dilation_px=1,
        min_contact_pixels=3,
    )

    assert merge_count == 1
    assert merged[3, 3] == merged[3, 6]
    assert merged[0, 0] != merged[3, 3]
    assert sorted(np.unique(merged).tolist()) == [0, 1, 2]


def test_tile_seam_merge_respects_minimum_contact_pixels():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[4:6, 3:5] = 1
    labels[4:6, 5:7] = 2
    tiles = [
        RoiBounds(y0=0, x0=0, y1=10, x1=5),
        RoiBounds(y0=0, x0=5, y1=10, x1=10),
    ]

    merged, merge_count = panel._merge_tile_seam_labels(
        labels,
        tiles,
        dilation_px=1,
        min_contact_pixels=20,
    )

    assert merge_count == 0
    np.testing.assert_array_equal(merged, labels)


def test_tile_seam_merge_reconnects_horizontal_split_object():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    labels = np.zeros((10, 10), dtype=np.uint32)
    labels[2:5, 2:8] = 1
    labels[5:8, 2:8] = 2
    tiles = [
        RoiBounds(y0=0, x0=0, y1=5, x1=10),
        RoiBounds(y0=5, x0=0, y1=10, x1=10),
    ]

    merged, merge_count = panel._merge_tile_seam_labels(
        labels,
        tiles,
        dilation_px=1,
        min_contact_pixels=3,
    )

    assert merge_count == 1
    assert merged[3, 3] == merged[6, 3]
