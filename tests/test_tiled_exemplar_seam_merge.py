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


def test_result_boxes_for_tile_clips_to_real_tile_area():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    from napari_sam3_assistant.core.models import Sam3Result, Sam3Task

    result = Sam3Result(
        task=Sam3Task.EXEMPLAR,
        boxes_xyxy=np.asarray(
            [
                [2, 2, 10, 10],      # exemplar-side box, outside tile
                [18, 3, 28, 13],     # tile-side box
                [12, 1, 20, 6],      # crosses into tile, clipped
            ],
            dtype=np.float32,
        ),
        scores=np.asarray([0.1, 0.9, 0.5], dtype=np.float32),
    )

    boxes, scores = panel._result_boxes_for_tile(result, tile_origin=(0, 16), tile_hw=(20, 20))

    np.testing.assert_allclose(
        boxes,
        np.asarray(
            [
                [2, 3, 12, 13],
                [0, 1, 4, 6],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(scores, np.asarray([0.9, 0.5], dtype=np.float32))


def test_dedupe_tiled_boxes_removes_high_iou_duplicates():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    boxes = np.asarray(
        [
            [10, 10, 30, 30],
            [11, 11, 31, 31],
            [80, 80, 100, 100],
        ],
        dtype=np.float32,
    )

    deduped = panel._dedupe_tiled_boxes(boxes, iou_threshold=0.8)

    assert deduped.shape == (2, 4)
    assert any(np.allclose(row, [80, 80, 100, 100]) for row in deduped)
