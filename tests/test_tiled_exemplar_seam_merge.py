from types import SimpleNamespace

import numpy as np

from napari_sam3_assistant.core.coordinates import RoiBounds, infer_image_selection
from napari_sam3_assistant.core.models import (
    BoxPrompt,
    ExemplarPrompt,
    PromptBundle,
    Sam3Result,
    Sam3Task,
)
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


def test_compose_tile_labels_preserves_touching_sam3_instances():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    composed = np.zeros((4, 6), dtype=np.uint32)
    local_labels = np.zeros((4, 6), dtype=np.uint32)
    local_labels[1:3, 1:3] = 7
    local_labels[1:3, 3:5] = 11

    next_id = panel._compose_tile_labels(
        composed,
        local_labels,
        RoiBounds(y0=0, x0=0, y1=4, x1=6),
        20,
    )

    assert next_id == 22
    assert composed[1, 2] == 20
    assert composed[1, 3] == 21
    assert composed[1, 2] != composed[1, 3]


def test_tile_containing_target_exemplar_uses_natural_pixels_and_local_box():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    panel._cache_context_for_layer = lambda *_args, **_kwargs: {}
    image_data = np.arange(20 * 24, dtype=np.uint16).reshape(20, 24)
    image_layer = SimpleNamespace(name="image", data=image_data)
    selection = infer_image_selection("image", image_data.shape)
    exemplar = ExemplarPrompt(
        roi=image_data[8:11, 9:13],
        y0=8.0,
        x0=9.0,
        y1=11.0,
        x1=13.0,
    )
    bundle = PromptBundle(
        task=Sam3Task.EXEMPLAR,
        image=selection,
        boxes=[BoxPrompt(y0=8.0, x0=9.0, y1=11.0, x1=13.0)],
        exemplars=[exemplar],
    )
    bounds = RoiBounds(y0=5, x0=6, y1=15, x1=18)

    class RecordingAdapter:
        def run_image(self, image, prompt_bundle, *, cache_context=None):
            self.image = np.asarray(image).copy()
            self.bundle = prompt_bundle
            return Sam3Result(
                task=Sam3Task.EXEMPLAR,
                labels=np.zeros(np.asarray(image).shape[:2], dtype=np.uint32),
            )

    adapter = RecordingAdapter()
    panel._infer_tiled_exemplar_tile(
        adapter,
        image_layer,
        bundle,
        bounds,
        np.asarray(exemplar.roi),
        target_exemplar=exemplar,
    )

    np.testing.assert_array_equal(adapter.image, image_data[5:15, 6:18])
    assert adapter.bundle.boxes == [
        BoxPrompt(y0=3.0, x0=3.0, y1=6.0, x1=7.0)
    ]


def test_tile_outside_target_exemplar_keeps_context_and_small_box():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    panel._cache_context_for_layer = lambda *_args, **_kwargs: {}
    image_data = np.arange(300 * 300, dtype=np.uint32).reshape(300, 300)
    image_layer = SimpleNamespace(name="image", data=image_data)
    selection = infer_image_selection("image", image_data.shape)
    exemplar = ExemplarPrompt(
        roi=image_data[20:40, 30:50],
        y0=20.0,
        x0=30.0,
        y1=40.0,
        x1=50.0,
    )
    bundle = PromptBundle(
        task=Sam3Task.EXEMPLAR,
        image=selection,
        boxes=[BoxPrompt(y0=20.0, x0=30.0, y1=40.0, x1=50.0)],
        exemplars=[exemplar],
    )
    bounds = RoiBounds(y0=172, x0=172, y1=300, x1=300)

    class RecordingAdapter:
        def run_image(self, image, prompt_bundle, *, cache_context=None):
            self.image = np.asarray(image).copy()
            self.bundle = prompt_bundle
            return Sam3Result(
                task=Sam3Task.EXEMPLAR,
                labels=np.zeros(np.asarray(image).shape[:2], dtype=np.uint32),
            )

    adapter = RecordingAdapter()
    panel._infer_tiled_exemplar_tile(
        adapter,
        image_layer,
        bundle,
        bounds,
        np.asarray(exemplar.roi),
        target_exemplar=exemplar,
    )

    assert adapter.image.shape == (128, 264)
    prompt_box = adapter.bundle.boxes[0]
    assert prompt_box == BoxPrompt(y0=20.0, x0=30.0, y1=40.0, x1=50.0)
    assert prompt_box.y1 < 128
    assert prompt_box.x1 < 128


def test_current_roi_and_full_scan_tile_use_identical_adapter_inputs():
    panel = AdvancedModePanel.__new__(AdvancedModePanel)
    panel._cache_context_for_layer = (
        lambda _layer, _bundle, roi_bounds=None: {
            "roi": (
                roi_bounds.y0,
                roi_bounds.x0,
                roi_bounds.y1,
                roi_bounds.x1,
            )
        }
    )
    image_data = np.arange(12 * 14, dtype=np.uint8).reshape(12, 14)
    image_layer = SimpleNamespace(name="image", data=image_data)
    selection = infer_image_selection("image", image_data.shape)
    bundle = PromptBundle(
        task=Sam3Task.EXEMPLAR,
        image=selection,
        boxes=[BoxPrompt(y0=6.0, x0=7.0, y1=8.0, x1=9.0)],
        exemplars=[
            ExemplarPrompt(
                roi=image_data[6:8, 7:9],
                y0=6.0,
                x0=7.0,
                y1=8.0,
                x1=9.0,
            )
        ],
    )
    exemplar = image_data[6:8, 7:9]
    target_exemplar = bundle.exemplars[0]
    tiles = panel._tile_bounds_for_image((12, 14), (6, 6), 0.25)
    bounds, tile_index, tile_count = panel._preview_tile_for_anchor(
        (12, 14), (6, 6), 0.25, (7.0, 8.0)
    )

    assert bounds == tiles[tile_index - 1]
    assert tile_count == len(tiles)

    class RecordingAdapter:
        def __init__(self):
            self.calls = []

        def run_image(self, image, prompt_bundle, *, cache_context=None):
            self.calls.append(
                (
                    np.asarray(image).copy(),
                    prompt_bundle,
                    dict(cache_context or {}),
                )
            )
            labels = np.ones(np.asarray(image).shape[:2], dtype=np.uint32)
            return Sam3Result(task=Sam3Task.EXEMPLAR, labels=labels)

    adapter = RecordingAdapter()
    preview = panel._infer_tiled_exemplar_tile(
        adapter,
        image_layer,
        bundle,
        bounds,
        exemplar,
        target_exemplar=target_exemplar,
    )
    scan = panel._infer_tiled_exemplar_tile(
        adapter,
        image_layer,
        bundle,
        tiles[tile_index - 1],
        exemplar,
        target_exemplar=target_exemplar,
    )

    preview_image, preview_bundle, preview_context = adapter.calls[0]
    scan_image, scan_bundle, scan_context = adapter.calls[1]
    np.testing.assert_array_equal(preview_image, scan_image)
    np.testing.assert_array_equal(
        preview_image,
        image_data[bounds.y0 : bounds.y1, bounds.x0 : bounds.x1],
    )
    assert preview_bundle.boxes == scan_bundle.boxes
    assert preview_bundle.image == scan_bundle.image
    assert preview_context == scan_context
    np.testing.assert_array_equal(preview[1], scan[1])
