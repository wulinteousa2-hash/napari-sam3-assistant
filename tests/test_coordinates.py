import numpy as np

from napari_sam3_assistant.core.coordinates import (
    CoordinateMapper,
    RoiBounds,
    centered_roi_bounds,
    extract_2d_image,
    extract_2d_roi,
    globalize_result_arrays,
    infer_image_selection,
    localize_bundle_to_roi,
    roi_anchor_from_bundle,
    to_rgb_uint8,
)
from napari_sam3_assistant.core.models import BoxPrompt, MaskPrompt, PointPrompt, PromptBundle, Sam3Task


def test_rgb_image_is_not_treated_as_stack():
    selection = infer_image_selection("rgb", (64, 96, 3), dims_current_step=(0, 0, 0))

    assert selection.frame_axis is None
    assert selection.channel_axis == 2
    assert selection.spatial_axes == (0, 1)
    assert selection.frame_index is None


def test_trailing_singleton_channel_is_not_treated_as_spatial_axis():
    selection = infer_image_selection("ome-zarr", (60000, 60000, 1))

    assert selection.frame_axis is None
    assert selection.channel_axis == 2
    assert selection.spatial_axes == (0, 1)


def test_stack_rgb_selection_uses_leading_frame_axis():
    selection = infer_image_selection(
        "rgb-stack",
        (5, 64, 96, 3),
        dims_current_step=(3, 0, 0, 0),
    )

    assert selection.frame_axis == 0
    assert selection.frame_index == 3
    assert selection.channel_axis == 3
    assert selection.spatial_axes == (1, 2)


def test_extract_2d_image_preserves_trailing_rgb_channels():
    data = np.zeros((4, 8, 9, 3), dtype=np.uint8)
    data[2, :, :, 1] = 255
    selection = infer_image_selection(
        "rgb-stack",
        data.shape,
        dims_current_step=(2, 0, 0, 0),
    )

    extracted = extract_2d_image(data, selection)

    assert extracted.shape == (8, 9, 3)
    assert extracted[..., 1].max() == 255


def test_extract_2d_roi_preserves_trailing_singleton_channel():
    data = np.arange(10 * 12, dtype=np.uint16).reshape(10, 12, 1)
    selection = infer_image_selection("ome-zarr", data.shape)
    bounds = RoiBounds(y0=2, x0=3, y1=6, x1=8)

    roi = extract_2d_roi(data, selection, bounds)

    assert roi.shape == (4, 5, 1)
    np.testing.assert_array_equal(roi[..., 0], data[2:6, 3:8, 0])


def test_box_coordinate_formats_match_sam3_image_and_video_expectations():
    selection = infer_image_selection("image", (100, 200))
    mapper = CoordinateMapper(selection)
    box = BoxPrompt(y0=10, x0=20, y1=30, x1=60)

    assert mapper.box_to_normalized_cxcywh(box, (100, 200)) == (0.2, 0.2, 0.2, 0.2)
    assert mapper.box_to_normalized_xywh(box, (100, 200)) == (0.1, 0.1, 0.2, 0.2)


def test_to_rgb_uint8_converts_float_grayscale():
    image = np.asarray([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)

    rgb = to_rgb_uint8(image)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] == 0
    assert rgb[-1, -1, 0] == 255


def test_to_rgb_uint8_converts_singleton_channel_grayscale():
    image = np.asarray([[[0], [10]], [[20], [30]]], dtype=np.uint16)

    rgb = to_rgb_uint8(image)

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])


def test_extract_2d_roi_slices_only_requested_xy_region():
    data = np.arange(2 * 10 * 12, dtype=np.uint16).reshape(2, 10, 12)
    selection = infer_image_selection("stack", data.shape, dims_current_step=(1, 0, 0))
    bounds = RoiBounds(y0=2, x0=3, y1=6, x1=8)

    roi = extract_2d_roi(data, selection, bounds)

    assert roi.shape == (4, 5)
    np.testing.assert_array_equal(roi, data[1, 2:6, 3:8])


def test_extract_2d_roi_uses_first_level_for_multiscale_data():
    level0 = np.arange(10 * 12, dtype=np.uint16).reshape(10, 12)
    multiscale = [level0, level0[::2, ::2]]
    selection = infer_image_selection("multiscale", level0.shape)
    bounds = RoiBounds(y0=2, x0=3, y1=6, x1=8)

    roi = extract_2d_roi(multiscale, selection, bounds)

    np.testing.assert_array_equal(roi, level0[2:6, 3:8])


def test_localize_bundle_to_roi_converts_global_prompts_to_local_coordinates():
    selection = infer_image_selection("large", (100, 120))
    bundle = PromptBundle(
        task=Sam3Task.REFINE,
        image=selection,
        points=[PointPrompt(y=45, x=55)],
        boxes=[BoxPrompt(y0=40, x0=50, y1=70, x1=90)],
    )
    bounds = RoiBounds(y0=32, x0=40, y1=96, x1=104)

    local = localize_bundle_to_roi(bundle, bounds, (64, 64, 3))

    assert local.image.data_shape == (64, 64, 3)
    assert local.points[0].y == 13
    assert local.points[0].x == 15
    assert local.boxes[0].y0 == 8
    assert local.boxes[0].x0 == 10


def test_globalize_result_arrays_writes_roi_labels_to_global_canvas():
    local_labels = np.zeros((4, 5), dtype=np.uint32)
    local_labels[1:3, 2:4] = 7
    bounds = RoiBounds(y0=2, x0=3, y1=6, x1=8)

    labels, masks, boxes = globalize_result_arrays(
        labels=local_labels,
        masks=None,
        boxes_xyxy=np.asarray([[1, 1, 4, 3]], dtype=np.float32),
        bounds=bounds,
        image_hw=(10, 12),
    )

    assert masks is None
    assert labels.shape == (10, 12)
    assert labels[3, 5] == 7
    assert labels[0, 0] == 0
    np.testing.assert_allclose(boxes, np.asarray([[4, 3, 7, 5]], dtype=np.float32))


def test_centered_roi_bounds_clamps_to_image_edges():
    bounds = centered_roi_bounds(5, 5, image_hw=(100, 120), roi_hw=(64, 64))

    assert bounds == RoiBounds(y0=0, x0=0, y1=64, x1=64)


def test_roi_anchor_from_bundle_uses_labels_mask_bbox_center():
    selection = infer_image_selection("large", (100, 120))
    mask = np.zeros((100, 120), dtype=bool)
    mask[20:31, 40:61] = True
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_2D,
        image=selection,
        masks=[MaskPrompt(mask=mask)],
    )

    assert roi_anchor_from_bundle(bundle) == (25.0, 50.0)


def test_localize_bundle_to_roi_crops_labels_mask_to_local_roi():
    selection = infer_image_selection("large", (100, 120))
    mask = np.zeros((100, 120), dtype=bool)
    mask[20:31, 40:61] = True
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_2D,
        image=selection,
        masks=[MaskPrompt(mask=mask)],
    )
    bounds = RoiBounds(y0=16, x0=32, y1=48, x1=64)

    local = localize_bundle_to_roi(bundle, bounds, (32, 32, 3))

    assert len(local.masks) == 1
    assert local.masks[0].mask.shape == (32, 32)
    assert local.masks[0].mask[4:15, 8:29].all()
