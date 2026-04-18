import numpy as np

from napari_sam3_assistant.core.coordinates import (
    CoordinateMapper,
    extract_2d_image,
    infer_image_selection,
    to_rgb_uint8,
)
from napari_sam3_assistant.core.models import BoxPrompt


def test_rgb_image_is_not_treated_as_stack():
    selection = infer_image_selection("rgb", (64, 96, 3), dims_current_step=(0, 0, 0))

    assert selection.frame_axis is None
    assert selection.channel_axis == 2
    assert selection.spatial_axes == (0, 1)
    assert selection.frame_index is None


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
