from types import SimpleNamespace

import numpy as np

from napari_sam3_assistant.core.models import PromptPolarity, Sam3Task
from napari_sam3_assistant.services.prompt_collector import PromptCollector


class FakeLayers(dict):
    pass


class ShapeOnlyData:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __array__(self, dtype=None):
        raise AssertionError("Large lazy data should not be materialized while collecting prompts.")


def _viewer(layers, current_step=(0, 0, 0)):
    return SimpleNamespace(layers=FakeLayers(layers), dims=SimpleNamespace(current_step=current_step))


def test_collect_text_prompt_without_prompt_layers():
    image = SimpleNamespace(name="image", data=np.zeros((16, 20), dtype=np.uint8))
    viewer = _viewer({"image": image})

    bundle = PromptCollector().collect(
        viewer,
        image_layer_name="image",
        task=Sam3Task.TEXT,
        text="  myelin ring  ",
    )

    assert bundle.task == Sam3Task.TEXT
    assert bundle.text is not None
    assert bundle.text.text == "myelin ring"
    assert bundle.has_prompt()


def test_collect_uses_lazy_shape_without_materializing_image_data():
    image = SimpleNamespace(name="ome-zarr", data=ShapeOnlyData((60000, 60000, 1)))
    viewer = _viewer({"image": image})

    bundle = PromptCollector().collect(
        viewer,
        image_layer_name="image",
        task=Sam3Task.TEXT,
        text="cell",
    )

    assert bundle.image.data_shape == (60000, 60000, 1)
    assert bundle.image.channel_axis == 2
    assert bundle.image.spatial_axes == (0, 1)


def test_points_layer_polarity_properties_are_collected():
    image = SimpleNamespace(name="image", data=np.zeros((16, 20), dtype=np.uint8))
    points = SimpleNamespace(
        name="points",
        data=np.asarray([[4.0, 5.0], [8.0, 9.0]]),
        properties={"polarity": np.asarray(["positive", "negative"], dtype=object)},
    )
    viewer = _viewer({"image": image, "points": points})

    bundle = PromptCollector().collect(
        viewer,
        image_layer_name="image",
        task=Sam3Task.REFINE,
        points_layer_name="points",
    )

    assert [(p.y, p.x, p.polarity) for p in bundle.points] == [
        (4.0, 5.0, PromptPolarity.POSITIVE),
        (8.0, 9.0, PromptPolarity.NEGATIVE),
    ]


def test_exemplar_shapes_are_collected_as_boxes_and_roi_metadata():
    image_data = np.arange(10 * 12, dtype=np.uint8).reshape(10, 12)
    image = SimpleNamespace(name="image", data=image_data)
    rectangle = np.asarray(
        [
            [2.0, 3.0],
            [2.0, 7.0],
            [6.0, 7.0],
            [6.0, 3.0],
        ]
    )
    shapes = SimpleNamespace(name="shapes", data=[rectangle], shape_type=["rectangle"])
    viewer = _viewer({"image": image, "shapes": shapes})

    bundle = PromptCollector().collect(
        viewer,
        image_layer_name="image",
        task=Sam3Task.EXEMPLAR,
        shapes_layer_name="shapes",
    )

    assert len(bundle.boxes) == 1
    assert len(bundle.exemplars) == 1
    assert bundle.boxes[0].y0 == 2.0
    assert bundle.boxes[0].x0 == 3.0
    assert bundle.boxes[0].y1 == 6.0
    assert bundle.boxes[0].x1 == 7.0
    assert bundle.exemplars[0].roi.shape == (4, 4)


def test_labels_prompt_selects_current_frame_from_stack():
    image = SimpleNamespace(name="image", data=np.zeros((3, 8, 8), dtype=np.uint8))
    labels_data = np.zeros((3, 8, 8), dtype=np.uint8)
    labels_data[2, 1:3, 4:6] = 1
    labels = SimpleNamespace(name="labels", data=labels_data)
    viewer = _viewer({"image": image, "labels": labels}, current_step=(2, 0, 0))

    bundle = PromptCollector().collect(
        viewer,
        image_layer_name="image",
        task=Sam3Task.REFINE,
        labels_layer_name="labels",
    )

    assert len(bundle.masks) == 1
    assert bundle.masks[0].frame_index == 2
    assert bundle.masks[0].mask.sum() == 4
