import numpy as np

from napari_sam3_assistant.core.models import Sam3Result, Sam3Task
from napari_sam3_assistant.services.layer_writer import LayerWriter


class FakeLabelsLayer:
    def __init__(self, data, name, **kwargs):
        self.data = data
        self.name = name
        self.scale = kwargs.get("scale")
        self.translate = kwargs.get("translate")
        self.rotate = kwargs.get("rotate")
        self.shear = kwargs.get("shear")
        self.affine = kwargs.get("affine")
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class FakeShapesLayer:
    def __init__(self, data, name, **kwargs):
        self.data = data
        self.name = name
        self.shape_type = kwargs.get("shape_type")
        self.properties = kwargs.get("properties")


class FakeImageLayer:
    def __init__(
        self,
        name,
        *,
        scale=(1.0, 1.0),
        translate=(0.0, 0.0),
        rotate=None,
        shear=None,
        affine=None,
    ):
        self.name = name
        self.scale = scale
        self.translate = translate
        self.rotate = rotate
        self.shear = shear
        self.affine = affine


class FakeLayers(dict):
    def __iter__(self):
        return iter(self.values())


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayers()

    def add_labels(self, data, name, **kwargs):
        layer = FakeLabelsLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer

    def add_shapes(self, data, name, **kwargs):
        layer = FakeShapesLayer(data, name, **kwargs)
        self.layers[name] = layer
        return layer


def test_video_frame_writer_does_not_erase_existing_mask_with_empty_update():
    viewer = FakeViewer()
    writer = LayerWriter(viewer)
    first = np.zeros((4, 5), dtype=np.uint32)
    first[1:3, 1:4] = 7

    writer.write_video_frame_result(
        Sam3Result(task=Sam3Task.SEGMENT_3D, frame_index=1, labels=first),
        (3, 4, 5),
        labels_name="SAM3 propagated preview labels",
    )
    writer.write_video_frame_result(
        Sam3Result(
            task=Sam3Task.SEGMENT_3D,
            frame_index=1,
            labels=np.zeros((4, 5), dtype=np.uint32),
        ),
        (3, 4, 5),
        labels_name="SAM3 propagated preview labels",
    )

    layer = viewer.layers["SAM3 propagated preview labels"]
    np.testing.assert_array_equal(layer.data[1], first)


def test_video_frame_writer_allows_nonempty_update_to_replace_existing_mask():
    viewer = FakeViewer()
    writer = LayerWriter(viewer)
    first = np.zeros((4, 5), dtype=np.uint32)
    first[1:3, 1:4] = 7
    second = np.zeros((4, 5), dtype=np.uint32)
    second[0:2, 0:2] = 3

    writer.write_video_frame_result(
        Sam3Result(task=Sam3Task.SEGMENT_3D, frame_index=1, labels=first),
        (3, 4, 5),
        labels_name="SAM3 propagated preview labels",
    )
    writer.write_video_frame_result(
        Sam3Result(task=Sam3Task.SEGMENT_3D, frame_index=1, labels=second),
        (3, 4, 5),
        labels_name="SAM3 propagated preview labels",
    )

    layer = viewer.layers["SAM3 propagated preview labels"]
    np.testing.assert_array_equal(layer.data[1], second)


def test_global_image_result_copies_source_image_transform():
    viewer = FakeViewer()
    viewer.layers["large"] = FakeImageLayer(
        "large",
        scale=(2.0, 3.0),
        translate=(40.0, 50.0),
    )
    writer = LayerWriter(viewer)
    labels = np.ones((4, 5), dtype=np.uint32)

    writer.write_result(
        Sam3Result(
            task=Sam3Task.EXEMPLAR,
            labels=labels,
            metadata={
                "image_layer": "large",
                "result_space": "global_image",
            },
        ),
        labels_name="SAM3 tiled exemplar labels",
    )

    layer = viewer.layers["SAM3 tiled exemplar labels"]
    assert layer.scale == (2.0, 3.0)
    assert layer.translate == (40.0, 50.0)


def test_global_image_2d_result_from_3d_source_uses_spatial_transform_only():
    viewer = FakeViewer()
    viewer.layers["stack"] = FakeImageLayer(
        "stack",
        scale=(4.0, 2.0, 3.0),
        translate=(9.0, 40.0, 50.0),
        rotate=np.eye(3),
        shear=(0.1, 0.2, 0.3),
        affine=object(),
    )
    writer = LayerWriter(viewer)
    labels = np.ones((4, 5), dtype=np.uint32)

    writer.write_result(
        Sam3Result(
            task=Sam3Task.EXEMPLAR,
            labels=labels,
            frame_index=4,
            metadata={
                "image_layer": "stack",
                "result_space": "global_image",
            },
        ),
        labels_name="SAM3 preview labels",
    )

    layer = viewer.layers["SAM3 preview labels"]
    assert layer.scale == (2.0, 3.0)
    assert layer.translate == (40.0, 50.0)
    np.testing.assert_array_equal(layer.rotate, np.eye(2))
    assert layer.shear is None
    assert layer.affine is None


def test_tiled_exemplar_result_writes_preview_boxes_layer():
    viewer = FakeViewer()
    writer = LayerWriter(viewer)
    labels = np.zeros((20, 20), dtype=np.uint32)
    labels[4:10, 5:12] = 1

    writer.write_result(
        Sam3Result(
            task=Sam3Task.EXEMPLAR,
            labels=labels,
            boxes_xyxy=np.asarray([[5, 4, 12, 10]], dtype=np.float32),
            scores=np.asarray([0.8], dtype=np.float32),
            metadata={"large_image_tiled_scan": True},
        ),
        labels_name="SAM3 tiled exemplar labels",
        boxes_name="SAM3 tiled exemplar boxes",
    )

    assert "SAM3 tiled exemplar boxes" in viewer.layers
    layer = viewer.layers["SAM3 tiled exemplar boxes"]
    assert layer.shape_type == "rectangle"
    assert len(layer.data) == 1
    np.testing.assert_allclose(
        layer.data[0],
        np.asarray([[4, 5], [4, 12], [10, 12], [10, 5]], dtype=float),
    )
    np.testing.assert_allclose(layer.properties["score"], np.asarray([0.8], dtype=np.float32))
