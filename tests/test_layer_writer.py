import numpy as np

from napari_sam3_assistant.core.models import Sam3Result, Sam3Task
from napari_sam3_assistant.services.layer_writer import LayerWriter


class FakeLabelsLayer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class FakeLayers(dict):
    def __iter__(self):
        return iter(self.values())


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayers()

    def add_labels(self, data, name, **_kwargs):
        layer = FakeLabelsLayer(data, name)
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
