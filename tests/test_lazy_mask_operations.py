from types import SimpleNamespace

import numpy as np

from napari_sam3_assistant.mask_operations.mask_cleanup_tab import MaskCleanupTab


class LazyArray:
    __module__ = "zarr.testing"
    def __init__(self, shape, value=7):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(np.uint32)
        self.value = int(value)
        self.requested = []

    def __array__(self, dtype=None):
        raise AssertionError("the full lazy array must not be materialized")

    def __getitem__(self, indexer):
        self.requested.append(indexer)
        if isinstance(indexer, tuple) and all(isinstance(item, int) for item in indexer):
            return np.asarray(self.value, dtype=self.dtype)
        requested_shape = []
        for size, selector in zip(self.shape, indexer, strict=True):
            if isinstance(selector, int):
                continue
            start, stop, step = selector.indices(size)
            requested_shape.append(len(range(start, stop, step)))
        return np.full(tuple(requested_shape), self.value, dtype=self.dtype)


def test_small_lazy_snapshot_is_not_materialized():
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=LazyArray((100, 100)))

    assert tab._snapshot_layer_data(layer) is None
    assert layer.data.requested == []


def test_large_lazy_snapshot_checks_shape_without_materializing():
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=LazyArray((84_175, 79_966)))

    assert tab._snapshot_layer_data(layer) is None
    assert layer.data.requested == []


def test_label_value_lookup_reads_only_one_lazy_element():
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=LazyArray((84_175, 79_966), value=11))

    value = tab._label_value_near(layer, np.asarray((123.0, 456.0)))

    assert value == 11
    assert layer.data.requested == [(123, 456)]


def test_scoped_source_image_is_sliced_before_numpy_conversion():
    class Combo:
        def currentData(self):
            return "image"

    image = LazyArray((84_175, 79_966), value=3)
    labels = LazyArray((84_175, 79_966), value=1)
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.source_image_combo = Combo()
    tab.viewer = SimpleNamespace(
        layers={"image": SimpleNamespace(data=image)}
    )

    region = (slice(100, 200), slice(300, 500))
    result = tab._source_image_for_scoped_labels(
        SimpleNamespace(data=labels),
        region,
    )

    assert result.shape == (100, 200)
    assert image.requested == [region]
    assert labels.requested == []
