from types import SimpleNamespace

from napari_sam3_assistant.mask_operations.mask_cleanup_tab import MaskCleanupTab
from napari_sam3_assistant.mask_operations.models import ComponentRecord


def test_current_slice_component_records_display_global_z_y_x():
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=SimpleNamespace(shape=(5, 100, 100)))
    record = ComponentRecord(
        component_id=1,
        label_value=7,
        area=12,
        centroid_y=2.0,
        centroid_x=4.0,
        bbox=((1, 5), (3, 8)),
        ndim=2,
    )

    display = tab._component_records_for_display(
        layer,
        (3, slice(10, 20), slice(30, 50)),
        [record],
    )[0]

    assert display.centroid_z == 3.0
    assert display.centroid_y == 12.0
    assert display.centroid_x == 34.0
    assert display.z_min == 3
    assert display.z_max == 3
    assert display.bbox == ((3, 4), (11, 15), (33, 38))
    assert display.ndim == 3


def test_z_range_component_records_display_global_z_y_x():
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=SimpleNamespace(shape=(5, 100, 100)))
    record = ComponentRecord(
        component_id=1,
        label_value=7,
        area=12,
        centroid_z=1.0,
        centroid_y=2.0,
        centroid_x=4.0,
        z_min=0,
        z_max=1,
        bbox=((0, 2), (1, 5), (3, 8)),
        ndim=3,
    )

    display = tab._component_records_for_display(
        layer,
        (slice(2, 5), slice(10, 20), slice(30, 50)),
        [record],
    )[0]

    assert display.centroid_z == 3.0
    assert display.centroid_y == 12.0
    assert display.centroid_x == 34.0
    assert display.z_min == 2
    assert display.z_max == 3
    assert display.bbox == ((2, 4), (11, 15), (33, 38))


def test_current_slice_scoped_data_slices_lazy_source_before_numpy_conversion():
    class LazyArray:
        shape = (5, 100, 100)
        ndim = 3
        dtype = "uint16"

        def __init__(self):
            self.requested = None

        def __array__(self, dtype=None):
            raise AssertionError("full lazy mask should not be materialized before scoping")

        def __getitem__(self, indexer):
            self.requested = indexer
            import numpy as np

            return np.ones((100, 100), dtype=np.uint16)

    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    lazy = LazyArray()
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.scope_combo = Combo("current_slice")
    tab.work_region_combo = Combo("full")
    tab.viewer = SimpleNamespace(dims=SimpleNamespace(current_step=(2, 0, 0)))
    layer = SimpleNamespace(data=lazy)

    sub, indexer, offset = tab._scoped_data(layer)

    assert lazy.requested == (2, slice(None), slice(None))
    assert sub.shape == (100, 100)
    assert indexer == (2, slice(None), slice(None))
    assert offset == (0, 0)
