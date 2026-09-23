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



def test_huge_volume_guard_blocks_full_or_whole_volume_scope():
    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=SimpleNamespace(shape=(500, 2500, 2500)))
    tab.scope_combo = Combo("current_slice")
    tab.work_region_combo = Combo("full")

    assert "Manual ROI" in tab._unsafe_huge_volume_scope_message(layer)

    tab.scope_combo = Combo("whole_volume")
    tab.work_region_combo = Combo("manual")
    assert "Current slice" in tab._unsafe_huge_volume_scope_message(layer)


def test_huge_volume_guard_allows_current_slice_manual_roi():
    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    class Spin:
        def __init__(self, value):
            self._value = value

        def value(self):
            return self._value

    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    layer = SimpleNamespace(data=SimpleNamespace(shape=(500, 2500, 2500)))
    tab.scope_combo = Combo("current_slice")
    tab.work_region_combo = Combo("manual")
    tab.work_y0_spin = Spin(0)
    tab.work_y1_spin = Spin(256)
    tab.work_x0_spin = Spin(640)
    tab.work_x1_spin = Spin(896)

    assert tab._unsafe_huge_volume_scope_message(layer) is None
    assert tab._selected_scope_shape(layer) == (256, 256)


def test_drawn_work_region_slices_handles_large_mask_bounds_without_materializing():
    import numpy as np

    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    roi_vertices = np.array(
        [
            [6071.215, 7336.469],
            [6071.215, -317.58984],
            [-209.04492, -317.58984],
            [-209.04492, 7336.469],
        ],
        dtype=np.float32,
    )
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.work_region_combo = Combo("drawn")
    tab.work_roi_combo = Combo("roi")
    tab.viewer = SimpleNamespace(layers={"roi": SimpleNamespace(data=[roi_vertices])})

    region = tab._work_region_slices(SimpleNamespace(shape=(16_384, 16_384)))

    assert region == (slice(0, 6073), slice(0, 7338))


def test_huge_volume_guard_allows_current_slice_drawn_roi():
    import numpy as np

    class Combo:
        def __init__(self, value):
            self.value = value

        def currentData(self):
            return self.value

    roi_vertices = np.array([[10, 20], [10, 120], [90, 120], [90, 20]], dtype=np.float32)
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.scope_combo = Combo("current_slice")
    tab.work_region_combo = Combo("drawn")
    tab.work_roi_combo = Combo("roi")
    tab.viewer = SimpleNamespace(layers={"roi": SimpleNamespace(data=[roi_vertices])})
    layer = SimpleNamespace(data=SimpleNamespace(shape=(500, 2500, 2500)))

    assert tab._unsafe_huge_volume_scope_message(layer) is None
    assert tab._selected_scope_shape(layer) == (81, 101)


def test_activate_local_cleanup_region_sets_manual_roi_from_click_percent():
    class Combo:
        def __init__(self, value, options=None):
            self.value = value
            self.options = list(options or [value])
            if value not in self.options:
                self.options.append(value)
            self.index = self.options.index(value)
            self.blocked = False

        def currentData(self):
            return self.options[self.index]

        def findData(self, value):
            try:
                return self.options.index(value)
            except ValueError:
                return -1

        def setCurrentIndex(self, index):
            self.index = int(index)

        def currentIndex(self):
            return self.index

        def count(self):
            return len(self.options)

        def blockSignals(self, blocked):
            self.blocked = bool(blocked)

    class Spin:
        def __init__(self):
            self._value = 0

        def value(self):
            return self._value

        def setValue(self, value):
            self._value = int(value)

        def blockSignals(self, _blocked):
            pass

    class Label:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.work_region_combo = Combo("full", ["full", "manual", "drawn"])
    tab.scope_combo = Combo("whole_volume", ["current_slice", "z_range", "whole_volume"])
    tab.local_region_size_combo = Combo(5, [5, 10, 20, 50])
    tab.work_y0_spin = Spin()
    tab.work_y1_spin = Spin()
    tab.work_x0_spin = Spin()
    tab.work_x1_spin = Spin()
    tab.status_label = Label()
    tab._log = lambda _message: None
    tab._on_work_region_changed = lambda: None
    layer = SimpleNamespace(name="mask", data=SimpleNamespace(shape=(16_384, 16_384)))

    assert tab._activate_local_cleanup_region(layer, (10, 20)) is True

    assert tab.work_region_combo.currentData() == "manual"
    assert tab.scope_combo.currentData() == "whole_volume"
    assert tab.work_y0_spin.value() == 0
    assert tab.work_y1_spin.value() == 819
    assert tab.work_x0_spin.value() == 0
    assert tab.work_x1_spin.value() == 819
    assert tab._last_local_cleanup_center == (10, 20)
    assert "819 x 819" in tab.status_label.text


def test_activate_local_cleanup_region_uses_current_slice_for_3d_masks():
    class Combo:
        def __init__(self, value, options):
            self.options = list(options)
            self.index = self.options.index(value)

        def currentData(self):
            return self.options[self.index]

        def findData(self, value):
            try:
                return self.options.index(value)
            except ValueError:
                return -1

        def setCurrentIndex(self, index):
            self.index = int(index)

        def blockSignals(self, _blocked):
            pass

    class Spin:
        def __init__(self):
            self._value = 0

        def value(self):
            return self._value

        def setValue(self, value):
            self._value = int(value)

        def blockSignals(self, _blocked):
            pass

    class Label:
        def setText(self, _text):
            pass

    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.work_region_combo = Combo("full", ["full", "manual", "drawn"])
    tab.scope_combo = Combo("whole_volume", ["current_slice", "z_range", "whole_volume"])
    tab.local_region_size_combo = Combo(20, [5, 10, 20, 50])
    tab.work_y0_spin = Spin()
    tab.work_y1_spin = Spin()
    tab.work_x0_spin = Spin()
    tab.work_x1_spin = Spin()
    tab.status_label = Label()
    tab._log = lambda _message: None
    tab._on_work_region_changed = lambda: None
    layer = SimpleNamespace(name="mask", data=SimpleNamespace(shape=(10, 2500, 2500)))

    assert tab._activate_local_cleanup_region(layer, (3, 1000, 1200)) is True

    assert tab.scope_combo.currentData() == "current_slice"
    assert tab.work_region_combo.currentData() == "manual"
    assert tab.work_y0_spin.value() == 750
    assert tab.work_y1_spin.value() == 1250
    assert tab.work_x0_spin.value() == 950
    assert tab.work_x1_spin.value() == 1450

def test_ome_zarr_write_target_blocks_different_store_by_default(tmp_path):
    import pytest

    from napari_sam3_assistant.huge_volume import HugeVolumeMaskStore

    pytest.importorskip("zarr")

    class CheckBox:
        def isChecked(self):
            return False

    source = HugeVolumeMaskStore.create(tmp_path / "source.ome.zarr", shape=(2, 8, 8), chunks=(1, 4, 4))
    HugeVolumeMaskStore.create(tmp_path / "other.ome.zarr", shape=(2, 8, 8), chunks=(1, 4, 4))

    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.allow_different_output_store_check = CheckBox()
    layer = SimpleNamespace(data=source.array)

    with pytest.raises(ValueError, match="differs from the loaded mask source"):
        tab._validate_ome_zarr_write_target(layer, tmp_path / "other.ome.zarr", "s0")


def test_ome_zarr_region_write_uses_loaded_array_path_and_persists(tmp_path):
    import numpy as np
    import pytest

    from napari_sam3_assistant.huge_volume import HugeVolumeMaskStore

    pytest.importorskip("zarr")

    class CheckBox:
        def isChecked(self):
            return False

    store = HugeVolumeMaskStore.create(tmp_path / "mask.ome.zarr", shape=(2, 8, 8), chunks=(1, 4, 4))
    tab = MaskCleanupTab.__new__(MaskCleanupTab)
    tab.allow_different_output_store_check = CheckBox()
    tab._pending_region_edits = {}
    layer = SimpleNamespace(data=store.array)
    edited = np.full((2, 2), 7, dtype=np.uint32)

    tab._write_working_region_to_ome_zarr(
        layer,
        edited,
        (1, slice(2, 4), slice(3, 5)),
        tmp_path / "mask.ome.zarr",
        "s0",
    )

    reopened = HugeVolumeMaskStore.open(tmp_path / "mask.ome.zarr", array_path="s0")
    assert np.array_equal(reopened.array[1, 2:4, 3:5], edited)
