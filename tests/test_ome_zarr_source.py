import numpy as np
import pytest

from napari_sam3_assistant.huge_volume import (
    HugeVolumeMaskStore,
    inspect_ome_zarr_array,
    inspect_ome_zarr_path,
)


def test_inspect_ome_zarr_array_reports_store_array_shape_chunks_axes(tmp_path):
    pytest.importorskip("zarr")
    store = HugeVolumeMaskStore.create(
        tmp_path / "mask.ome.zarr",
        shape=(3, 20, 30),
        chunks=(1, 8, 8),
        dtype=np.uint32,
        scale=(1.0, 2.0, 3.0),
    )

    source = inspect_ome_zarr_array(store.array)

    assert source is not None
    assert source.store_path == tmp_path / "mask.ome.zarr"
    assert source.array_path == "s0"
    assert source.shape == (3, 20, 30)
    assert source.chunks == (1, 8, 8)
    assert source.dtype == "uint32"
    assert source.axes == ("z", "y", "x")


def test_inspect_ome_zarr_path_reports_selected_array(tmp_path):
    pytest.importorskip("zarr")
    HugeVolumeMaskStore.create(
        tmp_path / "mask.ome.zarr",
        shape=(2, 10, 12),
        chunks=(1, 5, 6),
        dtype=np.uint16,
    )

    source = inspect_ome_zarr_path(tmp_path / "mask.ome.zarr", "/s0")

    assert source.store_path == tmp_path / "mask.ome.zarr"
    assert source.array_path == "s0"
    assert source.shape == (2, 10, 12)
    assert source.chunks == (1, 5, 6)
    assert source.axes == ("z", "y", "x")
