import numpy as np
import pytest

from napari_sam3_assistant.huge_volume import HugeVolumeMaskStore


def test_huge_volume_mask_store_writes_tile_labels_without_overwriting_overlap(tmp_path):
    zarr = pytest.importorskip("zarr")
    store = HugeVolumeMaskStore.create(
        tmp_path / "mask.ome.zarr",
        shape=(2, 8, 9),
        chunks=(1, 4, 4),
        scale=(2.0, 3.0, 4.0),
    )
    first = np.zeros((4, 5), dtype=np.uint32)
    first[1:3, 1:4] = 1
    next_id, written = store.write_tile_labels(0, 2, 3, first, next_object_id=1)

    assert next_id == 2
    assert written == 6

    second = np.zeros((4, 5), dtype=np.uint32)
    second[0:3, 0:3] = 1
    next_id, written = store.write_tile_labels(0, 3, 4, second, next_object_id=next_id)

    assert next_id == 3
    assert written == 3

    root = zarr.open_group(tmp_path / "mask.ome.zarr", mode="r")
    saved = root["s0"][:]
    assert saved.shape == (2, 8, 9)
    assert int(np.count_nonzero(saved[0] == 1)) == 6
    assert int(np.count_nonzero(saved[0] == 2)) == 3
    assert saved[1].max() == 0
    scale = root.attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [2.0, 3.0, 4.0]
