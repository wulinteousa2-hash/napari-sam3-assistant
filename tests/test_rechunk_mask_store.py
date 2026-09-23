import numpy as np
import pytest

from napari_sam3_assistant.huge_volume import rechunk_ome_zarr_mask


def test_rechunk_ome_zarr_mask_preserves_nested_dataset_and_metadata(tmp_path):
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    destination_path = tmp_path / "destination.ome.zarr"
    root = zarr.open_group(str(source_path), mode="w")
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{"axes": [{"name": "y"}, {"name": "x"}], "datasets": [{"path": "s0"}]}],
    }
    labels_group = root.require_group("labels")
    labels_group.attrs["image-label"] = {"source": {"image": "../../"}}
    nested_group = labels_group.require_group("labels")
    source = nested_group.create_array(
        "s0",
        shape=(13, 17),
        chunks=(8, 9),
        dtype=np.uint32,
        fill_value=0,
    )
    expected = np.zeros((13, 17), dtype=np.uint32)
    expected[2:7, 4:12] = 9
    source[:] = expected

    progress = []
    result = rechunk_ome_zarr_mask(
        source_path,
        destination_path,
        array_path="labels/labels/s0",
        xy_chunk=4,
        progress=lambda done, total: progress.append((done, total)),
    )

    destination_root = zarr.open_group(str(destination_path), mode="r")
    destination = destination_root["labels/labels/s0"]
    assert destination.chunks == (4, 4)
    np.testing.assert_array_equal(destination[:], expected)
    assert destination_root.attrs["ome"] == root.attrs["ome"]
    assert destination_root["labels"].attrs["image-label"] == labels_group.attrs["image-label"]
    assert destination_root.attrs["sam3_rechunk"]["complete"] is True
    assert result.source_chunks == (8, 9)
    assert result.destination_chunks == (4, 4)
    assert progress[-1] == (result.chunks_processed, result.chunks_processed)


def test_rechunk_ome_zarr_mask_refuses_existing_destination(tmp_path):
    zarr = pytest.importorskip("zarr")
    source_path = tmp_path / "source.ome.zarr"
    destination_path = tmp_path / "destination.ome.zarr"
    source_root = zarr.open_group(str(source_path), mode="w")
    source_root.create_array("s0", shape=(4, 4), chunks=(2, 2), dtype=np.uint8)
    zarr.open_group(str(destination_path), mode="w")

    with pytest.raises(FileExistsError):
        rechunk_ome_zarr_mask(source_path, destination_path)
