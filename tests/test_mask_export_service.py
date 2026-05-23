import numpy as np
import pytest
import tifffile

from napari_sam3_assistant.mask_operations.export_service import MaskExportService


def test_tiff_export_preserves_uint32_label_values(tmp_path):
    data = np.asarray([[0, 1], [70_000, 4_000_000_000]], dtype=np.uint64)
    path = MaskExportService().export(data, tmp_path / "labels.tif", "TIFF")

    saved = tifffile.imread(path)
    assert saved.dtype == np.uint32
    np.testing.assert_array_equal(saved, data.astype(np.uint32))


def test_png_export_rejects_labels_above_uint16(tmp_path):
    data = np.asarray([[0, 70_000]], dtype=np.uint32)

    with pytest.raises(ValueError, match="Use TIFF or NumPy"):
        MaskExportService().export(data, tmp_path / "labels.png", "PNG")


def test_ome_zarr_export_writes_multiscale_mask_with_scale(tmp_path):
    zarr = pytest.importorskip("zarr")
    data = np.zeros((128, 160), dtype=np.uint16)
    data[10:20, 30:50] = 7

    path = MaskExportService().export(
        data,
        tmp_path / "labels.ome.zarr",
        "OME-Zarr",
        scale=(6.0, 6.0),
    )

    assert path.name == "labels.ome.zarr"
    root = zarr.open_group(path, mode="r")
    assert root["s0"].shape == data.shape
    np.testing.assert_array_equal(root["s0"][:], data)
    multiscales = root.attrs["ome"]["multiscales"]
    assert multiscales[0]["axes"][0]["name"] == "y"
    assert multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [6.0, 6.0]
