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
