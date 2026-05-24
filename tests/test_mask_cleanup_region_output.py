from pathlib import Path


def test_mask_cleanup_region_output_exposes_disk_backed_save_workflow():
    source = Path("src/napari_sam3_assistant/mask_operations/mask_cleanup_tab.py").read_text(encoding="utf-8")
    panel = Path("src/napari_sam3_assistant/mask_operations/panel.py").read_text(encoding="utf-8")

    assert "HugeVolumeRoiTab" not in panel
    assert "Huge Volume ROI" not in panel
    assert "Region Output" in source
    assert "OME-Zarr write-back" in source
    assert "TIFF export" in source
    assert "Save Working Region" in source
    assert "def save_working_region" in source
    assert "def _write_working_region_to_ome_zarr" in source
    assert "HugeVolumeMaskStore.open" in source
    assert "store.array[z0:z1, y0:y1, x0:x1]" in source
    assert "_pending_region_edits.pop" in source
    assert "export_service.export(sub, path_text, \"TIFF\")" in source
