from pathlib import Path


MASK_CLEANUP_SOURCE = Path("src/napari_sam3_assistant/mask_operations/mask_cleanup_tab.py")


def test_context_menu_mouse_action_neutralizes_labels_paint_mode():
    source = MASK_CLEANUP_SOURCE.read_text(encoding="utf-8")
    right_click_block = source.split("if self._is_right_mouse_event(event):", 1)[1].split(
        "mode = self._cleanup_subtab()",
        1,
    )[0]

    assert right_click_block.count("self._neutralize_labels_mouse_tool(layer, event)") == 2
    assert "finally:" in right_click_block
    assert "layer.mode = \"pick\"" in source
    assert "event.handled = True" in source
    assert "native.accept()" in source


def test_axon_context_menu_inserts_qaction_not_text():
    source = MASK_CLEANUP_SOURCE.read_text(encoding="utf-8")
    axon_block = source.split('if mode == "axon":', 1)[1].split(
        "def handle_local_action",
        1,
    )[0]

    assert "QAction(" in axon_block
    assert "menu.insertAction(" in axon_block
    assert 'f"Cut axon from clicked point to {target_text}"' in axon_block
    assert "cut_axon_action.setToolTip" in axon_block
    assert "cut_axon_action = menu.insertAction" not in axon_block
    assert "assign_local_action,\n                cut_axon_action," in axon_block


def test_components_working_region_uses_roi_scoped_writeback():
    source = MASK_CLEANUP_SOURCE.read_text(encoding="utf-8")

    assert "Working Region" in source
    assert 'self.work_region_combo.addItem("Manual ROI", "manual")' in source
    assert 'self.work_region_combo.addItem("Drawn ROI", "drawn")' in source
    assert "def _work_region_slices" in source
    assert "def _replace_layer_region_data" in source
    assert "source[indexer] = updated_region" in source
    assert "_pending_region_edits" in source
    assert "current.copy()" not in source.split("def _replace_scoped_layer_data", 1)[1].split(
        "def _current_z",
        1,
    )[0]
    assert 'history.append(("region", indexer, np.asarray(region).copy()))' in source
