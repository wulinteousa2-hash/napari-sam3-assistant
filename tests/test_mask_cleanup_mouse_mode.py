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
