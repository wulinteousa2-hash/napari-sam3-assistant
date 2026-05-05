from pathlib import Path

from napari_sam3_assistant.widgets.main_widget import _is_cuda_kernel_image_error


MAIN_WIDGET_SOURCE = Path("src/napari_sam3_assistant/widgets/main_widget.py")
ADVANCED_WIDGET_SOURCE = Path("src/napari_sam3_assistant/widgets/advanced/advanced_mode_panel.py")
SIMPLE_RUN_SOURCE = Path("src/napari_sam3_assistant/widgets/simple/simple_run_panel.py")
NAPARI_MANIFEST = Path("src/napari_sam3_assistant/napari.yaml")


def test_widget_uses_clear_model_and_prompt_action_labels():
    main_source = MAIN_WIDGET_SOURCE.read_text(encoding="utf-8")
    advanced_source = ADVANCED_WIDGET_SOURCE.read_text(encoding="utf-8")
    simple_run_source = SIMPLE_RUN_SOURCE.read_text(encoding="utf-8")
    manifest_source = NAPARI_MANIFEST.read_text(encoding="utf-8")

    assert 'QRadioButton("Simple")' in Path("src/napari_sam3_assistant/widgets/mode_switch_bar.py").read_text(encoding="utf-8")
    assert 'QRadioButton("Advanced")' in Path("src/napari_sam3_assistant/widgets/mode_switch_bar.py").read_text(encoding="utf-8")
    assert "Step 1. Model Actions" in advanced_source
    assert "Step 2. Task Setup" in advanced_source
    assert "Step 3. Prompt Tools" in advanced_source
    assert "Step 4. Run" in advanced_source
    assert "Step 5. Results" in advanced_source
    assert "Step 6. Mask Operations" not in advanced_source
    assert "SAM3 Mask Operations" in manifest_source
    assert "Mask Ops" in simple_run_source
    assert "Log. Activity" in advanced_source
    assert "collapsibleStepBadge" in advanced_source
    assert "activityIndicator" in advanced_source
    assert "Activity: SAM3 preview running..." in advanced_source
    assert "Frame" in advanced_source
    assert "Layer" in advanced_source
    assert "Prompt" in advanced_source
    assert "Object ID" in advanced_source
    assert "Score" in advanced_source
    assert "Area" in advanced_source
    assert "Target image" in advanced_source
    assert "Only change these if your image layout or detection behavior needs manual tuning." in advanced_source
    assert "Detection threshold" in advanced_source
    assert "Enable large-image local inference" in advanced_source
    assert "512 x 512" in advanced_source
    assert "1024 x 1024" in advanced_source
    assert "2048 x 2048" in advanced_source
    assert "Large-image mode ON: local ROI inference" in advanced_source
    assert "Large-image mode OFF: full-image inference" in advanced_source
    assert "SAM3 active ROI" in advanced_source
    assert "Type text prompt, then press Enter to run..." in advanced_source
    assert "textPromptInput" in advanced_source
    assert "returnPressed.connect(self._run_current_task)" in advanced_source
    assert "Text prompt returned zero objects" in advanced_source
    assert "Copy Clipboard" in advanced_source
    assert "Export CSV" in advanced_source
    assert "Model type" in advanced_source
    assert "SAM3.0 2D/3D/video" in advanced_source
    assert "SAM3.1 video multiplex" in advanced_source
    assert "SAM3.1 video multiplex supports 3D/video propagation" in advanced_source
    assert "Load 2D Model" in advanced_source
    assert "Load 3D/Video Model" in advanced_source
    assert "Batch all image layers" in advanced_source
    assert "Batch text prompts" in advanced_source
    assert "multiTextPromptInput" in advanced_source
    assert "Running {len(bundles)} batch job(s)" in advanced_source
    assert "SAM3 preview labels [" in advanced_source
    assert "Saved {saved} batch label layer(s)." in advanced_source
    assert "Create Prompt Layer" in advanced_source
    assert "Clear Preview" in advanced_source
    assert "Preview output" in advanced_source
    assert "Save && Clean" in advanced_source
    assert "Open Folder" in advanced_source
    assert "quick_mask_output_dir" in advanced_source
    assert "MaskExportService" in advanced_source
    assert "PNG is only available for 2D masks. Using TIFF for this 3D/video preview." in advanced_source
    assert "Completed {action}; removed {removed} preview layer(s)." in advanced_source
    assert "Start 3D Propagation" in advanced_source
    assert "Propagate Existing Session" in advanced_source
    assert "Start a new SAM3 video session from the current frame prompt" in advanced_source
    assert "Reuse the current SAM3 video session without adding a new prompt" in advanced_source
    assert "Save Result as Labels" not in main_source + advanced_source
    assert "Apply mode to selected points" in advanced_source
    assert "Add the first point to start Live Points; first run may load the model." in advanced_source
    assert "T = next point mode only. Shift+T = flip selected/latest point and rerun." in advanced_source
    assert "shortcuts_enabled_callback=self._live_refinement_shortcuts_enabled" in advanced_source
    assert 'self.shared_context.get_mode() != "advanced"' in advanced_source
    assert "Qt.ApplicationShortcut" in Path("src/napari_sam3_assistant/widgets/live_point_refinement.py").read_text(encoding="utf-8")
    assert "Qt.ApplicationShortcut" in Path("src/napari_sam3_assistant/widgets/simple/simple_mode_panel.py").read_text(encoding="utf-8")
    assert "Run a new 3D/video preview before propagating again." in advanced_source
    assert "Cancelled 3D/video task; run preview again to start a new SAM3 session." in advanced_source


def test_widget_does_not_expose_dummy_mask_debug_action():
    source = MAIN_WIDGET_SOURCE.read_text(encoding="utf-8") + ADVANCED_WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "Add Dummy Mask Layer" not in source
    assert "_add_dummy_mask" not in source


def test_loading_model_does_not_implicitly_create_prompt_layer():
    source = ADVANCED_WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "def _on_image_initialized" in source
    assert "def _on_video_initialized" in source
    image_handler = source.split("def _on_image_initialized", 1)[1].split(
        "def _on_video_initialized",
        1,
    )[0]
    video_handler = source.split("def _on_video_initialized", 1)[1].split(
        "def _collect_bundle",
        1,
    )[0]

    assert "_initialize_prompt_layer()" not in image_handler
    assert "_initialize_prompt_layer()" not in video_handler


def test_cuda_kernel_image_error_detection():
    error = RuntimeError("CUDA error: no kernel image is available for execution on the device")

    assert _is_cuda_kernel_image_error(error)
    assert not _is_cuda_kernel_image_error(RuntimeError("some other CUDA error"))
