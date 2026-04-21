from pathlib import Path

from napari_sam3_assistant.widgets.main_widget import _is_cuda_kernel_image_error


WIDGET_SOURCE = Path("src/napari_sam3_assistant/widgets/main_widget.py")


def test_widget_uses_clear_model_and_prompt_action_labels():
    source = WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "1. Model Setup" in source
    assert "2. Task Setup" in source
    assert "3. Prompt Tools" in source
    assert "4. Run" in source
    assert "5. Results" in source
    assert "Step 6. Mask Operations" in source
    assert "Log. Activity" in source
    assert "MaskOperationsPanel" in source
    assert "collapsibleStepBadge" in source
    assert "activityIndicator" in source
    assert "Activity: SAM3 preview running..." in source
    assert "Frame" in source
    assert "Layer" in source
    assert "Prompt" in source
    assert "Object ID" in source
    assert "Score" in source
    assert "Area" in source
    assert "Target image" in source
    assert "Advanced" in source
    assert "Only change these if your image layout or detection behavior needs manual tuning." in source
    assert "Detection threshold" in source
    assert "Enable large-image local inference" in source
    assert "512 x 512" in source
    assert "1024 x 1024" in source
    assert "2048 x 2048" in source
    assert "Large-image mode ON: local ROI inference" in source
    assert "Large-image mode OFF: full-image inference" in source
    assert "SAM3 active ROI" in source
    assert "Type text prompt, then press Enter to run..." in source
    assert "textPromptInput" in source
    assert "returnPressed.connect(self._run_current_task)" in source
    assert "Text prompt returned zero objects" in source
    assert "Copy Clipboard" in source
    assert "Export CSV" in source
    assert "Model type" in source
    assert "SAM3.0 2D/3D/video" in source
    assert "SAM3.1 video multiplex" in source
    assert "SAM3.1 video multiplex supports 3D/video propagation" in source
    assert "Load 2D Model" in source
    assert "Load 3D/Video Model" in source
    assert "Batch all image layers" in source
    assert "Batch text prompts" in source
    assert "multiTextPromptInput" in source
    assert "Running {len(bundles)} batch job(s)" in source
    assert "SAM3 preview labels [" in source
    assert "Saved {saved} batch label layer(s)." in source
    assert "Create Prompt Layer" in source
    assert "Clear Preview" in source
    assert "Start 3D Propagation" in source
    assert "Propagate Existing Session" in source
    assert "Start a new SAM3 video session from the current frame prompt" in source
    assert "Reuse the current SAM3 video session without adding a new prompt" in source
    assert "Save Result as Labels" not in source
    assert "Save Accepted Object" not in source
    assert "Apply mode to selected points" in source
    assert "Add the first point to start live refinement; first run may load the model." in source
    assert "T = next point mode only. Shift+T = flip selected/latest point and rerun." in source
    assert "Run a new 3D/video preview before propagating again." in source
    assert "Cancelled 3D/video task; run preview again to start a new SAM3 session." in source


def test_widget_does_not_expose_dummy_mask_debug_action():
    source = WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "Add Dummy Mask Layer" not in source
    assert "_add_dummy_mask" not in source


def test_loading_model_does_not_implicitly_create_prompt_layer():
    source = WIDGET_SOURCE.read_text(encoding="utf-8")

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
