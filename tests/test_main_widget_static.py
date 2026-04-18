from pathlib import Path

from napari_sam3_assistant.widgets.main_widget import _is_cuda_kernel_image_error


WIDGET_SOURCE = Path("src/napari_sam3_assistant/widgets/main_widget.py")


def test_widget_uses_clear_model_and_prompt_action_labels():
    source = WIDGET_SOURCE.read_text(encoding="utf-8")

    assert "Load Image Model" in source
    assert "Load 3D/Video Model" in source
    assert "Create Prompt Layer" in source
    assert "Clear Preview" in source
    assert "Save Result as Labels" in source


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
