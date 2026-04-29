from __future__ import annotations

import os


VALID_DEVICES = {"cuda", "cpu"}
DEVICE_OVERRIDE_ENV = "NAPARI_SAM3_ENABLE_DEVICE_OVERRIDE"
CUDA_CPU_ONLY_MESSAGE = (
    "GPU/CUDA is selected, but this PyTorch installation is CPU-only.\n"
    "Switch Device to CPU for SAM3.0 2D preview.\n"
    "SAM3.1 and 3D/video propagation require CUDA with an NVIDIA GPU."
)
CPU_3D_MESSAGE = (
    "3D/video propagation requires CUDA. CPU mode is experimental for SAM3.0 2D image workflows only."
)
CPU_SAM31_MESSAGE = (
    "SAM3.1 multiplex requires CUDA/3D-video mode. "
    "CPU mode is experimental for SAM3.0 2D image workflows only."
)
CPU_EXPERIMENTAL_2D_MESSAGE = (
    "Experimental CPU mode for SAM3.0 2D image workflows. Tested with point and "
    "box prompts. GPU/CUDA is still recommended for full SAM3 functionality."
)
UPSTREAM_CPU_CUDA_MESSAGE = (
    "CPU was selected and passed to SAM3, but the installed upstream SAM3 package still "
    "attempted to allocate CUDA tensors during image model construction. This appears "
    "to be an upstream SAM3 CPU-support limitation. Use CUDA/GPU if available, or use "
    "a SAM3 version patched for CPU-safe position encoding."
)
SAVED_CUDA_OVERRIDDEN_MESSAGE = (
    "Saved GPU/CUDA device setting overridden to CPU because CUDA is unavailable."
)


def normalize_requested_device(
    requested: str | None,
    cuda_available: bool,
) -> tuple[str, str | None]:
    device = requested if requested in VALID_DEVICES else None
    if device is None:
        return ("cuda" if cuda_available else "cpu"), None
    if device == "cuda" and not cuda_available:
        return "cpu", SAVED_CUDA_OVERRIDDEN_MESSAGE
    return device, None


def runtime_device(cuda_available: bool) -> str:
    return "cuda" if cuda_available else "cpu"


def manual_device_override_enabled() -> bool:
    value = os.environ.get(DEVICE_OVERRIDE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def device_indicator_tooltip(device: str, *, override_enabled: bool = False) -> str:
    if override_enabled:
        return (
            "Advanced device override is enabled by "
            f"{DEVICE_OVERRIDE_ENV}. Use only for backend testing."
        )
    if device == "cuda":
        return "CUDA PyTorch is available. GPU / CUDA is used for full SAM3 functionality."
    return (
        "CPU-only environment detected. CPU mode is experimental for SAM3.0 2D image "
        "workflows and requires a CPU-safe SAM3 backend."
    )


def is_cuda_not_compiled_error(error: BaseException) -> bool:
    text = str(error).lower()
    return "torch not compiled with cuda enabled" in text


def cpu_prompt_support_error(
    task: str,
    *,
    has_text: bool = False,
    has_points: bool = False,
    has_boxes: bool = False,
    has_masks: bool = False,
    has_exemplars: bool = False,
) -> str | None:
    if task == "3d_video_propagation":
        return CPU_3D_MESSAGE
    return None
