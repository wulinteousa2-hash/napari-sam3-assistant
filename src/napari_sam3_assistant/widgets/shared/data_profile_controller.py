from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from coordinates import infer_image_selection, base_image_data, extract_2d_image
    from models import Sam3Task
    from ui_state_models import DataProfile
except Exception:  # pragma: no cover
    from .coordinates import infer_image_selection, base_image_data, extract_2d_image  # type: ignore
    from .models import Sam3Task  # type: ignore
    from .ui_state_models import DataProfile  # type: ignore


@dataclass
class DataProfileController:
    large_image_threshold_px: int = 4096

    def build_profile(
        self,
        viewer: Any,
        image_layer_name: str,
        *,
        channel_axis: int | None = None,
    ) -> DataProfile:
        image_layer = viewer.layers[image_layer_name]
        image_data = base_image_data(image_layer.data)
        dims_current_step = tuple(getattr(viewer.dims, "current_step", ()))
        selection = infer_image_selection(
            layer_name=image_layer.name,
            data_shape=tuple(int(v) for v in image_data.shape),
            dims_current_step=dims_current_step,
            channel_axis=channel_axis,
        )

        frame = extract_2d_image(image_data, selection)
        image_hw = tuple(int(v) for v in frame.shape[:2]) if frame.ndim >= 2 else (0, 0)
        is_rgb_like = bool(frame.ndim == 3 and frame.shape[-1] in (3, 4))
        has_channel_axis = selection.channel_axis is not None
        has_frame_axis = selection.frame_axis is not None
        is_2d_only = not has_frame_axis
        is_stack_or_video = has_frame_axis
        is_large_image = max(image_hw) >= int(self.large_image_threshold_px)

        allowed_tasks = self._allowed_tasks(is_stack_or_video)
        suggested_task = Sam3Task.SEGMENT_3D if is_stack_or_video else Sam3Task.SEGMENT_2D
        suggested_model_type = "sam3.1" if is_stack_or_video else "sam3"

        notes: list[str] = []
        if is_rgb_like:
            notes.append("RGB-like image detected")
        if is_large_image:
            notes.append("Large image detected; local ROI inference may be preferable")
        if is_stack_or_video:
            notes.append("Frame axis detected; 3D/video propagation is available")
        else:
            notes.append("2D-only image; 3D/video propagation should stay hidden")

        return DataProfile(
            layer_name=image_layer.name,
            data_shape=tuple(int(v) for v in image_data.shape),
            ndim=int(image_data.ndim),
            is_rgb_like=is_rgb_like,
            has_channel_axis=has_channel_axis,
            has_frame_axis=has_frame_axis,
            is_2d_only=is_2d_only,
            is_stack_or_video=is_stack_or_video,
            is_large_image=is_large_image,
            image_hw=image_hw,
            allowed_tasks=allowed_tasks,
            suggested_task=suggested_task,
            suggested_model_type=suggested_model_type,
            notes=tuple(notes),
        )

    def _allowed_tasks(self, is_stack_or_video: bool) -> tuple[Sam3Task, ...]:
        if is_stack_or_video:
            return (
                Sam3Task.SEGMENT_2D,
                Sam3Task.SEGMENT_3D,
                Sam3Task.EXEMPLAR,
                Sam3Task.TEXT,
                Sam3Task.REFINE,
            )
        return (
            Sam3Task.SEGMENT_2D,
            Sam3Task.EXEMPLAR,
            Sam3Task.TEXT,
            Sam3Task.REFINE,
        )