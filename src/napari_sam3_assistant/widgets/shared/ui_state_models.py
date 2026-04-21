from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...core.models import Sam3Task


@dataclass(frozen=True)
class DataProfile:
    layer_name: str
    data_shape: tuple[int, ...]
    ndim: int
    is_rgb_like: bool
    has_channel_axis: bool
    has_frame_axis: bool
    is_2d_only: bool
    is_stack_or_video: bool
    is_large_image: bool
    image_hw: tuple[int, int]
    allowed_tasks: tuple[Sam3Task, ...]
    suggested_task: Sam3Task
    suggested_model_type: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class UiModeState:
    mode: str  # "simple" | "advanced"

    @property
    def is_simple(self) -> bool:
        return self.mode == "simple"

    @property
    def is_advanced(self) -> bool:
        return self.mode == "advanced"


@dataclass(frozen=True)
class ResultState:
    has_any_result: bool = False
    has_labels_result: bool = False
    has_video_session: bool = False
    result_count: int = 0
    metadata: dict[str, Any] | None = None
