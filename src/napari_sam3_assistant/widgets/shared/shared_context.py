from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .activity_status_controller import ActivityStatusController
from .result_visibility_controller import ResultVisibilityController
from .ui_state_models import ResultState, UiModeState


@dataclass
class SharedContext:
    """Shared runtime state used by both Simple and Advanced UI modes.

    This is intentionally presentation-agnostic. Both modes should read/write
    the same execution state through this object instead of keeping parallel
    copies of the same values.
    """

    viewer: Any | None = None
    settings: Any | None = None

    provider: Any | None = None
    checkpoint_service: Any | None = None
    prompt_state_service: Any | None = None
    prompt_collector: Any | None = None
    layer_writer: Any | None = None

    adapter: Any | None = None
    video_session: Any | None = None
    worker: Any | None = None
    worker_failed: bool = False

    image_runner: Any | None = None
    refinement_runner: Any | None = None
    video_runner: Any | None = None

    task_router: Any | None = None
    mode_change_callback: Any | None = None
    result_visibility: ResultVisibilityController = field(default_factory=ResultVisibilityController)
    active_rois: dict[str, Any] = field(default_factory=dict)
    activity_status: ActivityStatusController = field(default_factory=ActivityStatusController)
    ui_mode_state: UiModeState = field(default_factory=lambda: UiModeState("simple"))
    result_state: ResultState = field(default_factory=ResultState)
    transient: dict[str, Any] = field(default_factory=dict)

    def set_mode(self, mode: str) -> None:
        normalized = mode if mode in {"simple", "advanced"} else "simple"
        self.ui_mode_state = UiModeState(normalized)
        self.transient["ui_mode"] = normalized

    def get_mode(self, default: str = "simple") -> str:
        value = self.transient.get("ui_mode", default)
        return value if value in {"simple", "advanced"} else default

    def request_mode(self, mode: str) -> None:
        if callable(self.mode_change_callback):
            self.mode_change_callback(mode)
        else:
            self.set_mode(mode)
