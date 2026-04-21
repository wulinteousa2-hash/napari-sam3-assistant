from __future__ import annotations

from typing import Any

from qtpy.QtCore import QObject, Signal

from .ui_state_models import ResultState


class ResultVisibilityController(QObject):
    state_changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = ResultState()

    def current_result_state(self) -> ResultState:
        return self._state

    def on_result_written(self, result: Any) -> ResultState:
        has_labels = bool(getattr(result, "labels", None) is not None)
        is_empty = bool(result.is_empty()) if hasattr(result, "is_empty") else False
        count = 0
        object_ids = getattr(result, "object_ids", None)
        if object_ids is not None:
            try:
                count = len(object_ids)
            except Exception:
                count = 0
        elif getattr(result, "masks", None) is not None:
            try:
                masks = getattr(result, "masks")
                count = int(masks.shape[0]) if len(masks.shape) >= 3 else 0
            except Exception:
                count = 0
        self._state = ResultState(
            has_any_result=not is_empty,
            has_labels_result=self._state.has_labels_result or has_labels,
            has_video_session=self._state.has_video_session or bool(getattr(result, "session_id", None)),
            result_count=count,
            metadata=dict(getattr(result, "metadata", {}) or {}),
        )
        self.state_changed.emit(self._state)
        return self._state

    def on_results_cleared(self) -> ResultState:
        self._state = ResultState(
            has_any_result=False,
            has_labels_result=False,
            has_video_session=self._state.has_video_session,
            result_count=0,
            metadata=None,
        )
        self.state_changed.emit(self._state)
        return self._state

    def on_preview_layers_cleared(self) -> ResultState:
        self._state = ResultState(
            has_any_result=False,
            has_labels_result=False,
            has_video_session=self._state.has_video_session,
            result_count=0,
            metadata=None,
        )
        self.state_changed.emit(self._state)
        return self._state

    def on_video_session_changed(self, has_session: bool) -> ResultState:
        self._state = ResultState(
            has_any_result=self._state.has_any_result,
            has_labels_result=self._state.has_labels_result,
            has_video_session=bool(has_session),
            result_count=self._state.result_count,
            metadata=self._state.metadata,
        )
        self.state_changed.emit(self._state)
        return self._state
