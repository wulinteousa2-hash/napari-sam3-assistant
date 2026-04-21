from __future__ import annotations

from qtpy.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget

from ..shared.ui_state_models import ResultState
from .simple_mode_controller import SimpleModeController


class SimpleResultPanel(QGroupBox):
    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Results", parent)
        self.controller = controller
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.summary_label)
        self.setLayout(layout)

        self.controller.shared_context.result_visibility.state_changed.connect(
            self._on_result_state_changed
        )

    def refresh(self) -> None:
        self._set_summary(self.controller.shared_context.result_state)

    def _on_result_state_changed(self, state: object) -> None:
        if isinstance(state, ResultState):
            self.controller.shared_context.result_state = state
            self._set_summary(state)

    def _set_summary(self, state: ResultState) -> None:
        if not state.has_any_result:
            session_text = " Video session ready." if state.has_video_session else ""
            self.summary_label.setText(f"No preview result yet.{session_text}")
            return
        label_text = "labels" if state.has_labels_result else "preview"
        count_text = f"{state.result_count} object(s)" if state.result_count else "objects available"
        session_text = " Video session ready." if state.has_video_session else ""
        self.summary_label.setText(f"Latest {label_text}: {count_text}.{session_text}")
