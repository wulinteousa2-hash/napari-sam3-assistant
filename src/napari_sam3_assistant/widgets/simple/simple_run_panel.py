from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core.models import Sam3Task
from .simple_mode_controller import SimpleModeController


class SimpleRunPanel(QGroupBox):
    run_requested = Signal()

    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Run", parent)
        self.controller = controller

        self.status_label = QLabel(self.controller.shared_context.activity_status.status)
        self.status_label.setObjectName("activityIndicator")

        self.run_btn = QPushButton("Run Preview")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self.run_requested.emit)

        self.propagate_btn = QPushButton("Propagate")
        self.propagate_btn.setObjectName("runButton")
        self.propagate_btn.clicked.connect(self.controller.propagate_existing_session)

        self.clear_preview_btn = QPushButton("Clear Preview")
        self.clear_preview_btn.setObjectName("clearButton")
        self.clear_preview_btn.clicked.connect(self.controller.clear_preview_layers)

        self.clear_results_btn = QPushButton("Clear Results")
        self.clear_results_btn.setObjectName("clearButton")
        self.clear_results_btn.clicked.connect(self.controller.clear_results)

        self.mask_ops_btn = QPushButton("Mask Ops")
        self.mask_ops_btn.clicked.connect(self.controller.open_mask_operations)

        self._buttons = (
            self.run_btn,
            self.propagate_btn,
            self.clear_preview_btn,
            self.clear_results_btn,
            self.mask_ops_btn,
        )
        self.button_grid = QGridLayout()
        self.button_grid.setSpacing(6)
        self._relayout_buttons(include_propagate=True)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(self.button_grid)
        self.setLayout(layout)

        self.controller.shared_context.activity_status.status_changed.connect(
            self.status_label.setText
        )
        self.controller.shared_context.result_visibility.state_changed.connect(
            lambda _state: self.refresh()
        )

    def refresh(self) -> None:
        task = self.controller.current_task()
        state = self.controller.shared_context.result_state
        has_session = bool(
            state.has_video_session
            or getattr(self.controller.owner, "video_session", None) is not None
        )
        self.run_btn.setText("Start 3D" if task == Sam3Task.SEGMENT_3D else "Run Preview")
        self.propagate_btn.setVisible(task == Sam3Task.SEGMENT_3D)
        self.propagate_btn.setEnabled(task == Sam3Task.SEGMENT_3D and has_session)
        self.status_label.setText(self.controller.shared_context.activity_status.status)
        self._relayout_buttons(include_propagate=task == Sam3Task.SEGMENT_3D)

    def _relayout_buttons(self, *, include_propagate: bool) -> None:
        for button in self._buttons:
            self.button_grid.removeWidget(button)

        visible_buttons = [
            button
            for button in self._buttons
            if button is not self.propagate_btn or include_propagate
        ]
        for index, button in enumerate(visible_buttons):
            self.button_grid.addWidget(button, index // 3, index % 3)
