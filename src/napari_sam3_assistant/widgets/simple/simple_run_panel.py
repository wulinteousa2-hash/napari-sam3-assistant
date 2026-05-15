from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

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

        self.save_labels_btn = QPushButton("Save Labels")
        self.save_labels_btn.setObjectName("saveButton")
        self.save_labels_btn.clicked.connect(self.controller.save_preview_labels)

        self.save_clean_btn = QPushButton("Save && Clean")
        self.save_clean_btn.setObjectName("saveButton")
        self.save_clean_btn.clicked.connect(self.controller.save_and_release_preview)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Output folder for Save && Clean")
        self.output_folder_edit.editingFinished.connect(
            lambda: self.controller.set_preview_output_folder(self.output_folder_edit.text())
        )
        self.output_browse_btn = QPushButton("Choose")
        self.output_browse_btn.clicked.connect(self._browse_output_folder)
        self.output_format_combo = QComboBox()
        self.output_format_combo.currentTextChanged.connect(self.controller.set_preview_output_format)

        self.mask_ops_btn = QPushButton("Mask Ops")
        self.mask_ops_btn.clicked.connect(self.controller.open_mask_operations)

        self.activity_log = QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(110)
        self.activity_log.setPlaceholderText("Activity log")

        self._buttons = (
            self.run_btn,
            self.propagate_btn,
            self.save_labels_btn,
            self.save_clean_btn,
            self.clear_preview_btn,
            self.mask_ops_btn,
        )
        self.button_grid = QGridLayout()
        self.button_grid.setSpacing(6)
        self._relayout_buttons(include_propagate=True)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(self.button_grid)
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.output_folder_edit, 1)
        output_row.addWidget(self.output_browse_btn)
        output_row.addWidget(self.output_format_combo)
        layout.addLayout(output_row)
        layout.addWidget(self.activity_log)
        self.setLayout(layout)

        self.controller.shared_context.activity_status.status_changed.connect(
            self.status_label.setText
        )
        self.controller.shared_context.result_visibility.state_changed.connect(
            lambda _state: self.refresh()
        )
        self.controller.shared_context.activity_log.line_added.connect(self._append_log_line)
        self.controller.shared_context.activity_log.cleared.connect(self.activity_log.clear)
        self._reload_log()
        self._sync_output_controls()

    def refresh(self) -> None:
        task = self.controller.current_task()
        state = self.controller.shared_context.result_state
        has_session = bool(
            state.has_video_session
            or getattr(self.controller.owner, "video_session", None) is not None
        )
        if task == Sam3Task.SEGMENT_3D:
            self.run_btn.setText("Start 3D")
        elif task == Sam3Task.EXEMPLAR:
            self.run_btn.setText("Run ROI")
        else:
            self.run_btn.setText("Run")
        self.propagate_btn.setVisible(task == Sam3Task.SEGMENT_3D)
        self.propagate_btn.setEnabled(task == Sam3Task.SEGMENT_3D and has_session)
        self.save_labels_btn.setEnabled(state.has_any_result)
        self.save_clean_btn.setEnabled(state.has_any_result)
        self.status_label.setText(self.controller.shared_context.activity_status.status)
        self._sync_output_controls()
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

    def _append_log_line(self, line: str) -> None:
        self.activity_log.appendPlainText(line)

    def _reload_log(self) -> None:
        self.activity_log.setPlainText("\n".join(self.controller.shared_context.activity_log.recent(10)))

    def _sync_output_controls(self) -> None:
        self.controller.sync_preview_output_controls()
        old = self.output_folder_edit.blockSignals(True)
        try:
            self.output_folder_edit.setText(self.controller.preview_output_folder())
        finally:
            self.output_folder_edit.blockSignals(old)

        current = self.controller.preview_output_format()
        old = self.output_format_combo.blockSignals(True)
        try:
            items = self.controller.preview_output_format_items()
            existing = [self.output_format_combo.itemText(i) for i in range(self.output_format_combo.count())]
            if existing != items:
                self.output_format_combo.clear()
                self.output_format_combo.addItems(items)
            index = self.output_format_combo.findText(current)
            if index >= 0:
                self.output_format_combo.setCurrentIndex(index)
        finally:
            self.output_format_combo.blockSignals(old)

    def _browse_output_folder(self) -> None:
        self.controller.browse_preview_output_folder()
        self._sync_output_controls()
