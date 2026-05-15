from __future__ import annotations

from qtpy.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QPushButton, QWidget

from ...core.models import Sam3Task
from ..collapsible_panel import CollapsiblePanel
from .simple_mode_controller import SimpleModeController


class SimpleModelPanel(CollapsiblePanel):
    """Collapsed Simple-mode model setup with task-aware model selection."""

    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        self.controller = controller
        content = QWidget()
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        self.device_label = QLabel()
        self.choose_btn = QPushButton("Choose Model Folder")
        self.choose_btn.clicked.connect(self._choose_model_folder)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(self.choose_btn)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Using", self.model_label)
        form.addRow("Folder", folder_row)
        form.addRow("Device", self.device_label)
        content.setLayout(form)
        super().__init__("Model", content, collapsed=True, parent=parent)

    def refresh(self) -> None:
        task = self.controller.current_task()
        if task == Sam3Task.SEGMENT_3D:
            model_text = "SAM3.1 multiplex for 3D/video propagation"
        else:
            model_text = "SAM3.0 for 2D image workflows"
        self.model_label.setText(model_text)
        folder = self.controller.current_model_dir().strip()
        self.folder_label.setText(folder if folder else "No model folder selected")
        self.folder_label.setToolTip(folder)
        self.device_label.setText(self.controller.current_device().upper())

    def _choose_model_folder(self) -> None:
        self.controller.browse_model_dir()
        self.refresh()
