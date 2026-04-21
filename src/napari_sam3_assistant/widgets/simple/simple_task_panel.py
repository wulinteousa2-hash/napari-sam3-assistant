from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.models import Sam3Task
from .simple_mode_controller import SimpleModeController


class SimpleTaskPanel(QGroupBox):
    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Setup", parent)
        self.controller = controller
        self._task_buttons: dict[Sam3Task, QPushButton] = {}

        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.controller.refresh)

        image_row = QHBoxLayout()
        image_row.addWidget(self.image_combo, 1)
        image_row.addWidget(refresh_btn)

        task_grid = QGridLayout()
        task_grid.setSpacing(6)
        task_specs = (
            ("2D", Sam3Task.SEGMENT_2D),
            ("Text", Sam3Task.TEXT),
            ("Live Points", Sam3Task.REFINE),
            ("Exemplar", Sam3Task.EXEMPLAR),
            ("3D/Video", Sam3Task.SEGMENT_3D),
        )
        for index, (label, task) in enumerate(task_specs):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, task=task: self.controller.set_task(task))
            self._task_buttons[task] = button
            task_grid.addWidget(button, index // 3, index % 3)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(image_row)
        layout.addLayout(task_grid)
        layout.addWidget(self.summary_label)
        self.setLayout(layout)

    def refresh(self) -> None:
        current = self.controller.current_image_layer_name()
        names = self.controller.image_layer_names()
        if (not current or current not in names) and names:
            current = names[0]
            self.controller.set_image_layer(current)
        self.image_combo.blockSignals(True)
        try:
            self.image_combo.clear()
            for name in names:
                self.image_combo.addItem(name, name)
            if current:
                index = self.image_combo.findData(current)
                if index >= 0:
                    self.image_combo.setCurrentIndex(index)
        finally:
            self.image_combo.blockSignals(False)

        current_task = self.controller.current_task()
        for task, button in self._task_buttons.items():
            button.setChecked(task == current_task)

        self.summary_label.setText(self.controller.image_summary())

    def _on_image_changed(self) -> None:
        layer_name = self.image_combo.currentData()
        if layer_name:
            self.controller.set_image_layer(str(layer_name))
        self.summary_label.setText(self.controller.image_summary())
