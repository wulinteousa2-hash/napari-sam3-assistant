from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .simple_mode_controller import SimpleModeController


class SimpleTaskPanel(QGroupBox):
    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Target", parent)
        self.controller = controller

        self.image_combo = QComboBox()
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.controller.refresh)

        image_row = QHBoxLayout()
        image_row.addWidget(self.image_combo, 1)
        image_row.addWidget(refresh_btn)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(image_row)
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

        self.summary_label.setText(self.controller.image_summary())

    def _on_image_changed(self) -> None:
        layer_name = self.image_combo.currentData()
        if layer_name:
            self.controller.set_image_layer(str(layer_name))
        self.summary_label.setText(self.controller.image_summary())
