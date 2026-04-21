from __future__ import annotations

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget


class ModeSwitchBar(QWidget):
    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.simple_radio = QRadioButton("Simple")
        self.advanced_radio = QRadioButton("Advanced")

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.addButton(self.simple_radio)
        self.group.addButton(self.advanced_radio)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.simple_radio)
        layout.addWidget(self.advanced_radio)
        self.setLayout(layout)

        self.simple_radio.toggled.connect(self._emit_mode_if_changed)
        self.advanced_radio.toggled.connect(self._emit_mode_if_changed)

        self.set_mode("simple")

    def current_mode(self) -> str:
        return "advanced" if self.advanced_radio.isChecked() else "simple"

    def set_mode(self, mode: str) -> None:
        mode = mode if mode in {"simple", "advanced"} else "simple"
        old = self.blockSignals(True)
        try:
            self.simple_radio.setChecked(mode == "simple")
            self.advanced_radio.setChecked(mode == "advanced")
        finally:
            self.blockSignals(old)

    def _emit_mode_if_changed(self) -> None:
        mode = self.current_mode()
        self.mode_changed.emit(mode)
