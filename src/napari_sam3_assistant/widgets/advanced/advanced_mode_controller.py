from __future__ import annotations

from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget


class AdvancedModePanel(QWidget):
    """Wrapper placeholder for the existing full/manual UI.

    First migration step:
    - keep this panel simple
    - later move the current MainWidget layout into this class
    """

    def __init__(self, shared_context, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.shared_context = shared_context

        self.placeholder_label = QLabel(
            "Advanced mode placeholder.\n"
            "Next migration step: move the current full UI layout into this panel."
        )
        self.placeholder_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(self.placeholder_label)
        layout.addStretch(1)
        self.setLayout(layout)

    def refresh_from_shared_state(self) -> None:
        pass
