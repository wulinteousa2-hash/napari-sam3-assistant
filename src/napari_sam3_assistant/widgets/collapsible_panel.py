from __future__ import annotations

from qtpy.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget


class CollapsiblePanel(QWidget):
    """
    Compact collapsible panel with a small +/- toggle.

    Behavior:
    - "-" means expanded
    - "+" means collapsed
    """

    def __init__(
        self,
        title: str,
        content: QWidget,
        collapsed: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.content = content
        step_text, title_text = self._split_title(title)

        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(collapsed)
        self.toggle_button.setText("+" if collapsed else "−")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setObjectName("collapsibleToggle")
        self.toggle_button.clicked.connect(self._on_toggled)

        self.step_label = QLabel(step_text)
        self.step_label.setObjectName("collapsibleStepBadge")
        self.step_label.setVisible(bool(step_text))

        self.title_label = QLabel(title_text)
        self.title_label.setObjectName("collapsibleTitle")

        self.header_line = QFrame()
        self.header_line.setFrameShape(QFrame.HLine)
        self.header_line.setFrameShadow(QFrame.Sunken)
        self.header_line.setObjectName("collapsibleHeaderLine")

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.step_label)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.header_line, 1)

        self.body_frame = QFrame()
        self.body_frame.setObjectName("collapsibleBody")
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        body_layout.addWidget(self.content)
        self.body_frame.setLayout(body_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(header_layout)
        layout.addWidget(self.body_frame)
        self.setLayout(layout)

        self._on_toggled(collapsed)

    def _split_title(self, title: str) -> tuple[str, str]:
        normalized = title.strip()
        head, separator, tail = normalized.partition(".")
        if separator and head.strip() and tail.strip():
            return head.strip(), tail.strip()
        return "", normalized

    def _on_toggled(self, checked: bool) -> None:
        # checked = collapsed
        self.toggle_button.setText("+" if checked else "−")
        self.body_frame.setVisible(not checked)
