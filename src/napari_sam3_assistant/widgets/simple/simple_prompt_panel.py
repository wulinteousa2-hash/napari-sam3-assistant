from __future__ import annotations

from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.models import Sam3Task
from ..advanced.advanced_mode_panel import (
    PROMPT_BOX,
    PROMPT_LABELS,
    PROMPT_POINTS,
    PROMPT_TEXT,
)
from .simple_mode_controller import SimpleModeController


class SimplePromptPanel(QGroupBox):
    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Prompt", parent)
        self.controller = controller
        self._prompt_buttons: dict[str, QPushButton] = {}

        self.prompt_grid = QGridLayout()
        for index, (label, tool) in enumerate(
            (
                ("Points", PROMPT_POINTS),
                ("Box", PROMPT_BOX),
                ("Labels", PROMPT_LABELS),
                ("Text", PROMPT_TEXT),
            )
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, tool=tool: self.controller.set_prompt_tool(tool))
            self._prompt_buttons[tool] = button
            self.prompt_grid.addWidget(button, index // 3, index % 3)

        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setObjectName("textPromptInput")
        self.text_prompt_edit.setPlaceholderText("Text prompt")
        self.text_prompt_edit.editingFinished.connect(self._sync_text_prompt)
        self.text_prompt_edit.returnPressed.connect(self._run_from_text)

        self.point_polarity_combo = QComboBox()
        self.point_polarity_combo.addItem("Positive", "positive")
        self.point_polarity_combo.addItem("Negative", "negative")
        self.point_polarity_combo.currentIndexChanged.connect(self._on_polarity_changed)

        self.create_prompt_btn = QPushButton("Create Prompt Layer")
        self.create_prompt_btn.clicked.connect(self.controller.create_prompt_layer)
        self.clear_prompt_btn = QPushButton("Reset Prompt")
        self.clear_prompt_btn.setObjectName("clearButton")
        self.clear_prompt_btn.setToolTip(
            "Clears text/internal prompt state and resets the video session. "
            "It does not delete napari prompt layers."
        )
        self.clear_prompt_btn.clicked.connect(self.controller.clear_prompt_state)

        action_row = QHBoxLayout()
        action_row.addWidget(self.create_prompt_btn)
        action_row.addWidget(self.clear_prompt_btn)

        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addLayout(self.prompt_grid)
        layout.addWidget(self.text_prompt_edit)
        layout.addWidget(self.point_polarity_combo)
        layout.addLayout(action_row)
        layout.addWidget(self.hint_label)
        self.setLayout(layout)

    def refresh(self) -> None:
        task = self.controller.current_task()
        prompt_tool = self.controller.current_prompt_tool()
        if task == Sam3Task.TEXT:
            prompt_tool = PROMPT_TEXT
            self.controller.set_prompt_tool(PROMPT_TEXT)
        elif task == Sam3Task.REFINE:
            prompt_tool = PROMPT_POINTS
            self.controller.set_prompt_tool(PROMPT_POINTS)
        elif task == Sam3Task.EXEMPLAR:
            prompt_tool = PROMPT_BOX
            self.controller.set_prompt_tool(PROMPT_BOX)

        allowed = self._allowed_prompt_tools(task)
        for tool, button in self._prompt_buttons.items():
            visible = tool in allowed
            button.setVisible(visible)
            button.setChecked(tool == prompt_tool)

        text_visible = task == Sam3Task.TEXT or (
            prompt_tool == PROMPT_TEXT
            and task in {Sam3Task.SEGMENT_2D, Sam3Task.SEGMENT_3D}
        )
        self.text_prompt_edit.setVisible(text_visible)
        self.text_prompt_edit.setText(self.controller.current_text_prompt())
        self.point_polarity_combo.setVisible(prompt_tool == PROMPT_POINTS)
        self.point_polarity_combo.blockSignals(True)
        try:
            index = self.point_polarity_combo.findData(self.controller.current_point_polarity())
            if index >= 0:
                self.point_polarity_combo.setCurrentIndex(index)
        finally:
            self.point_polarity_combo.blockSignals(False)

        self.create_prompt_btn.setVisible(task in {Sam3Task.SEGMENT_2D, Sam3Task.SEGMENT_3D})
        self.clear_prompt_btn.setVisible(
            task == Sam3Task.SEGMENT_3D
            or (task == Sam3Task.SEGMENT_2D and prompt_tool != PROMPT_TEXT)
        )
        self.hint_label.setText(self._hint_text(task, prompt_tool))

    def _allowed_prompt_tools(self, task: Sam3Task) -> set[str]:
        if task == Sam3Task.TEXT:
            return set()
        if task == Sam3Task.REFINE:
            return set()
        if task == Sam3Task.EXEMPLAR:
            return set()
        if task == Sam3Task.SEGMENT_3D:
            return {PROMPT_POINTS, PROMPT_BOX, PROMPT_TEXT}
        return {PROMPT_POINTS, PROMPT_BOX, PROMPT_LABELS, PROMPT_TEXT}

    def _hint_text(self, task: Sam3Task, prompt_tool: str) -> str:
        if task == Sam3Task.TEXT:
            return "Enter a short concept and press Enter."
        if task == Sam3Task.REFINE:

            return (
                "Click one point to start the mask.<br>"
                "A result appears right away.<br><br>"
                "<b>Positive</b>: keep the object.<br>"
                "<b>Negative</b>: remove wrong areas.<br><br>"
                "Use only a few points—too many can weaken or remove the mask.<br><br>"
                "<b>T</b>: change next point mode.<br>"
                "<b>Shift+T</b>: flip the selected/latest point and rerun."
            )

        if task == Sam3Task.EXEMPLAR:
            return (
                "A boxes layer is created automatically.\n"
                "Draw a box around an example object.\n"
                "SAM uses it as an example when you run preview."
            )
        if task == Sam3Task.SEGMENT_3D:
            return "Select a frame, add points or boxes, then start propagation."
        if prompt_tool == PROMPT_LABELS:
            return "Paint non-zero labels in the prompt layer."
        return "Create a prompt layer or enter text, then run preview."

    def _sync_text_prompt(self) -> None:
        self.controller.set_text_prompt(self.text_prompt_edit.text())

    def sync_to_shared_state(self) -> None:
        self._sync_text_prompt()
        self.controller.set_prompt_tool(self.controller.current_prompt_tool())
        if self.point_polarity_combo.isVisible():
            self.controller.set_point_polarity(
                self.point_polarity_combo.currentData() or "positive"
            )

    def _run_from_text(self) -> None:
        self.sync_to_shared_state()
        self.controller.run_current_task()

    def _on_polarity_changed(self) -> None:
        self.controller.set_point_polarity(self.point_polarity_combo.currentData() or "positive")
