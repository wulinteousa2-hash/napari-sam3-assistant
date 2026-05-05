from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from qtpy.QtWidgets import QShortcut

from ..shared.shared_context import SharedContext
from .simple_mode_controller import SimpleModeController
from .simple_prompt_panel import SimplePromptPanel
from .simple_result_panel import SimpleResultPanel
from .simple_run_panel import SimpleRunPanel
from .simple_task_panel import SimpleTaskPanel


SIMPLE_CONTENT_WIDTH = 520


class SimpleModePanel(QWidget):
    """Guided Simple presentation backed by the shared task router."""

    def __init__(self, shared_context: SharedContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.shared_context = shared_context
        self._layer_events_connected = False
        self.controller = SimpleModeController(shared_context, self)
        self.task_panel = SimpleTaskPanel(self.controller, self)
        self.prompt_panel = SimplePromptPanel(self.controller, self)
        self.run_panel = SimpleRunPanel(self.controller, self)
        self.result_panel = SimpleResultPanel(self.controller, self)
        self.content = QWidget(self)
        self.content.setMinimumWidth(SIMPLE_CONTENT_WIDTH)
        self.content.setMaximumWidth(SIMPLE_CONTENT_WIDTH)
        self.content.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.toggle_point_shortcut = QShortcut(QKeySequence("T"), self)
        self.toggle_point_shortcut.setContext(Qt.ApplicationShortcut)
        self.toggle_point_shortcut.activated.connect(self._toggle_next_point_mode_shortcut)
        self.flip_point_shortcut = QShortcut(QKeySequence("Shift+T"), self)
        self.flip_point_shortcut.setContext(Qt.ApplicationShortcut)
        self.flip_point_shortcut.activated.connect(self._flip_existing_point_polarity_shortcut)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addWidget(self.task_panel)
        content_layout.addWidget(self.prompt_panel)
        content_layout.addWidget(self.run_panel)
        content_layout.addWidget(self.result_panel)
        content_layout.addStretch(1)
        self.content.setLayout(content_layout)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self.content, 0, Qt.AlignLeft | Qt.AlignTop)
        outer_layout.addStretch(1)
        self.setLayout(outer_layout)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.controller.state_changed.connect(self.refresh_from_shared_state)
        self.run_panel.run_requested.connect(self._run_current_task)
        self._connect_viewer_events()

    def refresh_from_shared_state(self) -> None:
        self.task_panel.refresh()
        self.prompt_panel.refresh()
        self.run_panel.refresh()
        self.result_panel.refresh()

    def _run_current_task(self) -> None:
        self.prompt_panel.sync_to_shared_state()
        self.controller.run_current_task()

    def _toggle_next_point_mode_shortcut(self) -> None:
        if self.shared_context.get_mode() != "simple":
            return
        self.controller.toggle_next_point_mode()

    def _flip_existing_point_polarity_shortcut(self) -> None:
        if self.shared_context.get_mode() != "simple":
            return
        self.controller.flip_existing_point_polarity()

    def _connect_viewer_events(self) -> None:
        viewer = self.shared_context.viewer
        if viewer is None or self._layer_events_connected:
            return
        events = getattr(viewer.layers, "events", None)
        if events is not None:
            for event_name in ("inserted", "removed", "reordered"):
                event = getattr(events, event_name, None)
                if event is None:
                    continue
                try:
                    event.connect(self._on_layers_changed)
                except Exception:
                    pass
        selection = getattr(viewer.layers, "selection", None)
        selection_events = getattr(selection, "events", None)
        if selection_events is not None:
            for event_name in ("active", "changed"):
                event = getattr(selection_events, event_name, None)
                if event is None:
                    continue
                try:
                    event.connect(self._on_active_layer_changed)
                except Exception:
                    pass
        self._layer_events_connected = True

    def _on_layers_changed(self, event=None) -> None:
        self.controller.refresh_from_viewer(prefer_active=True)

    def _on_active_layer_changed(self, event=None) -> None:
        self.controller.refresh_from_viewer(prefer_active=True)
