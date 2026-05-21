from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from qtpy.QtWidgets import QShortcut

from ...core.models import Sam3Task
from ..live_points_accept import LivePointsContextMenu
from ..shared.shared_context import SharedContext
from .simple_mode_controller import SimpleModeController
from .simple_model_panel import SimpleModelPanel
from .simple_run_panel import SimpleRunPanel
from .simple_task_panel import SimpleTaskPanel
from .simple_workflow_panel import SimpleWorkflowPanel


SIMPLE_CONTENT_WIDTH = 520


class SimpleModePanel(QWidget):
    """Guided Simple presentation backed by the shared task router."""

    def __init__(self, shared_context: SharedContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.shared_context = shared_context
        self._layer_events_connected = False
        self.controller = SimpleModeController(shared_context, self)
        self._live_points_context_menu = LivePointsContextMenu(
            self,
            self.controller,
            self._live_points_context_menu_enabled,
        )
        self._live_points_mouse_callback = self._handle_live_points_mouse_action
        self.model_panel = SimpleModelPanel(self.controller, self)
        self.task_panel = SimpleTaskPanel(self.controller, self)
        self.workflow_panel = SimpleWorkflowPanel(self.controller, self)
        self.run_panel = SimpleRunPanel(self.controller, self)
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
        content_layout.addWidget(self.model_panel)
        content_layout.addWidget(self.task_panel)
        content_layout.addWidget(self.workflow_panel)
        content_layout.addWidget(self.run_panel)
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
        self.model_panel.refresh()
        self.task_panel.refresh()
        self.workflow_panel.refresh()
        self.run_panel.refresh()
        self._sync_live_points_mouse_callback()

    def _run_current_task(self) -> None:
        self.workflow_panel.sync_to_shared_state()
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

    def _sync_live_points_mouse_callback(self) -> None:
        layer = self.controller.current_points_layer() if self._live_points_context_menu_enabled() else None
        self._live_points_context_menu.sync_mouse_callback(layer)

    def _disconnect_live_points_mouse_callback(self) -> None:
        self._live_points_context_menu.disconnect_mouse_callback()

    def _handle_live_points_mouse_action(self, layer, event):
        self._live_points_context_menu.handle_mouse_action(layer, event)

    def _live_points_context_menu_enabled(self) -> bool:
        return (
            self.shared_context.get_mode() == "simple"
            and self.controller.current_task() == Sam3Task.REFINE
        )
