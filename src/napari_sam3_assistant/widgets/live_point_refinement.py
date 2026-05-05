from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable

from qtpy.QtCore import QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut


class LivePointRefinementController(QObject):
    def __init__(
        self,
        parent: Any,
        *,
        run_preview_callback: Callable[[], None],
        toggle_next_mode_callback: Callable[[], None],
        flip_existing_point_callback: Callable[[], None],
        is_enabled_callback: Callable[[], bool],
        shortcuts_enabled_callback: Callable[[], bool] | None = None,
        debounce_ms: int = 200,
    ) -> None:
        super().__init__(parent)

        self._run_preview_callback = run_preview_callback
        self._toggle_next_mode_callback = toggle_next_mode_callback
        self._flip_existing_point_callback = flip_existing_point_callback
        self._is_enabled_callback = is_enabled_callback
        self._shortcuts_enabled_callback = shortcuts_enabled_callback or is_enabled_callback

        self._points_layer: Any | None = None
        self._events_suspended = 0

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._run_if_enabled)

        self._shortcut = QShortcut(QKeySequence("T"), parent)
        self._shortcut.setContext(Qt.ApplicationShortcut)
        self._shortcut.activated.connect(self._toggle_next_mode_if_enabled)

        self._flip_shortcut = QShortcut(QKeySequence("Shift+T"), parent)
        self._flip_shortcut.setContext(Qt.ApplicationShortcut)
        self._flip_shortcut.activated.connect(self._flip_existing_if_enabled)

    def set_points_layer(self, layer: Any | None) -> None:
        self._disconnect_points_layer()
        self._points_layer = layer
        self._connect_points_layer()

    def shutdown(self) -> None:
        self._debounce_timer.stop()
        self._disconnect_points_layer()

    def _connect_points_layer(self) -> None:
        if self._points_layer is None:
            return

        events = getattr(self._points_layer, "events", None)
        if events is None:
            return

        for event_name in ("data", "properties", "set_data"):
            event = getattr(events, event_name, None)
            if event is None:
                continue
            try:
                event.connect(self._on_points_changed)
            except Exception:
                pass

    def _disconnect_points_layer(self) -> None:
        if self._points_layer is None:
            return

        events = getattr(self._points_layer, "events", None)
        if events is not None:
            for event_name in ("data", "properties", "set_data"):
                event = getattr(events, event_name, None)
                if event is None:
                    continue
                try:
                    event.disconnect(self._on_points_changed)
                except Exception:
                    pass

        self._points_layer = None

    def _on_points_changed(self, event: Any = None) -> None:
        if self._events_suspended:
            return
        if not self._is_enabled_callback():
            return
        self._debounce_timer.start()

    def _run_if_enabled(self) -> None:
        if not self._is_enabled_callback():
            return
        self._run_preview_callback()

    def _toggle_next_mode_if_enabled(self) -> None:
        if not self._shortcuts_enabled_callback():
            return
        self._toggle_next_mode_callback()

    def _flip_existing_if_enabled(self) -> None:
        if not self._shortcuts_enabled_callback():
            return
        self._flip_existing_point_callback()
        
    def request_preview(self) -> None:
        if not self._is_enabled_callback():
            return
        self._debounce_timer.start()

    @contextmanager
    def suspend_events(self):
        self._events_suspended += 1
        try:
            yield
        finally:
            self._events_suspended = max(0, self._events_suspended - 1)
