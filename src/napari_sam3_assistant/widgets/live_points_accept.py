from __future__ import annotations

from typing import Any, Callable

import numpy as np
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QMenu, QWidget


class LivePointsAcceptService:
    def __init__(
        self,
        *,
        viewer_getter: Callable[[], Any],
        points_layer_getter: Callable[[], Any | None],
        preview_layer_getter: Callable[[], Any | None],
        clear_preview_callback: Callable[[], None] | None = None,
        clear_prompt_callback: Callable[[], None] | None = None,
        activate_layer_callback: Callable[[], None] | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._viewer_getter = viewer_getter
        self._points_layer_getter = points_layer_getter
        self._preview_layer_getter = preview_layer_getter
        self._clear_preview_callback = clear_preview_callback
        self._clear_prompt_callback = clear_prompt_callback
        self._activate_layer_callback = activate_layer_callback
        self._log_callback = log_callback
        self._live_accept_undo: list[np.ndarray] = []

    def current_points_layer(self) -> Any | None:
        return self._points_layer_getter()

    def first_preview_labels_layer(self) -> Any | None:
        return self._preview_layer_getter()

    def has_preview_labels_layer(self) -> bool:
        return self.first_preview_labels_layer() is not None

    def accept_live_preview(self, coords: tuple[int, ...] | None = None, *, clear_prompt: bool = True) -> bool:
        viewer = self._viewer_getter()
        preview = self.first_preview_labels_layer()
        if viewer is None or preview is None:
            self._log("No Live Points preview labels found to accept.")
            return False

        preview_data = np.asarray(preview.data)
        if not np.any(preview_data):
            self._log("Live Points preview is empty; nothing accepted.")
            return False

        mask = preview_data > 0
        if coords is not None and len(coords) == preview_data.ndim:
            in_bounds = all(0 <= coord < size for coord, size in zip(coords, preview_data.shape, strict=False))
            if in_bounds:
                value = int(preview_data[coords])
                if value > 0:
                    mask = preview_data == value

        accepted = self._live_accepted_layer(preview_data.shape)
        accepted_data = np.asarray(accepted.data)
        self._live_accept_undo.append(accepted_data.copy())
        if len(self._live_accept_undo) > 20:
            self._live_accept_undo.pop(0)

        object_id = int(accepted_data.max()) + 1 if accepted_data.size else 1
        write_mask = mask & (accepted_data == 0)
        changed = int(np.count_nonzero(write_mask))
        if changed == 0:
            write_mask = mask
            changed = int(np.count_nonzero(write_mask))
        accepted_data = accepted_data.copy()
        accepted_data[write_mask] = object_id
        accepted.data = accepted_data
        try:
            accepted.refresh()
        except Exception:
            pass

        if self._clear_preview_callback is not None:
            self._clear_preview_callback()
        if clear_prompt:
            self.clear_live_points_prompt()
        else:
            self._activate_live_points_layer()
        suffix = " Prompt cleared." if clear_prompt else ""
        self._log(f"Accepted Live Points preview as object {object_id} ({changed} pixel(s)).{suffix}")
        return True

    def undo_live_accept(self) -> bool:
        viewer = self._viewer_getter()
        if viewer is None or not self._live_accept_undo:
            self._log("No accepted Live Points object to undo.")
            return False
        try:
            layer = viewer.layers["SAM3 live accepted labels"]
        except Exception:
            return False
        layer.data = self._live_accept_undo.pop()
        try:
            layer.refresh()
        except Exception:
            pass
        self._activate_live_points_layer()
        self._log("Undid last accepted Live Points object.")
        return True

    def clear_live_points_prompt(self) -> None:
        if self._clear_prompt_callback is not None:
            self._clear_prompt_callback()

    def _live_accepted_layer(self, shape: tuple[int, ...]) -> Any:
        viewer = self._viewer_getter()
        assert viewer is not None
        name = "SAM3 live accepted labels"
        try:
            layer = viewer.layers[name]
            if tuple(np.asarray(layer.data).shape) == tuple(shape):
                return layer
        except Exception:
            pass
        data = np.zeros(shape, dtype=np.uint32)
        return viewer.add_labels(data, name=name)

    def _activate_live_points_layer(self) -> None:
        if self._activate_layer_callback is not None:
            self._activate_layer_callback()

    def _log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(message)


class LivePointsContextMenu:
    def __init__(
        self,
        parent: QWidget,
        controller: Any,
        enabled_callback: Callable[[], bool],
    ) -> None:
        self._parent = parent
        self._controller = controller
        self._enabled_callback = enabled_callback
        self._mouse_layer: Any | None = None
        self._mouse_callback = self.handle_mouse_action

    def sync_mouse_callback(self, layer: Any | None) -> None:
        self.disconnect_mouse_callback()
        if layer is None or not self._enabled_callback():
            return
        callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if callbacks is None:
            return
        if self._mouse_callback not in callbacks:
            callbacks.append(self._mouse_callback)
        self._mouse_layer = layer

    def disconnect_mouse_callback(self) -> None:
        layer = self._mouse_layer
        if layer is None:
            return
        callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if callbacks is not None and self._mouse_callback in callbacks:
            callbacks.remove(self._mouse_callback)
        self._mouse_layer = None

    def handle_mouse_action(self, layer: Any, event: Any) -> None:
        if not self._enabled_callback():
            return
        if not _is_right_mouse_event(event):
            return
        self.open(event)

    def open(self, event: Any) -> None:
        coords = _preview_coords_from_event(event, self._controller.first_preview_labels_layer())
        menu = QMenu(self._parent)
        accept_clear_action = menu.addAction("Accept + Clear")
        accept_only_action = menu.addAction("Accept Only")
        clear_action = menu.addAction("Clear Prompt")
        undo_action = menu.addAction("Undo Last Accept")
        if coords is None:
            has_preview = self._controller.first_preview_labels_layer() is not None
            accept_clear_action.setEnabled(has_preview)
            accept_only_action.setEnabled(has_preview)
        selected = menu.exec_(_event_global_position(event))
        _mark_event_handled(event)
        if selected == accept_clear_action:
            self._controller.accept_live_preview(coords, clear_prompt=True)
        elif selected == accept_only_action:
            self._controller.accept_live_preview(coords, clear_prompt=False)
        elif selected == clear_action:
            self._controller.clear_live_points_prompt()
        elif selected == undo_action:
            self._controller.undo_live_accept()


def _preview_coords_from_event(event: Any, preview: Any | None) -> tuple[int, ...] | None:
    if preview is None:
        return None
    position = getattr(event, "position", None)
    if position is None:
        return None
    try:
        data_position = preview.world_to_data(position)
    except Exception:
        data_position = position
    data = getattr(preview, "data", None)
    shape = tuple(getattr(data, "shape", ()) or ())
    if not shape:
        return None
    coords = tuple(int(round(float(value))) for value in data_position[-len(shape) :])
    if len(coords) != len(shape):
        return None
    if any(coord < 0 or coord >= size for coord, size in zip(coords, shape, strict=False)):
        return None
    return coords


def _is_right_mouse_event(event: Any) -> bool:
    button = getattr(event, "button", None)
    if button is None:
        return False
    name = getattr(button, "name", None)
    value = str(name if name else button).lower().replace(" ", "").replace("_", "")
    return value in {"2", "right", "rightbutton", "mousebutton.right", "mousebutton.rightbutton"}


def _event_global_position(event: Any) -> Any:
    native = getattr(event, "native", None)
    if native is not None:
        for attr in ("globalPosition", "globalPos"):
            try:
                pos = getattr(native, attr)()
                if hasattr(pos, "toPoint"):
                    return pos.toPoint()
                return pos
            except Exception:
                pass
    return QCursor.pos()


def _mark_event_handled(event: Any) -> None:
    try:
        event.handled = True
    except Exception:
        pass
    native = getattr(event, "native", None)
    if native is not None:
        has_mouse_button = hasattr(native, "button") or hasattr(native, "buttons")
        if not has_mouse_button:
            return
        try:
            native.accept()
        except Exception:
            pass
