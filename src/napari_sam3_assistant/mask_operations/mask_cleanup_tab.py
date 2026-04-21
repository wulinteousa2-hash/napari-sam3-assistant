from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .cleanup_service import MaskCleanupService
from .component_analysis_service import ComponentAnalysisService
from .component_table_widget import ComponentTableWidget
from .utils import labels_layer_names, safe_get_layer


UNDO_HISTORY_LIMIT = 20


class MaskCleanupTab(QWidget):
    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.analysis = ComponentAnalysisService()
        self.cleanup = MaskCleanupService()
        self._mouse_layer = None
        self._mouse_callback = self._delete_label_on_right_click
        self._undo_history: dict[int, list[np.ndarray]] = {}
        self._last_layer_data: dict[int, np.ndarray] = {}
        self._tracked_layer = None
        self._history_callback = self._on_tracked_layer_data_changed
        self._suppress_history_event = False
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        current = self.target_combo.currentData()
        self.target_combo.clear()
        for name in labels_layer_names(self.viewer):
            self.target_combo.addItem(name, name)
        index = self.target_combo.findData(current)
        if index >= 0:
            self.target_combo.setCurrentIndex(index)
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_delete_callback()

    def analyze_layer(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer for component analysis.")
            return
        records = self.analysis.analyze(layer.data)
        self.component_table.set_records(records)
        self._log(f"Analyzed {len(records)} connected component(s) in {layer.name}.")

    def delete_selected_components(self) -> None:
        layer = self._target_layer()
        ids = self.component_table.selected_component_ids()
        if layer is None or not ids:
            self._log("Select component rows to delete.")
            return
        masks = self.analysis.component_masks(ids)
        data = self.cleanup.delete_components(layer.data, masks)
        if self._replace_layer_data(layer, data, "delete selected components"):
            self._log(f"Deleted {len(masks)} selected component(s) from {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Selected component delete made no mask changes.")

    def remove_small_objects(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        data = self.cleanup.remove_small_objects(layer.data, self.min_size_spin.value())
        if self._replace_layer_data(layer, data, "remove small objects"):
            self._log(f"Removed components smaller than {self.min_size_spin.value()} pixels from {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Remove small objects made no mask changes.")

    def fill_holes(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        data = self.cleanup.fill_holes(layer.data, self.hole_size_spin.value())
        if self._replace_layer_data(layer, data, "fill holes"):
            self._log(f"Filled holes up to {self.hole_size_spin.value()} pixels in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Fill holes made no mask changes.")

    def smooth_mask(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        data = self.cleanup.smooth(layer.data, self.smoothing_spin.value())
        if self._replace_layer_data(layer, data, "smooth mask"):
            self._log(f"Smoothed mask {layer.name} with radius {self.smoothing_spin.value()}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Smooth mask made no mask changes.")

    def keep_largest_object(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        data = self.cleanup.keep_largest_object(layer.data)
        if self._replace_layer_data(layer, data, "keep largest object"):
            self._log(f"Kept largest connected component in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Keep largest object made no mask changes.")

    def apply_relabel(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        try:
            source_values = self._parse_values(self.values_to_replace_edit.text())
        except ValueError as exc:
            self._log(str(exc))
            return
        data, changed = self.cleanup.relabel_values(layer.data, source_values, self.new_value_spin.value())
        if changed and self._replace_layer_data(layer, data, "relabel values"):
            self._log(f"Relabeled {changed} pixel(s) in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Relabel made no mask changes.")

    def change_selected_values(self) -> None:
        values: list[int] = []
        for index in self.unique_values_table.selectionModel().selectedRows():
            item = self.unique_values_table.item(index.row(), 0)
            if item is not None:
                values.append(int(item.text()))
        self.values_to_replace_edit.setText(",".join(str(value) for value in values))
        self.apply_relabel()

    def refresh_unique_values(self) -> None:
        layer = self._target_layer()
        self.unique_values_table.setRowCount(0)
        if layer is None:
            return
        values, counts = np.unique(layer.data, return_counts=True)
        for value, count in zip(values, counts, strict=False):
            if int(value) == 0:
                continue
            row = self.unique_values_table.rowCount()
            self.unique_values_table.insertRow(row)
            self.unique_values_table.setItem(row, 0, QTableWidgetItem(str(int(value))))
            self.unique_values_table.setItem(row, 1, QTableWidgetItem(str(int(count))))

    def _sync_mouse_delete_callback(self) -> None:
        self._disconnect_mouse_delete_callback()
        if not getattr(self, "right_click_delete_check", None):
            return
        if not self.right_click_delete_check.isChecked():
            return
        layer = self._target_layer()
        if layer is None:
            return
        callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if callbacks is None:
            self._log("Selected Labels layer does not expose mouse callbacks.")
            return
        if self._mouse_callback not in callbacks:
            callbacks.append(self._mouse_callback)
        self._mouse_layer = layer
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "pick"
        except Exception:
            pass
        self._log(f"Right-click delete armed for '{layer.name}'.")

    def _disconnect_mouse_delete_callback(self) -> None:
        layer = self._mouse_layer
        if layer is None:
            return
        callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if callbacks is not None and self._mouse_callback in callbacks:
            callbacks.remove(self._mouse_callback)
        self._mouse_layer = None

    def _delete_label_on_right_click(self, layer, event):
        if not self.right_click_delete_check.isChecked():
            return
        if layer is not self._target_layer():
            return
        if not self._is_right_click(event):
            return
        label_value = self._label_value_at_event(layer, event)
        if label_value <= 0:
            self._log("Right-clicked background; no label removed.")
            return
        data = np.asarray(layer.data).copy()
        removed = int(np.count_nonzero(data == label_value))
        if removed == 0:
            return
        data[data == label_value] = 0
        if self._replace_layer_data(layer, data, "right-click delete"):
            self._log(f"Removed label value {label_value} from '{layer.name}' ({removed} pixel(s)).")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()

    def _is_right_click(self, event) -> bool:
        button = getattr(event, "button", None)
        return button in {2, "right", "Right", "right_button"}

    def _label_value_at_event(self, layer, event) -> int:
        position = getattr(event, "position", None)
        if position is None:
            return 0
        try:
            data_position = layer.world_to_data(position)
        except Exception:
            data_position = position
        data = np.asarray(layer.data)
        coords = tuple(int(round(float(value))) for value in data_position[-data.ndim :])
        if len(coords) != data.ndim:
            return 0
        for coord, size in zip(coords, data.shape, strict=False):
            if coord < 0 or coord >= size:
                return 0
        return int(data[coords])

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        target_form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_layer_changed)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        analyze_btn = QPushButton("Analyze Layer")
        analyze_btn.clicked.connect(self.analyze_layer)
        delete_btn = QPushButton("Delete Selected Components")
        delete_btn.clicked.connect(self.delete_selected_components)
        self.undo_btn = QPushButton("Undo Last Edit")
        self.undo_btn.setToolTip("Restore the selected Labels layer to its previous Mask Cleanup state.")
        self.undo_btn.clicked.connect(self.undo_last_edit)
        self.undo_btn.setEnabled(False)
        self.right_click_delete_check = QCheckBox("Right-click Delete")
        self.right_click_delete_check.setToolTip(
            "Right-click a label object in the selected target Labels layer to remove that label value."
        )
        self.right_click_delete_check.toggled.connect(
            lambda _checked: self._sync_mouse_delete_callback()
        )
        target_row = QHBoxLayout()
        target_row.addWidget(refresh_btn)
        target_row.addWidget(analyze_btn)
        target_row.addWidget(delete_btn)
        target_row.addWidget(self.undo_btn)
        target_row.addWidget(self.right_click_delete_check)
        target_form.addRow("Target labels layer", self.target_combo)
        target_form.addRow(target_row)
        root.addLayout(target_form)

        self.component_table = ComponentTableWidget(
            delete_callback=self.delete_selected_components,
            locate_callback=self.locate_component,
        )
        root.addWidget(self.component_table)

        operations = QFormLayout()
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 2_147_483_647)
        self.min_size_spin.setValue(64)
        remove_small_btn = QPushButton("Remove Small Objects")
        remove_small_btn.clicked.connect(self.remove_small_objects)
        self.hole_size_spin = QSpinBox()
        self.hole_size_spin.setRange(0, 2_147_483_647)
        self.hole_size_spin.setValue(256)
        fill_btn = QPushButton("Fill Holes")
        fill_btn.clicked.connect(self.fill_holes)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(1, 10)
        self.smoothing_spin.setValue(1)
        smooth_btn = QPushButton("Smooth Mask")
        smooth_btn.clicked.connect(self.smooth_mask)
        keep_btn = QPushButton("Keep Largest Object")
        keep_btn.clicked.connect(self.keep_largest_object)
        operations.addRow("Minimum size", self.min_size_spin)
        operations.addRow(remove_small_btn)
        operations.addRow("Hole size", self.hole_size_spin)
        operations.addRow(fill_btn)
        operations.addRow("Smoothing radius", self.smoothing_spin)
        operations.addRow(smooth_btn)
        operations.addRow(keep_btn)
        root.addLayout(operations)

        self.unique_values_table = QTableWidget(0, 2)
        self.unique_values_table.setHorizontalHeaderLabels(["Value", "Pixels"])
        root.addWidget(self.unique_values_table)
        relabel_form = QFormLayout()
        self.values_to_replace_edit = QLineEdit()
        self.values_to_replace_edit.setPlaceholderText("1,2,3,4")
        self.new_value_spin = QSpinBox()
        self.new_value_spin.setRange(0, 2_147_483_647)
        self.new_value_spin.setValue(1)
        apply_btn = QPushButton("Apply Relabel")
        apply_btn.clicked.connect(self.apply_relabel)
        selected_btn = QPushButton("Change Selected To")
        selected_btn.clicked.connect(self.change_selected_values)
        relabel_form.addRow("Values to replace", self.values_to_replace_edit)
        relabel_form.addRow("New value", self.new_value_spin)
        relabel_buttons = QHBoxLayout()
        relabel_buttons.addWidget(apply_btn)
        relabel_buttons.addWidget(selected_btn)
        relabel_form.addRow(relabel_buttons)
        root.addLayout(relabel_form)
        self.setLayout(root)

    def _target_layer(self):
        return safe_get_layer(self.viewer, self.target_combo.currentData())

    def _on_target_layer_changed(self, _index: int) -> None:
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_delete_callback()
        self._update_undo_state()

    def undo_last_edit(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer to undo.")
            return
        history = self._undo_history.get(id(layer), [])
        if not history:
            self._log(f"No Mask Cleanup undo history for {layer.name}.")
            self._update_undo_state()
            return
        previous = history.pop()
        self._suppress_history_event = True
        try:
            layer.data = previous
            layer.refresh()
            self._last_layer_data[id(layer)] = np.asarray(layer.data).copy()
        finally:
            self._suppress_history_event = False
        self._log(f"Undid last Mask Cleanup edit on {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()
        self._refresh_all()
        self._update_undo_state()

    def _replace_layer_data(self, layer, data, action: str) -> bool:
        current = np.asarray(layer.data)
        updated = np.asarray(data)
        if current.shape == updated.shape and np.array_equal(current, updated):
            return False
        self._push_undo_state(layer, action)
        self._suppress_history_event = True
        try:
            layer.data = updated
            layer.refresh()
            self._last_layer_data[id(layer)] = np.asarray(layer.data).copy()
        finally:
            self._suppress_history_event = False
        return True

    def _push_undo_state(self, layer, action: str) -> None:
        self._append_undo_state(layer, np.asarray(layer.data).copy(), action)

    def _append_undo_state(self, layer, data: np.ndarray, action: str) -> None:
        history = self._undo_history.setdefault(id(layer), [])
        history.append(np.asarray(data).copy())
        if len(history) > UNDO_HISTORY_LIMIT:
            del history[0 : len(history) - UNDO_HISTORY_LIMIT]
        self._log(f"Saved undo point for {layer.name}: {action}.")
        self._update_undo_state()

    def _update_undo_state(self) -> None:
        if not hasattr(self, "undo_btn"):
            return
        layer = self._target_layer()
        has_history = bool(layer is not None and self._undo_history.get(id(layer)))
        self.undo_btn.setEnabled(has_history)

    def _track_target_layer(self) -> None:
        layer = self._target_layer()
        if layer is self._tracked_layer:
            self._update_undo_state()
            return
        self._disconnect_history_layer()
        self._tracked_layer = layer
        if layer is None:
            self._update_undo_state()
            return
        self._last_layer_data[id(layer)] = np.asarray(layer.data).copy()
        events = getattr(layer, "events", None)
        data_event = getattr(events, "data", None)
        connect = getattr(data_event, "connect", None)
        if connect is not None:
            try:
                connect(self._history_callback)
            except Exception:
                pass
        self._update_undo_state()

    def _disconnect_history_layer(self) -> None:
        layer = self._tracked_layer
        if layer is None:
            return
        events = getattr(layer, "events", None)
        data_event = getattr(events, "data", None)
        disconnect = getattr(data_event, "disconnect", None)
        if disconnect is not None:
            try:
                disconnect(self._history_callback)
            except Exception:
                pass
        self._tracked_layer = None

    def _on_tracked_layer_data_changed(self, _event=None) -> None:
        if self._suppress_history_event:
            return
        layer = self._tracked_layer
        if layer is None:
            return
        layer_id = id(layer)
        previous = self._last_layer_data.get(layer_id)
        current = np.asarray(layer.data).copy()
        if previous is None:
            self._last_layer_data[layer_id] = current
            return
        if previous.shape == current.shape and np.array_equal(previous, current):
            return
        self._append_undo_state(layer, previous, "manual label edit")
        self._last_layer_data[layer_id] = current

    def locate_component(self, component_id: int) -> None:
        layer = self._target_layer()
        mask = self.analysis.component_mask(component_id)
        if layer is None or mask is None:
            self._log("Analyze a target Labels layer before locating a component.")
            return
        coords = np.argwhere(mask)
        if coords.size == 0:
            self._log(f"Component {component_id} is empty or no longer exists.")
            return
        data_position = coords.mean(axis=0)
        label_value = self._label_value_near(layer, data_position)
        self._center_view_on_data_position(layer, data_position)
        detail = f" label {label_value}" if label_value > 0 else ""
        self._log(f"Located component {component_id}{detail} at centroid {self._format_position(data_position)}.")

    def _center_view_on_data_position(self, layer, data_position: np.ndarray) -> None:
        data_tuple = tuple(float(value) for value in data_position)
        try:
            world_position = tuple(float(value) for value in layer.data_to_world(data_tuple))
        except Exception:
            world_position = data_tuple

        dims = getattr(self.viewer, "dims", None)
        displayed = self._displayed_axes(dims, len(world_position))
        if dims is not None:
            for axis, value in enumerate(data_position):
                if axis not in displayed:
                    self._set_dim_step(dims, axis, int(round(float(value))))

        camera = getattr(self.viewer, "camera", None)
        if camera is not None:
            center = tuple(world_position[axis] for axis in displayed if axis < len(world_position))
            if center:
                try:
                    camera.center = center
                except Exception:
                    pass

        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "pick"
        except Exception:
            pass

    def _displayed_axes(self, dims, ndim: int) -> tuple[int, ...]:
        if dims is not None:
            displayed = getattr(dims, "displayed", None)
            if displayed is not None:
                return tuple(int(axis) for axis in displayed)
            ndisplay = int(getattr(dims, "ndisplay", 2))
        else:
            ndisplay = 2
        return tuple(range(max(0, ndim - ndisplay), ndim))

    def _set_dim_step(self, dims, axis: int, value: int) -> None:
        try:
            dims.set_current_step(axis, value)
            return
        except Exception:
            pass
        try:
            current_step = list(dims.current_step)
            if axis < len(current_step):
                current_step[axis] = value
                dims.current_step = tuple(current_step)
        except Exception:
            pass

    def _label_value_near(self, layer, data_position: np.ndarray) -> int:
        data = np.asarray(layer.data)
        coords = tuple(int(round(float(value))) for value in data_position[-data.ndim :])
        if len(coords) != data.ndim:
            return 0
        for coord, size in zip(coords, data.shape, strict=False):
            if coord < 0 or coord >= size:
                return 0
        return int(data[coords])

    def _format_position(self, data_position: np.ndarray) -> str:
        return ", ".join(f"{float(value):.1f}" for value in data_position)

    def _parse_values(self, text: str) -> list[int]:
        values = []
        for token in text.replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Label values must be integers. Invalid value: {token!r}") from exc
        return values
