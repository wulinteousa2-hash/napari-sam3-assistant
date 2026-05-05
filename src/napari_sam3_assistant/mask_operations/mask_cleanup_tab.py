from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
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
    """Friction-light cleanup for 2D/3D binary, semantic, and instance masks."""

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
        self._last_analysis_indexer: object = Ellipsis
        self._last_analysis_offset: tuple[int, ...] | None = None
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
        self._sync_scope_controls()
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_delete_callback()

    def analyze_layer(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer for component analysis.")
            return
        sub, indexer, offset = self._scoped_data(layer)
        records = self.analysis.analyze(sub)
        self._last_analysis_indexer = indexer
        self._last_analysis_offset = offset
        self.component_table.set_records(records)
        scope = self._scope_label(layer)
        self._log(f"Analyzed {len(records)} connected component(s) in {layer.name} ({scope}).")

    def delete_selected_components(self) -> None:
        layer = self._target_layer()
        ids = self.component_table.selected_component_ids()
        if layer is None or not ids:
            self._log("Select component rows to delete.")
            return
        masks = self.analysis.component_masks(ids)
        sub, indexer, _offset = self._scoped_data(layer)
        try:
            cleaned = self.cleanup.delete_components(sub, masks)
        except ValueError as exc:
            self._log(str(exc) + " Re-run Analyze Layer after changing scope or target layer.")
            return
        if self._replace_scoped_layer_data(layer, cleaned, indexer, "delete selected components"):
            self._log(f"Deleted {len(masks)} selected component(s) from {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Selected component delete made no mask changes.")

    def remove_small_objects(self) -> None:
        self._apply_scoped_cleanup(
            lambda sub: self.cleanup.remove_small_objects(sub, self.min_size_spin.value()),
            f"Removed components smaller than {self.min_size_spin.value()} pixels/voxels",
            "remove small objects",
        )

    def fill_holes(self) -> None:
        self._apply_scoped_cleanup(
            lambda sub: self.cleanup.fill_holes(sub, self.hole_size_spin.value()),
            f"Filled holes up to {self.hole_size_spin.value()} pixels/voxels",
            "fill holes",
        )

    def smooth_mask(self) -> None:
        self._apply_scoped_cleanup(
            lambda sub: self.cleanup.smooth(sub, self.smoothing_spin.value()),
            f"Smoothed mask with radius {self.smoothing_spin.value()}",
            "smooth mask",
        )

    def keep_largest_object(self) -> None:
        self._apply_scoped_cleanup(
            self.cleanup.keep_largest_object,
            "Kept largest connected component",
            "keep largest object",
        )

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
        sub, indexer, _offset = self._scoped_data(layer)
        data, changed = self.cleanup.relabel_values(sub, source_values, self.new_value_spin.value())
        if changed and self._replace_scoped_layer_data(layer, data, indexer, "relabel values"):
            self._log(f"Relabeled {changed} pixel(s)/voxel(s) in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
        else:
            self._log("Relabel made no mask changes.")

    def change_selected_values(self) -> None:
        values = self._selected_unique_values()
        if not values:
            self._log("Select one or more label values in the value table.")
            return
        self.values_to_replace_edit.setText(",".join(str(value) for value in values))
        self.apply_relabel()

    def delete_selected_values(self) -> None:
        values = self._selected_unique_values()
        if not values:
            self._log("Select one or more label values to delete.")
            return
        layer = self._target_layer()
        if layer is None:
            return
        sub, indexer, _offset = self._scoped_data(layer)
        data, changed = self.cleanup.delete_values(sub, values)
        if changed and self._replace_scoped_layer_data(layer, data, indexer, "delete selected values"):
            self._log(f"Deleted label value(s) {values} from {layer.name} ({changed} pixel(s)/voxel(s)).")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Delete selected values made no mask changes.")

    def keep_selected_values_only(self) -> None:
        values = self._selected_unique_values()
        if not values:
            self._log("Select one or more label values to keep.")
            return
        layer = self._target_layer()
        if layer is None:
            return
        sub, indexer, _offset = self._scoped_data(layer)
        data, changed = self.cleanup.keep_values(sub, values)
        if changed and self._replace_scoped_layer_data(layer, data, indexer, "keep selected values only"):
            self._log(f"Kept only label value(s) {values} in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Keep selected values made no mask changes.")

    def convert_nonzero_to_new_value(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        data, changed = self.cleanup.convert_nonzero_to_value(sub, self.new_value_spin.value())
        if changed and self._replace_scoped_layer_data(layer, data, indexer, "convert non-zero to class"):
            self._log(f"Converted non-zero labels to class value {self.new_value_spin.value()} in {layer.name}.")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Convert non-zero made no mask changes.")

    def refresh_unique_values(self) -> None:
        layer = self._target_layer()
        self.unique_values_table.setRowCount(0)
        if layer is None:
            return
        sub, _indexer, _offset = self._scoped_data(layer)
        values, counts = np.unique(sub, return_counts=True)
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
        sub, indexer, _offset = self._scoped_data(layer)
        data, removed = self.cleanup.delete_values(sub, [label_value])
        if removed == 0:
            return
        if self._replace_scoped_layer_data(layer, data, indexer, "right-click delete"):
            self._log(
                f"Removed label value {label_value} from '{layer.name}' in {self._scope_label(layer)} "
                f"({removed} pixel(s)/voxel(s))."
            )
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
        root.setSpacing(6)
        target_form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_layer_changed)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Current slice", "current_slice")
        self.scope_combo.addItem("Z range", "z_range")
        self.scope_combo.addItem("Whole volume", "whole_volume")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        self.z_start_spin = QSpinBox()
        self.z_end_spin = QSpinBox()
        self.z_start_spin.valueChanged.connect(lambda _value: self.refresh_unique_values())
        self.z_end_spin.valueChanged.connect(lambda _value: self.refresh_unique_values())
        z_row = QHBoxLayout()
        z_row.addWidget(self.z_start_spin)
        z_row.addWidget(QLabel("to"))
        z_row.addWidget(self.z_end_spin)
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
            "Right-click a label object in the selected target Labels layer to remove that label value within the selected scope."
        )
        self.right_click_delete_check.toggled.connect(lambda _checked: self._sync_mouse_delete_callback())
        target_row = QHBoxLayout()
        target_row.addWidget(refresh_btn)
        target_row.addWidget(analyze_btn)
        target_row.addWidget(delete_btn)
        target_row.addWidget(self.undo_btn)
        target_row.addWidget(self.right_click_delete_check)
        target_form.addRow("Target labels layer", self.target_combo)
        target_form.addRow("Operation scope", self.scope_combo)
        target_form.addRow("Z range", z_row)
        target_form.addRow(target_row)
        root.addLayout(target_form)

        self.component_table = ComponentTableWidget(
            delete_callback=self.delete_selected_components,
            locate_callback=self.locate_component,
        )
        self.component_table.setMinimumHeight(170)
        root.addWidget(self.component_table)

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
        operations = QGridLayout()
        operations.setHorizontalSpacing(8)
        operations.setVerticalSpacing(5)
        self._add_operation_row(operations, 0, "Minimum size", self.min_size_spin, remove_small_btn)
        self._add_operation_row(operations, 1, "Hole size", self.hole_size_spin, fill_btn)
        self._add_operation_row(operations, 2, "Smoothing radius", self.smoothing_spin, smooth_btn)
        operations.addWidget(keep_btn, 3, 2)
        operations.setColumnStretch(2, 1)
        root.addLayout(operations)

        self.unique_values_table = QTableWidget(0, 2)
        self.unique_values_table.setObjectName("maskValueTable")
        self.unique_values_table.setHorizontalHeaderLabels(["Value", "Pixels/Voxels"])
        self.unique_values_table.setAlternatingRowColors(True)
        self.unique_values_table.verticalHeader().setDefaultSectionSize(24)
        self.unique_values_table.verticalHeader().setMinimumSectionSize(22)
        self.unique_values_table.setMaximumHeight(170)
        self.unique_values_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.unique_values_table.setStyleSheet(
            """
            QTableWidget#maskValueTable {
                background: #1f242c;
                alternate-background-color: #2a3038;
                color: #eef2f7;
                gridline-color: #3b444f;
                selection-background-color: #2f6f8f;
                selection-color: #ffffff;
            }
            QTableWidget#maskValueTable::item {
                padding: 3px 4px;
            }
            QTableWidget#maskValueTable::item:selected {
                background: #2f6f8f;
                color: #ffffff;
            }
            """
        )
        root.addWidget(self.unique_values_table)
        relabel_form = QFormLayout()
        self.values_to_replace_edit = QLineEdit()
        self.values_to_replace_edit.setPlaceholderText("1,2,3,4")
        self.new_value_spin = QSpinBox()
        self.new_value_spin.setRange(0, 2_147_483_647)
        self.new_value_spin.setValue(1)
        apply_btn = QPushButton("Apply Relabel")
        apply_btn.clicked.connect(self.apply_relabel)
        selected_btn = QPushButton("Assign Selected To")
        selected_btn.clicked.connect(self.change_selected_values)
        delete_values_btn = QPushButton("Delete Selected Values")
        delete_values_btn.clicked.connect(self.delete_selected_values)
        keep_values_btn = QPushButton("Keep Selected Only")
        keep_values_btn.clicked.connect(self.keep_selected_values_only)
        convert_btn = QPushButton("Convert Non-zero To Class")
        convert_btn.clicked.connect(self.convert_nonzero_to_new_value)
        relabel_form.addRow("Values to replace", self.values_to_replace_edit)
        relabel_form.addRow("New class/value", self.new_value_spin)
        relabel_buttons = QHBoxLayout()
        relabel_buttons.addWidget(apply_btn)
        relabel_buttons.addWidget(selected_btn)
        relabel_buttons.addWidget(delete_values_btn)
        relabel_buttons.addWidget(keep_values_btn)
        relabel_buttons.addWidget(convert_btn)
        relabel_form.addRow(relabel_buttons)
        root.addLayout(relabel_form)
        self.setLayout(root)

    def _add_operation_row(self, layout: QGridLayout, row: int, label_text: str, spin_box: QSpinBox, button: QPushButton) -> None:
        label = QLabel(label_text)
        spin_box.setMinimumWidth(84)
        spin_box.setMaximumWidth(140)
        button.setMinimumWidth(180)
        layout.addWidget(label, row, 0)
        layout.addWidget(spin_box, row, 1)
        layout.addWidget(button, row, 2)

    def _apply_scoped_cleanup(self, callback, success_prefix: str, action: str) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        data = callback(sub)
        if self._replace_scoped_layer_data(layer, data, indexer, action):
            self._log(f"{success_prefix} in {layer.name} ({self._scope_label(layer)}).")
            self.analyze_layer()
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log(f"{action} made no mask changes.")

    def _target_layer(self):
        return safe_get_layer(self.viewer, self.target_combo.currentData())

    def _on_target_layer_changed(self, _index: int) -> None:
        self._sync_scope_controls()
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_delete_callback()
        self._update_undo_state()

    def _on_scope_changed(self, _index: int) -> None:
        self._sync_scope_controls()
        self.refresh_unique_values()

    def _sync_scope_controls(self) -> None:
        layer = self._target_layer()
        data = np.asarray(layer.data) if layer is not None else None
        has_z = data is not None and data.ndim >= 3
        z_axis = data.ndim - 3 if has_z else 0
        z_max = int(data.shape[z_axis] - 1) if has_z else 0
        for spin in (self.z_start_spin, self.z_end_spin):
            old = min(int(spin.value()), z_max)
            spin.blockSignals(True)
            spin.setRange(0, z_max)
            spin.setValue(old)
            spin.setEnabled(has_z and self.scope_combo.currentData() == "z_range")
            spin.blockSignals(False)
        self.scope_combo.setEnabled(has_z)
        if not has_z:
            idx = self.scope_combo.findData("whole_volume")
            if idx >= 0:
                self.scope_combo.setCurrentIndex(idx)
        elif self.scope_combo.currentData() == "current_slice":
            z = self._current_z(data, z_axis)
            self.z_start_spin.setValue(z)
            self.z_end_spin.setValue(z)

    def _scoped_data(self, layer) -> tuple[np.ndarray, object, tuple[int, ...]]:
        arr = np.asarray(layer.data)
        if arr.ndim < 3:
            return arr.copy(), Ellipsis, tuple(0 for _ in range(arr.ndim))
        scope = self.scope_combo.currentData()
        z_axis = arr.ndim - 3
        if scope == "whole_volume":
            return arr.copy(), Ellipsis, tuple(0 for _ in range(arr.ndim))
        if scope == "current_slice":
            z = self._current_z(arr, z_axis)
            indexer = [slice(None)] * arr.ndim
            indexer[z_axis] = z
            offset = [0] * arr.ndim
            offset[z_axis] = z
            reduced_offset = [v for axis, v in enumerate(offset) if axis != z_axis]
            return arr[tuple(indexer)].copy(), tuple(indexer), tuple(reduced_offset)
        z0 = min(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        z1 = max(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        indexer = [slice(None)] * arr.ndim
        indexer[z_axis] = slice(z0, z1 + 1)
        offset = [0] * arr.ndim
        offset[z_axis] = z0
        return arr[tuple(indexer)].copy(), tuple(indexer), tuple(offset)

    def _replace_scoped_layer_data(self, layer, scoped_data: np.ndarray, indexer: object, action: str) -> bool:
        current = np.asarray(layer.data)
        if indexer is Ellipsis:
            updated = np.asarray(scoped_data)
        else:
            updated = current.copy()
            updated[indexer] = scoped_data
        return self._replace_layer_data(layer, updated, action)

    def _current_z(self, arr: np.ndarray, z_axis: int) -> int:
        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return 0
        try:
            return max(0, min(int(dims.current_step[z_axis]), arr.shape[z_axis] - 1))
        except Exception:
            return 0

    def _scope_label(self, layer) -> str:
        data = np.asarray(layer.data)
        if data.ndim < 3:
            return "2D layer"
        scope = self.scope_combo.currentData()
        z_axis = data.ndim - 3
        if scope == "whole_volume":
            return "whole volume"
        if scope == "current_slice":
            return f"Z={self._current_z(data, z_axis)}"
        z0 = min(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        z1 = max(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        return f"Z={z0}-{z1}"

    def _selected_unique_values(self) -> list[int]:
        values: list[int] = []
        for index in self.unique_values_table.selectionModel().selectedRows():
            item = self.unique_values_table.item(index.row(), 0)
            if item is not None:
                values.append(int(item.text()))
        return values

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
        self._update_undo_state()
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
        if self._last_analysis_offset is not None and len(self._last_analysis_offset) == len(data_position):
            data_position = data_position + np.asarray(self._last_analysis_offset, dtype=float)
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
