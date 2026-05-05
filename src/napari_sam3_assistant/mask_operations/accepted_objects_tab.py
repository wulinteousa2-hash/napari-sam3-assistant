from __future__ import annotations

from typing import Callable

import numpy as np
from napari.layers import Image
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .registry_service import AcceptedObjectRegistry
from .utils import accepted_metadata, preview_labels_layer_names, safe_get_layer, unique_layer_name


class AcceptedObjectsTab(QWidget):
    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.registry = AcceptedObjectRegistry(viewer)
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        self._set_combo_items(self.source_combo, preview_labels_layer_names(self.viewer))
        self._set_combo_items(self.target_combo, self.registry.layer_names())
        self.target_combo.setEnabled(self.save_mode_combo.currentText() != "Save as new layer")
        self._sync_z_controls()

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(lambda _index: self._sync_z_controls())
        self.object_name_edit = QLineEdit()
        self.object_name_edit.setPlaceholderText("object_001")
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("myelin")
        self.class_value_spin = QSpinBox()
        self.class_value_spin.setRange(1, 2_147_483_647)
        self.class_value_spin.setValue(1)

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Current slice", "current_slice")
        self.scope_combo.addItem("Z range", "z_range")
        self.scope_combo.addItem("Whole volume", "whole_volume")
        self.scope_combo.currentIndexChanged.connect(lambda _index: self._sync_z_controls())
        self.z_start_spin = QSpinBox()
        self.z_end_spin = QSpinBox()
        self.z_start_spin.setRange(0, 0)
        self.z_end_spin.setRange(0, 0)
        z_row = QHBoxLayout()
        z_row.addWidget(self.z_start_spin)
        z_row.addWidget(QLabel("to"))
        z_row.addWidget(self.z_end_spin)

        self.conversion_combo = QComboBox()
        self.conversion_combo.addItem("Convert all non-zero to class value", "nonzero_to_class")
        self.conversion_combo.addItem("Preserve source labels", "preserve")
        self.conversion_combo.addItem("Keep only listed source values", "keep_values")
        self.conversion_combo.addItem("Delete listed source values", "delete_values")
        self.source_values_edit = QLineEdit()
        self.source_values_edit.setPlaceholderText("optional: 3,7,12")

        self.save_mode_combo = QComboBox()
        self.save_mode_combo.addItems(["Save as new layer", "Append to existing accepted layer", "Replace existing accepted layer"])
        self.save_mode_combo.currentTextChanged.connect(lambda _text: self.refresh())
        self.target_combo = QComboBox()
        save_btn = QPushButton("Save Accepted Object")
        save_btn.clicked.connect(self.save_accepted_object)
        refresh_btn = QPushButton("Refresh Accepted Layers")
        refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        buttons.addWidget(save_btn)
        buttons.addWidget(refresh_btn)

        layout.addRow("Source preview layer", self.source_combo)
        layout.addRow("Object name", self.object_name_edit)
        layout.addRow("Class name", self.class_name_edit)
        layout.addRow("Class value", self.class_value_spin)
        layout.addRow("Dimensional scope", self.scope_combo)
        layout.addRow("Z range", z_row)
        layout.addRow("Class conversion", self.conversion_combo)
        layout.addRow("Source values", self.source_values_edit)
        layout.addRow("Save mode", self.save_mode_combo)
        layout.addRow("Target layer", self.target_combo)
        layout.addRow(QLabel(""), buttons)
        self.setLayout(layout)

    def save_accepted_object(self) -> None:
        source_name = self.source_combo.currentData()
        source = safe_get_layer(self.viewer, source_name)
        if source is None:
            self._log("Select a source preview Labels layer.")
            return
        object_name = self.object_name_edit.text().strip() or source_name or "accepted_object"
        class_name = self.class_name_edit.text().strip() or "class"
        class_value = int(self.class_value_spin.value())
        try:
            data = self._prepared_save_data(source, class_value)
        except ValueError as exc:
            self._log(str(exc))
            return
        metadata = accepted_metadata(
            object_name=object_name,
            class_name=class_name,
            class_value=class_value,
            source_image_layer=self._source_image_layer_name(source),
            source_preview_layer=source.name,
            source_task=str(getattr(source, "metadata", {}).get("source_task", "")),
        )
        metadata.update(
            {
                "dimensional_scope": self.scope_combo.currentData(),
                "z_range": [int(self.z_start_spin.value()), int(self.z_end_spin.value())],
                "class_conversion": self.conversion_combo.currentData(),
                "source_values": self._parse_values(self.source_values_edit.text()),
            }
        )
        mode = self.save_mode_combo.currentText()
        if mode == "Save as new layer":
            name = unique_layer_name(self.viewer, f"{class_name}_{object_name}")
            layer = self.viewer.add_labels(data, name=name, metadata=metadata)
            self._copy_layer_geometry(source, layer)
            self._log(f"Saved accepted object layer: {layer.name}")
        else:
            target_name = self.target_combo.currentData()
            target = safe_get_layer(self.viewer, target_name)
            if target is None:
                self._log("Select an accepted target layer for append or replace.")
                return
            if data.shape != target.data.shape:
                self._log(f"Shape mismatch: source {data.shape}, target {target.data.shape}.")
                return
            if mode.startswith("Append"):
                merged = np.asarray(target.data).copy()
                merged[data != 0] = data[data != 0]
                target.data = merged
            else:
                target.data = data
            self.registry.write_metadata(target, metadata)
            target.refresh()
            self._log(f"Updated accepted object layer: {target.name}")
        self.refresh()
        self._refresh_all()

    def _prepared_save_data(self, source, class_value: int) -> np.ndarray:
        source_arr = np.asarray(source.data)
        scoped, indexer = self._scope_data(source_arr)
        converted = self._convert_values(scoped, class_value)
        if indexer is Ellipsis:
            return converted.astype(np.uint32, copy=False)
        out = np.zeros(source_arr.shape, dtype=np.uint32)
        out[indexer] = converted.astype(np.uint32, copy=False)
        return out

    def _convert_values(self, arr: np.ndarray, class_value: int) -> np.ndarray:
        mode = self.conversion_combo.currentData()
        values = self._parse_values(self.source_values_edit.text())
        out = np.asarray(arr).copy()
        if mode == "nonzero_to_class":
            return np.where(out != 0, int(class_value), 0).astype(np.uint32)
        if mode == "preserve":
            return out.astype(np.uint32, copy=False)
        if not values:
            raise ValueError("Enter source label values for this conversion mode, for example: 3,7,12.")
        mask = np.isin(out, np.asarray(values, dtype=out.dtype))
        if mode == "keep_values":
            return np.where(mask, int(class_value), 0).astype(np.uint32)
        if mode == "delete_values":
            out[mask] = 0
            return out.astype(np.uint32, copy=False)
        return out.astype(np.uint32, copy=False)

    def _scope_data(self, arr: np.ndarray) -> tuple[np.ndarray, object]:
        if arr.ndim < 3:
            return arr.copy(), Ellipsis
        scope = self.scope_combo.currentData()
        z_axis = arr.ndim - 3
        if scope == "whole_volume":
            return arr.copy(), Ellipsis
        if scope == "current_slice":
            z = self._current_z(arr, z_axis)
            indexer = [slice(None)] * arr.ndim
            indexer[z_axis] = z
            return arr[tuple(indexer)].copy(), tuple(indexer)
        z0 = min(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        z1 = max(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        indexer = [slice(None)] * arr.ndim
        indexer[z_axis] = slice(z0, z1 + 1)
        return arr[tuple(indexer)].copy(), tuple(indexer)

    def _sync_z_controls(self) -> None:
        source = safe_get_layer(self.viewer, self.source_combo.currentData())
        arr = np.asarray(source.data) if source is not None else None
        has_z = arr is not None and arr.ndim >= 3
        z_max = int(arr.shape[arr.ndim - 3] - 1) if has_z else 0
        for spin in (self.z_start_spin, self.z_end_spin):
            old = min(int(spin.value()), z_max)
            spin.setRange(0, z_max)
            spin.setValue(old)
            spin.setEnabled(has_z and self.scope_combo.currentData() == "z_range")
        self.scope_combo.setEnabled(has_z)
        if has_z and self.scope_combo.currentData() == "current_slice":
            z = self._current_z(arr, arr.ndim - 3)
            self.z_start_spin.setValue(z)
            self.z_end_spin.setValue(z)
        if not has_z:
            self.scope_combo.setCurrentIndex(self.scope_combo.findData("whole_volume"))

    def _current_z(self, arr: np.ndarray, z_axis: int) -> int:
        dims = getattr(self.viewer, "dims", None)
        if dims is None:
            return 0
        try:
            return max(0, min(int(dims.current_step[z_axis]), arr.shape[z_axis] - 1))
        except Exception:
            return 0

    def _parse_values(self, text: str) -> list[int]:
        values: list[int] = []
        for token in text.replace(";", ",").replace(" ", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Label values must be integers. Invalid value: {token!r}") from exc
        return values

    def _source_image_layer_name(self, source) -> str:
        metadata = getattr(source, "metadata", {})
        if isinstance(metadata, dict) and metadata.get("source_image_layer"):
            return str(metadata["source_image_layer"])
        if self.viewer is None:
            return ""
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                return layer.name
        return ""

    def _copy_layer_geometry(self, source, target) -> None:
        for attr in ("scale", "translate", "rotate", "shear", "affine"):
            try:
                setattr(target, attr, getattr(source, attr))
            except Exception:
                pass

    def _set_combo_items(self, combo: QComboBox, names: list[str]) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name, name)
        index = combo.findData(current)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
