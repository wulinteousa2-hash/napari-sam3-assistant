from __future__ import annotations

from typing import Callable

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
from .utils import accepted_metadata, copy_labels_data, preview_labels_layer_names, safe_get_layer, unique_layer_name


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

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.source_combo = QComboBox()
        self.object_name_edit = QLineEdit()
        self.object_name_edit.setPlaceholderText("object_001")
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("myelin")
        self.class_value_spin = QSpinBox()
        self.class_value_spin.setRange(0, 2_147_483_647)
        self.class_value_spin.setValue(1)
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
        data = copy_labels_data(source)
        metadata = accepted_metadata(
            object_name=object_name,
            class_name=class_name,
            class_value=class_value,
            source_image_layer=self._source_image_layer_name(source),
            source_preview_layer=source.name,
            source_task=str(getattr(source, "metadata", {}).get("source_task", "")),
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
                merged = target.data.copy()
                merged[data != 0] = data[data != 0]
                target.data = merged
            else:
                target.data = data
            self.registry.write_metadata(target, metadata)
            target.refresh()
            self._log(f"Updated accepted object layer: {target.name}")
        self.refresh()
        self._refresh_all()

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
