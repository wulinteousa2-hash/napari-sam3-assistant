from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .utils import copy_layer_geometry, labels_layer_names, safe_get_layer, unique_layer_name


class ClassMergeTab(QWidget):
    """Action-first layer merge for 2D/3D Labels masks.

    This tab is for the common SAM3 workflow where repeated prompts create
    several Labels layers for the same biological class or adjacent spatial
    regions. It merges selected layers directly, without requiring accepted /
    rejected review states.
    """

    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        selected = {item.data(256) for item in self.layer_list.selectedItems()}
        self.layer_list.clear()
        text_filter = self.layer_filter_edit.text().strip().lower()
        for name in labels_layer_names(self.viewer):
            if text_filter and text_filter not in name.lower():
                continue
            item = QListWidgetItem(name)
            item.setData(256, name)
            self.layer_list.addItem(item)
            if name in selected:
                item.setSelected(True)

    def merge_selected(self) -> None:
        names = [item.data(256) for item in self.layer_list.selectedItems()]
        if not names:
            self._log("Select at least one Labels layer to merge.")
            return
        layers = [safe_get_layer(self.viewer, name) for name in names]
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            self._log("Selected Labels layers were not found.")
            return
        shape = np.asarray(layers[0].data).shape
        arrays: list[np.ndarray] = []
        for layer in layers:
            arr = np.asarray(layer.data)
            if arr.shape != shape:
                self._log(f"Layer shape mismatch: {layer.name} has {arr.shape}, expected {shape}.")
                return
            arrays.append(self._prepare_layer_array(arr))
        out = self._merge_arrays(arrays, self.overlap_combo.currentData())
        base_name = self.output_name_edit.text().strip() or "merged_class_mask"
        name = unique_layer_name(self.viewer, base_name)
        layer = self.viewer.add_labels(
            out,
            name=name,
            metadata={
                "sam3_role": "class_working_mask",
                "merge_source_layers": names,
                "merge_conversion": self.conversion_combo.currentData(),
                "class_value": int(self.class_value_spin.value()),
                "overlap_rule": self.overlap_combo.currentData(),
            },
        )
        copy_layer_geometry(layers[0], layer)
        self._log(f"Merged {len(names)} Labels layer(s) into class mask: {layer.name}")
        self._refresh_all()

    def _prepare_layer_array(self, arr: np.ndarray) -> np.ndarray:
        mode = self.conversion_combo.currentData()
        if mode == "nonzero_to_class":
            return np.where(arr != 0, int(self.class_value_spin.value()), 0).astype(np.uint32, copy=False)
        if mode == "binary":
            return (arr != 0).astype(np.uint8)
        return arr.astype(np.uint32 if arr.max(initial=0) > 65535 else np.uint16, copy=False)

    def _merge_arrays(self, arrays: list[np.ndarray], overlap_rule: str) -> np.ndarray:
        out = np.zeros(arrays[0].shape, dtype=np.uint32)
        if overlap_rule in {"earlier_wins", "class_priority"}:
            for arr in arrays:
                mask = (arr != 0) & (out == 0)
                out[mask] = arr[mask]
            return out
        if overlap_rule == "set_background":
            counts = np.zeros(arrays[0].shape, dtype=np.uint16)
            for arr in arrays:
                counts += arr != 0
            for arr in arrays:
                mask = (arr != 0) & (counts == 1)
                out[mask] = arr[mask]
            return out
        # Default: later selected layer wins.
        for arr in arrays:
            mask = arr != 0
            out[mask] = arr[mask]
        return out

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.layer_filter_edit = QLineEdit()
        self.layer_filter_edit.setPlaceholderText("optional layer-name filter")
        self.layer_filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.conversion_combo = QComboBox()
        self.conversion_combo.addItem("Convert each layer non-zero to target class value", "nonzero_to_class")
        self.conversion_combo.addItem("Preserve source label values", "preserve")
        self.conversion_combo.addItem("Binary output", "binary")
        self.class_value_spin = QSpinBox()
        self.class_value_spin.setRange(1, 2_147_483_647)
        self.class_value_spin.setValue(1)
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItem("Later selected layer wins", "later_wins")
        self.overlap_combo.addItem("Earlier selected layer wins", "earlier_wins")
        self.overlap_combo.addItem("Set overlap to background", "set_background")
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("myelin_class_mask")

        merge_btn = QPushButton("Merge Selected Layers")
        merge_btn.clicked.connect(self.merge_selected)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        buttons.addWidget(merge_btn)
        buttons.addWidget(refresh_btn)

        layout.addRow("Layer filter", self.layer_filter_edit)
        layout.addRow("Labels layers", self.layer_list)
        layout.addRow("Merge conversion", self.conversion_combo)
        layout.addRow("Target class value", self.class_value_spin)
        layout.addRow("Overlap rule", self.overlap_combo)
        layout.addRow("Output layer name", self.output_name_edit)
        layout.addRow(buttons)
        self.setLayout(layout)
