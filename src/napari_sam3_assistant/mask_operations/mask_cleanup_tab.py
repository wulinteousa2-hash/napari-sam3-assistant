from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
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


class MaskCleanupTab(QWidget):
    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.analysis = ComponentAnalysisService()
        self.cleanup = MaskCleanupService()
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
        layer.data = self.cleanup.delete_components(layer.data, masks)
        layer.refresh()
        self._log(f"Deleted {len(masks)} selected component(s) from {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()
        self._refresh_all()

    def remove_small_objects(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        layer.data = self.cleanup.remove_small_objects(layer.data, self.min_size_spin.value())
        layer.refresh()
        self._log(f"Removed components smaller than {self.min_size_spin.value()} pixels from {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()

    def fill_holes(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        layer.data = self.cleanup.fill_holes(layer.data, self.hole_size_spin.value())
        layer.refresh()
        self._log(f"Filled holes up to {self.hole_size_spin.value()} pixels in {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()

    def smooth_mask(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        layer.data = self.cleanup.smooth(layer.data, self.smoothing_spin.value())
        layer.refresh()
        self._log(f"Smoothed mask {layer.name} with radius {self.smoothing_spin.value()}.")
        self.analyze_layer()
        self.refresh_unique_values()

    def keep_largest_object(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        layer.data = self.cleanup.keep_largest_object(layer.data)
        layer.refresh()
        self._log(f"Kept largest connected component in {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()

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
        layer.data, changed = self.cleanup.relabel_values(layer.data, source_values, self.new_value_spin.value())
        layer.refresh()
        self._log(f"Relabeled {changed} pixel(s) in {layer.name}.")
        self.analyze_layer()
        self.refresh_unique_values()

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

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        target_form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(lambda _index: self.refresh_unique_values())
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        analyze_btn = QPushButton("Analyze Layer")
        analyze_btn.clicked.connect(self.analyze_layer)
        delete_btn = QPushButton("Delete Selected Components")
        delete_btn.clicked.connect(self.delete_selected_components)
        target_row = QHBoxLayout()
        target_row.addWidget(refresh_btn)
        target_row.addWidget(analyze_btn)
        target_row.addWidget(delete_btn)
        target_form.addRow("Target labels layer", self.target_combo)
        target_form.addRow(target_row)
        root.addLayout(target_form)

        self.component_table = ComponentTableWidget(delete_callback=self.delete_selected_components)
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
