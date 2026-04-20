from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from .export_service import MaskExportService
from .merge_service import MaskMergeService
from .models import MergeOptions
from .utils import labels_layer_names, safe_get_layer, unique_layer_name


class FinalMergeExportTab(QWidget):
    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.merge_service = MaskMergeService(viewer)
        self.export_service = MaskExportService()
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        selected = {item.data(256) for item in self.layer_list.selectedItems()}
        self.layer_list.clear()
        for name in labels_layer_names(self.viewer):
            item = QListWidgetItem(name)
            item.setData(256, name)
            self.layer_list.addItem(item)
            if name in selected:
                item.setSelected(True)

    def merge_saved_objects(self):
        names = [item.data(256) for item in self.layer_list.selectedItems()]
        options = MergeOptions(mode=self.mode_combo.currentData(), overlap_rule=self.overlap_combo.currentData())
        data = self.merge_service.merge_final_masks(names, options)
        name = unique_layer_name(self.viewer, self.output_name_edit.text().strip() or "final_training_mask")
        layer = self.viewer.add_labels(
            data,
            name=name,
            metadata={
                "sam3_role": "final_training_mask",
                "merge_mode": options.mode,
                "overlap_rule": options.overlap_rule,
                "source_class_masks": names,
            },
        )
        self._log(f"Merged {len(names)} class mask layer(s) into final mask: {layer.name}")
        self._refresh_all()
        return layer

    def merge_and_export(self) -> None:
        try:
            layer = self.merge_saved_objects()
            self.export_layer(layer)
        except ValueError as exc:
            self._log(str(exc))

    def export_current_output(self) -> None:
        layer = safe_get_layer(self.viewer, self.output_name_edit.text().strip())
        if layer is None:
            self._log("Output layer was not found. Merge first or choose an existing output name.")
            return
        self.export_layer(layer)

    def export_layer(self, layer) -> None:
        path = self.export_path_edit.text().strip()
        if not path:
            self._log("Choose an export path first.")
            return
        try:
            target = self.export_service.export(layer.data, path, self.format_combo.currentText())
        except ValueError as exc:
            self._log(str(exc))
            return
        self._log(f"Exported mask '{layer.name}' to {target}")

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Semantic", "semantic")
        self.mode_combo.addItem("Instance", "instance")
        self.mode_combo.addItem("Binary", "binary")
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItem("Class priority order wins", "class_priority")
        self.overlap_combo.addItem("Earlier selected layer wins", "earlier_wins")
        self.overlap_combo.addItem("Larger component wins", "larger_component")
        self.overlap_combo.addItem("Smaller component wins", "smaller_component")
        self.overlap_combo.addItem("Set overlap to background", "set_background")
        self.output_name_edit = QLineEdit("final_training_mask")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["TIFF", "PNG", "NumPy (.npy)"])
        self.export_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_export_path)
        path_row = QHBoxLayout()
        path_row.addWidget(self.export_path_edit)
        path_row.addWidget(browse_btn)
        merge_btn = QPushButton("Merge Saved Objects")
        merge_btn.clicked.connect(lambda: self._run_with_log_errors(self.merge_saved_objects))
        merge_export_btn = QPushButton("Merge and Export")
        merge_export_btn.clicked.connect(self.merge_and_export)
        export_btn = QPushButton("Export Output")
        export_btn.clicked.connect(self.export_current_output)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        actions = QHBoxLayout()
        actions.addWidget(merge_btn)
        actions.addWidget(merge_export_btn)
        actions.addWidget(export_btn)
        actions.addWidget(refresh_btn)
        layout.addRow("Cleaned class masks", self.layer_list)
        layout.addRow("Merge mode", self.mode_combo)
        layout.addRow("Overlap rule", self.overlap_combo)
        layout.addRow("Output layer name", self.output_name_edit)
        layout.addRow("Export format", self.format_combo)
        layout.addRow("Export path", path_row)
        layout.addRow(actions)
        self.setLayout(layout)

    def _browse_export_path(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export mask", "", "Masks (*.tif *.tiff *.png *.npy)")
        if path:
            self.export_path_edit.setText(path)

    def _run_with_log_errors(self, callback) -> None:
        try:
            callback()
        except ValueError as exc:
            self._log(str(exc))
