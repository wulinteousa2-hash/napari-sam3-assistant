from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QWidget,
)

from .merge_service import MaskMergeService
from .models import MergeOptions
from .registry_service import AcceptedObjectRegistry
from .utils import safe_get_layer, unique_layer_name


class ClassMergeTab(QWidget):
    """Merge reviewed/accepted object layers into class-level working masks."""

    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.registry = AcceptedObjectRegistry(viewer)
        self.merge_service = MaskMergeService(viewer)
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        selected = {item.data(256) for item in self.layer_list.selectedItems()}
        self.layer_list.clear()
        class_filter = self.class_filter_edit.text().strip()
        for name in self.registry.layer_names(class_name=class_filter, status="accepted"):
            item = QListWidgetItem(name)
            item.setData(256, name)
            self.layer_list.addItem(item)
            if name in selected:
                item.setSelected(True)

    def merge_selected(self) -> None:
        names = [item.data(256) for item in self.layer_list.selectedItems()]
        if not names:
            self._log("Select at least one accepted object layer to merge.")
            return
        try:
            data, metadata = self.merge_service.merge_accepted_objects(
                names,
                MergeOptions(mode="semantic", overlap_rule=self.overlap_combo.currentData()),
            )
        except ValueError as exc:
            self._log(str(exc))
            return
        base_name = self.output_name_edit.text().strip() or "class_working_mask"
        name = unique_layer_name(self.viewer, base_name)
        layer = self.viewer.add_labels(data, name=name, metadata=metadata)
        self._log(f"Merged {len(names)} accepted object layer(s) into class mask: {layer.name}")
        self._refresh_all()

    def reject_selected(self) -> None:
        names = [item.data(256) for item in self.layer_list.selectedItems()]
        if not names:
            self._log("Select accepted object layer(s) to reject.")
            return
        for name in names:
            layer = safe_get_layer(self.viewer, name)
            if layer is not None:
                self.registry.set_review_status(layer, status="rejected")
                layer.visible = False
        self._log(f"Rejected {len(names)} accepted object layer(s); they will be excluded from merge.")
        self.refresh()
        self._refresh_all()

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.class_filter_edit = QLineEdit()
        self.class_filter_edit.setPlaceholderText("optional exact class-name filter")
        self.class_filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("myelin_final_working")
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItem("Earlier selected object wins", "earlier_wins")
        self.overlap_combo.addItem("Later selected object wins", "later_wins")
        self.overlap_combo.addItem("Larger component wins", "larger_component")
        self.overlap_combo.addItem("Smaller component wins", "smaller_component")
        self.overlap_combo.addItem("Set overlap to background", "set_background")
        merge_btn = QPushButton("Merge Accepted Objects")
        merge_btn.clicked.connect(self.merge_selected)
        reject_btn = QPushButton("Reject Selected")
        reject_btn.clicked.connect(self.reject_selected)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        buttons.addWidget(merge_btn)
        buttons.addWidget(reject_btn)
        buttons.addWidget(refresh_btn)
        layout.addRow("Class filter", self.class_filter_edit)
        layout.addRow("Accepted-object layers", self.layer_list)
        layout.addRow("Overlap rule", self.overlap_combo)
        layout.addRow("Output layer name", self.output_name_edit)
        layout.addRow(buttons)
        self.setLayout(layout)
