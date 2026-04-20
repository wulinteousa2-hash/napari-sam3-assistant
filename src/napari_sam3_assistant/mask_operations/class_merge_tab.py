from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import QAbstractItemView, QFormLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QWidget

from .merge_service import MaskMergeService
from .registry_service import AcceptedObjectRegistry
from .utils import unique_layer_name


class ClassMergeTab(QWidget):
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
        self.layer_list.clear()
        class_filter = self.class_filter_edit.text().strip()
        for name in self.registry.layer_names(class_name=class_filter):
            item = QListWidgetItem(name)
            item.setData(256, name)
            self.layer_list.addItem(item)

    def _build_ui(self) -> None:
        layout = QFormLayout()
        self.class_filter_edit = QLineEdit()
        self.class_filter_edit.setPlaceholderText("optional class-name filter")
        self.class_filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.layer_list = QListWidget()
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.output_name_edit = QLineEdit()
        self.output_name_edit.setPlaceholderText("myelin_final_working")
        merge_btn = QPushButton("Merge Accepted Objects")
        merge_btn.clicked.connect(self.merge_selected)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        buttons.addWidget(merge_btn)
        buttons.addWidget(refresh_btn)
        layout.addRow("Class filter", self.class_filter_edit)
        layout.addRow("Accepted-object layers", self.layer_list)
        layout.addRow("Output layer name", self.output_name_edit)
        layout.addRow(buttons)
        self.setLayout(layout)

    def merge_selected(self) -> None:
        names = [item.data(256) for item in self.layer_list.selectedItems()]
        try:
            data, metadata = self.merge_service.merge_accepted_objects(names)
        except ValueError as exc:
            self._log(str(exc))
            return
        base_name = self.output_name_edit.text().strip() or "class_working_mask"
        name = unique_layer_name(self.viewer, base_name)
        self.viewer.add_labels(data, name=name, metadata=metadata)
        self._log(f"Merged {len(names)} accepted object layer(s) into class mask: {name}")
        self._refresh_all()
