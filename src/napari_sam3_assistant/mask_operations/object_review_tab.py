from __future__ import annotations

from typing import Callable

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .registry_service import AcceptedObjectRegistry
from .utils import labels_layer_names, safe_get_layer


class NumericItem(QTableWidgetItem):
    def __lt__(self, other) -> bool:
        left = self.data(Qt.UserRole)
        right = other.data(Qt.UserRole)
        if left is not None and right is not None:
            return left < right
        return super().__lt__(other)


class ObjectReviewTab(QWidget):
    """Fast human triage for SAM3-generated object layers.

    This tab handles the common case where SAM3 creates many mask/object layers
    and the user needs to quickly accept good objects, reject wrong objects,
    assign/correct class identity, delete obvious failures, and jump to objects
    in napari before downstream cleanup/merge.
    """

    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.registry = AcceptedObjectRegistry(viewer)
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        selected_names = set(self._selected_layer_names())
        records = self.registry.review_records(include_final_outputs=self.include_outputs_check.isChecked())
        status_filter = self.status_filter_combo.currentData()
        class_filter = self.class_filter_edit.text().strip().lower()
        text_filter = self.text_filter_edit.text().strip().lower()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for record in records:
            if status_filter and record.review_status != status_filter:
                continue
            if class_filter and record.class_name.lower() != class_filter:
                continue
            if text_filter and text_filter not in record.layer_name.lower() and text_filter not in record.object_name.lower():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            self._set_text(row, 0, record.layer_name, record.layer_name)
            self._set_text(row, 1, record.review_status, record.layer_name)
            self._set_text(row, 2, record.class_name or "—", record.layer_name)
            self._set_numeric(row, 3, record.class_value, record.layer_name)
            self._set_numeric(row, 4, record.nonzero_count, record.layer_name)
            self._set_numeric(row, 5, record.ndim, record.layer_name)
            self._set_text(row, 6, record.shape_text, record.layer_name)
            self._set_text(row, 7, record.sam3_role or "—", record.layer_name)
            if record.layer_name in selected_names:
                self.table.selectRow(row)
        self.table.setSortingEnabled(True)
        self._sync_summary()

    def accept_selected(self) -> None:
        self._set_selected_status("accepted")

    def reject_selected(self) -> None:
        self._set_selected_status("rejected")

    def mark_needs_edit_selected(self) -> None:
        self._set_selected_status("needs_edit")

    def clear_review_selected(self) -> None:
        self._set_selected_status("unreviewed")

    def delete_selected_layers(self) -> None:
        names = self._selected_layer_names()
        if not names:
            self._log("Select one or more mask layers to delete.")
            return
        deleted = 0
        for name in names:
            layer = safe_get_layer(self.viewer, name)
            if layer is None:
                continue
            try:
                self.viewer.layers.remove(layer)
                deleted += 1
            except Exception as exc:
                self._log(f"Could not delete layer {name}: {exc}")
        self._log(f"Deleted {deleted} selected mask layer(s).")
        self.refresh()
        self._refresh_all()

    def apply_class_to_selected(self) -> None:
        names = self._selected_layer_names()
        if not names:
            self._log("Select one or more mask layers before assigning a class.")
            return
        class_name = self.class_name_edit.text().strip()
        class_value = int(self.class_value_spin.value())
        if not class_name:
            self._log("Enter a class name before assigning a class.")
            return
        for name in names:
            layer = safe_get_layer(self.viewer, name)
            if layer is None:
                continue
            self.registry.set_review_status(
                layer,
                status=self._status_for_layer(layer),
                class_name=class_name,
                class_value=class_value,
            )
        self._log(f"Assigned class '{class_name}' value {class_value} to {len(names)} layer(s).")
        self.refresh()
        self._refresh_all()

    def show_only_selected(self) -> None:
        names = set(self._selected_layer_names())
        if not names:
            self._log("Select one or more layers to isolate.")
            return
        for layer_name in labels_layer_names(self.viewer):
            layer = safe_get_layer(self.viewer, layer_name)
            if layer is not None:
                layer.visible = layer.name in names
        self.locate_selected()
        self._log(f"Showing only {len(names)} selected mask layer(s).")

    def hide_rejected(self) -> None:
        hidden = 0
        for record in self.registry.review_records(include_final_outputs=True):
            layer = safe_get_layer(self.viewer, record.layer_name)
            if layer is not None and record.review_status == "rejected":
                layer.visible = False
                hidden += 1
        self._log(f"Hid {hidden} rejected mask layer(s).")

    def show_all_masks(self) -> None:
        count = 0
        for layer_name in labels_layer_names(self.viewer):
            layer = safe_get_layer(self.viewer, layer_name)
            if layer is not None:
                layer.visible = True
                count += 1
        self._log(f"Showing {count} labels layer(s).")

    def locate_selected(self) -> None:
        names = self._selected_layer_names()
        if not names:
            self._log("Select a mask layer to locate.")
            return
        layer = safe_get_layer(self.viewer, names[0])
        if layer is None:
            self._log("Selected layer was not found.")
            return
        coords = np.argwhere(np.asarray(layer.data) != 0)
        if coords.size == 0:
            self._log(f"Layer {layer.name} has no non-zero mask pixels/voxels.")
            return
        center = coords.mean(axis=0)
        self._center_view_on_data_position(layer, center)
        try:
            self.viewer.layers.selection.active = layer
            layer.visible = True
            layer.mode = "pick"
        except Exception:
            pass
        self._log(f"Located {layer.name} at {self._format_position(center)}.")

    def _set_selected_status(self, status: str) -> None:
        names = self._selected_layer_names()
        if not names:
            self._log("Select one or more mask layers first.")
            return
        class_name = self.class_name_edit.text().strip()
        class_value = int(self.class_value_spin.value())
        for name in names:
            layer = safe_get_layer(self.viewer, name)
            if layer is None:
                continue
            self.registry.set_review_status(
                layer,
                status=status,
                class_name=class_name if class_name else None,
                class_value=class_value,
            )
            if status == "rejected" and self.auto_hide_rejected_check.isChecked():
                layer.visible = False
        self._log(f"Marked {len(names)} layer(s) as {status}.")
        self.refresh()
        self._refresh_all()

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        filter_form = QFormLayout()
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItem("All statuses", "")
        self.status_filter_combo.addItem("Unreviewed", "unreviewed")
        self.status_filter_combo.addItem("Accepted", "accepted")
        self.status_filter_combo.addItem("Needs edit", "needs_edit")
        self.status_filter_combo.addItem("Rejected", "rejected")
        self.status_filter_combo.currentIndexChanged.connect(lambda _index: self.refresh())
        self.class_filter_edit = QLineEdit()
        self.class_filter_edit.setPlaceholderText("optional exact class-name filter")
        self.class_filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.text_filter_edit = QLineEdit()
        self.text_filter_edit.setPlaceholderText("optional layer/object text filter")
        self.text_filter_edit.textChanged.connect(lambda _text: self.refresh())
        self.include_outputs_check = QCheckBox("include final/output masks")
        self.include_outputs_check.toggled.connect(lambda _checked: self.refresh())
        filter_form.addRow("Status filter", self.status_filter_combo)
        filter_form.addRow("Class filter", self.class_filter_edit)
        filter_form.addRow("Text filter", self.text_filter_edit)
        filter_form.addRow(QLabel(""), self.include_outputs_check)
        root.addLayout(filter_form)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Layer", "Status", "Class", "Value", "Pixels/Voxels", "Dims", "Shape", "Role"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self.locate_selected())
        root.addWidget(self.table)

        class_row = QHBoxLayout()
        self.class_name_edit = QLineEdit()
        self.class_name_edit.setPlaceholderText("myelin / axon / mito / error")
        self.class_value_spin = QSpinBox()
        self.class_value_spin.setRange(1, 2_147_483_647)
        self.class_value_spin.setValue(1)
        apply_class_btn = QPushButton("Assign Class")
        apply_class_btn.clicked.connect(self.apply_class_to_selected)
        class_row.addWidget(QLabel("Class"))
        class_row.addWidget(self.class_name_edit)
        class_row.addWidget(QLabel("Value"))
        class_row.addWidget(self.class_value_spin)
        class_row.addWidget(apply_class_btn)
        root.addLayout(class_row)

        action_row = QHBoxLayout()
        accept_btn = QPushButton("Accept")
        accept_btn.clicked.connect(self.accept_selected)
        reject_btn = QPushButton("Reject")
        reject_btn.clicked.connect(self.reject_selected)
        needs_edit_btn = QPushButton("Needs Edit")
        needs_edit_btn.clicked.connect(self.mark_needs_edit_selected)
        clear_btn = QPushButton("Unreview")
        clear_btn.clicked.connect(self.clear_review_selected)
        locate_btn = QPushButton("Locate")
        locate_btn.clicked.connect(self.locate_selected)
        delete_btn = QPushButton("Delete Layer")
        delete_btn.clicked.connect(self.delete_selected_layers)
        action_row.addWidget(accept_btn)
        action_row.addWidget(reject_btn)
        action_row.addWidget(needs_edit_btn)
        action_row.addWidget(clear_btn)
        action_row.addWidget(locate_btn)
        action_row.addWidget(delete_btn)
        root.addLayout(action_row)

        visibility_row = QHBoxLayout()
        isolate_btn = QPushButton("Show Only Selected")
        isolate_btn.clicked.connect(self.show_only_selected)
        hide_rejected_btn = QPushButton("Hide Rejected")
        hide_rejected_btn.clicked.connect(self.hide_rejected)
        show_all_btn = QPushButton("Show All Masks")
        show_all_btn.clicked.connect(self.show_all_masks)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        self.auto_hide_rejected_check = QCheckBox("auto-hide rejected")
        self.auto_hide_rejected_check.setChecked(True)
        visibility_row.addWidget(isolate_btn)
        visibility_row.addWidget(hide_rejected_btn)
        visibility_row.addWidget(show_all_btn)
        visibility_row.addWidget(refresh_btn)
        visibility_row.addWidget(self.auto_hide_rejected_check)
        root.addLayout(visibility_row)

        self.summary_label = QLabel("No masks loaded.")
        root.addWidget(self.summary_label)
        self.setLayout(root)

    def _on_selection_changed(self) -> None:
        names = self._selected_layer_names()
        if len(names) != 1:
            return
        layer = safe_get_layer(self.viewer, names[0])
        if layer is None:
            return
        metadata = getattr(layer, "metadata", {}) if isinstance(getattr(layer, "metadata", None), dict) else {}
        if metadata.get("class_name"):
            self.class_name_edit.setText(str(metadata.get("class_name")))
        if metadata.get("class_value"):
            try:
                self.class_value_spin.setValue(int(metadata.get("class_value")))
            except Exception:
                pass

    def _selected_layer_names(self) -> list[str]:
        names: list[str] = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is not None:
                names.append(str(item.data(Qt.UserRole)))
        return names

    def _set_text(self, row: int, col: int, text: str, layer_name: str) -> None:
        item = QTableWidgetItem(text)
        item.setData(Qt.UserRole, layer_name)
        self.table.setItem(row, col, item)

    def _set_numeric(self, row: int, col: int, value: int, layer_name: str) -> None:
        item = NumericItem(str(int(value)))
        item.setData(Qt.UserRole, layer_name if col == 0 else int(value))
        item.setData(Qt.UserRole + 1, layer_name)
        self.table.setItem(row, col, item)

    def _status_for_layer(self, layer) -> str:
        metadata = getattr(layer, "metadata", {}) if isinstance(getattr(layer, "metadata", None), dict) else {}
        role = str(metadata.get("sam3_role") or "")
        if role == "accepted_object":
            return "accepted"
        if role == "rejected_object":
            return "rejected"
        return str(metadata.get("review_status") or "unreviewed")

    def _sync_summary(self) -> None:
        counts = {"unreviewed": 0, "accepted": 0, "needs_edit": 0, "rejected": 0}
        for record in self.registry.review_records(include_final_outputs=self.include_outputs_check.isChecked()):
            counts[record.review_status] = counts.get(record.review_status, 0) + 1
        self.summary_label.setText(
            "Review summary: "
            f"accepted={counts.get('accepted', 0)}, "
            f"needs_edit={counts.get('needs_edit', 0)}, "
            f"rejected={counts.get('rejected', 0)}, "
            f"unreviewed={counts.get('unreviewed', 0)}"
        )

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

    def _format_position(self, data_position: np.ndarray) -> str:
        return ", ".join(f"{float(value):.1f}" for value in data_position)
