from __future__ import annotations

from collections.abc import Callable

import numpy as np
from napari import current_viewer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from .candidate_consolidation_service import CandidateConsolidationService
from .candidate_table_widget import CandidateTableWidget
from .utils import copy_layer_geometry, is_labels_layer, safe_get_layer, unique_layer_name


DEFAULT_INCLUDED_STATUSES = {"keep", "nested_child", "parent"}
STATUS_TOOLTIPS = {
    "keep": "Include this candidate in the final isolated objects layer.",
    "reject": "Exclude this candidate from the final isolated objects layer.",
    "duplicate": "Mark as a duplicate; excluded from final output by default.",
    "parent": "Keep as a larger parent object; included by default, usually trimmed around children.",
    "nested_child": "Keep as a smaller object inside a parent; included by default.",
    "fragment": "Mark as a partial fragment; excluded from final output by default.",
}


class MaskIsolationTab(QWidget):
    """Candidate consolidation workflow for aligned SAM3 mask layers."""

    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.service = CandidateConsolidationService(viewer)
        self._manual_status_ids: set[int] = set()
        self._build_ui()
        self.refresh()

    def refresh(self) -> None:
        self._ensure_viewer()
        selected = {item.data(256) for item in self.layer_list.selectedItems()}
        self.layer_list.clear()
        names = self._mask_layer_names()
        for name in names:
            item = QListWidgetItem(name)
            item.setData(256, name)
            self.layer_list.addItem(item)
            if name in selected:
                item.setSelected(True)
        if names:
            message = f"Mask Isolation: refreshed {len(names)} mask layer(s)."
            self.summary_label.setText(message)
            self._log(message)
        else:
            total = len(getattr(self.viewer, "layers", []) or []) if self.viewer is not None else 0
            message = (
                f"Mask Isolation: no usable mask layers found among {total} viewer layer(s). "
                "Use Labels layers or integer/binary mask image layers."
            )
            self.summary_label.setText(message)
            self._log(message)

    def harvest_masks(self) -> None:
        self._ensure_viewer()
        names = self._selected_layer_names()
        if not names:
            self._log("Mask Isolation: select at least one Labels layer or enable all visible Labels layers.")
            return
        self.summary_label.setText(f"Harvesting {len(names)} mask layer(s)...")
        self.harvest_progress.setRange(0, 100)
        self.harvest_progress.setValue(0)
        self.harvest_progress.setVisible(True)
        QApplication.processEvents()
        try:
            records = self.service.harvest_layers(
                names,
                min_area=self.min_area_spin.value(),
                progress_callback=self._update_harvest_progress,
            )
        except ValueError as exc:
            self._log(f"Mask Isolation: {exc}")
            self.summary_label.setText("No mask set loaded.")
            self.harvest_progress.setValue(0)
            self.table.set_records([])
            return
        self._manual_status_ids = set()
        self.table.set_records(records)
        shape_text = "x".join(str(value) for value in (self.service.shape or ()))
        self.summary_label.setText(f"Loaded {len(self.service.source_layer_names)} mask layer(s), {len(records)} candidate object(s), shape: {shape_text}")
        self.harvest_progress.setValue(100)
        self._log(f"Mask Isolation harvested {len(records)} candidate object(s) from {len(self.service.source_layer_names)} layer(s).")

    def _update_harvest_progress(self, completed: int, total: int, message: str) -> None:
        value = 100 if total <= 0 else int(round(100 * min(completed, total) / total))
        self.harvest_progress.setValue(value)
        self.harvest_progress.setFormat(f"{value}% - {message}")
        self.summary_label.setText(message)
        QApplication.processEvents()

    def classify_candidates(self) -> None:
        if not self.service.records:
            self._log("Harvest masks before classifying candidates.")
            return
        manual_statuses = {
            record.candidate_id: record.status
            for record in self.service.records
            if record.candidate_id in self._manual_status_ids
        }
        self.service.classify_candidates(
            duplicate_iou=self.duplicate_iou_spin.value(),
            nested_containment=self.nested_containment_spin.value(),
            fragment_iou_min=self.fragment_iou_spin.value(),
            area_ratio_threshold=self.area_ratio_spin.value(),
        )
        for record in self.service.records:
            if record.candidate_id in manual_statuses:
                record.status = manual_statuses[record.candidate_id]
        self.table.set_records(self.service.records)
        counts: dict[str, int] = {}
        for record in self.service.records:
            counts[record.status] = counts.get(record.status, 0) + 1
        summary = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))
        self._log(f"Mask Isolation classified {len(self.service.records)} candidate(s): {summary}.")

    def create_isolated_objects_layer(self) -> None:
        if not self.service.records:
            self._log("Harvest masks before creating an isolated objects layer.")
            return
        self.service.sort_rule = self.sort_combo.currentData()
        parent_handling = self.parent_combo.currentData()
        if parent_handling == "separate_parent_layer":
            self._log("Parent handling is set to separate parent layer; creating main layer with trimmed parent behavior.")
        overlap_rule = self.overlap_combo.currentData()
        service_parent_handling = "trim_parent" if parent_handling == "separate_parent_layer" else parent_handling
        data, mapping = self.service.create_isolated_label_data(
            DEFAULT_INCLUDED_STATUSES,
            overlap_rule=overlap_rule,
            parent_handling=service_parent_handling,
        )
        if data.size == 0:
            self._log("No harvested candidate data is available for output.")
            return
        base_name = self.output_name_edit.text().strip() or "isolated_objects"
        name = unique_layer_name(self.viewer, base_name)
        layer = self.viewer.add_labels(
            data.astype(np.uint32, copy=False),
            name=name,
            metadata=self.service.output_metadata(DEFAULT_INCLUDED_STATUSES, mapping, overlap_rule, parent_handling),
        )
        self._copy_reference_geometry(layer)
        self.table.set_records(self.service.records)
        self._log(f"Created isolated object layer {name} with {len(mapping)} object(s) from {len(self.service.records)} candidate(s).")
        if parent_handling == "separate_parent_layer":
            self.create_parent_layer()
        self._refresh_all()

    def create_parent_layer(self) -> None:
        parent_ids = [record.candidate_id for record in self.service.records if record.status == "parent"]
        if not parent_ids:
            self._log("No parent candidates are available for a parent-only layer.")
            return
        data = self.service.selected_candidates_label_data(parent_ids)
        name = unique_layer_name(self.viewer, "isolated_parent_candidates")
        layer = self.viewer.add_labels(
            data.astype(np.uint32, copy=False),
            name=name,
            metadata={
                "sam3_role": "isolated_parent_candidates",
                "source_candidates": [int(candidate_id) for candidate_id in parent_ids],
                "created_from": "Mask Isolation",
            },
        )
        self._copy_reference_geometry(layer)
        self._log(f"Created parent candidate layer {name} with {len(parent_ids)} parent object(s).")
        self._refresh_all()

    def create_child_layer(self) -> None:
        child_ids = [record.candidate_id for record in self.service.records if record.status == "nested_child"]
        if not child_ids:
            self._log("No nested child candidates are available for a child-only layer.")
            return
        data = self.service.selected_candidates_label_data(child_ids)
        name = unique_layer_name(self.viewer, "isolated_child_candidates")
        layer = self.viewer.add_labels(
            data.astype(np.uint32, copy=False),
            name=name,
            metadata={
                "sam3_role": "isolated_child_candidates",
                "source_candidates": [int(candidate_id) for candidate_id in child_ids],
                "created_from": "Mask Isolation",
            },
        )
        self._copy_reference_geometry(layer)
        self._log(f"Created child candidate layer {name} with {len(child_ids)} child object(s).")
        self._refresh_all()

    def isolate_selected(self) -> None:
        candidate_ids = self.table.selected_candidate_ids()
        if not candidate_ids:
            self._log("Select one or more candidate rows to isolate.")
            return
        data = self.service.selected_candidates_label_data(candidate_ids)
        if data.size == 0:
            self._log("No harvested candidate data is available for selected isolation.")
            return
        name = unique_layer_name(self.viewer, "isolated_selected_candidates")
        layer = self.viewer.add_labels(
            data.astype(np.uint32, copy=False),
            name=name,
            metadata={
                "sam3_role": "isolated_selected_candidates",
                "source_candidates": [int(candidate_id) for candidate_id in candidate_ids],
                "created_from": "Mask Isolation",
            },
        )
        self._copy_reference_geometry(layer)
        self._log(f"Created selected candidate layer {name} with {len(candidate_ids)} candidate(s).")
        self._refresh_all()

    def locate_candidate(self, candidate_id: int | None = None) -> None:
        if candidate_id is None:
            ids = self.table.selected_candidate_ids()
            if not ids:
                self._log("Select a candidate row to locate.")
                return
            candidate_id = ids[0]
        record = self._record(candidate_id)
        mask = self.service.candidate_mask(candidate_id)
        if record is None or mask is None:
            self._log(f"Candidate {candidate_id} is not available in the harvested mask set.")
            return
        coords = np.argwhere(mask)
        if coords.size == 0:
            self._log(f"Candidate {candidate_id} is empty.")
            return
        layer = safe_get_layer(self.viewer, record.source_layer_name)
        if layer is None:
            self._log(f"Source layer {record.source_layer_name!r} for candidate {candidate_id} was not found.")
            return
        center = coords.mean(axis=0)
        layer.visible = True
        self._center_view_on_data_position(layer, center)
        self.table.select_candidate_id(candidate_id)
        centroid_text = ", ".join(f"{float(value):.1f}" for value in center)
        self._log(
            f"Located candidate {candidate_id} from {record.source_layer_name}, "
            f"source label {record.source_label_value}, centroid {centroid_text}."
        )

    def apply_status_to_selected(self, status: str) -> None:
        candidate_ids = self.table.selected_candidate_ids()
        if not candidate_ids:
            self._log("Select one or more candidate rows before changing status.")
            return
        self.service.set_status(candidate_ids, status)
        self._manual_status_ids.update(int(candidate_id) for candidate_id in candidate_ids)
        self.table.set_records(self.service.records)
        self._log(f"Marked {len(candidate_ids)} candidate(s) as {status}.")

    def _handle_table_action(self, action: str) -> None:
        if action == "locate":
            self.locate_candidate()
        elif action == "isolate_selected":
            self.isolate_selected()
        else:
            self.apply_status_to_selected(action)

    def _selected_layer_names(self) -> list[str]:
        self._ensure_viewer()
        if self.use_visible_check.isChecked():
            if self.viewer is None:
                return []
            visible_names = []
            label_names = set(self._mask_layer_names())
            for layer in self.viewer.layers:
                if layer.name in label_names and bool(getattr(layer, "visible", False)):
                    visible_names.append(layer.name)
            return visible_names
        return [item.data(256) for item in self.layer_list.selectedItems()]

    def _mask_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        return [layer.name for layer in self.viewer.layers if self._is_usable_mask_layer(layer)]

    def _is_usable_mask_layer(self, layer) -> bool:
        if is_labels_layer(layer):
            return True
        if str(getattr(layer, "_type_string", "")).lower() == "labels":
            return True
        data = getattr(layer, "data", None)
        if data is None:
            return False
        try:
            arr = np.asarray(data)
        except Exception:
            return False
        if arr.ndim < 2 or arr.size == 0:
            return False
        if arr.dtype == np.bool_:
            return True
        if np.issubdtype(arr.dtype, np.integer):
            return self._has_mask_like_values(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            return False
        return self._has_integer_valued_mask_data(arr)

    def _has_mask_like_values(self, arr: np.ndarray) -> bool:
        values = np.unique(arr)
        nonzero = values[values != 0]
        if len(nonzero) == 0:
            return False
        if len(values) <= 32:
            return True
        return False

    def _has_integer_valued_mask_data(self, arr: np.ndarray) -> bool:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return False
        values = np.unique(finite)
        nonzero = values[values != 0]
        if len(nonzero) == 0 or len(values) > 32:
            return False
        return bool(np.allclose(values, np.round(values)))

    def _ensure_viewer(self) -> None:
        if self.viewer is not None:
            return
        self.viewer = current_viewer()
        self.service.viewer = self.viewer

    def _copy_reference_geometry(self, target_layer) -> None:
        source = safe_get_layer(self.viewer, self.service.reference_layer_name)
        if source is not None:
            copy_layer_geometry(source, target_layer)

    def _record(self, candidate_id: int):
        for record in self.service.records:
            if record.candidate_id == int(candidate_id):
                return record
        return None

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

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(6)

        input_form = QFormLayout()
        self.use_visible_check = QCheckBox("Use all visible Labels layers")
        self.use_visible_check.setToolTip("Use every visible mask layer instead of manually selecting layers below.")
        self.layer_list = QListWidget()
        self.layer_list.setToolTip("Select the mask layers that belong to the same image and coordinate system.")
        self.layer_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.setToolTip("Reload the list of usable mask layers from the viewer.")
        refresh_btn.clicked.connect(self.refresh)
        harvest_btn = QPushButton("Harvest Masks")
        harvest_btn.setToolTip("Collect selected mask layers and split them into individual candidate objects.")
        harvest_btn.clicked.connect(self.harvest_masks)
        input_buttons = QHBoxLayout()
        input_buttons.addWidget(refresh_btn)
        input_buttons.addWidget(harvest_btn)
        self.summary_label = QLabel("No mask set loaded.")
        self.summary_label.setToolTip("Shows the current Mask Isolation state and recent action result.")
        self.harvest_progress = QProgressBar()
        self.harvest_progress.setRange(0, 100)
        self.harvest_progress.setValue(0)
        self.harvest_progress.setTextVisible(True)
        self.harvest_progress.setVisible(False)
        input_form.addRow(self.use_visible_check)
        input_form.addRow("Labels layers", self.layer_list)
        input_form.addRow(input_buttons)
        input_form.addRow("Summary", self.summary_label)
        input_form.addRow("Harvest progress", self.harvest_progress)
        root.addLayout(input_form)

        settings = QGridLayout()
        settings.setHorizontalSpacing(8)
        settings.setVerticalSpacing(5)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 2_147_483_647)
        self.min_area_spin.setValue(64)
        self.min_area_spin.setToolTip("Ignore tiny candidates below this pixel/voxel count during harvest.")
        self.duplicate_iou_spin = self._double_spin(0.85)
        self.duplicate_iou_spin.setToolTip("Higher values mark only near-identical overlaps as duplicates.")
        self.nested_containment_spin = self._double_spin(0.90)
        self.nested_containment_spin.setToolTip("Containment needed to call a smaller mask a child inside a parent.")
        self.fragment_iou_spin = self._double_spin(0.15)
        self.fragment_iou_spin.setToolTip("Minimum overlap for detecting small partial fragments.")
        self.area_ratio_spin = self._double_spin(0.75)
        self.area_ratio_spin.setToolTip("Duplicate masks must be similar in size above this ratio.")
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Area small → large", "area_small_to_large")
        self.sort_combo.addItem("Area large → small", "area_large_to_small")
        self.sort_combo.setToolTip("Controls output write order. Small first protects child objects.")
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItem("Small objects win", "small_objects_win")
        self.overlap_combo.addItem("Later candidates win", "later_wins")
        self.overlap_combo.addItem("Set conflict to background", "set_conflict_background")
        self.overlap_combo.setToolTip("Choose what happens when included candidates share pixels.")
        self.parent_combo = QComboBox()
        self.parent_combo.addItem("Trim parent where child exists", "trim_parent")
        self.parent_combo.addItem("Keep parent full in separate parent layer", "separate_parent_layer")
        self.parent_combo.addItem("Parent overwrites children", "parent_overwrites_children")
        self.parent_combo.setToolTip("Default trims parent pixels under children so child objects stay visible.")
        classify_btn = QPushButton("Classify Candidates")
        classify_btn.setToolTip("Analyze overlaps and mark candidates as keep, duplicate, child, parent, or fragment.")
        classify_btn.clicked.connect(self.classify_candidates)
        self._add_setting_row(settings, 0, "Minimum area", self.min_area_spin, "Duplicate IoU", self.duplicate_iou_spin)
        self._add_setting_row(settings, 1, "Nested containment", self.nested_containment_spin, "Fragment IoU minimum", self.fragment_iou_spin)
        self._add_setting_row(settings, 2, "Area ratio threshold", self.area_ratio_spin, "Sort rule", self.sort_combo)
        self._add_setting_row(settings, 3, "Output overlap rule", self.overlap_combo, "Parent handling", self.parent_combo)
        settings.addWidget(classify_btn, 4, 3)
        root.addLayout(settings)

        self.table = CandidateTableWidget(locate_callback=self.locate_candidate, action_callback=self._handle_table_action)
        self.table.setToolTip("Review candidates. Status controls whether each candidate is included in final output.")
        self.table.setMinimumHeight(220)
        root.addWidget(self.table)

        actions = QHBoxLayout()
        self.table_context_check = QCheckBox("Enable table right-click actions")
        self.table_context_check.setChecked(True)
        self.table_context_check.setToolTip("When on: right-click selected candidate rows to keep, reject, locate, or isolate them.")
        self.table_context_check.toggled.connect(self.table.set_actions_enabled)
        actions.addWidget(self.table_context_check)
        for label, status in (
            ("Keep", "keep"),
            ("Reject", "reject"),
            ("Mark Duplicate", "duplicate"),
            ("Mark Parent", "parent"),
            ("Mark Child", "nested_child"),
            ("Mark Fragment", "fragment"),
        ):
            button = QPushButton(label)
            button.setToolTip(STATUS_TOOLTIPS[status])
            button.clicked.connect(lambda _checked=False, status=status: self.apply_status_to_selected(status))
            actions.addWidget(button)
        locate_btn = QPushButton("Locate Candidate")
        locate_btn.setToolTip("Center the viewer on the selected candidate in its source layer.")
        locate_btn.clicked.connect(lambda _checked=False: self.locate_candidate())
        isolate_btn = QPushButton("Isolate Selected")
        isolate_btn.setToolTip("Create a temporary layer containing only the selected candidates.")
        isolate_btn.clicked.connect(self.isolate_selected)
        actions.addWidget(locate_btn)
        actions.addWidget(isolate_btn)
        root.addLayout(actions)

        output_form = QFormLayout()
        self.output_name_edit = QLineEdit("isolated_objects")
        self.output_name_edit.setToolTip("Name for the final combined isolated object layer.")
        create_btn = QPushButton("Create Isolated Objects Layer")
        create_btn.setToolTip("Create one final layer from included statuses: keep, child, and parent.")
        create_btn.clicked.connect(self.create_isolated_objects_layer)
        parent_btn = QPushButton("Create Parent Layer")
        parent_btn.setToolTip("Optional: create a separate layer with only parent candidates.")
        parent_btn.clicked.connect(self.create_parent_layer)
        child_btn = QPushButton("Create Child Layer")
        child_btn.setToolTip("Optional: create a separate layer with only nested child candidates.")
        child_btn.clicked.connect(self.create_child_layer)
        output_buttons = QHBoxLayout()
        output_buttons.addWidget(create_btn)
        output_buttons.addWidget(parent_btn)
        output_buttons.addWidget(child_btn)
        output_form.addRow("Output layer name", self.output_name_edit)
        output_form.addRow(output_buttons)
        root.addLayout(output_form)
        self.setLayout(root)

    def _double_spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(3)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    def _add_setting_row(self, layout: QGridLayout, row: int, label_a: str, widget_a, label_b: str, widget_b) -> None:
        layout.addWidget(QLabel(label_a), row, 0)
        layout.addWidget(widget_a, row, 1)
        layout.addWidget(QLabel(label_b), row, 2)
        layout.addWidget(widget_b, row, 3)
