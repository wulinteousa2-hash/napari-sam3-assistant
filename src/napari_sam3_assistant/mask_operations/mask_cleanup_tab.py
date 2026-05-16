from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidgetSelectionRange,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .cleanup_service import MaskCleanupService
from .component_analysis_service import ComponentAnalysisService
from .fast_component_index_service import FastComponentIndex, FastComponentIndexService
from .component_table_widget import ComponentTableWidget
from .models import AxonHoleCandidate
from .utils import copy_layer_geometry, image_layer_names, labels_layer_names, safe_get_layer, shapes_layer_names, unique_layer_name
from ..widgets.collapsible_panel import CollapsiblePanel


UNDO_HISTORY_LIMIT = 20
UNDO_FULL_SNAPSHOT_PIXEL_LIMIT = 25_000_000


class MaskCleanupTab(QWidget):
    """Friction-light cleanup for 2D/3D binary, semantic, and instance masks."""

    def __init__(self, viewer, log_callback: Callable[[str], None], refresh_callback: Callable[[], None]) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback
        self._refresh_all = refresh_callback
        self.analysis = ComponentAnalysisService()
        self.fast_index_builder = FastComponentIndexService()
        self.cleanup = MaskCleanupService()
        self._mouse_layer = None
        self._mouse_callback = self._handle_mouse_action
        self._mouse_double_click_callback = self._handle_mouse_double_click
        self._undo_history: dict[int, list[np.ndarray]] = {}
        self._last_layer_data: dict[int, np.ndarray] = {}
        self._tracked_layer = None
        self._history_callback = self._on_tracked_layer_data_changed
        self._suppress_history_event = False
        self._last_analysis_indexer: object = Ellipsis
        self._last_analysis_offset: tuple[int, ...] | None = None
        self._fast_index: FastComponentIndex | None = None
        self._fast_index_layer_id: int | None = None
        self._fast_index_scope_key: tuple | None = None
        self._component_index_stale = True
        self._batch_axon_candidates: list[AxonHoleCandidate] = []
        self._batch_axon_masks: dict[int, np.ndarray] = {}
        self._batch_preview_layer_name: str | None = None
        self._batch_preview_indexer: object = Ellipsis
        self._batch_preview_shape: tuple[int, ...] | None = None
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
        current_image = self.source_image_combo.currentData()
        self.source_image_combo.clear()
        for name in image_layer_names(self.viewer):
            self.source_image_combo.addItem(name, name)
        image_index = self.source_image_combo.findData(current_image)
        if image_index >= 0:
            self.source_image_combo.setCurrentIndex(image_index)
        current_roi = self.batch_roi_combo.currentData() if hasattr(self, "batch_roi_combo") else None
        current_work_roi = self.work_roi_combo.currentData() if hasattr(self, "work_roi_combo") else None
        if hasattr(self, "batch_roi_combo"):
            self.batch_roi_combo.clear()
            self.batch_roi_combo.addItem("No ROI shape", "")
            for name in shapes_layer_names(self.viewer):
                self.batch_roi_combo.addItem(name, name)
            roi_index = self.batch_roi_combo.findData(current_roi)
            if roi_index >= 0:
                self.batch_roi_combo.setCurrentIndex(roi_index)
        if hasattr(self, "work_roi_combo"):
            self.work_roi_combo.clear()
            self.work_roi_combo.addItem("No ROI shape", "")
            for name in shapes_layer_names(self.viewer):
                self.work_roi_combo.addItem(name, name)
            work_roi_index = self.work_roi_combo.findData(current_work_roi)
            if work_roi_index >= 0:
                self.work_roi_combo.setCurrentIndex(work_roi_index)
        self._sync_scope_controls()
        self._sync_work_region_controls()
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_action_callback()

    def analyze_layer(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer for component analysis.")
            return
        sub, indexer, offset = self._scoped_data(layer)
        scope = self._scope_label(layer)
        self.status_label.setText(f"Analyzing {layer.name} ({scope})...")
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("0% - Preparing component analysis...")
        self.analysis_progress.setVisible(True)
        QApplication.processEvents()
        self._fast_index = self.fast_index_builder.build(
            sub,
            progress_callback=self._update_analysis_progress,
        )
        self._fast_index_layer_id = id(layer)
        self._fast_index_scope_key = self._scope_key(layer, indexer)
        self._component_index_stale = False
        self._last_analysis_indexer = indexer
        self._last_analysis_offset = offset
        records = self._fast_index.active_records()
        self.component_table.set_records(records)
        self.status_label.setText(f"Fast index ready: {len(records)} component(s) indexed in {scope}.")
        self.analysis_progress.setValue(100)
        self.analysis_progress.setFormat(f"100% - Indexed {len(records)} component(s)")
        self._log(f"Built fast click index for {len(records)} connected component(s) in {layer.name} ({scope}).")

    def _update_analysis_progress(self, completed: int, total: int, message: str) -> None:
        value = 100 if total <= 0 else int(round(100 * min(completed, total) / total))
        self.analysis_progress.setValue(value)
        self.analysis_progress.setFormat(f"{value}% - {message}")
        self.status_label.setText(message)
        QApplication.processEvents()

    def delete_selected_components(self) -> None:
        layer = self._target_layer()
        ids = self.component_table.selected_component_ids()
        if layer is None or not ids:
            self._log("Select component rows to delete.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        fast_index = self._fresh_fast_index(layer, sub, indexer)
        if fast_index is None:
            self._log("Component index is stale or not built. Click Analyze Layer before deleting selected components.")
            return
        cleaned, changed = fast_index.delete_components(sub, ids)
        if changed and self._replace_scoped_layer_data(layer, cleaned, indexer, "delete selected components", invalidate_fast_index=False):
            self.component_table.set_records(fast_index.active_records())
            self.refresh_unique_values()
            self._log(f"Deleted {len(ids)} selected component(s) from {layer.name} ({changed} pixel(s)/voxel(s)).")
        else:
            self._log("Selected component delete made no mask changes.")

    def assign_selected_components_to_current_value(self) -> None:
        layer = self._target_layer()
        ids = self.component_table.selected_component_ids()
        if layer is None or not ids:
            self._log("Select component rows to assign.")
            return
        new_value = int(self.assignment_value_spin.value())
        sub, indexer, _offset = self._scoped_data(layer)
        fast_index = self._fresh_fast_index(layer, sub, indexer)
        if fast_index is None:
            self._log("Component index is stale or not built. Click Analyze Layer before assigning selected components.")
            return
        data, changed = fast_index.relabel_components(sub, ids, new_value)
        if changed and self._replace_scoped_layer_data(layer, data, indexer, "assign selected components", invalidate_fast_index=False):
            self.component_table.set_records(fast_index.active_records())
            self.refresh_unique_values()
            self._select_value_row(new_value)
            self._log(
                f"Assigned {len(ids)} selected component(s) to value {new_value} in {layer.name} "
                f"({changed} pixel(s)/voxel(s))."
            )
            self._refresh_all()
        else:
            self._log("Selected component assignment made no mask changes.")

    def set_assignment_value(self, value: int) -> None:
        self.assignment_value_spin.setValue(int(value))
        self.status_label.setText(f"Assignment value set to {int(value)}.")

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
            self.refresh_unique_values()
            self._refresh_all()
        else:
            self._log("Convert non-zero made no mask changes.")

    def preview_batch_axons(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        image = self._source_image_for_scoped_labels(layer, indexer)
        if image is None:
            self._log("Select a source image layer before batch axon preview.")
            return
        try:
            roi_mask = self._batch_roi_mask(layer, indexer, sub.shape)
            candidates, masks, preview = self.cleanup.propose_axon_holes(
                sub,
                image,
                threshold_percent=int(self.axon_threshold_spin.value()),
                max_fraction_percent=int(self.axon_max_fraction_spin.value()),
                min_object_size=int(self.batch_min_object_spin.value()),
                min_confidence_percent=int(self.batch_min_confidence_spin.value()),
                roi_mask=roi_mask,
                progress_callback=self._update_batch_preview_progress,
            )
        except ValueError as exc:
            self._log(str(exc))
            return
        self._batch_axon_candidates = candidates
        self._batch_axon_masks = masks
        self._batch_preview_indexer = indexer
        self._batch_preview_shape = tuple(preview.shape)
        self._set_batch_axon_records(candidates)
        self._write_batch_preview_layer(layer, preview, indexer)
        counts: dict[str, int] = {}
        for candidate in candidates:
            counts[candidate.status] = counts.get(candidate.status, 0) + 1
        summary = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "none"
        self.status_label.setText(f"Batch axon preview ready: {len(candidates)} proposal(s), {summary}.")
        self._log(f"Batch axon preview created {len(candidates)} proposal(s): {summary}.")

    def create_myelin_axon_layers(self) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        image = self._source_image_for_scoped_labels(layer, indexer)
        if image is None:
            self._log("Select a source image layer before creating myelin/axon layers.")
            return
        try:
            roi_mask = self._batch_roi_mask(layer, indexer, sub.shape)
            candidates, masks, preview = self.cleanup.propose_axon_holes(
                sub,
                image,
                threshold_percent=int(self.axon_threshold_spin.value()),
                max_fraction_percent=int(self.axon_max_fraction_spin.value()),
                min_object_size=int(self.batch_min_object_spin.value()),
                min_confidence_percent=int(self.batch_min_confidence_spin.value()),
                roi_mask=roi_mask,
                progress_callback=self._update_batch_preview_progress,
            )
        except ValueError as exc:
            self._log(str(exc))
            return

        self._batch_axon_candidates = candidates
        self._batch_axon_masks = masks
        self._batch_preview_indexer = indexer
        self._batch_preview_shape = tuple(preview.shape)
        self._set_batch_axon_records(candidates)

        min_confidence = max(0.0, min(float(self.batch_min_confidence_spin.value()), 100.0)) / 100.0
        accepted = [
            candidate
            for candidate in candidates
            if candidate.status == "confident"
            and candidate.confidence >= min_confidence
            and candidate.candidate_id in masks
        ]
        if not accepted:
            self._log("No confident myelin/axon proposals found. Open Advanced review to inspect candidates.")
            self.status_label.setText("No confident proposals found; review candidates before creating layers.")
            return

        myelin_local = np.asarray(sub).copy()
        axon_local = np.zeros_like(myelin_local, dtype=np.uint32)
        for candidate in accepted:
            mask = masks.get(candidate.candidate_id)
            if mask is None:
                continue
            slices = tuple(slice(int(lo), int(hi)) for lo, hi in candidate.bbox)
            myelin_slice = myelin_local[slices]
            axon_slice = axon_local[slices]
            myelin_slice[mask] = 0
            axon_slice[mask] = int(candidate.label_value)

        myelin_full = self._full_data_from_scoped_result(layer, myelin_local, indexer)
        axon_full = self._full_data_from_scoped_result(layer, axon_local, indexer)

        myelin_name = unique_layer_name(self.viewer, "myelin_rings")
        axon_name = unique_layer_name(self.viewer, "axons")
        source_name = str(layer.name)
        common_metadata = {
            "created_from": "Mask Cleanup / Myelin / Axon Rings",
            "source_layer": source_name,
            "source_image_layer": str(self.source_image_combo.currentData() or ""),
            "candidate_count": len(candidates),
            "accepted_candidate_count": len(accepted),
            "min_confidence_percent": int(self.batch_min_confidence_spin.value()),
            "seed_tolerance_percent": int(self.axon_threshold_spin.value()),
            "max_axon_percent": int(self.axon_max_fraction_spin.value()),
            "min_object_size": int(self.batch_min_object_spin.value()),
            "candidate_ids": [int(candidate.candidate_id) for candidate in accepted],
        }
        myelin_layer = self.viewer.add_labels(
            myelin_full,
            name=myelin_name,
            metadata={**common_metadata, "sam3_role": "myelin_ring_labels"},
        )
        axon_layer = self.viewer.add_labels(
            axon_full,
            name=axon_name,
            metadata={**common_metadata, "sam3_role": "axon_labels"},
        )
        copy_layer_geometry(layer, myelin_layer)
        copy_layer_geometry(layer, axon_layer)

        axon_pixels = int(np.count_nonzero(axon_full))
        review_count = sum(1 for candidate in candidates if candidate.status in {"review", "failed"})
        self.status_label.setText(
            f"Created {myelin_name} and {axon_name}: {len(accepted)} accepted axon object(s), "
            f"{axon_pixels} axon pixel(s)/voxel(s)."
        )
        self._log(
            f"Created myelin/axon layers from {source_name}: {myelin_name}, {axon_name}; "
            f"{len(accepted)} accepted of {len(candidates)} candidate(s), {review_count} left for optional review."
        )
        self._refresh_all()

    def apply_confident_batch_axons(self) -> None:
        self._apply_batch_axons(candidate_ids=None, action_name="apply confident batch axon holes")

    def apply_selected_batch_axons(self) -> None:
        ids = self._selected_batch_axon_ids()
        if not ids:
            self._log("Select one or more batch axon proposal rows to apply.")
            return
        self._apply_batch_axons(candidate_ids=ids, action_name="apply selected batch axon holes")

    def _apply_batch_axons(self, *, candidate_ids: set[int] | None, action_name: str) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        if not self._batch_axon_candidates:
            self._log("Run Preview Batch Axons before applying confident proposals.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        output_value = int(self.axon_value_spin.value()) if self.axon_assign_class_check.isChecked() else 0
        data, applied, changed = self.cleanup.apply_axon_hole_candidates(
            sub,
            self._batch_axon_candidates,
            self._batch_axon_masks,
            output_value=output_value,
            min_confidence_percent=int(self.batch_min_confidence_spin.value()),
            candidate_ids=candidate_ids,
        )
        if applied <= 0 or changed <= 0:
            self._log("No batch axon proposals were applied.")
            return
        if self._replace_scoped_layer_data(layer, data, indexer, action_name):
            self._invalidate_fast_index()
            self.refresh_unique_values()
            self._refresh_all()
            self.status_label.setText(
                f"Applied {applied} batch axon hole(s), changed {changed} pixel(s)/voxel(s). "
                "Component table is stale; click Analyze Layer to rebuild."
            )
            self._log(f"Applied {applied} batch axon hole(s) to {layer.name} ({changed} pixel(s)/voxel(s)).")

    def fill_selected_batch_axon_holes(self) -> None:
        ids = self._selected_batch_axon_ids()
        if not ids:
            self._log("Select one or more batch axon proposal rows to fill.")
            return
        candidates, masks, filled = self.cleanup.fill_axon_candidate_holes(
            self._batch_axon_candidates,
            self._batch_axon_masks,
            ids,
            min_confidence_percent=int(self.batch_min_confidence_spin.value()),
            max_fraction_percent=int(self.axon_max_fraction_spin.value()),
        )
        self._batch_axon_candidates = candidates
        self._batch_axon_masks = masks
        self._set_batch_axon_records(candidates)
        self._refresh_batch_preview_layer()
        self.status_label.setText(f"Filled {filled} hole pixel(s) in selected batch axon proposal(s).")
        self._log(f"Filled {filled} hole pixel(s) in selected batch axon proposal(s).")

    def skip_selected_batch_axons(self) -> None:
        ids = self._selected_batch_axon_ids()
        if not ids:
            self._log("Select one or more batch axon proposal rows to skip.")
            return
        self._mark_batch_axon_candidates(ids, "skipped", "user skipped")

    def skip_low_confidence_batch_axons(self) -> None:
        threshold = float(self.batch_min_confidence_spin.value()) / 100.0
        ids = {
            candidate.candidate_id
            for candidate in self._batch_axon_candidates
            if candidate.confidence < threshold
        }
        if not ids:
            self._log("No low-confidence batch axon proposals to skip.")
            return
        self._mark_batch_axon_candidates(ids, "skipped", "below confidence threshold")

    def _mark_batch_axon_candidates(self, ids: set[int], status: str, reason: str) -> None:
        selected = {int(value) for value in ids}
        self._batch_axon_candidates = [
            replace(candidate, status=status, reason=reason)
            if candidate.candidate_id in selected
            else candidate
            for candidate in self._batch_axon_candidates
        ]
        for candidate_id in selected:
            self._batch_axon_masks.pop(candidate_id, None)
        self._set_batch_axon_records(self._batch_axon_candidates)
        self._refresh_batch_preview_layer()
        self.status_label.setText(f"Marked {len(selected)} batch axon proposal(s) as {status}.")

    def select_visible_batch_axons(self) -> None:
        rows = self.batch_axon_table.rowCount()
        if rows <= 0:
            return
        self.batch_axon_table.clearSelection()
        self.batch_axon_table.setRangeSelected(
            QTableWidgetSelectionRange(0, 0, rows - 1, self.batch_axon_table.columnCount() - 1),
            True,
        )
        self.status_label.setText(f"Selected {rows} visible batch axon proposal row(s).")

    def select_batch_axons_by_status(self, status: str) -> None:
        self.batch_axon_table.clearSelection()
        selected_rows = 0
        for row in range(self.batch_axon_table.rowCount()):
            item = self.batch_axon_table.item(row, 0)
            if item is not None and item.text() == status:
                self.batch_axon_table.selectRow(row)
                selected_rows += 1
        self.status_label.setText(f"Selected {selected_rows} batch axon proposal row(s) with status {status}.")

    def export_selected_batch_axon_layer(self) -> None:
        ids = self._selected_batch_axon_ids()
        if not ids:
            self._log("Select one or more batch axon proposal rows to export.")
            return
        self._export_batch_axon_layer(ids, "selected")

    def export_batch_axon_layer_by_status(self, status: str) -> None:
        ids = {
            candidate.candidate_id
            for candidate in self._batch_axon_candidates
            if candidate.status == status
        }
        if not ids:
            self._log(f"No batch axon proposals with status {status}.")
            return
        self._export_batch_axon_layer(ids, status)

    def _export_batch_axon_layer(self, ids: set[int], group_name: str) -> None:
        target_layer = self._target_layer()
        if target_layer is None or self._batch_preview_shape is None:
            self._log("Run Preview Batch Axons before exporting proposal layers.")
            return
        preview = np.zeros(self._batch_preview_shape, dtype=np.uint32)
        for output_id, candidate_id in enumerate(sorted(int(value) for value in ids), start=1):
            candidate = self._batch_axon_candidate(candidate_id)
            mask = self._batch_axon_masks.get(candidate_id)
            if candidate is None or mask is None:
                continue
            preview_slice = preview[tuple(slice(int(lo), int(hi)) for lo, hi in candidate.bbox)]
            preview_slice[mask] = int(output_id)
        full = np.zeros_like(np.asarray(target_layer.data), dtype=np.uint32)
        if self._batch_preview_indexer is Ellipsis:
            full = preview
        else:
            full[self._batch_preview_indexer] = preview
        name = unique_layer_name(self.viewer, f"batch_axon_{group_name}")
        layer = self.viewer.add_labels(
            full,
            name=name,
            metadata={
                "sam3_role": "batch_axon_export",
                "batch_group": group_name,
                "source_layer": target_layer.name,
                "candidate_ids": sorted(int(value) for value in ids),
            },
        )
        copy_layer_geometry(target_layer, layer)
        self.status_label.setText(f"Exported {len(ids)} batch axon proposal(s) to {name}.")

    def _update_batch_preview_progress(self, completed: int, total: int, message: str) -> None:
        if hasattr(self, "status_label"):
            self.status_label.setText(message)
        QApplication.processEvents()

    def _set_batch_axon_records(self, candidates: list[AxonHoleCandidate]) -> None:
        self.batch_axon_table.setSortingEnabled(False)
        self.batch_axon_table.setRowCount(0)
        for candidate in candidates:
            row = self.batch_axon_table.rowCount()
            self.batch_axon_table.insertRow(row)
            values = [
                candidate.status,
                str(candidate.candidate_id),
                f"{candidate.confidence * 100:.0f}%",
                str(candidate.axon_area),
                str(candidate.object_area),
                f"{candidate.axon_fraction * 100:.1f}%",
                candidate.reason,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(256, int(candidate.candidate_id))
                self.batch_axon_table.setItem(row, col, item)
        self.batch_axon_table.setSortingEnabled(True)

    def _selected_batch_axon_ids(self) -> set[int]:
        selected: set[int] = set()
        selection = self.batch_axon_table.selectionModel()
        if selection is None:
            return selected
        for index in selection.selectedRows():
            item = self.batch_axon_table.item(index.row(), 1)
            if item is not None:
                selected.add(int(item.text()))
        return selected

    def _batch_axon_candidate(self, candidate_id: int) -> AxonHoleCandidate | None:
        for candidate in self._batch_axon_candidates:
            if candidate.candidate_id == int(candidate_id):
                return candidate
        return None

    def locate_selected_batch_axon(self) -> None:
        ids = self._selected_batch_axon_ids()
        if not ids:
            self._log("Select a batch axon proposal row to locate.")
            return
        self.locate_batch_axon(next(iter(sorted(ids))))

    def locate_batch_axon_from_cell(self, row: int, _column: int) -> None:
        item = self.batch_axon_table.item(row, 1)
        if item is None:
            return
        self.locate_batch_axon(int(item.text()))

    def locate_batch_axon(self, candidate_id: int) -> None:
        candidate = self._batch_axon_candidate(candidate_id)
        layer = self._target_layer()
        if candidate is None or layer is None:
            return
        data_position = self._scoped_position_to_layer_coords(candidate.seed, self._batch_preview_indexer, np.asarray(layer.data).ndim)
        self._center_view_on_data_position(layer, np.asarray(data_position, dtype=float))
        preview_layer = safe_get_layer(self.viewer, self._batch_preview_layer_name)
        if preview_layer is not None:
            preview_layer.visible = True
        self.status_label.setText(
            f"Located batch axon proposal {candidate.candidate_id}: {candidate.status}, "
            f"confidence {candidate.confidence * 100:.0f}%."
        )

    def _scoped_position_to_layer_coords(self, scoped: tuple[int, ...], indexer: object, ndim: int) -> tuple[int, ...]:
        if indexer is Ellipsis or not isinstance(indexer, tuple):
            return tuple(int(value) for value in scoped)
        out: list[int] = []
        scoped_axis = 0
        for selector in indexer:
            if isinstance(selector, int):
                out.append(int(selector))
            elif isinstance(selector, slice):
                start = 0 if selector.start is None else int(selector.start)
                out.append(start + int(scoped[scoped_axis]))
                scoped_axis += 1
            else:
                out.append(int(scoped[scoped_axis]))
                scoped_axis += 1
        while len(out) < ndim and scoped_axis < len(scoped):
            out.append(int(scoped[scoped_axis]))
            scoped_axis += 1
        return tuple(out)

    def _write_batch_preview_layer(self, target_layer, preview: np.ndarray, indexer: object) -> None:
        if preview.size == 0:
            return
        full = np.zeros_like(np.asarray(target_layer.data), dtype=np.uint32)
        if indexer is Ellipsis:
            full = preview.astype(np.uint32, copy=False)
        else:
            full[indexer] = preview.astype(np.uint32, copy=False)
        existing = safe_get_layer(self.viewer, self._batch_preview_layer_name)
        if existing is not None:
            existing.data = full
            existing.refresh()
            return
        name = unique_layer_name(self.viewer, "batch_axon_hole_preview")
        layer = self.viewer.add_labels(
            full,
            name=name,
            metadata={"sam3_role": "batch_axon_hole_preview", "created_from": "Mask Cleanup"},
        )
        copy_layer_geometry(target_layer, layer)
        self._batch_preview_layer_name = name

    def _full_data_from_scoped_result(self, target_layer, scoped_data: np.ndarray, indexer: object) -> np.ndarray:
        full = np.zeros_like(np.asarray(target_layer.data), dtype=np.uint32)
        if indexer is Ellipsis:
            return np.asarray(scoped_data, dtype=np.uint32)
        full[indexer] = np.asarray(scoped_data, dtype=np.uint32)
        return full

    def _refresh_batch_preview_layer(self) -> None:
        layer = self._target_layer()
        if layer is None or self._batch_preview_shape is None:
            return
        preview = np.zeros(self._batch_preview_shape, dtype=np.uint32)
        for candidate in self._batch_axon_candidates:
            if candidate.status == "skipped":
                continue
            mask = self._batch_axon_masks.get(candidate.candidate_id)
            if mask is None:
                continue
            slices = tuple(slice(int(lo), int(hi)) for lo, hi in candidate.bbox)
            preview_slice = preview[slices]
            preview_slice[mask] = int(candidate.candidate_id)
        self._write_batch_preview_layer(layer, preview, self._batch_preview_indexer)

    def _batch_roi_mask(self, label_layer, indexer: object, shape: tuple[int, ...]) -> np.ndarray | None:
        roi_layer = safe_get_layer(self.viewer, self.batch_roi_combo.currentData())
        if roi_layer is None:
            return None
        data = list(getattr(roi_layer, "data", []) or [])
        if not data:
            return None
        full_mask = np.zeros(np.asarray(label_layer.data).shape, dtype=bool)
        for vertices in data:
            arr = np.asarray(vertices)
            if arr.size == 0:
                continue
            coord_count = min(int(arr.shape[-1]), full_mask.ndim)
            coords = arr[..., -coord_count:]
            mins = np.floor(np.min(coords, axis=0)).astype(int)
            maxs = np.ceil(np.max(coords, axis=0)).astype(int) + 1
            slices = [slice(None)] * full_mask.ndim
            start_axis = full_mask.ndim - coord_count
            for offset, axis in enumerate(range(start_axis, full_mask.ndim)):
                size = full_mask.shape[axis]
                lo = max(0, min(size, int(mins[offset])))
                hi = max(lo, min(size, int(maxs[offset])))
                slices[axis] = slice(lo, hi)
            full_mask[tuple(slices)] = True
        if indexer is Ellipsis:
            roi = full_mask
        else:
            roi = full_mask[indexer]
        if roi.shape != tuple(shape) or not np.any(roi):
            return None
        return roi


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

    def _sync_mouse_action_callback(self) -> None:
        self._disconnect_mouse_action_callback()
        layer = self._target_layer()
        if layer is None:
            return

        installed = False
        drag_callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if drag_callbacks is not None:
            if self._mouse_callback not in drag_callbacks:
                drag_callbacks.append(self._mouse_callback)
            installed = True

        double_click_callbacks = getattr(layer, "mouse_double_click_callbacks", None)
        if double_click_callbacks is not None:
            if self._mouse_double_click_callback not in double_click_callbacks:
                double_click_callbacks.append(self._mouse_double_click_callback)
            installed = True

        if not installed:
            self._log("Selected Labels layer does not expose mouse callbacks.")
            return

        self._mouse_layer = layer
        try:
            self.viewer.layers.selection.active = layer
            layer.mode = "pick"
        except Exception:
            pass
        self.status_label.setText(self._mouse_mode_status())

    def _disconnect_mouse_action_callback(self) -> None:
        layer = self._mouse_layer
        if layer is None:
            return
        drag_callbacks = getattr(layer, "mouse_drag_callbacks", None)
        if drag_callbacks is not None and self._mouse_callback in drag_callbacks:
            drag_callbacks.remove(self._mouse_callback)
        double_click_callbacks = getattr(layer, "mouse_double_click_callbacks", None)
        if double_click_callbacks is not None and self._mouse_double_click_callback in double_click_callbacks:
            double_click_callbacks.remove(self._mouse_double_click_callback)
        self._mouse_layer = None

    def _handle_mouse_double_click(self, layer, event):
        if not self._mouse_enabled_for_layer(layer):
            return
        self._pick_clicked_mask(layer, event, source="double-click")

    def _handle_mouse_action(self, layer, event):
        if not self._mouse_enabled_for_layer(layer):
            return

        event_type = str(getattr(event, "type", "")).lower()
        if "double" in event_type:
            self._pick_clicked_mask(layer, event, source="double-click")
            return

        if self._is_right_mouse_event(event):
            self._neutralize_labels_mouse_tool(layer, event)
            try:
                self._open_canvas_context_menu(layer, event)
            finally:
                self._neutralize_labels_mouse_tool(layer, event)
            return
        mode = self._cleanup_subtab()
        if mode == "axon" and self.axon_cut_enable_check.isChecked():
            self._apply_mouse_action(layer, event, "cut_axon")
            return

        # Fixed, frictionless canvas behavior:
        # - double-click: select clicked mask/component row
        # - right-click: context menu for cleanup actions
        # - left-click/other click: select clicked mask/component row
        # No user-facing mouse-action or mouse-button dropdowns are needed.
        self._pick_clicked_mask(layer, event, source="click")

    def _mouse_enabled_for_layer(self, layer) -> bool:
        return layer is self._target_layer()

    def _neutralize_labels_mouse_tool(self, layer, event=None) -> None:
        """Leave labels context-menu actions in a non-painting mouse mode."""
        try:
            self.viewer.layers.selection.active = layer
        except Exception:
            pass
        try:
            layer.mode = "pick"
        except Exception:
            pass
        if event is not None:
            try:
                event.handled = True
            except Exception:
                pass
            native = getattr(event, "native", None)
            if native is not None:
                try:
                    native.accept()
                except Exception:
                    pass

    def _any_canvas_tool_enabled(self) -> bool:
        delete_enabled = bool(
            getattr(self, "mouse_action_enable_check", None) is not None
            and self.mouse_action_enable_check.isChecked()
        )
        assign_enabled = bool(
            getattr(self, "canvas_assign_enable_check", None) is not None
            and self.canvas_assign_enable_check.isChecked()
        )
        axon_enabled = bool(
            getattr(self, "axon_cut_enable_check", None) is not None
            and self.axon_cut_enable_check.isChecked()
        )
        return delete_enabled or assign_enabled or axon_enabled

    def _pick_clicked_mask(self, layer, event, *, source: str = "click") -> tuple[int, int | None] | None:
        coords = self._event_data_coords(layer, event)
        if coords is None:
            return None
        label_value = self._label_value_at_coords(layer, coords)
        if label_value <= 0:
            self.status_label.setText("Clicked background; no mask selected.")
            return None
        sub, indexer, _offset = self._scoped_data(layer)
        scoped_coords = self._coords_to_scoped_coords(layer, coords, indexer)
        if scoped_coords is None:
            self.status_label.setText("Clicked mask is outside the selected operation scope.")
            return None

        self._select_value_row(label_value)
        mode = self._cleanup_subtab()
        component_id = None
        if mode == "components":
            fast_index = self._fresh_fast_index(layer, sub, indexer)
            if fast_index is None:
                self.status_label.setText("Component index is stale or not built. Click Analyze Layer before component picking.")
                self._log("Component mouse pick skipped because the component index is stale or not built.")
                return label_value, None
            component_id = fast_index.component_id_at(scoped_coords)
            if component_id is not None:
                self.component_table.select_component_id(component_id)
        detail = f", component {component_id}" if component_id else ""
        self.status_label.setText(f"Selected value {label_value}{detail}.")
        self._log(f"{source.capitalize()} selected label value {label_value}{detail} in {layer.name}.")
        return label_value, component_id

    def _open_canvas_context_menu(self, layer, event) -> None:
        picked = self._pick_clicked_mask(layer, event, source="right-click")
        if picked is None:
            return
        label_value, component_id = picked
        mode = self._cleanup_subtab()
        menu = QMenu(self)
        quick_assign_actions: dict[object, int] = {}

        if mode == "values":
            relabel_value_action = menu.addAction(f"Relabel clicked value {label_value} to {int(self.new_value_spin.value())}")
            delete_value_action = menu.addAction(f"Delete clicked value {label_value}")
            keep_value_action = menu.addAction(f"Keep value {label_value} only")
            select_action = menu.addAction("Select value row only")
            selected_action = menu.exec_(self._event_global_position(event))
            if selected_action == relabel_value_action:
                self.values_to_replace_edit.setText(str(label_value))
                self.apply_relabel()
            elif selected_action == delete_value_action:
                self._apply_mouse_action(layer, event, "delete_value")
            elif selected_action == keep_value_action:
                self._apply_mouse_action(layer, event, "keep_value_only")
            elif selected_action == select_action:
                self._select_value_row(label_value)
            return

        if mode == "components":
            if component_id is None:
                self.status_label.setText("Component index is stale or not built. Click Analyze Layer before component actions.")
                return
            current_value = int(self.assignment_value_spin.value())
            assign_current_action = menu.addAction(f"Assign indexed component to {current_value}")
            assign_current_action.setToolTip("Relabel only the clicked connected component.")
            quick_menu = menu.addMenu("Assign clicked object to")
            for value in range(1, 7):
                action = quick_menu.addAction(str(value))
                action.setToolTip(f"Relabel only the clicked connected component to {value}.")
                quick_assign_actions[action] = value
            menu.addSeparator()
            delete_component_action = menu.addAction("Delete indexed component")
            delete_component_action.setToolTip("Set only the clicked connected component to background.")
            locate_action = menu.addAction("Select table row only")
            selected_action = menu.exec_(self._event_global_position(event))
            if selected_action == assign_current_action:
                self._apply_mouse_action(layer, event, "assign_component")
            elif selected_action in quick_assign_actions:
                self.set_assignment_value(quick_assign_actions[selected_action])
                self._apply_mouse_action(layer, event, "assign_component")
            elif selected_action == delete_component_action:
                self._apply_mouse_action(layer, event, "delete_component")
            elif selected_action == locate_action:
                self.component_table.select_component_id(component_id)
            return

        assign_local_action = menu.addAction(f"Assign clicked local object to {int(self.assignment_value_spin.value())}")
        delete_local_action = menu.addAction("Delete clicked local object")
        delete_value_action = menu.addAction(f"Delete all pixels/voxels with value {label_value}")
        select_action = menu.addAction("Select value row only")
        cut_axon_action = None
        if mode == "axon":
            target_text = (
                f"class {int(self.axon_value_spin.value())}"
                if self.axon_assign_class_check.isChecked()
                else "background"
            )
            menu.insertSeparator(assign_local_action)
            cut_axon_action = QAction(
                f"Cut axon from clicked point to {target_text}",
                menu,
            )
            cut_axon_action.setToolTip("Grow the seed-similar region from the clicked point inside the clicked mask.")
            menu.insertAction(
                assign_local_action,
                cut_axon_action,
            )
        selected_action = menu.exec_(self._event_global_position(event))
        if cut_axon_action is not None and selected_action == cut_axon_action:
            self._apply_mouse_action(layer, event, "cut_axon")
        elif selected_action == assign_local_action:
            self._apply_mouse_action(layer, event, "assign_local_component")
        elif selected_action == delete_local_action:
            self._apply_mouse_action(layer, event, "delete_local_component")
        elif selected_action == delete_value_action:
            self._apply_mouse_action(layer, event, "delete_value")
        elif selected_action == select_action:
            self._select_value_row(label_value)

    def _apply_mouse_action(self, layer, event, action: str) -> None:
        coords = self._event_data_coords(layer, event)
        if coords is None:
            return
        label_value = self._label_value_at_coords(layer, coords)
        if label_value <= 0:
            self._log("Clicked background; no mask action applied.")
            return

        sub, indexer, _offset = self._scoped_data(layer)
        scoped_coords = self._coords_to_scoped_coords(layer, coords, indexer)
        if scoped_coords is None:
            self._log("Clicked mask is outside the selected operation scope.")
            return

        if action == "pick":
            self._pick_clicked_mask(layer, event, source="click")
            return

        if action == "cut_axon":
            image = self._source_image_for_scoped_labels(layer, indexer)
            if image is None:
                self._log("Select a source image layer for axon hole cutting.")
                return
            output_value = int(self.axon_value_spin.value()) if self.axon_assign_class_check.isChecked() else 0
            self.status_label.setText(
                f"Cutting axon from click at {coords}: label value {label_value}, "
                f"target {'class ' + str(output_value) if output_value else 'background'}..."
            )
            self._log(
                f"Axon hole click accepted at {coords} in {layer.name}; "
                f"value {label_value}."
            )
            try:
                component_mask = self.cleanup.connected_label_mask(sub, scoped_coords, label_value)
                data, changed, axon_pixels = self.cleanup.cut_seed_similar_region(
                    sub,
                    image,
                    component_mask,
                    scoped_coords,
                    output_value=output_value,
                    threshold_percent=int(self.axon_threshold_spin.value()),
                    max_fraction_percent=int(self.axon_max_fraction_spin.value()),
                )
            except ValueError as exc:
                self._log(str(exc))
                return
            invalidate_fast_index = True
            target_text = f"class {output_value}" if output_value else "background"
            message = f"Cut clicked axon region to {target_text} ({axon_pixels} pixel(s)/voxel(s))"
        elif action in {"assign_local_component", "delete_local_component"}:
            component_mask = self.cleanup.connected_label_mask(sub, scoped_coords, label_value)
            data = sub.copy()
            if action == "assign_local_component":
                new_value = int(self.assignment_value_spin.value())
                changed = int(np.count_nonzero(component_mask & (data != new_value)))
                data[component_mask] = new_value
                message = f"Assigned clicked local object from value {label_value} to {new_value}"
            else:
                changed = int(np.count_nonzero(component_mask & (data != 0)))
                data[component_mask] = 0
                message = f"Deleted clicked local object with value {label_value}"
            invalidate_fast_index = True
        elif action == "delete_value":
            data, changed = self.cleanup.delete_values(sub, [label_value])
            invalidate_fast_index = True
            message = f"Deleted clicked value {label_value}"
        elif action == "keep_value_only":
            data, changed = self.cleanup.keep_values(sub, [label_value])
            invalidate_fast_index = True
            message = f"Kept clicked value {label_value} only"
        else:
            fast_index = self._fresh_fast_index(layer, sub, indexer)
            if fast_index is None:
                self._log("Component index is stale or not built. Click Analyze Layer before component actions.")
                self.status_label.setText("Component index is stale or not built. Click Analyze Layer before component actions.")
                return
            component_id = fast_index.component_id_at(scoped_coords)
            invalidate_fast_index = False

            if action == "assign_component":
                if component_id is None:
                    self._log("Clicked mask component is not indexed. Click Analyze Layer / Rebuild Index and try again.")
                    return
                new_value = int(self.assignment_value_spin.value())
                data, changed = fast_index.relabel_components(sub, [component_id], new_value)
                message = f"Assigned clicked component {component_id} from value {label_value} to {new_value}"
            elif action == "delete_component":
                if component_id is None:
                    self._log("Clicked mask component is not indexed. Click Analyze Layer / Rebuild Index and try again.")
                    return
                data, changed = fast_index.delete_components(sub, [component_id])
                message = f"Deleted clicked component {component_id} from value {label_value}"
            else:
                return

        if changed and self._replace_scoped_layer_data(
            layer,
            data,
            indexer,
            action.replace("_", " "),
            invalidate_fast_index=invalidate_fast_index,
        ):
            self._log(f"{message} in {layer.name} ({self._scope_label(layer)}; {changed} pixel(s)/voxel(s)).")
            if action in {"cut_axon", "assign_local_component", "delete_local_component", "delete_value", "keep_value_only"}:
                self._invalidate_fast_index()
                self.refresh_unique_values()
                self._refresh_all()
                self.status_label.setText(f"{message}. Component table is stale; click Analyze Layer to rebuild.")
            else:
                self.component_table.set_records(fast_index.active_records())
                self.refresh_unique_values()
                if action == "assign_component":
                    self._select_value_row(int(self.assignment_value_spin.value()))
                else:
                    self._select_value_row(label_value)
        else:
            self._log(f"{message} made no mask changes.")

    def _ensure_fast_index(self, layer, sub: np.ndarray, indexer: object) -> FastComponentIndex:
        scope_key = self._scope_key(layer, indexer)
        if (
            self._fast_index is None
            or self._fast_index_layer_id != id(layer)
            or self._fast_index_scope_key != scope_key
            or self._fast_index.shape != tuple(sub.shape)
        ):
            self._fast_index = self.fast_index_builder.build(sub)
            self._fast_index_layer_id = id(layer)
            self._fast_index_scope_key = scope_key
            self._last_analysis_indexer = indexer
            self.component_table.set_records(self._fast_index.active_records())
            self.status_label.setText(
                f"Fast index ready: {len(self._fast_index.active_records())} component(s) indexed."
            )
        return self._fast_index

    def _fresh_fast_index(self, layer, sub: np.ndarray, indexer: object) -> FastComponentIndex | None:
        scope_key = self._scope_key(layer, indexer)
        if (
            self._component_index_stale
            or self._fast_index is None
            or self._fast_index_layer_id != id(layer)
            or self._fast_index_scope_key != scope_key
            or self._fast_index.shape != tuple(sub.shape)
        ):
            return None
        return self._fast_index

    def _invalidate_fast_index(self) -> None:
        self._fast_index = None
        self._fast_index_layer_id = None
        self._fast_index_scope_key = None
        self._component_index_stale = True

    def _scope_key(self, layer, indexer: object) -> tuple:
        if indexer is Ellipsis:
            indexer_key = "all"
        elif isinstance(indexer, tuple):
            indexer_key = tuple(
                ("slice", selector.start, selector.stop, selector.step)
                if isinstance(selector, slice)
                else ("int", int(selector))
                if isinstance(selector, int)
                else ("other", repr(selector))
                for selector in indexer
            )
        else:
            indexer_key = repr(indexer)
        data = np.asarray(layer.data)
        return (id(layer), layer.name, tuple(data.shape), str(data.dtype), indexer_key)

    def _event_global_position(self, event):
        """Return a reliable global Qt position for a napari canvas mouse event."""
        native = getattr(event, "native", None)
        if native is not None:
            for attr in ("globalPosition", "globalPos"):
                try:
                    pos = getattr(native, attr)()
                    if hasattr(pos, "toPoint"):
                        return pos.toPoint()
                    return pos
                except Exception:
                    pass
        return QCursor.pos()

    def _is_left_mouse_event(self, event) -> bool:
        return self._normalized_mouse_button(event) in {"1", "left", "leftbutton", "mousebutton.left", "mousebutton.left_button"}

    def _is_right_mouse_event(self, event) -> bool:
        return self._normalized_mouse_button(event) in {"2", "right", "rightbutton", "mousebutton.right", "mousebutton.right_button"}

    def _normalized_mouse_button(self, event) -> str:
        button = getattr(event, "button", None)
        if button is None:
            return ""
        name = getattr(button, "name", None)
        if name:
            return str(name).lower().replace("_", "")
        return str(button).lower().replace(" ", "").replace("_", "")

    def _event_data_coords(self, layer, event) -> tuple[int, ...] | None:
        position = getattr(event, "position", None)
        if position is None:
            return None
        try:
            data_position = layer.world_to_data(position)
        except Exception:
            data_position = position
        data = np.asarray(layer.data)
        coords = tuple(int(round(float(value))) for value in data_position[-data.ndim :])
        if len(coords) != data.ndim:
            return None
        for coord, size in zip(coords, data.shape, strict=False):
            if coord < 0 or coord >= size:
                return None
        return coords

    def _label_value_at_coords(self, layer, coords: tuple[int, ...]) -> int:
        data = np.asarray(layer.data)
        if len(coords) != data.ndim:
            return 0
        for coord, size in zip(coords, data.shape, strict=False):
            if coord < 0 or coord >= size:
                return 0
        return int(data[coords])

    def _coords_to_scoped_coords(self, layer, coords: tuple[int, ...], indexer: object) -> tuple[int, ...] | None:
        data = np.asarray(layer.data)
        if indexer is Ellipsis:
            return coords
        if not isinstance(indexer, tuple):
            return coords
        scoped: list[int] = []
        for axis, selector in enumerate(indexer):
            coord = coords[axis]
            if isinstance(selector, slice):
                start = 0 if selector.start is None else int(selector.start)
                stop = data.shape[axis] if selector.stop is None else int(selector.stop)
                if coord < start or coord >= stop:
                    return None
                scoped.append(coord - start)
            elif isinstance(selector, int):
                if coord != selector:
                    return None
            else:
                scoped.append(coord)
        return tuple(scoped)

    def _select_value_row(self, label_value: int) -> bool:
        self.unique_values_table.clearSelection()
        for row in range(self.unique_values_table.rowCount()):
            item = self.unique_values_table.item(row, 0)
            if item is not None and int(item.text()) == int(label_value):
                self.unique_values_table.selectRow(row)
                self.unique_values_table.scrollToItem(item)
                return True
        return False

    def _select_component_row_at_scoped_coord(self, sub: np.ndarray, scoped_coords: tuple[int, ...]) -> int | None:
        layer = self._target_layer()
        if layer is None:
            return None
        fast_index = self._fresh_fast_index(layer, sub, self._last_analysis_indexer)
        if fast_index is None:
            return None
        component_id = fast_index.component_id_at(scoped_coords)
        if component_id is not None:
            self.component_table.select_component_id(component_id)
        return component_id

    def _build_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(6)
        target_form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.currentIndexChanged.connect(self._on_target_layer_changed)
        self.source_image_combo = QComboBox()
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
        self.canvas_assign_enable_check = QCheckBox("Enable canvas assign tools")
        self.canvas_assign_enable_check.setToolTip(
            "When on: right-click a mask to assign only the clicked object to the current class value."
        )
        self.canvas_assign_enable_check.setChecked(False)
        self.canvas_assign_enable_check.toggled.connect(lambda _checked: self._sync_mouse_action_callback())
        self.mouse_action_enable_check = QCheckBox("Enable canvas right-click delete tools")
        self.mouse_action_enable_check.setToolTip(
            "When on: double-click a mask to select it; right-click a mask to open delete actions."
        )
        self.mouse_action_enable_check.setChecked(False)
        self.mouse_action_enable_check.toggled.connect(lambda _checked: self._sync_mouse_action_callback())
        self.axon_cut_enable_check = QCheckBox("Enable axon hole click tool")
        self.axon_cut_enable_check.setToolTip(
            "Myelin/axon workflow only. When on: left-click inside the axon region "
            "to set the seed-similar inner region to background or an axon class."
        )
        self.axon_cut_enable_check.setChecked(False)
        self.axon_cut_enable_check.toggled.connect(lambda _checked: self._sync_mouse_action_callback())
        target_row = QHBoxLayout()
        target_row.addWidget(refresh_btn)
        target_row.addWidget(self.undo_btn)
        target_form.addRow("Target labels layer", self.target_combo)
        target_form.addRow("Source image layer", self.source_image_combo)
        target_form.addRow("Operation scope", self.scope_combo)
        target_form.addRow("Z range", z_row)
        target_form.addRow(target_row)
        root.addLayout(target_form)

        self.cleanup_tabs = QTabWidget()
        self.cleanup_tabs.currentChanged.connect(lambda _index: self._on_cleanup_subtab_changed())
        components_tab = QWidget()
        components_layout = QVBoxLayout()
        components_tab.setLayout(components_layout)
        local_tab = QWidget()
        local_layout = QVBoxLayout()
        local_tab.setLayout(local_layout)
        axon_tab = QWidget()
        axon_layout = QVBoxLayout()
        axon_tab.setLayout(axon_layout)
        values_tab = QWidget()
        values_layout = QVBoxLayout()
        values_tab.setLayout(values_layout)
        self.cleanup_tabs.addTab(components_tab, "Components")
        self.cleanup_tabs.addTab(local_tab, "Local Edit")
        self.cleanup_tabs.addTab(axon_tab, "Myelin / Axon Rings")
        self.cleanup_tabs.addTab(values_tab, "Values")
        root.addWidget(self.cleanup_tabs)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("Component analysis idle")
        region_form = QFormLayout()
        self.work_region_combo = QComboBox()
        self.work_region_combo.addItem("Full mask", "full")
        self.work_region_combo.addItem("Manual ROI", "manual")
        self.work_region_combo.addItem("Drawn ROI", "drawn")
        self.work_region_combo.setToolTip(
            "Choose the part of the target mask analyzed by Components. "
            "ROI modes keep analysis fast on very large masks and write edits back to the original layer."
        )
        self.work_region_combo.currentIndexChanged.connect(self._on_work_region_changed)
        self.work_roi_combo = QComboBox()
        self.work_roi_combo.setToolTip("Shapes layer used as the working region when Work on is Drawn ROI.")
        self.work_roi_combo.currentIndexChanged.connect(lambda _index: self._on_work_region_changed())
        self.work_y0_spin = QSpinBox()
        self.work_y1_spin = QSpinBox()
        self.work_x0_spin = QSpinBox()
        self.work_x1_spin = QSpinBox()
        for spin in (self.work_y0_spin, self.work_y1_spin, self.work_x0_spin, self.work_x1_spin):
            spin.setRange(0, 2_147_483_647)
            spin.valueChanged.connect(lambda _value: self._on_work_region_changed())
        y_row = QHBoxLayout()
        y_row.addWidget(self.work_y0_spin)
        y_row.addWidget(QLabel("to"))
        y_row.addWidget(self.work_y1_spin)
        x_row = QHBoxLayout()
        x_row.addWidget(self.work_x0_spin)
        x_row.addWidget(QLabel("to"))
        x_row.addWidget(self.work_x1_spin)
        region_form.addRow("Work on", self.work_region_combo)
        region_form.addRow("ROI shape", self.work_roi_combo)
        region_form.addRow("Y", y_row)
        region_form.addRow("X", x_row)
        components_layout.addWidget(QLabel("Working Region"))
        components_layout.addLayout(region_form)
        components_header = QHBoxLayout()
        components_header.addWidget(analyze_btn)
        components_header.addWidget(delete_btn)
        components_header.addStretch(1)
        components_layout.addLayout(components_header)
        components_layout.addWidget(self.analysis_progress)

        self.component_table = ComponentTableWidget(
            delete_callback=self.delete_selected_components,
            assign_callback=self.assign_selected_components_to_current_value,
            locate_callback=self.locate_component,
        )
        self.component_table.setMinimumHeight(170)
        components_layout.addWidget(self.component_table)

        assignment = QHBoxLayout()
        assignment.addWidget(QLabel("Assignment value"))
        self.assignment_value_spin = QSpinBox()
        self.assignment_value_spin.setRange(0, 2_147_483_647)
        self.assignment_value_spin.setValue(1)
        self.assignment_value_spin.setToolTip("Class value to assign to selected or clicked objects.")
        assignment.addWidget(self.assignment_value_spin)
        for value in range(1, 7):
            button = QPushButton(str(value))
            button.setToolTip(f"Set assignment value to {value}.")
            button.clicked.connect(lambda _checked=False, value=value: self.set_assignment_value(value))
            assignment.addWidget(button)
        assign_selected_btn = QPushButton("Assign Selected Components")
        assign_selected_btn.setToolTip("Relabel selected component rows to the current assignment value.")
        assign_selected_btn.clicked.connect(self.assign_selected_components_to_current_value)
        assignment.addWidget(assign_selected_btn)
        assignment.addStretch(1)
        root.addLayout(assignment)

        local_mouse_row = QHBoxLayout()
        local_mouse_row.addWidget(self.canvas_assign_enable_check)
        local_mouse_row.addWidget(self.mouse_action_enable_check)
        local_mouse_row.addStretch(1)
        local_layout.addLayout(local_mouse_row)

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
        components_layout.addLayout(operations)

        axon_note = QLabel(
            "One-click myelin/axon output for ring-shaped masks. This workflow creates new "
            "myelin and axon Labels layers and does not edit the original target layer."
        )
        axon_note.setWordWrap(True)
        axon_layout.addWidget(axon_note)
        axon_primary_row = QHBoxLayout()
        self.create_myelin_axon_btn = QPushButton("Create Myelin + Axon Layers")
        self.create_myelin_axon_btn.setToolTip(
            "Run automatic axon-hole detection, keep confident proposals, and create "
            "separate myelin_rings and axons Labels layers without changing the source labels."
        )
        self.create_myelin_axon_btn.clicked.connect(self.create_myelin_axon_layers)
        axon_primary_row.addWidget(self.create_myelin_axon_btn)
        axon_primary_row.addWidget(self.axon_cut_enable_check)
        axon_primary_row.addStretch(1)
        axon_layout.addLayout(axon_primary_row)

        axon_row = QHBoxLayout()
        axon_row.addWidget(QLabel("Seed tolerance %"))
        self.axon_threshold_spin = QSpinBox()
        self.axon_threshold_spin.setRange(0, 100)
        self.axon_threshold_spin.setValue(35)
        self.axon_threshold_spin.setToolTip(
            "Higher values grow farther from the clicked seed intensity before stopping at a different-intensity rim."
        )
        axon_row.addWidget(self.axon_threshold_spin)
        axon_row.addWidget(QLabel("Max axon %"))
        self.axon_max_fraction_spin = QSpinBox()
        self.axon_max_fraction_spin.setRange(1, 100)
        self.axon_max_fraction_spin.setValue(75)
        self.axon_max_fraction_spin.setToolTip("Reject the cut if the detected axon exceeds this share of the clicked mask.")
        axon_row.addWidget(self.axon_max_fraction_spin)
        self.axon_assign_class_check = QCheckBox("Assign axon class")
        self.axon_assign_class_check.setChecked(False)
        self.axon_assign_class_check.setToolTip("When off, the detected axon is set to background value 0.")
        axon_row.addWidget(self.axon_assign_class_check)
        axon_row.addWidget(QLabel("Axon value"))
        self.axon_value_spin = QSpinBox()
        self.axon_value_spin.setRange(1, 2_147_483_647)
        self.axon_value_spin.setValue(2)
        self.axon_value_spin.setToolTip("Class value used when Assign axon class is enabled.")
        axon_row.addWidget(self.axon_value_spin)
        axon_row.addStretch(1)
        axon_layout.addLayout(axon_row)

        axon_filter_row = QHBoxLayout()
        axon_filter_row.addWidget(QLabel("Min object"))
        self.batch_min_object_spin = QSpinBox()
        self.batch_min_object_spin.setRange(1, 2_147_483_647)
        self.batch_min_object_spin.setValue(32)
        self.batch_min_object_spin.setToolTip(
            "Ignore candidate objects smaller than this many pixels/voxels."
        )
        axon_filter_row.addWidget(self.batch_min_object_spin)
        axon_filter_row.addWidget(QLabel("Min confidence %"))
        self.batch_min_confidence_spin = QSpinBox()
        self.batch_min_confidence_spin.setRange(0, 100)
        self.batch_min_confidence_spin.setValue(70)
        self.batch_min_confidence_spin.setToolTip(
            "Confidence cutoff for the one-click output and Apply Confident. "
            "Lower values accept more proposals; higher values are more conservative."
        )
        axon_filter_row.addWidget(self.batch_min_confidence_spin)
        axon_filter_row.addWidget(QLabel("ROI shape"))
        self.batch_roi_combo = QComboBox()
        self.batch_roi_combo.setToolTip("Optional Shapes layer; candidates are limited to objects whose seed is inside the ROI bbox.")
        axon_filter_row.addWidget(self.batch_roi_combo)
        axon_filter_row.addStretch(1)
        axon_layout.addLayout(axon_filter_row)

        advanced_review = QWidget()
        advanced_review_layout = QVBoxLayout()
        advanced_review.setLayout(advanced_review_layout)

        batch_row = QHBoxLayout()
        self.batch_preview_btn = QPushButton("Preview Batch Axons")
        self.batch_preview_btn.setToolTip(
            "Scan the current scope for myelinated objects and create proposed axon holes. "
            "This only updates the preview layer and table; it does not edit the target labels."
        )
        self.batch_preview_btn.clicked.connect(self.preview_batch_axons)
        self.batch_apply_btn = QPushButton("Apply Confident")
        self.batch_apply_btn.setToolTip(
            "Apply all non-skipped proposals whose confidence is at or above Min confidence %. "
            "This edits the target labels and can be undone with Undo Last Edit."
        )
        self.batch_apply_btn.clicked.connect(self.apply_confident_batch_axons)
        self.batch_apply_selected_btn = QPushButton("Apply Selected")
        self.batch_apply_selected_btn.setToolTip(
            "Apply only the selected proposal rows, even if they are below Min confidence %. "
            "Use this after visually checking specific rows."
        )
        self.batch_apply_selected_btn.clicked.connect(self.apply_selected_batch_axons)
        self.batch_locate_btn = QPushButton("Locate Selected")
        self.batch_locate_btn.setToolTip(
            "Center the napari view on the selected proposal and show its preview layer. "
            "This does not edit labels."
        )
        self.batch_locate_btn.clicked.connect(self.locate_selected_batch_axon)
        self.batch_fill_selected_btn = QPushButton("Fill Selected Holes")
        self.batch_fill_selected_btn.setToolTip(
            "Repair selected proposals by filling holes inside the proposed axon mask. "
            "Use when the preview axon contains small myelin/noise gaps; this updates the preview only."
        )
        self.batch_fill_selected_btn.clicked.connect(self.fill_selected_batch_axon_holes)
        self.batch_skip_selected_btn = QPushButton("Skip Selected")
        self.batch_skip_selected_btn.setToolTip(
            "Mark selected proposals as skipped and remove them from the preview. "
            "Skipped proposals are not applied by Apply Confident."
        )
        self.batch_skip_selected_btn.clicked.connect(self.skip_selected_batch_axons)
        self.batch_skip_low_btn = QPushButton("Skip Low Confidence")
        self.batch_skip_low_btn.setToolTip(
            "Mark all proposals below Min confidence % as skipped and remove them from the preview. "
            "Use this to hide obvious uncertain candidates before reviewing the rest."
        )
        self.batch_skip_low_btn.clicked.connect(self.skip_low_confidence_batch_axons)
        self.batch_select_visible_btn = QPushButton("Select Visible")
        self.batch_select_visible_btn.setToolTip("Select every currently visible table row after sorting or filtering.")
        self.batch_select_visible_btn.clicked.connect(self.select_visible_batch_axons)
        self.batch_export_selected_btn = QPushButton("Export Selected")
        self.batch_export_selected_btn.setToolTip(
            "Create a new Labels layer from the selected proposal masks. "
            "This does not edit the target labels."
        )
        self.batch_export_selected_btn.clicked.connect(self.export_selected_batch_axon_layer)
        batch_row.addWidget(self.batch_preview_btn)
        batch_row.addWidget(self.batch_apply_btn)
        batch_row.addWidget(self.batch_apply_selected_btn)
        batch_row.addWidget(self.batch_locate_btn)
        batch_row.addWidget(self.batch_fill_selected_btn)
        batch_row.addWidget(self.batch_skip_selected_btn)
        batch_row.addWidget(self.batch_skip_low_btn)
        batch_row.addWidget(self.batch_select_visible_btn)
        batch_row.addWidget(self.batch_export_selected_btn)
        batch_row.addStretch(1)
        advanced_review_layout.addLayout(batch_row)

        batch_status_row = QHBoxLayout()
        for status in ("confident", "review", "failed", "skipped"):
            select_btn = QPushButton(f"Select {status.title()}")
            select_btn.setToolTip(f"Select all visible proposal rows with status {status}.")
            select_btn.clicked.connect(lambda _checked=False, status=status: self.select_batch_axons_by_status(status))
            export_btn = QPushButton(f"Export {status.title()}")
            export_btn.setToolTip(f"Create a new Labels layer containing proposal masks with status {status}.")
            export_btn.clicked.connect(lambda _checked=False, status=status: self.export_batch_axon_layer_by_status(status))
            batch_status_row.addWidget(select_btn)
            batch_status_row.addWidget(export_btn)
        batch_status_row.addStretch(1)
        advanced_review_layout.addLayout(batch_status_row)

        self.batch_axon_table = QTableWidget(0, 7)
        self.batch_axon_table.setHorizontalHeaderLabels(
            ["Status", "Object", "Confidence", "Axon px", "Object px", "Axon %", "Reason"]
        )
        self.batch_axon_table.setAlternatingRowColors(True)
        self.batch_axon_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_axon_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_axon_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_axon_table.setSortingEnabled(True)
        self.batch_axon_table.cellClicked.connect(self.locate_batch_axon_from_cell)
        self.batch_axon_table.verticalHeader().setDefaultSectionSize(24)
        self.batch_axon_table.verticalHeader().setMinimumSectionSize(22)
        self.batch_axon_table.setMinimumHeight(140)
        self.batch_axon_table.setStyleSheet(
            """
            QTableWidget {
                background: #1f242c;
                alternate-background-color: #2a3038;
                color: #eef2f7;
                gridline-color: #3b444f;
                selection-background-color: #2f6f8f;
                selection-color: #ffffff;
            }
            QTableWidget::item {
                padding: 3px 4px;
            }
            QTableWidget::item:selected {
                background: #2f6f8f;
                color: #ffffff;
            }
            QHeaderView::section {
                background: #303741;
                color: #f4f7fb;
                border: 1px solid #3b444f;
                padding: 3px 4px;
            }
            """
        )
        advanced_review_layout.addWidget(self.batch_axon_table)
        axon_layout.addWidget(CollapsiblePanel("Advanced review", advanced_review, collapsed=True))

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
        values_layout.addWidget(self.unique_values_table)
        self.status_label = QLabel("Canvas assignment/delete tools are off.")
        root.addWidget(self.status_label)
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
        values_layout.addLayout(relabel_form)
        self.setLayout(root)

    def _add_operation_row(self, layout: QGridLayout, row: int, label_text: str, spin_box: QSpinBox, button: QPushButton) -> None:
        label = QLabel(label_text)
        spin_box.setMinimumWidth(84)
        spin_box.setMaximumWidth(140)
        button.setMinimumWidth(180)
        layout.addWidget(label, row, 0)
        layout.addWidget(spin_box, row, 1)
        layout.addWidget(button, row, 2)

    def _cleanup_subtab(self) -> str:
        tabs = getattr(self, "cleanup_tabs", None)
        if tabs is None:
            return "components"
        return {0: "components", 1: "local", 2: "axon", 3: "values"}.get(int(tabs.currentIndex()), "components")

    def _on_cleanup_subtab_changed(self) -> None:
        self._sync_mouse_action_callback()
        if hasattr(self, "status_label"):
            self.status_label.setText(self._mouse_mode_status())

    def _mouse_mode_status(self) -> str:
        mode = self._cleanup_subtab()
        if mode == "local":
            return "Mouse mode: Local Edit. Click tools use local flood-fill and do not rebuild the component table."
        if mode == "axon":
            return "Mouse mode: Myelin / Axon Rings. Axon click tools are opt-in and specific to ring-shaped masks."
        if mode == "components":
            return "Mouse mode: Components. Requires a fresh Analyze Layer index for component row selection/actions."
        return "Mouse mode: Values. Clicks select or edit label values without component analysis."

    def _apply_scoped_cleanup(self, callback, success_prefix: str, action: str) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        data = callback(sub)
        if self._replace_scoped_layer_data(layer, data, indexer, action):
            self._log(f"{success_prefix} in {layer.name} ({self._scope_label(layer)}).")
            self.refresh_unique_values()
            self._refresh_all()
            self.status_label.setText(f"{success_prefix}. Component table is stale; click Analyze Layer to rebuild.")
        else:
            self._log(f"{action} made no mask changes.")

    def _target_layer(self):
        return safe_get_layer(self.viewer, self.target_combo.currentData())

    def _on_target_layer_changed(self, _index: int) -> None:
        self._invalidate_fast_index()
        self.component_table.set_records([])
        self._sync_scope_controls()
        self._sync_work_region_controls()
        self.refresh_unique_values()
        self._track_target_layer()
        self._sync_mouse_action_callback()
        self._update_undo_state()

    def _on_scope_changed(self, _index: int) -> None:
        self._invalidate_fast_index()
        self._sync_scope_controls()
        self.refresh_unique_values()

    def _on_work_region_changed(self, _index: int | None = None) -> None:
        self._invalidate_fast_index()
        self._sync_work_region_controls()
        self.component_table.set_records([])
        if hasattr(self, "status_label"):
            self.status_label.setText("Working region changed. Click Analyze Layer to rebuild the component table.")

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

    def _sync_work_region_controls(self) -> None:
        if not hasattr(self, "work_region_combo"):
            return
        layer = self._target_layer()
        data = np.asarray(layer.data) if layer is not None else None
        height = int(data.shape[-2]) if data is not None and data.ndim >= 2 else 0
        width = int(data.shape[-1]) if data is not None and data.ndim >= 2 else 0
        mode = self.work_region_combo.currentData()
        manual = mode == "manual"
        drawn = mode == "drawn"
        for spin, maximum in (
            (self.work_y0_spin, height),
            (self.work_y1_spin, height),
            (self.work_x0_spin, width),
            (self.work_x1_spin, width),
        ):
            old = min(int(spin.value()), maximum)
            spin.blockSignals(True)
            spin.setRange(0, maximum)
            spin.setValue(old)
            spin.setEnabled(manual and maximum > 0)
            spin.blockSignals(False)
        if manual:
            if int(self.work_y1_spin.value()) <= int(self.work_y0_spin.value()) and height > 0:
                self.work_y1_spin.blockSignals(True)
                self.work_y1_spin.setValue(height)
                self.work_y1_spin.blockSignals(False)
            if int(self.work_x1_spin.value()) <= int(self.work_x0_spin.value()) and width > 0:
                self.work_x1_spin.blockSignals(True)
                self.work_x1_spin.setValue(width)
                self.work_x1_spin.blockSignals(False)
        self.work_roi_combo.setEnabled(drawn)

    def _scoped_data(self, layer) -> tuple[np.ndarray, object, tuple[int, ...]]:
        arr = np.asarray(layer.data)
        indexer: list[object] = [slice(None)] * arr.ndim
        offset = [0] * arr.ndim
        reduced_axes: set[int] = set()
        if arr.ndim >= 3:
            scope = self.scope_combo.currentData()
            z_axis = arr.ndim - 3
            if scope == "current_slice":
                z = self._current_z(arr, z_axis)
                indexer[z_axis] = z
                offset[z_axis] = z
                reduced_axes.add(z_axis)
            elif scope == "z_range":
                z0 = min(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
                z1 = max(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
                indexer[z_axis] = slice(z0, z1 + 1)
                offset[z_axis] = z0
        work_region = self._work_region_slices(arr)
        if work_region is not None:
            y_slice, x_slice = work_region
            indexer[-2] = y_slice
            indexer[-1] = x_slice
            offset[-2] = 0 if y_slice.start is None else int(y_slice.start)
            offset[-1] = 0 if x_slice.start is None else int(x_slice.start)
        if all(selector == slice(None) for selector in indexer):
            return arr.copy(), Ellipsis, tuple(0 for _ in range(arr.ndim))
        scoped_indexer = tuple(indexer)
        scoped_offset = tuple(value for axis, value in enumerate(offset) if axis not in reduced_axes)
        return arr[scoped_indexer].copy(), scoped_indexer, scoped_offset

    def _work_region_slices(self, arr: np.ndarray) -> tuple[slice, slice] | None:
        if arr.ndim < 2 or not hasattr(self, "work_region_combo"):
            return None
        mode = self.work_region_combo.currentData()
        if mode == "manual":
            y0 = int(self.work_y0_spin.value())
            y1 = int(self.work_y1_spin.value())
            x0 = int(self.work_x0_spin.value())
            x1 = int(self.work_x1_spin.value())
        elif mode == "drawn":
            bounds = self._drawn_work_region_bounds(arr)
            if bounds is None:
                return None
            y0, x0, y1, x1 = bounds
        else:
            return None
        height, width = int(arr.shape[-2]), int(arr.shape[-1])
        y0, y1 = sorted((max(0, min(height, y0)), max(0, min(height, y1))))
        x0, x1 = sorted((max(0, min(width, x0)), max(0, min(width, x1))))
        if y1 <= y0 or x1 <= x0:
            return None
        if y0 == 0 and x0 == 0 and y1 == height and x1 == width:
            return None
        return slice(y0, y1), slice(x0, x1)

    def _drawn_work_region_bounds(self, arr: np.ndarray) -> tuple[int, int, int, int] | None:
        roi_layer = safe_get_layer(self.viewer, self.work_roi_combo.currentData())
        if roi_layer is None:
            return None
        data = list(getattr(roi_layer, "data", []) or [])
        if not data:
            return None
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for vertices in data:
            points = np.asarray(vertices)
            if points.size == 0:
                continue
            coord_count = min(int(points.shape[-1]), arr.ndim)
            coords = points[..., -coord_count:]
            yx = coords[..., -2:]
            mins.append(np.floor(np.min(yx, axis=0)).astype(int))
            maxs.append(np.ceil(np.max(yx, axis=0)).astype(int) + 1)
        if not mins:
            return None
        lo = np.min(np.stack(mins, axis=0), axis=0)
        hi = np.max(np.stack(maxs, axis=0), axis=0)
        return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])

    def _source_image_for_scoped_labels(self, label_layer, indexer: object) -> np.ndarray | None:
        image_layer = safe_get_layer(self.viewer, self.source_image_combo.currentData())
        if image_layer is None:
            return None
        image = np.asarray(image_layer.data)
        labels = np.asarray(label_layer.data)
        if indexer is Ellipsis:
            return image
        if not isinstance(indexer, tuple):
            return image
        if image.shape == labels.shape:
            return image[indexer]
        if image.ndim == labels.ndim + 1 and image.shape[-1] in (3, 4) and image.shape[:-1] == labels.shape:
            return image[indexer + (slice(None),)]
        if image.ndim == labels.ndim + 1 and image.shape[0] in (3, 4) and image.shape[1:] == labels.shape:
            return image[(slice(None),) + indexer]
        scoped_shape = np.asarray(label_layer.data[indexer]).shape
        if image.shape == scoped_shape:
            return image
        return image

    def _replace_scoped_layer_data(
        self,
        layer,
        scoped_data: np.ndarray,
        indexer: object,
        action: str,
        *,
        invalidate_fast_index: bool = True,
    ) -> bool:
        current = np.asarray(layer.data)
        if indexer is Ellipsis:
            updated = np.asarray(scoped_data)
            return self._replace_layer_data(layer, updated, action, invalidate_fast_index=invalidate_fast_index)
        else:
            return self._replace_layer_region_data(
                layer,
                scoped_data,
                indexer,
                action,
                invalidate_fast_index=invalidate_fast_index,
            )

    def _replace_layer_region_data(
        self,
        layer,
        scoped_data: np.ndarray,
        indexer: object,
        action: str,
        *,
        invalidate_fast_index: bool = True,
    ) -> bool:
        current = np.asarray(layer.data)
        previous_region = np.asarray(current[indexer]).copy()
        updated_region = np.asarray(scoped_data)
        if previous_region.shape == updated_region.shape and np.array_equal(previous_region, updated_region):
            return False
        self._append_region_undo_state(layer, indexer, previous_region, action)
        self._suppress_history_event = True
        try:
            current[indexer] = updated_region
            layer.refresh()
            self._last_layer_data[id(layer)] = self._snapshot_layer_data(layer)
        finally:
            self._suppress_history_event = False
        self._update_undo_state()
        if invalidate_fast_index:
            self._invalidate_fast_index()
        return True

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
        region = self._work_region_label(data)
        if data.ndim < 3:
            return f"2D layer{region}"
        scope = self.scope_combo.currentData()
        z_axis = data.ndim - 3
        if scope == "whole_volume":
            return f"whole volume{region}"
        if scope == "current_slice":
            return f"Z={self._current_z(data, z_axis)}{region}"
        z0 = min(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        z1 = max(int(self.z_start_spin.value()), int(self.z_end_spin.value()))
        return f"Z={z0}-{z1}{region}"

    def _work_region_label(self, data: np.ndarray) -> str:
        region = self._work_region_slices(data)
        if region is None:
            return ""
        y_slice, x_slice = region
        return f", ROI y={y_slice.start}:{y_slice.stop}, x={x_slice.start}:{x_slice.stop}"

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
            if isinstance(previous, tuple) and len(previous) == 3 and previous[0] == "region":
                _tag, indexer, region = previous
                data = np.asarray(layer.data)
                data[indexer] = region
            else:
                layer.data = previous
            layer.refresh()
            self._last_layer_data[id(layer)] = self._snapshot_layer_data(layer)
        finally:
            self._suppress_history_event = False
        self._invalidate_fast_index()
        self._log(f"Undid last Mask Cleanup edit on {layer.name}.")
        self.refresh_unique_values()
        self._refresh_all()
        self.status_label.setText(f"Undo restored {layer.name}. Component table is stale; click Analyze Layer to rebuild.")
        self.analysis_progress.setValue(0)
        self.analysis_progress.setFormat("Undo complete - component index stale")
        self._update_undo_state()

    def _replace_layer_data(self, layer, data, action: str, *, invalidate_fast_index: bool = True) -> bool:
        current = np.asarray(layer.data)
        updated = np.asarray(data)
        if current.shape == updated.shape and np.array_equal(current, updated):
            return False
        self._push_undo_state(layer, action)
        self._suppress_history_event = True
        try:
            layer.data = updated
            layer.refresh()
            self._last_layer_data[id(layer)] = self._snapshot_layer_data(layer)
        finally:
            self._suppress_history_event = False
        self._update_undo_state()
        if invalidate_fast_index:
            self._invalidate_fast_index()
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

    def _append_region_undo_state(self, layer, indexer: object, region: np.ndarray, action: str) -> None:
        history = self._undo_history.setdefault(id(layer), [])
        history.append(("region", indexer, np.asarray(region).copy()))
        if len(history) > UNDO_HISTORY_LIMIT:
            del history[0 : len(history) - UNDO_HISTORY_LIMIT]
        self._log(f"Saved ROI undo point for {layer.name}: {action}.")
        self._update_undo_state()

    def _snapshot_layer_data(self, layer) -> np.ndarray | None:
        data = np.asarray(layer.data)
        if data.size > UNDO_FULL_SNAPSHOT_PIXEL_LIMIT:
            return None
        return data.copy()

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
        self._last_layer_data[id(layer)] = self._snapshot_layer_data(layer)
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
        current = self._snapshot_layer_data(layer)
        if previous is None:
            self._last_layer_data[layer_id] = current
            return
        if current is None:
            return
        if previous.shape == current.shape and np.array_equal(previous, current):
            return
        self._append_undo_state(layer, previous, "manual label edit")
        self._last_layer_data[layer_id] = current
        self._invalidate_fast_index()

    def locate_component(self, component_id: int) -> None:
        layer = self._target_layer()
        if layer is None:
            self._log("Select a target Labels layer before locating a component.")
            return
        sub, indexer, _offset = self._scoped_data(layer)
        fast_index = self._fresh_fast_index(layer, sub, indexer)
        if fast_index is None:
            self._log("Component index is stale or not built. Click Analyze Layer before locating components.")
            return
        mask = fast_index.component_mask(component_id)
        record = fast_index.record(component_id)
        if mask is None or record is None:
            self._log("Analyze/Rebuild the fast index before locating this component.")
            return
        coords = np.argwhere(mask)
        if coords.size == 0:
            self._log(f"Component {component_id} is empty or no longer exists.")
            return
        bbox_start = np.asarray([lo for lo, _hi in record.bbox], dtype=float)
        data_position = coords.mean(axis=0) + bbox_start
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
