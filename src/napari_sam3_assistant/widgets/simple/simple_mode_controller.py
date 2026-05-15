from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QFileDialog

from napari.layers import Image
import torch

from ...core.models import Sam3Task
from ...device_utils import runtime_device
from ..advanced.advanced_mode_panel import (
    PROMPT_BOX,
    PROMPT_LABELS,
    PROMPT_POINTS,
    PROMPT_TEXT,
)
from ..shared.shared_context import SharedContext

SIMPLE_MODEL_DIR_KEY = "simple_model_dir"
SIMPLE_DEVICE_KEY = "simple_device"


class SimpleModeController(QObject):
    state_changed = Signal()

    def __init__(self, shared_context: SharedContext, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.shared_context = shared_context
        self._live_accept_undo: list[np.ndarray] = []

    @property
    def owner(self) -> Any:
        router = self.shared_context.task_router
        if router is None:
            return None
        return router.execution_owner()

    def refresh(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_refresh_layers"):
            owner._refresh_layers(silent=True)
        self.state_changed.emit()

    def refresh_from_viewer(self, *, prefer_active: bool = False) -> None:
        if prefer_active:
            self.select_active_image_layer()
        self.refresh()

    def select_active_image_layer(self) -> bool:
        viewer = self.shared_context.viewer
        if viewer is None:
            return False
        active = getattr(getattr(viewer.layers, "selection", None), "active", None)
        if not self._is_image_layer(active):
            return False
        self.set_image_layer(str(active.name))
        return True

    def image_layer_names(self) -> list[str]:
        owner = self.owner
        if owner is None or not hasattr(owner, "_layer_names"):
            return []
        return list(owner._layer_names({"image"}))

    def first_image_layer_name(self) -> str:
        names = self.image_layer_names()
        return names[0] if names else ""

    def current_image_layer_name(self) -> str:
        owner = self.owner
        if owner is None or not hasattr(owner, "_current_image_layer_name"):
            return ""
        return str(owner._current_image_layer_name() or "")

    def set_image_layer(self, layer_name: str) -> None:
        owner = self.owner
        if owner is None or not layer_name:
            return
        combo = getattr(owner, "image_layer_combo", None)
        if combo is not None:
            self._select_combo_data(combo, layer_name)
        self.state_changed.emit()

    def _is_image_layer(self, layer: Any) -> bool:
        if layer is None:
            return False
        if isinstance(layer, Image):
            return True
        layer_type = layer.__class__.__name__.lower()
        type_string = str(getattr(layer, "_type_string", "")).lower()
        return layer_type == "image" or type_string == "image"

    def image_summary(self) -> str:
        owner = self.owner
        viewer = self.shared_context.viewer
        layer_name = self.current_image_layer_name()
        if viewer is None or not layer_name:
            return "No image layer selected."
        try:
            layer = viewer.layers[layer_name]
        except Exception:
            return "Selected image layer is not available."
        data = getattr(layer, "data", None)
        shape = tuple(getattr(data, "shape", ()) or ())
        if not shape:
            return f"{layer_name}: image data shape unavailable."
        task_label = self.task_label(self.current_task())
        shape_text = " x ".join(str(int(value)) for value in shape)
        return f"{layer_name}: {shape_text}. Current task: {task_label}."

    def current_task(self) -> Sam3Task:
        owner = self.owner
        if owner is None or not hasattr(owner, "_current_task"):
            return Sam3Task.SEGMENT_2D
        return owner._current_task()

    def set_task(self, task: Sam3Task) -> None:
        owner = self.owner
        if owner is None:
            return
        changed = self.current_task() != task
        combo = getattr(owner, "task_combo", None)
        if combo is not None:
            self._select_combo_data(combo, task)
        if task in {Sam3Task.REFINE, Sam3Task.EXEMPLAR}:
            self.create_prompt_layer()
        if changed:
            self.state_changed.emit()

    def configure_task(
        self,
        task: Sam3Task,
        *,
        prompt_tool: str | None = None,
        large_image: bool | None = None,
        model_type: str | None = None,
    ) -> None:
        self.set_task(task)
        if prompt_tool is not None:
            self.set_prompt_tool(prompt_tool)
        if large_image is not None:
            self.set_large_image_enabled(large_image)
        self.set_model_type(model_type or self._model_type_for_task(task))

    def set_model_type(self, model_type: str) -> None:
        owner = self.owner
        combo = getattr(owner, "model_type_combo", None)
        changed = combo is not None and combo.currentData() != model_type
        if combo is not None:
            self._select_combo_data(combo, model_type)
        if owner is not None and hasattr(owner, "_sync_model_type_controls"):
            owner._sync_model_type_controls()
        if changed:
            self.state_changed.emit()

    def _model_type_for_task(self, task: Sam3Task) -> str:
        return "sam3.1" if task == Sam3Task.SEGMENT_3D else "sam3"

    def current_prompt_tool(self) -> str:
        owner = self.owner
        combo = getattr(owner, "prompt_tool_combo", None)
        if combo is None:
            return PROMPT_POINTS
        return combo.currentData() or PROMPT_POINTS

    def set_prompt_tool(self, tool: str) -> None:
        owner = self.owner
        if owner is None:
            return
        changed = self.current_prompt_tool() != tool
        combo = getattr(owner, "prompt_tool_combo", None)
        if combo is not None:
            self._select_combo_data(combo, tool)
        if changed:
            self.state_changed.emit()

    def current_text_prompt(self) -> str:
        owner = self.owner
        edit = getattr(owner, "text_prompt_edit", None)
        return "" if edit is None else str(edit.text())

    def set_text_prompt(self, text: str) -> None:
        owner = self.owner
        edit = getattr(owner, "text_prompt_edit", None)
        if edit is not None:
            edit.setText(text)
        if owner is not None and hasattr(owner, "_set_text_prompt"):
            owner._set_text_prompt()

    def set_point_polarity(self, polarity: str) -> None:
        owner = self.owner
        changed = self.current_point_polarity() != polarity
        combo = getattr(owner, "point_polarity_combo", None)
        if combo is not None:
            self._select_combo_data(combo, polarity)
        if changed:
            self.state_changed.emit()

    def current_point_polarity(self) -> str:
        owner = self.owner
        combo = getattr(owner, "point_polarity_combo", None)
        if combo is None:
            return "positive"
        return combo.currentData() or "positive"

    def create_prompt_layer(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_initialize_prompt_layer"):
            owner._initialize_prompt_layer()
        self.refresh()

    def clear_prompt_state(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_clear_prompts"):
            owner._clear_prompts()
        self.refresh()

    def clear_live_points_prompt(self) -> None:
        layer = self.current_points_layer()
        if layer is not None:
            try:
                if hasattr(self.owner, "live_point_refinement"):
                    with self.owner.live_point_refinement.suspend_events():
                        layer.data = np.empty((0, layer.ndim), dtype=float)
                else:
                    layer.data = np.empty((0, layer.ndim), dtype=float)
            except Exception:
                pass
            try:
                layer.selected_data = set()
            except Exception:
                pass
            try:
                layer.refresh()
            except Exception:
                pass
        self._activate_live_points_layer()
        owner = self.owner
        if owner is not None and hasattr(owner, "_log"):
            owner._log("Live Points prompt cleared.")

    def browse_model_dir(self) -> None:
        owner = self.owner
        if owner is None:
            return
        selected = QFileDialog.getExistingDirectory(
            owner,
            "Select SAM3 model directory",
            self.current_model_dir(),
        )
        if not selected:
            return
        if self.shared_context.settings is not None:
            self.shared_context.settings.setValue(SIMPLE_MODEL_DIR_KEY, selected)
        if hasattr(owner, "_log"):
            owner._log(f"Selected Simple mode SAM3 model directory: {selected}")
        self.state_changed.emit()

    def current_model_dir(self) -> str:
        if self.shared_context.settings is None:
            return ""
        return self.shared_context.settings.value(SIMPLE_MODEL_DIR_KEY, "", type=str)

    def current_device(self) -> str:
        return runtime_device(torch.cuda.is_available())

    def set_device(self, device: str) -> None:
        if self.shared_context.settings is not None:
            self.shared_context.settings.setValue(SIMPLE_DEVICE_KEY, self.current_device())
        self.state_changed.emit()

    def run_current_task(self) -> None:
        owner = self.owner
        if not self._apply_simple_model_settings():
            return
        if owner is not None and hasattr(owner, "_sync_live_refinement_layer"):
            owner._sync_live_refinement_layer()
        router = self.shared_context.task_router
        if router is not None:
            router.run_current_task()

    def toggle_next_point_mode(self) -> None:
        owner = self.owner
        if not self._live_refinement_enabled():
            return
        if owner is not None and hasattr(owner, "_toggle_next_point_mode"):
            owner._toggle_next_point_mode()
        self.state_changed.emit()

    def flip_existing_point_polarity(self) -> None:
        owner = self.owner
        if not self._live_refinement_enabled():
            return
        if owner is not None and hasattr(owner, "_flip_existing_point_polarity"):
            owner._flip_existing_point_polarity()
        self.state_changed.emit()

    def _live_refinement_enabled(self) -> bool:
        owner = self.owner
        if owner is None or not hasattr(owner, "_live_refinement_enabled"):
            return False
        return bool(owner._live_refinement_enabled())

    def _apply_simple_model_settings(self) -> bool:
        owner = self.owner
        if owner is None:
            return False

        simple_model_dir = self.current_model_dir().strip()
        model_dir_edit = getattr(owner, "model_dir_edit", None)
        model_type_combo = getattr(owner, "model_type_combo", None)
        device_combo = getattr(owner, "device_combo", None)
        model_type = self._model_type_for_task(self.current_task())

        if model_type_combo is not None:
            old = model_type_combo.blockSignals(True)
            try:
                index = model_type_combo.findData(model_type)
                if index >= 0:
                    model_type_combo.setCurrentIndex(index)
            finally:
                model_type_combo.blockSignals(old)

        if model_dir_edit is not None:
            model_dir_edit.setText(simple_model_dir)

        if device_combo is not None:
            old = device_combo.blockSignals(True)
            try:
                index = device_combo.findData(self.current_device())
                if index >= 0:
                    device_combo.setCurrentIndex(index)
            finally:
                device_combo.blockSignals(old)

        if hasattr(owner, "_sync_model_type_controls"):
            owner._sync_model_type_controls()

        if simple_model_dir:
            return True

        if hasattr(owner, "_log"):
            owner._log(
                "Select a Simple mode SAM3 model folder before running. "
                "Use a SAM3.1 multiplex folder for 3D Multiplex."
            )
        self.shared_context.activity_status.set_ready()
        return False

    def set_large_image_enabled(self, enabled: bool) -> None:
        owner = self.owner
        check = getattr(owner, "large_image_check", None)
        if check is not None:
            check.setChecked(bool(enabled))
        elif owner is not None and hasattr(owner, "_on_large_image_mode_changed"):
            owner._on_large_image_mode_changed()
        self.state_changed.emit()

    def large_image_enabled(self) -> bool:
        owner = self.owner
        check = getattr(owner, "large_image_check", None)
        return bool(check is not None and check.isChecked())

    def set_batch_all_image_layers_enabled(self, enabled: bool) -> None:
        check = getattr(self.owner, "batch_all_images_check", None)
        if check is not None:
            check.setChecked(bool(enabled))
        self.state_changed.emit()

    def batch_all_image_layers_enabled(self) -> bool:
        check = getattr(self.owner, "batch_all_images_check", None)
        return bool(check is not None and check.isChecked())

    def roi_size_items(self) -> list[tuple[str, object]]:
        combo = getattr(self.owner, "roi_size_combo", None)
        if combo is None:
            return []
        return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]

    def current_roi_size(self) -> object:
        combo = getattr(self.owner, "roi_size_combo", None)
        return None if combo is None else combo.currentData()

    def set_roi_size(self, value: object) -> None:
        combo = getattr(self.owner, "roi_size_combo", None)
        if combo is not None:
            self._select_combo_data(combo, value)
        self.state_changed.emit()

    def current_tile_overlap(self) -> int:
        spin = getattr(self.owner, "tile_overlap_spin", None)
        return 15 if spin is None else int(spin.value())

    def set_tile_overlap(self, value: int) -> None:
        spin = getattr(self.owner, "tile_overlap_spin", None)
        if spin is not None:
            spin.setValue(int(value))

    def merge_tile_seams_enabled(self) -> bool:
        check = getattr(self.owner, "merge_tile_seams_check", None)
        return bool(check is not None and check.isChecked())

    def set_merge_tile_seams_enabled(self, enabled: bool) -> None:
        check = getattr(self.owner, "merge_tile_seams_check", None)
        if check is not None:
            check.setChecked(bool(enabled))

    def set_exemplar_source(self, source: str) -> None:
        combo = getattr(self.owner, "exemplar_source_combo", None)
        if combo is None:
            return
        changed = combo.currentData() != source
        self._select_combo_data(combo, source)
        if changed:
            self.state_changed.emit()

    def current_exemplar_source(self) -> str:
        combo = getattr(self.owner, "exemplar_source_combo", None)
        return "target" if combo is None else str(combo.currentData() or "target")

    def crop_image_layer_names(self) -> list[str]:
        target = self.current_image_layer_name()
        return [name for name in self.image_layer_names() if name != target]

    def set_crop_image_layer(self, layer_name: str) -> None:
        combo = getattr(self.owner, "exemplar_crop_layer_combo", None)
        if combo is not None:
            self._select_combo_data(combo, layer_name)
        self.state_changed.emit()

    def current_crop_image_layer(self) -> str:
        combo = getattr(self.owner, "exemplar_crop_layer_combo", None)
        return "" if combo is None else str(combo.currentData() or "")

    def crop_region_items(self) -> list[tuple[str, object]]:
        combo = getattr(self.owner, "exemplar_crop_region_combo", None)
        if combo is None:
            return []
        return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]

    def current_crop_region(self) -> object:
        combo = getattr(self.owner, "exemplar_crop_region_combo", None)
        return "whole" if combo is None else combo.currentData()

    def set_crop_region(self, value: object) -> None:
        combo = getattr(self.owner, "exemplar_crop_region_combo", None)
        if combo is not None:
            self._select_combo_data(combo, value)
        self.state_changed.emit()

    def run_tiled_exemplar_scan(self) -> None:
        owner = self.owner
        if not self._apply_simple_model_settings():
            return
        self.configure_task(Sam3Task.EXEMPLAR, prompt_tool=PROMPT_BOX, large_image=True)
        if owner is not None and hasattr(owner, "_run_batch_local_exemplar_task"):
            owner._run_batch_local_exemplar_task()

    def save_preview_labels(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_save_preview_labels"):
            owner._save_preview_labels()

    def save_and_release_preview(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_save_preview_mask_and_release_memory"):
            owner._save_preview_mask_and_release_memory()

    def browse_preview_output_folder(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_browse_preview_output_folder"):
            owner._browse_preview_output_folder()

    def preview_output_folder(self) -> str:
        edit = getattr(self.owner, "preview_output_folder_edit", None)
        return "" if edit is None else str(edit.text())

    def set_preview_output_folder(self, folder: str) -> None:
        owner = self.owner
        edit = getattr(owner, "preview_output_folder_edit", None)
        if edit is not None:
            edit.setText(folder)
        if owner is not None and hasattr(owner, "_save_settings"):
            owner._save_settings()

    def preview_output_format_items(self) -> list[str]:
        combo = getattr(self.owner, "preview_output_format_combo", None)
        if combo is None:
            return ["TIFF", "NumPy (.npy)", "PNG"]
        return [combo.itemText(index) for index in range(combo.count())]

    def preview_output_format(self) -> str:
        combo = getattr(self.owner, "preview_output_format_combo", None)
        return "TIFF" if combo is None else str(combo.currentText())

    def set_preview_output_format(self, text: str) -> None:
        owner = self.owner
        combo = getattr(owner, "preview_output_format_combo", None)
        if combo is not None:
            index = combo.findText(text)
            if index >= 0:
                combo.setCurrentIndex(index)
        if owner is not None and hasattr(owner, "_on_preview_output_format_changed"):
            owner._on_preview_output_format_changed(text)

    def sync_preview_output_controls(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_sync_preview_output_controls"):
            owner._sync_preview_output_controls()

    def open_saved_mask_folder(self) -> None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_open_saved_mask_folder"):
            owner._open_saved_mask_folder()

    def current_points_layer(self) -> Any | None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_current_points_layer"):
            return owner._current_points_layer()
        return None

    def first_preview_labels_layer(self) -> Any | None:
        owner = self.owner
        if owner is not None and hasattr(owner, "_first_preview_labels_layer"):
            return owner._first_preview_labels_layer()
        viewer = self.shared_context.viewer
        if viewer is None:
            return None
        for name in ("SAM3 preview labels", "SAM3 tiled exemplar labels", "SAM3 propagated preview labels"):
            try:
                return viewer.layers[name]
            except Exception:
                pass
        return None

    def has_preview_labels_layer(self) -> bool:
        return self.first_preview_labels_layer() is not None

    def accept_live_preview(self, coords: tuple[int, ...] | None = None, *, clear_prompt: bool = True) -> bool:
        viewer = self.shared_context.viewer
        preview = self.first_preview_labels_layer()
        owner = self.owner
        if viewer is None or preview is None:
            if owner is not None and hasattr(owner, "_log"):
                owner._log("No Live Points preview labels found to accept.")
            return False

        preview_data = np.asarray(preview.data)
        if not np.any(preview_data):
            if owner is not None and hasattr(owner, "_log"):
                owner._log("Live Points preview is empty; nothing accepted.")
            return False

        mask = preview_data > 0
        if coords is not None and len(coords) == preview_data.ndim:
            in_bounds = all(0 <= coord < size for coord, size in zip(coords, preview_data.shape, strict=False))
            if in_bounds:
                value = int(preview_data[coords])
                if value > 0:
                    mask = preview_data == value

        accepted = self._live_accepted_layer(preview_data.shape)
        accepted_data = np.asarray(accepted.data)
        self._live_accept_undo.append(accepted_data.copy())
        if len(self._live_accept_undo) > 20:
            self._live_accept_undo.pop(0)

        object_id = int(accepted_data.max()) + 1 if accepted_data.size else 1
        write_mask = mask & (accepted_data == 0)
        changed = int(np.count_nonzero(write_mask))
        if changed == 0:
            write_mask = mask
            changed = int(np.count_nonzero(write_mask))
        accepted_data = accepted_data.copy()
        accepted_data[write_mask] = object_id
        accepted.data = accepted_data
        try:
            accepted.refresh()
        except Exception:
            pass

        if owner is not None and hasattr(owner, "_clear_preview_layers"):
            owner._clear_preview_layers()
        if clear_prompt:
            self.clear_live_points_prompt()
        else:
            self._activate_live_points_layer()
        if owner is not None and hasattr(owner, "_log"):
            suffix = " Prompt cleared." if clear_prompt else ""
            owner._log(f"Accepted Live Points preview as object {object_id} ({changed} pixel(s)).{suffix}")
        return True

    def undo_live_accept(self) -> bool:
        viewer = self.shared_context.viewer
        if viewer is None or not self._live_accept_undo:
            owner = self.owner
            if owner is not None and hasattr(owner, "_log"):
                owner._log("No accepted Live Points object to undo.")
            return False
        try:
            layer = viewer.layers["SAM3 live accepted labels"]
        except Exception:
            return False
        layer.data = self._live_accept_undo.pop()
        try:
            layer.refresh()
        except Exception:
            pass
        self._activate_live_points_layer()
        owner = self.owner
        if owner is not None and hasattr(owner, "_log"):
            owner._log("Undid last accepted Live Points object.")
        return True

    def _live_accepted_layer(self, shape: tuple[int, ...]):
        viewer = self.shared_context.viewer
        assert viewer is not None
        name = "SAM3 live accepted labels"
        try:
            layer = viewer.layers[name]
            if tuple(np.asarray(layer.data).shape) == tuple(shape):
                return layer
        except Exception:
            pass
        data = np.zeros(shape, dtype=np.uint32)
        return viewer.add_labels(data, name=name)

    def _activate_live_points_layer(self) -> None:
        layer = self.current_points_layer()
        viewer = self.shared_context.viewer
        if layer is None or viewer is None:
            return
        try:
            viewer.layers.selection.active = layer
        except Exception:
            pass
        try:
            layer.mode = "add"
        except Exception:
            pass

    def propagate_existing_session(self) -> None:
        router = self.shared_context.task_router
        if router is not None:
            router.propagate_existing_session()

    def clear_preview_layers(self) -> None:
        router = self.shared_context.task_router
        if router is not None:
            router.clear_preview_layers()

    def clear_results(self) -> None:
        router = self.shared_context.task_router
        if router is not None:
            router.clear_results()

    def open_mask_operations(self) -> None:
        router = self.shared_context.task_router
        if router is not None:
            router.open_mask_operations()

    def task_label(self, task: Sam3Task) -> str:
        labels = {
            Sam3Task.SEGMENT_2D: "2D",
            Sam3Task.TEXT: "Text",
            Sam3Task.REFINE: "Refine",
            Sam3Task.EXEMPLAR: "Exemplar",
            Sam3Task.SEGMENT_3D: "3D/Video",
        }
        return labels.get(task, str(task.value))

    def _select_combo_data(self, combo: Any, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
