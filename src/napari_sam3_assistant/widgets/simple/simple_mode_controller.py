from __future__ import annotations

from typing import Any

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

    def browse_model_dir(self) -> None:
        owner = self.owner
        if owner is None:
            return
        selected = QFileDialog.getExistingDirectory(
            owner,
            "Select SAM3.0 model directory",
            self.current_model_dir(),
        )
        if not selected:
            return
        if self.shared_context.settings is not None:
            self.shared_context.settings.setValue(SIMPLE_MODEL_DIR_KEY, selected)
        if hasattr(owner, "_log"):
            owner._log(f"Selected Simple mode SAM3.0 model directory: {selected}")
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

        if model_type_combo is not None:
            old = model_type_combo.blockSignals(True)
            try:
                index = model_type_combo.findData("sam3")
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
                "Simple mode uses SAM3.0 only. Select a Simple mode SAM3.0 model folder before running."
            )
        self.shared_context.activity_status.set_ready()
        return False

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
