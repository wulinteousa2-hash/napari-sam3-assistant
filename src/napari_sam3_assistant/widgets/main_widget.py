from __future__ import annotations

from pathlib import Path
from typing import Any

from napari import current_viewer
from napari.viewer import Viewer
import torch
from qtpy.QtCore import QSettings, QSize
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..device_utils import (
    device_indicator_tooltip,
    manual_device_override_enabled,
    runtime_device,
)
from ..notifications import TaskCompleteSound
from ..providers.sam3_repo_provider import Sam3RepoProvider
from ..services.checkpoint_service import CheckpointService
from ..services.layer_writer import LayerWriter
from ..services.prompt_collector import PromptCollector
from ..services.prompt_state_service import PromptStateService
from .advanced.advanced_mode_panel import (
    SETTINGS_APP,
    SETTINGS_ORG,
    SAM3_WIDGET_STYLE,
    AdvancedModePanel,
    _is_cuda_kernel_image_error,
)
from .mode_switch_bar import ModeSwitchBar
from .shared.shared_context import SharedContext
from .shared.task_router import TaskRouter
from .simple.simple_mode_panel import SimpleModePanel
from .simple.simple_mode_controller import SIMPLE_MODEL_DIR_KEY


MODE_SETTINGS_KEY = "workspace_mode"
SIMPLE_MODE_WIDTH = 520
SIMPLE_MODE_DOCK_WIDTH = 550
QT_MAX_WIDGET_SIZE = 16777215


class ModeStackedWidget(QStackedWidget):
    """Stack that sizes to the active mode instead of the largest hidden page."""

    def sizeHint(self) -> QSize:
        widget = self.currentWidget()
        if widget is None:
            return super().sizeHint()
        return widget.sizeHint()

    def minimumSizeHint(self) -> QSize:
        widget = self.currentWidget()
        if widget is None:
            return super().minimumSizeHint()
        return widget.minimumSizeHint()


class MainWidget(QWidget):
    """Top-level shell that hosts the Simple and Advanced mode panels."""

    def __init__(
        self,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer or viewer or current_viewer()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.shared_context = self._build_shared_context()
        self.shared_context.mode_change_callback = self._set_mode

        self.mode_switch_bar = ModeSwitchBar(self)
        self.model_status_label = QLabel("Model: not set")
        self.model_status_label.setObjectName("statusLabel")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("SAM3.0", "sam3")
        self.model_type_combo.addItem("SAM3.1", "sam3.1")
        self.model_type_combo.currentIndexChanged.connect(self._on_top_model_type_changed)
        self.model_folder_btn = QPushButton("Model Folder")
        self.model_folder_btn.clicked.connect(self._browse_current_model_dir)
        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU / CUDA", "cuda")
        self.device_combo.addItem("CPU (2D only)", "cpu")
        self.device_combo.setEnabled(manual_device_override_enabled())
        self.device_combo.setToolTip(
            device_indicator_tooltip(
                runtime_device(torch.cuda.is_available()),
                override_enabled=manual_device_override_enabled(),
            )
        )
        self.device_combo.currentIndexChanged.connect(self._on_top_device_changed)
        self.sound_check = QCheckBox("Sound")
        self.sound_check.setToolTip("Play a short completion chime when a SAM3 task finishes.")

        self.stack = ModeStackedWidget(self)
        self.advanced_panel = AdvancedModePanel(
            self.shared_context,
            napari_viewer=self.viewer,
            parent=self,
        )
        self.shared_context.task_router = TaskRouter(
            self.advanced_panel,
            self.shared_context,
        )
        self.simple_panel = SimpleModePanel(self.shared_context, self)
        self.stack.addWidget(self.advanced_panel)
        self.stack.addWidget(self.simple_panel)

        self._build_ui()
        self._connect_signals()

        mode = self._restore_mode()
        self.mode_switch_bar.set_mode(mode)
        self._set_mode(mode)

    def _build_shared_context(self) -> SharedContext:
        return SharedContext(
            viewer=self.viewer,
            settings=self.settings,
            provider=Sam3RepoProvider(),
            checkpoint_service=CheckpointService(),
            prompt_state_service=PromptStateService(),
            prompt_collector=PromptCollector(),
            layer_writer=LayerWriter(self.viewer) if self.viewer is not None else None,
            task_complete_sound=TaskCompleteSound(self.settings),
        )

    def _build_ui(self) -> None:
        self.setStyleSheet(SAM3_WIDGET_STYLE)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self.top_strip = self._build_top_strip()
        layout.addWidget(self.top_strip)
        layout.addWidget(self.stack, 1)
        self.setLayout(layout)

    def _build_top_strip(self) -> QWidget:
        strip = QFrame()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.mode_switch_bar)
        layout.addStretch(1)
        layout.addWidget(self.model_status_label)
        layout.addWidget(self.model_type_combo)
        layout.addWidget(self.model_folder_btn)
        layout.addWidget(QLabel("Device"))
        layout.addWidget(self.device_combo)
        layout.addWidget(self.sound_check)
        strip.setLayout(layout)
        return strip

    def _connect_signals(self) -> None:
        self.mode_switch_bar.mode_changed.connect(self._set_mode)
        self.simple_panel.controller.state_changed.connect(self._sync_top_controls)
        self.sound_check.toggled.connect(self._on_sound_toggled)

    def _restore_mode(self) -> str:
        mode = self.settings.value(MODE_SETTINGS_KEY, "simple", type=str)
        return mode if mode in {"simple", "advanced"} else "simple"

    def _save_mode(self, mode: str) -> None:
        self.settings.setValue(MODE_SETTINGS_KEY, mode)

    def _set_mode(self, mode: str) -> None:
        mode = mode if mode in {"simple", "advanced"} else "simple"
        if hasattr(self, "mode_switch_bar") and self.mode_switch_bar.current_mode() != mode:
            self.mode_switch_bar.set_mode(mode)
        self.shared_context.set_mode(mode)
        self._save_mode(mode)
        if mode == "advanced":
            self.advanced_panel._restore_settings()
            self.advanced_panel._sync_model_type_controls()
        self.stack.setCurrentWidget(
            self.advanced_panel if mode == "advanced" else self.simple_panel
        )
        self._sync_top_controls()
        current = self.stack.currentWidget()
        refresh = getattr(current, "refresh_from_shared_state", None)
        if callable(refresh):
            refresh()
        self._sync_mode_size_constraints(mode)
        self.stack.updateGeometry()
        self.updateGeometry()
        self.adjustSize()
        parent = self.parentWidget()
        while parent is not None:
            parent.updateGeometry()
            parent.adjustSize()
            parent = parent.parentWidget()

    def _sync_top_controls(self) -> None:
        mode = self.shared_context.get_mode()
        self.model_status_label.setVisible(mode == "advanced")
        self.model_type_combo.setVisible(mode == "advanced")
        if mode == "advanced":
            self.model_type_combo.blockSignals(True)
            try:
                index = self.model_type_combo.findData(
                    self.advanced_panel._current_model_type()
                )
                if index >= 0:
                    self.model_type_combo.setCurrentIndex(index)
            finally:
                self.model_type_combo.blockSignals(False)

        device = self._current_top_device(mode)
        self.device_combo.blockSignals(True)
        try:
            index = self.device_combo.findData(device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
            self.device_combo.setEnabled(manual_device_override_enabled())
            self.device_combo.setToolTip(
                device_indicator_tooltip(
                    device,
                    override_enabled=manual_device_override_enabled(),
                )
            )
        finally:
            self.device_combo.blockSignals(False)

        text, tooltip = self._current_model_status(mode)
        self.model_status_label.setText(text)
        self.model_status_label.setToolTip(tooltip)
        self.model_folder_btn.setToolTip(tooltip or "Select model folder")
        sound = self.shared_context.task_complete_sound
        if sound is not None:
            self.sound_check.blockSignals(True)
            try:
                self.sound_check.setChecked(sound.is_enabled())
            finally:
                self.sound_check.blockSignals(False)

    def _sync_mode_size_constraints(self, mode: str) -> None:
        dock = self._containing_dock_widget()
        if mode == "simple":
            self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
            self.setMinimumWidth(SIMPLE_MODE_WIDTH)
            self.setMaximumWidth(SIMPLE_MODE_WIDTH)
            self.top_strip.setMinimumWidth(SIMPLE_MODE_WIDTH)
            self.top_strip.setMaximumWidth(SIMPLE_MODE_WIDTH)
            self.stack.setMinimumWidth(SIMPLE_MODE_WIDTH)
            self.stack.setMaximumWidth(SIMPLE_MODE_WIDTH)
            self.simple_panel.setMinimumWidth(SIMPLE_MODE_WIDTH)
            self.simple_panel.setMaximumWidth(SIMPLE_MODE_WIDTH)
            if dock is not None:
                dock.setMinimumWidth(SIMPLE_MODE_DOCK_WIDTH)
                dock.setMaximumWidth(SIMPLE_MODE_DOCK_WIDTH)
                dock.adjustSize()
            return

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        self.top_strip.setMinimumWidth(0)
        self.top_strip.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        self.stack.setMinimumWidth(0)
        self.stack.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        self.simple_panel.setMinimumWidth(0)
        self.simple_panel.setMaximumWidth(QT_MAX_WIDGET_SIZE)
        if dock is not None:
            dock.setMinimumWidth(0)
            dock.setMaximumWidth(QT_MAX_WIDGET_SIZE)
            dock.adjustSize()

    def _containing_dock_widget(self) -> QDockWidget | None:
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QDockWidget):
                return parent
            parent = parent.parentWidget()
        return None

    def _current_top_device(self, mode: str) -> str:
        if mode == "advanced":
            value = self.advanced_panel.device_combo.currentData()
            if manual_device_override_enabled() and value in {"cuda", "cpu"}:
                return value
        return runtime_device(torch.cuda.is_available())

    def _current_model_status(self, mode: str) -> tuple[str, str]:
        if mode == "advanced":
            model_dir = self.advanced_panel.model_dir_edit.text().strip()
            model_type = self.advanced_panel._current_model_type()
            label = "SAM3.1" if model_type == "sam3.1" else "SAM3.0"
            ok = self.advanced_panel._checkpoint_path_from_model_dir(model_dir, model_type) is not None
        else:
            model_dir = self.settings.value(SIMPLE_MODEL_DIR_KEY, "", type=str).strip()
            label = "SAM3.0"
            ok = any((Path(model_dir) / name).exists() for name in ("sam3.pt", "model.safetensors"))

        if not model_dir:
            return "Model: not set", "No model folder selected"
        suffix = "ok" if ok else "missing"
        return f"Model: {label} {suffix}", model_dir

    def _browse_current_model_dir(self) -> None:
        if self.shared_context.get_mode() == "advanced":
            self.advanced_panel._browse_model_dir()
        else:
            self.simple_panel.controller.browse_model_dir()
        self._sync_top_controls()

    def _on_top_device_changed(self) -> None:
        if not manual_device_override_enabled():
            self._sync_top_controls()
            return
        device = self.device_combo.currentData() or "cuda"
        if self.shared_context.get_mode() == "advanced":
            combo = self.advanced_panel.device_combo
            old = combo.blockSignals(True)
            try:
                index = combo.findData(device)
                if index >= 0:
                    combo.setCurrentIndex(index)
            finally:
                combo.blockSignals(old)
            self.advanced_panel._on_device_changed()
        else:
            self.simple_panel.controller.set_device(device)
        self._sync_top_controls()

    def _on_top_model_type_changed(self) -> None:
        if self.shared_context.get_mode() != "advanced":
            return
        value = self.model_type_combo.currentData() or "sam3"
        combo = self.advanced_panel.model_type_combo
        old = combo.blockSignals(True)
        try:
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(old)
        self.advanced_panel._on_model_type_changed()
        self._sync_top_controls()

    def _on_sound_toggled(self, checked: bool) -> None:
        sound = self.shared_context.task_complete_sound
        if sound is not None:
            sound.set_enabled(checked)

    def __getattr__(self, name: str) -> Any:
        advanced_panel = self.__dict__.get("advanced_panel")
        if advanced_panel is not None and hasattr(advanced_panel, name):
            return getattr(advanced_panel, name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
