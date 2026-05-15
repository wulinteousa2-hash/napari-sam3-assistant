from __future__ import annotations
import csv
import gc
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from napari import current_viewer
from napari.layers import Image, Labels, Points, Shapes
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
import torch
from qtpy.QtGui import QDesktopServices, QKeySequence
from qtpy.QtCore import QSettings, Qt, QUrl
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QShortcut
)

from ...adapters import Sam3Adapter, Sam3AdapterConfig, cuda_compatibility_issue
from ...core.coordinates import (
    RoiBounds,
    box_roi_bounds,
    centered_roi_bounds,
    extract_2d_image,
    extract_2d_roi,
    extract_video_xy_roi,
    globalize_result_arrays,
    infer_image_selection,
    localize_bundle_to_roi,
    roi_anchor_from_bundle,
    selection_frame_count,
    selection_video_output_shape,
)
from ...device_utils import (
    CPU_3D_MESSAGE,
    CPU_EXPERIMENTAL_2D_MESSAGE,
    CPU_SAM31_MESSAGE,
    CUDA_CPU_ONLY_MESSAGE,
    cpu_prompt_support_error,
    device_indicator_tooltip,
    manual_device_override_enabled,
    normalize_requested_device,
    runtime_device,
)
from ...core.models import BoxPrompt, PromptBundle, Sam3Result, Sam3Session, Sam3Task
from ...core.diagnostics import Sam3Diagnostics
from ...providers.sam3_repo_provider import Sam3RepoProvider
from ...services.checkpoint_service import CheckpointService
from ...services.layer_writer import LayerWriter
from ...services.prompt_collector import PromptCollector
from ...services.prompt_state_service import PromptStateService
from ...mask_operations import MaskOperationsPanel
from ...mask_operations.export_service import MaskExportService
from ...notifications import TaskCompleteSound
from ..collapsible_panel import CollapsiblePanel
from ..live_point_refinement import LivePointRefinementController
from ..shared.activity_status_controller import ActivityStatusController
from ..shared.shared_context import SharedContext

NONE_LABEL = "(none)"
SETTINGS_ORG = "napari-sam3-assistant"
SETTINGS_APP = "sam3-assistant"
PROMPT_POINTS = "points"
PROMPT_BOX = "box"
PROMPT_LABELS = "labels"
PROMPT_TEXT = "text"
MAX_RESULTS_TABLE_ROWS_PER_RESULT = 100
MAX_RESULTS_TABLE_TOTAL_ROWS = 2000
SAM3_WIDGET_STYLE = """
MainWidget {
    background: #141821;
    color: #d8dee9;
}
QGroupBox {
    background: #1a2130;
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #c9d1dc;
    background: transparent;
    border: none;
}
QLabel {
    color: #c9d1dc;
    font-weight: 400;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #101722;
    color: #e6edf5;
    border: 1px solid #344154;
    border-radius: 6px;
    padding: 5px 7px;
    selection-background-color: #3b6f8f;
    font-weight: 400;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #6f8fae;
}
QLineEdit#textPromptInput {
    background: #172333;
    color: #f1f5f9;
    border: 1px solid #7aa2c7;
    border-left: 4px solid #8fb3d9;
    padding: 7px 9px;
    font-weight: 400;
}
QLineEdit#textPromptInput:focus {
    background: #1b2a3d;
    border: 1px solid #a0b9d4;
    border-left: 4px solid #a0b9d4;
}
QTextEdit#multiTextPromptInput {
    background: #121c29;
    color: #e6edf5;
    border: 1px solid #536b84;
    border-left: 4px solid #6f8fae;
    padding: 6px 8px;
}
QTextEdit#multiTextPromptInput:focus {
    background: #172333;
    border: 1px solid #8aa7c2;
    border-left: 4px solid #8aa7c2;
}
QPushButton {
    background: #273244;
    color: #eef2f7;
    border: 1px solid #3c4a5f;
    border-radius: 7px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #334057;
    border-color: #71849c;
}
QPushButton:pressed {
    background: #3d5f7a;
}
QPushButton:disabled {
    background: #222936;
    color: #727c8b;
    border-color: #343d4c;
}
QPushButton#runButton {
    background: #2f5f7c;
    border-color: #7aa2c7;
}
QPushButton#runButton:hover {
    background: #3a6f90;
}
QPushButton#saveButton {
    background: #315f45;
    border-color: #75a589;
}
QPushButton#saveButton:hover {
    background: #3a7052;
}
QPushButton#clearButton {
    background: #6a522f;
    border-color: #a68a58;
}
QPushButton#clearButton:hover {
    background: #7a6038;
}
QPushButton#cancelButton {
    background: #6a3434;
    border-color: #a66b6b;
}
QPushButton#cancelButton:hover {
    background: #7a3d3d;
}
QCheckBox {
    color: #c9d1dc;
    spacing: 8px;
    font-weight: 400;
}
QTextEdit#statusBox {
    background: #0f1621;
    color: #bfd2e3;
    border: 1px solid #2e4b61;
    border-radius: 8px;
    font-family: "DejaVu Sans Mono", "Menlo", monospace;
}
QLabel#statusLabel {
    color: #9db6cf;
    font-weight: 700;
}
QTableWidget#resultsTable {
    background: #101722;
    alternate-background-color: #151e2b;
    color: #eef2f7;
    gridline-color: #334154;
    border: 1px solid #2e4b61;
    border-radius: 8px;
    selection-background-color: #0369a1;
}
QHeaderView::section {
    background: #315868;
    color: #edf6fb;
    border: 0;
    border-right: 1px solid #2a4b58;
    padding: 5px 7px;
    font-weight: 700;
}

QGroupBox::indicator {
    width: 14px;
    height: 14px;
}

QGroupBox::indicator:unchecked {
    background: #0f172a;
    border: 1px solid #7aa2c7;
    border-radius: 3px;
}

QGroupBox::indicator:checked {
    background: #7aa2c7;
    border: 1px solid #7aa2c7;
    border-radius: 3px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 2px 8px;
    color: #9db6cf;
    background: #141821;
    border-radius: 6px;
    font-weight: 700;
}
QCheckBox {
    color: #c9d1dc;
    spacing: 8px;
    font-weight: 400;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
}

QCheckBox::indicator:unchecked {
    background: transparent;
    border: 1px solid #64748b;
    border-radius: 3px;
}

QCheckBox::indicator:checked {
    background: #475569;
    border: 1px solid #64748b;
    border-radius: 3px;
}

QFrame#collapsibleBody {
    background: #1a2130;
    border: 1px solid #2d3748;
    border-left: 4px solid #5e7892;
    border-radius: 10px;
}

QToolButton#collapsibleToggle {
    background: #101722;
    color: #eef2f7;
    border: 1px solid #6f8fae;
    border-radius: 9px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    font-weight: 700;
    padding: 0px;
}

QToolButton#collapsibleToggle:hover {
    border-color: #9db6cf;
    color: #f1f5f9;
}

QLabel#collapsibleStepBadge {
    background: #40566e;
    color: #f1f5f9;
    border: 1px solid #6f8fae;
    border-radius: 9px;
    padding: 3px 8px;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.4px;
}

QLabel#collapsibleTitle {
    color: #eef2f7;
    font-size: 13px;
    font-weight: 800;
    padding: 2px 2px;
}

QFrame#collapsibleHeaderLine {
    color: #2d3748;
}

QFrame#collapsibleBody {
    background: #1a2130;
    border: 1px solid #2d3748;
    border-left: 4px solid #5e7892;
    border-radius: 10px;
}


QLabel#activityIndicator {
    color: #d7e7f3;
    font-weight: 700;
    padding: 5px 8px;
    background: #132638;
    border: 1px solid #4d6f8c;
    border-radius: 8px;
}
"""


class AdvancedModePanel(QWidget):
    def __init__(
        self,
        shared_context: SharedContext | None = None,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.shared_context = shared_context
        shared_viewer = shared_context.viewer if shared_context is not None else None
        self.viewer = napari_viewer or viewer or shared_viewer or current_viewer()

        self.provider = (
            shared_context.provider
            if shared_context is not None and shared_context.provider is not None
            else Sam3RepoProvider()
        )
        self.checkpoint_service = (
            shared_context.checkpoint_service
            if shared_context is not None and shared_context.checkpoint_service is not None
            else CheckpointService()
        )
        self.prompt_state_service = (
            shared_context.prompt_state_service
            if shared_context is not None and shared_context.prompt_state_service is not None
            else PromptStateService()
        )
        self.prompt_collector = (
            shared_context.prompt_collector
            if shared_context is not None and shared_context.prompt_collector is not None
            else PromptCollector()
        )
        self.layer_writer = (
            shared_context.layer_writer
            if shared_context is not None and shared_context.layer_writer is not None
            else LayerWriter(self.viewer) if self.viewer is not None else None
        )
        self.mask_export_service = MaskExportService()

        self.adapter: Sam3Adapter | None = (
            shared_context.adapter if shared_context is not None else None
        )
        self.video_session: Sam3Session | None = (
            shared_context.video_session if shared_context is not None else None
        )
        self._worker: Any | None = None
        self._worker_failed = False
        self._layer_events_connected = False
        self._active_rois: dict[str, RoiBounds] = (
            shared_context.active_rois if shared_context is not None else {}
        )
        self._last_quick_mask_path: Path | None = None
        self.settings = (
            shared_context.settings
            if shared_context is not None and shared_context.settings is not None
            else QSettings(SETTINGS_ORG, SETTINGS_APP)
        )
        self.task_complete_sound = (
            shared_context.task_complete_sound
            if shared_context is not None and shared_context.task_complete_sound is not None
            else TaskCompleteSound(self.settings)
        )
        self.activity_status = (
            shared_context.activity_status
            if shared_context is not None
            else ActivityStatusController()
        )
        self._sync_shared_runtime_state()

        self._build_ui()
        self._restore_settings()
        self._connect_layer_events()
        self._refresh_layers()
        self._sync_model_type_controls()
        self._on_task_changed()
        self._on_prompt_tool_changed()

        self.live_point_refinement = LivePointRefinementController(
            self,
            run_preview_callback=self._run_live_refinement_preview,
            toggle_next_mode_callback=self._toggle_next_point_mode,
            flip_existing_point_callback=self._flip_existing_point_polarity,
            is_enabled_callback=self._live_refinement_enabled,
            shortcuts_enabled_callback=self._live_refinement_shortcuts_enabled,
        )
        self._sync_live_refinement_layer()
        self._set_live_refinement_status("Activity: idle")


    def _sync_shared_runtime_state(self) -> None:
        if self.shared_context is None:
            return
        self.shared_context.viewer = self.viewer
        self.shared_context.settings = self.settings
        self.shared_context.provider = self.provider
        self.shared_context.checkpoint_service = self.checkpoint_service
        self.shared_context.prompt_state_service = self.prompt_state_service
        self.shared_context.prompt_collector = self.prompt_collector
        self.shared_context.layer_writer = self.layer_writer
        self.shared_context.task_complete_sound = self.task_complete_sound
        self.shared_context.adapter = self.adapter
        self.shared_context.video_session = self.video_session
        self.shared_context.worker = self._worker
        self.shared_context.worker_failed = self._worker_failed
        self.shared_context.active_rois = self._active_rois


    def _build_ui(self) -> None:
        self.setStyleSheet(SAM3_WIDGET_STYLE)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        left_column = QVBoxLayout()
        left_column.setSpacing(8)
        right_column = QVBoxLayout()
        right_column.setSpacing(8)

        backend_group = self._build_backend_group()
        task_group = self._build_task_group()
        prompt_group = self._build_prompt_group()
        actions_group = self._build_actions_group()
        results_group = self._build_results_group()

        left_column.addWidget(CollapsiblePanel("Step 1. Model Actions", backend_group, collapsed=True))
        left_column.addWidget(CollapsiblePanel("Step 2. Task Setup", task_group, collapsed=False))
        left_column.addWidget(CollapsiblePanel("Step 3. Prompt Tools", prompt_group, collapsed=False))
        
        right_column.addWidget(CollapsiblePanel("Step 4. Run and Save", actions_group, collapsed=False))
        right_column.addWidget(CollapsiblePanel("Step 5. Results", results_group, collapsed=False))

        self.status_box = QTextEdit()
        self.status_box.setObjectName("statusBox")
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(150)



        status_container = QWidget()
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status_box)
        status_container.setLayout(status_layout)

        right_column.addWidget(CollapsiblePanel("Log. Activity", status_container, collapsed=True))

        left_column.addStretch(1)
        right_column.addStretch(1)
        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)
        layout.addLayout(columns)

        layout.addStretch(1)
        self.setLayout(layout)

    def _open_mask_operations(self) -> None:
        panel = self._ensure_mask_operations_surface()
        panel.set_viewer(self.viewer)
        panel.tabs.setCurrentWidget(panel.mask_cleanup_tab)
        panel.show()
        panel.raise_()
        panel.activateWindow()
        self._log("Opened Mask Operations for preview-mask cleanup and export.")

    def _ensure_mask_operations_surface(self) -> MaskOperationsPanel:
        panel = self.shared_context.transient.get("mask_operations_panel")
        if isinstance(panel, MaskOperationsPanel):
            dock = self.shared_context.transient.get("mask_operations_dock")
            if dock is not None:
                try:
                    dock.show()
                    dock.raise_()
                except Exception:
                    pass
            return panel

        if self.viewer is None:
            self.viewer = current_viewer()

        panel = MaskOperationsPanel(self.viewer, log_callback=self._log)
        panel.setWindowTitle("SAM3 Mask Operations")
        self.shared_context.transient["mask_operations_panel"] = panel

        window = getattr(self.viewer, "window", None) if self.viewer is not None else None
        add_dock_widget = getattr(window, "add_dock_widget", None)
        if callable(add_dock_widget):
            dock = add_dock_widget(
                panel,
                name="SAM3 Mask Operations",
                area="right",
            )
            if hasattr(dock, "setFloating"):
                dock.setFloating(True)
            if hasattr(dock, "resize"):
                dock.resize(780, 720)
            self.shared_context.transient["mask_operations_dock"] = dock
        else:
            panel.resize(760, 720)
        return panel




    def _set_live_refinement_status(self, text: str) -> None:
        if hasattr(self, "live_refinement_status_label"):
            display_text = text if len(text) <= 72 else f"{text[:69]}..."
            self.live_refinement_status_label.setText(display_text)
            self.live_refinement_status_label.setToolTip(text)

    def _live_refinement_enabled(self) -> bool:
        return (
            self._current_task() == Sam3Task.REFINE
            and self.prompt_tool_combo.currentData() == PROMPT_POINTS
            and self._worker is None
        )

    def _live_refinement_shortcuts_enabled(self) -> bool:
        if self.shared_context is not None and self.shared_context.get_mode() != "advanced":
            return False
        return self._live_refinement_enabled()

    def _toggle_next_point_mode(self) -> None:
        current = self.point_polarity_combo.currentData() or "positive"
        new_value = "negative" if current == "positive" else "positive"
        index = self.point_polarity_combo.findData(new_value)
        if index >= 0:
            self.point_polarity_combo.setCurrentIndex(index)

        self._set_current_point_polarity()
        self._log(f"Next point mode: {new_value.capitalize()}")

    def _flip_existing_point_polarity(self) -> None:
        layer = self._current_points_layer()
        if layer is None:
            self._log("No points layer selected.")
            return
        data = np.asarray(layer.data)
        if len(data) == 0:
            self._log("No points available to flip.")
            return

        selected = sorted(getattr(layer, "selected_data", []))
        indices = selected or [len(data) - 1]
        properties = dict(getattr(layer, "properties", {}) or {})
        values = self._point_polarity_values(layer)
        for idx in indices:
            values[idx] = "negative" if values[idx] == "positive" else "positive"
        properties["polarity"] = np.asarray(values, dtype=object)
        layer.properties = properties
        layer.refresh_colors()
        self._log(f"Flipped {len(indices)} point(s); rerunning Live Points.")
        self._run_live_refinement_preview()

    def _sync_live_refinement_layer(self) -> None:
        if self.viewer is None:
            self.live_point_refinement.set_points_layer(None)
            return

        layer_name = self._optional_combo_data(self.points_layer_combo)
        if not layer_name:
            self.live_point_refinement.set_points_layer(None)
            return

        try:
            layer = self.viewer.layers[layer_name]
        except (KeyError, ValueError):
            layer = None

        self.live_point_refinement.set_points_layer(layer)

    def _run_live_refinement_preview(self) -> None:
        if not self._live_refinement_enabled():
            return
        self._set_live_refinement_status("Activity: Live Points running...")
        self._run_current_task()
    
    def _step_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        font = group.font()
        font.setPointSize(max(font.pointSize() + 1, 11))
        font.setBold(False)
        group.setFont(font)
        return group

    def _build_backend_group(self) -> QGroupBox:
        group = self._step_group("Validate, load, or unload the selected model")
        layout = QVBoxLayout()

        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Select local SAM3 model directory...")

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("SAM3.0 2D/3D/video", "sam3")
        self.model_type_combo.addItem("SAM3.1 video multiplex", "sam3.1")

        btn_row = QHBoxLayout()
        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self._validate_model_dir)

        self.load_image_btn = QPushButton("Load 2D Model")
        self.load_image_btn.clicked.connect(self._load_image_adapter)

        self.load_video_btn = QPushButton("Load 3D/Video Model")
        self.load_video_btn.clicked.connect(self._load_video_adapter)

        unload_btn = QPushButton("Unload")
        unload_btn.clicked.connect(self._unload_adapter)
        unload_btn.setObjectName("clearButton")

        btn_row.addWidget(validate_btn)
        btn_row.addWidget(self.load_image_btn)
        btn_row.addWidget(self.load_video_btn)
        btn_row.addWidget(unload_btn)

        self.lazy_load_check = QCheckBox("Load model when running")
        self.lazy_load_check.setChecked(True)

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
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        layout.addLayout(btn_row)
        layout.addWidget(self.lazy_load_check)
        group.setLayout(layout)
        return group

    def _build_task_group(self) -> QGroupBox:
        group = self._step_group("Choose the SAM3 task and target image")
        layout = QVBoxLayout()
        task_layout = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItem("2D segmentation", Sam3Task.SEGMENT_2D)
        self.task_combo.addItem("3D/video propagation", Sam3Task.SEGMENT_3D)
        self.task_combo.addItem("Exemplar segmentation", Sam3Task.EXEMPLAR)
        self.task_combo.addItem("Text segmentation", Sam3Task.TEXT)
        self.task_combo.addItem("Live Points (positive/negative)", Sam3Task.REFINE)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)

        self.image_layer_combo = QComboBox()
        self.image_layer_combo.currentIndexChanged.connect(self._on_target_image_changed)
        self.batch_all_images_check = QCheckBox("Batch all image layers")
        self.batch_all_images_check.setToolTip(
            "Run the current prompt setup on every napari Image layer. "
            "Each image gets its own SAM3 preview output layers."
        )

        self.large_image_check = QCheckBox("Enable large-image local inference")
        self.large_image_check.setChecked(False)
        self.large_image_check.setToolTip(
            "When enabled, SAM3 runs only on a local XY ROI around point/box prompts "
            "and writes the result back into global image coordinates. For 3D/video, "
            "the same fixed XY ROI is used across all frames."
        )
        self.large_image_check.toggled.connect(self._on_large_image_mode_changed)

        self.roi_size_combo = QComboBox()
        self.roi_size_combo.addItem("512 x 512", (512, 512))
        self.roi_size_combo.addItem("1024 x 1024", (1024, 1024))
        self.roi_size_combo.addItem("2048 x 2048", (2048, 2048))
        self.roi_size_combo.addItem("4096 x 4096", (4096, 4096))
        self.roi_size_combo.addItem("8192 x 8192", (8192, 8192))
        self.roi_size_combo.setCurrentIndex(1)
        self.roi_size_combo.setEnabled(False)
        self.roi_size_combo.currentIndexChanged.connect(self._on_large_image_mode_changed)

        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 50)
        self.tile_overlap_spin.setValue(15)
        self.tile_overlap_spin.setSuffix("%")
        self.tile_overlap_spin.setEnabled(False)
        self.tile_overlap_spin.setToolTip(
            "Overlap between local tiles for full-image tiled exemplar scans. "
            "Overlap helps objects near tile edges."
        )

        self.merge_tile_seams_check = QCheckBox("Merge seam-split objects")
        self.merge_tile_seams_check.setChecked(True)
        self.merge_tile_seams_check.setEnabled(False)
        self.merge_tile_seams_check.setToolTip(
            "After tiled exemplar scanning, reconnect labels that were split by "
            "straight tile boundaries. This checks only narrow tile seam bands."
        )

        self.exemplar_source_combo = QComboBox()
        self.exemplar_source_combo.addItem("Use box from target image", "target")
        self.exemplar_source_combo.addItem("Use separate crop image", "crop")
        self.exemplar_source_combo.setToolTip(
            "For tiled exemplar scans, choose whether the example comes from the "
            "target image or from a separate crop image layer."
        )
        self.exemplar_source_combo.currentIndexChanged.connect(self._on_exemplar_source_changed)

        self.exemplar_crop_layer_combo = QComboBox()
        self.exemplar_crop_layer_combo.setToolTip(
            "Small Image layer to use as the exemplar source while scanning the target image."
        )

        self.exemplar_crop_region_combo = QComboBox()
        self.exemplar_crop_region_combo.addItem("Whole crop image", "whole")
        self.exemplar_crop_region_combo.addItem("Box from Shapes layer", "box")
        self.exemplar_crop_region_combo.setToolTip(
            "Use the full crop as the exemplar, or crop one box from the selected Shapes layer."
        )

        self.channel_axis_spin = QSpinBox()
        self.channel_axis_spin.setRange(-1, 8)
        self.channel_axis_spin.setValue(-1)
        channel_axis_tip = (
            "Which data axis is color/channel. Use -1 for grayscale or RGB/RGBA auto-detect. "
            "Examples: C,H,W -> 0; Z,C,H,W or T,C,H,W -> 1."
        )
        self.channel_axis_spin.setToolTip(channel_axis_tip)

        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setRange(0.01, 0.95)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setDecimals(2)
        self.confidence_threshold_spin.setValue(0.35)
        self.confidence_threshold_spin.setToolTip(
            "Detection confidence threshold for SAM3 grounding. Lower values can help "
            "text prompts return candidates; higher values reduce weak detections."
        )
        self.confidence_threshold_spin.valueChanged.connect(lambda _value: self._save_settings())

        self.propagation_direction_combo = QComboBox()
        self.propagation_direction_combo.addItems(["both", "forward", "backward"])
        self.sam31_diagnostics_check = QCheckBox("Log SAM3.1 diagnostics")
        self.sam31_diagnostics_check.setToolTip(
            "Log CUDA state, SAM3.1 session internals, and per-frame propagation timing."
        )
        self.sam31_diagnostics_check.toggled.connect(lambda _checked: self._save_settings())

        task_layout.addRow("Task", self.task_combo)
        task_layout.addRow("Target image to scan", self.image_layer_combo)
        task_layout.addRow("", self.batch_all_images_check)
        task_layout.addRow("", self.large_image_check)
        task_layout.addRow("ROI size", self.roi_size_combo)
        task_layout.addRow("Tile overlap", self.tile_overlap_spin)
        task_layout.addRow("", self.merge_tile_seams_check)
        task_layout.addRow("Exemplar source", self.exemplar_source_combo)
        task_layout.addRow("Exemplar crop image", self.exemplar_crop_layer_combo)
        task_layout.addRow("Crop region", self.exemplar_crop_region_combo)

        advanced_content = QWidget()
        advanced_layout = QFormLayout()
        advanced_note = QLabel(
            "Only change these if your image layout or detection behavior needs manual tuning."
        )
        advanced_note.setWordWrap(True)
        advanced_note.setStyleSheet("color: #9aa7b6; font-size: 11px;")

        advanced_layout.addRow("", advanced_note)
        advanced_layout.addRow("Channel axis", self.channel_axis_spin)
        hint = QLabel("Leave -1 unless your image has an explicit channel dimension.")
        hint.setToolTip(channel_axis_tip)
        advanced_layout.addRow("", hint)
        advanced_layout.addRow("Detection threshold", self.confidence_threshold_spin)
        advanced_layout.addRow("3D direction", self.propagation_direction_combo)
        advanced_layout.addRow("", self.sam31_diagnostics_check)
        advanced_content.setLayout(advanced_layout)

        self._task_setup_form = task_layout
        self._roi_size_row_label = task_layout.labelForField(self.roi_size_combo)
        self._tile_overlap_row_label = task_layout.labelForField(self.tile_overlap_spin)
        self._exemplar_source_row_label = task_layout.labelForField(self.exemplar_source_combo)
        self._exemplar_crop_layer_row_label = task_layout.labelForField(self.exemplar_crop_layer_combo)
        self._exemplar_crop_region_row_label = task_layout.labelForField(self.exemplar_crop_region_combo)
        self._propagation_direction_row_label = advanced_layout.labelForField(
            self.propagation_direction_combo
        )

        layout.addLayout(task_layout)
        layout.addWidget(CollapsiblePanel("Advanced", advanced_content, collapsed=True))
        group.setLayout(layout)
        self._sync_task_setup_visibility()
        return group

    def _build_prompt_layer_selector_group(self) -> QWidget:
        container = QWidget()
        layout = QFormLayout()

        self.points_layer_combo = QComboBox()
        self.points_layer_combo.currentIndexChanged.connect(self._on_points_layer_changed)
        self.shapes_layer_combo = QComboBox()
        self.labels_layer_combo = QComboBox()

        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)

        layout.addRow("Points", self.points_layer_combo)
        layout.addRow("Shapes", self.shapes_layer_combo)
        layout.addRow("Labels", self.labels_layer_combo)
        layout.addRow(refresh_btn)
        container.setLayout(layout)
        return container

    def _build_prompt_group(self) -> QGroupBox:
        group = self._step_group("Guide SAM3 with prompts")
        layout = QFormLayout()

        self.prompt_tool_combo = QComboBox()
        self.prompt_tool_combo.addItem("Points (positive/negative)", PROMPT_POINTS)
        self.prompt_tool_combo.addItem("Box", PROMPT_BOX)
        self.prompt_tool_combo.addItem("Labels mask", PROMPT_LABELS)
        self.prompt_tool_combo.addItem("Text only", PROMPT_TEXT)
        self.prompt_tool_combo.currentIndexChanged.connect(self._on_prompt_tool_changed)

        self.point_polarity_combo = QComboBox()
        self.point_polarity_combo.addItem("Positive", "positive")
        self.point_polarity_combo.addItem("Negative", "negative")
        self.point_polarity_combo.currentIndexChanged.connect(self._set_current_point_polarity)

        self.refinement_hint_label = QLabel(
            "Add the first point to start Live Points; first run may load the model.\n"
            "Next point mode affects the next point only.\n"
            "T = next point mode only. Shift+T = flip selected/latest point and rerun."
        )
        self.refinement_hint_label.setWordWrap(True)
        self.refinement_hint_label.setStyleSheet("color: #93c5fd; font-size: 11px;")
        self.refinement_hint_label.hide()  

        init_prompt_btn = QPushButton("Create Prompt Layer")
        init_prompt_btn.clicked.connect(self._initialize_prompt_layer)

        apply_polarity_btn = QPushButton("Apply mode to selected points")
        apply_polarity_btn.clicked.connect(self._apply_polarity_to_selected_points)

        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setObjectName("textPromptInput")
        self.text_prompt_edit.setPlaceholderText("Type text prompt, then press Enter to run...")
        self.text_prompt_edit.setToolTip(
            "Text prompt for SAM3 grounding. Use a short noun phrase such as "
            "'nucleus' or 'myelin sheath'. Press Enter to run preview."
        )
        self.text_prompt_edit.editingFinished.connect(self._set_text_prompt)
        self.text_prompt_edit.returnPressed.connect(self._run_current_task)

        self.multi_text_prompt_edit = QTextEdit()
        self.multi_text_prompt_edit.setObjectName("multiTextPromptInput")
        self.multi_text_prompt_edit.setPlaceholderText(
            "Optional batch prompts: one prompt per line, then Press Ctrl+Enter to run batch text prompts.")
        self.multi_text_prompt_edit.setToolTip(
            "Optional multi-text mode.Enter one concept per line. With Batch all image "
            "layers enabled, every prompt is run on every image layer."
        )
        self.multi_text_prompt_edit.setMaximumHeight(78)
        self.multi_text_run_shortcut = QShortcut(
            QKeySequence("Ctrl+Return"),
            self.multi_text_prompt_edit,
        )
        self.multi_text_run_shortcut.activated.connect(self._run_current_task)

        self.multi_text_run_shortcut_2 = QShortcut(
            QKeySequence("Ctrl+Enter"),
            self.multi_text_prompt_edit,
        )
        self.multi_text_run_shortcut_2.activated.connect(self._run_current_task)
        clear_btn = QPushButton("Clear Text / Prompt State")
        clear_btn.clicked.connect(self._clear_prompts)

        layout.addRow("Prompt type", self.prompt_tool_combo)
        #layout.addRow("Point type", self.point_polarity_combo)
        layout.addRow("Next point mode", self.point_polarity_combo)
        layout.addRow("", self.refinement_hint_label)

        layout.addRow(init_prompt_btn)
        layout.addRow(apply_polarity_btn)
        layout.addRow("Text prompt", self.text_prompt_edit)
        layout.addRow("Batch text prompts", self.multi_text_prompt_edit)
        layout.addRow(
            CollapsiblePanel(
                "Prompt Layers",
                self._build_prompt_layer_selector_group(),
                collapsed=True,
            )
        )
        layout.addRow(clear_btn)
        group.setLayout(layout)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = self._step_group("Run preview or propagation")
        layout = QVBoxLayout()

        self.live_refinement_status_label = QLabel("Activity: idle")
        self.live_refinement_status_label.setObjectName("activityIndicator")
        self.live_refinement_status_label.setMaximumWidth(460)
        self.live_refinement_status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.live_refinement_status_label.setToolTip(
            "Shows Live Points and SAM3 worker activity so long model runs are visible."
        )

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run Preview")
        self.run_btn.setObjectName("runButton")
        self.run_btn.setToolTip("Run the selected SAM3 task and write preview layers.")
        self.run_btn.clicked.connect(self._run_current_task)

        self.propagate_btn = QPushButton("Propagate Existing Session")
        self.propagate_btn.setObjectName("runButton")
        self.propagate_btn.setToolTip(
            "Reuse the current SAM3 video session without adding a new prompt. "
            "Use after a successful 3D propagation run."
        )
        self.propagate_btn.clicked.connect(self._propagate_existing_session)

        self.batch_local_exemplar_btn = QPushButton("Scan Full Image by Tiles")
        self.batch_local_exemplar_btn.setObjectName("runButton")
        self.batch_local_exemplar_btn.setToolTip(
            "For 2D exemplar segmentation with large-image mode: scan every tile in "
            "the full image using the current exemplar box, then compose one full-size mask."
        )
        self.batch_local_exemplar_btn.clicked.connect(self._run_batch_local_exemplar_task)
        self.batch_local_exemplar_btn.setVisible(False)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.clicked.connect(self._cancel_worker)

        clear_preview_btn = QPushButton("Clear Preview")
        clear_preview_btn.setObjectName("clearButton")
        clear_preview_btn.clicked.connect(self._clear_preview_layers)

        row.addWidget(self.run_btn)
        row.addWidget(self.propagate_btn)
        row.addWidget(self.batch_local_exemplar_btn)
        row.addWidget(clear_preview_btn)
        row.addWidget(cancel_btn)

        layout.addWidget(self.live_refinement_status_label)
        layout.addLayout(row)
        layout.addWidget(self._build_preview_output_group())
        group.setLayout(layout)
        return group

    def _build_preview_output_group(self) -> QFrame:
        frame = QFrame()
        self.preview_output_panel = frame
        frame.setObjectName("previewOutputPanel")
        layout = QFormLayout()
        layout.setContentsMargins(0, 6, 0, 0)

        self.preview_output_folder_edit = QLineEdit()
        self.preview_output_folder_edit.setPlaceholderText("Choose an output folder")
        self.preview_output_folder_edit.setMinimumWidth(0)
        self.preview_output_folder_edit.setMaximumWidth(260)
        self.preview_output_folder_edit.editingFinished.connect(self._save_settings)
        self.preview_output_browse_btn = QPushButton("Choose")
        self.preview_output_browse_btn.setToolTip("Choose where quick-acquired preview masks are saved.")
        self.preview_output_browse_btn.clicked.connect(self._browse_preview_output_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.preview_output_folder_edit)
        folder_row.addWidget(self.preview_output_browse_btn)

        self.preview_output_format_combo = QComboBox()
        self.preview_output_format_combo.addItems(["TIFF", "NumPy (.npy)", "PNG"])
        self.preview_output_format_combo.setMaximumWidth(160)
        self.preview_output_format_combo.currentTextChanged.connect(self._on_preview_output_format_changed)

        self.preview_output_filename_edit = QLineEdit()
        self.preview_output_filename_edit.setMinimumWidth(0)
        self.preview_output_filename_edit.setMaximumWidth(260)
        self.preview_output_filename_edit.setToolTip(
            "Filename for the current preview mask. The extension is added from the selected format."
        )

        self.save_release_btn = QPushButton("Save && Clean")
        self.save_release_btn.setObjectName("saveButton")
        self.save_release_btn.setToolTip(
            "Save the preview mask, remove temporary ROI memory, and unload the SAM3 model. "
            "If Load model when running is checked, the next run reloads automatically."
        )
        self.save_release_btn.clicked.connect(self._save_preview_mask_and_release_memory)

        self.saved_preview_path_label = QLabel("")
        self.saved_preview_path_label.setMinimumWidth(0)
        self.saved_preview_path_label.setMaximumWidth(220)
        self.saved_preview_path_label.setWordWrap(False)
        self.saved_preview_path_label.setVisible(False)
        self.open_saved_folder_btn = QPushButton("Open Folder")
        self.open_saved_folder_btn.setToolTip("Open the folder containing the saved mask file.")
        self.open_saved_folder_btn.clicked.connect(self._open_saved_mask_folder)
        self.open_saved_folder_btn.setVisible(False)

        save_row = QHBoxLayout()
        save_row.addWidget(self.save_release_btn)
        save_row.addStretch(1)
        self.saved_preview_row = QHBoxLayout()
        self.saved_preview_row.addWidget(self.saved_preview_path_label)
        self.saved_preview_row.addWidget(self.open_saved_folder_btn)
        self.saved_preview_row.addStretch(1)

        self.preview_output_folder_row = folder_row
        layout.addRow("Preview output", self.preview_output_folder_row)
        layout.addRow("Format", self.preview_output_format_combo)
        layout.addRow("Filename", self.preview_output_filename_edit)
        layout.addRow(save_row)
        layout.addRow(self.saved_preview_row)
        self.preview_output_folder_label = layout.labelForField(self.preview_output_folder_row)
        self.preview_output_format_label = layout.labelForField(self.preview_output_format_combo)
        self.preview_output_filename_label = layout.labelForField(self.preview_output_filename_edit)
        frame.setLayout(layout)
        frame.setVisible(False)
        return frame

    def _build_results_group(self) -> QGroupBox:
        group = self._step_group("Review and save results")
        layout = QVBoxLayout()

        self.results_table = QTableWidget(0, 6)
        self.results_table.setObjectName("resultsTable")
        self.results_table.setHorizontalHeaderLabels(["Layer", "Prompt", "Frame", "Object ID", "Score", "Area"])
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.verticalHeader().setVisible(False)


        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setMinimumHeight(110)
        self.results_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.results_table.setWordWrap(False)
        self.results_table.setCornerButtonEnabled(False)
        clear_results_btn = QPushButton("Clear Results")
        clear_results_btn.setObjectName("clearButton")
        clear_results_btn.clicked.connect(self._clear_results_table)

        copy_results_btn = QPushButton("Copy Clipboard")
        copy_results_btn.clicked.connect(self._copy_results_to_clipboard)

        export_results_btn = QPushButton("Export CSV")
        export_results_btn.clicked.connect(self._export_results_csv)

        action_row = QHBoxLayout()
        action_row.addWidget(clear_results_btn)
        action_row.addWidget(copy_results_btn)
        action_row.addWidget(export_results_btn)

        layout.addWidget(self.results_table)
        layout.addLayout(action_row)
        group.setLayout(layout)
        return group

    def _browse_model_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select SAM3 model directory",
            str(Path.home()),
        )
        if selected:
            self.model_dir_edit.setText(selected)
            self._save_settings()
            self._log(f"Selected model directory: {selected}")

    def _browse_preview_output_folder(self) -> None:
        current = self.preview_output_folder_edit.text().strip() if hasattr(self, "preview_output_folder_edit") else ""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select preview mask output folder",
            current or str(Path.home()),
        )
        if selected:
            self.preview_output_folder_edit.setText(selected)
            self._save_settings()
            self._update_preview_output_filename()

    def _on_preview_output_format_changed(self, _text: str) -> None:
        self._save_settings()
        self._update_preview_output_filename()

    def _validate_model_dir(self) -> None:
        model_dir = self.model_dir_edit.text().strip()
        result = self.checkpoint_service.validate(
            model_dir,
            model_type=self._current_model_type(),
        )
        if result.ok:
            self._save_settings()
        self._log(result.message)

    def _load_image_adapter(self) -> None:
        try:
            adapter = self._ensure_adapter(reset=True)
        except Exception as exc:
            self._log(f"Cannot load image model: {exc}")
            return

        @thread_worker
        def load_model() -> str:
            needs_interactivity = self._current_task() in {
                Sam3Task.REFINE,
                Sam3Task.EXEMPLAR,
            }
            adapter.load_image(
                enable_instance_interactivity=needs_interactivity
            )
            return "SAM3 image model loaded."

        self._start_worker(
            load_model(),
            on_returned=self._on_image_initialized,
            activity_status=ActivityStatusController.LOADING_MODEL,
        )

    def _load_video_adapter(self) -> None:
        try:
            adapter = self._ensure_adapter(reset=True)
        except Exception as exc:
            self._log(f"Cannot load video model: {exc}")
            return

        @thread_worker
        def load_model() -> str:
            adapter.load_video()
            return "SAM3 video predictor loaded."

        self._start_worker(
            load_model(),
            on_returned=self._on_video_initialized,
            activity_status=ActivityStatusController.LOADING_MODEL,
        )

    def _unload_adapter(self) -> None:
        self._cancel_worker()
        if self.adapter is not None:
            self.adapter.unload()
        self.adapter = None
        self.video_session = None
        self._sync_shared_runtime_state()
        self.provider.unload()
        self._log("SAM3 adapter unloaded.")

    def _set_text_prompt(self) -> None:
        text = self.text_prompt_edit.text()
        self.prompt_state_service.set_text_prompt(text)
        self._log(f"Prompt state: {self.prompt_state_service.summary()}")

    def _clear_prompts(self) -> None:
        self.prompt_state_service.clear()
        self.text_prompt_edit.clear()
        self.multi_text_prompt_edit.clear()
        self._reset_video_session()
        self._log("Prompt state cleared.")

    def _refresh_layers(self, silent: bool = False) -> None:
        if self.viewer is None:
            self.viewer = current_viewer()
            if self.viewer is not None and self.layer_writer is None:
                self.layer_writer = LayerWriter(self.viewer)
                self._sync_shared_runtime_state()
                self._connect_layer_events()

        self._set_combo_items(self.image_layer_combo, self._layer_names({"image"}), include_none=False)
        if hasattr(self, "exemplar_crop_layer_combo"):
            self._refresh_exemplar_crop_layer_combo()
        self._set_combo_items(self.points_layer_combo, self._layer_names({"points"}))
        self._set_combo_items(self.shapes_layer_combo, self._layer_names({"shapes"}))
        self._set_combo_items(self.labels_layer_combo, self._layer_names({"labels"}))
        if hasattr(self, "relabel_layer_combo"):
            self._set_combo_items(self.relabel_layer_combo, self._layer_names({"labels"}), include_none=False)
        if hasattr(self, "mask_operations_panel"):
            self.mask_operations_panel.set_viewer(self.viewer)
        layer_count = 0 if self.viewer is None else len(self.viewer.layers)
        if not silent:
            self._log(f"Layer selectors refreshed. Viewer layers: {layer_count}.")
        if hasattr(self, "live_point_refinement"):
            self._sync_live_refinement_layer()
        self._sync_preview_output_controls()
        self._sync_task_setup_visibility()

    def _refresh_exemplar_crop_layer_combo(self) -> None:
        if not hasattr(self, "exemplar_crop_layer_combo"):
            return
        target_name = self._current_image_layer_name()
        crop_names = [name for name in self._layer_names({"image"}) if name != target_name]
        previous = self.exemplar_crop_layer_combo.currentData()
        self._set_combo_items(self.exemplar_crop_layer_combo, crop_names, include_none=True)
        if previous not in crop_names and len(crop_names) == 1:
            self._select_combo_data(self.exemplar_crop_layer_combo, crop_names[0])

    def _on_task_changed(self) -> None:
        task = self._current_task()
        self._sync_task_setup_visibility()
        self._sync_run_controls()
        if task == Sam3Task.TEXT:
            self.prompt_tool_combo.setCurrentIndex(self.prompt_tool_combo.findData(PROMPT_TEXT))
        elif task == Sam3Task.EXEMPLAR:
            self.prompt_tool_combo.setCurrentIndex(self.prompt_tool_combo.findData(PROMPT_BOX))
        elif task == Sam3Task.REFINE:
            self.prompt_tool_combo.setCurrentIndex(self.prompt_tool_combo.findData(PROMPT_POINTS))
        self._on_prompt_tool_changed()
        self._log(self._task_guidance(task))
        if hasattr(self, "live_point_refinement"):
            self._sync_live_refinement_layer()

    def _on_target_image_changed(self, *_args: Any) -> None:
        self._refresh_exemplar_crop_layer_combo()
        self._sync_task_setup_visibility()
        self._sync_run_controls()

    def _sync_task_setup_visibility(self) -> None:
        if not hasattr(self, "roi_size_combo") or not hasattr(self, "propagation_direction_combo"):
            return

        large_image_enabled = self._large_image_mode_enabled()
        is_exemplar = self._current_task() == Sam3Task.EXEMPLAR
        show_exemplar_source = large_image_enabled and is_exemplar
        use_external_crop = show_exemplar_source and self._external_exemplar_source_enabled()
        self.roi_size_combo.setEnabled(large_image_enabled)
        self.roi_size_combo.setVisible(large_image_enabled)
        if hasattr(self, "tile_overlap_spin"):
            self.tile_overlap_spin.setEnabled(large_image_enabled)
            self.tile_overlap_spin.setVisible(large_image_enabled)
        if hasattr(self, "merge_tile_seams_check"):
            self.merge_tile_seams_check.setEnabled(large_image_enabled)
            self.merge_tile_seams_check.setVisible(large_image_enabled)
        if hasattr(self, "_roi_size_row_label") and self._roi_size_row_label is not None:
            self._roi_size_row_label.setVisible(large_image_enabled)
        if hasattr(self, "_tile_overlap_row_label") and self._tile_overlap_row_label is not None:
            self._tile_overlap_row_label.setVisible(large_image_enabled)
        if hasattr(self, "exemplar_source_combo"):
            self.exemplar_source_combo.setEnabled(show_exemplar_source)
            self.exemplar_source_combo.setVisible(show_exemplar_source)
        if hasattr(self, "exemplar_crop_layer_combo"):
            self.exemplar_crop_layer_combo.setEnabled(use_external_crop)
            self.exemplar_crop_layer_combo.setVisible(use_external_crop)
        if hasattr(self, "exemplar_crop_region_combo"):
            self.exemplar_crop_region_combo.setEnabled(use_external_crop)
            self.exemplar_crop_region_combo.setVisible(use_external_crop)
        if hasattr(self, "_exemplar_source_row_label") and self._exemplar_source_row_label is not None:
            self._exemplar_source_row_label.setVisible(show_exemplar_source)
        if (
            hasattr(self, "_exemplar_crop_layer_row_label")
            and self._exemplar_crop_layer_row_label is not None
        ):
            self._exemplar_crop_layer_row_label.setVisible(use_external_crop)
        if (
            hasattr(self, "_exemplar_crop_region_row_label")
            and self._exemplar_crop_region_row_label is not None
        ):
            self._exemplar_crop_region_row_label.setVisible(use_external_crop)

        is_video = self._current_task() == Sam3Task.SEGMENT_3D
        self.propagation_direction_combo.setEnabled(is_video)
        self.propagation_direction_combo.setVisible(is_video)
        if hasattr(self, "sam31_diagnostics_check"):
            self.sam31_diagnostics_check.setEnabled(is_video)
            self.sam31_diagnostics_check.setVisible(is_video)
        if (
            hasattr(self, "_propagation_direction_row_label")
            and self._propagation_direction_row_label is not None
        ):
            self._propagation_direction_row_label.setVisible(is_video)

    def _sync_run_controls(self) -> None:
        if not hasattr(self, "run_btn") or not hasattr(self, "propagate_btn"):
            return
        is_video = self._current_task() == Sam3Task.SEGMENT_3D
        has_session = self.video_session is not None
        running = self._worker is not None
        if is_video:
            self.run_btn.setText("Start 3D Propagation")
            self.run_btn.setToolTip(
                "Start a new SAM3 video session from the current frame prompt and "
                "propagate through the stack."
            )
            self.propagate_btn.setVisible(True)
            self.propagate_btn.setEnabled(not running and has_session)
        else:
            if self._current_task() == Sam3Task.EXEMPLAR and self._large_image_mode_enabled():
                self.run_btn.setText("Run Current ROI Only")
                self.run_btn.setToolTip(
                    "Run exemplar segmentation only around the current exemplar box. "
                    "Use Scan Full Image by Tiles to cover the whole image."
                )
            else:
                self.run_btn.setText("Run Preview")
                self.run_btn.setToolTip("Run the selected SAM3 task and write preview layers.")
            self.propagate_btn.setVisible(False)
            self.propagate_btn.setEnabled(False)
        if hasattr(self, "batch_local_exemplar_btn"):
            enabled = (
                not running
                and not is_video
                and self._current_task() == Sam3Task.EXEMPLAR
                and self._large_image_mode_enabled()
            )
            if self._external_exemplar_source_enabled():
                self.batch_local_exemplar_btn.setText("Scan Target Image by Tiles")
                self.batch_local_exemplar_btn.setToolTip(
                    "Scan the selected target image by tiles using a separate crop image "
                    "as the visual exemplar source."
                )
            else:
                self.batch_local_exemplar_btn.setText("Scan Full Image by Tiles")
                self.batch_local_exemplar_btn.setToolTip(
                    "For 2D exemplar segmentation with large-image mode: scan every tile in "
                    "the full image using the current exemplar box, then compose one full-size mask."
                )
            self.batch_local_exemplar_btn.setVisible(
                self._current_task() == Sam3Task.EXEMPLAR and self._large_image_mode_enabled()
            )
            self.batch_local_exemplar_btn.setEnabled(enabled)
        self._sync_preview_output_controls()


    def _sync_model_type_controls(self) -> None:
        if not hasattr(self, "load_image_btn") or not hasattr(self, "load_video_btn"):
            return
        is_sam31 = self._current_model_type() == "sam3.1"
        self.load_image_btn.setEnabled(not is_sam31)
        self.load_image_btn.setToolTip(
            "SAM3.1 video multiplex is for 3D/video propagation. "
            "Choose SAM3.0 2D/3D/video for 2D image tasks."
            if is_sam31
            else "Load the SAM3.0 2D image model."
        )
        self.load_video_btn.setToolTip(
            "Load the SAM3.1 multiplex video predictor."
            if is_sam31
            else "Load the SAM3.0 3D/video predictor."
        )

    def _on_prompt_tool_changed(self) -> None:
        tool = self.prompt_tool_combo.currentData()
        is_points = tool == PROMPT_POINTS
        self.point_polarity_combo.setEnabled(is_points)
        text_enabled = tool == PROMPT_TEXT or self._current_task() in {Sam3Task.TEXT, Sam3Task.SEGMENT_2D, Sam3Task.SEGMENT_3D}
        self.text_prompt_edit.setEnabled(text_enabled)
        self.multi_text_prompt_edit.setEnabled(self._current_task() == Sam3Task.TEXT)

        show_refinement_hint = (
            self._current_task() == Sam3Task.REFINE
            and self.prompt_tool_combo.currentData() == PROMPT_POINTS
        )
        self.refinement_hint_label.setVisible(show_refinement_hint)

        if hasattr(self, "live_point_refinement"):
            self._sync_live_refinement_layer()

    def _run_current_task(self) -> None:
        self.activity_status.set_starting_task()
        if self._current_model_type() == "sam3.1" and self._current_task() != Sam3Task.SEGMENT_3D:
            self._log(
                "SAM3.1 video multiplex supports 3D/video propagation in this plugin. "
                "Choose task '3D/video propagation', or switch Model type to "
                "'SAM3.0 2D/3D/video' for 2D image tasks."
            )
            self.activity_status.set_ready()
            return

        if self.batch_all_images_check.isChecked() or self._multi_text_prompts():
            self._run_batch_current_task()
            return

        try:
            bundle = self._collect_bundle()
        except Exception as exc:
            self._log(f"Cannot collect prompts: {exc}")
            self.activity_status.set_ready()
            return

        if not bundle.has_prompt():
            self._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            self.activity_status.set_ready()
            return
        cpu_error = self._cpu_bundle_support_error(bundle)
        if cpu_error:
            self._log(cpu_error)
            self.activity_status.set_ready()
            return

        self._clear_results_table()
        if bundle.task == Sam3Task.SEGMENT_3D:
            self._run_video_task(bundle)
        else:
            self._run_image_task(bundle)

    def _run_batch_current_task(self) -> None:
        if self._current_task() == Sam3Task.SEGMENT_3D:
            self._log("Batch all image layers is for 2D image tasks. Use one stack for 3D/video propagation.")
            self.activity_status.set_ready()
            return
        if self._current_task() == Sam3Task.REFINE:
            self._log("Batch mode is disabled for Live Points. Select one image for point correction.")
            self.activity_status.set_ready()
            return
        if self._multi_text_prompts() and self._current_task() != Sam3Task.TEXT:
            self._log("Multi-text batch prompts require task 'Text segmentation'.")
            self.activity_status.set_ready()
            return
        try:
            bundles = self._collect_batch_bundles()
        except Exception as exc:
            self._log(f"Cannot collect batch prompts: {exc}")
            self.activity_status.set_ready()
            return
        if not bundles:
            self._log("No image layers found for batch segmentation.")
            self.activity_status.set_ready()
            return
        if not any(bundle.has_prompt() for bundle in bundles):
            self._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            self.activity_status.set_ready()
            return
        cpu_errors = [self._cpu_bundle_support_error(bundle) for bundle in bundles]
        cpu_error = next((error for error in cpu_errors if error), None)
        if cpu_error:
            self._log(cpu_error)
            self.activity_status.set_ready()
            return
        self._clear_results_table()
        self._run_batch_image_task(bundles)

    def _collect_batch_bundles(self) -> list[PromptBundle]:
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        channel_axis = self.channel_axis_spin.value()
        bundles: list[PromptBundle] = []
        image_layer_names = (
            self._layer_names({"image"})
            if self.batch_all_images_check.isChecked()
            else [self._current_image_layer_name()]
        )
        text_prompts = self._multi_text_prompts() or [self.text_prompt_edit.text()]
        for layer_name in image_layer_names:
            if not layer_name:
                continue
            for text_prompt in text_prompts:
                bundle = self.prompt_collector.collect(
                    self.viewer,
                    image_layer_name=layer_name,
                    task=self._current_task(),
                    points_layer_name=self._optional_combo_data(self.points_layer_combo),
                    shapes_layer_name=self._optional_combo_data(self.shapes_layer_combo),
                    labels_layer_name=self._optional_combo_data(self.labels_layer_combo),
                    text=text_prompt,
                    channel_axis=None if channel_axis < 0 else channel_axis,
                    collect_exemplar_rois=not self._large_image_mode_enabled(),
                )
                bundles.append(bundle)
        return bundles

    def _run_batch_image_task(self, bundles: list[PromptBundle]) -> None:
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run batch image task: {exc}")
            return
        local_jobs: dict[int, RoiBounds] = {}
        if self._large_image_mode_enabled():
            for index, bundle in enumerate(bundles):
                anchor = roi_anchor_from_bundle(bundle)
                if anchor is None:
                    continue
                bounds = self._active_or_new_roi_bounds(
                    bundle,
                    anchor,
                    self._selection_image_hw(bundle.image),
                    self._selected_roi_size(),
                )
                self._active_rois[bundle.image.layer_name] = bounds
                local_jobs[index] = bounds
            if local_jobs:
                self._show_active_roi_overlay(
                    "batch",
                    None,
                    extra_bounds=[
                        (bundles[index].image.layer_name, bounds)
                        for index, bounds in local_jobs.items()
                    ],
                )

        @thread_worker
        def run_batch():
            for index, bundle in enumerate(bundles):
                image_layer = self.viewer.layers[bundle.image.layer_name]
                bounds = local_jobs.get(index)
                if bounds is not None:
                    self._ensure_image_adapter_loaded_for_bundle(adapter, bundle)
                    roi_data = extract_2d_roi(image_layer.data, bundle.image, bounds)
                    local_bundle = localize_bundle_to_roi(bundle, bounds, tuple(roi_data.shape))
                    result = adapter.run_image(
                        roi_data,
                        local_bundle,
                        cache_context=self._cache_context_for_layer(
                            image_layer,
                            bundle,
                            roi_bounds=bounds,
                        ),
                    )
                    labels, masks, boxes = globalize_result_arrays(
                        labels=result.labels,
                        masks=result.masks,
                        boxes_xyxy=result.boxes_xyxy,
                        bounds=bounds,
                        image_hw=self._selection_image_hw(bundle.image),
                    )
                    result.labels = labels
                    result.masks = masks
                    result.boxes_xyxy = boxes
                    result.metadata["large_image_roi"] = (bounds.y0, bounds.x0, bounds.y1, bounds.x1)
                    result.metadata["large_image_mode"] = True
                    result.metadata["large_image_hw"] = self._selection_image_hw(bundle.image)
                    result.metadata["result_space"] = "global_image"
                else:
                    self._ensure_image_adapter_loaded_for_bundle(adapter, bundle)
                    result = adapter.run_image(
                        image_layer.data,
                        bundle,
                        cache_context=self._cache_context_for_layer(image_layer, bundle),
                    )
                result.metadata["image_layer"] = bundle.image.layer_name
                if bundle.text and bundle.text.text:
                    result.metadata["batch_prompt"] = bundle.text.text
                yield result

        worker = run_batch()
        worker.yielded.connect(self._write_batch_image_result)
        self._start_worker(worker)
        image_count = len({bundle.image.layer_name for bundle in bundles})
        prompt_count = len({bundle.text.text for bundle in bundles if bundle.text and bundle.text.text}) or 1
        self._log(
            f"Running {len(bundles)} batch job(s): {image_count} image layer(s), "
            f"{prompt_count} prompt(s)."
        )
        if self._large_image_mode_enabled():
            self._log(
                f"Large-image mode ON: local ROI inference for {len(local_jobs)} "
                "anchored batch job(s); jobs without point/box anchors use full-image inference."
            )
        else:
            self._log("Large-image mode OFF: full-image inference.")

    def _run_image_task(self, bundle: PromptBundle) -> None:
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        if self._large_image_mode_enabled():
            anchor = roi_anchor_from_bundle(bundle)
            if anchor is not None:
                if bundle.task == Sam3Task.EXEMPLAR:
                    self._log(
                        "Run Current ROI Only will segment one local ROI around the exemplar. "
                        "Click Scan Full Image by Tiles to scan the entire image."
                    )
                self._run_large_image_task(bundle, anchor)
                return
            self._log(
                "Large-image mode ON, but no point or box ROI anchor was found. "
                "Using full-image inference for this task."
            )
        image_layer = self.viewer.layers[bundle.image.layer_name]
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run image task: {exc}")
            return

        @thread_worker
        def run_image() -> Sam3Result:
            self._ensure_image_adapter_loaded_for_bundle(adapter, bundle)
            result = adapter.run_image(
                image_layer.data,
                bundle,
                cache_context=self._cache_context_for_layer(image_layer, bundle),
            )
            result.metadata["image_layer"] = bundle.image.layer_name
            return result

        worker = run_image()
        worker.returned.connect(self._write_image_result)
        self._start_worker(worker)
        self._log(f"Running {bundle.task.value} on image layer '{bundle.image.layer_name}'.")
        if self._large_image_mode_enabled():
            self._log("Large-image mode OFF for this run: no local ROI anchor available.")
        else:
            self._log("Large-image mode OFF: full-image inference.")

    def _run_large_image_task(self, bundle: PromptBundle, anchor: tuple[float, float]) -> None:
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        image_layer = self.viewer.layers[bundle.image.layer_name]
        image_hw = self._selection_image_hw(bundle.image)
        roi_size = self._selected_roi_size()
        bounds = self._active_or_new_roi_bounds(bundle, anchor, image_hw, roi_size)
        self._active_rois[bundle.image.layer_name] = bounds
        self._show_active_roi_overlay(bundle.image.layer_name, bounds)
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run local ROI task: {exc}")
            return

        @thread_worker
        def run_local_roi() -> Sam3Result:
            self._ensure_image_adapter_loaded_for_bundle(adapter, bundle)
            roi_data = extract_2d_roi(image_layer.data, bundle.image, bounds)
            local_bundle = localize_bundle_to_roi(bundle, bounds, tuple(roi_data.shape))
            result = adapter.run_image(
                roi_data,
                local_bundle,
                cache_context=self._cache_context_for_layer(
                    image_layer,
                    bundle,
                    roi_bounds=bounds,
                ),
            )
            labels, masks, boxes = globalize_result_arrays(
                labels=result.labels,
                masks=result.masks,
                boxes_xyxy=result.boxes_xyxy,
                bounds=bounds,
                image_hw=image_hw,
            )
            result.labels = labels
            result.masks = masks
            result.boxes_xyxy = boxes
            result.metadata["image_layer"] = bundle.image.layer_name
            result.metadata["large_image_roi"] = (bounds.y0, bounds.x0, bounds.y1, bounds.x1)
            result.metadata["large_image_mode"] = True
            result.metadata["large_image_hw"] = image_hw
            result.metadata["result_space"] = "global_image"
            return result

        worker = run_local_roi()
        worker.returned.connect(self._write_image_result)
        self._start_worker(worker)
        self._log(
            f"Large-image mode ON: local ROI inference ({bounds.width} x {bounds.height}); "
            f"ROI y={bounds.y0}:{bounds.y1}, x={bounds.x0}:{bounds.x1}."
        )

    def _run_batch_local_exemplar_task(self) -> None:
        self.activity_status.set_starting_task()
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            self.activity_status.set_ready()
            return
        if self._current_task() != Sam3Task.EXEMPLAR:
            self._log("Full-image tiled scan is only available for Exemplar segmentation.")
            self.activity_status.set_ready()
            return
        if not self._large_image_mode_enabled():
            self._log("Enable large-image local inference before scanning the full image by tiles.")
            self.activity_status.set_ready()
            return
        roi_hw = self._selected_roi_size()
        overlap_fraction = float(self.tile_overlap_spin.value()) / 100.0
        merge_tile_seams = bool(self.merge_tile_seams_check.isChecked())
        target_layer_names = self._tiled_exemplar_target_layer_names()
        if not target_layer_names:
            self._log("No image layers found for tiled exemplar scanning.")
            self.activity_status.set_ready()
            return
        jobs: list[dict[str, Any]] = []
        external_exemplar: tuple[np.ndarray, str] | None = None
        try:
            if self._external_exemplar_source_enabled():
                external_exemplar = self._collect_external_exemplar_patch()
            for layer_name in target_layer_names:
                bundle = self._collect_bundle_for_tiled_exemplar(layer_name)
                if external_exemplar is None:
                    exemplar, exemplar_source_name = self._collect_tiled_exemplar_patch(bundle)
                else:
                    exemplar, exemplar_source_name = external_exemplar
                cpu_error = self._cpu_bundle_support_error(bundle)
                if cpu_error:
                    raise RuntimeError(cpu_error)
                image_layer = self.viewer.layers[bundle.image.layer_name]
                image_hw = self._selection_image_hw(bundle.image)
                exemplar_hw = tuple(int(value) for value in np.asarray(exemplar).shape[:2])
                if exemplar_hw[0] > roi_hw[0] or exemplar_hw[1] > roi_hw[1]:
                    raise RuntimeError(
                        "The exemplar crop is larger than the tile size. "
                        "Select a smaller crop or increase ROI size."
                    )
                tiles = self._tile_bounds_for_image(image_hw, roi_hw, overlap_fraction)
                if not tiles:
                    raise RuntimeError(f"No tiles were generated for image layer '{layer_name}'.")
                jobs.append(
                    {
                        "bundle": bundle,
                        "image_layer": image_layer,
                        "image_hw": image_hw,
                        "exemplar": np.asarray(exemplar),
                        "exemplar_source_name": exemplar_source_name,
                        "tiles": tiles,
                    }
                )
        except Exception as exc:
            self._log(f"Cannot collect tiled exemplar scan inputs: {exc}")
            self.activity_status.set_ready()
            return
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run tiled exemplar scan: {exc}")
            self.activity_status.set_ready()
            return
        batch_mode = len(jobs) > 1
        self._clear_results_table()
        self._show_active_roi_overlay(
            "batch" if batch_mode else jobs[0]["bundle"].image.layer_name,
            None,
            extra_bounds=[
                (job["bundle"].image.layer_name, bounds)
                for job in jobs
                for bounds in job["tiles"]
            ],
        )

        @thread_worker
        def run_tiled_exemplar():
            total_jobs = len(jobs)
            for job_index, job in enumerate(jobs, start=1):
                bundle = job["bundle"]
                image_layer = job["image_layer"]
                image_hw = job["image_hw"]
                exemplar = job["exemplar"]
                exemplar_source_name = job["exemplar_source_name"]
                tiles = job["tiles"]
                self._ensure_image_adapter_loaded_for_bundle(adapter, bundle)
                composed = np.zeros(image_hw, dtype=np.uint32)
                next_object_id = 1
                total_tiles = len(tiles)
                for tile_index, bounds in enumerate(tiles, start=1):
                    yield (
                        f"Tiled exemplar scan {job_index}/{total_jobs} "
                        f"'{bundle.image.layer_name}' tile {tile_index}/{total_tiles}: "
                        f"y={bounds.y0}:{bounds.y1}, x={bounds.x0}:{bounds.x1}"
                    )
                    tile = extract_2d_roi(image_layer.data, bundle.image, bounds)
                    augmented, tile_origin, exemplar_box = self._augmented_exemplar_tile(tile, exemplar)
                    tile_bundle = self._bundle_for_augmented_exemplar(bundle, augmented, exemplar_box)
                    result = adapter.run_image(
                        augmented,
                        tile_bundle,
                        cache_context=self._cache_context_for_layer(
                            image_layer,
                            bundle,
                            roi_bounds=bounds,
                        ),
                    )
                    local_labels = self._result_labels_for_tile(
                        result,
                        augmented.shape,
                        tile_origin,
                        (bounds.height, bounds.width),
                    )
                    next_object_id = self._compose_tile_labels(
                        composed,
                        local_labels,
                        bounds,
                        next_object_id,
                    )
                seam_merge_count = 0
                if merge_tile_seams:
                    composed, seam_merge_count = self._merge_tile_seam_labels(
                        composed,
                        tiles,
                        dilation_px=2,
                        min_contact_pixels=8,
                    )
                yield Sam3Result(
                    task=Sam3Task.EXEMPLAR,
                    labels=composed,
                    metadata={
                        "image_layer": bundle.image.layer_name,
                        "large_image_mode": True,
                        "large_image_tiled_scan": True,
                        "large_image_hw": image_hw,
                        "tile_count": total_tiles,
                        "tile_size": roi_hw,
                        "tile_overlap_percent": int(self.tile_overlap_spin.value()),
                        "tile_seam_merge_enabled": merge_tile_seams,
                        "tile_seam_merge_count": seam_merge_count,
                        "tile_seam_merge_dilation_px": 2,
                        "tile_seam_merge_min_contact_pixels": 8,
                        "exemplar_source_layer": exemplar_source_name,
                        "external_exemplar_source": self._external_exemplar_source_enabled(),
                        "batch_tiled_exemplar": batch_mode,
                        "result_space": "global_image",
                    },
                )

        def handle_tiled_exemplar_output(payload: object) -> None:
            if isinstance(payload, Sam3Result):
                if batch_mode:
                    self._write_batch_image_result(payload)
                else:
                    self._write_image_result(payload)
                return
            self._log(str(payload))

        worker = run_tiled_exemplar()
        worker.yielded.connect(handle_tiled_exemplar_output)
        self._start_worker(worker)
        total_tiles = sum(len(job["tiles"]) for job in jobs)
        self._log(
            f"Started full-image tiled exemplar scan: {len(jobs)} image layer(s), {total_tiles} tile(s), "
            f"tile size {roi_hw[1]} x {roi_hw[0]}, overlap {self.tile_overlap_spin.value()}%, "
            f"seam merge {'ON' if merge_tile_seams else 'OFF'}."
        )
        if self._external_exemplar_source_enabled():
            exemplar_source_name = jobs[0]["exemplar_source_name"]
            self._log(
                f"Using crop layer '{exemplar_source_name}' as exemplar source and scanning "
                f"{len(jobs)} target image layer(s) by tiles."
            )

    def _tiled_exemplar_target_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        if not self.batch_all_images_check.isChecked():
            return [self._current_image_layer_name()]
        crop_layer_name = (
            self._optional_combo_data(self.exemplar_crop_layer_combo)
            if self._external_exemplar_source_enabled()
            else ""
        )
        return [name for name in self._layer_names({"image"}) if name and name != crop_layer_name]

    def _collect_bundle_for_tiled_exemplar(self, image_layer_name: str | None = None) -> PromptBundle:
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        image_layer_name = image_layer_name or self._current_image_layer_name()
        if not image_layer_name:
            raise RuntimeError("Select an image layer first.")
        channel_axis = self.channel_axis_spin.value()
        return self.prompt_collector.collect(
            self.viewer,
            image_layer_name=image_layer_name,
            task=Sam3Task.EXEMPLAR,
            points_layer_name=None,
            shapes_layer_name=self._optional_combo_data(self.shapes_layer_combo),
            labels_layer_name=None,
            text="",
            channel_axis=None if channel_axis < 0 else channel_axis,
            collect_exemplar_rois=True,
        )

    def _collect_tiled_exemplar_patch(self, bundle: PromptBundle) -> tuple[np.ndarray, str]:
        if self._external_exemplar_source_enabled():
            return self._collect_external_exemplar_patch()
        if not bundle.exemplars:
            raise RuntimeError("Draw at least one exemplar box in the selected Shapes layer before tiled scanning.")
        return np.asarray(bundle.exemplars[0].roi), bundle.image.layer_name

    def _collect_external_exemplar_patch(self) -> tuple[np.ndarray, str]:
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        crop_layer_name = self._optional_combo_data(self.exemplar_crop_layer_combo)
        if not crop_layer_name:
            raise RuntimeError("Select a crop image layer to use as the exemplar source.")
        target_layer_name = self._current_image_layer_name()
        if crop_layer_name == target_layer_name:
            raise RuntimeError(
                f"Target image and exemplar crop image are both '{target_layer_name}'. "
                "Set Target image to the large image to scan, and Exemplar crop image "
                "to the small crop layer."
            )
        region_mode = self.exemplar_crop_region_combo.currentData()
        if region_mode == "box":
            channel_axis = self.channel_axis_spin.value()
            crop_bundle = self.prompt_collector.collect(
                self.viewer,
                image_layer_name=crop_layer_name,
                task=Sam3Task.EXEMPLAR,
                points_layer_name=None,
                shapes_layer_name=self._optional_combo_data(self.shapes_layer_combo),
                labels_layer_name=None,
                text="",
                channel_axis=None if channel_axis < 0 else channel_axis,
                collect_exemplar_rois=True,
            )
            if not crop_bundle.exemplars:
                raise RuntimeError(
                    "Draw at least one box in the selected Shapes layer, or choose Whole crop image."
                )
            return np.asarray(crop_bundle.exemplars[0].roi), crop_layer_name

        crop_layer = self.viewer.layers[crop_layer_name]
        selection = self._image_selection_for_layer(crop_layer)
        patch = extract_2d_image(crop_layer.data, selection)
        patch = np.asarray(patch)
        if patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
            raise RuntimeError("The selected crop image layer is empty.")
        return patch, crop_layer_name

    def _tile_bounds_for_image(
        self,
        image_hw: tuple[int, int],
        roi_hw: tuple[int, int],
        overlap_fraction: float,
    ) -> list[RoiBounds]:
        height, width = image_hw
        tile_h = min(int(roi_hw[0]), height)
        tile_w = min(int(roi_hw[1]), width)
        stride_y = max(1, int(round(tile_h * (1.0 - overlap_fraction))))
        stride_x = max(1, int(round(tile_w * (1.0 - overlap_fraction))))
        y_starts = self._tile_starts(height, tile_h, stride_y)
        x_starts = self._tile_starts(width, tile_w, stride_x)
        return [
            RoiBounds(y0=y, x0=x, y1=min(height, y + tile_h), x1=min(width, x + tile_w))
            for y in y_starts
            for x in x_starts
        ]

    def _tile_starts(self, size: int, tile_size: int, stride: int) -> list[int]:
        if size <= tile_size:
            return [0]
        starts = list(range(0, max(1, size - tile_size + 1), stride))
        last = size - tile_size
        if starts[-1] != last:
            starts.append(last)
        return sorted(set(int(value) for value in starts))

    def _augmented_exemplar_tile(
        self,
        tile: np.ndarray,
        exemplar: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int], BoxPrompt]:
        tile_arr = np.asarray(tile)
        exemplar_arr = np.asarray(exemplar)
        if tile_arr.ndim == 2 and exemplar_arr.ndim == 3 and exemplar_arr.shape[-1] in (1, 3, 4):
            exemplar_arr = exemplar_arr[..., 0]
        if tile_arr.ndim == 3 and tile_arr.shape[-1] in (1, 3, 4) and exemplar_arr.ndim == 2:
            exemplar_arr = np.repeat(exemplar_arr[..., None], tile_arr.shape[-1], axis=-1)
        gap = 8
        exemplar_h = min(int(exemplar_arr.shape[0]), int(tile_arr.shape[0]))
        exemplar_w = min(int(exemplar_arr.shape[1]), int(tile_arr.shape[1]))
        exemplar_crop = exemplar_arr[:exemplar_h, :exemplar_w]
        out_shape = (tile_arr.shape[0], exemplar_w + gap + tile_arr.shape[1], *tile_arr.shape[2:])
        augmented = np.zeros(out_shape, dtype=tile_arr.dtype)
        augmented[:exemplar_h, :exemplar_w, ...] = exemplar_crop
        tile_x0 = exemplar_w + gap
        augmented[: tile_arr.shape[0], tile_x0 : tile_x0 + tile_arr.shape[1], ...] = tile_arr
        box = BoxPrompt(y0=0.0, x0=0.0, y1=float(exemplar_h), x1=float(exemplar_w))
        return augmented, (0, tile_x0), box

    def _bundle_for_augmented_exemplar(
        self,
        source_bundle: PromptBundle,
        augmented: np.ndarray,
        exemplar_box: BoxPrompt,
    ) -> PromptBundle:
        selection = infer_image_selection(
            layer_name=source_bundle.image.layer_name,
            data_shape=tuple(int(value) for value in augmented.shape),
            channel_axis=augmented.ndim - 1 if augmented.ndim >= 3 and augmented.shape[-1] in (1, 3, 4) else None,
        )
        return PromptBundle(
            task=Sam3Task.EXEMPLAR,
            image=selection,
            boxes=[exemplar_box],
        )

    def _result_labels_for_tile(
        self,
        result: Sam3Result,
        augmented_shape: tuple[int, ...],
        tile_origin: tuple[int, int],
        tile_hw: tuple[int, int],
    ) -> np.ndarray:
        label_image = np.zeros(augmented_shape[:2], dtype=np.uint32)
        if result.labels is not None:
            labels = np.asarray(result.labels)
            if labels.ndim == 2:
                label_image = labels.astype(np.uint32, copy=False)
        elif result.masks is not None:
            masks = np.asarray(result.masks)
            if masks.ndim == 2:
                label_image[masks > 0] = 1
            elif masks.ndim >= 3:
                for mask_index, mask in enumerate(masks, start=1):
                    label_image[np.asarray(mask) > 0] = int(mask_index)
        y0, x0 = tile_origin
        tile_h, tile_w = tile_hw
        return np.asarray(label_image[y0 : y0 + tile_h, x0 : x0 + tile_w], dtype=np.uint32)

    def _compose_tile_labels(
        self,
        composed: np.ndarray,
        local_labels: np.ndarray,
        bounds: RoiBounds,
        next_object_id: int,
    ) -> int:
        components, count = ndi.label(np.asarray(local_labels) != 0)
        if count == 0:
            return next_object_id
        target = composed[bounds.y0:bounds.y1, bounds.x0:bounds.x1]
        for component_id in range(1, int(count) + 1):
            mask = components == component_id
            write_mask = mask & (target == 0)
            if not np.any(write_mask):
                continue
            target[write_mask] = int(next_object_id)
            next_object_id += 1
        return next_object_id

    def _merge_tile_seam_labels(
        self,
        labels: np.ndarray,
        tiles: list[RoiBounds],
        dilation_px: int = 2,
        min_contact_pixels: int = 8,
    ) -> tuple[np.ndarray, int]:
        label_data = np.asarray(labels, dtype=np.uint32)
        if label_data.ndim != 2 or not tiles:
            return label_data, 0

        height, width = label_data.shape
        vertical_boundaries = sorted({
            int(value)
            for bounds in tiles
            for value in (bounds.x0, bounds.x1)
            if 0 < int(value) < width
        })
        horizontal_boundaries = sorted({
            int(value)
            for bounds in tiles
            for value in (bounds.y0, bounds.y1)
            if 0 < int(value) < height
        })
        if not vertical_boundaries and not horizontal_boundaries:
            return label_data, 0

        parent: dict[int, int] = {}

        def find(value: int) -> int:
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

        for x in vertical_boundaries:
            for (a, b), count in self._vertical_seam_label_pairs(label_data, x, dilation_px).items():
                if count >= min_contact_pixels:
                    union(a, b)
        for y in horizontal_boundaries:
            for (a, b), count in self._horizontal_seam_label_pairs(label_data, y, dilation_px).items():
                if count >= min_contact_pixels:
                    union(a, b)

        merge_groups = {
            find(value)
            for value in parent
            if find(value) != value or any(find(other) == value and other != value for other in parent)
        }
        if not merge_groups:
            return label_data, 0

        values = np.unique(label_data)
        values = values[values != 0]
        root_to_output: dict[int, int] = {}
        mapping = np.zeros(int(values.max()) + 1, dtype=np.uint32)
        next_id = 1
        for value in values:
            root = find(int(value))
            if root not in root_to_output:
                root_to_output[root] = next_id
                next_id += 1
            mapping[int(value)] = root_to_output[root]
        return mapping[label_data], len(merge_groups)

    def _vertical_seam_label_pairs(
        self,
        labels: np.ndarray,
        x: int,
        dilation_px: int,
    ) -> dict[tuple[int, int], int]:
        height, width = labels.shape
        max_gap = max(0, int(dilation_px))
        counts: dict[tuple[int, int], int] = {}
        left_columns = range(max(0, x - max_gap - 1), x)
        right_columns = range(x, min(width, x + max_gap + 1))
        for left_x in left_columns:
            for right_x in right_columns:
                for dy in range(-max_gap, max_gap + 1):
                    if dy >= 0:
                        left = labels[: height - dy, left_x]
                        right = labels[dy:, right_x]
                    else:
                        left = labels[-dy:, left_x]
                        right = labels[: height + dy, right_x]
                    self._accumulate_label_pair_counts(counts, left, right)
        return counts

    def _horizontal_seam_label_pairs(
        self,
        labels: np.ndarray,
        y: int,
        dilation_px: int,
    ) -> dict[tuple[int, int], int]:
        height, width = labels.shape
        max_gap = max(0, int(dilation_px))
        counts: dict[tuple[int, int], int] = {}
        top_rows = range(max(0, y - max_gap - 1), y)
        bottom_rows = range(y, min(height, y + max_gap + 1))
        for top_y in top_rows:
            for bottom_y in bottom_rows:
                for dx in range(-max_gap, max_gap + 1):
                    if dx >= 0:
                        top = labels[top_y, : width - dx]
                        bottom = labels[bottom_y, dx:]
                    else:
                        top = labels[top_y, -dx:]
                        bottom = labels[bottom_y, : width + dx]
                    self._accumulate_label_pair_counts(counts, top, bottom)
        return counts

    def _accumulate_label_pair_counts(
        self,
        counts: dict[tuple[int, int], int],
        a: np.ndarray,
        b: np.ndarray,
    ) -> None:
        valid = (a != 0) & (b != 0) & (a != b)
        if not np.any(valid):
            return
        aa = np.asarray(a[valid], dtype=np.uint64)
        bb = np.asarray(b[valid], dtype=np.uint64)
        low = np.minimum(aa, bb)
        high = np.maximum(aa, bb)
        keys, key_counts = np.unique((low << np.uint64(32)) | high, return_counts=True)
        for key, count in zip(keys, key_counts, strict=False):
            packed = int(key)
            label_a = packed >> 32
            label_b = packed & 0xFFFFFFFF
            counts[(label_a, label_b)] = counts.get((label_a, label_b), 0) + int(count)

    def _video_roi_bounds_for_bundle(self, bundle: PromptBundle) -> RoiBounds | None:
        if not self._large_image_mode_enabled():
            return None
        anchor = roi_anchor_from_bundle(bundle)
        if anchor is None:
            self._log(
                "Large-image mode ON, but no point or box ROI anchor was found. "
                "Using full-stack 3D/video propagation."
            )
            return None
        image_hw = self._selection_image_hw(bundle.image)
        bounds = self._active_or_new_roi_bounds(
            bundle,
            anchor,
            image_hw,
            self._selected_roi_size(),
        )
        self._active_rois[bundle.image.layer_name] = bounds
        self._show_active_roi_overlay(bundle.image.layer_name, bounds)
        return bounds

    def _globalize_video_roi_result(
        self,
        result: Sam3Result,
        original_bundle: PromptBundle,
        bounds: RoiBounds | None,
    ) -> None:
        if bounds is None:
            return
        labels, masks, boxes = globalize_result_arrays(
            labels=result.labels,
            masks=result.masks,
            boxes_xyxy=result.boxes_xyxy,
            bounds=bounds,
            image_hw=self._selection_image_hw(original_bundle.image),
        )
        result.labels = labels
        result.masks = masks
        result.boxes_xyxy = boxes
        result.metadata["large_image_roi"] = (bounds.y0, bounds.x0, bounds.y1, bounds.x1)
        result.metadata["large_image_mode"] = True
        result.metadata["large_image_hw"] = self._selection_image_hw(original_bundle.image)
        result.metadata["result_space"] = "global_video_frame"

    def _localize_bundle_for_existing_video_roi(
        self,
        bundle: PromptBundle,
        session: Sam3Session,
    ) -> PromptBundle:
        roi = session.metadata.get("large_image_roi")
        if not roi:
            return bundle
        bounds = RoiBounds(y0=int(roi[0]), x0=int(roi[1]), y1=int(roi[2]), x1=int(roi[3]))
        local_shape = session.metadata.get("local_video_shape")
        if local_shape is None:
            local_shape = (
                selection_frame_count(bundle.image),
                bounds.height,
                bounds.width,
            )
        return localize_bundle_to_roi(bundle, bounds, tuple(int(value) for value in local_shape))

    def _globalize_existing_video_roi_result(
        self,
        result: Sam3Result,
        original_bundle: PromptBundle,
        session: Sam3Session,
    ) -> None:
        roi = session.metadata.get("large_image_roi")
        if not roi:
            return
        bounds = RoiBounds(y0=int(roi[0]), x0=int(roi[1]), y1=int(roi[2]), x1=int(roi[3]))
        self._globalize_video_roi_result(result, original_bundle, bounds)

    def _run_video_task(self, bundle: PromptBundle) -> None:
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        image_layer = self.viewer.layers[bundle.image.layer_name]
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run video task: {exc}")
            return
        direction = self.propagation_direction_combo.currentText()
        video_data = image_layer.data
        video_bundle = bundle
        video_roi_bounds = self._video_roi_bounds_for_bundle(bundle)
        if video_roi_bounds is not None:
            try:
                video_data = extract_video_xy_roi(image_layer.data, bundle.image, video_roi_bounds)
                video_bundle = localize_bundle_to_roi(bundle, video_roi_bounds, tuple(video_data.shape))
            except Exception as exc:
                self._log(f"Cannot run 3D/video local ROI task: {exc}")
                return
        diagnostics = Sam3Diagnostics(self._log) if self._sam31_diagnostics_enabled() else None
        if diagnostics is not None:
            diagnostics.log(
                "SAM3.1 image source: "
                f"{diagnostics.describe_image_source(video_data)}"
            )
            diagnostics.log_prompt_diagnostics(video_bundle)

        @thread_worker
        def run_video():
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before start_video_session")
                diagnostics.log_runtime_diagnostics(adapter, stage="before session start")
                session_t0 = time.perf_counter()
            session = adapter.start_video_session(video_data, video_bundle)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 session start", session_t0)
                diagnostics.log(
                    "SAM3.1 session ready: "
                    f"{session.session_id}; prompt_frame={video_bundle.image.frame_index or 0}; "
                    f"boxes={len(getattr(video_bundle, 'boxes', []) or [])}; "
                    f"points={len(getattr(video_bundle, 'points', []) or [])}."
                )
                diagnostics.log_session_diagnostics(adapter, session, stage="after session start")
                prompt_t0 = time.perf_counter()
            prompt_result = adapter.add_video_prompt(video_bundle, session)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 prompt insertion", prompt_t0)
                diagnostics.log_session_diagnostics(adapter, session, stage="after prompt insertion")
            self._globalize_video_roi_result(prompt_result, bundle, video_roi_bounds)
            prompt_result.metadata["image_layer"] = bundle.image.layer_name
            yield prompt_result
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before propagation")
                diagnostics.log_runtime_diagnostics(adapter, stage="before propagation")
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(video_bundle, session, direction=direction)
                )
            else:
                iterator = adapter.propagate_video(video_bundle, session, direction=direction)
            for result in iterator:
                self._globalize_video_roi_result(result, bundle, video_roi_bounds)
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            if video_roi_bounds is not None:
                session.metadata["large_image_roi"] = (
                    video_roi_bounds.y0,
                    video_roi_bounds.x0,
                    video_roi_bounds.y1,
                    video_roi_bounds.x1,
                )
                session.metadata["large_image_hw"] = self._selection_image_hw(bundle.image)
                session.metadata["local_video_shape"] = tuple(int(value) for value in getattr(video_data, "shape", ()))
            return session

        worker = run_video()
        worker.yielded.connect(self._write_video_result)
        worker.returned.connect(self._set_video_session)
        self._start_worker(worker)
        if video_roi_bounds is not None:
            self._log(
                f"3D/video local ROI ON: using fixed XY ROI "
                f"({video_roi_bounds.width} x {video_roi_bounds.height}) across all frames; "
                f"y={video_roi_bounds.y0}:{video_roi_bounds.y1}, "
                f"x={video_roi_bounds.x0}:{video_roi_bounds.x1}."
            )
        self._log(f"Started video propagation from frame {bundle.image.frame_index or 0}.")

    def _propagate_existing_session(self) -> None:
        if self.video_session is None:
            self._log("No active SAM3 video session. Run a 3D/video task first.")
            return
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot propagate session: {exc}")
            return
        if not adapter.has_video_session(self.video_session):
            self._reset_video_session()
            self._log(
                "The previous SAM3 video session is no longer active. "
                "Run a new 3D/video preview before propagating again."
            )
            return
        try:
            bundle = self._collect_bundle()
        except Exception as exc:
            self._log(f"Cannot collect prompts: {exc}")
            return
        session = self.video_session
        direction = self.propagation_direction_combo.currentText()
        video_bundle = self._localize_bundle_for_existing_video_roi(bundle, session)
        diagnostics = Sam3Diagnostics(self._log) if self._sam31_diagnostics_enabled() else None
        if diagnostics is not None:
            diagnostics.log_prompt_diagnostics(video_bundle)
            diagnostics.log_session_diagnostics(adapter, session, stage="before existing-session propagation")
            diagnostics.log_cuda_diagnostics("before existing-session propagation")
            diagnostics.log_runtime_diagnostics(adapter, stage="before existing-session propagation")

        @thread_worker
        def propagate():
            if diagnostics is not None:
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(video_bundle, session, direction=direction),
                    label="SAM3.1 existing-session propagation",
                )
            else:
                iterator = adapter.propagate_video(video_bundle, session, direction=direction)
            for result in iterator:
                self._globalize_existing_video_roi_result(result, bundle, session)
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = propagate()
        worker.yielded.connect(self._write_video_result)
        worker.returned.connect(self._set_video_session)
        self._start_worker(worker)
        self._log(f"Propagating existing session {session.session_id}.")

    def _write_image_result(self, result: Sam3Result) -> None:
        if self.layer_writer is None:
            return
        if result.is_empty():
            self.activity_status.set_no_objects_found()
            self._update_result_visibility(result)
            self._log("SAM3 returned no result.")
            self._log(self._result_summary(result))
            self._log_large_image_result_guidance(result)
            self._log_text_result_guidance(result)
            return
        update_boxes = self._should_update_boxes_for_result(result)
        labels_name = "SAM3 preview labels"
        mask_name = "SAM3 preview masks"
        boxes_name = "SAM3 preview boxes"
        if result.metadata.get("large_image_tiled_scan"):
            labels_name = "SAM3 tiled exemplar labels"
            mask_name = "SAM3 tiled exemplar masks"
            boxes_name = "SAM3 tiled exemplar boxes"
        self.layer_writer.write_result(
            result,
            labels_name=labels_name,
            mask_name=mask_name,
            boxes_name=boxes_name,
            update_boxes=update_boxes,
        )
        if self._current_task() == Sam3Task.REFINE:
            self._activate_points_layer_for_live_refinement()
        self._append_result_rows(result)
        self._update_result_visibility(result)
        self.activity_status.set_preview_ready()
        self._last_quick_mask_path = None
        self._sync_preview_output_controls()
        self._log(self._result_summary(result))
        if result.metadata.get("large_image_tiled_scan"):
            tile_count = int(result.metadata.get("tile_count") or 0)
            nonzero = int(np.count_nonzero(result.labels)) if result.labels is not None else 0
            self._log(f"Tiled exemplar scan wrote {nonzero} labeled pixel(s) from {tile_count} tile(s).")
            if result.metadata.get("tile_seam_merge_enabled"):
                merge_count = int(result.metadata.get("tile_seam_merge_count") or 0)
                self._log(f"Tile seam merge reconnected {merge_count} split object group(s).")
        self._log_large_image_result_guidance(result)
        self._log_text_result_guidance(result)

    def _write_batch_image_result(self, result: Sam3Result) -> None:
        if self.layer_writer is None:
            return
        image_layer_name = str(result.metadata.get("image_layer") or "")
        prompt = str(result.metadata.get("batch_prompt") or result.metadata.get("text_prompt_used") or "")
        suffix = self._safe_layer_suffix(
            f"{image_layer_name} - {prompt}" if prompt else image_layer_name or "image"
        )
        if result.is_empty():
            self.activity_status.set_no_objects_found()
            self._update_result_visibility(result)
            target = f"{image_layer_name} / {prompt}" if prompt else image_layer_name
            self._log(f"SAM3 returned no result for '{target}'.")
            self._log(self._result_summary(result))
            self._log_large_image_result_guidance(result)
            self._log_text_result_guidance(result)
            return
        self.layer_writer.write_result(
            result,
            labels_name=f"SAM3 preview labels [{suffix}]",
            mask_name=f"SAM3 preview masks [{suffix}]",
            boxes_name=f"SAM3 preview boxes [{suffix}]",
            update_boxes=self._should_update_boxes_for_result(result),
        )
        self._append_result_rows(result)
        self._update_result_visibility(result)
        self.activity_status.set_preview_ready()
        self._last_quick_mask_path = None
        self._sync_preview_output_controls()
        target = f"{image_layer_name} / {prompt}" if prompt else image_layer_name
        self._log(f"{target}: {self._result_summary(result)}")
        self._log_large_image_result_guidance(result)
        self._log_text_result_guidance(result)

    def _write_video_result(self, result: Sam3Result) -> None:
        if self.layer_writer is None or self.viewer is None:
            return
        image_layer_name = result.metadata.get("image_layer", self._current_image_layer_name())
        image_layer = self.viewer.layers[image_layer_name]
        selection = self._image_selection_for_layer(image_layer)
        output_shape = selection_video_output_shape(selection)
        self.layer_writer.write_video_frame_result(
            result,
            output_shape,
            labels_name="SAM3 propagated preview labels",
        )
        self._append_result_rows(result)
        self._update_result_visibility(result)
        self.activity_status.set_preview_ready()
        self._last_quick_mask_path = None
        self._sync_preview_output_controls()
        self._log(self._result_summary(result))

    def _set_video_session(self, session: Sam3Session) -> None:
        self.video_session = session
        self._sync_shared_runtime_state()
        self._update_video_session_visibility(True)
        self._sync_run_controls()
        self._log(f"Video session ready: {session.session_id}")

    def _reset_video_session(self) -> None:
        self.video_session = None
        if self.adapter is not None:
            self.adapter.video_session = None
        self._sync_shared_runtime_state()
        self._update_video_session_visibility(False)
        self._sync_run_controls()

    def _update_result_visibility(self, result: Sam3Result) -> None:
        if self.shared_context is None:
            return
        state = self.shared_context.result_visibility.on_result_written(result)
        self.shared_context.result_state = state

    def _update_video_session_visibility(self, has_session: bool) -> None:
        if self.shared_context is None:
            return
        state = self.shared_context.result_visibility.on_video_session_changed(has_session)
        self.shared_context.result_state = state

    def _on_image_initialized(self, message: str) -> None:
        self._log(message)
        self._log("Next: choose a task, create a prompt layer if needed, then run preview.")

    def _on_video_initialized(self, message: str) -> None:
        self._log(message)
        self._log("Next: select a frame/slice, add prompts, then run preview or propagate.")

    def _collect_bundle(self) -> PromptBundle:
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")

        image_layer_name = self._current_image_layer_name()
        if not image_layer_name:
            raise RuntimeError("Select an image layer first.")

        channel_axis = self.channel_axis_spin.value()
        return self.prompt_collector.collect(
            self.viewer,
            image_layer_name=image_layer_name,
            task=self._current_task(),
            points_layer_name=self._optional_combo_data(self.points_layer_combo),
            shapes_layer_name=self._optional_combo_data(self.shapes_layer_combo),
            labels_layer_name=self._optional_combo_data(self.labels_layer_combo),
            text=self.text_prompt_edit.text(),
            channel_axis=None if channel_axis < 0 else channel_axis,
            collect_exemplar_rois=not self._large_image_mode_enabled(),
        )

    def _safe_layer_suffix(self, name: str) -> str:
        suffix = "".join(char if char.isalnum() or char in ("-", "_", " ") else "_" for char in name)
        suffix = " ".join(suffix.split())
        return suffix or "image"

    def _large_image_mode_enabled(self) -> bool:
        return bool(hasattr(self, "large_image_check") and self.large_image_check.isChecked())

    def _external_exemplar_source_enabled(self) -> bool:
        return bool(
            hasattr(self, "exemplar_source_combo")
            and self.exemplar_source_combo.currentData() == "crop"
        )

    def _sam31_diagnostics_enabled(self) -> bool:
        return bool(
            hasattr(self, "sam31_diagnostics_check")
            and self.sam31_diagnostics_check.isChecked()
            and self._current_task() == Sam3Task.SEGMENT_3D
        )

    def _selected_roi_size(self) -> tuple[int, int]:
        if not hasattr(self, "roi_size_combo"):
            return (1024, 1024)
        value = self.roi_size_combo.currentData()
        if isinstance(value, tuple) and len(value) == 2:
            return int(value[0]), int(value[1])
        return (1024, 1024)

    def _on_large_image_mode_changed(self, *_args: Any) -> None:
        enabled = self._large_image_mode_enabled()
        self._sync_task_setup_visibility()
        self._sync_run_controls()
        if enabled:
            width, height = self._selected_roi_size()
            self._log(f"Large-image mode ON: local ROI inference ({width} x {height}).")
        else:
            self._active_rois.clear()
            self._clear_active_roi_overlay()
            self._log("Large-image mode OFF: full-image inference.")

    def _on_exemplar_source_changed(self, *_args: Any) -> None:
        self._sync_task_setup_visibility()
        self._sync_run_controls()

    def _selection_image_hw(self, selection) -> tuple[int, int]:
        y_axis, x_axis = selection.spatial_axes
        return int(selection.data_shape[y_axis]), int(selection.data_shape[x_axis])

    def _cache_context_for_layer(
        self,
        image_layer: Any,
        bundle: PromptBundle,
        *,
        roi_bounds: RoiBounds | None = None,
    ) -> dict[str, Any]:
        roi_tuple = None
        if roi_bounds is not None:
            roi_tuple = (int(roi_bounds.y0), int(roi_bounds.x0), int(roi_bounds.y1), int(roi_bounds.x1))
        return {
            "layer_name": str(getattr(image_layer, "name", bundle.image.layer_name) or bundle.image.layer_name),
            "layer_identity": id(image_layer),
            "frame_index": bundle.image.frame_index,
            "channel_index": bundle.image.channel_index,
            "roi_bounds": roi_tuple,
        }


    def _current_image_canvas_shape(self, image_layer: Any) -> tuple[int, int]:
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        channel_axis = self.channel_axis_spin.value()
        base_data = self._base_layer_data(image_layer.data)
        selection = infer_image_selection(
            layer_name=image_layer.name,
            data_shape=tuple(base_data.shape),
            dims_current_step=tuple(self.viewer.dims.current_step),
            channel_axis=None if channel_axis < 0 else channel_axis,
        )
        frame = np.asarray(extract_2d_image(base_data, selection))
        if frame.ndim == 2:
            return tuple(int(v) for v in frame.shape)
        return int(frame.shape[0]), int(frame.shape[1])

    def _base_layer_data(self, data: Any) -> Any:
        if isinstance(data, (list, tuple)) and data:
            return data[0]
        try:
            if hasattr(data, "_data") and hasattr(data, "__len__") and len(data):
                return data[0]
        except Exception:
            pass
        return data

    def _active_or_new_roi_bounds(
        self,
        bundle: PromptBundle,
        anchor: tuple[float, float],
        image_hw: tuple[int, int],
        roi_size: tuple[int, int],
    ) -> RoiBounds:
        current = self._active_rois.get(bundle.image.layer_name)
        anchor_y, anchor_x = anchor
        if current is not None and current.contains_yx(anchor_y, anchor_x):
            return current
        if bundle.boxes and not bundle.points:
            return box_roi_bounds(bundle.boxes[-1], image_hw=image_hw, roi_hw=roi_size)
        return centered_roi_bounds(anchor_y, anchor_x, image_hw=image_hw, roi_hw=roi_size)

    def _show_active_roi_overlay(
        self,
        image_layer_name: str,
        bounds: RoiBounds | None,
        *,
        extra_bounds: list[tuple[str, RoiBounds]] | None = None,
    ) -> None:
        if self.viewer is None:
            return
        if extra_bounds is not None:
            rows = extra_bounds
        elif bounds is not None:
            rows = [(image_layer_name, bounds)]
        else:
            rows = []
        rectangles = [roi_bounds.as_rectangle() for _name, roi_bounds in rows]
        properties = {"source": np.asarray([name for name, _bounds in rows], dtype=object)}
        name = "SAM3 active ROI"
        try:
            layer = self.viewer.layers[name]
        except (KeyError, ValueError):
            layer = self.viewer.add_shapes(
                rectangles,
                shape_type="rectangle",
                name=name,
                edge_color="#d6a657",
                face_color="#d6a65722",
                properties=properties,
            )
        else:
            layer.data = rectangles
            layer.properties = properties
        self._set_layer_mode(layer, "select")

    def _clear_active_roi_overlay(self) -> None:
        if self.viewer is None:
            return
        try:
            layer = self.viewer.layers["SAM3 active ROI"]
        except (KeyError, ValueError):
            return
        self.viewer.layers.remove(layer)

    def _multi_text_prompts(self) -> list[str]:
        if not hasattr(self, "multi_text_prompt_edit"):
            return []
        prompts: list[str] = []
        seen: set[str] = set()
        for line in self.multi_text_prompt_edit.toPlainText().splitlines():
            prompt = " ".join(line.strip().split())
            if not prompt or prompt.startswith("#"):
                continue
            key = prompt.lower()
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
        return prompts

    def _ensure_adapter(self, *, reset: bool = False) -> Sam3Adapter:
        config = self._adapter_config()
        if reset or self.adapter is None or self.adapter.config != config:
            if self.adapter is not None:
                self.adapter.unload()
            self.adapter = Sam3Adapter(config)
            self._sync_shared_runtime_state()
        return self.adapter

    def _ensure_image_adapter_loaded_for_bundle(
        self,
        adapter: Sam3Adapter,
        bundle: PromptBundle,
    ) -> None:
        needs_interactivity = self._bundle_needs_instance_interactivity(bundle)
        if adapter.image_processor is None:
            adapter.load_image(enable_instance_interactivity=needs_interactivity)
        elif needs_interactivity and not adapter._has_instance_interactivity():
            adapter.load_image(enable_instance_interactivity=True)

    def _bundle_needs_instance_interactivity(self, bundle: PromptBundle) -> bool:
        return bool(
            bundle.points
            or bundle.masks
            or bundle.task in {Sam3Task.REFINE, Sam3Task.EXEMPLAR}
        )

    def _adapter_config(self) -> Sam3AdapterConfig:
        model_dir = self.model_dir_edit.text().strip()
        model_type = self._current_model_type()
        checkpoint = self._checkpoint_path_from_model_dir(model_dir, model_type) if model_dir else None
        if checkpoint is None:
            expected = ", ".join(self._expected_weight_names(model_type))
            raise RuntimeError(
                f"Select a {self.model_type_combo.currentText()} directory containing: {expected}."
            )
        device = self._current_runtime_device()
        self._log(
            "Selected device: "
            f"{device}; torch.cuda.is_available()={torch.cuda.is_available()}; "
            f"torch.version.cuda={torch.version.cuda}"
        )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(CUDA_CPU_ONLY_MESSAGE)
        if device == "cpu" and self._current_task() == Sam3Task.SEGMENT_3D:
            raise RuntimeError(CPU_3D_MESSAGE)
        if device == "cpu" and model_type == "sam3.1":
            raise RuntimeError(CPU_SAM31_MESSAGE)
        bpe_path = self._bpe_path_from_model_dir(model_dir)
        if device == "cpu":
            self._log(CPU_EXPERIMENTAL_2D_MESSAGE)
            self._log(
                "CPU SAM3.0 image config: "
                f"model_type={model_type}; device={device}; "
                f"checkpoint={checkpoint}; bpe_path={bpe_path}"
            )
        cuda_issue = cuda_compatibility_issue()
        if device == "cuda" and cuda_issue:
            self._log(
                "CUDA selected despite PyTorch architecture warning. "
                f"Continuing on GPU for testing: {cuda_issue}"
            )
        return Sam3AdapterConfig(
            checkpoint_path=checkpoint,
            bpe_path=bpe_path,
            device=device,
            confidence_threshold=float(self.confidence_threshold_spin.value()),
            load_from_hf=False,
        )

    def _cpu_bundle_support_error(self, bundle: PromptBundle) -> str | None:
        if self._current_runtime_device() != "cpu":
            return None
        return cpu_prompt_support_error(
            bundle.task.value,
            has_text=bool(bundle.text and bundle.text.text.strip()),
            has_points=bool(bundle.points),
            has_boxes=bool(bundle.boxes),
            has_masks=bool(bundle.masks),
            has_exemplars=bool(bundle.exemplars),
        )

    def _current_runtime_device(self) -> str:
        if manual_device_override_enabled():
            device = self.device_combo.currentData()
            if device in {"cuda", "cpu"}:
                return device
        return runtime_device(torch.cuda.is_available())

    def _checkpoint_path_from_model_dir(
        self,
        model_dir: str,
        model_type: str | None = None,
    ) -> Path | None:
        path = Path(model_dir)
        for name in self._expected_weight_names(model_type):
            candidate = path / name
            if candidate.exists():
                return candidate
        return None


    def _bpe_path_from_model_dir(self, model_dir: str) -> Path | None:
        path = Path(model_dir)

        candidates = (
            path / "bpe_simple_vocab_16e6.txt.gz",
            path / "merges.txt.gz",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        merges = path / "merges.txt"
        if merges.exists():
            gz_path = path / "bpe_simple_vocab_16e6.txt.gz"
            try:
                import gzip

                with open(merges, "rb") as src, gzip.open(gz_path, "wb") as dst:
                    dst.write(src.read())

                self._log(f"Created SAM3 BPE tokenizer file: {gz_path}")
                return gz_path
            except Exception as exc:
                self._log(f"Could not create SAM3 BPE tokenizer from merges.txt: {exc}")
                return None

        return None

    def _expected_weight_names(self, model_type: str | None = None) -> tuple[str, ...]:
        if model_type == "sam3.1":
            return ("sam3.1_multiplex.pt",)
        if model_type == "sam3":
            return ("sam3.pt", "model.safetensors")
        return ("sam3.1_multiplex.pt", "sam3.pt", "model.safetensors")

    def _current_model_type(self) -> str:
        return self.model_type_combo.currentData() or "sam3"

    def _restore_settings(self) -> None:
        model_dir = self.settings.value("model_dir", "", type=str)
        self.model_dir_edit.setText(model_dir or "")
        model_type = self.settings.value("model_type", "sam3", type=str)
        if manual_device_override_enabled():
            default_device = runtime_device(torch.cuda.is_available())
            requested_device = self.settings.value("device", default_device, type=str)
            device, warning = normalize_requested_device(
                requested_device,
                torch.cuda.is_available(),
            )
            if warning:
                self._log(warning)
                self.settings.setValue("device", device)
        else:
            device = runtime_device(torch.cuda.is_available())
        threshold = self.settings.value("confidence_threshold", 0.35, type=float)
        sam31_diagnostics = self.settings.value("sam31_diagnostics", False, type=bool)

        old_model_signals = self.model_type_combo.blockSignals(True)
        old_device_signals = self.device_combo.blockSignals(True)
        old_threshold_signals = self.confidence_threshold_spin.blockSignals(True)
        old_diagnostics_signals = self.sam31_diagnostics_check.blockSignals(True)
        try:
            index = self.model_type_combo.findData(model_type)
            if index >= 0:
                self.model_type_combo.setCurrentIndex(index)
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
            self.confidence_threshold_spin.setValue(float(threshold))
            self.sam31_diagnostics_check.setChecked(bool(sam31_diagnostics))
        finally:
            self.model_type_combo.blockSignals(old_model_signals)
            self.device_combo.blockSignals(old_device_signals)
            self.confidence_threshold_spin.blockSignals(old_threshold_signals)
            self.sam31_diagnostics_check.blockSignals(old_diagnostics_signals)

        output_dir = self.settings.value("quick_mask_output_dir", "", type=str)
        if hasattr(self, "preview_output_folder_edit"):
            self.preview_output_folder_edit.setText(output_dir or "")
        output_format = self.settings.value("quick_mask_output_format", "TIFF", type=str)
        if hasattr(self, "preview_output_format_combo"):
            index = self.preview_output_format_combo.findText(output_format)
            if index >= 0:
                self.preview_output_format_combo.setCurrentIndex(index)

        if (
            self._current_model_type() == "sam3.1"
            and hasattr(self, "task_combo")
            and self._current_task() != Sam3Task.SEGMENT_3D
        ):
            index = self.task_combo.findData(Sam3Task.SEGMENT_3D)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)

    def _save_settings(self) -> None:
        self.settings.setValue("model_dir", self.model_dir_edit.text().strip())
        self.settings.setValue("model_type", self._current_model_type())
        self.settings.setValue("confidence_threshold", float(self.confidence_threshold_spin.value()))
        device = self.device_combo.currentData()
        self.settings.setValue("device", device or "")
        if hasattr(self, "preview_output_folder_edit"):
            self.settings.setValue("quick_mask_output_dir", self.preview_output_folder_edit.text().strip())
        if hasattr(self, "preview_output_format_combo"):
            self.settings.setValue("quick_mask_output_format", self.preview_output_format_combo.currentText())
        if hasattr(self, "sam31_diagnostics_check"):
            self.settings.setValue("sam31_diagnostics", self.sam31_diagnostics_check.isChecked())

    def _on_model_type_changed(self) -> None:
        self._save_settings()
        if (
            self._current_model_type() == "sam3.1"
            and hasattr(self, "task_combo")
            and self._current_task() != Sam3Task.SEGMENT_3D
        ):
            index = self.task_combo.findData(Sam3Task.SEGMENT_3D)
            if index >= 0:
                self.task_combo.setCurrentIndex(index)
            self._log("SAM3.1 video multiplex selected; task set to 3D/video propagation.")
        self._sync_model_type_controls()
        if self.adapter is not None:
            self._unload_adapter()
            self._log("Model type changed; unloaded SAM3 model so it reloads from the selected folder.")

    def _on_device_changed(self) -> None:
        if not manual_device_override_enabled():
            return
        self._save_settings()
        if self.adapter is not None:
            self._unload_adapter()
            self._log("Device changed; unloaded SAM3 model so it reloads on the selected device.")

    def _initialize_prompt_layer(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return

        task = self._current_task()
        tool = self.prompt_tool_combo.currentData()
        layer_name: str | None = None
        preserve_target_name = (
            self._current_image_layer_name()
            if task == Sam3Task.EXEMPLAR and self._external_exemplar_source_enabled()
            else ""
        )

        if tool == PROMPT_POINTS:
            if task == Sam3Task.REFINE:
                self._ensure_refinement_preview_labels()
            layer = self._ensure_points_prompt_layer()
            layer_name = layer.name
            self.viewer.layers.selection.active = layer
            self._set_layer_mode(layer, "add")
            if hasattr(self, "live_point_refinement"):
                self.live_point_refinement.set_points_layer(layer)
            self._log(
                "Created/selected SAM3 points layer. Add the first point to start "
                "Live Points; first run may take longer while the model loads."
            )
        elif tool == PROMPT_BOX:
            layer = self._ensure_shapes_prompt_layer()
            layer_name = layer.name
            self.viewer.layers.selection.active = layer
            self._set_layer_mode(layer, "add_rectangle")
            if task == Sam3Task.EXEMPLAR:
                if self._external_exemplar_source_enabled():
                    crop_name = self._optional_combo_data(self.exemplar_crop_layer_combo) or "(select crop image)"
                    target_name = self._current_image_layer_name() or "(select target image)"
                    self._log(
                        "Created/selected SAM3 boxes layer for crop exemplars. "
                        f"Draw ROI boxes on crop image '{crop_name}', then click "
                        f"Scan Target Image by Tiles to scan target '{target_name}'."
                    )
                else:
                    self._log(
                        "Created/selected SAM3 boxes layer for exemplars. Draw ROI boxes "
                        "around example objects, then click Run Preview."
                    )
            else:
                self._log(
                    "Created/selected SAM3 boxes layer. Draw box prompts, then click Run Preview."
                )
        elif tool == PROMPT_LABELS:
            layer = self._ensure_labels_prompt_layer()
            layer_name = layer.name
            self.viewer.layers.selection.active = layer
            self._set_layer_mode(layer, "paint")
            self._log(
                "Created/selected SAM3 labels prompt layer. Paint non-zero pixels, "
                "then click Run Preview."
            )
        else:
            self.text_prompt_edit.setFocus()
            self._log("Text prompt mode selected. Enter a phrase, then click Run Preview.")

        self._refresh_layers()
        if preserve_target_name:
            self._select_combo_data(self.image_layer_combo, preserve_target_name)
            self._refresh_exemplar_crop_layer_combo()
        if layer_name is not None:
            if tool == PROMPT_POINTS:
                self._select_combo_data(self.points_layer_combo, layer_name)
                self._set_current_point_polarity()
                self._sync_live_refinement_layer()
                self.viewer.layers.selection.active = self.viewer.layers[layer_name]
                self._set_layer_mode(self.viewer.layers[layer_name], "add")
                self._set_live_refinement_status("Activity: Live Points armed. Add a point to start.")
            elif tool == PROMPT_BOX:
                self._select_combo_data(self.shapes_layer_combo, layer_name)
            elif tool == PROMPT_LABELS:
                self._select_combo_data(self.labels_layer_combo, layer_name)

    def _ensure_points_prompt_layer(self) -> Points:
        assert self.viewer is not None
        name = "SAM3 points"
        try:
            layer = self.viewer.layers[name]
            if isinstance(layer, Points):
                self._set_current_point_polarity()
                return layer
        except (KeyError, ValueError):
            pass

        layer = self.viewer.add_points(
            np.empty((0, 2), dtype=float),
            name=name,
            properties={"polarity": np.asarray([], dtype=object)},
            property_choices={"polarity": ["positive", "negative"]},
            face_color="polarity",
            face_color_cycle=["#2fb344", "#e03131"],
            symbol="disc",
            size=12,
        )
        self._set_current_point_polarity()
        return layer

    def _ensure_refinement_preview_labels(self) -> None:
        if self.viewer is None:
            return
        try:
            image_name = self._current_image_layer_name()
            image_layer = self.viewer.layers[image_name]
        except Exception:
            return
        try:
            self.viewer.layers["SAM3 preview labels"]
            return
        except (KeyError, ValueError):
            pass
        frame_shape = self._current_image_canvas_shape(image_layer)
        labels = self.viewer.add_labels(
            np.zeros(frame_shape, dtype=np.uint32),
            name="SAM3 preview labels",
        )
        labels.visible = True

    def _activate_points_layer_for_live_refinement(self) -> None:
        layer = self._current_points_layer()
        if layer is None or self.viewer is None:
            return
        self.viewer.layers.selection.active = layer
        self._set_layer_mode(layer, "add")

    def _ensure_shapes_prompt_layer(self) -> Shapes:
        assert self.viewer is not None
        name = "SAM3 boxes"
        try:
            layer = self.viewer.layers[name]
            if isinstance(layer, Shapes):
                return layer
        except (KeyError, ValueError):
            pass

        layer = self.viewer.add_shapes(
            name=name,
            shape_type="rectangle",
            edge_color="#2f9e44",
            face_color="#2f9e4433",
        )
        return layer

    def _ensure_labels_prompt_layer(self) -> Labels:
        assert self.viewer is not None
        image_name = self._current_image_layer_name()
        if not image_name:
            raise RuntimeError("Select an image layer before creating a labels prompt layer.")
        image_layer = self.viewer.layers[image_name]
        name = "SAM3 mask prompt"
        try:
            layer = self.viewer.layers[name]
            if isinstance(layer, Labels):
                return layer
        except (KeyError, ValueError):
            pass

        data = np.zeros(self._current_image_canvas_shape(image_layer), dtype=np.uint8)
        layer = self.viewer.add_labels(data, name=name)
        return layer

    def _set_current_point_polarity(self) -> None:
        layer = self._current_points_layer()
        if layer is None:
            return
        polarity = self.point_polarity_combo.currentData() or "positive"
        if hasattr(self, "live_point_refinement"):
            with self.live_point_refinement.suspend_events():
                layer.current_properties = {"polarity": np.asarray([polarity], dtype=object)}
        else:
            layer.current_properties = {"polarity": np.asarray([polarity], dtype=object)}

    def _on_points_layer_changed(self) -> None:
        self._set_current_point_polarity()
        if hasattr(self, "live_point_refinement"):
            self._sync_live_refinement_layer()

    def _apply_polarity_to_selected_points(self) -> None:
        layer = self._current_points_layer()
        if layer is None:
            self._log("No points layer selected.")
            return
        selected = sorted(getattr(layer, "selected_data", []))
        if not selected:
            self._log("Select one or more points before applying the positive/negative type.")
            return

        polarity = self.point_polarity_combo.currentData() or "positive"
        properties = dict(getattr(layer, "properties", {}) or {})
        values = self._point_polarity_values(layer)
        for idx in selected:
            values[idx] = polarity
        properties["polarity"] = np.asarray(values, dtype=object)
        layer.properties = properties
        layer.refresh_colors()
        self._log(f"Updated {len(selected)} selected point(s) to {polarity}.")
        if hasattr(self, "live_point_refinement"):
            self.live_point_refinement.request_preview()

    def _current_points_layer(self) -> Points | None:
        if self.viewer is None:
            return None
        layer_name = self._optional_combo_data(self.points_layer_combo)
        if not layer_name:
            return None
        try:
            layer = self.viewer.layers[layer_name]
        except (KeyError, ValueError):
            return None
        return layer if isinstance(layer, Points) else None

    def _point_polarity_values(self, layer: Points) -> list[str]:
        properties = dict(getattr(layer, "properties", {}) or {})
        values = [str(value) for value in list(properties.get("polarity", []))]
        if len(values) < len(layer.data):
            values.extend(["positive"] * (len(layer.data) - len(values)))
        return [
            "negative" if str(value).strip().lower() == "negative" else "positive"
            for value in values[: len(layer.data)]
        ]

    def _task_guidance(self, task: Sam3Task) -> str:
        if task == Sam3Task.TEXT:
            return "Text segmentation: enter a phrase and click Run Preview. No prompt layer is required."
        if task == Sam3Task.EXEMPLAR:
            return "Exemplar segmentation: create a box prompt layer and draw ROI boxes around examples."
        if task == Sam3Task.REFINE:
            return (
                "Live Points: create a points layer and add a point to start point correction. "
                "Use T for next point mode; Shift+T flips selected/latest point and reruns."
            )
        if task == Sam3Task.SEGMENT_3D:
            return (
                "3D/video: select the frame/slice, then use one SAM3.0 box, SAM3.1 "
                "box prompts, or up to 16 points for one object before propagating. "
                "Labels-mask prompts are 2D only."
            )
        return "2D segmentation: use text, points, boxes, or a labels-mask prompt, then run preview."

    def _select_combo_data(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_layer_mode(self, layer: Any, mode: str) -> None:
        try:
            layer.mode = mode
        except Exception:
            pass

    def _start_worker(
        self,
        worker: Any,
        on_returned: Any | None = None,
        activity_status: str | None = None,
    ) -> None:
        self._cancel_worker()
        self._worker = worker
        self._worker_failed = False
        self._sync_shared_runtime_state()
        if on_returned is not None:
            worker.returned.connect(on_returned)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        if activity_status is not None:
            self.activity_status.set_status(activity_status)
            self._set_live_refinement_status(f"Activity: {activity_status}")
        else:
            self._set_activity_running_message()
        worker.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            try:
                self._worker.quit()
            except Exception:
                pass
            self._worker = None
            self._set_running(False)
            self._sync_shared_runtime_state()
            self.activity_status.set_ready()
            if self._current_task() == Sam3Task.SEGMENT_3D:
                self._reset_video_session()
                self._log("Cancelled 3D/video task; run preview again to start a new SAM3 session.")

    def _on_worker_error(self, error: Any) -> None:
        self._worker_failed = True
        self.activity_status.set_task_failed()
        self._sync_shared_runtime_state()
        self._log(f"SAM3 task failed: {error}")
        if self._is_missing_video_session_error(error):
            self._reset_video_session()
            self._log(
                "The SAM3 video session expired or was cancelled. "
                "Run a new 3D/video preview before propagating again."
            )
        if _is_cuda_kernel_image_error(error):
            self._log(
                "CUDA kernel compatibility failure detected. The selected GPU is visible, "
                "but at least one PyTorch, torchvision, or SAM3 CUDA kernel was not built "
                "for this device architecture. Select CPU for this workflow, or install "
                "a build that supports the GPU."
            )
        if self._current_task() == Sam3Task.REFINE and self.prompt_tool_combo.currentData() == PROMPT_POINTS:
            self._set_live_refinement_status("Activity: failed")

    def _is_missing_video_session_error(self, error: Any) -> bool:
        text = str(error)
        return "Cannot find session" in text and "might have expired" in text

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_running(False)
        if self._worker_failed:
            self.activity_status.set_task_failed()
            self._log("SAM3 task stopped after failure.")
        else:
            self.activity_status.finish_success()
            self._log("SAM3 task finished.")
            self.task_complete_sound.play_task_complete()
        self._sync_shared_runtime_state()

        if self._current_task() == Sam3Task.REFINE and self.prompt_tool_combo.currentData() == PROMPT_POINTS:
            self._set_live_refinement_status("Activity: Live Points ready")
        elif self._worker_failed:
            self._set_live_refinement_status("Activity: failed")
        else:
            self._set_live_refinement_status("Activity: idle")

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self._sync_run_controls()
        self.setCursor(Qt.BusyCursor if running else Qt.ArrowCursor)

    def _set_activity_running_message(self) -> None:
        task = self._current_task()
        if task == Sam3Task.REFINE:
            self.activity_status.set_running_preview()
            self._set_live_refinement_status("Activity: Live Points running...")
        elif task == Sam3Task.SEGMENT_3D:
            self.activity_status.set_starting_3d_propagation()
            self._set_live_refinement_status("Activity: 3D/video propagation running...")
        else:
            self.activity_status.set_running_preview()
            self._set_live_refinement_status("Activity: SAM3 preview running...")

    def _clear_preview_layers(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        preview_names = (
            "SAM3 preview labels",
            "SAM3 preview masks",
            "SAM3 preview boxes",
            "SAM3 tiled exemplar labels",
            "SAM3 tiled exemplar masks",
            "SAM3 tiled exemplar boxes",
            "SAM3 propagated preview labels",
        )
        preview_prefixes = (
            "SAM3 preview labels [",
            "SAM3 preview masks [",
            "SAM3 preview boxes [",
        )
        removed = 0
        for layer in list(self.viewer.layers):
            name = getattr(layer, "name", "")
            if name not in preview_names and not any(name.startswith(prefix) for prefix in preview_prefixes):
                continue
            self.viewer.layers.remove(layer)
            removed += 1
        if removed:
            if self.shared_context is not None:
                state = self.shared_context.result_visibility.on_preview_layers_cleared()
                self.shared_context.result_state = state
            self._last_quick_mask_path = None
            self._sync_preview_output_controls()
            self._log(f"Cleared {removed} SAM3 preview layer(s). Prompts and saved labels were kept.")
        else:
            self._log("No SAM3 preview layers found to clear.")

    def _sync_preview_output_controls(self) -> None:
        if not hasattr(self, "save_release_btn"):
            return
        preview_layer = self._first_preview_labels_layer()
        has_preview = preview_layer is not None
        has_saved_path = self._last_quick_mask_path is not None
        self.save_release_btn.setEnabled(has_preview and self._worker is None)
        self.preview_output_filename_edit.setEnabled(has_preview)
        self.preview_output_format_combo.setEnabled(has_preview)
        self.preview_output_folder_edit.setEnabled(has_preview)
        self.preview_output_browse_btn.setEnabled(has_preview)
        self.save_release_btn.setVisible(has_preview)
        self.preview_output_filename_edit.setVisible(has_preview)
        self.preview_output_format_combo.setVisible(has_preview)
        self.preview_output_folder_edit.setVisible(has_preview)
        self.preview_output_browse_btn.setVisible(has_preview)
        self.preview_output_folder_label.setVisible(has_preview)
        self.preview_output_format_label.setVisible(has_preview)
        self.preview_output_filename_label.setVisible(has_preview)
        self.saved_preview_path_label.setVisible(has_saved_path and not has_preview)
        self.open_saved_folder_btn.setVisible(has_saved_path and not has_preview)
        if has_saved_path:
            self.saved_preview_path_label.setText(f"Saved: {self._last_quick_mask_path.name}")
            self.saved_preview_path_label.setToolTip(str(self._last_quick_mask_path))
            self.open_saved_folder_btn.setToolTip(str(self._last_quick_mask_path.parent))
        self.preview_output_panel.setVisible(has_preview or has_saved_path)
        if has_preview:
            self._sync_preview_output_formats(preview_layer)
            self._update_preview_output_filename()

    def _first_preview_labels_layer(self) -> Any | None:
        if self.viewer is None:
            return None
        preferred_names = (
            "SAM3 preview labels",
            "SAM3 tiled exemplar labels",
            "SAM3 propagated preview labels",
        )
        for name in preferred_names:
            try:
                return self.viewer.layers[name]
            except (KeyError, ValueError):
                pass
        for layer in self.viewer.layers:
            name = getattr(layer, "name", "")
            if name.startswith("SAM3 preview labels ["):
                return layer
        return None

    def _update_preview_output_filename(self) -> None:
        if not hasattr(self, "preview_output_filename_edit"):
            return
        preview_layer = self._first_preview_labels_layer()
        if preview_layer is None:
            return
        folder = Path(self.preview_output_folder_edit.text().strip() or Path.home())
        base = self._quick_mask_base_name(preview_layer)
        stem = self._next_quick_mask_stem(base, folder, self.preview_output_format_combo.currentText())
        self.preview_output_filename_edit.setText(self._filename_for_format(stem, self.preview_output_format_combo.currentText()))

    def _sync_preview_output_formats(self, preview_layer: Any) -> None:
        previous = self.preview_output_format_combo.currentText()
        formats = ["TIFF", "NumPy (.npy)"]
        if np.asarray(preview_layer.data).ndim == 2:
            formats.append("PNG")
        current_items = [
            self.preview_output_format_combo.itemText(index)
            for index in range(self.preview_output_format_combo.count())
        ]
        if current_items != formats:
            old_signals = self.preview_output_format_combo.blockSignals(True)
            self.preview_output_format_combo.clear()
            self.preview_output_format_combo.addItems(formats)
            index = self.preview_output_format_combo.findText(previous)
            if index < 0:
                index = 0
                if previous == "PNG":
                    self._log("PNG is only available for 2D masks. Using TIFF for this 3D/video preview.")
            self.preview_output_format_combo.setCurrentIndex(index)
            self.preview_output_format_combo.blockSignals(old_signals)
            self._save_settings()

    def _save_preview_mask_and_release_memory(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        preview = self._first_preview_labels_layer()
        if preview is None:
            self._log("No preview Labels layer found to save.")
            self._sync_preview_output_controls()
            return
        output_dir = self.preview_output_folder_edit.text().strip()
        if not output_dir:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Select preview mask output folder",
                str(Path.home()),
            )
            if not selected:
                self._log("Choose an output folder before saving the preview mask.")
                return
            self.preview_output_folder_edit.setText(selected)
            output_dir = selected
        folder = Path(output_dir)
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(f"Cannot create output folder: {exc}")
            return

        filename = self.preview_output_filename_edit.text().strip()
        if not filename:
            self._update_preview_output_filename()
            filename = self.preview_output_filename_edit.text().strip()
        target_path = folder / filename
        data = preview.data.copy()
        saved_layer_name = self._unique_layer_name(Path(filename).stem)
        try:
            exported = self.mask_export_service.export(
                data,
                target_path,
                self.preview_output_format_combo.currentText(),
            )
        except Exception as exc:
            self._log(f"Could not save preview mask: {exc}")
            return
        self.viewer.add_labels(data, name=saved_layer_name)

        self._last_quick_mask_path = exported
        self._save_settings()
        removed = self._remove_preview_layers()
        self._release_preview_memory()
        self._unload_adapter()
        action = "clean"
        activity = "Activity: Saved. Model unloaded."
        self._set_live_refinement_status(activity)
        if hasattr(self, "live_refinement_status_label"):
            self.live_refinement_status_label.setToolTip(f"Saved to: {exported}")
        self._log(
            f"Saved preview mask to layer '{saved_layer_name}' and file: {exported}. "
            f"Completed {action}; removed {removed} preview layer(s)."
        )
        self._sync_preview_output_controls()

    def _open_saved_mask_folder(self) -> None:
        if self._last_quick_mask_path is None:
            self._log("No saved preview mask path is available.")
            return
        folder = self._last_quick_mask_path.parent
        if not folder.exists():
            self._log(f"Saved mask folder was not found: {folder}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            self._log(f"Could not open saved mask folder: {folder}")

    def _remove_preview_layers(self) -> int:
        if self.viewer is None:
            return 0
        removed = 0
        for layer in list(self.viewer.layers):
            name = getattr(layer, "name", "")
            if name in {
                "SAM3 preview labels",
                "SAM3 preview masks",
                "SAM3 preview boxes",
                "SAM3 propagated preview labels",
            } or name.startswith(("SAM3 preview labels [", "SAM3 preview masks [", "SAM3 preview boxes [")):
                self.viewer.layers.remove(layer)
                removed += 1
        if removed and self.shared_context is not None:
            state = self.shared_context.result_visibility.on_preview_layers_cleared()
            self.shared_context.result_state = state
        return removed

    def _release_preview_memory(self) -> None:
        gc.collect()
        try:
            import torch
        except Exception:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _quick_mask_base_name(self, preview_layer: Any) -> str:
        image_name = self._current_image_layer_name()
        preview_name = getattr(preview_layer, "name", "preview_mask")
        if preview_name.startswith("SAM3 preview labels ["):
            image_name = preview_name.removeprefix("SAM3 preview labels [").removesuffix("]")
        base = image_name or preview_name
        if self._large_image_mode_enabled():
            base = f"{base}_roi"
        return f"{self._safe_file_stem(base)}_mask"

    def _next_quick_mask_stem(self, base: str, folder: Path, fmt: str) -> str:
        for index in range(1, 10000):
            stem = f"{base}_{index:02d}"
            if not (folder / self._filename_for_format(stem, fmt)).exists():
                return stem
        return f"{base}_9999"

    def _filename_for_format(self, stem: str, fmt: str) -> str:
        fmt_key = fmt.lower()
        if fmt_key in {"numpy (.npy)", "npy"}:
            return f"{Path(stem).stem}.npy"
        if fmt_key == "png":
            return f"{Path(stem).stem}.png"
        return f"{Path(stem).stem}.tif"

    def _safe_file_stem(self, value: str) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
        return stem or "sam3_preview_mask"

    def _unique_layer_name(self, base: str) -> str:
        if self.viewer is None:
            return base
        existing = {getattr(layer, "name", "") for layer in self.viewer.layers}
        if base not in existing:
            return base
        for index in range(1, 10000):
            name = f"{base}_{index:02d}"
            if name not in existing:
                return name
        return f"{base}_9999"

    def _save_preview_labels(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        for preview_name, saved_name in (
            ("SAM3 preview labels", "SAM3 saved labels"),
            ("SAM3 propagated preview labels", "SAM3 saved propagated labels"),
        ):
            try:
                preview = self.viewer.layers[preview_name]
            except (KeyError, ValueError):
                continue
            data = preview.data.copy()
            try:
                existing = self.viewer.layers[saved_name]
            except (KeyError, ValueError):
                self.viewer.add_labels(data, name=saved_name)
            else:
                existing.data = data
            self._log(f"Saved labels to layer: {saved_name}")
            return
        saved = 0
        for preview in list(self.viewer.layers):
            preview_name = getattr(preview, "name", "")
            if not preview_name.startswith("SAM3 preview labels ["):
                continue
            suffix = preview_name.removeprefix("SAM3 preview labels ").strip()
            saved_name = f"SAM3 saved labels {suffix}"
            data = preview.data.copy()
            try:
                existing = self.viewer.layers[saved_name]
            except (KeyError, ValueError):
                self.viewer.add_labels(data, name=saved_name)
            else:
                existing.data = data
            saved += 1
        if saved:
            self._log(f"Saved {saved} batch label layer(s).")
            return
        self._log("No SAM3 preview labels found to save.")

    def _clear_results_table(self) -> None:
        if hasattr(self, "results_table"):
            self.results_table.setRowCount(0)
        if self.shared_context is not None:
            state = self.shared_context.result_visibility.on_results_cleared()
            self.shared_context.result_state = state

    def _refresh_relabel_layers(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        self._set_combo_items(self.relabel_layer_combo, self._layer_names({"labels"}), include_none=False)
        count = self.relabel_layer_combo.count()
        self._log(f"Relabel layer selector refreshed. Labels layers: {count}.")

    def _merge_label_values(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        layer_name = self.relabel_layer_combo.currentData()
        if not layer_name:
            self._log("Select a Labels layer to relabel.")
            return
        try:
            layer = self.viewer.layers[layer_name]
        except (KeyError, ValueError):
            self._log(f"Labels layer not found: {layer_name}")
            return
        if not isinstance(layer, Labels):
            self._log(f"Selected layer is not a Labels layer: {layer_name}")
            return
        try:
            source_values = self._parse_label_values(self.relabel_source_edit.text())
        except ValueError as exc:
            self._log(str(exc))
            return
        target_value = int(self.relabel_target_spin.value())
        if not source_values:
            self._log("Enter one or more source label values, for example: 3,4,5,6.")
            return

        data = np.asarray(layer.data).copy()
        mask = np.isin(data, np.asarray(source_values, dtype=data.dtype))
        changed = int(np.count_nonzero(mask))
        if changed == 0:
            self._log(f"No pixels found with label values: {', '.join(map(str, source_values))}.")
            return
        data[mask] = target_value
        layer.data = data
        layer.refresh()
        self._log(
            f"Relabeled {changed} pixel(s) in '{layer_name}': "
            f"{', '.join(map(str, source_values))} -> {target_value}."
        )
        self._refresh_relabel_layers()

    def _parse_label_values(self, text: str) -> list[int]:
        normalized = text.replace(";", ",").replace(" ", ",")
        values = []
        for token in normalized.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"Label values must be integers. Invalid value: {token!r}") from exc
        return values

    def _copy_results_to_clipboard(self) -> None:
        rows = self._results_table_rows()
        if len(rows) <= 1:
            self._log("No results to copy.")
            return
        text = "\n".join("\t".join(row) for row in rows)
        QApplication.clipboard().setText(text)
        self._log(f"Copied {len(rows) - 1} result row(s) to clipboard.")

    def _export_results_csv(self) -> None:
        rows = self._results_table_rows()
        if len(rows) <= 1:
            self._log("No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export SAM3 results",
            str(Path.home() / "sam3_results.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
        self._log(f"Exported {len(rows) - 1} result row(s) to CSV: {path}")

    def _results_table_rows(self) -> list[list[str]]:
        headers = [
            self.results_table.horizontalHeaderItem(column).text()
            for column in range(self.results_table.columnCount())
        ]
        rows = [headers]
        for row in range(self.results_table.rowCount()):
            values = []
            for column in range(self.results_table.columnCount()):
                item = self.results_table.item(row, column)
                values.append(item.text() if item is not None else "")
            rows.append(values)
        return rows
    
    def _append_result_rows(self, result: Sam3Result) -> None:
        rows = self._result_rows(result)
        if not rows:
            return

        remaining = MAX_RESULTS_TABLE_TOTAL_ROWS - self.results_table.rowCount()
        if remaining <= 0:
            self._log("Results table row limit reached; skipping further UI appends.")
            return

        rows = rows[:remaining]

        table = self.results_table
        table.setUpdatesEnabled(False)
        table.blockSignals(True)
        table.setSortingEnabled(False)

        start_row = table.rowCount()
        table.setRowCount(start_row + len(rows))

        try:
            for row_offset, row_values in enumerate(rows):
                row_index = start_row + row_offset
                for column, value in enumerate(row_values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row_index, column, item)
        finally:
            table.blockSignals(False)
            table.setUpdatesEnabled(True)

    def _result_rows(self, result: Sam3Result) -> list[tuple[str, str, str, str, str, str]]:
        object_ids = self._result_object_ids(result)
        total_objects = len(object_ids)

        if total_objects > MAX_RESULTS_TABLE_ROWS_PER_RESULT:
            object_ids = object_ids[:MAX_RESULTS_TABLE_ROWS_PER_RESULT]
            truncated = True
        else:
            truncated = False

        scores = self._result_scores(result, len(object_ids))
        areas = self._result_areas(result, object_ids)
        layer = str(result.metadata.get("image_layer") or "-")
        prompt = str(
            result.metadata.get("batch_prompt")
            or result.metadata.get("text_prompt_used")
            or "-"
        )
        frame = str(result.frame_index) if result.frame_index is not None else "-"

        rows = []
        for index, object_id in enumerate(object_ids):
            score = scores[index] if index < len(scores) else None
            area = areas[index] if index < len(areas) else None
            rows.append(
                (
                    layer or "-",
                    prompt or "-",
                    frame,
                    str(int(object_id)),
                    "-" if score is None else f"{float(score):.3f}",
                    "-" if area is None else str(int(area)),
                )
            )

        if truncated:
            self._log(
                f"Results table capped at {MAX_RESULTS_TABLE_ROWS_PER_RESULT} rows "
                f"(total objects: {total_objects}). Export logic can be extended later if needed."
            )

        return rows

    def _result_object_ids(self, result: Sam3Result) -> np.ndarray:
        if result.object_ids is not None:
            return np.asarray(result.object_ids).reshape(-1)
        if result.masks is not None and np.asarray(result.masks).ndim >= 3:
            return np.arange(1, np.asarray(result.masks).shape[0] + 1)
        if result.labels is not None:
            ids = np.unique(np.asarray(result.labels))
            return ids[ids != 0]
        return np.asarray([], dtype=np.int64)

    def _result_scores(self, result: Sam3Result, count: int) -> list[float | None]:
        if result.scores is None:
            return [None] * count
        scores = np.asarray(result.scores).reshape(-1)
        return [float(value) for value in scores[:count]]

    def _result_areas(self, result: Sam3Result, object_ids: np.ndarray) -> list[int | None]:
        if result.labels is not None:
            labels = np.asarray(result.labels)
            flat = labels.reshape(-1)
            if flat.size == 0:
                return [None] * len(object_ids)

            max_id = int(flat.max())
            if max_id <= 0:
                return [None] * len(object_ids)

            counts = np.bincount(flat.astype(np.int64), minlength=max_id + 1)
            return [
                int(counts[int(object_id)]) if 0 <= int(object_id) < len(counts) else 0
                for object_id in object_ids
            ]

        if result.masks is not None:
            masks = np.asarray(result.masks)
            if masks.ndim >= 3:
                return [int(np.count_nonzero(mask)) for mask in masks[: len(object_ids)]]

        return [None] * len(object_ids)

    def _should_update_boxes_for_result(self, result: Sam3Result) -> bool:
        if self._current_task() != Sam3Task.REFINE:
            return True
        if result.metadata.get("large_image_mode"):
            return False
        return False

    def _current_task(self) -> Sam3Task:
        return self.task_combo.currentData()

    def _current_image_layer_name(self) -> str:
        return self.image_layer_combo.currentData() or ""

    def _optional_combo_data(self, combo: QComboBox) -> str | None:
        value = combo.currentData()
        return value if value else None

    def _layer_names(self, type_names: set[str]) -> list[str]:
        if self.viewer is None:
            return []
        names = []
        for layer in self.viewer.layers:
            if self._layer_matches(layer, type_names):
                names.append(layer.name)
        return names

    def _layer_matches(self, layer: Any, type_names: set[str]) -> bool:
        if "image" in type_names and isinstance(layer, Image):
            return True
        if "points" in type_names and isinstance(layer, Points):
            return True
        if "shapes" in type_names and isinstance(layer, Shapes):
            return True
        if "labels" in type_names and isinstance(layer, Labels):
            return True

        layer_type = layer.__class__.__name__.lower()
        type_string = str(getattr(layer, "_type_string", "")).lower()
        return layer_type in type_names or type_string in type_names

    def _connect_layer_events(self) -> None:
        if self.viewer is None or self._layer_events_connected:
            return
        events = getattr(self.viewer.layers, "events", None)
        if events is None:
            return
        for event_name in ("inserted", "removed", "reordered"):
            event = getattr(events, event_name, None)
            if event is None:
                continue
            try:
                event.connect(self._on_layers_changed)
            except ValueError:
                pass
        self._layer_events_connected = True

    def _on_layers_changed(self, event: Any = None) -> None:
        self._refresh_layers(silent=True)

    def _set_combo_items(
        self,
        combo: QComboBox,
        names: list[str],
        *,
        include_none: bool = True,
    ) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if include_none:
            combo.addItem(NONE_LABEL, None)
        for name in names:
            combo.addItem(name, name)
        if current:
            index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _image_selection_for_layer(self, image_layer: Any):
        if self.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        channel_axis = self.channel_axis_spin.value()
        base_data = self._base_layer_data(image_layer.data)
        return infer_image_selection(
            layer_name=image_layer.name,
            data_shape=tuple(base_data.shape),
            dims_current_step=tuple(self.viewer.dims.current_step),
            channel_axis=None if channel_axis < 0 else channel_axis,
        )

    def _result_summary(self, result: Sam3Result) -> str:
        count = self._result_count(result)
        frame = f" frame={result.frame_index}" if result.frame_index is not None else ""
        session = f" session={result.session_id}" if result.session_id else ""
        return f"SAM3 result:{frame}{session} objects={count}"

    def _result_count(self, result: Sam3Result) -> int:
        if result.object_ids is not None:
            return len(result.object_ids)
        if result.masks is not None and result.masks.ndim >= 3:
            return result.masks.shape[0]
        return 0

    def _log_text_result_guidance(self, result: Sam3Result) -> None:
        if result.task != Sam3Task.TEXT:
            return
        prompt = result.metadata.get("text_prompt_used")
        threshold = result.metadata.get("text_threshold_used")
        if prompt and threshold is not None:
            self._log(f"Text prompt used by SAM3: '{prompt}' at threshold {float(threshold):.2f}.")
        if self._result_count(result) == 0:
            self._log(
                "Text prompt returned zero objects. Try a short noun phrase, lower "
                "Detection threshold, or use a box/exemplar prompt for microscopy-specific structures."
            )

    def _log_large_image_result_guidance(self, result: Sam3Result) -> None:
        roi = result.metadata.get("large_image_roi")
        if not roi:
            return
        y0, x0, y1, x1 = roi
        self._log(f"Active ROI bounds: y={y0}:{y1}, x={x0}:{x1}.")

    def _log(self, message: str) -> None:
        if self.shared_context is not None:
            self.shared_context.activity_log.append(message)
        self.status_box.append(message)


def _is_cuda_kernel_image_error(error: Any) -> bool:
    text = str(error).lower()
    return (
        "no kernel image is available for execution on the device" in text
        or "cudaerrornokernelimagefordevice" in text
    )
