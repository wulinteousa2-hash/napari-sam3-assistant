from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

import numpy as np
from napari import current_viewer
from napari.layers import Image, Labels, Points, Shapes
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..adapters import Sam3Adapter, Sam3AdapterConfig, cuda_compatibility_issue
from ..core.coordinates import extract_2d_image, infer_image_selection
from ..core.models import PromptBundle, Sam3Result, Sam3Session, Sam3Task
from ..providers.sam3_repo_provider import Sam3RepoProvider
from ..services.checkpoint_service import CheckpointService
from ..services.layer_writer import LayerWriter
from ..services.prompt_collector import PromptCollector
from ..services.prompt_state_service import PromptStateService
from .collapsible_panel import CollapsiblePanel
from .live_point_refinement import LivePointRefinementController

NONE_LABEL = "(none)"
SETTINGS_ORG = "napari-sam3-assistant"
SETTINGS_APP = "sam3-assistant"
PROMPT_POINTS = "points"
PROMPT_BOX = "box"
PROMPT_LABELS = "labels"
PROMPT_TEXT = "text"

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


class MainWidget(QWidget):
    def __init__(
        self,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer or viewer or current_viewer()

        self.provider = Sam3RepoProvider()
        self.checkpoint_service = CheckpointService()
        self.prompt_state_service = PromptStateService()
        self.prompt_collector = PromptCollector()
        self.layer_writer = LayerWriter(self.viewer) if self.viewer is not None else None

        self.adapter: Sam3Adapter | None = None
        self.video_session: Sam3Session | None = None
        self._worker: Any | None = None
        self._worker_failed = False
        self._layer_events_connected = False
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)

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
        )
        self._sync_live_refinement_layer()
        self._set_live_refinement_status("Activity: idle")


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
        layers_group = self._build_layers_group()
        prompt_group = self._build_prompt_group()
        actions_group = self._build_actions_group()
        results_group = self._build_results_group()

        left_column.addWidget(CollapsiblePanel("Step 1. Model Setup", backend_group, collapsed=False))
        left_column.addWidget(CollapsiblePanel("Step 2. Task", task_group, collapsed=False))
        left_column.addWidget(CollapsiblePanel("Step 3. Layers", layers_group, collapsed=False))

        right_column.addWidget(CollapsiblePanel("Step 4. Prompt Tools", prompt_group, collapsed=False))
        right_column.addWidget(CollapsiblePanel("Step 5. Run", actions_group, collapsed=False))
        right_column.addWidget(CollapsiblePanel("Step 6. Results", results_group, collapsed=False))

        self.status_box = QTextEdit()
        self.status_box.setObjectName("statusBox")
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(150)



        status_container = QWidget()
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addWidget(self.status_box)
        status_container.setLayout(status_layout)

        right_column.addWidget(CollapsiblePanel("Step 7. Status", status_container, collapsed=True))

        left_column.addStretch(1)
        right_column.addStretch(1)
        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)
        layout.addLayout(columns)

        layout.addStretch(1)
        self.setLayout(layout)
        self._log("SAM3 Assistant widget initialized.")




    def _set_live_refinement_status(self, text: str) -> None:
        if hasattr(self, "live_refinement_status_label"):
            self.live_refinement_status_label.setText(text)

    def _live_refinement_enabled(self) -> bool:
        return (
            self._current_task() == Sam3Task.REFINE
            and self.prompt_tool_combo.currentData() == PROMPT_POINTS
            and self._worker is None
        )

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
        self._log(f"Flipped {len(indices)} point(s); rerunning refinement.")
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
        self._set_live_refinement_status("Activity: live refinement running...")
        self._run_current_task()
    
    def _step_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        font = group.font()
        font.setPointSize(max(font.pointSize() + 1, 11))
        font.setBold(False)
        group.setFont(font)
        return group

    def _build_backend_group(self) -> QGroupBox:
        group = self._step_group("Model path, type, and device")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Select local SAM3 model directory...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model_dir)
        row.addWidget(self.model_dir_edit)
        row.addWidget(browse_btn)

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
        self.device_combo.addItem("Auto", None)
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        layout.addLayout(row)
        layout.addWidget(QLabel("Model type"))
        layout.addWidget(self.model_type_combo)
        layout.addLayout(btn_row)
        layout.addWidget(QLabel("Device"))
        layout.addWidget(self.device_combo)
        layout.addWidget(self.lazy_load_check)
        group.setLayout(layout)
        return group

    def _build_task_group(self) -> QGroupBox:
        group = self._step_group("Choose the SAM3 task")
        layout = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItem("2D segmentation", Sam3Task.SEGMENT_2D)
        self.task_combo.addItem("3D/video propagation", Sam3Task.SEGMENT_3D)
        self.task_combo.addItem("Exemplar segmentation", Sam3Task.EXEMPLAR)
        self.task_combo.addItem("Text segmentation", Sam3Task.TEXT)
        self.task_combo.addItem("Refinement (live point correction)", Sam3Task.REFINE)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)

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

        layout.addRow("Task", self.task_combo)
        layout.addRow("Channel axis (-1 auto/no channel)", self.channel_axis_spin)
        hint = QLabel("Leave -1 unless your image has an explicit channel dimension.")
        hint.setToolTip(channel_axis_tip)
        layout.addRow("", hint)
        layout.addRow("Detection threshold", self.confidence_threshold_spin)
        layout.addRow("3D direction", self.propagation_direction_combo)
        group.setLayout(layout)
        return group

    def _build_layers_group(self) -> QGroupBox:
        group = self._step_group("Review detected napari layers")
        layout = QFormLayout()

        self.image_layer_combo = QComboBox()
        self.points_layer_combo = QComboBox()
        self.points_layer_combo.currentIndexChanged.connect(self._on_points_layer_changed)
        self.shapes_layer_combo = QComboBox()
        self.labels_layer_combo = QComboBox()
        self.batch_all_images_check = QCheckBox("Batch all image layers")
        self.batch_all_images_check.setToolTip(
            "Run the current prompt setup on every napari Image layer. "
            "Each image gets its own SAM3 preview output layers."
        )

        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)

        layout.addRow("Image", self.image_layer_combo)
        layout.addRow("", self.batch_all_images_check)
        layout.addRow("Points", self.points_layer_combo)
        layout.addRow("Shapes", self.shapes_layer_combo)
        layout.addRow("Labels", self.labels_layer_combo)
        layout.addRow(refresh_btn)
        group.setLayout(layout)
        return group

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
            "Add the first point to start live refinement; first run may load the model.\n"
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
        self.multi_text_prompt_edit.setPlaceholderText("Optional batch prompts: one prompt per line, e.g.\ncat\ndog")
        self.multi_text_prompt_edit.setToolTip(
            "Optional multi-text mode. Enter one concept per line. With Batch all image "
            "layers enabled, every prompt is run on every image layer."
        )
        self.multi_text_prompt_edit.setMaximumHeight(78)

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
        layout.addRow(clear_btn)
        group.setLayout(layout)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = self._step_group("Run preview or propagation")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.run_btn = QPushButton("Run Preview")
        self.run_btn.setObjectName("runButton")
        self.run_btn.clicked.connect(self._run_current_task)

        self.propagate_btn = QPushButton("Propagate Stack/Video")
        self.propagate_btn.setObjectName("runButton")
        self.propagate_btn.clicked.connect(self._propagate_existing_session)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancelButton")
        cancel_btn.clicked.connect(self._cancel_worker)

        save_btn = QPushButton("Save Result as Labels")
        save_btn.setObjectName("saveButton")
        save_btn.clicked.connect(self._save_preview_labels)

        clear_preview_btn = QPushButton("Clear Preview")
        clear_preview_btn.setObjectName("clearButton")
        clear_preview_btn.clicked.connect(self._clear_preview_layers)

        row.addWidget(self.run_btn)
        row.addWidget(self.propagate_btn)
        row.addWidget(clear_preview_btn)
        row.addWidget(save_btn)
        row.addWidget(cancel_btn)

        self.live_refinement_status_label = QLabel("Activity: idle")
        self.live_refinement_status_label.setObjectName("activityIndicator")
        self.live_refinement_status_label.setToolTip(
            "Shows live refinement and SAM3 worker activity so long model runs are visible."
        )

        layout.addLayout(row)
        layout.addWidget(self.live_refinement_status_label)
        group.setLayout(layout)
        return group

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

        relabel_form = QFormLayout()
        self.relabel_layer_combo = QComboBox()
        self.relabel_source_edit = QLineEdit()
        self.relabel_source_edit.setPlaceholderText("Example: 3,4,5,6")
        self.relabel_target_spin = QSpinBox()
        self.relabel_target_spin.setRange(0, 2_147_483_647)
        self.relabel_target_spin.setValue(3)

        refresh_relabel_btn = QPushButton("Refresh Label Layers")
        refresh_relabel_btn.clicked.connect(self._refresh_relabel_layers)
        apply_relabel_btn = QPushButton("Merge Label Values")
        apply_relabel_btn.clicked.connect(self._merge_label_values)

        relabel_buttons = QHBoxLayout()
        relabel_buttons.addWidget(refresh_relabel_btn)
        relabel_buttons.addWidget(apply_relabel_btn)

        relabel_form.addRow("Relabel layer", self.relabel_layer_combo)
        relabel_form.addRow("Values to replace", self.relabel_source_edit)
        relabel_form.addRow("New value", self.relabel_target_spin)

        layout.addWidget(self.results_table)
        layout.addLayout(action_row)
        layout.addLayout(relabel_form)
        layout.addLayout(relabel_buttons)
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
            adapter.load_image(enable_instance_interactivity=self._current_task() == Sam3Task.REFINE)
            return "SAM3 image model loaded."

        self._start_worker(load_model(), on_returned=self._on_image_initialized)

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

        self._start_worker(load_model(), on_returned=self._on_video_initialized)

    def _unload_adapter(self) -> None:
        self._cancel_worker()
        if self.adapter is not None:
            self.adapter.unload()
        self.adapter = None
        self.video_session = None
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
        self._log("Prompt state cleared.")

    def _refresh_layers(self) -> None:
        if self.viewer is None:
            self.viewer = current_viewer()
            if self.viewer is not None and self.layer_writer is None:
                self.layer_writer = LayerWriter(self.viewer)
                self._connect_layer_events()

        self._set_combo_items(self.image_layer_combo, self._layer_names({"image"}), include_none=False)
        self._set_combo_items(self.points_layer_combo, self._layer_names({"points"}))
        self._set_combo_items(self.shapes_layer_combo, self._layer_names({"shapes"}))
        self._set_combo_items(self.labels_layer_combo, self._layer_names({"labels"}))
        if hasattr(self, "relabel_layer_combo"):
            self._set_combo_items(self.relabel_layer_combo, self._layer_names({"labels"}), include_none=False)
        layer_count = 0 if self.viewer is None else len(self.viewer.layers)
        self._log(f"Layer selectors refreshed. Viewer layers: {layer_count}.")
        if hasattr(self, "live_point_refinement"):
            self._sync_live_refinement_layer()

    def _on_task_changed(self) -> None:
        task = self._current_task()
        is_video = task == Sam3Task.SEGMENT_3D
        self.propagation_direction_combo.setEnabled(is_video)
        self.propagate_btn.setEnabled(is_video and self._worker is None)
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
        if self._current_model_type() == "sam3.1" and self._current_task() != Sam3Task.SEGMENT_3D:
            self._log(
                "SAM3.1 video multiplex supports 3D/video propagation in this plugin. "
                "Choose task '3D/video propagation', or switch Model type to "
                "'SAM3.0 2D/3D/video' for 2D image tasks."
            )
            return

        if self.batch_all_images_check.isChecked() or self._multi_text_prompts():
            self._run_batch_current_task()
            return

        try:
            bundle = self._collect_bundle()
        except Exception as exc:
            self._log(f"Cannot collect prompts: {exc}")
            return

        if not bundle.has_prompt():
            self._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            return

        self._clear_results_table()
        if bundle.task == Sam3Task.SEGMENT_3D:
            self._run_video_task(bundle)
        else:
            self._run_image_task(bundle)

    def _run_batch_current_task(self) -> None:
        if self._current_task() == Sam3Task.SEGMENT_3D:
            self._log("Batch all image layers is for 2D image tasks. Use one stack for 3D/video propagation.")
            return
        if self._current_task() == Sam3Task.REFINE:
            self._log("Batch mode is disabled for live refinement. Select one image for point refinement.")
            return
        if self._multi_text_prompts() and self._current_task() != Sam3Task.TEXT:
            self._log("Multi-text batch prompts require task 'Text segmentation'.")
            return
        try:
            bundles = self._collect_batch_bundles()
        except Exception as exc:
            self._log(f"Cannot collect batch prompts: {exc}")
            return
        if not bundles:
            self._log("No image layers found for batch segmentation.")
            return
        if not any(bundle.has_prompt() for bundle in bundles):
            self._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
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

        @thread_worker
        def run_batch():
            for bundle in bundles:
                image_layer = self.viewer.layers[bundle.image.layer_name]
                result = adapter.run_image(image_layer.data, bundle)
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

    def _run_image_task(self, bundle: PromptBundle) -> None:
        if self.viewer is None or self.layer_writer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        image_layer = self.viewer.layers[bundle.image.layer_name]
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot run image task: {exc}")
            return

        @thread_worker
        def run_image() -> Sam3Result:
            result = adapter.run_image(image_layer.data, bundle)
            result.metadata["image_layer"] = bundle.image.layer_name
            return result

        worker = run_image()
        worker.returned.connect(self._write_image_result)
        self._start_worker(worker)
        self._log(f"Running {bundle.task.value} on image layer '{bundle.image.layer_name}'.")

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

        @thread_worker
        def run_video():
            session = adapter.start_video_session(image_layer.data, bundle)
            prompt_result = adapter.add_video_prompt(bundle, session)
            prompt_result.metadata["image_layer"] = bundle.image.layer_name
            yield prompt_result
            for result in adapter.propagate_video(bundle, session, direction=direction):
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = run_video()
        worker.yielded.connect(self._write_video_result)
        worker.returned.connect(self._set_video_session)
        self._start_worker(worker)
        self._log(f"Started video propagation from frame {bundle.image.frame_index or 0}.")

    def _propagate_existing_session(self) -> None:
        if self.video_session is None:
            self._log("No active SAM3 video session. Run a 3D/video task first.")
            return
        try:
            bundle = self._collect_bundle()
        except Exception as exc:
            self._log(f"Cannot collect prompts: {exc}")
            return
        try:
            adapter = self._ensure_adapter()
        except Exception as exc:
            self._log(f"Cannot propagate session: {exc}")
            return
        session = self.video_session
        direction = self.propagation_direction_combo.currentText()

        @thread_worker
        def propagate():
            for result in adapter.propagate_video(bundle, session, direction=direction):
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
            self._log("SAM3 returned no result.")
            self._log(self._result_summary(result))
            self._log_text_result_guidance(result)
            return
        self.layer_writer.write_result(
            result,
            labels_name="SAM3 preview labels",
            mask_name="SAM3 preview masks",
            boxes_name="SAM3 preview boxes",
        )
        if self._current_task() == Sam3Task.REFINE:
            self._activate_points_layer_for_live_refinement()
        self._append_result_rows(result)
        self._log(self._result_summary(result))
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
            target = f"{image_layer_name} / {prompt}" if prompt else image_layer_name
            self._log(f"SAM3 returned no result for '{target}'.")
            self._log(self._result_summary(result))
            self._log_text_result_guidance(result)
            return
        self.layer_writer.write_result(
            result,
            labels_name=f"SAM3 preview labels [{suffix}]",
            mask_name=f"SAM3 preview masks [{suffix}]",
            boxes_name=f"SAM3 preview boxes [{suffix}]",
        )
        self._append_result_rows(result)
        target = f"{image_layer_name} / {prompt}" if prompt else image_layer_name
        self._log(f"{target}: {self._result_summary(result)}")
        self._log_text_result_guidance(result)

    def _write_video_result(self, result: Sam3Result) -> None:
        if self.layer_writer is None or self.viewer is None:
            return
        image_layer_name = result.metadata.get("image_layer", self._current_image_layer_name())
        image_layer = self.viewer.layers[image_layer_name]
        frame_axis = self._current_frame_axis(image_layer.data)
        output_shape = self._video_output_shape(image_layer.data.shape, frame_axis)
        self.layer_writer.write_video_frame_result(
            result,
            output_shape,
            labels_name="SAM3 propagated preview labels",
        )
        self._append_result_rows(result)
        self._log(self._result_summary(result))

    def _set_video_session(self, session: Sam3Session) -> None:
        self.video_session = session
        self._log(f"Video session ready: {session.session_id}")

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
        )

    def _safe_layer_suffix(self, name: str) -> str:
        suffix = "".join(char if char.isalnum() or char in ("-", "_", " ") else "_" for char in name)
        suffix = " ".join(suffix.split())
        return suffix or "image"

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
        return self.adapter

    def _adapter_config(self) -> Sam3AdapterConfig:
        model_dir = self.model_dir_edit.text().strip()
        model_type = self._current_model_type()
        checkpoint = self._checkpoint_path_from_model_dir(model_dir, model_type) if model_dir else None
        if checkpoint is None:
            expected = ", ".join(self._expected_weight_names(model_type))
            raise RuntimeError(
                f"Select a {self.model_type_combo.currentText()} directory containing: {expected}."
            )
        device = self.device_combo.currentData()
        cuda_issue = cuda_compatibility_issue()
        if device == "cuda" and cuda_issue:
            self._log(
                "CUDA selected despite PyTorch architecture warning. "
                f"Continuing on GPU for testing: {cuda_issue}"
            )
        if device is None and cuda_issue:
            device = "cpu"
            self._log(f"Auto device selected CPU because CUDA is unavailable for this PyTorch build: {cuda_issue}")
        return Sam3AdapterConfig(
            checkpoint_path=checkpoint,
            device=device,
            confidence_threshold=float(self.confidence_threshold_spin.value()),
            load_from_hf=False,
        )

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
        if model_dir:
            self.model_dir_edit.setText(model_dir)
        model_type = self.settings.value("model_type", "sam3", type=str)
        index = self.model_type_combo.findData(model_type)
        if index >= 0:
            self.model_type_combo.setCurrentIndex(index)
        device = self.settings.value("device", "", type=str)
        if device:
            index = self.device_combo.findData(device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        threshold = self.settings.value("confidence_threshold", 0.35, type=float)
        self.confidence_threshold_spin.setValue(float(threshold))

    def _save_settings(self) -> None:
        self.settings.setValue("model_dir", self.model_dir_edit.text().strip())
        self.settings.setValue("model_type", self._current_model_type())
        self.settings.setValue("confidence_threshold", float(self.confidence_threshold_spin.value()))
        device = self.device_combo.currentData()
        self.settings.setValue("device", device or "")

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
                "Created/selected SAM3 points layer. Add the first point to start live "
                "refinement; first run may take longer while the model loads."
            )
        elif tool == PROMPT_BOX:
            layer = self._ensure_shapes_prompt_layer()
            layer_name = layer.name
            self.viewer.layers.selection.active = layer
            self._set_layer_mode(layer, "add_rectangle")
            if task == Sam3Task.EXEMPLAR:
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
        if layer_name is not None:
            if tool == PROMPT_POINTS:
                self._select_combo_data(self.points_layer_combo, layer_name)
                self._set_current_point_polarity()
                self._sync_live_refinement_layer()
                self.viewer.layers.selection.active = self.viewer.layers[layer_name]
                self._set_layer_mode(self.viewer.layers[layer_name], "add")
                self._set_live_refinement_status("Activity: live refinement armed. Add a point to start.")
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
        channel_axis = self.channel_axis_spin.value()
        selection = infer_image_selection(
            layer_name=image_layer.name,
            data_shape=tuple(image_layer.data.shape),
            dims_current_step=tuple(self.viewer.dims.current_step),
            channel_axis=None if channel_axis < 0 else channel_axis,
        )
        frame = extract_2d_image(np.asarray(image_layer.data), selection)
        labels = self.viewer.add_labels(
            np.zeros(frame.shape[-2:], dtype=np.uint32),
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

        data = np.zeros(image_layer.data.shape[-2:], dtype=np.uint8)
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
                "Refinement: create a points layer and add a point to start live preview. "
                "Use T for next point mode; Shift+T flips selected/latest point and reruns."
            )
        if task == Sam3Task.SEGMENT_3D:
            return "3D/video: select the frame/slice, add prompts there, then run preview or propagate."
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

    def _start_worker(self, worker: Any, on_returned: Any | None = None) -> None:
        self._cancel_worker()
        self._worker = worker
        self._worker_failed = False
        if on_returned is not None:
            worker.returned.connect(on_returned)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
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

    def _on_worker_error(self, error: Any) -> None:
        self._worker_failed = True
        self._log(f"SAM3 task failed: {error}")
        if _is_cuda_kernel_image_error(error):
            self._log(
                "CUDA kernel compatibility failure detected. The selected GPU is visible, "
                "but at least one PyTorch, torchvision, or SAM3 CUDA kernel was not built "
                "for this device architecture. Select CPU for this workflow, or install "
                "a build that supports the GPU."
            )
        if self._current_task() == Sam3Task.REFINE and self.prompt_tool_combo.currentData() == PROMPT_POINTS:
            self._set_live_refinement_status("Activity: failed")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_running(False)
        if self._worker_failed:
            self._log("SAM3 task stopped after failure.")
        else:
            self._log("SAM3 task finished.")

        if self._current_task() == Sam3Task.REFINE and self.prompt_tool_combo.currentData() == PROMPT_POINTS:
            self._set_live_refinement_status("Activity: live refinement ready")
        elif self._worker_failed:
            self._set_live_refinement_status("Activity: failed")
        else:
            self._set_live_refinement_status("Activity: idle")

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.propagate_btn.setEnabled(not running and self._current_task() == Sam3Task.SEGMENT_3D)
        self.setCursor(Qt.BusyCursor if running else Qt.ArrowCursor)

    def _set_activity_running_message(self) -> None:
        task = self._current_task()
        if task == Sam3Task.REFINE:
            self._set_live_refinement_status("Activity: live refinement running...")
        elif task == Sam3Task.SEGMENT_3D:
            self._set_live_refinement_status("Activity: 3D/video propagation running...")
        else:
            self._set_live_refinement_status("Activity: SAM3 preview running...")

    def _clear_preview_layers(self) -> None:
        if self.viewer is None:
            self._log("No napari viewer was provided to the widget.")
            return
        preview_names = (
            "SAM3 preview labels",
            "SAM3 preview masks",
            "SAM3 preview boxes",
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
            self._log(f"Cleared {removed} SAM3 preview layer(s). Prompts and saved labels were kept.")
        else:
            self._log("No SAM3 preview layers found to clear.")

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
        for row in rows:
            row_index = self.results_table.rowCount()
            self.results_table.insertRow(row_index)
            for column, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row_index, column, item)

    def _result_rows(self, result: Sam3Result) -> list[tuple[str, str, str, str, str, str]]:
        object_ids = self._result_object_ids(result)
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
            return [int(np.count_nonzero(labels == int(object_id))) for object_id in object_ids]
        if result.masks is not None:
            masks = np.asarray(result.masks)
            if masks.ndim >= 3:
                return [int(np.count_nonzero(mask)) for mask in masks[: len(object_ids)]]
        return [None] * len(object_ids)

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
        for event_name in ("inserted", "removed", "changed", "reordered"):
            event = getattr(events, event_name, None)
            if event is None:
                continue
            try:
                event.connect(self._on_layers_changed)
            except ValueError:
                pass
        self._layer_events_connected = True

    def _on_layers_changed(self, event: Any = None) -> None:
        self._refresh_layers()

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

    def _current_frame_axis(self, data: Any) -> int | None:
        ndim = len(data.shape)
        return ndim - 3 if ndim >= 3 else None

    def _video_output_shape(
        self,
        image_shape: tuple[int, ...],
        frame_axis: int | None,
    ) -> tuple[int, int, int]:
        if frame_axis is None:
            return (1, image_shape[-2], image_shape[-1])
        return (image_shape[frame_axis], image_shape[-2], image_shape[-1])

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

    def _log(self, message: str) -> None:
        self.status_box.append(message)


def _is_cuda_kernel_image_error(error: Any) -> bool:
    text = str(error).lower()
    return (
        "no kernel image is available for execution on the device" in text
        or "cudaerrornokernelimagefordevice" in text
    )
