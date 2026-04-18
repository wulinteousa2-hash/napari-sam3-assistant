from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from napari import current_viewer
from napari.layers import Image, Labels, Points, Shapes
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtCore import QSettings, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..adapters import Sam3Adapter, Sam3AdapterConfig, cuda_compatibility_issue
from ..core.models import PromptBundle, Sam3Result, Sam3Session, Sam3Task
from ..providers.sam3_repo_provider import Sam3RepoProvider
from ..services.checkpoint_service import CheckpointService
from ..services.layer_writer import LayerWriter
from ..services.prompt_collector import PromptCollector
from ..services.prompt_state_service import PromptStateService


NONE_LABEL = "(none)"
SETTINGS_ORG = "napari-sam3-assistant"
SETTINGS_APP = "sam3-assistant"
PROMPT_POINTS = "points"
PROMPT_BOX = "box"
PROMPT_LABELS = "labels"
PROMPT_TEXT = "text"

SAM3_WIDGET_STYLE = """
MainWidget {
    background: #111827;
    color: #e5e7eb;
}
QGroupBox {
    background: #172033;
    border: 1px solid #314158;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 2px 8px;
    color: #67e8f9;
    background: #111827;
    border-radius: 6px;
}
QLabel {
    color: #d1d5db;
}
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background: #0b1220;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 7px;
    selection-background-color: #0ea5e9;
}
QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #38bdf8;
}
QPushButton {
    background: #26364f;
    color: #f8fafc;
    border: 1px solid #3b516f;
    border-radius: 7px;
    padding: 6px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #314563;
    border-color: #60a5fa;
}
QPushButton:pressed {
    background: #1d4ed8;
}
QPushButton:disabled {
    background: #1f2937;
    color: #6b7280;
    border-color: #374151;
}
QPushButton#runButton {
    background: #0369a1;
    border-color: #38bdf8;
}
QPushButton#runButton:hover {
    background: #0284c7;
}
QPushButton#saveButton {
    background: #166534;
    border-color: #22c55e;
}
QPushButton#saveButton:hover {
    background: #15803d;
}
QPushButton#clearButton {
    background: #854d0e;
    border-color: #f59e0b;
}
QPushButton#clearButton:hover {
    background: #a16207;
}
QPushButton#cancelButton {
    background: #7f1d1d;
    border-color: #ef4444;
}
QPushButton#cancelButton:hover {
    background: #991b1b;
}
QCheckBox {
    color: #d1d5db;
    spacing: 8px;
}
QTextEdit#statusBox {
    background: #07111f;
    color: #bae6fd;
    border: 1px solid #164e63;
    border-radius: 8px;
    font-family: "DejaVu Sans Mono", "Menlo", monospace;
}
QLabel#statusLabel {
    color: #93c5fd;
    font-weight: 700;
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
        self._on_task_changed()

    def _build_ui(self) -> None:
        self.setStyleSheet(SAM3_WIDGET_STYLE)
        layout = QVBoxLayout()

        layout.addWidget(self._build_backend_group())
        layout.addWidget(self._build_task_group())
        layout.addWidget(self._build_layers_group())
        layout.addWidget(self._build_prompt_group())
        layout.addWidget(self._build_actions_group())

        self.status_box = QTextEdit()
        self.status_box.setObjectName("statusBox")
        self.status_box.setReadOnly(True)
        self.status_box.setMinimumHeight(150)
        status_label = QLabel("Status")
        status_label.setObjectName("statusLabel")
        layout.addWidget(status_label)
        layout.addWidget(self.status_box)

        self.setLayout(layout)
        self._log("SAM3 Assistant widget initialized.")

    def _step_group(self, title: str) -> QGroupBox:
        group = QGroupBox(title)
        font = group.font()
        font.setPointSize(max(font.pointSize() + 1, 11))
        font.setBold(True)
        group.setFont(font)
        return group

    def _build_backend_group(self) -> QGroupBox:
        group = self._step_group("1. Model Setup")
        layout = QVBoxLayout()

        row = QHBoxLayout()
        self.model_dir_edit = QLineEdit()
        self.model_dir_edit.setPlaceholderText("Select local SAM3 model directory...")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_model_dir)
        row.addWidget(self.model_dir_edit)
        row.addWidget(browse_btn)

        btn_row = QHBoxLayout()
        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self._validate_model_dir)

        load_image_btn = QPushButton("Load 2D Model")
        load_image_btn.clicked.connect(self._load_image_adapter)

        load_video_btn = QPushButton("Load 3D/Video Model")
        load_video_btn.clicked.connect(self._load_video_adapter)

        unload_btn = QPushButton("Unload")
        unload_btn.clicked.connect(self._unload_adapter)
        unload_btn.setObjectName("clearButton")

        btn_row.addWidget(validate_btn)
        btn_row.addWidget(load_image_btn)
        btn_row.addWidget(load_video_btn)
        btn_row.addWidget(unload_btn)

        self.lazy_load_check = QCheckBox("Load model when running")
        self.lazy_load_check.setChecked(True)

        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto", None)
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        layout.addLayout(row)
        layout.addLayout(btn_row)
        layout.addWidget(QLabel("Device"))
        layout.addWidget(self.device_combo)
        layout.addWidget(self.lazy_load_check)
        group.setLayout(layout)
        return group

    def _build_task_group(self) -> QGroupBox:
        group = self._step_group("2. Task")
        layout = QFormLayout()

        self.task_combo = QComboBox()
        self.task_combo.addItem("2D segmentation", Sam3Task.SEGMENT_2D)
        self.task_combo.addItem("3D/video propagation", Sam3Task.SEGMENT_3D)
        self.task_combo.addItem("Exemplar segmentation", Sam3Task.EXEMPLAR)
        self.task_combo.addItem("Text segmentation", Sam3Task.TEXT)
        self.task_combo.addItem("Refinement", Sam3Task.REFINE)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)

        self.channel_axis_spin = QSpinBox()
        self.channel_axis_spin.setRange(-1, 8)
        self.channel_axis_spin.setValue(-1)
        channel_axis_tip = (
            "Which data axis is color/channel. Use -1 for grayscale or RGB/RGBA auto-detect. "
            "Examples: C,H,W -> 0; Z,C,H,W or T,C,H,W -> 1."
        )
        self.channel_axis_spin.setToolTip(channel_axis_tip)

        self.propagation_direction_combo = QComboBox()
        self.propagation_direction_combo.addItems(["both", "forward", "backward"])

        layout.addRow("Task", self.task_combo)
        layout.addRow("Channel axis (-1 auto/no channel)", self.channel_axis_spin)
        hint = QLabel("Leave -1 unless your image has an explicit channel dimension.")
        hint.setToolTip(channel_axis_tip)
        layout.addRow("", hint)
        layout.addRow("3D direction", self.propagation_direction_combo)
        group.setLayout(layout)
        return group

    def _build_layers_group(self) -> QGroupBox:
        group = self._step_group("3. Layers")
        layout = QFormLayout()

        self.image_layer_combo = QComboBox()
        self.points_layer_combo = QComboBox()
        self.shapes_layer_combo = QComboBox()
        self.labels_layer_combo = QComboBox()

        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)

        layout.addRow("Image", self.image_layer_combo)
        layout.addRow("Points", self.points_layer_combo)
        layout.addRow("Shapes", self.shapes_layer_combo)
        layout.addRow("Labels", self.labels_layer_combo)
        layout.addRow(refresh_btn)
        group.setLayout(layout)
        return group

    def _build_prompt_group(self) -> QGroupBox:
        group = self._step_group("4. Prompt Tools")
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

        init_prompt_btn = QPushButton("Create Prompt Layer")
        init_prompt_btn.clicked.connect(self._initialize_prompt_layer)

        apply_polarity_btn = QPushButton("Apply Positive/Negative to Selected Points")
        apply_polarity_btn.clicked.connect(self._apply_polarity_to_selected_points)

        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setPlaceholderText("Optional text prompt...")
        self.text_prompt_edit.editingFinished.connect(self._set_text_prompt)

        clear_btn = QPushButton("Clear Text / Prompt State")
        clear_btn.clicked.connect(self._clear_prompts)

        layout.addRow("Prompt type", self.prompt_tool_combo)
        layout.addRow("Point type", self.point_polarity_combo)
        layout.addRow(init_prompt_btn)
        layout.addRow(apply_polarity_btn)
        layout.addRow("Text", self.text_prompt_edit)
        layout.addRow(clear_btn)
        group.setLayout(layout)
        return group

    def _build_actions_group(self) -> QGroupBox:
        group = self._step_group("5. Run")
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

        layout.addLayout(row)
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
        result = self.checkpoint_service.validate(model_dir)
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
        layer_count = 0 if self.viewer is None else len(self.viewer.layers)
        self._log(f"Layer selectors refreshed. Viewer layers: {layer_count}.")

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

    def _on_prompt_tool_changed(self) -> None:
        tool = self.prompt_tool_combo.currentData()
        is_points = tool == PROMPT_POINTS
        self.point_polarity_combo.setEnabled(is_points)
        self.text_prompt_edit.setEnabled(tool == PROMPT_TEXT or self._current_task() in {Sam3Task.TEXT, Sam3Task.SEGMENT_2D, Sam3Task.SEGMENT_3D})

    def _run_current_task(self) -> None:
        try:
            bundle = self._collect_bundle()
        except Exception as exc:
            self._log(f"Cannot collect prompts: {exc}")
            return

        if not bundle.has_prompt():
            self._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            return

        if bundle.task == Sam3Task.SEGMENT_3D:
            self._run_video_task(bundle)
        else:
            self._run_image_task(bundle)

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
            return adapter.run_image(image_layer.data, bundle)

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
            return
        self.layer_writer.write_result(
            result,
            labels_name="SAM3 preview labels",
            mask_name="SAM3 preview masks",
            boxes_name="SAM3 preview boxes",
        )
        self._log(self._result_summary(result))

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

    def _ensure_adapter(self, *, reset: bool = False) -> Sam3Adapter:
        config = self._adapter_config()
        if reset or self.adapter is None or self.adapter.config != config:
            if self.adapter is not None:
                self.adapter.unload()
            self.adapter = Sam3Adapter(config)
        return self.adapter

    def _adapter_config(self) -> Sam3AdapterConfig:
        model_dir = self.model_dir_edit.text().strip()
        checkpoint = self._checkpoint_path_from_model_dir(model_dir) if model_dir else None
        if checkpoint is None:
            raise RuntimeError(
                "Select a local SAM3 model directory containing sam3.pt, "
                "model.safetensors, or pytorch_model.bin."
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
            load_from_hf=False,
        )

    def _checkpoint_path_from_model_dir(self, model_dir: str) -> Path | None:
        path = Path(model_dir)
        for name in ("sam3.pt", "model.safetensors", "pytorch_model.bin"):
            candidate = path / name
            if candidate.exists():
                return candidate
        return None

    def _restore_settings(self) -> None:
        model_dir = self.settings.value("model_dir", "", type=str)
        if model_dir:
            self.model_dir_edit.setText(model_dir)
        device = self.settings.value("device", "", type=str)
        if device:
            index = self.device_combo.findData(device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def _save_settings(self) -> None:
        self.settings.setValue("model_dir", self.model_dir_edit.text().strip())
        device = self.device_combo.currentData()
        self.settings.setValue("device", device or "")

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
            layer = self._ensure_points_prompt_layer()
            layer_name = layer.name
            self.viewer.layers.selection.active = layer
            self._set_layer_mode(layer, "add")
            self._log(
                "Created/selected SAM3 points layer. Select Positive/Negative, add points, "
                "then click Run Preview."
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
        if self.viewer is None:
            return
        layer_name = self._optional_combo_data(self.points_layer_combo)
        if not layer_name:
            return
        try:
            layer = self.viewer.layers[layer_name]
        except (KeyError, ValueError):
            return
        if not isinstance(layer, Points):
            return
        polarity = self.point_polarity_combo.currentData() or "positive"
        layer.current_properties = {"polarity": np.asarray([polarity], dtype=object)}

    def _apply_polarity_to_selected_points(self) -> None:
        if self.viewer is None:
            return
        layer_name = self._optional_combo_data(self.points_layer_combo)
        if not layer_name:
            self._log("No points layer selected.")
            return
        layer = self.viewer.layers[layer_name]
        if not isinstance(layer, Points):
            self._log("Selected prompt layer is not a Points layer.")
            return
        selected = sorted(getattr(layer, "selected_data", []))
        if not selected:
            self._log("Select one or more points before applying the positive/negative type.")
            return

        polarity = self.point_polarity_combo.currentData() or "positive"
        properties = dict(getattr(layer, "properties", {}) or {})
        values = list(properties.get("polarity", ["positive"] * len(layer.data)))
        if len(values) < len(layer.data):
            values.extend(["positive"] * (len(layer.data) - len(values)))
        for idx in selected:
            values[idx] = polarity
        properties["polarity"] = np.asarray(values, dtype=object)
        layer.properties = properties
        layer.refresh_colors()
        self._log(f"Set {len(selected)} selected point(s) to {polarity}.")

    def _task_guidance(self, task: Sam3Task) -> str:
        if task == Sam3Task.TEXT:
            return "Text segmentation: enter a phrase and click Run Preview. No prompt layer is required."
        if task == Sam3Task.EXEMPLAR:
            return "Exemplar segmentation: create a box prompt layer and draw ROI boxes around examples."
        if task == Sam3Task.REFINE:
            return "Refinement: create a points layer, add positive/negative points, then run preview."
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

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._set_running(False)
        if self._worker_failed:
            self._log("SAM3 task stopped after failure.")
        else:
            self._log("SAM3 task finished.")

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.propagate_btn.setEnabled(not running and self._current_task() == Sam3Task.SEGMENT_3D)
        self.setCursor(Qt.BusyCursor if running else Qt.ArrowCursor)

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
        removed = 0
        for name in preview_names:
            try:
                layer = self.viewer.layers[name]
            except (KeyError, ValueError):
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
        self._log("No SAM3 preview labels found to save.")

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
        count = 0
        if result.object_ids is not None:
            count = len(result.object_ids)
        elif result.masks is not None and result.masks.ndim >= 3:
            count = result.masks.shape[0]
        frame = f" frame={result.frame_index}" if result.frame_index is not None else ""
        session = f" session={result.session_id}" if result.session_id else ""
        return f"SAM3 result:{frame}{session} objects={count}"

    def _log(self, message: str) -> None:
        self.status_box.append(message)


def _is_cuda_kernel_image_error(error: Any) -> bool:
    text = str(error).lower()
    return (
        "no kernel image is available for execution on the device" in text
        or "cudaerrornokernelimagefordevice" in text
    )
