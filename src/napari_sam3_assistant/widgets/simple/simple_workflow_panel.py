from __future__ import annotations

from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.models import Sam3Task
from ..advanced.advanced_mode_panel import PROMPT_BOX, PROMPT_LABELS, PROMPT_POINTS, PROMPT_TEXT
from .simple_mode_controller import SimpleModeController


class SimpleWorkflowPanel(QGroupBox):
    """Task-focused Simple UI that hides unrelated prompt controls."""

    def __init__(self, controller: SimpleModeController, parent: QWidget | None = None) -> None:
        super().__init__("Tasks", parent)
        self.controller = controller
        self._syncing = False

        self.tabs = QTabWidget()

        self.slice_prompt_combo = QComboBox()
        self.slice_prompt_combo.addItem("Points", PROMPT_POINTS)
        self.slice_prompt_combo.addItem("Box", PROMPT_BOX)
        self.slice_prompt_combo.addItem("Labels", PROMPT_LABELS)
        self.slice_prompt_combo.currentIndexChanged.connect(lambda _i: self._configure_current_tab())
        self.tabs.addTab(self._build_slice_tab(), "2D Slice")

        self.exemplar_source_group = QButtonGroup(self)
        self.exemplar_target_radio = QRadioButton("Draw box on target")
        self.exemplar_crop_radio = QRadioButton("Use crop image")
        self.exemplar_target_radio.setChecked(True)
        self.exemplar_source_group.addButton(self.exemplar_target_radio)
        self.exemplar_source_group.addButton(self.exemplar_crop_radio)
        self.exemplar_source_group.setExclusive(True)
        self.exemplar_target_radio.toggled.connect(lambda _checked: self._on_exemplar_source_changed())
        self.exemplar_crop_radio.toggled.connect(lambda _checked: self._on_exemplar_source_changed())
        self.crop_image_combo = QComboBox()
        self.crop_image_combo.currentIndexChanged.connect(
            lambda _i: self.controller.set_crop_image_layer(str(self.crop_image_combo.currentData() or ""))
        )
        self.crop_region_combo = QComboBox()
        self.crop_region_combo.currentIndexChanged.connect(
            lambda _i: self.controller.set_crop_region(self.crop_region_combo.currentData())
        )
        self.exemplar_large_check = QCheckBox("Enable local/tiled inference")
        self.exemplar_large_check.setChecked(True)
        self.exemplar_large_check.toggled.connect(lambda checked: self.controller.set_large_image_enabled(bool(checked)))
        self.exemplar_batch_layers_check = QCheckBox("Batch all image layers")
        self.exemplar_batch_layers_check.setToolTip(
            "Run the exemplar workflow on every Image layer already loaded in napari. "
            "Use this when each layer is a separate target image."
        )
        self.exemplar_batch_layers_check.toggled.connect(
            lambda checked: self.controller.set_batch_all_image_layers_enabled(bool(checked))
        )
        self.roi_size_combo = QComboBox()
        self.roi_size_combo.currentIndexChanged.connect(lambda _i: self.controller.set_roi_size(self.roi_size_combo.currentData()))
        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 50)
        self.tile_overlap_spin.setSuffix("%")
        self.tile_overlap_spin.valueChanged.connect(self.controller.set_tile_overlap)
        self.merge_seams_check = QCheckBox("Merge seam-split objects")
        self.merge_seams_check.toggled.connect(self.controller.set_merge_tile_seams_enabled)
        self.tabs.addTab(self._build_exemplar_tab(), "Exemplar")

        self.text_prompt_edit = QLineEdit()
        self.text_prompt_edit.setObjectName("textPromptInput")
        self.text_prompt_edit.setPlaceholderText("Text prompt")
        self.text_prompt_edit.editingFinished.connect(self._sync_text_prompt)
        self.text_prompt_edit.returnPressed.connect(self._run_from_text)
        self.tabs.addTab(self._build_text_tab(), "Text")

        self.point_polarity_combo = QComboBox()
        self.point_polarity_combo.addItem("Positive", "positive")
        self.point_polarity_combo.addItem("Negative", "negative")
        self.point_polarity_combo.currentIndexChanged.connect(self._on_polarity_changed)
        self.tabs.addTab(self._build_live_points_tab(), "Live Points")

        self.multiplex_prompt_combo = QComboBox()
        self.multiplex_prompt_combo.addItem("Points", PROMPT_POINTS)
        self.multiplex_prompt_combo.addItem("Box", PROMPT_BOX)
        self.multiplex_prompt_combo.currentIndexChanged.connect(lambda _i: self._configure_current_tab())
        self.tabs.addTab(self._build_multiplex_tab(), "3D Multiplex")

        self.tabs.addTab(self._build_cleanup_tab(), "Cleanup")
        self.tabs.tabBar().moveTab(1, 0)
        self.tabs.tabBar().moveTab(3, 1)
        self.tabs.tabBar().moveTab(4, 2)
        self.tabs.tabBar().moveTab(5, 3)
        self.tabs.currentChanged.connect(self._on_task_tab_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def refresh(self) -> None:
        self._syncing = True
        try:
            task = self.controller.current_task()
            tab_index = {
                Sam3Task.EXEMPLAR: 0,
                Sam3Task.REFINE: 1,
                Sam3Task.SEGMENT_3D: 2,
                Sam3Task.SEGMENT_2D: 4,
                Sam3Task.TEXT: 5,
            }.get(task, 0)
            if self.tabs.currentIndex() != tab_index:
                self.tabs.setCurrentIndex(tab_index)

            self.text_prompt_edit.setText(self.controller.current_text_prompt())
            self._sync_combo_items(self.roi_size_combo, self.controller.roi_size_items(), self.controller.current_roi_size())
            self.tile_overlap_spin.setValue(self.controller.current_tile_overlap())
            self.merge_seams_check.setChecked(self.controller.merge_tile_seams_enabled())
            self.exemplar_large_check.setChecked(self.controller.large_image_enabled())
            self.exemplar_batch_layers_check.setChecked(self.controller.batch_all_image_layers_enabled())
            self._sync_combo_items(
                self.crop_image_combo,
                [(name, name) for name in self.controller.crop_image_layer_names()],
                self.controller.current_crop_image_layer(),
            )
            self._sync_combo_items(
                self.crop_region_combo,
                self.controller.crop_region_items(),
                self.controller.current_crop_region(),
            )
            source = self.controller.current_exemplar_source()
            self.exemplar_crop_radio.setChecked(source == "crop")
            self.exemplar_target_radio.setChecked(source != "crop")
            self._set_combo_data(self.point_polarity_combo, self.controller.current_point_polarity())
            self._sync_exemplar_controls()
        finally:
            self._syncing = False

    def sync_to_shared_state(self) -> None:
        self._sync_text_prompt()
        self._configure_current_tab()
        if self.tabs.currentIndex() == 1:
            self.controller.set_point_polarity(self.point_polarity_combo.currentData() or "positive")

    def _build_slice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        form.addRow("Prompt", self.slice_prompt_combo)
        layout.addLayout(form)
        row = QHBoxLayout()
        create_btn = QPushButton("Create Prompt Layer")
        create_btn.clicked.connect(self.controller.create_prompt_layer)
        row.addWidget(create_btn)
        clear_btn = QPushButton("Reset Prompt")
        clear_btn.setObjectName("clearButton")
        clear_btn.clicked.connect(self.controller.clear_prompt_state)
        row.addWidget(clear_btn)
        layout.addLayout(row)
        layout.addWidget(self._hint("Runs on the current Z/T slice only. Choose points, a box, or labels."))
        tab.setLayout(layout)
        return tab

    def _build_exemplar_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        source_row = QHBoxLayout()
        source_row.addWidget(self.exemplar_target_radio)
        source_row.addWidget(self.exemplar_crop_radio)
        layout.addLayout(source_row)
        form = QFormLayout()
        form.addRow("Crop image", self.crop_image_combo)
        form.addRow("Crop region", self.crop_region_combo)
        form.addRow("", self.exemplar_large_check)
        form.addRow("", self.exemplar_batch_layers_check)
        form.addRow("Tile size", self.roi_size_combo)
        form.addRow("Tile overlap", self.tile_overlap_spin)
        form.addRow("", self.merge_seams_check)
        self.crop_image_label = form.labelForField(self.crop_image_combo)
        self.crop_region_label = form.labelForField(self.crop_region_combo)
        layout.addLayout(form)
        row = QHBoxLayout()
        box_btn = QPushButton("Create Box Layer")
        box_btn.clicked.connect(self.controller.create_prompt_layer)
        row.addWidget(box_btn)
        layout.addLayout(row)
        layout.addWidget(self._hint("Draw one exemplar box, or use a crop image, then run from the Run panel."))
        tab.setLayout(layout)
        return tab

    def _build_text_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.text_prompt_edit)
        layout.addWidget(self._hint("Runs text segmentation on the selected 2D image or current stack slice."))
        tab.setLayout(layout)
        return tab

    def _build_live_points_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        form.addRow("Next point", self.point_polarity_combo)
        layout.addLayout(form)
        row = QHBoxLayout()
        point_btn = QPushButton("Create Points Layer")
        point_btn.clicked.connect(self.controller.create_prompt_layer)
        row.addWidget(point_btn)
        reset_btn = QPushButton("Reset Prompt")
        reset_btn.setObjectName("clearButton")
        reset_btn.clicked.connect(self.controller.clear_prompt_state)
        row.addWidget(reset_btn)
        layout.addLayout(row)
        layout.addWidget(self._hint("Click one point to start. Use T for next point mode and Shift+T to flip selected/latest point."))
        tab.setLayout(layout)
        return tab

    def _build_multiplex_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        form = QFormLayout()
        form.addRow("Prompt", self.multiplex_prompt_combo)
        layout.addLayout(form)
        row = QHBoxLayout()
        prompt_btn = QPushButton("Create Prompt Layer")
        prompt_btn.clicked.connect(self.controller.create_prompt_layer)
        row.addWidget(prompt_btn)
        reset_btn = QPushButton("Reset Session")
        reset_btn.setObjectName("clearButton")
        reset_btn.clicked.connect(self.controller.clear_prompt_state)
        row.addWidget(reset_btn)
        layout.addLayout(row)
        layout.addWidget(self._hint("Uses SAM3.1 multiplex. The current viewer frame becomes the prompt frame for propagation."))
        tab.setLayout(layout)
        return tab

    def _build_cleanup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        open_btn = QPushButton("Open Mask Operations")
        open_btn.clicked.connect(self.controller.open_mask_operations)
        layout.addWidget(open_btn)
        layout.addWidget(self._hint("Open task-specific mask cleanup, myelin/axon review, merge, and export tools."))
        tab.setLayout(layout)
        return tab

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        return label

    def _on_task_tab_changed(self, _index: int) -> None:
        self._configure_current_tab()

    def _configure_current_tab(self) -> None:
        if self._syncing:
            return
        index = self.tabs.currentIndex()
        if index == 0:
            self.controller.configure_task(
                Sam3Task.EXEMPLAR,
                prompt_tool=PROMPT_BOX,
                large_image=self.exemplar_large_check.isChecked(),
            )
            self._on_exemplar_source_changed()
        elif index == 1:
            self.controller.configure_task(Sam3Task.REFINE, prompt_tool=PROMPT_POINTS, large_image=False)
        elif index == 2:
            self.controller.configure_task(
                Sam3Task.SEGMENT_3D,
                prompt_tool=self.multiplex_prompt_combo.currentData() or PROMPT_POINTS,
                large_image=False,
                model_type="sam3.1",
            )
        elif index == 4:
            self.controller.configure_task(
                Sam3Task.SEGMENT_2D,
                prompt_tool=self.slice_prompt_combo.currentData() or PROMPT_POINTS,
                large_image=False,
            )
        elif index == 5:
            self.controller.configure_task(Sam3Task.TEXT, prompt_tool=PROMPT_TEXT, large_image=False)

    def _on_exemplar_source_changed(self) -> None:
        if self._syncing:
            return
        self.controller.set_exemplar_source("crop" if self.exemplar_crop_radio.isChecked() else "target")
        self._sync_exemplar_controls()

    def _sync_exemplar_controls(self) -> None:
        use_crop = self.exemplar_crop_radio.isChecked()
        self.crop_image_combo.setEnabled(use_crop)
        self.crop_region_combo.setEnabled(use_crop)
        self.crop_image_combo.setVisible(use_crop)
        self.crop_region_combo.setVisible(use_crop)
        for label in (getattr(self, "crop_image_label", None), getattr(self, "crop_region_label", None)):
            if label is not None:
                label.setVisible(use_crop)
        large = self.exemplar_large_check.isChecked()
        self.exemplar_batch_layers_check.setEnabled(large)
        self.roi_size_combo.setEnabled(large)
        self.tile_overlap_spin.setEnabled(large)
        self.merge_seams_check.setEnabled(large)

    def _run_tiled_exemplar(self) -> None:
        self.sync_to_shared_state()
        self.controller.run_tiled_exemplar_scan()

    def _sync_text_prompt(self) -> None:
        self.controller.set_text_prompt(self.text_prompt_edit.text())

    def _run_from_text(self) -> None:
        self.sync_to_shared_state()
        self.controller.run_current_task()

    def _on_polarity_changed(self) -> None:
        self.controller.set_point_polarity(self.point_polarity_combo.currentData() or "positive")

    def _sync_combo_items(self, combo: QComboBox, items: list[tuple[str, object]], current: object) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            for label, value in items:
                combo.addItem(label, value)
            self._set_combo_data(combo, current)
        finally:
            combo.blockSignals(False)

    def _set_combo_data(self, combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
