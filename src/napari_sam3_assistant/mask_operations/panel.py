from __future__ import annotations

from typing import Callable

from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .class_merge_tab import ClassMergeTab
from .final_merge_export_tab import FinalMergeExportTab
from .mask_cleanup_tab import MaskCleanupTab


class MaskOperationsPanel(QWidget):
    def __init__(self, viewer=None, log_callback: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.viewer = viewer
        self._log = log_callback or (lambda message: None)
        self.tabs = QTabWidget()

        # Action-first workflow:
        # 1) clean/relabel objects inside one Labels layer,
        # 2) merge multiple Labels layers when SAM3 produced separate layers,
        # 3) export final masks.
        self.mask_cleanup_tab = MaskCleanupTab(viewer, self._log, self.refresh_all)
        self.class_merge_tab = ClassMergeTab(viewer, self._log, self.refresh_all)
        self.final_merge_export_tab = FinalMergeExportTab(viewer, self._log, self.refresh_all)

        self.tabs.addTab(self.mask_cleanup_tab, "Mask Cleanup / Multiclass")
        self.tabs.addTab(self.class_merge_tab, "Merge Layers")
        self.tabs.addTab(self.final_merge_export_tab, "Final Merge / Export")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def refresh_all(self) -> None:
        for tab in self._all_tabs():
            tab.refresh()

    def set_viewer(self, viewer) -> None:
        self.viewer = viewer
        for tab in self._all_tabs():
            tab.viewer = viewer
            if hasattr(tab, "registry"):
                tab.registry.viewer = viewer
            if hasattr(tab, "merge_service"):
                tab.merge_service.viewer = viewer
                tab.merge_service.registry.viewer = viewer
        self.refresh_all()

    def _all_tabs(self):
        return (
            self.mask_cleanup_tab,
            self.class_merge_tab,
            self.final_merge_export_tab,
        )