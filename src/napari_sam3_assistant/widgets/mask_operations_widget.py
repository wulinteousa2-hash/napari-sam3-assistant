from __future__ import annotations

from napari import current_viewer
from napari.viewer import Viewer
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ..mask_operations import MaskOperationsPanel


class MaskOperationsWidget(QWidget):
    """Standalone napari widget for SAM3 mask cleanup, merge, and export."""

    def __init__(
        self,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer or viewer or current_viewer()
        self.panel = MaskOperationsPanel(self.viewer)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.panel)
        self.setLayout(layout)

    def refresh_from_viewer(self) -> None:
        self.viewer = self.viewer or current_viewer()
        self.panel.set_viewer(self.viewer)
