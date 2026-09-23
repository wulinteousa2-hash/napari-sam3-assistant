from __future__ import annotations

from pathlib import Path
from typing import Any

from napari import current_viewer
from napari.viewer import Viewer
from qtpy.QtCore import QSettings
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workspace import create_portable_snapshot, load_workspace, save_workspace

SETTINGS_ORG = "napari-sam3-assistant"
SETTINGS_APP = "sam3-assistant"
RECENT_WORKSPACES_KEY = "workspace/recent_manifests"
LAST_WORKSPACE_KEY = "workspace/last_manifest"
RECENT_LIMIT = 10


class WorkspaceManagerWidget(QWidget):
    """Independent SAM3 workspace/session manager for lazy writable mask data."""

    def __init__(
        self,
        napari_viewer: Viewer | None = None,
        viewer: Viewer | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__()
        self.viewer = napari_viewer or viewer or current_viewer()
        self.settings = settings or QSettings(SETTINGS_ORG, SETTINGS_APP)
        last = str(self.settings.value(LAST_WORKSPACE_KEY, "", type=str) or "").strip()
        self.workspace_path: Path | None = Path(last).expanduser() if last else None

        self.current_label = QLabel()
        self.current_label.setWordWrap(True)
        self.status_label = QLabel(
            "Normal Save writes a small manifest only. Writable masks remain in their OME-Zarr stores."
        )
        self.status_label.setWordWrap(True)
        self.recent_list = QListWidget()
        self.recent_list.setMinimumHeight(120)
        self.recent_list.itemDoubleClicked.connect(lambda _item: self.open_selected_recent())

        new_button = QPushButton("New")
        open_button = QPushButton("Open")
        save_button = QPushButton("Save")
        save_as_button = QPushButton("Save As")
        recent_button = QPushButton("Open Recent")
        snapshot_button = QPushButton("Portable Snapshot")

        new_button.setToolTip("Clear the viewer and start an unsaved SAM3 workspace.")
        open_button.setToolTip("Load a SAM3 manifest and open referenced masks lazily in writable mode.")
        save_button.setToolTip("Save only the manifest; new in-memory Labels are persisted once as OME-Zarr.")
        save_as_button.setToolTip("Write another manifest that references the same durable data stores.")
        recent_button.setToolTip("Open the workspace selected in the Recent list.")
        snapshot_button.setToolTip("Explicitly copy the manifest and all referenced local data into one folder.")

        new_button.clicked.connect(self.new_workspace)
        open_button.clicked.connect(self.open_workspace)
        save_button.clicked.connect(self.save)
        save_as_button.clicked.connect(self.save_as)
        recent_button.clicked.connect(self.open_selected_recent)
        snapshot_button.clicked.connect(self.create_snapshot)

        first_row = QHBoxLayout()
        first_row.addWidget(new_button)
        first_row.addWidget(open_button)
        first_row.addWidget(save_button)
        first_row.addWidget(save_as_button)

        second_row = QHBoxLayout()
        second_row.addWidget(recent_button)
        second_row.addWidget(snapshot_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(QLabel("SAM3 Workspace Manager"))
        layout.addWidget(self.current_label)
        layout.addLayout(first_row)
        layout.addWidget(QLabel("Recent workspaces"))
        layout.addWidget(self.recent_list)
        layout.addLayout(second_row)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self._refresh_current_label()
        self._refresh_recent_list()

    def new_workspace(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if len(self.viewer.layers) and not self._confirm_clear(
            "Start a new SAM3 workspace? Current layers will be removed from the viewer. "
            "Writable Zarr data already saved on disk will not be deleted."
        ):
            return
        self.viewer.layers.clear()
        self.workspace_path = None
        self.settings.remove(LAST_WORKSPACE_KEY)
        self._refresh_current_label()
        self._set_status("Started a new empty SAM3 workspace.")

    def open_workspace(self) -> None:
        initial = str(self.workspace_path.parent) if self.workspace_path else ""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Open SAM3 Workspace",
            initial,
            "SAM3 workspace (*.sam3.json *.json)",
        )
        if chosen:
            self._load_path(Path(chosen))

    def open_selected_recent(self) -> None:
        item = self.recent_list.currentItem()
        if item is None:
            self._set_status("Select a recent workspace first.")
            return
        self._load_path(Path(item.text()))

    def save(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if self.workspace_path is None:
            self.save_as()
            return
        self._save_path(self.workspace_path)

    def save_as(self) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        initial = str(self.workspace_path or Path.cwd() / "workspace.sam3.json")
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Save SAM3 Workspace As",
            initial,
            "SAM3 workspace (*.sam3.json)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if not str(path).lower().endswith(".json"):
            path = path.with_suffix(".sam3.json")
        self._save_path(path)

    def create_snapshot(self) -> None:
        if self.workspace_path is None:
            self._set_status("Save the workspace before creating a portable snapshot.")
            return
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Empty Portable Snapshot Folder",
            str(self.workspace_path.parent),
        )
        if not chosen:
            return
        destination = Path(chosen)
        if destination.exists() and any(destination.iterdir()):
            self._set_status("Portable snapshot folder must be empty.")
            return
        answer = QMessageBox.question(
            self,
            "Create Portable Snapshot",
            "This explicitly copies every referenced local image and mask. "
            "Large TIFF or OME-Zarr data may require substantial time and disk space. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._run(
            lambda: create_portable_snapshot(self.workspace_path, destination),
            success=lambda result: (
                f"Created portable snapshot at {result['path']} with "
                f"{result['copied_sources']} copied source(s)."
            ),
        )

    def _save_path(self, path: Path) -> None:
        self._run(
            lambda: save_workspace(self.viewer, path),
            success=lambda result: (
                f"Saved {result['saved_layers']} layer(s) to {result['path']}. "
                f"Skipped {len(result['skipped_layers'])} layer(s)."
            ),
            completed_path=path,
        )

    def _load_path(self, path: Path) -> None:
        if self.viewer is None:
            self._set_status("No napari viewer is available.")
            return
        if len(self.viewer.layers) and not self._confirm_clear(
            "Open this SAM3 workspace? Current layers will be removed from the viewer. "
            "Writable Zarr data already saved on disk will not be deleted."
        ):
            return
        self._run(
            lambda: load_workspace(self.viewer, path, clear_existing=True),
            success=lambda result: (
                (
                    "Imported legacy workspace and loaded "
                    if result.get("imported_legacy")
                    else "Loaded "
                )
                + f"{len(result['restored_layers'])} layer(s) from {result['path']}. "
                f"Skipped {len(result['skipped_layers'])} layer(s)."
            ),
            completed_path=path,
        )

    def _run(
        self,
        operation,
        *,
        success,
        completed_path: Path | None = None,
    ) -> None:
        try:
            result = operation()
        except Exception as exc:
            self._set_status(str(exc))
            return
        if completed_path is not None:
            self.workspace_path = completed_path.expanduser()
            self._remember(self.workspace_path)
            self._refresh_current_label()
            self._refresh_recent_list()
        self._set_status(success(result))

    def _confirm_clear(self, message: str) -> bool:
        answer = QMessageBox.question(
            self,
            "SAM3 Workspace",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _recent_paths(self) -> list[str]:
        value = self.settings.value(RECENT_WORKSPACES_KEY, [])
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in list(value or []) if str(item).strip()]

    def _remember(self, path: Path) -> None:
        normalized = str(path.expanduser().resolve())
        recent = [item for item in self._recent_paths() if item != normalized]
        recent.insert(0, normalized)
        self.settings.setValue(RECENT_WORKSPACES_KEY, recent[:RECENT_LIMIT])
        self.settings.setValue(LAST_WORKSPACE_KEY, normalized)

    def _refresh_recent_list(self) -> None:
        self.recent_list.clear()
        for path in self._recent_paths():
            self.recent_list.addItem(path)

    def _refresh_current_label(self) -> None:
        text = str(self.workspace_path) if self.workspace_path else "Unsaved workspace"
        self.current_label.setText(f"Current: {text}")

    def _set_status(self, message: str) -> None:
        self.status_label.setText(str(message))
        if self.viewer is not None:
            try:
                self.viewer.status = str(message)
            except Exception:
                pass
