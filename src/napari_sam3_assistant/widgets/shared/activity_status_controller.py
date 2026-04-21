from __future__ import annotations

from qtpy.QtCore import QObject, Signal


class ActivityStatusController(QObject):
    """Shared, presentation-neutral activity status for both UI modes."""

    READY = "Ready"
    STARTING_TASK = "Starting task..."
    LOADING_MODEL = "Loading model..."
    RUNNING_PREVIEW = "Running SAM3 preview..."
    STARTING_3D_PROPAGATION = "Starting 3D propagation..."
    PREVIEW_READY = "Preview ready"
    NO_OBJECTS_FOUND = "No objects found"
    TASK_FAILED = "Task failed"

    status_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._status = self.READY

    @property
    def status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        self.status_changed.emit(status)

    def set_ready(self) -> None:
        self.set_status(self.READY)

    def set_starting_task(self) -> None:
        self.set_status(self.STARTING_TASK)

    def set_loading_model(self) -> None:
        self.set_status(self.LOADING_MODEL)

    def set_running_preview(self) -> None:
        self.set_status(self.RUNNING_PREVIEW)

    def set_starting_3d_propagation(self) -> None:
        self.set_status(self.STARTING_3D_PROPAGATION)

    def set_preview_ready(self) -> None:
        self.set_status(self.PREVIEW_READY)

    def set_no_objects_found(self) -> None:
        self.set_status(self.NO_OBJECTS_FOUND)

    def set_task_failed(self) -> None:
        self.set_status(self.TASK_FAILED)

    def finish_success(self) -> None:
        if self._status not in {self.PREVIEW_READY, self.NO_OBJECTS_FOUND}:
            self.set_ready()
