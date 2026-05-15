from __future__ import annotations

from collections import deque

from qtpy.QtCore import QObject, Signal


class ActivityLogController(QObject):
    """Small shared log buffer for compact task-focused UI surfaces."""

    line_added = Signal(str)
    cleared = Signal()

    def __init__(self, limit: int = 80) -> None:
        super().__init__()
        self._lines: deque[str] = deque(maxlen=limit)

    def append(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self._lines.append(text)
        self.line_added.emit(text)

    def clear(self) -> None:
        self._lines.clear()
        self.cleared.emit()

    def recent(self, limit: int = 10) -> list[str]:
        if limit <= 0:
            return []
        return list(self._lines)[-limit:]
