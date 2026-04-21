from __future__ import annotations

from typing import Any


class TaskRouter:
    """Thin shared execution bridge for both UI modes.

    This file intentionally avoids duplicating runner logic. It delegates into
    the existing MainWidget-style methods or runner objects already present in
    the current system.
    """

    def __init__(self, owner: Any, shared_context: Any) -> None:
        self.owner = owner
        self.shared_context = shared_context

    def set_owner(self, owner: Any) -> None:
        self.owner = owner

    def execution_owner(self) -> Any:
        return self.owner

    def run_current_task(self) -> None:
        if hasattr(self.owner, "_run_current_task"):
            self.owner._run_current_task()

    def propagate_existing_session(self) -> None:
        if hasattr(self.owner, "_propagate_existing_session"):
            self.owner._propagate_existing_session()

    def clear_preview_layers(self) -> None:
        if hasattr(self.owner, "_clear_preview_layers"):
            self.owner._clear_preview_layers()

    def clear_results(self) -> None:
        if hasattr(self.owner, "_clear_results_table"):
            self.owner._clear_results_table()

    def open_mask_operations(self) -> None:
        if hasattr(self.owner, "_open_mask_operations"):
            self.owner._open_mask_operations()
