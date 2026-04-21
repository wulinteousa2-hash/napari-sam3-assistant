from __future__ import annotations

from typing import Any

import numpy as np

from ..core.models import Sam3Task


class RefinementTaskRunner:
    def __init__(self, widget: Any) -> None:
        self.widget = widget

    def live_refinement_enabled(self) -> bool:
        w = self.widget
        return (
            w._current_task() == Sam3Task.REFINE
            and w.prompt_tool_combo.currentData() == "points"
            and w._worker is None
        )

    def toggle_next_point_mode(self) -> None:
        w = self.widget
        current = w.point_polarity_combo.currentData() or "positive"
        new_value = "negative" if current == "positive" else "positive"
        index = w.point_polarity_combo.findData(new_value)
        if index >= 0:
            w.point_polarity_combo.setCurrentIndex(index)

        w._set_current_point_polarity()
        w._log(f"Next point mode: {new_value.capitalize()}")

    def flip_existing_point_polarity(self) -> None:
        w = self.widget
        layer = w._current_points_layer()
        if layer is None:
            w._log("No points layer selected.")
            return
        data = np.asarray(layer.data)
        if len(data) == 0:
            w._log("No points available to flip.")
            return

        selected = sorted(getattr(layer, "selected_data", []))
        indices = selected or [len(data) - 1]
        properties = dict(getattr(layer, "properties", {}) or {})
        values = w._point_polarity_values(layer)
        for idx in indices:
            values[idx] = "negative" if values[idx] == "positive" else "positive"
        properties["polarity"] = np.asarray(values, dtype=object)
        layer.properties = properties
        layer.refresh_colors()
        w._log(f"Flipped {len(indices)} point(s); rerunning Live Points.")
        self.run_live_refinement_preview()

    def sync_live_refinement_layer(self) -> None:
        w = self.widget
        if w.viewer is None:
            w.live_point_refinement.set_points_layer(None)
            return

        layer_name = w._optional_combo_data(w.points_layer_combo)
        if not layer_name:
            w.live_point_refinement.set_points_layer(None)
            return

        try:
            layer = w.viewer.layers[layer_name]
        except (KeyError, ValueError):
            layer = None

        w.live_point_refinement.set_points_layer(layer)

    def run_live_refinement_preview(self) -> None:
        w = self.widget
        if not self.live_refinement_enabled():
            return
        w._set_live_refinement_status("Activity: Live Points running...")
        w.image_runner.run_current_task()
