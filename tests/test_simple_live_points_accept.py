from types import SimpleNamespace

import numpy as np

from napari_sam3_assistant.widgets.shared.shared_context import SharedContext
from napari_sam3_assistant.widgets.simple.simple_mode_controller import SimpleModeController


class FakeLayer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.mode = "add"
        self.ndim = data.shape[1] if data.ndim == 2 and data.shape[0] == 0 else data.ndim
        self.selected_data = set()
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class FakeLayers(dict):
    def __iter__(self):
        return iter(self.values())

    def remove(self, layer):
        self.pop(layer.name, None)


class FakeViewer:
    def __init__(self):
        self.layers = FakeLayers()
        self.layers.selection = SimpleNamespace(active=None)

    def add_labels(self, data, name):
        layer = FakeLayer(data, name)
        self.layers[name] = layer
        return layer


class FakeOwner:
    def __init__(self, viewer, points):
        self.viewer = viewer
        self.points = points
        self.logs = []

    def _first_preview_labels_layer(self):
        return self.viewer.layers.get("SAM3 preview labels")

    def _current_points_layer(self):
        return self.points

    def _clear_preview_layers(self):
        layer = self.viewer.layers.get("SAM3 preview labels")
        if layer is not None:
            self.viewer.layers.remove(layer)

    def _log(self, message):
        self.logs.append(message)


class FakeRouter:
    def __init__(self, owner):
        self._owner = owner

    def execution_owner(self):
        return self._owner


def test_live_points_accept_clear_accumulates_preview_and_resets_points():
    viewer = FakeViewer()
    preview = viewer.add_labels(np.zeros((8, 8), dtype=np.uint32), "SAM3 preview labels")
    preview.data[2:5, 3:6] = 4
    points = FakeLayer(np.asarray([[3.0, 4.0]]), "SAM3 points")
    owner = FakeOwner(viewer, points)
    controller = SimpleModeController(SharedContext(viewer=viewer, task_router=FakeRouter(owner)))

    assert controller.accept_live_preview((3, 4), clear_prompt=True)

    accepted = viewer.layers["SAM3 live accepted labels"]
    assert int(accepted.data.max()) == 1
    assert np.count_nonzero(accepted.data == 1) == 9
    assert "SAM3 preview labels" not in viewer.layers
    assert points.data.shape == (0, 2)
    assert viewer.layers.selection.active is points
    assert points.mode == "add"

    assert controller.undo_live_accept()
    assert not np.any(viewer.layers["SAM3 live accepted labels"].data)
