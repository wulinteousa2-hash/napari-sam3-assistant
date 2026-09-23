import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_sam3_assistant.workspace import (
    WorkspaceError,
    create_portable_snapshot,
    load_workspace,
    save_workspace,
)


class Labels:
    def __init__(self, data, name="labels"):
        self.data = data
        self.name = name
        self.visible = True
        self.opacity = 0.7
        self.blending = "translucent"
        self.scale = (1.0,) * len(data.shape)
        self.translate = (0.0,) * len(data.shape)
        self.metadata = {}
        self.source = SimpleNamespace(path=None, reader_plugin=None)


class LayerList(list):
    def __init__(self, values=()):
        super().__init__(values)
        self.selection = SimpleNamespace(active=self[0] if self else None)

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)


class FakeViewer:
    def __init__(self, layers=()):
        self.layers = LayerList(layers)
        self.dims = SimpleNamespace(current_step=(0, 0))
        self.status = ""

    def add_labels(self, data, name):
        layer = Labels(data, name)
        self.layers.append(layer)
        return layer


def test_first_save_persists_numpy_labels_once_with_1024_chunks(tmp_path):
    pytest.importorskip("zarr")
    data = np.zeros((1100, 1200), dtype=np.uint32)
    data[100:130, 200:240] = 5
    layer = Labels(data, "axon mask")
    viewer = FakeViewer([layer])
    path = tmp_path / "study.sam3.json"

    first = save_workspace(viewer, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    storage = payload["layers"][0]["storage"]
    store_path = tmp_path / storage["path"]

    assert first["saved_layers"] == 1
    assert storage["kind"] == "zarr"
    assert storage["chunks"] == [1024, 1024]
    assert store_path.exists()
    assert not isinstance(layer.data, np.ndarray)
    np.testing.assert_array_equal(layer.data[90:140, 190:250], data[90:140, 190:250])

    stores_before = sorted(first_path.name for first_path in store_path.parent.glob("*.ome.zarr"))
    second = save_workspace(viewer, path)
    stores_after = sorted(second_path.name for second_path in store_path.parent.glob("*.ome.zarr"))

    assert second["saved_layers"] == 1
    assert stores_after == stores_before


def test_save_accepts_numpy_scale_and_translate_properties(tmp_path):
    pytest.importorskip("zarr")
    layer = Labels(np.ones((12, 14), dtype=np.uint8), "mask")
    layer.scale = np.asarray((2.0, 3.0))
    layer.translate = np.asarray((4.0, 5.0))
    path = tmp_path / "study.sam3.json"

    save_workspace(FakeViewer([layer]), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["layers"][0]["scale"] == [2.0, 3.0]
    assert payload["layers"][0]["translate"] == [4.0, 5.0]


def test_save_does_not_replace_existing_manifest_when_a_layer_fails(tmp_path):
    class UnsupportedLayer:
        name = "unsupported"
        visible = True
        opacity = 1.0
        blending = "translucent"
        scale = np.asarray((1.0, 1.0))
        translate = np.asarray((0.0, 0.0))

    path = tmp_path / "study.sam3.json"
    original = '{"preserve": true}\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(WorkspaceError, match="existing manifest was left unchanged"):
        save_workspace(FakeViewer([UnsupportedLayer()]), path)

    assert path.read_text(encoding="utf-8") == original


def test_load_rejects_empty_manifest_without_clearing_open_layers(tmp_path):
    existing = Labels(np.ones((4, 5), dtype=np.uint8), "already open")
    viewer = FakeViewer([existing])
    path = tmp_path / "empty.sam3.json"
    path.write_text(
        json.dumps(
            {
                "format": "napari-sam3-workspace",
                "version": 1,
                "layers": [],
                "skipped_layers": [
                    {"name": "mask", "reason": "serialization failed"}
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceError, match="contains no layers"):
        load_workspace(viewer, path)

    assert list(viewer.layers) == [existing]


def test_load_reopens_labels_writable_without_numpy_materialization(tmp_path):
    zarr = pytest.importorskip("zarr")
    source_viewer = FakeViewer([Labels(np.ones((20, 30), dtype=np.uint16), "mask")])
    path = tmp_path / "study.sam3.json"
    save_workspace(source_viewer, path)

    viewer = FakeViewer()
    result = load_workspace(viewer, path)

    assert result["restored_layers"] == ["mask"]
    restored = viewer.layers[0]
    assert type(restored.data).__module__.startswith("zarr")
    restored.data[4:6, 7:9] = 12

    payload = json.loads(path.read_text(encoding="utf-8"))
    storage = payload["layers"][0]["storage"]
    stored = zarr.open_array(
        str(tmp_path / storage["path"] / storage["array_path"]),
        mode="r",
    )
    assert np.all(stored[4:6, 7:9] == 12)


def test_save_as_references_same_durable_mask_store(tmp_path):
    pytest.importorskip("zarr")
    viewer = FakeViewer([Labels(np.ones((10, 12), dtype=np.uint8), "mask")])
    first_path = tmp_path / "first.sam3.json"
    second_path = tmp_path / "second.sam3.json"

    save_workspace(viewer, first_path)
    save_workspace(viewer, second_path)

    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    first_store = (first_path.parent / first["layers"][0]["storage"]["path"]).resolve()
    second_store = (second_path.parent / second["layers"][0]["storage"]["path"]).resolve()

    assert first_store == second_store
    assert len(list(tmp_path.glob("*_sam3_data/*.ome.zarr"))) == 1


def test_save_rebinds_external_read_only_zarr_labels_as_writable(tmp_path):
    zarr = pytest.importorskip("zarr")
    external = tmp_path / "external.ome.zarr"
    source = zarr.open_array(
        str(external),
        mode="w",
        shape=(12, 14),
        chunks=(6, 7),
        dtype=np.uint16,
    )
    source[2:4, 3:6] = 8
    layer = Labels(zarr.open_array(str(external), mode="r"), "external mask")
    assert layer.data.read_only is True

    workspace = tmp_path / "study.sam3.json"
    save_workspace(FakeViewer([layer]), workspace)

    assert layer.data.read_only is False
    layer.data[0, 0] = 3
    assert int(zarr.open_array(str(external), mode="r")[0, 0]) == 3


def test_portable_snapshot_is_explicit_self_contained_copy(tmp_path):
    pytest.importorskip("zarr")
    viewer = FakeViewer([Labels(np.eye(16, dtype=np.uint8), "mask")])
    workspace = tmp_path / "source.sam3.json"
    save_workspace(viewer, workspace)

    destination = tmp_path / "portable"
    result = create_portable_snapshot(workspace, destination)
    portable = json.loads(
        (destination / "workspace.sam3.json").read_text(encoding="utf-8")
    )
    storage = portable["layers"][0]["storage"]

    assert result["copied_sources"] == 1
    assert not Path(storage["path"]).is_absolute()
    assert (destination / storage["path"]).exists()


def test_loads_legacy_shared_workspace_and_converts_on_next_save(tmp_path):
    zarr = pytest.importorskip("zarr")
    asset_root = tmp_path / "workspace_assets"
    store_path = asset_root / "layer_001_snapshot_labels.ome.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    labels = root.require_group("labels").require_group("labels").create_array(
        "s0",
        shape=(20, 30),
        chunks=(10, 10),
        dtype=np.uint32,
    )
    labels[3:6, 4:8] = 7
    manifest_path = tmp_path / "workspace.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "viewer": {
                    "dims_current_step": [0, 0],
                    "selected_layer_name": "one",
                },
                "layers": [
                    {
                        "layer_type": "Labels",
                        "name": "one",
                        "visible": True,
                        "opacity": 0.7,
                        "blending": "translucent",
                        "scale": [1.0, 1.0],
                        "translate": [0.0, 0.0],
                        "source": {"path": None, "reader_plugin": None},
                        "asset_path": "layer_001_snapshot_labels.ome.zarr",
                        "asset_format": "ome-zarr",
                        "asset_dataset": "labels/labels/s0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    viewer = FakeViewer()
    result = load_workspace(viewer, manifest_path)

    assert result["imported_legacy"] is True
    assert result["restored_layers"] == ["one"]
    assert viewer.layers[0].data.read_only is False
    assert int(viewer.layers[0].data[3, 4]) == 7

    save_workspace(viewer, manifest_path)
    converted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert converted["format"] == "napari-sam3-workspace"
    assert converted["layers"][0]["storage"]["kind"] == "zarr"
    assert (
        tmp_path / converted["layers"][0]["storage"]["path"]
    ).resolve() == store_path.resolve()


def test_workspace_widget_is_registered_as_independent_plugin_entry():
    manifest = (
        Path(__file__).parents[1]
        / "src"
        / "napari_sam3_assistant"
        / "napari.yaml"
    ).read_text(encoding="utf-8")

    assert "make_workspace_widget" in manifest
    assert "SAM3 Assistant: Workspace Manager" in manifest
