from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

WORKSPACE_FORMAT = "napari-sam3-workspace"
WORKSPACE_VERSION = 1
DEFAULT_XY_CHUNK = 1024
SAM3_STORAGE_METADATA_KEY = "sam3_workspace_storage"


class WorkspaceError(RuntimeError):
    """Raised when a SAM3 workspace cannot be saved or restored safely."""


def save_workspace(
    viewer: Any,
    destination: str | Path,
    *,
    xy_chunk: int = DEFAULT_XY_CHUNK,
) -> dict[str, Any]:
    """Save a lightweight manifest and persist new Labels layers once.

    Existing Zarr-backed layers remain external references. In-memory Labels
    layers are written chunk-by-chunk into a sibling durable-data directory and
    rebound to the writable Zarr array so later saves do not copy mask pixels.
    """

    path = _manifest_path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_root = path.with_name(f"{_workspace_stem(path)}_sam3_data")
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for index, layer in enumerate(list(viewer.layers)):
        try:
            record = _serialize_layer(
                layer,
                manifest_path=path,
                data_root=data_root,
                layer_index=index,
                xy_chunk=int(xy_chunk),
            )
        except Exception as exc:
            skipped.append(
                {"name": str(getattr(layer, "name", f"layer-{index}")), "reason": str(exc)}
            )
            continue
        if record is None:
            skipped.append(
                {
                    "name": str(getattr(layer, "name", f"layer-{index}")),
                    "reason": f"unsupported layer type {type(layer).__name__}",
                }
            )
            continue
        records.append(record)

    active = getattr(getattr(viewer.layers, "selection", None), "active", None)
    if skipped:
        details = "; ".join(
            f"{item['name']}: {item['reason']}" for item in skipped
        )
        raise WorkspaceError(
            f"Workspace was not saved because {len(skipped)} layer(s) could not be serialized. "
            f"The existing manifest was left unchanged. {details}"
        )

    dims = getattr(viewer, "dims", None)
    payload = {
        "format": WORKSPACE_FORMAT,
        "version": WORKSPACE_VERSION,
        "viewer": {
            "dims_current_step": [
                int(value) for value in _as_sequence(getattr(dims, "current_step", ()))
            ],
            "selected_layer_name": str(getattr(active, "name", "")) if active is not None else "",
        },
        "layers": records,
        "skipped_layers": skipped,
    }
    _atomic_write_json(path, payload)
    return {
        "path": str(path),
        "saved_layers": len(records),
        "skipped_layers": skipped,
        "data_root": str(data_root),
    }


def load_workspace(
    viewer: Any,
    source: str | Path,
    *,
    clear_existing: bool = True,
) -> dict[str, Any]:
    """Load a SAM3 manifest while keeping Zarr Labels writable and lazy."""

    path, payload = read_workspace(source)
    records = payload.get("layers")
    if not isinstance(records, list) or not records:
        skipped = payload.get("skipped_layers") or []
        details = "; ".join(
            f"{item.get('name', 'layer')}: {item.get('reason', 'unknown error')}"
            for item in skipped
            if isinstance(item, dict)
        )
        suffix = f" Previous save errors: {details}" if details else ""
        raise WorkspaceError(
            f"Workspace contains no layers and was not loaded: {path}.{suffix}"
        )
    if clear_existing:
        _clear_layers(viewer)

    restored: list[str] = []
    skipped: list[dict[str, str]] = []
    for record in records:
        name = str(record.get("name") or "layer")
        try:
            layer = _restore_layer(viewer, record, manifest_path=path)
            if layer is None:
                raise WorkspaceError("record did not produce a layer")
            _apply_common_state(layer, record)
            restored.append(str(getattr(layer, "name", name)))
        except Exception as exc:
            skipped.append({"name": name, "reason": str(exc)})

    _restore_viewer_state(viewer, payload)
    return {
        "path": str(path),
        "restored_layers": restored,
        "skipped_layers": skipped,
        "imported_legacy": bool(payload.get("imported_from")),
    }


def read_workspace(source: str | Path) -> tuple[Path, dict[str, Any]]:
    path = _manifest_path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != WORKSPACE_FORMAT:
        if _is_legacy_shared_workspace(payload):
            return path, _import_legacy_shared_workspace(path, payload)
        raise WorkspaceError(f"{path} is not a recognized SAM3 or legacy shared workspace manifest.")
    version = int(payload.get("version", 0))
    if version != WORKSPACE_VERSION:
        raise WorkspaceError(
            f"Unsupported SAM3 workspace version {version}; expected {WORKSPACE_VERSION}."
        )
    return path, payload

def _is_legacy_shared_workspace(payload: dict[str, Any]) -> bool:
    return (
        int(payload.get("version", 0)) == 2
        and isinstance(payload.get("layers"), list)
        and payload.get("format") in (None, "")
    )


def _import_legacy_shared_workspace(
    manifest_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Map the version-2 shared/Myelin manifest to SAM3 storage records."""

    asset_root = manifest_path.with_name(f"{manifest_path.stem}_assets")
    converted_layers: list[dict[str, Any]] = []
    for original in payload.get("layers", []):
        if not isinstance(original, dict):
            continue
        record = dict(original)
        if isinstance(record.get("storage"), dict):
            converted_layers.append(record)
            continue

        asset_path = str(record.get("asset_path") or "").strip()
        asset_dataset = str(record.get("asset_dataset") or "").strip().strip("/")
        source = record.get("source") or {}
        source_path = str(source.get("path") or "").strip()
        layer_type = str(record.get("layer_type") or record.get("inline_kind") or "")

        if asset_path:
            record["storage"] = {
                "kind": "zarr",
                "path": _encode_path(asset_root / asset_path, manifest_path.parent),
                "array_path": asset_dataset,
                "mode": "r+" if layer_type == "Labels" else "r",
            }
        elif source_path:
            record["storage"] = {
                "kind": "file",
                "path": _encode_path(source_path, manifest_path.parent),
                "reader_plugin": source.get("reader_plugin"),
            }
        elif layer_type in {"Shapes", "Points"} or record.get("inline_kind") in {
            "Shapes",
            "Points",
        }:
            record["storage"] = {"kind": "inline"}
        else:
            record["storage"] = {"kind": "unsupported"}
        converted_layers.append(record)

    return {
        "format": WORKSPACE_FORMAT,
        "version": WORKSPACE_VERSION,
        "imported_from": "shared-workspace-v2",
        "viewer": dict(payload.get("viewer") or {}),
        "layers": converted_layers,
        "skipped_layers": list(payload.get("skipped_layers") or []),
    }


def create_portable_snapshot(
    workspace_path: str | Path,
    destination: str | Path,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Copy a workspace and every referenced local data source into one folder."""

    source_manifest, payload = read_workspace(workspace_path)
    target_root = Path(destination).expanduser()
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"Portable snapshot destination is not empty: {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    target_data = target_root / "data"
    target_data.mkdir(parents=True, exist_ok=True)

    records = list(payload.get("layers", []))
    copied: dict[Path, Path] = {}
    for index, record in enumerate(records, start=1):
        storage = record.get("storage") or {}
        kind = str(storage.get("kind") or "")
        if kind not in {"file", "zarr"}:
            if progress is not None:
                progress(index, len(records), str(record.get("name", "layer")))
            continue
        source_path = _resolve_path(storage.get("path"), source_manifest.parent)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if source_path in copied:
            copied_path = copied[source_path]
        else:
            suffix = "".join(source_path.suffixes)
            base = _safe_name(str(record.get("name") or source_path.stem))
            copied_path = target_data / f"{index:03d}_{base}{suffix}"
            if source_path.is_dir():
                shutil.copytree(source_path, copied_path)
            else:
                shutil.copy2(source_path, copied_path)
            copied[source_path] = copied_path
        storage["path"] = _encode_path(copied_path, target_root)
        record["storage"] = storage
        if progress is not None:
            progress(index, len(records), str(record.get("name", "layer")))

    portable_manifest = target_root / "workspace.sam3.json"
    _atomic_write_json(portable_manifest, payload)
    return {
        "path": str(portable_manifest),
        "copied_sources": len(copied),
        "layers": len(records),
    }


def _serialize_layer(
    layer: Any,
    *,
    manifest_path: Path,
    data_root: Path,
    layer_index: int,
    xy_chunk: int,
) -> dict[str, Any] | None:
    layer_type = type(layer).__name__
    record = _common_record(layer, layer_type)

    if layer_type == "Labels":
        reference = _zarr_reference(layer)
        if reference is None:
            data_root.mkdir(parents=True, exist_ok=True)
            target = _new_mask_path(data_root, layer_index, str(getattr(layer, "name", "labels")))
            array = _persist_labels_to_zarr(
                layer.data,
                target,
                xy_chunk=xy_chunk,
                scale=tuple(float(value) for value in _as_sequence(getattr(layer, "scale", ()))),
            )
            layer.data = array
            reference = {"path": target, "array_path": "s0"}
        else:
            _ensure_writable_zarr_layer(layer, reference)
        _remember_layer_storage(layer, reference)
        record["storage"] = {
            "kind": "zarr",
            "path": _encode_path(reference["path"], manifest_path.parent),
            "array_path": str(reference.get("array_path") or ""),
            "mode": "r+",
            "shape": [int(value) for value in layer.data.shape],
            "chunks": [int(value) for value in layer.data.chunks],
            "dtype": str(layer.data.dtype),
        }
        return record

    if layer_type == "Image":
        reference = _zarr_reference(layer)
        if reference is not None:
            record["storage"] = {
                "kind": "zarr",
                "path": _encode_path(reference["path"], manifest_path.parent),
                "array_path": str(reference.get("array_path") or ""),
                "mode": "r",
            }
            record.update(_image_state(layer))
            return record
        source_path, reader_plugin = _layer_file_source(layer)
        if source_path is None:
            raise WorkspaceError(
                "Fileless Image layers are not copied during normal Save; export the image or use Portable Snapshot."
            )
        record["storage"] = {
            "kind": "file",
            "path": _encode_path(source_path, manifest_path.parent),
            "reader_plugin": reader_plugin,
        }
        record.update(_image_state(layer))
        return record

    if layer_type == "Shapes":
        record["storage"] = {"kind": "inline"}
        record.update(
            {
                "data": _json_ready(_as_sequence(getattr(layer, "data", []))),
                "shape_type": _json_ready(getattr(layer, "shape_type", [])),
                "features": _json_ready(getattr(layer, "features", {})),
                "edge_width": _json_ready(getattr(layer, "edge_width", 1.0)),
                "edge_color": _json_ready(getattr(layer, "edge_color", None)),
                "face_color": _json_ready(getattr(layer, "face_color", None)),
            }
        )
        return record

    if layer_type == "Points":
        record["storage"] = {"kind": "inline"}
        record.update(
            {
                "data": _json_ready(getattr(layer, "data", [])),
                "features": _json_ready(getattr(layer, "features", {})),
                "size": _json_ready(getattr(layer, "size", 10)),
                "symbol": _json_ready(getattr(layer, "symbol", "o")),
                "face_color": _json_ready(getattr(layer, "face_color", None)),
                "border_color": _json_ready(getattr(layer, "border_color", None)),
            }
        )
        return record

    return None


def _restore_layer(viewer: Any, record: dict[str, Any], *, manifest_path: Path):
    layer_type = str(record.get("layer_type") or "")
    name = str(record.get("name") or layer_type or "layer")
    storage = record.get("storage") or {}
    kind = str(storage.get("kind") or "")

    if kind == "zarr":
        try:
            import zarr
        except Exception as exc:
            raise WorkspaceError("Loading Zarr workspace data requires zarr.") from exc
        store_path = _resolve_path(storage.get("path"), manifest_path.parent)
        array_path = str(storage.get("array_path") or "").strip().strip("/")
        target = store_path / array_path if array_path else store_path
        mode = "r+" if layer_type == "Labels" else "r"
        data = zarr.open_array(str(target), mode=mode)
        if layer_type == "Labels":
            layer = viewer.add_labels(data, name=name)
            _remember_layer_storage(
                layer, {"path": store_path, "array_path": array_path}
            )
            return layer
        if layer_type == "Image":
            return viewer.add_image(data, name=name, rgb=bool(record.get("rgb", False)))

    if kind == "file":
        source = _resolve_path(storage.get("path"), manifest_path.parent)
        if not source.exists():
            raise FileNotFoundError(source)
        loaded = viewer.open(
            [str(source)],
            stack=False,
            plugin=storage.get("reader_plugin") or None,
            layer_type=layer_type.lower() or None,
        )
        return loaded[-1] if loaded else None

    if kind == "inline" and layer_type == "Shapes":
        return viewer.add_shapes(
            data=record.get("data") or [],
            shape_type=record.get("shape_type") or "polygon",
            edge_width=record.get("edge_width", 1.0),
            edge_color=record.get("edge_color"),
            face_color=record.get("face_color"),
            features=record.get("features") or {},
            name=name,
        )

    if kind == "inline" and layer_type == "Points":
        return viewer.add_points(
            data=np.asarray(record.get("data") or [], dtype=float),
            features=record.get("features") or {},
            size=record.get("size", 10),
            symbol=record.get("symbol", "o"),
            face_color=record.get("face_color"),
            border_color=record.get("border_color"),
            name=name,
        )
    return None


def _persist_labels_to_zarr(
    source: Any,
    destination: Path,
    *,
    xy_chunk: int,
    scale: tuple[float, ...],
):
    try:
        import zarr
    except Exception as exc:
        raise WorkspaceError("Saving Labels data requires zarr.") from exc

    shape = tuple(int(value) for value in getattr(source, "shape", ()))
    if len(shape) not in (2, 3):
        raise WorkspaceError(f"SAM3 workspace Labels must be 2D or 3D; got {shape}.")
    if destination.exists():
        raise FileExistsError(
            f"Refusing to replace existing durable mask store: {destination}"
        )
    temp = destination.with_name(f"{destination.name}.__tmp__")
    if temp.exists():
        shutil.rmtree(temp)

    chunks = list(shape)
    chunks[-2] = min(shape[-2], max(1, int(xy_chunk)))
    chunks[-1] = min(shape[-1], max(1, int(xy_chunk)))
    if len(shape) == 3:
        chunks[0] = 1

    root = zarr.open_group(str(temp), mode="w")
    array = root.create_array(
        "s0",
        shape=shape,
        chunks=tuple(chunks),
        dtype=np.dtype(getattr(source, "dtype", np.uint32)),
        fill_value=0,
    )
    axes_names = ("y", "x") if len(shape) == 2 else ("z", "y", "x")
    axes = [{"name": name, "type": "space"} for name in axes_names]
    dataset: dict[str, Any] = {"path": "s0"}
    if len(scale) == len(shape):
        dataset["coordinateTransformations"] = [
            {"type": "scale", "scale": [float(value) for value in scale]}
        ]
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{"name": destination.stem, "axes": axes, "datasets": [dataset]}],
    }

    grid = tuple(
        (size + chunk - 1) // chunk for size, chunk in zip(shape, chunks, strict=True)
    )
    for index in np.ndindex(grid):
        region = tuple(
            slice(i * chunk, min(size, (i + 1) * chunk))
            for i, chunk, size in zip(index, chunks, shape, strict=True)
        )
        block = np.asarray(source[region])
        if block.size and np.any(block):
            array[region] = block

    temp.replace(destination)
    return zarr.open_array(str(destination / "s0"), mode="r+")


def _ensure_writable_zarr_layer(layer: Any, reference: dict[str, Any]) -> None:
    data = getattr(layer, "data", None)
    if not bool(getattr(data, "read_only", False)):
        return
    try:
        import zarr
    except Exception as exc:
        raise WorkspaceError("Opening writable Labels data requires zarr.") from exc
    store_path = Path(reference["path"]).expanduser()
    array_path = str(reference.get("array_path") or "").strip().strip("/")
    target = store_path / array_path if array_path else store_path
    try:
        writable = zarr.open_array(str(target), mode="r+")
    except Exception as exc:
        raise WorkspaceError(
            f"Labels store is not writable: {target}. Create a writable SAM3 copy first."
        ) from exc
    layer.data = writable


def _zarr_reference(layer: Any) -> dict[str, Any] | None:
    metadata = getattr(layer, "metadata", {}) or {}
    remembered = metadata.get(SAM3_STORAGE_METADATA_KEY) if isinstance(metadata, dict) else None
    if isinstance(remembered, dict) and remembered.get("path"):
        path = Path(str(remembered["path"])).expanduser()
        return {"path": path, "array_path": str(remembered.get("array_path") or "")}

    data = getattr(layer, "data", None)
    module = type(data).__module__.lower()
    name = type(data).__name__.lower()
    if "zarr" not in module and "zarr" not in name:
        return None
    store = getattr(data, "store", None)
    root = getattr(store, "root", None)
    if root is None:
        root = getattr(store, "path", None)
    if root is None:
        return None
    return {
        "path": Path(str(root)).expanduser(),
        "array_path": str(getattr(data, "path", "") or "").strip().strip("/"),
    }


def _remember_layer_storage(layer: Any, reference: dict[str, Any]) -> None:
    metadata = dict(getattr(layer, "metadata", {}) or {})
    metadata[SAM3_STORAGE_METADATA_KEY] = {
        "path": str(Path(reference["path"]).expanduser()),
        "array_path": str(reference.get("array_path") or ""),
    }
    try:
        layer.metadata = metadata
    except Exception:
        pass


def _layer_file_source(layer: Any) -> tuple[Path | None, str | None]:
    source = getattr(layer, "source", None)
    value = getattr(source, "path", None)
    if not value:
        return None, None
    path = Path(str(value)).expanduser()
    return path, getattr(source, "reader_plugin", None)


def _common_record(layer: Any, layer_type: str) -> dict[str, Any]:
    return {
        "layer_type": layer_type,
        "name": str(getattr(layer, "name", layer_type)),
        "visible": bool(getattr(layer, "visible", True)),
        "opacity": float(getattr(layer, "opacity", 1.0)),
        "blending": str(getattr(layer, "blending", "translucent")),
        "scale": [float(value) for value in _as_sequence(getattr(layer, "scale", ()))],
        "translate": [
            float(value) for value in _as_sequence(getattr(layer, "translate", ()))
        ],
    }


def _image_state(layer: Any) -> dict[str, Any]:
    return {
        "contrast_limits": [
            float(value)
            for value in _as_sequence(getattr(layer, "contrast_limits", ()))
        ],
        "colormap": str(getattr(getattr(layer, "colormap", None), "name", "gray")),
        "gamma": float(getattr(layer, "gamma", 1.0)),
        "rgb": bool(getattr(layer, "rgb", False)),
    }


def _apply_common_state(layer: Any, record: dict[str, Any]) -> None:
    for attribute in ("name", "visible", "opacity", "blending"):
        if attribute in record:
            try:
                setattr(layer, attribute, record[attribute])
            except Exception:
                pass
    for attribute in ("scale", "translate"):
        value = record.get(attribute)
        if value:
            try:
                setattr(layer, attribute, tuple(value))
            except Exception:
                pass
    if str(record.get("layer_type")) == "Image":
        for attribute in ("contrast_limits", "gamma"):
            if attribute in record:
                try:
                    setattr(layer, attribute, record[attribute])
                except Exception:
                    pass
        if record.get("colormap"):
            try:
                layer.colormap = record["colormap"]
            except Exception:
                pass


def _restore_viewer_state(viewer: Any, payload: dict[str, Any]) -> None:
    viewer_state = payload.get("viewer") or {}
    steps = viewer_state.get("dims_current_step") or []
    if steps:
        try:
            viewer.dims.current_step = tuple(int(value) for value in steps)
        except Exception:
            pass
    selected = str(viewer_state.get("selected_layer_name") or "")
    if selected:
        try:
            viewer.layers.selection.active = viewer.layers[selected]
        except Exception:
            pass


def _clear_layers(viewer: Any) -> None:
    layers = viewer.layers
    clear = getattr(layers, "clear", None)
    if callable(clear):
        clear()
        return
    while len(layers):
        layers.remove(layers[0])


def _new_mask_path(data_root: Path, index: int, name: str) -> Path:
    base = data_root / f"{index:03d}_{_safe_name(name)}.ome.zarr"
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = data_root / f"{index:03d}_{_safe_name(name)}_{counter}.ome.zarr"
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or "layer"


def _manifest_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists() and path.suffix.lower() != ".json":
        path = path.with_suffix(".sam3.json")
    return path


def _workspace_stem(path: Path) -> str:
    name = path.name
    suffix = ".sam3.json"
    if name.lower().endswith(suffix):
        return name[: -len(suffix)] or "workspace"
    return path.stem or "workspace"


def _encode_path(path: str | Path, base: Path) -> str:
    target = Path(path).expanduser()
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(target.resolve())


def _resolve_path(value: Any, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise WorkspaceError("Workspace storage path is missing.")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        try:
            return _json_ready(value.to_dict())
        except Exception:
            pass
    return str(value)
