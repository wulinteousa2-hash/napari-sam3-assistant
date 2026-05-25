from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Any


@dataclass(frozen=True)
class OmeZarrMaskSource:
    """Metadata needed to safely write edited mask regions back to OME-Zarr."""

    store_path: Path | None
    array_path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...] | None
    dtype: str
    axes: tuple[str, ...] | None = None

    def display_text(self) -> str:
        bits = []
        if self.store_path is not None:
            bits.append(str(self.store_path))
        bits.append(f"array /{self.array_path}")
        bits.append("shape " + " x ".join(str(value) for value in self.shape))
        if self.chunks:
            bits.append("chunks " + " x ".join(str(value) for value in self.chunks))
        if self.axes:
            bits.append("axes " + "/".join(self.axes))
        bits.append(f"dtype {self.dtype}")
        return "; ".join(bits)


def inspect_ome_zarr_array(data: Any) -> OmeZarrMaskSource | None:
    """Return OME-Zarr array metadata for zarr-backed layer data, if available."""

    if not _looks_like_zarr_array(data):
        return None
    shape = _tuple_attr(data, "shape")
    if not shape:
        return None
    array_path = _array_path(data)
    store_path = _store_path(data)
    axes = _ome_axes_for_array(store_path, array_path)
    return OmeZarrMaskSource(
        store_path=store_path,
        array_path=array_path,
        shape=shape,
        chunks=_tuple_attr(data, "chunks"),
        dtype=str(getattr(data, "dtype", "unknown")),
        axes=axes,
    )


def inspect_ome_zarr_path(path: str | Path, array_path: str = "s0") -> OmeZarrMaskSource:
    """Open an OME-Zarr store and return metadata for one array path."""

    try:
        import zarr
    except Exception as exc:
        raise RuntimeError("OME-Zarr source inspection requires the 'zarr' package.") from exc

    target = Path(path)
    group = zarr.open_group(str(target), mode="r")
    try:
        array = group[array_path]
    except KeyError as exc:
        raise ValueError(f"OME-Zarr array '{array_path}' was not found in {target}.") from exc
    axes = _ome_axes_for_array(target, array_path)
    return OmeZarrMaskSource(
        store_path=target,
        array_path=_normalize_array_path(array_path),
        shape=_tuple_attr(array, "shape"),
        chunks=_tuple_attr(array, "chunks"),
        dtype=str(getattr(array, "dtype", "unknown")),
        axes=axes,
    )


def _looks_like_zarr_array(data: Any) -> bool:
    module = type(data).__module__.lower()
    name = type(data).__name__.lower()
    return "zarr" in module or "zarr" in name or (
        hasattr(data, "store") and hasattr(data, "shape") and hasattr(data, "dtype")
    )


def _tuple_attr(data: Any, attr: str) -> tuple[int, ...]:
    value = getattr(data, attr, None)
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in value)
    except TypeError:
        return ()


def _array_path(data: Any) -> str:
    for attr in ("path", "basename", "name"):
        value = getattr(data, attr, None)
        if value:
            return _normalize_array_path(str(value))
    return "s0"


def _normalize_array_path(value: str) -> str:
    cleaned = str(value).strip().strip("/")
    return cleaned or "s0"


def _store_path(data: Any) -> Path | None:
    store = getattr(data, "store", None)
    root = getattr(store, "root", None)
    if root is not None:
        return _path_from_value(root)
    path = getattr(store, "path", None)
    if path is not None:
        return _path_from_value(path)
    return None


def _path_from_value(value: Any) -> Path:
    text = str(value)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(text)

def _ome_axes_for_array(store_path: Path | None, array_path: str) -> tuple[str, ...] | None:
    if store_path is None:
        return None
    try:
        import zarr

        group = zarr.open_group(str(store_path), mode="r")
    except Exception:
        return None
    attrs = _attrs_to_dict(getattr(group, "attrs", {}))
    ome = attrs.get("ome")
    if not isinstance(ome, dict):
        return None
    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list):
        return None
    normalized = _normalize_array_path(array_path)
    for multiscale in multiscales:
        if not isinstance(multiscale, dict):
            continue
        datasets = multiscale.get("datasets")
        if isinstance(datasets, list):
            paths = {
                _normalize_array_path(str(dataset.get("path", "")))
                for dataset in datasets
                if isinstance(dataset, dict)
            }
            if paths and normalized not in paths:
                continue
        axes = multiscale.get("axes")
        if isinstance(axes, list):
            names = []
            for axis in axes:
                if isinstance(axis, dict) and axis.get("name"):
                    names.append(str(axis["name"]))
            if names:
                return tuple(names)
    return None


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    if hasattr(attrs, "asdict"):
        try:
            return dict(attrs.asdict())
        except Exception:
            pass
    try:
        return dict(attrs)
    except Exception:
        return {}
