from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Labels


ACCEPTED_ROLE = "accepted_object"


def is_labels_layer(layer: Any) -> bool:
    return isinstance(layer, Labels)


def labels_layer_names(viewer: Any) -> list[str]:
    if viewer is None:
        return []
    return [layer.name for layer in viewer.layers if is_labels_layer(layer)]


def preview_labels_layer_names(viewer: Any) -> list[str]:
    names = labels_layer_names(viewer)
    preview = [name for name in names if "preview" in name.lower()]
    return preview or names


def safe_get_layer(viewer: Any, name: str | None) -> Any | None:
    if viewer is None or not name:
        return None
    try:
        return viewer.layers[name]
    except (KeyError, ValueError):
        return None


def unique_layer_name(viewer: Any, base_name: str) -> str:
    if viewer is None:
        return base_name
    existing = {layer.name for layer in viewer.layers}
    if base_name not in existing:
        return base_name
    index = 2
    while f"{base_name} {index}" in existing:
        index += 1
    return f"{base_name} {index}"


def accepted_metadata(
    *,
    object_name: str,
    class_name: str,
    class_value: int,
    source_image_layer: str,
    source_preview_layer: str,
    source_task: str = "",
    created_from: str = "Mask Operations",
) -> dict[str, Any]:
    return {
        "sam3_role": ACCEPTED_ROLE,
        "object_name": object_name,
        "class_name": class_name,
        "class_value": int(class_value),
        "instance_kind": "object",
        "source_image_layer": source_image_layer,
        "source_preview_layer": source_preview_layer,
        "source_task": source_task,
        "created_from": created_from,
    }


def layer_metadata(layer: Any) -> dict[str, Any]:
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def copy_labels_data(layer: Labels) -> np.ndarray:
    return np.asarray(layer.data).copy()
