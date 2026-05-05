from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Labels


ACCEPTED_ROLE = "accepted_object"
REJECTED_ROLE = "rejected_object"
CLASS_MASK_ROLE = "class_working_mask"
FINAL_MASK_ROLE = "final_training_mask"
OVERLAP_ROLE = "overlap_map"
DEFAULT_REVIEW_STATUS = "unreviewed"
ACCEPTED_STATUS = "accepted"
REJECTED_STATUS = "rejected"
NEEDS_EDIT_STATUS = "needs_edit"
REVIEW_STATUSES = {DEFAULT_REVIEW_STATUS, ACCEPTED_STATUS, REJECTED_STATUS, NEEDS_EDIT_STATUS}


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
        "review_status": ACCEPTED_STATUS,
        "object_name": object_name,
        "class_name": class_name,
        "class_value": int(class_value),
        "instance_kind": "object",
        "source_image_layer": source_image_layer,
        "source_preview_layer": source_preview_layer,
        "source_task": source_task,
        "created_from": created_from,
    }


def review_metadata(
    *,
    status: str,
    object_name: str = "",
    class_name: str = "",
    class_value: int = 1,
    source_image_layer: str = "",
    source_preview_layer: str = "",
    created_from: str = "Mask Review",
) -> dict[str, Any]:
    status_key = normalize_review_status(status)
    role = ACCEPTED_ROLE if status_key == ACCEPTED_STATUS else REJECTED_ROLE if status_key == REJECTED_STATUS else "review_object"
    return {
        "sam3_role": role,
        "review_status": status_key,
        "object_name": object_name,
        "class_name": class_name,
        "class_value": int(class_value),
        "instance_kind": "object",
        "source_image_layer": source_image_layer,
        "source_preview_layer": source_preview_layer,
        "created_from": created_from,
    }


def normalize_review_status(status: str | None) -> str:
    key = str(status or "").strip().lower().replace(" ", "_")
    if key in REVIEW_STATUSES:
        return key
    return DEFAULT_REVIEW_STATUS


def layer_review_status(layer: Any) -> str:
    metadata = layer_metadata(layer)
    role = str(metadata.get("sam3_role") or "")
    if role == ACCEPTED_ROLE:
        return ACCEPTED_STATUS
    if role == REJECTED_ROLE:
        return REJECTED_STATUS
    return normalize_review_status(str(metadata.get("review_status") or DEFAULT_REVIEW_STATUS))


def layer_metadata(layer: Any) -> dict[str, Any]:
    metadata = getattr(layer, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def copy_labels_data(layer: Labels) -> np.ndarray:
    return np.asarray(layer.data).copy()


def copy_layer_geometry(source: Any, target: Any) -> None:
    for attr in ("scale", "translate", "rotate", "shear", "affine"):
        try:
            setattr(target, attr, getattr(source, attr))
        except Exception:
            pass
