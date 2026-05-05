from __future__ import annotations

from typing import Any

import numpy as np

from .models import AcceptedObjectRecord, ReviewObjectRecord
from .utils import ACCEPTED_ROLE, layer_metadata, layer_review_status, labels_layer_names, normalize_review_status, safe_get_layer


class AcceptedObjectRegistry:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer

    def is_accepted_layer(self, layer: Any) -> bool:
        metadata = layer_metadata(layer)
        return metadata.get("sam3_role") == ACCEPTED_ROLE and layer_review_status(layer) == "accepted"

    def records(self) -> list[AcceptedObjectRecord]:
        if self.viewer is None:
            return []
        records: list[AcceptedObjectRecord] = []
        for layer in self.viewer.layers:
            if not self.is_accepted_layer(layer):
                continue
            metadata = layer_metadata(layer)
            records.append(
                AcceptedObjectRecord(
                    layer_name=layer.name,
                    object_name=str(metadata.get("object_name") or layer.name),
                    class_name=str(metadata.get("class_name") or ""),
                    class_value=int(metadata.get("class_value") or 1),
                    source_image_layer=str(metadata.get("source_image_layer") or ""),
                    source_preview_layer=str(metadata.get("source_preview_layer") or ""),
                    review_status=layer_review_status(layer),
                )
            )
        return records

    def review_records(self, *, include_final_outputs: bool = False) -> list[ReviewObjectRecord]:
        if self.viewer is None:
            return []
        records: list[ReviewObjectRecord] = []
        output_roles = {"class_working_mask", "final_training_mask", "overlap_map"}
        for name in labels_layer_names(self.viewer):
            layer = safe_get_layer(self.viewer, name)
            if layer is None:
                continue
            metadata = layer_metadata(layer)
            role = str(metadata.get("sam3_role") or "")
            if not include_final_outputs and role in output_roles:
                continue
            arr = np.asarray(layer.data)
            records.append(
                ReviewObjectRecord(
                    layer_name=layer.name,
                    object_name=str(metadata.get("object_name") or layer.name),
                    class_name=str(metadata.get("class_name") or ""),
                    class_value=int(metadata.get("class_value") or 1),
                    review_status=layer_review_status(layer),
                    sam3_role=role,
                    ndim=int(arr.ndim),
                    shape_text=" × ".join(str(v) for v in arr.shape),
                    nonzero_count=int(np.count_nonzero(arr)),
                )
            )
        return records

    def layer_names(
        self,
        *,
        class_name: str = "",
        class_value: int | None = None,
        status: str = "accepted",
    ) -> list[str]:
        filtered = []
        class_key = class_name.strip().lower()
        status_key = normalize_review_status(status) if status else ""
        for record in self.records():
            if status_key and record.review_status != status_key:
                continue
            if class_key and record.class_name.lower() != class_key:
                continue
            if class_value is not None and record.class_value != class_value:
                continue
            filtered.append(record.layer_name)
        return filtered

    def metadata_for_layer(self, layer: Any) -> dict[str, Any]:
        return dict(layer_metadata(layer))

    def write_metadata(self, layer: Any, metadata: dict[str, Any]) -> None:
        existing = dict(layer_metadata(layer))
        existing.update(metadata)
        layer.metadata = existing

    def set_review_status(
        self,
        layer: Any,
        *,
        status: str,
        object_name: str | None = None,
        class_name: str | None = None,
        class_value: int | None = None,
    ) -> None:
        status_key = normalize_review_status(status)
        metadata = dict(layer_metadata(layer))
        if object_name is not None and object_name.strip():
            metadata["object_name"] = object_name.strip()
        elif not metadata.get("object_name"):
            metadata["object_name"] = layer.name
        if class_name is not None and class_name.strip():
            metadata["class_name"] = class_name.strip()
        if class_value is not None:
            metadata["class_value"] = int(class_value)
        elif not metadata.get("class_value"):
            metadata["class_value"] = 1
        metadata["review_status"] = status_key
        if status_key == "accepted":
            metadata["sam3_role"] = ACCEPTED_ROLE
        elif status_key == "rejected":
            metadata["sam3_role"] = "rejected_object"
        else:
            metadata["sam3_role"] = "review_object"
        metadata.setdefault("instance_kind", "object")
        metadata.setdefault("created_from", "Mask Review")
        layer.metadata = metadata
