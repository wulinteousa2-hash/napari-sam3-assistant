from __future__ import annotations

from typing import Any

from .models import AcceptedObjectRecord
from .utils import ACCEPTED_ROLE, layer_metadata


class AcceptedObjectRegistry:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer

    def is_accepted_layer(self, layer: Any) -> bool:
        return layer_metadata(layer).get("sam3_role") == ACCEPTED_ROLE

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
                )
            )
        return records

    def layer_names(self, *, class_name: str = "", class_value: int | None = None) -> list[str]:
        filtered = []
        class_key = class_name.strip().lower()
        for record in self.records():
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
