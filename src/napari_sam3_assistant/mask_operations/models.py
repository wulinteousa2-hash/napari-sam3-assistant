from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceptedObjectRecord:
    layer_name: str
    object_name: str
    class_name: str
    class_value: int
    source_image_layer: str = ""
    source_preview_layer: str = ""


@dataclass(frozen=True)
class ComponentRecord:
    component_id: int
    label_value: int
    area: int
    centroid_y: float
    centroid_x: float
    bbox: tuple[tuple[int, int], ...]

    @property
    def bbox_text(self) -> str:
        return ", ".join(f"{lo}:{hi}" for lo, hi in self.bbox)


@dataclass(frozen=True)
class MergeOptions:
    mode: str = "semantic"
    overlap_rule: str = "class_priority"


@dataclass(frozen=True)
class CleanupOptions:
    min_size: int = 16
    hole_size: int = 64
    smoothing_radius: int = 1
