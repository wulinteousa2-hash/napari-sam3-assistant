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
    review_status: str = "accepted"


@dataclass(frozen=True)
class ReviewObjectRecord:
    layer_name: str
    object_name: str
    class_name: str
    class_value: int
    review_status: str
    sam3_role: str
    ndim: int
    shape_text: str
    nonzero_count: int


@dataclass(frozen=True)
class ComponentRecord:
    component_id: int
    label_value: int
    area: int
    centroid_y: float
    centroid_x: float
    bbox: tuple[tuple[int, int], ...]
    centroid_z: float | None = None
    z_min: int | None = None
    z_max: int | None = None
    ndim: int = 2

    @property
    def bbox_text(self) -> str:
        return ", ".join(f"{lo}:{hi}" for lo, hi in self.bbox)

    @property
    def z_range_text(self) -> str:
        if self.z_min is None or self.z_max is None:
            return "—"
        return f"{self.z_min}:{self.z_max}"


@dataclass(frozen=True)
class AxonHoleCandidate:
    candidate_id: int
    label_value: int
    object_area: int
    axon_area: int
    axon_fraction: float
    confidence: float
    status: str
    reason: str
    seed: tuple[int, ...]
    bbox: tuple[tuple[int, int], ...]

    @property
    def bbox_text(self) -> str:
        return ", ".join(f"{lo}:{hi}" for lo, hi in self.bbox)


@dataclass
class CandidateObjectRecord:
    candidate_id: int
    source_layer_name: str
    source_label_value: int
    source_component_id: int
    area: int
    bbox: tuple[tuple[int, int], ...]
    centroid: tuple[float, ...]
    status: str = "keep"
    duplicate_of: int | None = None
    parent_candidate_id: int | None = None
    child_count: int = 0
    max_iou: float = 0.0
    max_containment: float = 0.0
    overlap_pixels: int = 0
    output_object_id: int | None = None

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
