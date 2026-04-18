from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Sam3Task(str, Enum):
    SEGMENT_2D = "2d_segmentation"
    SEGMENT_3D = "3d_video_propagation"
    EXEMPLAR = "exemplar_segmentation"
    TEXT = "text_segmentation"
    REFINE = "refinement"


class PromptPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class OutputKind(str, Enum):
    LABELS = "labels"
    IMAGE = "image"
    SHAPES = "shapes"
    TABLE = "table"


@dataclass(frozen=True)
class ImageSelection:
    layer_name: str
    data_shape: tuple[int, ...]
    frame_axis: int | None
    channel_axis: int | None
    spatial_axes: tuple[int, int]
    frame_index: int | None = None
    channel_index: int | None = None


@dataclass(frozen=True)
class PointPrompt:
    y: float
    x: float
    polarity: PromptPolarity = PromptPolarity.POSITIVE
    frame_index: int | None = None
    object_id: int | None = None

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BoxPrompt:
    y0: float
    x0: float
    y1: float
    x1: float
    polarity: PromptPolarity = PromptPolarity.POSITIVE
    frame_index: int | None = None
    object_id: int | None = None

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def xywh_normalized(self) -> tuple[float, float, float, float]:
        raise RuntimeError("Use CoordinateMapper.box_to_normalized_xywh with image size.")


@dataclass(frozen=True)
class MaskPrompt:
    mask: np.ndarray
    frame_index: int | None = None
    object_id: int | None = None


@dataclass(frozen=True)
class ExemplarPrompt:
    roi: np.ndarray
    y0: float
    x0: float
    y1: float
    x1: float
    frame_index: int | None = None
    label: str | None = None


@dataclass(frozen=True)
class TextPrompt:
    text: str


@dataclass
class PromptBundle:
    task: Sam3Task
    image: ImageSelection
    points: list[PointPrompt] = field(default_factory=list)
    boxes: list[BoxPrompt] = field(default_factory=list)
    masks: list[MaskPrompt] = field(default_factory=list)
    exemplars: list[ExemplarPrompt] = field(default_factory=list)
    text: TextPrompt | None = None

    def has_prompt(self) -> bool:
        return bool(
            self.points
            or self.boxes
            or self.masks
            or self.exemplars
            or (self.text and self.text.text.strip())
        )


@dataclass
class Sam3Session:
    task: Sam3Task
    image: ImageSelection
    session_id: str | None = None
    resource_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sam3Result:
    task: Sam3Task
    frame_index: int | None = None
    masks: np.ndarray | None = None
    labels: np.ndarray | None = None
    boxes_xyxy: np.ndarray | None = None
    scores: np.ndarray | None = None
    object_ids: np.ndarray | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (
            self.masks is None
            and self.labels is None
            and self.boxes_xyxy is None
            and self.object_ids is None
        )
