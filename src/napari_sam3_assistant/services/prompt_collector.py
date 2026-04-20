from __future__ import annotations

from typing import Any

import numpy as np

from ..core.coordinates import base_image_data, extract_2d_image, infer_image_selection
from ..core.models import (
    BoxPrompt,
    ExemplarPrompt,
    MaskPrompt,
    PointPrompt,
    PromptBundle,
    PromptPolarity,
    Sam3Task,
    TextPrompt,
)


POSITIVE_VALUES = {"positive", "pos", "foreground", "fg", "1", "true", "include"}
NEGATIVE_VALUES = {"negative", "neg", "background", "bg", "0", "false", "exclude"}


class PromptCollector:
    """Translate napari layers into task-level prompt data models."""

    def collect(
        self,
        viewer: Any,
        *,
        image_layer_name: str,
        task: Sam3Task,
        points_layer_name: str | None = None,
        shapes_layer_name: str | None = None,
        labels_layer_name: str | None = None,
        text: str = "",
        channel_axis: int | None = None,
        collect_exemplar_rois: bool = True,
    ) -> PromptBundle:
        image_layer = viewer.layers[image_layer_name]
        image_data = base_image_data(image_layer.data)
        selection = infer_image_selection(
            layer_name=image_layer.name,
            data_shape=self._data_shape(image_data),
            dims_current_step=tuple(viewer.dims.current_step),
            channel_axis=channel_axis,
        )

        bundle = PromptBundle(task=task, image=selection)
        if text.strip():
            bundle.text = TextPrompt(text.strip())

        if points_layer_name:
            bundle.points.extend(
                self._collect_points(viewer.layers[points_layer_name], selection.frame_index)
            )
        if shapes_layer_name:
            boxes, exemplars = self._collect_shapes(
                viewer.layers[shapes_layer_name],
                image_data,
                selection.frame_index,
                task,
                collect_exemplar_rois=collect_exemplar_rois,
            )
            bundle.boxes.extend(boxes)
            bundle.exemplars.extend(exemplars)
        if labels_layer_name:
            mask = self._collect_mask(viewer.layers[labels_layer_name], selection.frame_index)
            if mask is not None:
                bundle.masks.append(mask)

        return bundle

    def _collect_points(self, layer: Any, frame_index: int | None) -> list[PointPrompt]:
        data = np.asarray(layer.data)
        if data.size == 0:
            return []

        polarities = self._layer_polarities(layer, len(data))
        points: list[PointPrompt] = []
        for row, polarity in zip(data, polarities):
            y, x = self._last_yx(row)
            points.append(
                PointPrompt(y=float(y), x=float(x), polarity=polarity, frame_index=frame_index)
            )
        return points

    def _collect_shapes(
        self,
        layer: Any,
        image_data: np.ndarray,
        frame_index: int | None,
        task: Sam3Task,
        *,
        collect_exemplar_rois: bool = True,
    ) -> tuple[list[BoxPrompt], list[ExemplarPrompt]]:
        boxes: list[BoxPrompt] = []
        exemplars: list[ExemplarPrompt] = []
        data = list(layer.data)

        for vertices in data:
            arr = np.asarray(vertices)
            if arr.size == 0:
                continue
            y0, x0, y1, x1 = self._bounds_yx(arr)
            box = BoxPrompt(
                y0=y0,
                x0=x0,
                y1=y1,
                x1=x1,
                frame_index=frame_index,
            )

            if task == Sam3Task.EXEMPLAR:
                # Local SAM3 exposes exemplar/visual prompting as positive boxes.
                boxes.append(box)
                if not collect_exemplar_rois:
                    continue
                roi = self._crop_exemplar(image_data, y0, x0, y1, x1)
                exemplars.append(
                    ExemplarPrompt(
                        roi=roi,
                        y0=y0,
                        x0=x0,
                        y1=y1,
                        x1=x1,
                        frame_index=frame_index,
                    )
                )
            else:
                boxes.append(box)
        return boxes, exemplars

    def _collect_mask(self, layer: Any, frame_index: int | None) -> MaskPrompt | None:
        data = np.asarray(layer.data)
        if data.size == 0:
            return None
        mask = data > 0
        while mask.ndim > 2:
            mask = mask[frame_index or 0] if mask.shape[0] > (frame_index or 0) else mask[0]
        if not mask.any():
            return None
        return MaskPrompt(mask=mask.astype(bool), frame_index=frame_index)

    def _layer_polarities(self, layer: Any, n: int) -> list[PromptPolarity]:
        properties = getattr(layer, "properties", {}) or {}
        for key in ("polarity", "prompt", "label", "class", "kind"):
            values = properties.get(key)
            if values is None:
                continue
            polarities = [self._parse_polarity(value) for value in list(values)[:n]]
            if len(polarities) < n:
                polarities.extend([PromptPolarity.POSITIVE] * (n - len(polarities)))
            return polarities
        return [PromptPolarity.POSITIVE] * n

    def _parse_polarity(self, value: Any) -> PromptPolarity:
        text = str(value).strip().lower()
        if text in NEGATIVE_VALUES:
            return PromptPolarity.NEGATIVE
        if text in POSITIVE_VALUES:
            return PromptPolarity.POSITIVE
        return PromptPolarity.POSITIVE

    def _last_yx(self, row: np.ndarray) -> tuple[float, float]:
        if row.shape[0] < 2:
            raise ValueError("Point prompts must have at least y and x coordinates.")
        return float(row[-2]), float(row[-1])

    def _bounds_yx(self, vertices: np.ndarray) -> tuple[float, float, float, float]:
        if vertices.shape[-1] < 2:
            raise ValueError("Shape prompts must have at least y and x coordinates.")
        y = vertices[..., -2]
        x = vertices[..., -1]
        return float(np.min(y)), float(np.min(x)), float(np.max(y)), float(np.max(x))

    def _crop_exemplar(
        self,
        image_data: np.ndarray,
        y0: float,
        x0: float,
        y1: float,
        x1: float,
    ) -> np.ndarray:
        normalized = base_image_data(image_data)
        selection = infer_image_selection("exemplar-source", self._data_shape(normalized))
        image = extract_2d_image(normalized, selection)
        height, width = image.shape[-2:]
        iy0 = max(0, min(height, int(np.floor(y0))))
        iy1 = max(iy0 + 1, min(height, int(np.ceil(y1))))
        ix0 = max(0, min(width, int(np.floor(x0))))
        ix1 = max(ix0 + 1, min(width, int(np.ceil(x1))))
        return np.asarray(image[iy0:iy1, ix0:ix1])

    def _data_shape(self, data: Any) -> tuple[int, ...]:
        shape = getattr(data, "shape", None)
        if shape is not None:
            return tuple(int(v) for v in shape)
        return tuple(int(v) for v in np.asarray(data).shape)
