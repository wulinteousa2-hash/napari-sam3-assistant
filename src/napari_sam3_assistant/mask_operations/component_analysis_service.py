from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .models import ComponentRecord


class ComponentAnalysisService:
    """Connected-component analysis for 2D or nD napari Labels data.

    Label value 0 is treated as background. Every other integer value is treated
    as a candidate semantic class, instance id, or temporary SAM3 object id.
    """

    def __init__(self) -> None:
        self._component_masks: dict[int, np.ndarray] = {}

    def analyze(self, data: Any) -> list[ComponentRecord]:
        arr = np.asarray(data)
        self._component_masks = {}
        if arr.size == 0:
            return []
        records: list[ComponentRecord] = []
        visited = np.zeros(arr.shape, dtype=bool)
        component_id = 1
        for label_value in [int(v) for v in np.unique(arr) if int(v) != 0]:
            positions = np.argwhere((arr == label_value) & ~visited)
            for start in positions:
                start_tuple = tuple(int(v) for v in start)
                if visited[start_tuple] or int(arr[start_tuple]) != label_value:
                    continue
                coords = self._flood_component(arr, visited, start_tuple, label_value)
                if coords.size == 0:
                    continue
                mask = np.zeros(arr.shape, dtype=bool)
                mask[tuple(coords.T)] = True
                self._component_masks[component_id] = mask
                records.append(self._record(component_id, label_value, coords, arr.ndim))
                component_id += 1
        return records

    def component_mask(self, component_id: int) -> np.ndarray | None:
        return self._component_masks.get(component_id)

    def component_masks(self, component_ids: list[int]) -> list[np.ndarray]:
        return [mask for cid in component_ids if (mask := self.component_mask(cid)) is not None]

    def _record(self, component_id: int, label_value: int, coords: np.ndarray, ndim: int) -> ComponentRecord:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        centroid = coords.mean(axis=0)
        y_axis = -2 if coords.shape[1] >= 2 else 0
        x_axis = -1
        z_axis = coords.shape[1] - 3 if coords.shape[1] >= 3 else None
        return ComponentRecord(
            component_id=component_id,
            label_value=label_value,
            area=int(coords.shape[0]),
            centroid_y=float(centroid[y_axis]),
            centroid_x=float(centroid[x_axis]),
            centroid_z=float(centroid[z_axis]) if z_axis is not None else None,
            z_min=int(mins[z_axis]) if z_axis is not None else None,
            z_max=int(maxs[z_axis] - 1) if z_axis is not None else None,
            bbox=tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs, strict=False)),
            ndim=int(ndim),
        )

    def _flood_component(
        self,
        arr: np.ndarray,
        visited: np.ndarray,
        start: tuple[int, ...],
        label_value: int,
    ) -> np.ndarray:
        queue: deque[tuple[int, ...]] = deque([start])
        visited[start] = True
        coords: list[tuple[int, ...]] = []
        while queue:
            point = queue.popleft()
            coords.append(point)
            for neighbor in self._neighbors(point, arr.shape):
                if visited[neighbor] or int(arr[neighbor]) != label_value:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        return np.asarray(coords, dtype=np.intp)

    def _neighbors(self, point: tuple[int, ...], shape: tuple[int, ...]):
        for axis, value in enumerate(point):
            if value > 0:
                neighbor = list(point)
                neighbor[axis] -= 1
                yield tuple(neighbor)
            if value + 1 < shape[axis]:
                neighbor = list(point)
                neighbor[axis] += 1
                yield tuple(neighbor)
