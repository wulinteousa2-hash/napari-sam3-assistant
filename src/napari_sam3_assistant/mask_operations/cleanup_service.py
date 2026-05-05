from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .component_analysis_service import ComponentAnalysisService


class MaskCleanupService:
    def __init__(self) -> None:
        self.analysis = ComponentAnalysisService()

    def delete_components(self, data: Any, component_masks: list[np.ndarray]) -> np.ndarray:
        arr = np.asarray(data).copy()
        for mask in component_masks:
            if mask.shape != arr.shape:
                raise ValueError("Component mask shape does not match target layer shape.")
            arr[mask] = 0
        return arr

    def remove_small_objects(self, data: Any, min_size: int) -> np.ndarray:
        arr = np.asarray(data).copy()
        for record in self.analysis.analyze(arr):
            if record.area < int(min_size):
                mask = self.analysis.component_mask(record.component_id)
                if mask is not None:
                    arr[mask] = 0
        return arr

    def keep_largest_object(self, data: Any) -> np.ndarray:
        arr = np.asarray(data)
        records = self.analysis.analyze(arr)
        if not records:
            return arr.copy()
        largest = max(records, key=lambda record: record.area)
        out = np.zeros_like(arr)
        mask = self.analysis.component_mask(largest.component_id)
        if mask is not None:
            out[mask] = largest.label_value
        return out


    def delete_values(self, data: Any, values: list[int]) -> tuple[np.ndarray, int]:
        arr = np.asarray(data).copy()
        if not values:
            return arr, 0
        mask = np.isin(arr, np.asarray(values, dtype=arr.dtype))
        changed = int(np.count_nonzero(mask))
        arr[mask] = 0
        return arr, changed

    def keep_values(self, data: Any, values: list[int]) -> tuple[np.ndarray, int]:
        arr = np.asarray(data).copy()
        if not values:
            return arr, 0
        keep = np.isin(arr, np.asarray(values, dtype=arr.dtype)) | (arr == 0)
        changed = int(np.count_nonzero(~keep))
        arr[~keep] = 0
        return arr, changed

    def convert_nonzero_to_value(self, data: Any, target_value: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(data).copy()
        mask = arr != 0
        changed = int(np.count_nonzero(mask & (arr != int(target_value))))
        arr[mask] = int(target_value)
        return arr, changed

    def relabel_values(self, data: Any, source_values: list[int], target_value: int) -> tuple[np.ndarray, int]:
        arr = np.asarray(data).copy()
        if not source_values:
            return arr, 0
        mask = np.isin(arr, np.asarray(source_values, dtype=arr.dtype))
        changed = int(np.count_nonzero(mask))
        arr[mask] = int(target_value)
        return arr, changed

    def fill_holes(self, data: Any, max_hole_size: int) -> np.ndarray:
        arr = np.asarray(data).copy()
        for label_value in [int(v) for v in np.unique(arr) if int(v) != 0]:
            binary = arr == label_value
            filled = self._fill_binary_holes(binary, int(max_hole_size))
            arr[filled & ~binary] = label_value
        return arr

    def smooth(self, data: Any, radius: int) -> np.ndarray:
        arr = np.asarray(data).copy()
        iterations = max(1, int(radius))
        out = np.zeros_like(arr)
        for label_value in [int(v) for v in np.unique(arr) if int(v) != 0]:
            binary = arr == label_value
            smoothed = binary
            for _ in range(iterations):
                smoothed = self._binary_dilate(self._binary_erode(smoothed))
                smoothed = self._binary_erode(self._binary_dilate(smoothed))
            out[smoothed] = label_value
        return out

    def _fill_binary_holes(self, binary: np.ndarray, max_hole_size: int) -> np.ndarray:
        background = ~binary
        visited = np.zeros(binary.shape, dtype=bool)
        filled = binary.copy()
        for start in np.argwhere(background):
            start_tuple = tuple(int(v) for v in start)
            if visited[start_tuple]:
                continue
            coords, touches_border = self._flood_background(background, visited, start_tuple)
            if not touches_border and (max_hole_size <= 0 or len(coords) <= max_hole_size):
                filled[tuple(np.asarray(coords, dtype=np.intp).T)] = True
        return filled

    def _flood_background(
        self,
        background: np.ndarray,
        visited: np.ndarray,
        start: tuple[int, ...],
    ) -> tuple[list[tuple[int, ...]], bool]:
        queue: deque[tuple[int, ...]] = deque([start])
        visited[start] = True
        coords: list[tuple[int, ...]] = []
        touches_border = False
        while queue:
            point = queue.popleft()
            coords.append(point)
            if any(value == 0 or value + 1 == background.shape[axis] for axis, value in enumerate(point)):
                touches_border = True
            for neighbor in self._neighbors(point, background.shape):
                if visited[neighbor] or not background[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        return coords, touches_border

    def _binary_dilate(self, binary: np.ndarray) -> np.ndarray:
        out = binary.copy()
        for axis in range(binary.ndim):
            out |= np.roll(binary, 1, axis=axis)
            out |= np.roll(binary, -1, axis=axis)
        return self._clear_wrapped_edges(out, binary)

    def _binary_erode(self, binary: np.ndarray) -> np.ndarray:
        out = binary.copy()
        for axis in range(binary.ndim):
            out &= np.roll(binary, 1, axis=axis)
            out &= np.roll(binary, -1, axis=axis)
        return self._clear_wrapped_edges(out, binary)

    def _clear_wrapped_edges(self, out: np.ndarray, original: np.ndarray) -> np.ndarray:
        result = out.copy()
        for axis in range(original.ndim):
            head = [slice(None)] * original.ndim
            tail = [slice(None)] * original.ndim
            head[axis] = 0
            tail[axis] = -1
            result[tuple(head)] &= original[tuple(head)]
            result[tuple(tail)] &= original[tuple(tail)]
        return result

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
