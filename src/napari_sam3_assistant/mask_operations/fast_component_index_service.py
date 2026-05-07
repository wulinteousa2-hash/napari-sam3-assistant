from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .models import ComponentRecord


@dataclass
class FastComponentIndex:
    """In-memory component lookup for one scoped Labels array.

    This index is built once after Analyze/Rebuild and then used for O(1)
    canvas picking:

        component_id = component_id_map[scoped_coords]

    The component map uses the same shape as the scoped mask. Component deletes
    are applied by bbox so the whole image does not need to be scanned again.
    """

    shape: tuple[int, ...]
    component_id_map: np.ndarray
    records: dict[int, ComponentRecord] = field(default_factory=dict)
    deleted_component_ids: set[int] = field(default_factory=set)

    def active_records(self) -> list[ComponentRecord]:
        return [
            record
            for component_id, record in sorted(self.records.items())
            if component_id not in self.deleted_component_ids
        ]

    def component_id_at(self, scoped_coords: tuple[int, ...]) -> int | None:
        if len(scoped_coords) != len(self.shape):
            return None
        for coord, size in zip(scoped_coords, self.shape, strict=False):
            if coord < 0 or coord >= size:
                return None
        component_id = int(self.component_id_map[scoped_coords])
        if component_id <= 0 or component_id in self.deleted_component_ids:
            return None
        return component_id

    def record_at(self, scoped_coords: tuple[int, ...]) -> ComponentRecord | None:
        component_id = self.component_id_at(scoped_coords)
        if component_id is None:
            return None
        return self.records.get(component_id)

    def record(self, component_id: int) -> ComponentRecord | None:
        component_id = int(component_id)
        if component_id in self.deleted_component_ids:
            return None
        return self.records.get(component_id)

    def component_mask(self, component_id: int) -> np.ndarray | None:
        record = self.record(component_id)
        if record is None:
            return None
        slices = self._bbox_slices(record.bbox)
        return self.component_id_map[slices] == int(component_id)

    def delete_components(self, scoped_data: np.ndarray, component_ids: list[int]) -> tuple[np.ndarray, int]:
        out = np.asarray(scoped_data).copy()
        changed = 0
        for component_id in [int(value) for value in component_ids]:
            record = self.record(component_id)
            if record is None:
                continue
            slices = self._bbox_slices(record.bbox)
            local_component_map = self.component_id_map[slices]
            mask = local_component_map == component_id
            count = int(np.count_nonzero(mask))
            if count == 0:
                self.deleted_component_ids.add(component_id)
                continue
            out[slices][mask] = 0
            local_component_map[mask] = 0
            self.deleted_component_ids.add(component_id)
            changed += count
        return out, changed

    def delete_label_value(self, scoped_data: np.ndarray, label_value: int) -> tuple[np.ndarray, int]:
        out = np.asarray(scoped_data).copy()
        mask = out == int(label_value)
        changed = int(np.count_nonzero(mask))
        if changed == 0:
            return out, 0
        out[mask] = 0
        for component_id, record in self.records.items():
            if int(record.label_value) == int(label_value):
                self.deleted_component_ids.add(component_id)
        self.component_id_map[mask] = 0
        return out, changed

    def relabel_components(
        self,
        scoped_data: np.ndarray,
        component_ids: list[int],
        new_label_value: int,
    ) -> tuple[np.ndarray, int]:
        out = np.asarray(scoped_data).copy()
        changed = 0
        new_value = int(new_label_value)
        for component_id in [int(value) for value in component_ids]:
            record = self.record(component_id)
            if record is None:
                continue
            slices = self._bbox_slices(record.bbox)
            mask = self.component_id_map[slices] == component_id
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            out[slices][mask] = new_value
            self.records[component_id] = replace(record, label_value=new_value)
            changed += count
        return out, changed

    def _bbox_slices(self, bbox: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
        return tuple(slice(int(lo), int(hi)) for lo, hi in bbox)


class FastComponentIndexService:
    """Builds a fast click/pick index for 2D/3D/nD Labels data.

    Label value 0 is background. Non-zero label values are separated into
    connected components using face connectivity. This intentionally mirrors the
    existing ComponentAnalysisService behavior but stores a dense component-id
    lookup map so repeated mouse clicks avoid repeated flood-fill analysis.
    """

    def build(self, data: Any) -> FastComponentIndex:
        arr = np.asarray(data)
        component_id_map = np.zeros(arr.shape, dtype=np.int32)
        records: dict[int, ComponentRecord] = {}
        if arr.size == 0:
            return FastComponentIndex(shape=tuple(arr.shape), component_id_map=component_id_map, records=records)

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
                component_id_map[tuple(coords.T)] = component_id
                records[component_id] = self._record(component_id, label_value, coords, arr.ndim)
                component_id += 1
        return FastComponentIndex(shape=tuple(arr.shape), component_id_map=component_id_map, records=records)

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
