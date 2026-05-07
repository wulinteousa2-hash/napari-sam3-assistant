from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .models import CandidateObjectRecord
from .utils import safe_get_layer


class CandidateConsolidationService:
    """Harvest and consolidate globally unique mask candidates across layers."""

    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.records: list[CandidateObjectRecord] = []
        self.candidate_masks: dict[int, np.ndarray] = {}
        self.source_layer_names: list[str] = []
        self.shape: tuple[int, ...] | None = None
        self.reference_layer_name = ""
        self.duplicate_iou_threshold = 0.85
        self.nested_containment_threshold = 0.90
        self.sort_rule = "area_small_to_large"

    def harvest_layers(
        self,
        layer_names: list[str],
        min_area: int = 64,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[CandidateObjectRecord]:
        self.records = []
        self.candidate_masks = {}
        self.source_layer_names = []
        self.shape = None
        self.reference_layer_name = ""

        layer_infos = []
        for layer_name in layer_names:
            layer = safe_get_layer(self.viewer, layer_name)
            if layer is not None:
                data = np.asarray(layer.data)
                label_values = [int(value) for value in np.unique(data) if int(value) != 0]
                layer_infos.append((layer, data, label_values))
        expected_shape = None
        for layer, data, _label_values in layer_infos:
            data_shape = tuple(data.shape)
            if expected_shape is None:
                expected_shape = data_shape
            elif data_shape != expected_shape:
                raise ValueError(f"Layer shape mismatch: {layer.name} has {data_shape}, expected {expected_shape}.")

        total_steps = sum(max(1, len(label_values)) for _layer, _data, label_values in layer_infos)
        completed_steps = 0
        if progress_callback is not None:
            progress_callback(completed_steps, total_steps, "Preparing mask harvest...")
        next_candidate_id = 1
        for layer, data, label_values in layer_infos:
            if self.shape is None:
                self.shape = tuple(data.shape)
                self.reference_layer_name = str(layer.name)

            records, next_candidate_id = self._harvest_layer_fast(
                data,
                str(layer.name),
                label_values=label_values,
                min_area=int(min_area),
                next_candidate_id=next_candidate_id,
                progress_callback=progress_callback,
                completed_steps=completed_steps,
                total_steps=total_steps,
            )
            completed_steps += max(1, len(label_values))
            self.records.extend(records)

            self.source_layer_names.append(str(layer.name))
        if progress_callback is not None:
            progress_callback(total_steps, total_steps, "Mask harvest complete.")
        return self.records

    def classify_candidates(
        self,
        duplicate_iou: float = 0.85,
        nested_containment: float = 0.90,
        fragment_iou_min: float = 0.15,
        area_ratio_threshold: float = 0.75,
    ) -> list[CandidateObjectRecord]:
        self.duplicate_iou_threshold = float(duplicate_iou)
        self.nested_containment_threshold = float(nested_containment)
        for record in self.records:
            record.status = "keep"
            record.duplicate_of = None
            record.parent_candidate_id = None
            record.child_count = 0
            record.max_iou = 0.0
            record.max_containment = 0.0
            record.overlap_pixels = 0
            record.output_object_id = None

        pair_stats: list[tuple[CandidateObjectRecord, CandidateObjectRecord, float, float, float, float]] = []
        for index, a in enumerate(self.records):
            mask_a = self.candidate_masks.get(a.candidate_id)
            if mask_a is None:
                continue
            for b in self.records[index + 1 :]:
                if not self._bbox_intersects(a.bbox, b.bbox):
                    continue
                mask_b = self.candidate_masks.get(b.candidate_id)
                if mask_b is None:
                    continue
                bbox = self._intersection_bbox(a.bbox, b.bbox)
                if bbox is None:
                    continue
                sub_a = mask_a[self._local_slices(a.bbox, bbox)]
                sub_b = mask_b[self._local_slices(b.bbox, bbox)]
                intersection = int(np.count_nonzero(sub_a & sub_b))
                if intersection == 0:
                    continue
                union = int(a.area + b.area - intersection)
                iou = float(intersection / union) if union else 0.0
                containment_a = float(intersection / a.area) if a.area else 0.0
                containment_b = float(intersection / b.area) if b.area else 0.0
                area_ratio = min(a.area, b.area) / max(a.area, b.area)
                a.max_iou = max(a.max_iou, iou)
                b.max_iou = max(b.max_iou, iou)
                a.max_containment = max(a.max_containment, containment_a)
                b.max_containment = max(b.max_containment, containment_b)
                a.overlap_pixels += intersection
                b.overlap_pixels += intersection
                pair_stats.append((a, b, iou, containment_a, containment_b, float(area_ratio)))

        duplicate_pairs = sorted(pair_stats, key=lambda item: item[2], reverse=True)
        for a, b, iou, _containment_a, _containment_b, area_ratio in duplicate_pairs:
            if iou < duplicate_iou or area_ratio < area_ratio_threshold:
                continue
            if a.status in {"duplicate", "reject"} or b.status in {"duplicate", "reject"}:
                continue
            kept, duplicate = (a, b) if a.area <= b.area else (b, a)
            duplicate.status = "duplicate"
            duplicate.duplicate_of = kept.candidate_id

        nested_pairs: set[tuple[int, int]] = set()
        for a, b, _iou, containment_a, containment_b, _area_ratio in pair_stats:
            if a.status in {"duplicate", "reject"} or b.status in {"duplicate", "reject"}:
                continue
            smaller, larger, containment = (a, b, containment_a) if a.area <= b.area else (b, a, containment_b)
            if containment < nested_containment:
                continue
            smaller.status = "nested_child"
            smaller.parent_candidate_id = larger.candidate_id
            if larger.status not in {"duplicate", "reject"}:
                larger.status = "parent"
            larger.child_count += 1
            nested_pairs.add((smaller.candidate_id, larger.candidate_id))

        for a, b, iou, _containment_a, _containment_b, area_ratio in pair_stats:
            if iou < fragment_iou_min or iou >= duplicate_iou or area_ratio >= 0.50:
                continue
            smaller, larger = (a, b) if a.area <= b.area else (b, a)
            if (smaller.candidate_id, larger.candidate_id) in nested_pairs:
                continue
            if smaller.status not in {"duplicate", "nested_child", "parent", "reject"}:
                smaller.status = "fragment"
        return self.records

    def create_isolated_label_data(
        self,
        include_statuses: set[str],
        overlap_rule: str,
        parent_handling: str,
    ) -> tuple[np.ndarray, dict[int, int]]:
        if self.shape is None:
            return np.zeros((0,), dtype=np.uint32), {}
        out = np.zeros(self.shape, dtype=np.uint32)
        blocked = np.zeros(self.shape, dtype=bool)
        candidate_to_output_id: dict[int, int] = {}
        for record in self.records:
            record.output_object_id = None
        included = [record for record in self.records if record.status in include_statuses]
        included = self._sort_records_for_output(included)
        next_output_id = 1
        for record in included:
            mask = self.candidate_masks.get(record.candidate_id)
            if mask is None:
                continue
            slices = self._bbox_slices(record.bbox)
            if parent_handling == "parent_overwrites_children" and record.status == "parent":
                write_mask = mask
            elif overlap_rule == "later_wins":
                write_mask = mask & ~blocked[slices]
            elif overlap_rule == "set_conflict_background":
                conflict = mask & (out[slices] != 0)
                write_mask = mask & (out[slices] == 0) & ~blocked[slices]
                out_slice = out[slices]
                blocked_slice = blocked[slices]
                out_slice[conflict] = 0
                blocked_slice[conflict] = True
            else:
                write_mask = mask & (out[slices] == 0) & ~blocked[slices]
            if not np.any(write_mask):
                continue
            output_id = next_output_id
            next_output_id += 1
            out_slice = out[slices]
            out_slice[write_mask] = output_id
            record.output_object_id = int(output_id)
            candidate_to_output_id[record.candidate_id] = int(output_id)
        return out, candidate_to_output_id

    def candidate_mask(self, candidate_id: int) -> np.ndarray | None:
        local_mask = self.candidate_masks.get(int(candidate_id))
        if local_mask is None or self.shape is None:
            return None
        record = self._record(int(candidate_id))
        if record is None:
            return None
        mask = np.zeros(self.shape, dtype=bool)
        mask[self._bbox_slices(record.bbox)] = local_mask
        return mask

    def selected_candidates_label_data(self, candidate_ids: list[int]) -> np.ndarray:
        if self.shape is None:
            return np.zeros((0,), dtype=np.uint32)
        out = np.zeros(self.shape, dtype=np.uint32)
        for output_id, candidate_id in enumerate(candidate_ids, start=1):
            record = self._record(int(candidate_id))
            mask = self.candidate_masks.get(int(candidate_id))
            if record is not None and mask is not None:
                slices = self._bbox_slices(record.bbox)
                out_slice = out[slices]
                out_slice[mask & (out_slice == 0)] = int(output_id)
        return out

    def set_status(self, candidate_ids: list[int], status: str) -> None:
        allowed = {"keep", "duplicate", "fragment", "nested_child", "parent", "reject"}
        if status not in allowed:
            return
        ids = {int(candidate_id) for candidate_id in candidate_ids}
        for record in self.records:
            if record.candidate_id in ids:
                record.status = status

    def output_metadata(self, include_statuses: set[str], candidate_to_output_id: dict[int, int], overlap_rule: str, parent_handling: str) -> dict[str, Any]:
        return {
            "sam3_role": "isolated_candidate_objects",
            "created_from": "Mask Isolation",
            "source_layers": list(self.source_layer_names),
            "candidate_count": len(self.records),
            "included_statuses": sorted(include_statuses),
            "candidate_to_output_id": {int(k): int(v) for k, v in candidate_to_output_id.items()},
            "candidate_source_keys": {
                int(record.candidate_id): {
                    "source_layer_name": record.source_layer_name,
                    "source_label_value": int(record.source_label_value),
                    "source_component_id": int(record.source_component_id),
                    "status": record.status,
                    "duplicate_of": record.duplicate_of,
                    "parent_candidate_id": record.parent_candidate_id,
                    "child_count": int(record.child_count),
                    "output_object_id": record.output_object_id,
                }
                for record in self.records
            },
            "duplicate_iou_threshold": float(self.duplicate_iou_threshold),
            "nested_containment_threshold": float(self.nested_containment_threshold),
            "overlap_rule": overlap_rule,
            "parent_handling": parent_handling,
        }

    def _sort_records_for_output(self, records: list[CandidateObjectRecord]) -> list[CandidateObjectRecord]:
        reverse = self.sort_rule == "area_large_to_small"
        return sorted(records, key=lambda record: (record.area, record.candidate_id), reverse=reverse)

    def _harvest_layer_fast(
        self,
        data: np.ndarray,
        layer_name: str,
        *,
        label_values: list[int],
        min_area: int,
        next_candidate_id: int,
        progress_callback: Callable[[int, int, str], None] | None,
        completed_steps: int,
        total_steps: int,
    ) -> tuple[list[CandidateObjectRecord], int]:
        records: list[CandidateObjectRecord] = []
        structure = ndi.generate_binary_structure(data.ndim, 1)
        source_component_id = 1
        if not label_values and progress_callback is not None:
            progress_callback(completed_steps + 1, total_steps, f"No masks found in {layer_name}.")
        for label_offset, label_value in enumerate(label_values, start=1):
            if progress_callback is not None:
                progress_callback(
                    completed_steps + label_offset - 1,
                    total_steps,
                    f"Harvesting {layer_name}, label {label_value}...",
                )
            components, component_count = ndi.label(data == label_value, structure=structure)
            if component_count == 0:
                if progress_callback is not None:
                    progress_callback(
                        completed_steps + label_offset,
                        total_steps,
                        f"Finished {layer_name}, label {label_value}.",
                    )
                continue
            for component_index, component_slice in enumerate(ndi.find_objects(components), start=1):
                if component_slice is None:
                    continue
                local_component = components[component_slice] == component_index
                area = int(np.count_nonzero(local_component))
                if area < min_area:
                    source_component_id += 1
                    continue
                local_coords = np.argwhere(local_component)
                starts = np.asarray([selector.start or 0 for selector in component_slice], dtype=np.int64)
                global_coords = local_coords + starts
                bbox = tuple(
                    (int(selector.start or 0), int(selector.stop or 0))
                    for selector in component_slice
                )
                records.append(
                    CandidateObjectRecord(
                        candidate_id=next_candidate_id,
                        source_layer_name=layer_name,
                        source_label_value=int(label_value),
                        source_component_id=int(source_component_id),
                        area=area,
                        bbox=bbox,
                        centroid=tuple(float(value) for value in global_coords.mean(axis=0)),
                    )
                )
                self.candidate_masks[next_candidate_id] = local_component.astype(bool, copy=True)
                next_candidate_id += 1
                source_component_id += 1
            if progress_callback is not None:
                progress_callback(
                    completed_steps + label_offset,
                    total_steps,
                    f"Finished {layer_name}, label {label_value}.",
                )
        return records, next_candidate_id

    def _record(self, candidate_id: int) -> CandidateObjectRecord | None:
        for record in self.records:
            if record.candidate_id == int(candidate_id):
                return record
        return None

    def _bbox_intersects(self, a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
        return all(a_lo < b_hi and b_lo < a_hi for (a_lo, a_hi), (b_lo, b_hi) in zip(a, b, strict=False))

    def _intersection_bbox(
        self, a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]
    ) -> tuple[tuple[int, int], ...] | None:
        bbox = tuple((max(a_lo, b_lo), min(a_hi, b_hi)) for (a_lo, a_hi), (b_lo, b_hi) in zip(a, b, strict=False))
        if any(lo >= hi for lo, hi in bbox):
            return None
        return bbox

    def _bbox_slices(self, bbox: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
        return tuple(slice(lo, hi) for lo, hi in bbox)

    def _local_slices(
        self,
        source_bbox: tuple[tuple[int, int], ...],
        target_bbox: tuple[tuple[int, int], ...],
    ) -> tuple[slice, ...]:
        return tuple(
            slice(target_lo - source_lo, target_hi - source_lo)
            for (source_lo, _source_hi), (target_lo, target_hi) in zip(source_bbox, target_bbox, strict=False)
        )
