from __future__ import annotations

from collections.abc import Callable
from collections import deque
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .component_analysis_service import ComponentAnalysisService
from .models import AxonHoleCandidate


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

    def connected_label_mask(self, data: Any, seed: tuple[int, ...], label_value: int) -> np.ndarray:
        arr = np.asarray(data)
        if len(seed) != arr.ndim or any(coord < 0 or coord >= size for coord, size in zip(seed, arr.shape, strict=False)):
            raise ValueError("Seed is outside the target labels layer.")
        if int(arr[seed]) != int(label_value):
            raise ValueError("Seed does not match the clicked label value.")
        mask = np.zeros(arr.shape, dtype=bool)
        queue: deque[tuple[int, ...]] = deque([seed])
        mask[seed] = True
        while queue:
            point = queue.popleft()
            for neighbor in self._neighbors(point, arr.shape):
                if mask[neighbor] or int(arr[neighbor]) != int(label_value):
                    continue
                mask[neighbor] = True
                queue.append(neighbor)
        return mask

    def cut_seed_similar_region(
        self,
        labels: Any,
        image: Any,
        component_mask: np.ndarray,
        seed: tuple[int, ...],
        *,
        output_value: int,
        threshold_percent: int,
        max_fraction_percent: int,
    ) -> tuple[np.ndarray, int, int]:
        arr = np.asarray(labels).copy()
        component = np.asarray(component_mask, dtype=bool)
        if arr.shape != component.shape:
            raise ValueError("Component mask shape does not match target labels shape.")
        if len(seed) != arr.ndim or any(coord < 0 or coord >= size for coord, size in zip(seed, arr.shape, strict=False)):
            raise ValueError("Axon seed is outside the target labels layer.")
        if not component[seed]:
            raise ValueError("Axon seed must be inside the clicked SAM3 mask component.")

        gray = self._coerce_grayscale_image(image, arr.shape)
        candidate = self._similar_candidate_from_seed(gray, component, seed, int(threshold_percent))
        axon_mask = self._connected_seed_region(candidate, seed)
        axon_pixels = int(np.count_nonzero(axon_mask))
        component_pixels = int(np.count_nonzero(component))
        if axon_pixels == 0:
            return arr, 0, component_pixels
        max_pixels = max(1, int(round(component_pixels * max(1, int(max_fraction_percent)) / 100.0)))
        if axon_pixels > max_pixels:
            raise ValueError(
                f"Detected axon region is {axon_pixels} pixels/voxels, above the "
                f"{int(max_fraction_percent)}% limit for the clicked component."
            )

        changed_mask = axon_mask & (arr != int(output_value))
        changed = int(np.count_nonzero(changed_mask))
        arr[axon_mask] = int(output_value)
        return arr, changed, axon_pixels

    def cut_dark_region_from_seed(
        self,
        labels: Any,
        image: Any,
        component_mask: np.ndarray,
        seed: tuple[int, ...],
        *,
        output_value: int,
        threshold_percent: int,
        max_fraction_percent: int,
    ) -> tuple[np.ndarray, int, int]:
        return self.cut_seed_similar_region(
            labels,
            image,
            component_mask,
            seed,
            output_value=output_value,
            threshold_percent=threshold_percent,
            max_fraction_percent=max_fraction_percent,
        )

    def propose_axon_holes(
        self,
        labels: Any,
        image: Any,
        *,
        threshold_percent: int,
        max_fraction_percent: int,
        min_object_size: int,
        min_confidence_percent: int,
        roi_mask: np.ndarray | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> tuple[list[AxonHoleCandidate], dict[int, np.ndarray], np.ndarray]:
        arr = np.asarray(labels)
        gray = self._coerce_grayscale_image(image, arr.shape)
        if roi_mask is not None and np.asarray(roi_mask).shape != arr.shape:
            raise ValueError("Batch ROI mask shape does not match target labels shape.")
        roi = np.asarray(roi_mask, dtype=bool) if roi_mask is not None else None
        preview = np.zeros(arr.shape, dtype=np.uint32)
        candidates: list[AxonHoleCandidate] = []
        candidate_masks: dict[int, np.ndarray] = {}
        label_values = [int(value) for value in np.unique(arr) if int(value) != 0]
        total = max(1, len(label_values))
        if progress_callback is not None:
            progress_callback(0, total, "Preparing batch axon preview...")
        candidate_id = 1
        min_confidence = max(0.0, min(float(min_confidence_percent), 100.0)) / 100.0
        structure = ndi.generate_binary_structure(arr.ndim, 1)
        for completed, label_value in enumerate(label_values, start=1):
            component_labels, component_count = ndi.label(arr == label_value, structure=structure)
            objects = ndi.find_objects(component_labels)
            for component_number, slices in enumerate(objects, start=1):
                if slices is None:
                    continue
                local_component = component_labels[slices] == component_number
                object_area = int(np.count_nonzero(local_component))
                if object_area < int(min_object_size):
                    continue
                bbox = self._slices_to_bbox(slices)
                seed = self._seed_from_distance_transform(local_component, bbox)
                if roi is not None and not roi[seed]:
                    continue
                local_gray = gray[slices]
                local_seed = tuple(int(seed[axis] - bbox[axis][0]) for axis in range(arr.ndim))
                local_candidate = self._similar_candidate_from_seed(
                    local_gray,
                    local_component,
                    local_seed,
                    int(threshold_percent),
                )
                local_axon = self._connected_seed_region(local_candidate, local_seed)
                axon_area = int(np.count_nonzero(local_axon))
                confidence, status, reason = self._axon_candidate_score(
                    local_component,
                    local_axon,
                    object_area,
                    axon_area,
                    int(max_fraction_percent),
                    min_confidence,
                )
                candidate = AxonHoleCandidate(
                    candidate_id=candidate_id,
                    label_value=label_value,
                    object_area=object_area,
                    axon_area=axon_area,
                    axon_fraction=float(axon_area / object_area) if object_area else 0.0,
                    confidence=float(confidence),
                    status=status,
                    reason=reason,
                    seed=seed,
                    bbox=bbox,
                )
                candidates.append(candidate)
                if axon_area > 0:
                    candidate_masks[candidate_id] = local_axon
                    preview_slice = preview[slices]
                    preview_slice[local_axon] = candidate_id
                candidate_id += 1
            if progress_callback is not None:
                progress_callback(completed, total, f"Previewed label value {label_value} ({completed}/{total})")
        if progress_callback is not None:
            progress_callback(total, total, "Batch axon preview complete.")
        return candidates, candidate_masks, preview

    def apply_axon_hole_candidates(
        self,
        labels: Any,
        candidates: list[AxonHoleCandidate],
        candidate_masks: dict[int, np.ndarray],
        *,
        output_value: int,
        min_confidence_percent: int,
        candidate_ids: set[int] | None = None,
    ) -> tuple[np.ndarray, int, int]:
        out = np.asarray(labels).copy()
        min_confidence = max(0.0, min(float(min_confidence_percent), 100.0)) / 100.0
        selected_ids = {int(value) for value in candidate_ids} if candidate_ids is not None else None
        applied = 0
        changed = 0
        for candidate in candidates:
            if selected_ids is not None and candidate.candidate_id not in selected_ids:
                continue
            if candidate.status in {"skipped", "failed"}:
                continue
            if selected_ids is None and candidate.confidence < min_confidence:
                continue
            mask = candidate_masks.get(candidate.candidate_id)
            if mask is None:
                continue
            slices = self._bbox_slices(candidate.bbox)
            sub = out[slices]
            change_mask = mask & (sub != int(output_value))
            changed += int(np.count_nonzero(change_mask))
            sub[mask] = int(output_value)
            applied += 1
        return out, applied, changed

    def fill_axon_candidate_holes(
        self,
        candidates: list[AxonHoleCandidate],
        candidate_masks: dict[int, np.ndarray],
        candidate_ids: set[int],
        *,
        min_confidence_percent: int,
        max_fraction_percent: int,
    ) -> tuple[list[AxonHoleCandidate], dict[int, np.ndarray], int]:
        selected_ids = {int(value) for value in candidate_ids}
        min_confidence = max(0.0, min(float(min_confidence_percent), 100.0)) / 100.0
        max_fraction = max(0.01, min(float(max_fraction_percent), 100.0) / 100.0)
        updated_candidates: list[AxonHoleCandidate] = []
        updated_masks = dict(candidate_masks)
        filled_pixels = 0
        for candidate in candidates:
            if candidate.candidate_id not in selected_ids:
                updated_candidates.append(candidate)
                continue
            mask = updated_masks.get(candidate.candidate_id)
            if mask is None:
                updated_candidates.append(candidate)
                continue
            filled = ndi.binary_fill_holes(mask)
            added = int(np.count_nonzero(filled & ~mask))
            if added <= 0:
                updated_candidates.append(candidate)
                continue
            axon_area = int(np.count_nonzero(filled))
            axon_fraction = float(axon_area / candidate.object_area) if candidate.object_area else 0.0
            if axon_fraction > max_fraction:
                updated_candidates.append(
                    replace(
                        candidate,
                        axon_area=axon_area,
                        axon_fraction=axon_fraction,
                        confidence=0.35,
                        status="review",
                        reason="filled proposal exceeds max axon fraction",
                    )
                )
            else:
                confidence = max(float(candidate.confidence), 0.85)
                updated_candidates.append(
                    replace(
                        candidate,
                        axon_area=axon_area,
                        axon_fraction=axon_fraction,
                        confidence=confidence,
                        status="confident" if confidence >= min_confidence else "review",
                        reason="filled holes in axon proposal",
                    )
                )
                updated_masks[candidate.candidate_id] = filled
                filled_pixels += added
        return updated_candidates, updated_masks, filled_pixels

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

    def _coerce_grayscale_image(self, image: Any, target_shape: tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(image)
        if arr.shape == target_shape:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == len(target_shape) + 1 and arr.shape[-1] in (3, 4) and arr.shape[:-1] == target_shape:
            return arr[..., :3].astype(np.float32, copy=False).mean(axis=-1)
        if arr.ndim == len(target_shape) + 1 and arr.shape[0] in (3, 4) and arr.shape[1:] == target_shape:
            return arr[:3].astype(np.float32, copy=False).mean(axis=0)
        if arr.shape[-len(target_shape) :] == target_shape:
            reduced = arr
            while reduced.ndim > len(target_shape):
                reduced = reduced.mean(axis=0)
            return reduced.astype(np.float32, copy=False)
        raise ValueError(
            f"Source image shape {arr.shape} cannot be aligned to target labels shape {target_shape}."
        )

    def _similar_candidate_from_seed(
        self,
        image: np.ndarray,
        component: np.ndarray,
        seed: tuple[int, ...],
        threshold_percent: int,
    ) -> np.ndarray:
        values = image[component].astype(np.float32, copy=False)
        if values.size == 0:
            return np.zeros(component.shape, dtype=bool)
        seed_value = float(image[seed])
        high = float(np.percentile(values, 90.0))
        low = float(np.percentile(values, 10.0))
        span = max(high - low, np.finfo(np.float32).eps)
        fraction = max(0.0, min(float(threshold_percent), 100.0)) / 100.0
        tolerance = span * fraction
        return component & (np.abs(image - seed_value) <= tolerance)

    def _connected_seed_region(self, candidate: np.ndarray, seed: tuple[int, ...]) -> np.ndarray:
        if not bool(candidate[seed]):
            region = np.zeros(candidate.shape, dtype=bool)
            region[seed] = True
            return region
        structure = ndi.generate_binary_structure(candidate.ndim, 1)
        labeled, _count = ndi.label(candidate, structure=structure)
        component_id = int(labeled[seed])
        if component_id <= 0:
            return np.zeros(candidate.shape, dtype=bool)
        return labeled == component_id

    def _seed_from_distance_transform(
        self,
        component: np.ndarray,
        bbox: tuple[tuple[int, int], ...],
    ) -> tuple[int, ...]:
        padded = np.pad(component, 1, mode="constant", constant_values=False)
        distance = ndi.distance_transform_edt(padded)[tuple(slice(1, -1) for _axis in range(component.ndim))]
        local_seed = tuple(int(value) for value in np.unravel_index(int(np.argmax(distance)), distance.shape))
        return tuple(int(local_seed[axis] + bbox[axis][0]) for axis in range(component.ndim))

    def _axon_candidate_score(
        self,
        component: np.ndarray,
        axon: np.ndarray,
        object_area: int,
        axon_area: int,
        max_fraction_percent: int,
        min_confidence: float,
    ) -> tuple[float, str, str]:
        if axon_area <= 0:
            return 0.0, "failed", "no seed-similar axon region"
        fraction = float(axon_area / object_area) if object_area else 0.0
        max_fraction = max(0.01, min(float(max_fraction_percent), 100.0) / 100.0)
        if fraction > max_fraction:
            return 0.0, "failed", "axon region above max fraction"
        if fraction < 0.01:
            return 0.25, "review", "axon region very small"
        if self._touches_border(axon):
            return 0.35, "review", "axon region touches object bbox"
        ring_area = object_area - axon_area
        if ring_area <= 0:
            return 0.0, "failed", "no myelin remains after cut"
        confidence = 1.0
        confidence -= max(0.0, (fraction - 0.55) / max(0.01, max_fraction - 0.55)) * 0.35 if max_fraction > 0.55 else 0.0
        confidence -= 0.15 if fraction < 0.04 else 0.0
        confidence = max(0.0, min(1.0, confidence))
        status = "confident" if confidence >= min_confidence else "review"
        reason = "passes area and boundary checks" if status == "confident" else "below confidence threshold"
        return confidence, status, reason

    def _touches_border(self, mask: np.ndarray) -> bool:
        for axis in range(mask.ndim):
            head = [slice(None)] * mask.ndim
            tail = [slice(None)] * mask.ndim
            head[axis] = 0
            tail[axis] = -1
            if np.any(mask[tuple(head)]) or np.any(mask[tuple(tail)]):
                return True
        return False

    def _slices_to_bbox(self, slices: tuple[slice, ...]) -> tuple[tuple[int, int], ...]:
        return tuple((int(selector.start or 0), int(selector.stop or 0)) for selector in slices)

    def _bbox_slices(self, bbox: tuple[tuple[int, int], ...]) -> tuple[slice, ...]:
        return tuple(slice(int(lo), int(hi)) for lo, hi in bbox)

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
