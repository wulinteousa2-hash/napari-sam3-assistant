from __future__ import annotations

from typing import Any

import numpy as np

from .component_analysis_service import ComponentAnalysisService
from .models import MergeOptions
from .registry_service import AcceptedObjectRegistry
from .utils import safe_get_layer


class MaskMergeService:
    def __init__(self, viewer: Any) -> None:
        self.viewer = viewer
        self.registry = AcceptedObjectRegistry(viewer)

    def merge_accepted_objects(self, layer_names: list[str], options: MergeOptions | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        layers = [safe_get_layer(self.viewer, name) for name in layer_names]
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            raise ValueError("Select at least one accepted-object layer.")
        shape = np.asarray(layers[0].data).shape
        class_arrays: list[np.ndarray] = []
        classes: list[str] = []
        class_values: list[int] = []
        for layer in layers:
            data = np.asarray(layer.data)
            if data.shape != shape:
                raise ValueError(f"Layer shape mismatch: {layer.name} has {data.shape}, expected {shape}.")
            metadata = self.registry.metadata_for_layer(layer)
            if str(metadata.get("review_status") or "accepted") == "rejected":
                continue
            class_value = int(metadata.get("class_value") or 1)
            class_name = str(metadata.get("class_name") or "")
            if class_name and class_name not in classes:
                classes.append(class_name)
            if class_value not in class_values:
                class_values.append(class_value)
            class_arrays.append(np.where(data != 0, class_value, 0).astype(np.uint32, copy=False))
        if not class_arrays:
            raise ValueError("No accepted object data found to merge.")
        merge_options = options or MergeOptions(mode="semantic", overlap_rule="later_wins")
        if merge_options.overlap_rule == "later_wins":
            out = np.zeros(shape, dtype=np.uint32)
            for arr in class_arrays:
                out[arr != 0] = arr[arr != 0]
        else:
            out = self._merge_semantic(class_arrays, merge_options.overlap_rule)
        metadata = {
            "sam3_role": "class_working_mask",
            "review_status": "accepted",
            "class_name": ", ".join(classes),
            "class_value": class_values[0] if len(class_values) == 1 else class_values,
            "source_accepted_layers": list(layer_names),
            "overlap_rule": merge_options.overlap_rule,
        }
        return out, metadata

    def merge_final_masks(self, layer_names: list[str], options: MergeOptions) -> np.ndarray:
        layers = [safe_get_layer(self.viewer, name) for name in layer_names]
        layers = [layer for layer in layers if layer is not None]
        if not layers:
            raise ValueError("Select at least one class mask layer.")
        shape = np.asarray(layers[0].data).shape
        arrays = []
        for layer in layers:
            arr = np.asarray(layer.data)
            if arr.shape != shape:
                raise ValueError(f"Layer shape mismatch: {layer.name} has {arr.shape}, expected {shape}.")
            arrays.append(arr)
        if options.mode == "instance":
            return self._merge_instance(arrays, options.overlap_rule)
        if options.mode == "binary":
            semantic = self._merge_semantic(arrays, options.overlap_rule)
            return (semantic != 0).astype(np.uint8)
        return self._merge_semantic(arrays, options.overlap_rule)

    def _merge_semantic(self, arrays: list[np.ndarray], overlap_rule: str) -> np.ndarray:
        out = np.zeros(arrays[0].shape, dtype=np.uint32)
        if overlap_rule in {"class_priority", "earlier_wins"}:
            for arr in arrays:
                mask = (arr != 0) & (out == 0)
                out[mask] = arr[mask]
            return out
        if overlap_rule == "later_wins":
            for arr in arrays:
                mask = arr != 0
                out[mask] = arr[mask]
            return out
        if overlap_rule == "set_background":
            counts = np.zeros(arrays[0].shape, dtype=np.uint16)
            for arr in arrays:
                counts += arr != 0
            for arr in arrays:
                mask = (arr != 0) & (counts == 1)
                out[mask] = arr[mask]
            return out
        if overlap_rule in {"larger_component", "smaller_component"}:
            return self._merge_by_component_size(arrays, larger=overlap_rule == "larger_component")
        for arr in arrays:
            mask = (arr != 0) & (out == 0)
            out[mask] = arr[mask]
        return out

    def _merge_instance(self, arrays: list[np.ndarray], overlap_rule: str) -> np.ndarray:
        semantic = self._merge_semantic(arrays, overlap_rule)
        analysis = ComponentAnalysisService()
        out = np.zeros(semantic.shape, dtype=np.uint32)
        for index, record in enumerate(analysis.analyze(semantic), start=1):
            mask = analysis.component_mask(record.component_id)
            if mask is not None:
                out[mask] = index
        return out

    def _merge_by_component_size(self, arrays: list[np.ndarray], *, larger: bool) -> np.ndarray:
        out = np.zeros(arrays[0].shape, dtype=np.uint32)
        score = np.full(arrays[0].shape, -1 if larger else np.iinfo(np.int64).max, dtype=np.int64)
        analysis = ComponentAnalysisService()
        for arr in arrays:
            for record in analysis.analyze(arr):
                mask = analysis.component_mask(record.component_id)
                if mask is None:
                    continue
                take = mask & ((record.area > score) if larger else (record.area < score))
                out[take] = record.label_value
                score[take] = record.area
        return out
