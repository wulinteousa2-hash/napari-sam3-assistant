from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Iterator

import numpy as np


class Sam3Diagnostics:
    """Opt-in SAM3 runtime diagnostics for comparing video propagation paths."""

    def __init__(self, log: Callable[[str], None]) -> None:
        self._log = log

    def log(self, message: str) -> None:
        try:
            self._log(message)
        except Exception:
            pass

    def log_timing(self, label: str, start: float) -> None:
        self.log(f"{label} finished in {time.perf_counter() - start:.2f} sec.")

    def log_cuda_diagnostics(self, stage: str) -> None:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            if available:
                device_index = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_index)
                capability = torch.cuda.get_device_capability(device_index)
                allocated = int(torch.cuda.memory_allocated(device_index))
                reserved = int(torch.cuda.memory_reserved(device_index))
            else:
                device_index = None
                device_name = ""
                capability = None
                allocated = 0
                reserved = 0
            self.log(
                "SAM3.1 CUDA diagnostics "
                f"({stage}): available={available}, current_device={device_index}, "
                f"name={device_name}, torch_cuda={getattr(torch.version, 'cuda', None)}, "
                f"capability={capability}, allocated={allocated}, reserved={reserved}."
            )
        except Exception as exc:
            self.log(f"SAM3.1 CUDA diagnostics ({stage}) failed: {exc}")

    def log_runtime_diagnostics(self, adapter: Any, *, stage: str) -> None:
        predictor = getattr(adapter, "video_predictor", None)
        attrs = {
            "predictor_type": type(predictor).__name__ if predictor is not None else None,
            "model_type": type(getattr(predictor, "model", None)).__name__ if predictor is not None else None,
            "async_loading_frames": getattr(predictor, "async_loading_frames", None),
            "video_loader_type": getattr(predictor, "video_loader_type", None),
            "thread_id": threading.get_ident(),
            "env_USE_PERFLIB": os.environ.get("USE_PERFLIB"),
            "env_SAM3_FORCE_SDPA_FALLBACK": os.environ.get("SAM3_FORCE_SDPA_FALLBACK"),
            "env_SAM3_FORCE_SLOW_EDT": os.environ.get("SAM3_FORCE_SLOW_EDT"),
        }
        try:
            import torch

            attrs.update(
                {
                    "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                    "math_sdp": torch.backends.cuda.math_sdp_enabled(),
                    "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                }
            )
        except Exception as exc:
            attrs["torch_backend_error"] = str(exc)
        try:
            from sam3 import perflib

            attrs["perflib_enabled"] = getattr(perflib, "is_enabled", None)
        except Exception as exc:
            attrs["perflib_error"] = str(exc)
        self.log(f"SAM3.1 runtime diagnostics ({stage}): {attrs}")

    def log_session_diagnostics(self, adapter: Any, session: Any, *, stage: str) -> None:
        predictor = getattr(adapter, "video_predictor", None)
        sessions = getattr(predictor, "_all_inference_states", None)
        if not isinstance(sessions, dict) or session is None:
            self.log(f"SAM3.1 session diagnostics ({stage}): no session state dictionary.")
            return
        record = sessions.get(getattr(session, "session_id", None))
        state = record.get("state") if isinstance(record, dict) else None
        if not isinstance(state, dict):
            self.log(f"SAM3.1 session diagnostics ({stage}): no state for session.")
            return
        images = state.get("images")
        first_image = self._first_sequence_item(images)
        feature_cache = state.get("feature_cache")
        cached_features = state.get("cached_features")
        sam2_states = state.get("sam2_inference_states")
        first_sam2_state = self._first_sequence_item(sam2_states)
        details = {
            "session_id": getattr(session, "session_id", None),
            "state_keys": sorted(str(key) for key in state.keys())[:40],
            "num_frames": state.get("num_frames"),
            "image_size": state.get("image_size"),
            "orig_height": state.get("orig_height"),
            "orig_width": state.get("orig_width"),
            "video_height": state.get("video_height"),
            "video_width": state.get("video_width"),
            "device": str(state.get("device")),
            "storage_device": str(state.get("storage_device")),
            "offload_video_to_cpu": state.get("offload_video_to_cpu"),
            "offload_state_to_cpu": state.get("offload_state_to_cpu"),
            "images_type": type(images).__name__ if images is not None else None,
            "images_len": self._safe_len(images),
            "first_image": self._value_summary(first_image),
            "feature_cache_type": type(feature_cache).__name__ if feature_cache is not None else None,
            "feature_cache_len": self._safe_len(feature_cache),
            "feature_cache_keys": self._mapping_keys(feature_cache),
            "cached_features_type": type(cached_features).__name__ if cached_features is not None else None,
            "cached_features_len": self._safe_len(cached_features),
            "cached_features_keys": self._mapping_keys(cached_features),
            "sam2_inference_states_len": self._safe_len(sam2_states),
            "first_sam2_state_keys": self._mapping_keys(first_sam2_state),
            "obj_ids": self._short_value(state.get("obj_ids")),
            "obj_id_to_idx": self._short_value(state.get("obj_id_to_idx")),
            "tracker_metadata_keys": self._mapping_keys(state.get("tracker_metadata")),
        }
        self.log(f"SAM3.1 session diagnostics ({stage}): {details}")

    def log_prompt_diagnostics(self, bundle: Any) -> None:
        try:
            from .coordinates import CoordinateMapper

            height = bundle.image.data_shape[bundle.image.spatial_axes[0]]
            width = bundle.image.data_shape[bundle.image.spatial_axes[1]]
            mapper = CoordinateMapper(bundle.image)
            boxes = [
                {
                    "object_id": getattr(box, "object_id", None),
                    "yx": (float(box.y0), float(box.x0), float(box.y1), float(box.x1)),
                    "xyxy": mapper.box_to_xyxy(box),
                    "norm_xywh": mapper.box_to_normalized_xywh(box, (height, width)),
                }
                for box in getattr(bundle, "boxes", [])
            ]
            self.log(
                "SAM3.1 prompt diagnostics: "
                f"frame={bundle.image.frame_index}, image_hw={(height, width)}, boxes={boxes}, "
                f"points={len(getattr(bundle, 'points', []) or [])}."
            )
        except Exception as exc:
            self.log(f"SAM3.1 prompt diagnostics failed: {exc}")

    def describe_image_source(self, image: Any) -> str:
        image_type = type(image)
        module = getattr(image_type, "__module__", "")
        qualname = getattr(image_type, "__qualname__", image_type.__name__)
        shape = getattr(image, "shape", None)
        dtype = getattr(image, "dtype", None)
        chunks = getattr(image, "chunks", None)
        flags = []
        if isinstance(image, np.ndarray):
            flags.append("numpy")
            if isinstance(image, np.memmap):
                flags.append("memmap")
        if module.startswith("dask"):
            flags.append("dask")
        if module.startswith("zarr"):
            flags.append("zarr")
        if not flags:
            flags.append("lazy_or_custom" if hasattr(image, "__getitem__") and shape is not None else "unknown")
        return (
            f"type={module}.{qualname}, shape={shape}, dtype={dtype}, "
            f"chunks={chunks}, source={','.join(flags)}"
        )

    def iter_propagation_with_timing(
        self,
        iterator: Iterator[Any],
        *,
        label: str = "SAM3.1 propagation",
    ) -> Iterator[Any]:
        start = time.perf_counter()
        last = start
        count = 0
        for result in iterator:
            count += 1
            now = time.perf_counter()
            self.log(
                "SAM3.1 propagation yield: "
                f"index={count}, frame={getattr(result, 'frame_index', None)}, "
                f"dt={now - last:.3f} sec, elapsed={now - start:.2f} sec."
            )
            last = now
            yield result
        self.log(f"{label} finished in {time.perf_counter() - start:.2f} sec; frames={count}.")

    @staticmethod
    def _safe_len(value: Any) -> int | None:
        try:
            return len(value) if value is not None and hasattr(value, "__len__") else None
        except Exception:
            return None

    @staticmethod
    def _first_sequence_item(value: Any) -> Any:
        try:
            if value is not None and len(value):
                return value[0]
        except Exception:
            return None
        return None

    @staticmethod
    def _mapping_keys(value: Any) -> list[str] | None:
        if not isinstance(value, dict):
            return None
        return sorted(str(key) for key in value.keys())[:30]

    @classmethod
    def _value_summary(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "type": type(value).__name__,
            "shape": tuple(value.shape) if hasattr(value, "shape") else None,
            "device": str(getattr(value, "device", "")),
            "dtype": str(getattr(value, "dtype", "")),
            "len": cls._safe_len(value),
        }

    @staticmethod
    def _short_value(value: Any) -> str | None:
        if value is None:
            return None
        text = repr(value)
        return text if len(text) <= 300 else text[:300] + "..."
