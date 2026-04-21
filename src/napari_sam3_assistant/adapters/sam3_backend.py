from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

import numpy as np
from PIL import Image
import torch
from torch import nn

from ..core.coordinates import CoordinateMapper, extract_2d_image, to_rgb_uint8
from ..core.models import PromptBundle, PromptPolarity, Sam3Result, Sam3Session, Sam3Task


MAX_VIDEO_POINT_PROMPTS = 16


@dataclass
class Sam3AdapterConfig:
    checkpoint_path: Path | None = None
    bpe_path: Path | None = None
    device: str | None = None
    confidence_threshold: float = 0.5
    compile_model: bool = False
    load_from_hf: bool = False


class Sam3Adapter:
    """Narrow plugin-facing adapter around the local SAM3 package."""

    def __init__(self, config: Sam3AdapterConfig | None = None) -> None:
        self.config = config or Sam3AdapterConfig()
        self.image_model: Any | None = None
        self.image_processor: Any | None = None
        self.video_predictor: Any | None = None
        self.image_state: dict[str, Any] | None = None
        self._cached_image_state: dict[str, Any] | None = None
        self._cached_image_key: tuple[Any, ...] | None = None
        self.video_session: Sam3Session | None = None
        self._temp_video_dir: TemporaryDirectory[str] | None = None
        self._dtype_hook_handles: list[Any] = []

    def load_image(self, *, enable_instance_interactivity: bool = False) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if self._model_version() == "sam3.1":
            raise RuntimeError(
                "SAM3.1 multiplex checkpoints are supported for 3D/video propagation. "
                "Use a SAM3 3.0 image checkpoint such as sam3.pt for 2D image tasks."
            )

        device = self._resolved_device()
        kwargs = {
            "checkpoint_path": str(self.config.checkpoint_path)
            if self.config.checkpoint_path
            else None,
            "bpe_path": str(self.config.bpe_path) if self.config.bpe_path else None,
            "load_from_HF": self.config.load_from_hf,
            "enable_inst_interactivity": enable_instance_interactivity,
            "compile": self.config.compile_model,
            "device": device,
        }
        self.image_model = build_sam3_image_model(**kwargs)
        self._cached_image_state = None
        self._cached_image_key = None
        if device == "cpu":
            self._force_float32(self.image_model)
            self._install_cpu_float32_hooks(self.image_model)
        self.image_processor = Sam3Processor(
            self.image_model,
            device=str(device),
            confidence_threshold=self.config.confidence_threshold,
        )

    def load_video(self) -> None:
        device = self._resolved_device()
        if device == "cpu":
            raise RuntimeError(
                "The current SAM3 video predictor backend is CUDA-only. "
                "Use the image model for 2D tasks, or select CUDA with a compatible "
                "PyTorch build for 3D/video propagation."
            )
        kwargs = {
            "checkpoint_path": str(self.config.checkpoint_path)
            if self.config.checkpoint_path
            else None,
            "bpe_path": str(self.config.bpe_path) if self.config.bpe_path else None,
            "compile": self.config.compile_model,
        }
        gpu_id = self._cuda_device_index(device)
        if self._model_version() == "sam3.1":
            from sam3.model_builder import build_sam3_multiplex_video_predictor

            torch.cuda.set_device(gpu_id)
            self.video_predictor = build_sam3_multiplex_video_predictor(
                **kwargs,
                use_fa3=False,
                use_rope_real=True,
            )
        else:
            from sam3.model_builder import build_sam3_video_predictor

            self.video_predictor = build_sam3_video_predictor(
                **kwargs,
                gpus_to_use=[gpu_id],
            )

    def unload(self) -> None:
        self._remove_dtype_hooks()
        self.image_model = None
        self.image_processor = None
        self.image_state = None
        self._cached_image_state = None
        self._cached_image_key = None
        self.video_predictor = None
        self.video_session = None
        if self._temp_video_dir is not None:
            self._temp_video_dir.cleanup()
            self._temp_video_dir = None

    def run_image(
        self,
        image_data: np.ndarray,
        bundle: PromptBundle,
        *,
        cache_context: dict[str, Any] | None = None,
    ) -> Sam3Result:
        needs_interactivity = bool(bundle.points or bundle.masks)
        if self.image_processor is None:
            self.load_image(enable_instance_interactivity=needs_interactivity)
        elif needs_interactivity and not self._has_instance_interactivity():
            self.load_image(enable_instance_interactivity=True)

        frame = extract_2d_image(image_data, bundle.image)
        rgb = to_rgb_uint8(frame)
        state = self._initial_state_for_image(rgb, bundle, cache_context=cache_context)
        with self._inference_context():
            if bundle.text and bundle.text.text:
                prompt = self._text_prompt_for_model(bundle.text.text)
                state["_sam3_text_prompt_used"] = prompt
                state = self._set_text_prompt(prompt, state)
                self._normalize_state_tensors(state)
                if bundle.task == Sam3Task.TEXT and self._mask_count(state.get("masks")) == 0:
                    state = self._retry_text_prompt_with_lower_thresholds(prompt, state)

            mapper = CoordinateMapper(bundle.image)
            image_hw = rgb.shape[:2]
            for box in bundle.boxes:
                normalized = mapper.box_to_normalized_cxcywh(box, image_hw)
                state = self._add_geometric_prompt(
                    list(normalized),
                    box.polarity == PromptPolarity.POSITIVE,
                    state=state,
                )
                self._normalize_state_tensors(state)

            if bundle.points or bundle.masks:
                self._normalize_state_tensors(state)
                state = self._run_interactive_refinement(rgb, bundle, state)

        self.image_state = state
        return self._result_from_image_state(bundle, state)

    def _initial_state_for_image(
        self,
        rgb: np.ndarray,
        bundle: PromptBundle,
        *,
        cache_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cache_allowed = self._should_cache_image_state(bundle)
        cache_key = self._image_cache_key(rgb, bundle, cache_context=cache_context) if cache_allowed else None

        if (
            cache_allowed
            and cache_key is not None
            and self._cached_image_state is not None
            and self._cached_image_key == cache_key
        ):
            return self._clone_state(self._cached_image_state)

        pil_image = Image.fromarray(rgb)
        with self._inference_context():
            state = self.image_processor.set_image(pil_image)
        self._normalize_state_tensors(state)

        if cache_allowed and cache_key is not None:
            self._cached_image_key = cache_key
            self._cached_image_state = self._clone_state(state)
        else:
            self._cached_image_key = None
            self._cached_image_state = None
        return state

    def _should_cache_image_state(self, bundle: PromptBundle) -> bool:
        return bundle.task == Sam3Task.REFINE

    def _image_cache_key(
        self,
        rgb: np.ndarray,
        bundle: PromptBundle,
        *,
        cache_context: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        arr = np.ascontiguousarray(rgb)
        roi_bounds = None
        layer_identity = None
        layer_name = bundle.image.layer_name
        if cache_context:
            roi_bounds = cache_context.get("roi_bounds")
            layer_identity = cache_context.get("layer_identity")
            layer_name = str(cache_context.get("layer_name") or layer_name)
        fingerprint = hashlib.blake2b(
            arr.view(np.uint8),
            digest_size=16,
        ).hexdigest()
        checkpoint = str(self.config.checkpoint_path) if self.config.checkpoint_path else ""
        device = str(self._resolved_device(allow_cpu_fallback=True))
        return (
            layer_name,
            layer_identity,
            bundle.image.frame_index,
            bundle.image.channel_index,
            tuple(roi_bounds) if roi_bounds is not None else None,
            arr.shape,
            str(arr.dtype),
            checkpoint,
            device,
            fingerprint,
        )

    def _clone_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(state)

    def start_video_session(self, stack_data: np.ndarray, bundle: PromptBundle) -> Sam3Session:
        if self.video_predictor is None:
            self.load_video()

        self._temp_video_dir = TemporaryDirectory(prefix="napari-sam3-video-")
        video_dir = Path(self._temp_video_dir.name)
        self._write_stack_as_jpeg_dir(stack_data, bundle, video_dir)

        with self._inference_context():
            response = self.video_predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video_dir),
                }
            )
        session = Sam3Session(
            task=bundle.task,
            image=bundle.image,
            session_id=response["session_id"],
            resource_path=video_dir,
        )
        self.video_session = session
        return session

    def add_video_prompt(self, bundle: PromptBundle, session: Sam3Session) -> Sam3Result:
        if self.video_predictor is None:
            raise RuntimeError("Video predictor is not loaded.")
        if not session.session_id:
            raise RuntimeError("Video session has no SAM3 session id.")

        frame_index = bundle.image.frame_index or 0
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session.session_id,
            "frame_index": frame_index,
        }
        if bundle.masks:
            raise RuntimeError(
                "Labels-mask prompts are not supported by the SAM3 video predictor "
                "API used for 3D/video propagation. Use a box or point prompt on the "
                "selected frame, or run labels-mask prompting as a 2D image task."
            )
        if bundle.text and bundle.text.text:
            request["text"] = bundle.text.text

        if bundle.task == Sam3Task.EXEMPLAR and len(bundle.boxes) > 1:
            raise RuntimeError(
                "SAM3 video exemplar mode supports one initial visual box per prompted "
                "frame. Use one Shapes ROI, or run image exemplar mode for multiple "
                "exemplars."
            )
        if self._model_version() == "sam3" and bundle.boxes and len(bundle.boxes) > 1:
            raise RuntimeError(
                "SAM3.0 3D/video propagation supports one initial visual box per "
                "prompted frame. Use one box, or switch to SAM3.1 video multiplex for "
                "multi-box video prompts."
            )

        mapper = CoordinateMapper(bundle.image)
        height = bundle.image.data_shape[bundle.image.spatial_axes[0]]
        width = bundle.image.data_shape[bundle.image.spatial_axes[1]]

        if bundle.points:
            if len(bundle.points) > MAX_VIDEO_POINT_PROMPTS:
                raise RuntimeError(
                    "SAM3 video point prompts support up to "
                    f"{MAX_VIDEO_POINT_PROMPTS} points per request. Use fewer points "
                    "or split corrections into separate runs."
                )
            if (bundle.text and bundle.text.text) or bundle.boxes:
                raise RuntimeError(
                    "SAM3 video point prompts cannot be combined with text or box prompts "
                    "in one request. Use points alone for tracker refinement, or run a "
                    "separate text/box 3D prompt."
                )
            object_ids = {point.object_id for point in bundle.points if point.object_id is not None}
            if len(object_ids) > 1:
                raise RuntimeError(
                    "SAM3 video point prompts in one request must target one object id."
                )
            request["obj_id"] = next(iter(object_ids), 1)
            request["points"] = [
                mapper.point_to_normalized_xy(point.y, point.x, (height, width))
                for point in bundle.points
            ]
            request["point_labels"] = [
                1 if point.polarity == PromptPolarity.POSITIVE else 0
                for point in bundle.points
            ]
            self._ensure_video_prompt_frame_cache(session.session_id, frame_index)

        if bundle.boxes:
            request["bounding_boxes"] = [
                mapper.box_to_normalized_xywh(box, (height, width))
                for box in bundle.boxes
            ]
            request["bounding_box_labels"] = [
                1 if box.polarity == PromptPolarity.POSITIVE else 0 for box in bundle.boxes
            ]

        with self._inference_context():
            response = self.video_predictor.handle_request(request)
        return self._result_from_video_output(bundle, response, session.session_id)

    def _ensure_video_prompt_frame_cache(self, session_id: str, frame_index: int) -> None:
        sessions = getattr(self.video_predictor, "_all_inference_states", None)
        if not isinstance(sessions, dict):
            return
        session = sessions.get(session_id)
        if not isinstance(session, dict):
            return
        state = session.get("state")
        if not isinstance(state, dict):
            return
        cache = state.setdefault("cached_frame_outputs", {})
        if isinstance(cache, dict):
            num_frames = int(state.get("num_frames") or 0)
            if num_frames > 0:
                for idx in range(num_frames):
                    cache.setdefault(idx, {})
            else:
                cache.setdefault(frame_index, {})

    def propagate_video(
        self,
        bundle: PromptBundle,
        session: Sam3Session,
        *,
        direction: str = "both",
    ) -> Iterator[Sam3Result]:
        if self.video_predictor is None:
            raise RuntimeError("Video predictor is not loaded.")
        if not session.session_id:
            raise RuntimeError("Video session has no SAM3 session id.")

        with self._inference_context():
            stream = self.video_predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session.session_id,
                    "propagation_direction": direction,
                    "start_frame_index": bundle.image.frame_index or 0,
                }
            )
            for response in stream:
                yield self._result_from_video_output(bundle, response, session.session_id)

    def _run_interactive_refinement(
        self,
        rgb: np.ndarray,
        bundle: PromptBundle,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        predictor = getattr(self.image_model, "inst_interactive_predictor", None)
        tracker_model = getattr(predictor, "model", None)
        if predictor is None or tracker_model is None:
            raise RuntimeError(
                "SAM3 point refinement is unavailable: the local image model did not "
                "construct a usable interactive tracker. Use a box/text prompt, or run "
                "points through 3D/video mode."
            )

        point_coords = None
        point_labels = None
        if bundle.points:
            point_coords = np.asarray([point.xy for point in bundle.points], dtype=np.float32)
            point_labels = np.asarray(
                [
                    1 if point.polarity == PromptPolarity.POSITIVE else 0
                    for point in bundle.points
                ],
                dtype=np.int32,
            )

        box = None
        if bundle.boxes:
            box = np.asarray(
                [CoordinateMapper(bundle.image).box_to_xyxy(bundle.boxes[0])],
                dtype=np.float32,
            )

        mask_input = None
        masks, scores, low_res = self.image_model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=False,
            normalize_coords=True,
        )
        state["masks"] = np.asarray(masks) > 0
        state["scores"] = np.asarray(scores)
        state["masks_logits"] = np.asarray(low_res)
        return state

    def _set_text_prompt(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        if self._resolved_device(allow_cpu_fallback=True) != "cpu":
            return self.image_processor.set_text_prompt(prompt, state)

        text_outputs = self.image_model.backbone.forward_text(
            [prompt],
            device=self.image_processor.device,
        )
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.image_model._get_dummy_prompt()
        self._normalize_state_tensors(state)
        return self.image_processor._forward_grounding(state)

    def _retry_text_prompt_with_lower_thresholds(
        self,
        prompt: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        current = float(getattr(self.image_processor, "confidence_threshold", self.config.confidence_threshold))
        thresholds = []
        for threshold in (min(current, 0.35) * 0.5, 0.20, 0.10):
            threshold = max(0.01, round(float(threshold), 3))
            if threshold < current and threshold not in thresholds:
                thresholds.append(threshold)
        for threshold in thresholds:
            state = self.image_processor.set_confidence_threshold(threshold, state)
            state["_sam3_text_threshold_used"] = threshold
            state["_sam3_text_prompt_used"] = prompt
            self._normalize_state_tensors(state)
            if self._mask_count(state.get("masks")) > 0:
                self.image_processor.set_confidence_threshold(current)
                return state
        self.image_processor.set_confidence_threshold(current)
        return state

    def _text_prompt_for_model(self, prompt: str) -> str:
        text = " ".join(prompt.strip().split())
        lowered = text.lower()
        prefixes = (
            "segment all of the ",
            "segment all the ",
            "segment the ",
            "segment all ",
            "segment ",
            "detect all of the ",
            "detect all the ",
            "detect the ",
            "detect all ",
            "detect ",
            "find all of the ",
            "find all the ",
            "find the ",
            "find all ",
            "find ",
        )
        for prefix in prefixes:
            if lowered.startswith(prefix) and len(text) > len(prefix):
                return text[len(prefix):].strip(" .,:;")
        return text.strip(" .,:;")

    def _mask_count(self, masks: Any) -> int:
        if masks is None:
            return 0
        try:
            return len(masks)
        except TypeError:
            return int(np.asarray(masks).shape[0]) if np.asarray(masks).ndim else 0

    def _add_geometric_prompt(
        self,
        box: list[float],
        label: bool,
        *,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if self._resolved_device(allow_cpu_fallback=True) != "cpu":
            return self.image_processor.add_geometric_prompt(box, label, state)

        if "language_features" not in state["backbone_out"]:
            text_outputs = self.image_model.backbone.forward_text(
                ["visual"],
                device=self.image_processor.device,
            )
            state["backbone_out"].update(text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.image_model._get_dummy_prompt()

        boxes = torch.tensor(
            box,
            device=self.image_processor.device,
            dtype=torch.float32,
        ).view(1, 1, 4)
        labels = torch.tensor(
            [label],
            device=self.image_processor.device,
            dtype=torch.bool,
        ).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)
        self._normalize_state_tensors(state)
        return self.image_processor._forward_grounding(state)

    def _inference_context(self):
        device = str(self._resolved_device(allow_cpu_fallback=True))
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _resolved_device(self, *, allow_cpu_fallback: bool = False) -> str:
        requested = self.config.device
        issue = cuda_compatibility_issue()
        if requested:
            return requested
        if torch.cuda.is_available() and issue is None:
            return "cuda"
        if torch.cuda.is_available() and issue is not None:
            return "cpu"
        return self._model_device() if allow_cpu_fallback else "cpu"

    def _model_device(self) -> str:
        for model in (self.image_model, getattr(self.video_predictor, "model", None)):
            if model is None:
                continue
            try:
                return str(next(model.parameters()).device)
            except Exception:
                device = getattr(model, "device", None)
                if device is not None:
                    return str(device)
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _model_version(self) -> str:
        checkpoint = self.config.checkpoint_path
        if checkpoint is not None and "sam3.1" in checkpoint.name:
            return "sam3.1"
        return "sam3"

    def _cuda_device_index(self, device: str) -> int:
        parsed = torch.device(device)
        if parsed.type != "cuda":
            raise RuntimeError(f"SAM3 video predictor requires CUDA, got {device!r}.")
        if parsed.index is not None:
            return parsed.index
        try:
            return torch.cuda.current_device()
        except Exception:
            return 0

    def _force_float32(self, model: Any) -> None:
        if model is None:
            return
        try:
            model.float()
        except Exception:
            pass

    def _has_instance_interactivity(self) -> bool:
        predictor = getattr(self.image_model, "inst_interactive_predictor", None)
        return predictor is not None and getattr(predictor, "model", None) is not None

    def _install_cpu_float32_hooks(self, model: Any) -> None:
        self._remove_dtype_hooks()
        if not isinstance(model, nn.Module):
            return
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_pre_hook(self._linear_float32_pre_hook)
                self._dtype_hook_handles.append(handle)
            elif isinstance(module, nn.MultiheadAttention):
                handle = module.register_forward_pre_hook(self._mha_float32_pre_hook)
                self._dtype_hook_handles.append(handle)

    def _remove_dtype_hooks(self) -> None:
        for handle in self._dtype_hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._dtype_hook_handles = []

    def _linear_float32_pre_hook(self, module: nn.Linear, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            return inputs
        return (self._cast_float_tensor(inputs[0], module.weight.dtype), *inputs[1:])

    def _mha_float32_pre_hook(
        self,
        module: nn.MultiheadAttention,
        inputs: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        if len(inputs) < 3:
            return inputs
        dtype = module.in_proj_weight.dtype
        return (
            self._cast_float_tensor(inputs[0], dtype),
            self._cast_float_tensor(inputs[1], dtype),
            self._cast_float_tensor(inputs[2], dtype),
            *inputs[3:],
        )

    def _cast_float_tensor(self, value: Any, dtype: torch.dtype) -> Any:
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value

    def _normalize_state_tensors(self, state: dict[str, Any]) -> None:
        device = self._torch_device()
        dtype = torch.float32 if device.type == "cpu" else None
        self._convert_tensors(state, device=device, dtype=dtype)

    def _torch_device(self) -> torch.device:
        return torch.device(self._resolved_device(allow_cpu_fallback=True))

    def _convert_tensors(
        self,
        value: Any,
        *,
        device: torch.device,
        dtype: torch.dtype | None,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            target_dtype = dtype if value.is_floating_point() and dtype is not None else value.dtype
            if value.device != device or value.dtype != target_dtype:
                return value.to(device=device, dtype=target_dtype)
            return value
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = self._convert_tensors(item, device=device, dtype=dtype)
            return value
        if isinstance(value, list):
            for idx, item in enumerate(value):
                value[idx] = self._convert_tensors(item, device=device, dtype=dtype)
            return value
        if isinstance(value, tuple):
            return tuple(self._convert_tensors(item, device=device, dtype=dtype) for item in value)
        if hasattr(value, "__dict__"):
            for key, item in list(vars(value).items()):
                try:
                    setattr(value, key, self._convert_tensors(item, device=device, dtype=dtype))
                except Exception:
                    pass
        return value

    def _result_from_image_state(
        self,
        bundle: PromptBundle,
        state: dict[str, Any],
    ) -> Sam3Result:
        masks = self._to_numpy(state.get("masks"))
        boxes = self._to_numpy(state.get("boxes"))
        scores = self._to_numpy(state.get("scores"))
        if masks is not None and masks.ndim == 4:
            masks = masks[:, 0]
        labels = self._labels_from_masks(masks)
        return Sam3Result(
            task=bundle.task,
            frame_index=bundle.image.frame_index,
            masks=masks,
            labels=labels,
            boxes_xyxy=boxes,
            scores=scores,
            object_ids=np.arange(1, len(masks) + 1) if masks is not None else None,
            metadata={
                "text_prompt_used": state.get("_sam3_text_prompt_used"),
                "text_threshold_used": state.get(
                    "_sam3_text_threshold_used",
                    getattr(self.image_processor, "confidence_threshold", self.config.confidence_threshold),
                ),
            },
        )

    def _result_from_video_output(
        self,
        bundle: PromptBundle,
        response: dict[str, Any],
        session_id: str,
    ) -> Sam3Result:
        outputs = response.get("outputs", {})
        masks = self._to_numpy(outputs.get("out_binary_masks"))
        boxes_xywh = self._to_numpy(outputs.get("out_boxes_xywh"))
        boxes = self._video_boxes_xywh_to_xyxy(boxes_xywh, bundle.image.data_shape)
        object_ids = self._to_numpy(outputs.get("out_obj_ids"))
        scores = self._to_numpy(outputs.get("out_probs"))
        return Sam3Result(
            task=bundle.task,
            frame_index=response.get("frame_index"),
            masks=masks,
            labels=self._labels_from_masks(masks, object_ids=object_ids),
            boxes_xyxy=boxes,
            scores=scores,
            object_ids=object_ids,
            session_id=session_id,
            metadata={"frame_stats": outputs.get("frame_stats")},
        )

    def _write_stack_as_jpeg_dir(
        self,
        stack_data: np.ndarray,
        bundle: PromptBundle,
        output_dir: Path,
    ) -> None:
        arr = np.asarray(stack_data)
        frame_axis = bundle.image.frame_axis
        if frame_axis is None:
            frames = arr[None, ...]
        else:
            frames = np.moveaxis(arr, frame_axis, 0)

        for idx, frame in enumerate(frames):
            frame_2d = np.asarray(frame)
            if bundle.image.channel_axis is not None and frame_2d.ndim > 2:
                frame_2d = frame_2d[bundle.image.channel_index or 0]
            while frame_2d.ndim > 2 and frame_2d.shape[-1] not in (3, 4):
                frame_2d = frame_2d[0]
            rgb = to_rgb_uint8(frame_2d)
            Image.fromarray(rgb).save(output_dir / f"{idx:06d}.jpg", quality=95)

    def _video_boxes_xywh_to_xyxy(
        self,
        boxes_xywh: np.ndarray | None,
        data_shape: tuple[int, ...],
    ) -> np.ndarray | None:
        if boxes_xywh is None:
            return None
        height, width = data_shape[-2], data_shape[-1]
        boxes = np.asarray(boxes_xywh, dtype=np.float32).copy()
        x = boxes[:, 0] * width
        y = boxes[:, 1] * height
        w = boxes[:, 2] * width
        h = boxes[:, 3] * height
        return np.stack([x, y, x + w, y + h], axis=1)

    def _labels_from_masks(
        self,
        masks: np.ndarray | None,
        *,
        object_ids: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if masks is None:
            return None
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim == 2:
            return masks.astype(np.uint8)
        ids = None
        if object_ids is not None:
            ids = np.asarray(object_ids).reshape(-1)
        labels = np.zeros(masks.shape[-2:], dtype=np.uint32)
        for idx, mask in enumerate(masks, start=1):
            label_value = int(ids[idx - 1]) if ids is not None and idx - 1 < len(ids) else idx
            labels[np.asarray(mask).astype(bool)] = label_value
        return labels

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
            if value.dtype == torch.bfloat16:
                value = value.to(dtype=torch.float32)
            value = value.cpu().numpy()
        return np.asarray(value)


def cuda_compatibility_issue() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)
        arch = f"sm_{major}{minor}"
        supported = set(torch.cuda.get_arch_list())
    except Exception:
        return None
    if supported and arch not in supported:
        supported_text = ", ".join(sorted(supported))
        return (
            f"CUDA device '{device_name}' reports capability {major}.{minor} "
            f"({arch}), but this PyTorch build supports: {supported_text}."
        )
    return None
