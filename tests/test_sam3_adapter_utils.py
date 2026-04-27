from pathlib import Path
import logging
import sys
import types

import numpy as np
import torch

import napari_sam3_assistant.adapters.sam3_backend as backend
from napari_sam3_assistant.adapters import Sam3Adapter, Sam3AdapterConfig
from napari_sam3_assistant.core.coordinates import infer_image_selection
from napari_sam3_assistant.core.models import (
    BoxPrompt,
    PointPrompt,
    PromptBundle,
    PromptPolarity,
    MaskPrompt,
    Sam3Session,
    Sam3Task,
    TextPrompt,
)


def test_to_numpy_casts_bfloat16_to_float32_before_numpy_conversion():
    adapter = Sam3Adapter()
    tensor = torch.asarray([1.0, 2.0], dtype=torch.bfloat16)

    array = adapter._to_numpy(tensor)

    assert array.dtype == np.float32
    np.testing.assert_allclose(array, np.asarray([1.0, 2.0], dtype=np.float32))


def test_labels_from_masks_preserves_sam3_object_ids():
    adapter = Sam3Adapter()
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[0, 0:2, 0:2] = True
    masks[1, 2:4, 2:4] = True

    labels = adapter._labels_from_masks(masks, object_ids=np.asarray([7, 12]))

    assert labels[0, 0] == 7
    assert labels[3, 3] == 12
    assert labels[0, 3] == 0


def test_labels_from_masks_handles_singleton_channel_mask_dimension():
    adapter = Sam3Adapter()
    masks = np.zeros((1, 1, 4, 4), dtype=bool)
    masks[0, 0, 1:3, 1:3] = True

    labels = adapter._labels_from_masks(masks, object_ids=np.asarray([5]))

    assert labels.shape == (4, 4)
    assert labels[1, 1] == 5


def test_text_prompt_for_model_removes_instruction_prefix():
    adapter = Sam3Adapter()

    assert adapter._text_prompt_for_model("segment all the myelin ring") == "myelin ring"
    assert adapter._text_prompt_for_model("nucleus") == "nucleus"


def test_text_prompt_threshold_retry_restores_configured_threshold():
    class Processor:
        confidence_threshold = 0.35

        def __init__(self):
            self.thresholds = []

        def set_confidence_threshold(self, threshold, state=None):
            self.confidence_threshold = threshold
            self.thresholds.append(threshold)
            if state is not None:
                state["masks"] = np.ones((1, 2, 2), dtype=bool)
            return state

    adapter = Sam3Adapter(Sam3AdapterConfig(confidence_threshold=0.35))
    adapter.image_processor = Processor()
    state = {"masks": np.zeros((0, 2, 2), dtype=bool)}

    result = adapter._retry_text_prompt_with_lower_thresholds("myelin ring", state)

    assert adapter.image_processor.thresholds == [0.175, 0.35]
    assert result["_sam3_text_threshold_used"] == 0.175
    assert result["_sam3_text_prompt_used"] == "myelin ring"


def test_explicit_cuda_is_allowed_even_when_arch_warning_exists(monkeypatch):
    monkeypatch.setattr(backend, "cuda_compatibility_issue", lambda: "unsupported arch")

    adapter = Sam3Adapter(Sam3AdapterConfig(device="cuda"))

    assert adapter._resolved_device() == "cuda"


def test_auto_uses_cpu_when_cuda_arch_warning_exists(monkeypatch):
    monkeypatch.setattr(backend, "cuda_compatibility_issue", lambda: "unsupported arch")
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: True)

    adapter = Sam3Adapter(Sam3AdapterConfig(device=None))

    assert adapter._resolved_device() == "cpu"


def test_normalize_state_tensors_moves_nested_prompt_objects_to_adapter_device(monkeypatch):
    monkeypatch.setattr(backend, "cuda_compatibility_issue", lambda: "unsupported arch")
    monkeypatch.setattr(backend.torch.cuda, "is_available", lambda: True)
    adapter = Sam3Adapter(Sam3AdapterConfig(device=None))
    prompt = type("PromptLike", (), {})()
    prompt.box_embeddings = torch.ones((1, 1, 4), dtype=torch.bfloat16)
    state = {
        "backbone_out": {
            "language_features": torch.ones((1, 1, 2), dtype=torch.bfloat16),
        },
        "geometric_prompt": prompt,
    }

    adapter._normalize_state_tensors(state)

    assert state["backbone_out"]["language_features"].device.type == "cpu"
    assert state["backbone_out"]["language_features"].dtype == torch.float32
    assert prompt.box_embeddings.device.type == "cpu"
    assert prompt.box_embeddings.dtype == torch.float32


def test_load_video_does_not_pass_unsupported_device_keyword(monkeypatch):
    calls = {}

    def fake_build_sam3_video_predictor(**kwargs):
        calls.update(kwargs)
        return object()

    module = types.ModuleType("sam3.model_builder")
    module.build_sam3_video_predictor = fake_build_sam3_video_predictor
    monkeypatch.setitem(sys.modules, "sam3.model_builder", module)

    adapter = Sam3Adapter(Sam3AdapterConfig(device="cuda:2", compile_model=True))
    adapter.load_video()

    assert calls["compile"] is True
    assert calls["gpus_to_use"] == [2]
    assert "device" not in calls


def test_load_video_reports_cpu_backend_limitation():
    adapter = Sam3Adapter(Sam3AdapterConfig(device="cpu"))

    try:
        adapter.load_video()
    except RuntimeError as error:
        assert "video predictor backend is CUDA-only" in str(error)
    else:
        raise AssertionError("Expected CPU video loading to fail clearly")


def test_sam31_checkpoint_routes_to_multiplex_video_predictor(monkeypatch):
    calls = {}

    def fake_build_sam3_multiplex_video_predictor(**kwargs):
        calls.update(kwargs)
        return object()

    module = types.ModuleType("sam3.model_builder")
    module.build_sam3_multiplex_video_predictor = fake_build_sam3_multiplex_video_predictor
    monkeypatch.setitem(sys.modules, "sam3.model_builder", module)
    monkeypatch.setattr(backend.torch.cuda, "set_device", lambda device: None)

    adapter = Sam3Adapter(
        Sam3AdapterConfig(
            checkpoint_path=Path("/models/sam3.1_multiplex.pt"),
            device="cuda:1",
            compile_model=True,
        )
    )
    adapter.load_video()

    assert calls["checkpoint_path"] == "/models/sam3.1_multiplex.pt"
    assert calls["compile"] is True
    assert calls["use_fa3"] is False
    assert calls["use_rope_real"] is True
    assert "gpus_to_use" not in calls
    assert "device" not in calls


def test_start_video_session_drops_unsupported_offload_state_kwarg(caplog):
    class LegacyModel:
        def __init__(self):
            self.calls = []

        def init_state(self, resource_path):
            self.calls.append({"resource_path": resource_path})
            return {"resource_path": resource_path}

    class Predictor:
        def __init__(self):
            self.model = LegacyModel()

        def handle_request(self, request):
            self.model.init_state(
                resource_path=request["resource_path"],
                offload_state_to_cpu=False,
            )
            return {"session_id": "session-1"}

    selection = infer_image_selection("stack", (3, 4, 5))
    bundle = PromptBundle(task=Sam3Task.SEGMENT_3D, image=selection)
    adapter = Sam3Adapter()
    adapter.video_predictor = Predictor()

    caplog.set_level(logging.WARNING, logger=backend.__name__)
    session = adapter.start_video_session(np.zeros((3, 4, 5), dtype=np.uint8), bundle)

    assert session.session_id == "session-1"
    assert len(adapter.video_predictor.model.calls) == 1
    assert "resource_path" in adapter.video_predictor.model.calls[0]
    assert any(
        "does not support offload_state_to_cpu" in record.message for record in caplog.records
    )


def test_start_video_session_preserves_supported_offload_state_kwarg(caplog):
    class ModernModel:
        def __init__(self):
            self.calls = []

        def init_state(self, resource_path, offload_state_to_cpu=False):
            self.calls.append(
                {
                    "resource_path": resource_path,
                    "offload_state_to_cpu": offload_state_to_cpu,
                }
            )
            return {"resource_path": resource_path}

    class Predictor:
        def __init__(self):
            self.model = ModernModel()

        def handle_request(self, request):
            self.model.init_state(
                resource_path=request["resource_path"],
                offload_state_to_cpu=False,
            )
            return {"session_id": "session-1"}

    selection = infer_image_selection("stack", (3, 4, 5))
    bundle = PromptBundle(task=Sam3Task.SEGMENT_3D, image=selection)
    adapter = Sam3Adapter()
    adapter.video_predictor = Predictor()

    caplog.set_level(logging.WARNING, logger=backend.__name__)
    session = adapter.start_video_session(np.zeros((3, 4, 5), dtype=np.uint8), bundle)

    assert session.session_id == "session-1"
    assert adapter.video_predictor.model.calls == [
        {
            "resource_path": str(session.resource_path),
            "offload_state_to_cpu": False,
        }
    ]
    assert not caplog.records


def test_sam31_checkpoint_rejects_current_2d_image_loader():
    adapter = Sam3Adapter(
        Sam3AdapterConfig(
            checkpoint_path=Path("/models/sam3.1_multiplex.pt"),
            device="cpu",
        )
    )

    try:
        adapter.load_image()
    except RuntimeError as error:
        assert "SAM3.1 multiplex checkpoints are supported for 3D/video" in str(error)
    else:
        raise AssertionError("Expected SAM3.1 image loading to fail clearly")


def test_2d_box_prompts_use_instance_predictor_and_clip_masks(monkeypatch):
    class InteractiveModel:
        def __init__(self):
            self.calls = []
            self.inst_interactive_predictor = types.SimpleNamespace(model=object())

        def predict_inst(
            self,
            state,
            *,
            point_coords,
            point_labels,
            box,
            mask_input,
            multimask_output,
            normalize_coords,
        ):
            self.calls.append(
                {
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "box": np.asarray(box),
                    "multimask_output": multimask_output,
                    "normalize_coords": normalize_coords,
                }
            )
            mask = np.ones((1, 10, 20), dtype=bool)
            scores = np.asarray([0.9], dtype=np.float32)
            low_res = np.zeros((1, 10, 20), dtype=np.float32)
            return mask, scores, low_res

    adapter = Sam3Adapter()
    adapter.image_model = InteractiveModel()
    adapter.image_processor = object()
    monkeypatch.setattr(adapter, "_initial_state_for_image", lambda rgb, bundle, cache_context=None: {})
    monkeypatch.setattr(adapter, "_add_geometric_prompt", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("2D box-only prompts should not use geometric grounding")
    ))

    selection = infer_image_selection("image", (10, 20))
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_2D,
        image=selection,
        boxes=[
            BoxPrompt(y0=1, x0=2, y1=4, x1=5),
            BoxPrompt(y0=6, x0=10, y1=9, x1=14),
        ],
    )

    result = adapter.run_image(np.zeros((10, 20), dtype=np.uint8), bundle)

    assert len(adapter.image_model.calls) == 2
    np.testing.assert_array_equal(
        adapter.image_model.calls[0]["box"],
        np.asarray([[2.0, 1.0, 5.0, 4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        adapter.image_model.calls[1]["box"],
        np.asarray([[10.0, 6.0, 14.0, 9.0]], dtype=np.float32),
    )
    assert adapter.image_model.calls[0]["multimask_output"] is False
    assert adapter.image_model.calls[0]["normalize_coords"] is True
    assert result.masks.shape == (2, 10, 20)
    assert result.labels.shape == (10, 20)
    assert np.all(result.masks[0, 1:4, 2:5])
    assert not np.any(result.masks[0, :1, :])
    assert not np.any(result.masks[0, 4:, :])
    assert not np.any(result.masks[0, :, :2])
    assert not np.any(result.masks[0, :, 5:])
    assert np.all(result.masks[1, 6:9, 10:14])
    assert not np.any(result.masks[1, :6, :])
    assert not np.any(result.masks[1, 9:, :])
    assert not np.any(result.masks[1, :, :10])
    assert not np.any(result.masks[1, :, 14:])
    np.testing.assert_array_equal(
        result.boxes_xyxy,
        np.asarray([[2.0, 1.0, 5.0, 4.0], [10.0, 6.0, 14.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(result.object_ids, np.asarray([1, 2]))


def test_video_point_prompt_request_uses_normalized_points_and_object_id():
    requests = []

    class Predictor:
        _all_inference_states = {
            "session-1": {
                "state": {
                    "num_frames": 3,
                    "cached_frame_outputs": {},
                },
            },
        }

        def handle_request(self, request):
            requests.append(request)
            return {
                "frame_index": request["frame_index"],
                "outputs": {
                    "out_binary_masks": np.zeros((0, 10, 20), dtype=bool),
                    "out_obj_ids": np.asarray([], dtype=np.int64),
                    "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                },
            }

    selection = infer_image_selection("stack", (3, 10, 20), dims_current_step=(1, 0, 0))
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        points=[
            PointPrompt(y=2, x=4, polarity=PromptPolarity.POSITIVE),
            PointPrompt(y=8, x=10, polarity=PromptPolarity.NEGATIVE),
        ],
    )
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = Predictor()

    adapter.add_video_prompt(bundle, session)

    assert adapter.video_predictor._all_inference_states["session-1"]["state"][
        "cached_frame_outputs"
    ] == {0: {}, 1: {}, 2: {}}
    assert requests == [
        {
            "type": "add_prompt",
            "session_id": "session-1",
            "frame_index": 1,
            "obj_id": 1,
            "points": [(0.2, 0.2), (0.5, 0.8)],
            "point_labels": [1, 0],
        }
    ]


def test_video_point_prompt_uses_explicit_single_object_id():
    requests = []

    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {"frame_index": 0, "outputs": {}}

    selection = infer_image_selection("stack", (3, 10, 20))
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        points=[PointPrompt(y=2, x=4, object_id=7)],
    )
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = Predictor()

    adapter.add_video_prompt(bundle, session)

    assert requests[0]["obj_id"] == 7


def test_video_point_prompt_rejects_mixed_text_or_box_prompt():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        points=[PointPrompt(y=2, x=4)],
        boxes=[BoxPrompt(y0=1, x0=2, y1=3, x1=4)],
        text=TextPrompt("cell"),
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "cannot be combined with text or box prompts" in str(error)
    else:
        raise AssertionError("Expected mixed video point prompts to fail clearly")


def test_video_prompt_rejects_labels_mask_prompt():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        masks=[MaskPrompt(mask=np.ones((10, 20), dtype=bool))],
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "Labels-mask prompts are not supported" in str(error)
    else:
        raise AssertionError("Expected labels-mask video prompt to fail clearly")


def test_video_point_prompt_rejects_more_than_sixteen_points():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        points=[PointPrompt(y=index, x=index) for index in range(17)],
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "up to 16 points" in str(error)
    else:
        raise AssertionError("Expected more than sixteen video points to fail clearly")


def test_video_point_prompt_rejects_multiple_object_ids():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        points=[PointPrompt(y=2, x=4, object_id=7), PointPrompt(y=3, x=5, object_id=8)],
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "must target one object id" in str(error)
    else:
        raise AssertionError("Expected multiple object ids to fail clearly")


def test_sam30_video_box_prompt_rejects_multiple_initial_boxes():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        boxes=[
            BoxPrompt(y0=1, x0=2, y1=3, x1=4),
            BoxPrompt(y0=5, x0=6, y1=7, x1=8),
        ],
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "SAM3.0 3D/video propagation supports one initial visual box" in str(error)
    else:
        raise AssertionError("Expected multiple SAM3.0 video boxes to fail clearly")


def test_sam31_video_box_prompt_uses_tracker_box_points():
    requests = []

    class Predictor:
        _all_inference_states = {
            "session-1": {
                "state": {
                    "num_frames": 3,
                    "cached_frame_outputs": {},
                },
            },
        }

        def handle_request(self, request):
            requests.append(request)
            return {"frame_index": request["frame_index"], "outputs": {}}

    selection = infer_image_selection("stack", (3, 10, 20), dims_current_step=(1, 0, 0))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter(
        Sam3AdapterConfig(checkpoint_path=Path("/models/sam3.1_multiplex.pt"))
    )
    adapter.video_predictor = Predictor()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        boxes=[
            BoxPrompt(y0=1, x0=2, y1=3, x1=4),
            BoxPrompt(y0=5, x0=6, y1=7, x1=8),
        ],
    )

    adapter.add_video_prompt(bundle, session)

    assert adapter.video_predictor._all_inference_states["session-1"]["state"][
        "cached_frame_outputs"
    ] == {0: {}, 1: {}, 2: {}}
    assert requests == [
        {
            "type": "add_prompt",
            "session_id": "session-1",
            "frame_index": 1,
            "obj_id": 1,
            "points": [(0.1, 0.1), (0.2, 0.3)],
            "point_labels": [2, 3],
        },
        {
            "type": "add_prompt",
            "session_id": "session-1",
            "frame_index": 1,
            "obj_id": 2,
            "points": [(0.3, 0.5), (0.4, 0.7)],
            "point_labels": [2, 3],
        },
    ]


def test_sam31_video_text_box_prompt_keeps_semantic_box_request():
    requests = []

    class Predictor:
        def handle_request(self, request):
            requests.append(request)
            return {"frame_index": 0, "outputs": {}}

    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter(
        Sam3AdapterConfig(checkpoint_path=Path("/models/sam3.1_multiplex.pt"))
    )
    adapter.video_predictor = Predictor()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        text=TextPrompt("cell"),
        boxes=[BoxPrompt(y0=1, x0=2, y1=3, x1=4)],
    )

    adapter.add_video_prompt(bundle, session)

    assert requests[0]["text"] == "cell"
    assert requests[0]["bounding_boxes"] == [(0.1, 0.1, 0.1, 0.2)]
    assert requests[0]["bounding_box_labels"] == [1]


def test_video_box_prompt_rejects_negative_tracker_box():
    selection = infer_image_selection("stack", (3, 10, 20))
    session = Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="session-1")
    adapter = Sam3Adapter()
    adapter.video_predictor = object()
    bundle = PromptBundle(
        task=Sam3Task.SEGMENT_3D,
        image=selection,
        boxes=[BoxPrompt(y0=1, x0=2, y1=3, x1=4, polarity=PromptPolarity.NEGATIVE)],
    )

    try:
        adapter.add_video_prompt(bundle, session)
    except RuntimeError as error:
        assert "positive tracking boxes" in str(error)
    else:
        raise AssertionError("Expected negative video box prompt to fail clearly")


def test_has_video_session_checks_backend_session_registry():
    class Predictor:
        _all_inference_states = {"active-session": {"state": {}}}

    selection = infer_image_selection("stack", (3, 10, 20))
    adapter = Sam3Adapter()
    adapter.video_predictor = Predictor()

    assert adapter.has_video_session(
        Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="active-session")
    )
    assert not adapter.has_video_session(
        Sam3Session(task=Sam3Task.SEGMENT_3D, image=selection, session_id="expired-session")
    )
