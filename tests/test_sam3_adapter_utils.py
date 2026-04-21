from pathlib import Path
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


def test_sam31_video_box_prompt_allows_multiple_boxes():
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
        boxes=[
            BoxPrompt(y0=1, x0=2, y1=3, x1=4),
            BoxPrompt(y0=5, x0=6, y1=7, x1=8),
        ],
    )

    adapter.add_video_prompt(bundle, session)

    assert requests[0]["bounding_boxes"] == [
        (0.1, 0.1, 0.1, 0.2),
        (0.3, 0.5, 0.1, 0.2),
    ]
