import sys
import types

import numpy as np
import torch

import napari_sam3_assistant.adapters.sam3_backend as backend
from napari_sam3_assistant.adapters import Sam3Adapter, Sam3AdapterConfig


def test_to_numpy_casts_bfloat16_to_float32_before_numpy_conversion():
    adapter = Sam3Adapter()
    tensor = torch.asarray([1.0, 2.0], dtype=torch.bfloat16)

    array = adapter._to_numpy(tensor)

    assert array.dtype == np.float32
    np.testing.assert_allclose(array, np.asarray([1.0, 2.0], dtype=np.float32))


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
