from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import torch


@dataclass
class ProviderStatus:
    ready: bool
    message: str


class Sam3RepoProvider:
    """
    Official SAM3 repo provider scaffold.

    Current behavior:
    - validates a local SAM3 model directory
    - inspects config.json
    - records the model directory
    - avoids accidental remote Hugging Face loading

    Real model construction should be added only after the correct
    local image/video loading path is confirmed.
    """

    def __init__(self) -> None:
        self.model_dir: str | None = None
        self.loaded: bool = False
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: Any | None = None
        self.processor: Any | None = None

        self.architectures: list[str] = []
        self.model_type: str | None = None

    def validate_model_dir(self, model_dir: str) -> ProviderStatus:
        path = Path(model_dir)

        if not model_dir.strip():
            return ProviderStatus(False, "Model directory is empty.")

        if not path.exists():
            return ProviderStatus(False, f"Model directory does not exist: {model_dir}")

        if not path.is_dir():
            return ProviderStatus(False, f"Path is not a directory: {model_dir}")

        required = ["config.json", "processor_config.json"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            return ProviderStatus(False, f"Missing required files: {', '.join(missing)}")

        weight_candidates = ["model.safetensors", "sam3.pt", "pytorch_model.bin"]
        found_weights = [name for name in weight_candidates if (path / name).exists()]
        if not found_weights:
            return ProviderStatus(
                False,
                "No supported weight file found. Expected one of: "
                + ", ".join(weight_candidates),
            )

        return ProviderStatus(
            True,
            f"Model directory looks valid. Found weights: {', '.join(found_weights)}"
        )

    def _inspect_config(self, model_dir: str) -> ProviderStatus:
        config_path = Path(model_dir) / "config.json"

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            return ProviderStatus(False, f"Failed to read config.json: {e}")

        self.architectures = cfg.get("architectures", [])
        self.model_type = cfg.get("model_type")

        arch_text = ", ".join(self.architectures) if self.architectures else "unknown"
        type_text = self.model_type or "unknown"

        return ProviderStatus(
            True,
            f"Detected architecture: {arch_text} | model_type: {type_text}"
        )

    def load(self, model_dir: str) -> ProviderStatus:
        status = self.validate_model_dir(model_dir)
        if not status.ready:
            return status

        try:
            import sam3  # noqa: F401
        except Exception as e:
            return ProviderStatus(False, f"Failed to import SAM3 backend: {e}")

        inspect_status = self._inspect_config(model_dir)
        if not inspect_status.ready:
            return inspect_status

        self.model_dir = model_dir
        self.loaded = True
        self.model = None
        self.processor = None

        return ProviderStatus(
            True,
            f"SAM3 backend import OK. Local model directory registered: {model_dir}. "
            f"{inspect_status.message}. Real model construction is not wired yet."
        )

    def unload(self) -> ProviderStatus:
        self.model = None
        self.processor = None
        self.model_dir = None
        self.loaded = False
        self.architectures = []
        self.model_type = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ProviderStatus(True, "SAM3 backend unloaded.")

    def backend_summary(self) -> str:
        if not self.loaded:
            return "SAM3 backend not loaded."

        arch_text = ", ".join(self.architectures) if self.architectures else "unknown"
        type_text = self.model_type or "unknown"

        return (
            f"SAM3 backend registered | device={self.device} | "
            f"model_dir={self.model_dir} | architecture={arch_text} | model_type={type_text}"
        )

    def predict_text(self, image_pil, text_prompt: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Real SAM3 inference is not wired yet. "
            "First confirm the correct local image/video loading API."
        )