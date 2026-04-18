from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckpointValidationResult:
    ok: bool
    message: str


class CheckpointService:
    """
    Validate local SAM3 model/checkpoint directory.
    """

    REQUIRED_FILES = [
        "config.json",
        "processor_config.json",
    ]

    OPTIONAL_WEIGHT_FILES = [
        "model.safetensors",
        "sam3.pt",
        "pytorch_model.bin",
    ]

    def validate(self, model_dir: str) -> CheckpointValidationResult:
        path = Path(model_dir)

        if not model_dir.strip():
            return CheckpointValidationResult(False, "Model directory is empty.")

        if not path.exists():
            return CheckpointValidationResult(False, f"Path does not exist: {model_dir}")

        if not path.is_dir():
            return CheckpointValidationResult(False, f"Path is not a directory: {model_dir}")

        missing = [name for name in self.REQUIRED_FILES if not (path / name).exists()]
        if missing:
            return CheckpointValidationResult(
                False,
                f"Missing required config files: {', '.join(missing)}",
            )

        if not any((path / name).exists() for name in self.OPTIONAL_WEIGHT_FILES):
            return CheckpointValidationResult(
                False,
                "No weight file found. Expected one of: "
                + ", ".join(self.OPTIONAL_WEIGHT_FILES),
            )

        return CheckpointValidationResult(True, "Checkpoint directory looks valid.")