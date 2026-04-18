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
        "sam3.1_multiplex.pt",
        "model.safetensors",
        "sam3.pt",
    ]

    WEIGHT_FILES_BY_MODEL_TYPE = {
        "sam3": ["sam3.pt", "model.safetensors"],
        "sam3.1": ["sam3.1_multiplex.pt"],
    }

    def validate(
        self,
        model_dir: str,
        *,
        model_type: str | None = None,
    ) -> CheckpointValidationResult:
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

        expected_weights = (
            self.WEIGHT_FILES_BY_MODEL_TYPE.get(model_type, self.OPTIONAL_WEIGHT_FILES)
            if model_type
            else self.OPTIONAL_WEIGHT_FILES
        )
        if not any((path / name).exists() for name in expected_weights):
            return CheckpointValidationResult(
                False,
                "No weight file found. Expected one of: "
                + ", ".join(expected_weights),
            )

        return CheckpointValidationResult(True, "Model directory looks valid.")
