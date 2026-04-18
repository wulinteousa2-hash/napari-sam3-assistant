from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptState:
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    exemplar_layer_name: str | None = None
    text_prompt: str = ""


class PromptStateService:
    def __init__(self) -> None:
        self.state = PromptState()

    def clear(self) -> None:
        self.state = PromptState()

    def add_positive_point(self, y: float, x: float) -> None:
        self.state.positive_points.append((y, x))

    def add_negative_point(self, y: float, x: float) -> None:
        self.state.negative_points.append((y, x))

    def set_text_prompt(self, text: str) -> None:
        self.state.text_prompt = text.strip()

    def summary(self) -> str:
        return (
            f"positive={len(self.state.positive_points)}, "
            f"negative={len(self.state.negative_points)}, "
            f"boxes={len(self.state.boxes)}, "
            f"text={'yes' if self.state.text_prompt else 'no'}"
        )