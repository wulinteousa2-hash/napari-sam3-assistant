"""Downstream mask operations workflow for napari-sam3-assistant."""

__all__ = ["MaskOperationsPanel"]


def __getattr__(name: str):
    if name == "MaskOperationsPanel":
        from .panel import MaskOperationsPanel

        return MaskOperationsPanel
    raise AttributeError(name)
