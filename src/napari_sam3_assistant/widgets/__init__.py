from __future__ import annotations

__all__ = ["MainWidget"]


def __getattr__(name: str):
    if name == "MainWidget":
        from .main_widget import MainWidget

        return MainWidget
    raise AttributeError(name)
