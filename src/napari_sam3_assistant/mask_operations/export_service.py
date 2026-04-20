from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class MaskExportService:
    def export(self, data: Any, path: str | Path, fmt: str) -> Path:
        target = Path(path)
        arr = np.asarray(data)
        fmt_key = fmt.lower()
        if fmt_key in {"numpy (.npy)", "npy"}:
            if target.suffix.lower() != ".npy":
                target = target.with_suffix(".npy")
            np.save(target, arr)
            return target
        if fmt_key == "png":
            if arr.ndim != 2:
                raise ValueError("PNG export supports 2D masks only.")
            if target.suffix.lower() != ".png":
                target = target.with_suffix(".png")
            Image.fromarray(arr.astype(np.uint16 if arr.max(initial=0) > 255 else np.uint8)).save(target)
            return target
        if target.suffix.lower() not in {".tif", ".tiff"}:
            target = target.with_suffix(".tif")
        tif_arr = arr.astype(np.uint32 if arr.max(initial=0) > 65535 else np.uint16)
        if tif_arr.ndim == 2:
            Image.fromarray(tif_arr).save(target)
            return target
        if tif_arr.ndim == 3:
            frames = [Image.fromarray(tif_arr[index]) for index in range(tif_arr.shape[0])]
            frames[0].save(target, save_all=True, append_images=frames[1:])
            return target
        raise ValueError("TIFF export supports 2D masks or 3D stacks.")
