from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tifffile


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
            max_value = int(arr.max(initial=0))
            if max_value > np.iinfo(np.uint16).max:
                raise ValueError("PNG export supports label values up to 65535. Use TIFF or NumPy for this mask.")
            Image.fromarray(arr.astype(np.uint16 if max_value > 255 else np.uint8)).save(target)
            return target
        if target.suffix.lower() not in {".tif", ".tiff"}:
            target = target.with_suffix(".tif")
        tif_arr = self._tiff_labels_array(arr)
        if tif_arr.ndim in {2, 3}:
            tifffile.imwrite(target, tif_arr, bigtiff=True, photometric="minisblack")
            return target
        raise ValueError("TIFF export supports 2D masks or 3D stacks.")

    def _tiff_labels_array(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data)
        if arr.size == 0:
            return arr.astype(np.uint16, copy=False)
        min_value = int(arr.min(initial=0))
        max_value = int(arr.max(initial=0))
        if min_value < 0:
            raise ValueError("TIFF label export requires non-negative label values.")
        if max_value <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16, copy=False)
        if max_value <= np.iinfo(np.uint32).max:
            return arr.astype(np.uint32, copy=False)
        raise ValueError("TIFF label export supports label values up to uint32.")
