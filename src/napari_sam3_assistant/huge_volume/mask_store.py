from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class HugeVolumeMaskStore:
    """OME-Zarr-backed mask store for direct chunk write-back."""

    def __init__(self, path: str | Path, array: Any) -> None:
        self.path = Path(path)
        self.array = array

    @classmethod
    def open(cls, path: str | Path, array_path: str = "s0") -> "HugeVolumeMaskStore":
        try:
            import zarr
        except Exception as exc:
            raise RuntimeError("Huge-volume mask editing requires the 'zarr' package.") from exc

        target = Path(path)
        group = zarr.open_group(str(target), mode="r+")
        try:
            array = group[array_path]
        except KeyError as exc:
            raise ValueError(f"OME-Zarr mask array '{array_path}' was not found in {target}.") from exc
        if len(tuple(array.shape)) != 3:
            raise ValueError(
                f"OME-Zarr mask write-back expects a 3D z/y/x mask array; got shape {tuple(array.shape)}."
            )
        return cls(target, array)

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        shape: tuple[int, int, int],
        chunks: tuple[int, int, int],
        dtype: str | np.dtype = np.uint32,
        scale: tuple[float, float, float] | None = None,
    ) -> "HugeVolumeMaskStore":
        try:
            import zarr
        except Exception as exc:
            raise RuntimeError("Huge-volume mask output requires the 'zarr' package.") from exc

        target = Path(path)
        group = zarr.open_group(str(target), mode="w")
        array = group.create_array(
            "s0",
            shape=tuple(int(value) for value in shape),
            chunks=tuple(int(value) for value in chunks),
            dtype=np.dtype(dtype),
            fill_value=0,
        )
        axes = [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
        dataset: dict[str, Any] = {"path": "s0"}
        if scale is not None:
            dataset["coordinateTransformations"] = [
                {"type": "scale", "scale": [float(value) for value in scale]}
            ]
        group.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "name": target.stem,
                    "axes": axes,
                    "datasets": [dataset],
                }
            ],
        }
        return cls(target, array)

    def read_region(self, z_index: int, y_slice: slice, x_slice: slice) -> np.ndarray:
        return np.asarray(self.array[int(z_index), y_slice, x_slice])

    def write_region(self, z_index: int, y_slice: slice, x_slice: slice, data: np.ndarray) -> None:
        self.array[int(z_index), y_slice, x_slice] = np.asarray(data, dtype=self.array.dtype)

    def write_tile_labels(
        self,
        z_index: int,
        y0: int,
        x0: int,
        labels: np.ndarray,
        *,
        next_object_id: int,
    ) -> tuple[int, int]:
        from scipy import ndimage as ndi

        local = np.asarray(labels)
        if local.ndim != 2 or local.size == 0 or not np.any(local):
            return int(next_object_id), 0

        y1 = int(y0) + int(local.shape[0])
        x1 = int(x0) + int(local.shape[1])
        y_slice = slice(int(y0), y1)
        x_slice = slice(int(x0), x1)
        target = self.read_region(z_index, y_slice, x_slice)
        components, count = ndi.label(local != 0)
        written = 0
        object_id = int(next_object_id)
        for component_id in range(1, int(count) + 1):
            mask = components == component_id
            write_mask = mask & (target == 0)
            pixels = int(np.count_nonzero(write_mask))
            if pixels == 0:
                continue
            target[write_mask] = object_id
            written += pixels
            object_id += 1
        if written:
            self.write_region(z_index, y_slice, x_slice, target)
        return object_id, written
