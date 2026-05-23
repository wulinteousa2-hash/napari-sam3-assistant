from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import tifffile


class MaskExportService:
    def export(
        self,
        data: Any,
        path: str | Path,
        fmt: str,
        *,
        scale: tuple[float, ...] | None = None,
    ) -> Path:
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
        if fmt_key in {"ome-zarr", "ome zarr", "ome-zarr (.ome.zarr)", "ome zarr (.ome.zarr)"}:
            if target.name.endswith(".ome.zarr"):
                zarr_target = target
            else:
                zarr_target = target.with_suffix("")
                zarr_target = zarr_target.with_name(f"{zarr_target.name}.ome.zarr")
            self._write_ome_zarr(self._tiff_labels_array(arr), zarr_target, scale=scale)
            return zarr_target
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

    def _write_ome_zarr(
        self,
        data: np.ndarray,
        target: Path,
        *,
        scale: tuple[float, ...] | None = None,
    ) -> None:
        if data.ndim not in {2, 3}:
            raise ValueError("OME-Zarr export supports 2D masks or 3D stacks.")
        try:
            from ome_zarr.scale import Methods
            from ome_zarr.writer import write_image
        except Exception as exc:
            raise RuntimeError(
                "OME-Zarr export requires the 'ome-zarr' and 'zarr' packages."
            ) from exc

        axes = ["y", "x"] if data.ndim == 2 else ["z", "y", "x"]
        scale_factors = self._ome_zarr_scale_factors(data.shape)
        coordinate_transformations = self._ome_zarr_coordinate_transformations(
            axes,
            scale,
            scale_factors,
        )
        write_image(
            data,
            str(target),
            axes=axes,
            method=Methods.NEAREST,
            scale_factors=scale_factors,
            coordinate_transformations=coordinate_transformations,
            storage_options={"chunks": self._ome_zarr_chunks(data.shape)},
        )

    def _ome_zarr_scale_factors(self, shape: tuple[int, ...]) -> list[int]:
        min_spatial = min(int(shape[-2]), int(shape[-1]))
        return [factor for factor in (2, 4, 8, 16) if min_spatial // factor >= 64]

    def _ome_zarr_chunks(self, shape: tuple[int, ...]) -> tuple[int, ...]:
        if len(shape) == 2:
            return (min(2048, int(shape[0])), min(2048, int(shape[1])))
        return (1, min(1024, int(shape[-2])), min(1024, int(shape[-1])))

    def _ome_zarr_coordinate_transformations(
        self,
        axes: list[str],
        scale: tuple[float, ...] | None,
        scale_factors: list[int],
    ) -> list[list[dict[str, list[float] | str]]] | None:
        if scale is None:
            return None
        spatial_scale = [float(value) for value in scale[-len(axes):]]
        if len(spatial_scale) != len(axes):
            return None
        transforms: list[list[dict[str, list[float] | str]]] = []
        for factor in [1, *scale_factors]:
            level_scale = [
                value if axis == "z" else value * float(factor)
                for axis, value in zip(axes, spatial_scale)
            ]
            transforms.append([{"type": "scale", "scale": level_scale}])
        return transforms
