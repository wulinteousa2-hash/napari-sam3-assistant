from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import BoxPrompt, ImageSelection


@dataclass(frozen=True)
class CoordinateMapper:
    selection: ImageSelection

    def point_to_xy(self, y: float, x: float) -> tuple[float, float]:
        return (float(x), float(y))

    def box_to_xyxy(self, box: BoxPrompt) -> tuple[float, float, float, float]:
        y0, y1 = sorted((float(box.y0), float(box.y1)))
        x0, x1 = sorted((float(box.x0), float(box.x1)))
        return (x0, y0, x1, y1)

    def box_to_normalized_cxcywh(
        self,
        box: BoxPrompt,
        image_hw: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Return normalized center-x, center-y, width, height for image SAM3."""
        height, width = image_hw
        x0, y0, x1, y1 = self.box_to_xyxy(box)
        x0 = np.clip(x0, 0, width)
        x1 = np.clip(x1, 0, width)
        y0 = np.clip(y0, 0, height)
        y1 = np.clip(y1, 0, height)
        cx = ((x0 + x1) * 0.5) / width
        cy = ((y0 + y1) * 0.5) / height
        w = max(0.0, x1 - x0) / width
        h = max(0.0, y1 - y0) / height
        return (float(cx), float(cy), float(w), float(h))

    def box_to_normalized_xywh(
        self,
        box: BoxPrompt,
        image_hw: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Return normalized top-left-x, top-left-y, width, height for video SAM3."""
        height, width = image_hw
        x0, y0, x1, y1 = self.box_to_xyxy(box)
        x0 = np.clip(x0, 0, width)
        x1 = np.clip(x1, 0, width)
        y0 = np.clip(y0, 0, height)
        y1 = np.clip(y1, 0, height)
        w = max(0.0, x1 - x0) / width
        h = max(0.0, y1 - y0) / height
        return (float(x0 / width), float(y0 / height), float(w), float(h))


def infer_image_selection(
    layer_name: str,
    data_shape: tuple[int, ...],
    dims_current_step: tuple[int, ...] | None = None,
    channel_axis: int | None = None,
) -> ImageSelection:
    ndim = len(data_shape)
    if ndim < 2:
        raise ValueError("SAM3 requires an image layer with at least two spatial axes.")

    if channel_axis is None and ndim >= 3 and data_shape[-1] in (3, 4):
        channel_axis = ndim - 1

    if channel_axis == ndim - 1 and ndim >= 3:
        spatial_axes = (ndim - 3, ndim - 2)
    else:
        spatial_axes = (ndim - 2, ndim - 1)

    frame_axis = None
    frame_index = None

    if ndim >= 3:
        candidates = [axis for axis in range(ndim) if axis not in spatial_axes]
        if channel_axis in candidates:
            candidates.remove(channel_axis)
        if candidates:
            frame_axis = candidates[-1]
            if dims_current_step is not None and frame_axis < len(dims_current_step):
                frame_index = int(dims_current_step[frame_axis])
            else:
                frame_index = 0

    channel_index = None
    if channel_axis is not None:
        channel_index = (
            int(dims_current_step[channel_axis])
            if dims_current_step is not None and channel_axis < len(dims_current_step)
            else 0
        )

    return ImageSelection(
        layer_name=layer_name,
        data_shape=tuple(int(v) for v in data_shape),
        frame_axis=frame_axis,
        channel_axis=channel_axis,
        spatial_axes=spatial_axes,
        frame_index=frame_index,
        channel_index=channel_index,
    )


def extract_2d_image(data: np.ndarray, selection: ImageSelection) -> np.ndarray:
    arr = np.asarray(data)
    index: list[int | slice] = [slice(None)] * arr.ndim

    if selection.frame_axis is not None:
        index[selection.frame_axis] = selection.frame_index or 0
    if selection.channel_axis is not None and selection.channel_axis != arr.ndim - 1:
        index[selection.channel_axis] = selection.channel_index or 0

    sliced = arr[tuple(index)]
    while sliced.ndim > 2 and sliced.shape[-1] not in (3, 4):
        sliced = sliced[0]
    return np.asarray(sliced)


def to_rgb_uint8(image_2d_or_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_2d_or_rgb)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[-1] != 3:
        raise ValueError("Expected grayscale, RGB, or RGBA image data for SAM3.")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo)
    return (scaled * 255).astype(np.uint8)
