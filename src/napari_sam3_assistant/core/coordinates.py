from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .models import BoxPrompt, ImageSelection, MaskPrompt, PromptBundle


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


@dataclass(frozen=True)
class RoiBounds:
    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    def contains_yx(self, y: float, x: float) -> bool:
        return self.y0 <= y < self.y1 and self.x0 <= x < self.x1

    def as_rectangle(self) -> np.ndarray:
        return np.asarray(
            [
                [self.y0, self.x0],
                [self.y0, self.x1],
                [self.y1, self.x1],
                [self.y1, self.x0],
            ],
            dtype=float,
        )


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
    arr = np.asarray(base_image_data(data))
    index: list[int | slice] = [slice(None)] * arr.ndim

    if selection.frame_axis is not None:
        index[selection.frame_axis] = selection.frame_index or 0
    if selection.channel_axis is not None and selection.channel_axis != arr.ndim - 1:
        index[selection.channel_axis] = selection.channel_index or 0

    sliced = arr[tuple(index)]
    while sliced.ndim > 2 and sliced.shape[-1] not in (3, 4):
        sliced = sliced[0]
    return np.asarray(sliced)


def extract_2d_roi(
    data: np.ndarray,
    selection: ImageSelection,
    bounds: RoiBounds,
) -> np.ndarray:
    """Extract only the selected 2D ROI from a possibly lazy image array."""
    data = _base_image_data(data)
    ndim = len(selection.data_shape)
    index: list[int | slice] = [slice(None)] * ndim

    if selection.frame_axis is not None:
        index[selection.frame_axis] = selection.frame_index or 0
    if selection.channel_axis is not None and selection.channel_axis != ndim - 1:
        index[selection.channel_axis] = selection.channel_index or 0

    y_axis, x_axis = selection.spatial_axes
    index[y_axis] = slice(bounds.y0, bounds.y1)
    index[x_axis] = slice(bounds.x0, bounds.x1)

    sliced = np.asarray(base_image_data(data)[tuple(index)])
    while sliced.ndim > 2 and sliced.shape[-1] not in (3, 4):
        sliced = sliced[0]
    return sliced


def base_image_data(data):
    """Return the full-resolution level for napari multiscale image data."""
    if isinstance(data, (list, tuple)):
        return data[0]
    try:
        return data[0] if hasattr(data, "_data") and hasattr(data, "__len__") and len(data) else data
    except Exception:
        return data


def _base_image_data(data):
    """Backward-compatible alias for older imports."""
    return base_image_data(data)


def centered_roi_bounds(
    anchor_y: float,
    anchor_x: float,
    *,
    image_hw: tuple[int, int],
    roi_hw: tuple[int, int],
) -> RoiBounds:
    height, width = image_hw
    roi_h = min(int(roi_hw[0]), height)
    roi_w = min(int(roi_hw[1]), width)
    y0 = int(round(float(anchor_y) - roi_h / 2))
    x0 = int(round(float(anchor_x) - roi_w / 2))
    y0 = max(0, min(height - roi_h, y0))
    x0 = max(0, min(width - roi_w, x0))
    return RoiBounds(y0=y0, x0=x0, y1=y0 + roi_h, x1=x0 + roi_w)


def box_roi_bounds(
    box: BoxPrompt,
    *,
    image_hw: tuple[int, int],
    roi_hw: tuple[int, int],
) -> RoiBounds:
    y0, y1 = sorted((float(box.y0), float(box.y1)))
    x0, x1 = sorted((float(box.x0), float(box.x1)))
    box_h = max(1.0, y1 - y0)
    box_w = max(1.0, x1 - x0)
    roi_h = max(int(roi_hw[0]), int(np.ceil(box_h)))
    roi_w = max(int(roi_hw[1]), int(np.ceil(box_w)))
    return centered_roi_bounds(
        (y0 + y1) * 0.5,
        (x0 + x1) * 0.5,
        image_hw=image_hw,
        roi_hw=(roi_h, roi_w),
    )


def roi_anchor_from_bundle(bundle: PromptBundle) -> tuple[float, float] | None:
    if bundle.points:
        point = bundle.points[-1]
        return point.y, point.x
    if bundle.boxes:
        box = bundle.boxes[-1]
        return (float(box.y0) + float(box.y1)) * 0.5, (float(box.x0) + float(box.x1)) * 0.5
    return None


def localize_bundle_to_roi(
    bundle: PromptBundle,
    bounds: RoiBounds,
    roi_shape: tuple[int, ...],
) -> PromptBundle:
    channel_axis = len(roi_shape) - 1 if len(roi_shape) >= 3 and roi_shape[-1] in (3, 4) else None
    local_image = infer_image_selection(
        layer_name=bundle.image.layer_name,
        data_shape=tuple(int(v) for v in roi_shape),
        channel_axis=channel_axis,
    )
    points = [
        replace(point, y=float(point.y) - bounds.y0, x=float(point.x) - bounds.x0)
        for point in bundle.points
        if bounds.contains_yx(point.y, point.x)
    ]
    boxes = []
    for box in bundle.boxes:
        y0 = max(0.0, min(float(bounds.height), float(box.y0) - bounds.y0))
        y1 = max(0.0, min(float(bounds.height), float(box.y1) - bounds.y0))
        x0 = max(0.0, min(float(bounds.width), float(box.x0) - bounds.x0))
        x1 = max(0.0, min(float(bounds.width), float(box.x1) - bounds.x0))
        if abs(y1 - y0) > 0 and abs(x1 - x0) > 0:
            boxes.append(replace(box, y0=y0, x0=x0, y1=y1, x1=x1))

    masks = []
    for mask_prompt in bundle.masks:
        mask = np.asarray(mask_prompt.mask)
        if mask.ndim == 2:
            local_mask = mask[bounds.y0:bounds.y1, bounds.x0:bounds.x1]
            if local_mask.any():
                masks.append(MaskPrompt(mask=local_mask.astype(bool), frame_index=mask_prompt.frame_index))

    return PromptBundle(
        task=bundle.task,
        image=local_image,
        points=points,
        boxes=boxes,
        masks=masks,
        exemplars=[],
        text=bundle.text,
    )


def globalize_result_arrays(
    *,
    labels: np.ndarray | None,
    masks: np.ndarray | None,
    boxes_xyxy: np.ndarray | None,
    bounds: RoiBounds,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    global_labels = None
    if labels is not None:
        global_labels = np.zeros(image_hw, dtype=np.asarray(labels).dtype)
        global_labels[bounds.y0:bounds.y1, bounds.x0:bounds.x1] = np.asarray(labels)

    global_masks = None
    if masks is not None:
        local_masks = np.asarray(masks)
        if local_masks.ndim >= 3:
            global_masks = np.zeros((local_masks.shape[0], *image_hw), dtype=local_masks.dtype)
            global_masks[:, bounds.y0:bounds.y1, bounds.x0:bounds.x1] = local_masks
        elif local_masks.ndim == 2:
            global_masks = np.zeros(image_hw, dtype=local_masks.dtype)
            global_masks[bounds.y0:bounds.y1, bounds.x0:bounds.x1] = local_masks

    global_boxes = None
    if boxes_xyxy is not None:
        global_boxes = np.asarray(boxes_xyxy).copy()
        if global_boxes.size:
            global_boxes[:, [0, 2]] += bounds.x0
            global_boxes[:, [1, 3]] += bounds.y0

    return global_labels, global_masks, global_boxes


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
