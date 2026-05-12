from __future__ import annotations

import numpy as np
from napari.viewer import Viewer

from ..core.models import Sam3Result


class LayerWriter:
    def __init__(self, viewer: Viewer) -> None:
        self.viewer = viewer

    def add_dummy_mask(self, image_layer_name: str = "SAM3 Mask") -> None:
        data = np.zeros((256, 256), dtype=np.uint8)
        data[80:180, 100:200] = 1
        self.viewer.add_labels(data, name=image_layer_name)

    def write_result(
        self,
        result: Sam3Result,
        *,
        labels_name: str = "SAM3 labels",
        mask_name: str = "SAM3 mask probabilities",
        boxes_name: str = "SAM3 boxes",
        update_boxes: bool = True,
    ) -> None:
        transform = self._result_transform(result)

        if result.labels is not None:
            self._upsert_labels(labels_name, result.labels, transform=transform)
        elif result.masks is not None:
            self._upsert_image(mask_name, result.masks.astype(np.float32), transform=transform)

        if update_boxes and result.boxes_xyxy is not None and len(result.boxes_xyxy):
            self._upsert_boxes(boxes_name, result.boxes_xyxy, result)

    def write_video_frame_result(
        self,
        result: Sam3Result,
        output_shape: tuple[int, ...],
        *,
        labels_name: str = "SAM3 propagated labels",
    ) -> None:
        if result.labels is None or result.frame_index is None:
            return
        layer = self._get_layer(labels_name)
        if layer is None:
            data = np.zeros(output_shape, dtype=np.uint32)
            layer = self.viewer.add_labels(data, name=labels_name)
        existing = np.asarray(layer.data[result.frame_index])
        incoming = np.asarray(result.labels)
        if existing.any() and not incoming.any():
            return
        layer.data[result.frame_index] = result.labels
        layer.refresh()

    def _upsert_labels(
        self,
        name: str,
        data: np.ndarray,
        *,
        transform: dict[str, object] | None = None,
    ) -> None:
        arr = data.astype(np.uint32, copy=False)
        layer = self._get_layer(name)
        if layer is None:
            kwargs = {"name": name}
            if transform:
                kwargs.update(transform)
            self.viewer.add_labels(arr, **kwargs)
        else:
            layer.data = arr
            self._apply_transform(layer, transform, arr.ndim)
            layer.refresh()

    def _upsert_image(
        self,
        name: str,
        data: np.ndarray,
        *,
        transform: dict[str, object] | None = None,
    ) -> None:
        layer = self._get_layer(name)
        if layer is None:
            kwargs = {"name": name}
            if transform:
                kwargs.update(transform)
            self.viewer.add_image(data, **kwargs)
        else:
            layer.data = data
            self._apply_transform(layer, transform, data.ndim)
            layer.refresh()

    def _result_transform(self, result: Sam3Result) -> dict[str, object] | None:
        result_space = str(result.metadata.get("result_space") or "")
        if result_space == "global_image":
            image_layer_name = result.metadata.get("image_layer")
            if not image_layer_name:
                return None
            source = self._get_layer(str(image_layer_name))
            if source is None:
                return None
            return self._layer_transform(source)
        if result_space != "roi_local":
            return None
        roi = result.metadata.get("large_image_roi")
        if not roi:
            return None
        y0, x0, _y1, _x1 = roi
        return {"translate": (float(y0), float(x0))}

    def _layer_transform(self, layer) -> dict[str, object]:
        transform: dict[str, object] = {}
        for attr in ("scale", "translate", "rotate", "shear", "affine"):
            try:
                value = getattr(layer, attr)
            except Exception:
                continue
            if value is not None:
                transform[attr] = value
        return transform

    def _apply_transform(
        self,
        layer,
        transform: dict[str, object] | None,
        ndim: int,
    ) -> None:
        if transform:
            for attr, value in transform.items():
                try:
                    setattr(layer, attr, value)
                except Exception:
                    pass
            return
        try:
            layer.translate = (0.0,) * ndim
        except Exception:
            pass

    def _upsert_boxes(
        self,
        name: str,
        boxes_xyxy: np.ndarray,
        result: Sam3Result,
    ) -> None:
        rectangles = []
        for x0, y0, x1, y1 in np.asarray(boxes_xyxy):
            rectangles.append(
                np.asarray(
                    [
                        [y0, x0],
                        [y0, x1],
                        [y1, x1],
                        [y1, x0],
                    ],
                    dtype=float,
                )
            )

        properties = {}
        if result.scores is not None:
            properties["score"] = np.asarray(result.scores)
        if result.object_ids is not None:
            properties["object_id"] = np.asarray(result.object_ids)

        layer = self._get_layer(name)
        if layer is None:
            self.viewer.add_shapes(
                rectangles,
                shape_type="rectangle",
                name=name,
                properties=properties or None,
            )
        else:
            layer.data = rectangles
            if properties:
                layer.properties = properties

    def _get_layer(self, name: str):
        try:
            return self.viewer.layers[name]
        except (KeyError, ValueError):
            return None
