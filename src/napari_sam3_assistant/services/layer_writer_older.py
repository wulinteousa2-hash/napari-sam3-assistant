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
        translate = self._result_translate(result)

        if result.labels is not None:
            self._upsert_labels(labels_name, result.labels, translate=translate)
        elif result.masks is not None:
            self._upsert_image(mask_name, result.masks.astype(np.float32), translate=translate)

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
        layer.data[result.frame_index] = result.labels
        layer.refresh()

    def _upsert_labels(
        self,
        name: str,
        data: np.ndarray,
        *,
        translate: tuple[float, float] | None = None,
    ) -> None:
        arr = data.astype(np.uint32, copy=False)
        layer = self._get_layer(name)
        if layer is None:
            kwargs = {"name": name}
            if translate is not None:
                kwargs["translate"] = translate
            self.viewer.add_labels(arr, **kwargs)
        else:
            layer.data = arr
            if translate is not None:
                layer.translate = translate
            else:
                try:
                    layer.translate = (0.0,) * arr.ndim
                except Exception:
                    pass
            layer.refresh()

    def _upsert_image(
        self,
        name: str,
        data: np.ndarray,
        *,
        translate: tuple[float, float] | None = None,
    ) -> None:
        layer = self._get_layer(name)
        if layer is None:
            kwargs = {"name": name}
            if translate is not None:
                kwargs["translate"] = translate
            self.viewer.add_image(data, **kwargs)
        else:
            layer.data = data
            if translate is not None:
                layer.translate = translate
            else:
                try:
                    layer.translate = (0.0,) * data.ndim
                except Exception:
                    pass
            layer.refresh()

    def _result_translate(self, result: Sam3Result) -> tuple[float, float] | None:
        roi = result.metadata.get("large_image_roi")
        if not roi:
            return None
        y0, x0, _y1, _x1 = roi
        return (float(y0), float(x0))

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