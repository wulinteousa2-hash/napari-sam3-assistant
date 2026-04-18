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
    ) -> None:
        if result.labels is not None:
            self._upsert_labels(labels_name, result.labels)
        elif result.masks is not None:
            self._upsert_image(mask_name, result.masks.astype(np.float32))

        if result.boxes_xyxy is not None and len(result.boxes_xyxy):
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

    def _upsert_labels(self, name: str, data: np.ndarray) -> None:
        layer = self._get_layer(name)
        if layer is None:
            self.viewer.add_labels(data.astype(np.uint32, copy=False), name=name)
        else:
            layer.data = data.astype(np.uint32, copy=False)

    def _upsert_image(self, name: str, data: np.ndarray) -> None:
        layer = self._get_layer(name)
        if layer is None:
            self.viewer.add_image(data, name=name)
        else:
            layer.data = data

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
