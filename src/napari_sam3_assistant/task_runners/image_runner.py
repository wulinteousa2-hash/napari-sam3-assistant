from __future__ import annotations

from typing import Any

from napari.qt.threading import thread_worker

from ..core.coordinates import (
    RoiBounds,
    extract_2d_roi,
    globalize_result_arrays,
    localize_bundle_to_roi,
    roi_anchor_from_bundle,
)
from ..core.models import PromptBundle, Sam3Result, Sam3Task


class ImageTaskRunner:
    def __init__(self, widget: Any) -> None:
        self.widget = widget

    def run_current_task(self) -> None:
        w = self.widget
        if w._current_model_type() == "sam3.1" and w._current_task() != Sam3Task.SEGMENT_3D:
            w._log(
                "SAM3.1 video multiplex supports 3D/video propagation in this plugin. "
                "Choose task '3D/video propagation', or switch Model type to "
                "'SAM3.0 2D/3D/video' for 2D image tasks."
            )
            return

        if w.batch_all_images_check.isChecked() or w._multi_text_prompts():
            self.run_batch_current_task()
            return

        try:
            bundle = w._collect_bundle()
        except Exception as exc:
            w._log(f"Cannot collect prompts: {exc}")
            return

        if not bundle.has_prompt():
            w._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            return

        w._clear_results_table()
        if bundle.task == Sam3Task.SEGMENT_3D:
            w.video_runner.run_video_task(bundle)
        else:
            self.run_image_task(bundle)

    def run_batch_current_task(self) -> None:
        w = self.widget
        if w._current_task() == Sam3Task.SEGMENT_3D:
            w._log("Batch all image layers is for 2D image tasks. Use one stack for 3D/video propagation.")
            return
        if w._current_task() == Sam3Task.REFINE:
            w._log("Batch mode is disabled for live refinement. Select one image for point refinement.")
            return
        if w._multi_text_prompts() and w._current_task() != Sam3Task.TEXT:
            w._log("Multi-text batch prompts require task 'Text segmentation'.")
            return
        try:
            bundles = self.collect_batch_bundles()
        except Exception as exc:
            w._log(f"Cannot collect batch prompts: {exc}")
            return
        if not bundles:
            w._log("No image layers found for batch segmentation.")
            return
        if not any(bundle.has_prompt() for bundle in bundles):
            w._log("No prompts found. Add text, points, boxes, labels, or exemplar ROIs.")
            return
        w._clear_results_table()
        self.run_batch_image_task(bundles)

    def collect_batch_bundles(self) -> list[PromptBundle]:
        w = self.widget
        if w.viewer is None:
            raise RuntimeError("No napari viewer was provided to the widget.")
        channel_axis = w.channel_axis_spin.value()
        bundles: list[PromptBundle] = []
        image_layer_names = (
            w._layer_names({"image"})
            if w.batch_all_images_check.isChecked()
            else [w._current_image_layer_name()]
        )
        text_prompts = w._multi_text_prompts() or [w.text_prompt_edit.text()]
        for layer_name in image_layer_names:
            if not layer_name:
                continue
            for text_prompt in text_prompts:
                bundle = w.prompt_collector.collect(
                    w.viewer,
                    image_layer_name=layer_name,
                    task=w._current_task(),
                    points_layer_name=w._optional_combo_data(w.points_layer_combo),
                    shapes_layer_name=w._optional_combo_data(w.shapes_layer_combo),
                    labels_layer_name=w._optional_combo_data(w.labels_layer_combo),
                    text=text_prompt,
                    channel_axis=None if channel_axis < 0 else channel_axis,
                    collect_exemplar_rois=not w._large_image_mode_enabled(),
                )
                bundles.append(bundle)
        return bundles

    def run_batch_image_task(self, bundles: list[PromptBundle]) -> None:
        w = self.widget
        if w.viewer is None or w.layer_writer is None:
            w._log("No napari viewer was provided to the widget.")
            return
        try:
            adapter = w._ensure_adapter()
        except Exception as exc:
            w._log(f"Cannot run batch image task: {exc}")
            return
        local_jobs: dict[int, RoiBounds] = {}
        if w._large_image_mode_enabled():
            for index, bundle in enumerate(bundles):
                anchor = roi_anchor_from_bundle(bundle)
                if anchor is None:
                    continue
                bounds = w._active_or_new_roi_bounds(
                    bundle,
                    anchor,
                    w._selection_image_hw(bundle.image),
                    w._selected_roi_size(),
                )
                w._active_rois[bundle.image.layer_name] = bounds
                local_jobs[index] = bounds
            if local_jobs:
                w._show_active_roi_overlay(
                    "batch",
                    None,
                    extra_bounds=[
                        (bundles[index].image.layer_name, bounds)
                        for index, bounds in local_jobs.items()
                    ],
                )

        @thread_worker
        def run_batch():
            for index, bundle in enumerate(bundles):
                image_layer = w.viewer.layers[bundle.image.layer_name]
                bounds = local_jobs.get(index)
                if bounds is not None:
                    roi_data = extract_2d_roi(image_layer.data, bundle.image, bounds)
                    local_bundle = localize_bundle_to_roi(bundle, bounds, tuple(roi_data.shape))
                    result = adapter.run_image(
                        roi_data,
                        local_bundle,
                        cache_context=w._cache_context_for_layer(
                            image_layer,
                            bundle,
                            roi_bounds=bounds,
                        ),
                    )
                    labels, masks, boxes = globalize_result_arrays(
                        labels=result.labels,
                        masks=result.masks,
                        boxes_xyxy=result.boxes_xyxy,
                        bounds=bounds,
                        image_hw=w._selection_image_hw(bundle.image),
                    )
                    result.labels = labels
                    result.masks = masks
                    result.boxes_xyxy = boxes
                    result.metadata["large_image_roi"] = (bounds.y0, bounds.x0, bounds.y1, bounds.x1)
                    result.metadata["large_image_mode"] = True
                    result.metadata["large_image_hw"] = w._selection_image_hw(bundle.image)
                    result.metadata["result_space"] = "global_image"
                else:
                    result = adapter.run_image(
                        image_layer.data,
                        bundle,
                        cache_context=w._cache_context_for_layer(image_layer, bundle),
                    )
                result.metadata["image_layer"] = bundle.image.layer_name
                if bundle.text and bundle.text.text:
                    result.metadata["batch_prompt"] = bundle.text.text
                yield result

        worker = run_batch()
        worker.yielded.connect(w._write_batch_image_result)
        w._start_worker(worker)
        image_count = len({bundle.image.layer_name for bundle in bundles})
        prompt_count = len({bundle.text.text for bundle in bundles if bundle.text and bundle.text.text}) or 1
        w._log(
            f"Running {len(bundles)} batch job(s): {image_count} image layer(s), "
            f"{prompt_count} prompt(s)."
        )
        if w._large_image_mode_enabled():
            w._log(
                f"Large-image mode ON: local ROI inference for {len(local_jobs)} "
                "anchored batch job(s); jobs without point/box anchors use full-image inference."
            )
        else:
            w._log("Large-image mode OFF: full-image inference.")

    def run_image_task(self, bundle: PromptBundle) -> None:
        w = self.widget
        if w.viewer is None or w.layer_writer is None:
            w._log("No napari viewer was provided to the widget.")
            return
        if w._large_image_mode_enabled():
            anchor = roi_anchor_from_bundle(bundle)
            if anchor is not None:
                self.run_large_image_task(bundle, anchor)
                return
            w._log(
                "Large-image mode ON, but no point or box ROI anchor was found. "
                "Using full-image inference for this task."
            )
        image_layer = w.viewer.layers[bundle.image.layer_name]
        try:
            adapter = w._ensure_adapter()
        except Exception as exc:
            w._log(f"Cannot run image task: {exc}")
            return

        @thread_worker
        def run_image() -> Sam3Result:
            result = adapter.run_image(
                image_layer.data,
                bundle,
                cache_context=w._cache_context_for_layer(image_layer, bundle),
            )
            result.metadata["image_layer"] = bundle.image.layer_name
            return result

        worker = run_image()
        worker.returned.connect(w._write_image_result)
        w._start_worker(worker)
        w._log(f"Running {bundle.task.value} on image layer '{bundle.image.layer_name}'.")
        if w._large_image_mode_enabled():
            w._log("Large-image mode OFF for this run: no local ROI anchor available.")
        else:
            w._log("Large-image mode OFF: full-image inference.")

    def run_large_image_task(self, bundle: PromptBundle, anchor: tuple[float, float]) -> None:
        w = self.widget
        if w.viewer is None or w.layer_writer is None:
            w._log("No napari viewer was provided to the widget.")
            return
        image_layer = w.viewer.layers[bundle.image.layer_name]
        image_hw = w._selection_image_hw(bundle.image)
        roi_size = w._selected_roi_size()
        bounds = w._active_or_new_roi_bounds(bundle, anchor, image_hw, roi_size)
        w._active_rois[bundle.image.layer_name] = bounds
        w._show_active_roi_overlay(bundle.image.layer_name, bounds)
        try:
            adapter = w._ensure_adapter()
        except Exception as exc:
            w._log(f"Cannot run local ROI task: {exc}")
            return

        @thread_worker
        def run_local_roi() -> Sam3Result:
            roi_data = extract_2d_roi(image_layer.data, bundle.image, bounds)
            local_bundle = localize_bundle_to_roi(bundle, bounds, tuple(roi_data.shape))
            result = adapter.run_image(
                roi_data,
                local_bundle,
                cache_context=w._cache_context_for_layer(
                    image_layer,
                    bundle,
                    roi_bounds=bounds,
                ),
            )
            labels, masks, boxes = globalize_result_arrays(
                labels=result.labels,
                masks=result.masks,
                boxes_xyxy=result.boxes_xyxy,
                bounds=bounds,
                image_hw=image_hw,
            )
            result.labels = labels
            result.masks = masks
            result.boxes_xyxy = boxes
            result.metadata["image_layer"] = bundle.image.layer_name
            result.metadata["large_image_roi"] = (bounds.y0, bounds.x0, bounds.y1, bounds.x1)
            result.metadata["large_image_mode"] = True
            result.metadata["large_image_hw"] = image_hw
            result.metadata["result_space"] = "global_image"
            return result

        worker = run_local_roi()
        worker.returned.connect(w._write_image_result)
        w._start_worker(worker)
        w._log(
            f"Large-image mode ON: local ROI inference ({bounds.width} x {bounds.height}); "
            f"ROI y={bounds.y0}:{bounds.y1}, x={bounds.x0}:{bounds.x1}."
        )
