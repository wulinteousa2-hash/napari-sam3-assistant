from __future__ import annotations

import time
from typing import Any

from napari.qt.threading import thread_worker

from ..core.coordinates import extract_video_xy_roi, localize_bundle_to_roi
from ..core.diagnostics import Sam3Diagnostics
from ..core.models import PromptBundle


class VideoTaskRunner:
    def __init__(self, widget: Any) -> None:
        self.widget = widget

    def run_video_task(self, bundle: PromptBundle) -> None:
        w = self.widget
        if w.viewer is None or w.layer_writer is None:
            w._log("No napari viewer was provided to the widget.")
            return
        image_layer = w.viewer.layers[bundle.image.layer_name]
        try:
            adapter = w._ensure_adapter()
        except Exception as exc:
            w._log(f"Cannot run video task: {exc}")
            return
        direction = w.propagation_direction_combo.currentText()
        video_data = image_layer.data
        video_bundle = bundle
        video_roi_bounds = (
            w._video_roi_bounds_for_bundle(bundle)
            if hasattr(w, "_video_roi_bounds_for_bundle")
            else None
        )
        if video_roi_bounds is not None:
            try:
                video_data = extract_video_xy_roi(image_layer.data, bundle.image, video_roi_bounds)
                video_bundle = localize_bundle_to_roi(bundle, video_roi_bounds, tuple(video_data.shape))
            except Exception as exc:
                w._log(f"Cannot run 3D/video local ROI task: {exc}")
                return
        diagnostics = (
            Sam3Diagnostics(w._log)
            if getattr(w, "_sam31_diagnostics_enabled", lambda: False)()
            else None
        )
        if diagnostics is not None:
            diagnostics.log(
                "SAM3.1 image source: "
                f"{diagnostics.describe_image_source(video_data)}"
            )
            diagnostics.log_prompt_diagnostics(video_bundle)

        @thread_worker
        def run_video():
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before start_video_session")
                diagnostics.log_runtime_diagnostics(adapter, stage="before session start")
                session_t0 = time.perf_counter()
            session = adapter.start_video_session(video_data, video_bundle)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 session start", session_t0)
                diagnostics.log(
                    "SAM3.1 session ready: "
                    f"{session.session_id}; prompt_frame={video_bundle.image.frame_index or 0}; "
                    f"boxes={len(getattr(video_bundle, 'boxes', []) or [])}; "
                    f"points={len(getattr(video_bundle, 'points', []) or [])}."
                )
                diagnostics.log_session_diagnostics(adapter, session, stage="after session start")
                prompt_t0 = time.perf_counter()
            prompt_result = adapter.add_video_prompt(video_bundle, session)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 prompt insertion", prompt_t0)
                diagnostics.log_session_diagnostics(adapter, session, stage="after prompt insertion")
            if hasattr(w, "_globalize_video_roi_result"):
                w._globalize_video_roi_result(prompt_result, bundle, video_roi_bounds)
            prompt_result.metadata["image_layer"] = bundle.image.layer_name
            yield prompt_result
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before propagation")
                diagnostics.log_runtime_diagnostics(adapter, stage="before propagation")
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(video_bundle, session, direction=direction)
                )
            else:
                iterator = adapter.propagate_video(video_bundle, session, direction=direction)
            for result in iterator:
                if hasattr(w, "_globalize_video_roi_result"):
                    w._globalize_video_roi_result(result, bundle, video_roi_bounds)
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            if video_roi_bounds is not None:
                session.metadata["large_image_roi"] = (
                    video_roi_bounds.y0,
                    video_roi_bounds.x0,
                    video_roi_bounds.y1,
                    video_roi_bounds.x1,
                )
                session.metadata["large_image_hw"] = w._selection_image_hw(bundle.image)
                session.metadata["local_video_shape"] = tuple(int(value) for value in getattr(video_data, "shape", ()))
            return session

        worker = run_video()
        worker.yielded.connect(w._write_video_result)
        worker.returned.connect(w._set_video_session)
        w._start_worker(worker)
        if video_roi_bounds is not None:
            w._log(
                f"3D/video local ROI ON: using fixed XY ROI "
                f"({video_roi_bounds.width} x {video_roi_bounds.height}) across all frames; "
                f"y={video_roi_bounds.y0}:{video_roi_bounds.y1}, "
                f"x={video_roi_bounds.x0}:{video_roi_bounds.x1}."
            )
        w._log(f"Started video propagation from frame {bundle.image.frame_index or 0}.")

    def propagate_existing_session(self) -> None:
        w = self.widget
        if w.video_session is None:
            w._log("No active SAM3 video session. Run a 3D/video task first.")
            return
        try:
            bundle = w._collect_bundle()
        except Exception as exc:
            w._log(f"Cannot collect prompts: {exc}")
            return
        try:
            adapter = w._ensure_adapter()
        except Exception as exc:
            w._log(f"Cannot propagate session: {exc}")
            return
        session = w.video_session
        direction = w.propagation_direction_combo.currentText()
        video_bundle = (
            w._localize_bundle_for_existing_video_roi(bundle, session)
            if hasattr(w, "_localize_bundle_for_existing_video_roi")
            else bundle
        )
        diagnostics = (
            Sam3Diagnostics(w._log)
            if getattr(w, "_sam31_diagnostics_enabled", lambda: False)()
            else None
        )
        if diagnostics is not None:
            diagnostics.log_prompt_diagnostics(video_bundle)
            diagnostics.log_session_diagnostics(adapter, session, stage="before existing-session propagation")
            diagnostics.log_cuda_diagnostics("before existing-session propagation")
            diagnostics.log_runtime_diagnostics(adapter, stage="before existing-session propagation")

        @thread_worker
        def propagate():
            if diagnostics is not None:
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(video_bundle, session, direction=direction),
                    label="SAM3.1 existing-session propagation",
                )
            else:
                iterator = adapter.propagate_video(video_bundle, session, direction=direction)
            for result in iterator:
                if hasattr(w, "_globalize_existing_video_roi_result"):
                    w._globalize_existing_video_roi_result(result, bundle, session)
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = propagate()
        worker.yielded.connect(w._write_video_result)
        worker.returned.connect(w._set_video_session)
        w._start_worker(worker)
        w._log(f"Propagating existing session {session.session_id}.")
