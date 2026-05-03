from __future__ import annotations

import time
from typing import Any

from napari.qt.threading import thread_worker

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
        diagnostics = (
            Sam3Diagnostics(w._log)
            if getattr(w, "_sam31_diagnostics_enabled", lambda: False)()
            else None
        )
        if diagnostics is not None:
            diagnostics.log(
                "SAM3.1 image source: "
                f"{diagnostics.describe_image_source(image_layer.data)}"
            )
            diagnostics.log_prompt_diagnostics(bundle)

        @thread_worker
        def run_video():
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before start_video_session")
                diagnostics.log_runtime_diagnostics(adapter, stage="before session start")
                session_t0 = time.perf_counter()
            session = adapter.start_video_session(image_layer.data, bundle)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 session start", session_t0)
                diagnostics.log(
                    "SAM3.1 session ready: "
                    f"{session.session_id}; prompt_frame={bundle.image.frame_index or 0}; "
                    f"boxes={len(getattr(bundle, 'boxes', []) or [])}; "
                    f"points={len(getattr(bundle, 'points', []) or [])}."
                )
                diagnostics.log_session_diagnostics(adapter, session, stage="after session start")
                prompt_t0 = time.perf_counter()
            prompt_result = adapter.add_video_prompt(bundle, session)
            if diagnostics is not None:
                diagnostics.log_timing("SAM3.1 prompt insertion", prompt_t0)
                diagnostics.log_session_diagnostics(adapter, session, stage="after prompt insertion")
            prompt_result.metadata["image_layer"] = bundle.image.layer_name
            yield prompt_result
            if diagnostics is not None:
                diagnostics.log_cuda_diagnostics("before propagation")
                diagnostics.log_runtime_diagnostics(adapter, stage="before propagation")
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(bundle, session, direction=direction)
                )
            else:
                iterator = adapter.propagate_video(bundle, session, direction=direction)
            for result in iterator:
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = run_video()
        worker.yielded.connect(w._write_video_result)
        worker.returned.connect(w._set_video_session)
        w._start_worker(worker)
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
        diagnostics = (
            Sam3Diagnostics(w._log)
            if getattr(w, "_sam31_diagnostics_enabled", lambda: False)()
            else None
        )
        if diagnostics is not None:
            diagnostics.log_prompt_diagnostics(bundle)
            diagnostics.log_session_diagnostics(adapter, session, stage="before existing-session propagation")
            diagnostics.log_cuda_diagnostics("before existing-session propagation")
            diagnostics.log_runtime_diagnostics(adapter, stage="before existing-session propagation")

        @thread_worker
        def propagate():
            if diagnostics is not None:
                iterator = diagnostics.iter_propagation_with_timing(
                    adapter.propagate_video(bundle, session, direction=direction),
                    label="SAM3.1 existing-session propagation",
                )
            else:
                iterator = adapter.propagate_video(bundle, session, direction=direction)
            for result in iterator:
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = propagate()
        worker.yielded.connect(w._write_video_result)
        worker.returned.connect(w._set_video_session)
        w._start_worker(worker)
        w._log(f"Propagating existing session {session.session_id}.")
