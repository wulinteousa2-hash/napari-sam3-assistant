from __future__ import annotations

from typing import Any

from napari.qt.threading import thread_worker

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

        @thread_worker
        def run_video():
            session = adapter.start_video_session(image_layer.data, bundle)
            prompt_result = adapter.add_video_prompt(bundle, session)
            prompt_result.metadata["image_layer"] = bundle.image.layer_name
            yield prompt_result
            for result in adapter.propagate_video(bundle, session, direction=direction):
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

        @thread_worker
        def propagate():
            for result in adapter.propagate_video(bundle, session, direction=direction):
                result.metadata["image_layer"] = bundle.image.layer_name
                yield result
            return session

        worker = propagate()
        worker.yielded.connect(w._write_video_result)
        worker.returned.connect(w._set_video_session)
        w._start_worker(worker)
        w._log(f"Propagating existing session {session.session_id}.")
