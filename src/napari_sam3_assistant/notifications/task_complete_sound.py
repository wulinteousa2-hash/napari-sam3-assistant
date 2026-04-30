from __future__ import annotations

import math
import os
import platform
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import wave
from typing import Any

from qtpy.QtCore import QUrl

TASK_COMPLETE_SOUND_ENABLED_KEY = "task_complete_sound_enabled_v2"


class TaskCompleteSound:
    """Small, optional completion chime for long-running SAM3 tasks."""

    def __init__(self, settings: Any | None = None) -> None:
        self.settings = settings
        self._player: Any | None = None
        self._sound_path: Path | None = None
        self._sound_backend_available = True
        self._external_player = self._find_external_player()

    def is_enabled(self) -> bool:
        if self.settings is None:
            return True
        return bool(
            self.settings.value(
                TASK_COMPLETE_SOUND_ENABLED_KEY,
                False,
                type=bool,
            )
        )

    def set_enabled(self, enabled: bool) -> None:
        if self.settings is not None:
            self.settings.setValue(TASK_COMPLETE_SOUND_ENABLED_KEY, bool(enabled))

    def play_task_complete(self) -> None:
        if not self.is_enabled() or not self._sound_backend_available:
            return
        try:
            sound_path = self._ensure_sound_file()
            if self._play_with_external_player(sound_path):
                return
            if self._should_skip_qt_multimedia():
                self._sound_backend_available = False
                return
            player = self._qt_sound_player(sound_path)
            if player is None:
                return
            player.stop()
            player.play()
        except Exception:
            self._sound_backend_available = False

    def _find_external_player(self) -> tuple[str, ...] | None:
        if platform.system() != "Linux":
            return None
        for command in (("pw-play",), ("paplay",), ("aplay", "-q")):
            executable = shutil.which(command[0])
            if executable is not None:
                return (executable, *command[1:])
        return None

    def _play_with_external_player(self, sound_path: Path) -> bool:
        if self._external_player is None:
            return False
        try:
            subprocess.Popen(
                (*self._external_player, str(sound_path)),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            self._external_player = None
            return False

    def _should_skip_qt_multimedia(self) -> bool:
        if platform.system() != "Linux":
            return False
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return False
        alsa_plugin_dir = Path(conda_prefix) / "lib" / "alsa-lib"
        pipewire_plugin = alsa_plugin_dir / "libasound_module_pcm_pipewire.so"
        return alsa_plugin_dir.exists() and not pipewire_plugin.exists()

    def _qt_sound_player(self, sound_path: Path) -> Any | None:
        if self._player is not None:
            return self._player
        try:
            from qtpy.QtMultimedia import QSoundEffect
        except Exception:
            self._sound_backend_available = False
            return None

        player = QSoundEffect()
        player.setSource(QUrl.fromLocalFile(str(sound_path)))
        player.setVolume(0.24)
        self._player = player
        return player

    def _ensure_sound_file(self) -> Path:
        if self._sound_path is not None and self._sound_path.exists():
            return self._sound_path

        path = Path(tempfile.gettempdir()) / "napari_sam3_task_complete_chime_v2.wav"
        self._write_chime(path)
        self._sound_path = path
        return path

    def _write_chime(self, path: Path) -> None:
        sample_rate = 44100
        duration_seconds = 1.10
        sample_count = int(sample_rate * duration_seconds)
        max_amplitude = 32767

        frames = bytearray()

        for index in range(sample_count):
            t = index / sample_rate

            # One soft lower chime note.
            base = math.sin(2.0 * math.pi * 345.00 * t)

            # Very light upper partial for gentle bell character.
            harmonic = math.sin(2.0 * math.pi * 888.00 * t)

            attack = 0.010
            if t < attack:
                envelope = t / attack
            else:
                envelope = math.exp(-3.1 * (t - attack) / (duration_seconds - attack))

            fade_out_start = duration_seconds - 0.22
            if t > fade_out_start:
                fade_progress = (t - fade_out_start) / (duration_seconds - fade_out_start)
                fade_progress = min(max(fade_progress, 0.0), 1.0)
                final_fade = 0.5 * (1.0 + math.cos(math.pi * fade_progress))
                envelope *= final_fade

            tone = 0.975 * base + 0.025 * harmonic
            tone = math.tanh(tone * 0.70)

            sample = int(max_amplitude * 0.13 * envelope * tone)
            frames.extend(struct.pack("<h", sample))

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(frames)

    def _envelope(self, t: float, duration_seconds: float) -> float:
        attack = 0.012
        release_start = 0.13
        if t < attack:
            return t / attack
        if t < release_start:
            return 1.0
        release = max(duration_seconds - release_start, 0.001)
        return math.exp(-6.8 * (t - release_start) / release)
