import sys
from types import SimpleNamespace
import wave

from napari_sam3_assistant.notifications.task_complete_sound import (
    TASK_COMPLETE_SOUND_ENABLED_KEY,
    TaskCompleteSound,
)


class FakeSettings:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key, default=None, type=None):
        value = self.values.get(key, default)
        return type(value) if type is not None else value

    def setValue(self, key, value) -> None:
        self.values[key] = value


def test_task_complete_sound_enabled_setting():
    settings = FakeSettings()
    sound = TaskCompleteSound(settings)

    assert not sound.is_enabled()

    sound.set_enabled(True)
    assert settings.values[TASK_COMPLETE_SOUND_ENABLED_KEY] is True
    assert sound.is_enabled()

    sound.set_enabled(False)
    assert not sound.is_enabled()


def test_task_complete_sound_generates_short_wav(tmp_path):
    sound = TaskCompleteSound()

    path = tmp_path / "chime.wav"
    sound._write_chime(path)

    assert path.exists()
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 44100
        duration = wav_file.getnframes() / wav_file.getframerate()

    assert 1.0 <= duration <= 1.2


def test_task_complete_sound_skips_qt_when_conda_pipewire_plugin_missing(
    monkeypatch,
    tmp_path,
):
    conda_prefix = tmp_path / "env"
    (conda_prefix / "lib" / "alsa-lib").mkdir(parents=True)
    sound = TaskCompleteSound()

    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    assert sound._should_skip_qt_multimedia()


def test_task_complete_sound_prefers_pipewire_player(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "shutil.which",
        lambda command: "/usr/bin/pw-play" if command == "pw-play" else None,
    )

    sound = TaskCompleteSound()

    assert sound._external_player == ("/usr/bin/pw-play",)


def test_task_complete_sound_uses_winsound_first_on_windows(
    monkeypatch,
    tmp_path,
):
    calls = []
    fake_winsound = SimpleNamespace(
        SND_FILENAME=1,
        SND_ASYNC=2,
        SND_NODEFAULT=4,
        PlaySound=lambda path, flags: calls.append((path, flags)),
    )
    sound_path = tmp_path / "chime.wav"
    sound_path.write_bytes(b"RIFF")

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)

    sound = TaskCompleteSound()

    assert sound._play_with_windows_sound(sound_path)
    assert calls == [(str(sound_path), 7)]


def test_task_complete_sound_falls_back_when_winsound_fails(
    monkeypatch,
    tmp_path,
):
    player_calls = []

    class FakePlayer:
        def stop(self):
            player_calls.append("stop")

        def play(self):
            player_calls.append("play")

    def raise_error(path, flags):
        raise RuntimeError("sound unavailable")

    fake_winsound = SimpleNamespace(
        SND_FILENAME=1,
        SND_ASYNC=2,
        SND_NODEFAULT=4,
        PlaySound=raise_error,
    )
    sound_path = tmp_path / "chime.wav"
    sound_path.write_bytes(b"RIFF")

    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)

    sound = TaskCompleteSound()
    monkeypatch.setattr(sound, "_ensure_sound_file", lambda: sound_path)
    monkeypatch.setattr(sound, "_qt_sound_player", lambda path: FakePlayer())

    sound.play_task_complete()

    assert not sound._windows_sound_available
    assert sound._sound_backend_available
    assert player_calls == ["stop", "play"]
