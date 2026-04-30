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


def test_task_complete_sound_generates_short_wav():
    sound = TaskCompleteSound()

    path = sound._ensure_sound_file()

    assert path.exists()
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 44100
        duration = wav_file.getnframes() / wav_file.getframerate()

    assert 1.4 <= duration <= 1.5


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
