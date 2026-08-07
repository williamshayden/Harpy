from PySide6.QtCore import QObject, Signal

from harpy.config import DEFAULT_CONFIG, AppConfig
from harpy.gui.app import build_runtime
from harpy.gui.controller import AudioCommand
from harpy.gui.visualizer import SampleRingBuffer


class ComposedAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(self, config: AppConfig, history: SampleRingBuffer) -> None:
        super().__init__()
        self.config = config
        self.history = history
        self.commands: list[AudioCommand] = []
        self.started = False

    def submit(self, command: AudioCommand) -> None:
        self.commands.append(command)

    def start(self) -> None:
        self.started = True
        self.status_changed.emit("Fake speakers · 48 kHz", True)

    def shutdown(self) -> None:
        self.started = False


def test_build_runtime_owns_connected_components(qapp) -> None:
    created: list[ComposedAudioEngine] = []

    def factory(config: AppConfig, history: SampleRingBuffer) -> ComposedAudioEngine:
        engine = ComposedAudioEngine(config, history)
        created.append(engine)
        return engine

    runtime = build_runtime(qapp, DEFAULT_CONFIG, audio_factory=factory)
    assert runtime.config is DEFAULT_CONFIG
    assert runtime.audio is created[0]
    assert runtime.window.pitch_slider.value() == 60
    runtime.audio.start()
    assert runtime.audio.started
    assert runtime.window.play_button.isEnabled()
    runtime.window.close()
