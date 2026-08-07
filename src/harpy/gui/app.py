from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QMessageBox

from harpy.config import DEFAULT_CONFIG, AppConfig
from harpy.gui.controller import LabController
from harpy.gui.qt_audio import QtAudioEngine
from harpy.gui.visualizer import SampleRingBuffer
from harpy.gui.window import HarpyWindow

AudioFactory = Callable[[AppConfig, SampleRingBuffer], QtAudioEngine]


@dataclass(slots=True)
class LabRuntime:
    config: AppConfig
    sample_history: SampleRingBuffer
    audio: QtAudioEngine
    controller: LabController
    window: HarpyWindow


def build_runtime(
    app: QApplication,
    config: AppConfig = DEFAULT_CONFIG,
    *,
    audio_factory: AudioFactory = QtAudioEngine,
) -> LabRuntime:
    sample_history = SampleRingBuffer(max(config.render.sample_rate_hz, 4_096))
    audio = audio_factory(config, sample_history)
    controller = LabController(config, audio.submit)
    window = HarpyWindow(config, controller, audio, sample_history)
    app.aboutToQuit.connect(controller.force_stop)
    app.aboutToQuit.connect(audio.shutdown)
    return LabRuntime(config, sample_history, audio, controller, window)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(arguments)
    try:
        runtime = build_runtime(app)
    except (TypeError, ValueError) as error:
        QMessageBox.critical(None, "Harpy configuration error", str(error))
        return 2
    runtime.window.show()
    runtime.audio.start()
    return app.exec()
