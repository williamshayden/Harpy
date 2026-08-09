from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QMessageBox

from harpy.capture import CaptureCoordinator, SampleHistory
from harpy.config import DEFAULT_CONFIG, AppConfig
from harpy.gui.patch_dialogs import NativePatchDialogs, PatchDialogPort
from harpy.gui.window import HarpyWindow
from harpy.gui.workbench_controller import WorkbenchController
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.synth.models import RenderConfig, SynthPatch

if TYPE_CHECKING:
    from harpy.gui.qt_audio import QtAudioBackend

AudioFactory = Callable[[RenderConfig, SynthPatch, SampleHistory], "QtAudioBackend"]


@dataclass(slots=True)
class WorkbenchRuntime:
    config: AppConfig
    history: SampleHistory
    capture: CaptureCoordinator
    audio: QtAudioBackend
    controller: WorkbenchController
    window: HarpyWindow


def build_runtime(
    app: QApplication,
    config: AppConfig = DEFAULT_CONFIG,
    *,
    audio_factory: AudioFactory | None = None,
    patch_dialogs: PatchDialogPort | None = None,
) -> WorkbenchRuntime:
    if audio_factory is None:
        from harpy.gui.qt_audio import QtAudioBackend

        audio_factory = QtAudioBackend

    spec = WorkbenchSpec.from_tuning(config.tuning)
    history = SampleHistory(capacity_frames=config.analysis.fft_frames)
    capture = CaptureCoordinator(
        history,
        config.render.sample_rate_hz,
        config.analysis,
    )
    audio = audio_factory(config.render, config.patch, history)
    controller = WorkbenchController(
        spec,
        config.render,
        config.patch,
        capture,
        audio.submit,
    )
    window = HarpyWindow(
        controller,
        config.tuning,
        spec,
        config.render,
        patch_dialogs if patch_dialogs is not None else NativePatchDialogs(),
    )

    audio.availability_changed.connect(window.handle_audio_availability)
    audio.audio_failure.connect(window.handle_audio_failure)
    audio.force_stop_requested.connect(window.handle_force_stop)
    audio.voice_idle.connect(window.handle_voice_idle)
    window.shutdown_requested.connect(audio.shutdown)
    app.aboutToQuit.connect(window.prepare_shutdown)

    return WorkbenchRuntime(config, history, capture, audio, controller, window)


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
