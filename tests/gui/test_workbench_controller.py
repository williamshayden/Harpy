from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from harpy.analysis import AnalysisConfig, AudioObservation
from harpy.capture import CaptureCoordinator, CaptureState, SampleHistory
from harpy.gui.workbench_controller import WorkbenchController
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch
from harpy.tuning import Tuning

SAMPLE_RATE_HZ = 48_000
ANALYSIS_CONFIG = AnalysisConfig(
    waveform_window_seconds=4 / SAMPLE_RATE_HZ,
    fft_frames=4,
)


def make_controller(
    *,
    analyzer: Callable[..., AudioObservation] | None = None,
) -> tuple[WorkbenchController, list[AudioCommand], SampleHistory]:
    history = SampleHistory(capacity_frames=8)
    coordinator_kwargs: dict[str, object] = {}
    if analyzer is not None:
        coordinator_kwargs["analyzer"] = analyzer
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        **coordinator_kwargs,
    )
    commands: list[AudioCommand] = []
    controller = WorkbenchController(
        WorkbenchSpec.from_tuning(Tuning()),
        RenderConfig(sample_rate_hz=SAMPLE_RATE_HZ),
        SynthPatch(),
        coordinator,
        commands.append,
    )
    return controller, commands, history


def test_high_reference_tuning_cannot_compose_an_out_of_nyquist_workbench() -> None:
    history = SampleHistory(capacity_frames=8)
    capture = CaptureCoordinator(history, SAMPLE_RATE_HZ, ANALYSIS_CONFIG)
    commands: list[AudioCommand] = []
    spec = WorkbenchSpec.from_tuning(Tuning(reference_hz=30_000.0))

    with pytest.raises(ValueError, match="Nyquist"):
        WorkbenchController(
            spec,
            RenderConfig(sample_rate_hz=SAMPLE_RATE_HZ),
            SynthPatch(),
            capture,
            commands.append,
        )

    assert history.generation == 0
    assert commands == []


def test_initial_state_is_tuned_c3_and_semantically_idle() -> None:
    controller, commands, _ = make_controller()

    state = controller.state

    assert Tuning().describe_frequency(state.selected_frequency_hz).name == "C3"
    assert state.selected_frequency_hz == WorkbenchSpec.from_tuning(Tuning()).center_frequency_hz
    assert not state.gate_held
    assert not state.voice_may_be_active
    assert state.patch == SynthPatch()
    assert state.capture.state is CaptureState.EMPTY
    assert state.capture.observation is None
    assert state.audio_available
    assert state.audio_error is None
    assert commands == []


def test_idle_frequency_change_updates_state_without_audio_command() -> None:
    controller, commands, _ = make_controller()

    state = controller.set_frequency(330.0)

    assert state.selected_frequency_hz == 330.0
    assert commands == []


@pytest.mark.parametrize(
    "frequency_hz",
    [True, False, float("nan"), float("inf"), float("-inf"), 100.0, 600.0],
)
def test_frequency_change_rejects_nonfinite_boolean_and_out_of_range_values(
    frequency_hz: object,
) -> None:
    controller, commands, _ = make_controller()
    before = controller.state

    with pytest.raises(ValueError, match="frequency"):
        controller.set_frequency(frequency_hz)  # type: ignore[arg-type]

    assert controller.state == before
    assert commands == []


def test_press_and_release_are_idempotent_and_preserve_release_activity() -> None:
    controller, commands, _ = make_controller()

    pressed = controller.press_play()
    controller.press_play()
    released = controller.release_play()
    controller.release_play()

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert commands[0].frequency_hz == pressed.selected_frequency_hz
    assert commands[0].generation == pressed.capture.generation
    assert pressed.gate_held
    assert pressed.voice_may_be_active
    assert pressed.capture.state is CaptureState.MEASURING
    assert not released.gate_held
    assert released.voice_may_be_active


def test_frequency_change_retunes_without_retrigger_while_held() -> None:
    controller, commands, _ = make_controller()
    controller.press_play()

    state = controller.set_frequency(330.0)

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RETUNE,
    ]
    assert commands[-1].frequency_hz == 330.0
    assert commands[-1].generation is None
    assert state.gate_held


def test_frequency_change_retunes_during_release() -> None:
    controller, commands, _ = make_controller()
    controller.press_play()
    controller.release_play()

    state = controller.set_frequency(330.0)

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.RETUNE,
    ]
    assert not state.gate_held
    assert state.voice_may_be_active


def test_current_idle_callback_stops_retunes_and_makes_clear_idle() -> None:
    controller, commands, _ = make_controller()
    pressed = controller.press_play()
    generation = pressed.capture.generation
    controller.release_play()

    idle = controller.mark_voice_idle(generation)
    controller.set_frequency(330.0)
    cleared = controller.clear_measurement()

    assert not idle.voice_may_be_active
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.CLEAR_CAPTURE,
    ]
    assert cleared.capture.state is CaptureState.EMPTY
    assert commands[-1].generation == generation + 1 == cleared.capture.generation


def test_current_idle_callback_preserves_the_exact_live_capture() -> None:
    observation = AudioObservation(
        has_signal=True,
        waveform_samples=np.array([0.25], dtype=np.float32),
        waveform_time_ms=np.array([0.0]),
        spectrum_frequency_hz=np.array([220.0]),
        spectrum_level_dbfs=np.array([-12.0]),
        peak_amplitude_fs=0.25,
        peak_frequency_hz=220.0,
        peak_level_dbfs=-12.0,
    )
    controller, _, history = make_controller(analyzer=lambda *_args: observation)
    generation = controller.press_play().capture.generation
    assert history.append(np.ones(4, dtype=np.float32), generation)
    captured = controller.refresh_capture().capture
    controller.release_play()

    idle = controller.mark_voice_idle(generation)

    assert not idle.voice_may_be_active
    assert idle.capture is captured
    assert idle.capture.state is CaptureState.LIVE
    assert idle.capture.observation is observation


def test_delayed_prior_idle_callback_cannot_stop_a_retriggered_voice() -> None:
    controller, commands, _ = make_controller()
    prior_generation = controller.press_play().capture.generation
    controller.release_play()
    current_generation = controller.press_play().capture.generation

    stale_result = controller.mark_voice_idle(prior_generation)
    controller.set_frequency(330.0)

    assert current_generation == prior_generation + 1
    assert stale_result.voice_may_be_active
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RETUNE,
    ]


def test_active_clear_adopts_generation_for_later_idle_callback() -> None:
    controller, commands, _ = make_controller()
    note_generation = controller.press_play().capture.generation

    active_clear = controller.clear_measurement()
    clear_generation = active_clear.capture.generation
    stale_result = controller.mark_voice_idle(note_generation)
    controller.release_play()
    current_result = controller.mark_voice_idle(clear_generation)
    idle_clear = controller.clear_measurement()

    assert clear_generation == note_generation + 1
    assert active_clear.voice_may_be_active
    assert active_clear.capture.state is CaptureState.MEASURING
    assert stale_result.voice_may_be_active
    assert not current_result.voice_may_be_active
    assert idle_clear.capture.state is CaptureState.EMPTY
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.CLEAR_CAPTURE,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.CLEAR_CAPTURE,
    ]
    assert commands[1].generation == clear_generation
    assert commands[-1].generation == clear_generation + 1


def test_replace_patch_is_one_atomic_audio_command() -> None:
    controller, commands, _ = make_controller()
    controller.press_play()
    replacement = SynthPatch(output_gain_dbfs=-18.0)

    state = controller.replace_patch(replacement)

    assert commands[-1].kind is AudioCommandKind.REPLACE_PATCH
    assert commands[-1].patch == replacement
    assert commands[-1].generation == state.capture.generation
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert state.patch == replacement
    assert not state.gate_held
    assert not state.voice_may_be_active
    assert state.capture.state is CaptureState.EMPTY


def test_unrenderable_patch_leaves_controller_capture_and_commands_unchanged() -> None:
    controller, commands, _ = make_controller()
    controller.press_play()
    before_state = controller.state
    before_commands = tuple(commands)
    invalid = SynthPatch(envelope=EnvelopeConfig(attack_seconds=1e-12))

    with pytest.raises(ValueError, match="at least one frame"):
        controller.replace_patch(invalid)

    assert controller.state == before_state
    assert tuple(commands) == before_commands


def test_huge_finite_patch_duration_leaves_controller_state_and_capture_unchanged() -> None:
    controller, commands, history = make_controller()
    controller.press_play()
    before_state = controller.state
    before_commands = tuple(commands)
    before_generation = history.generation
    invalid = SynthPatch(envelope=EnvelopeConfig(attack_seconds=1e308))

    with pytest.raises(ValueError, match="attack_seconds"):
        controller.replace_patch(invalid)

    assert controller.state == before_state
    assert history.generation == before_generation
    assert tuple(commands) == before_commands


def test_refresh_capture_publishes_the_coordinators_current_observation() -> None:
    observation = AudioObservation(
        has_signal=True,
        waveform_samples=np.array([0.25], dtype=np.float32),
        waveform_time_ms=np.array([0.0]),
        spectrum_frequency_hz=np.array([220.0]),
        spectrum_level_dbfs=np.array([-12.0]),
        peak_amplitude_fs=0.25,
        peak_frequency_hz=220.0,
        peak_level_dbfs=-12.0,
    )
    controller, _, history = make_controller(analyzer=lambda *_args: observation)
    generation = controller.press_play().capture.generation
    assert history.append(np.ones(4, dtype=np.float32), generation)

    state = controller.refresh_capture()

    assert state.capture.state is CaptureState.LIVE
    assert state.capture.observation is observation


def test_force_stop_is_idempotent_and_allocates_one_reset_generation() -> None:
    controller, commands, _ = make_controller()
    note_generation = controller.press_play().capture.generation

    stopped = controller.force_stop()
    stopped_again = controller.force_stop()

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert commands[-1].generation == note_generation + 1 == stopped.capture.generation
    assert stopped == stopped_again
    assert not stopped.gate_held
    assert not stopped.voice_may_be_active
    assert stopped.capture.state is CaptureState.EMPTY


def test_force_stop_resets_capture_retained_after_voice_becomes_idle() -> None:
    controller, commands, _ = make_controller()
    generation = controller.press_play().capture.generation
    controller.release_play()
    controller.mark_voice_idle(generation)

    stopped = controller.force_stop()

    assert commands[-1].kind is AudioCommandKind.RESET
    assert commands[-1].generation == generation + 1 == stopped.capture.generation
    assert stopped.capture.state is CaptureState.EMPTY


def test_unavailable_audio_blocks_note_on_but_keeps_frequency_and_patch_editable() -> None:
    controller, commands, _ = make_controller()

    unavailable = controller.set_audio_availability(False, "device missing")
    blocked = controller.press_play()
    frequency = controller.set_frequency(330.0)
    replacement = SynthPatch(output_gain_dbfs=-18.0)
    edited = controller.replace_patch(replacement)
    recovered = controller.set_audio_availability(True, "healthy")

    assert not unavailable.audio_available
    assert unavailable.audio_error == "device missing"
    assert not blocked.gate_held
    assert not blocked.voice_may_be_active
    assert frequency.selected_frequency_hz == 330.0
    assert edited.patch == replacement
    assert [command.kind for command in commands] == [AudioCommandKind.REPLACE_PATCH]
    assert recovered.audio_available
    assert recovered.audio_error is None


def test_availability_update_does_not_force_stop_an_active_voice() -> None:
    controller, commands, _ = make_controller()
    controller.press_play()

    state = controller.set_audio_availability(False, "device failed")

    assert state.gate_held
    assert state.voice_may_be_active
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]


@pytest.mark.parametrize("availability_first", [True, False])
def test_device_failure_callback_orders_share_one_idempotent_force_stop_path(
    availability_first: bool,
) -> None:
    controller, commands, _ = make_controller()
    note_generation = controller.press_play().capture.generation

    if availability_first:
        controller.set_audio_availability(False, "device failed")
        stopped = controller.force_stop()
    else:
        stopped = controller.force_stop()
        controller.set_audio_availability(False, "device failed")
    controller.force_stop()
    final = controller.set_audio_availability(False, "device failed")

    resets = [command for command in commands if command.kind is AudioCommandKind.RESET]
    assert len(resets) == 1
    assert resets[0].generation == note_generation + 1 == stopped.capture.generation
    assert not final.audio_available
    assert final.audio_error == "device failed"
    assert final.capture.state is CaptureState.EMPTY
