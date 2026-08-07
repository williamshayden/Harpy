import pytest

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind, LabController
from harpy.pitch import MidiNote, Pitch


def make_controller() -> tuple[LabController, list[AudioCommand]]:
    commands: list[AudioCommand] = []
    return LabController(DEFAULT_CONFIG, commands.append), commands


def test_initial_state_is_middle_c_and_idle() -> None:
    controller, commands = make_controller()
    assert controller.state.note == MidiNote(60)
    assert controller.state.readout == "C3 · MIDI 60 · 261.626 Hz"
    assert not controller.state.gate_active
    assert controller.state.selector_enabled
    assert commands == []


def test_press_and_release_send_one_command_each() -> None:
    controller, commands = make_controller()
    controller.press_play()
    controller.press_play()
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert controller.state.gate_active
    assert not controller.state.selector_enabled
    controller.release_play()
    controller.release_play()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert not controller.state.gate_active
    assert controller.state.selector_enabled


def test_note_selection_is_blocked_while_gated() -> None:
    controller, _ = make_controller()
    controller.press_play()
    with pytest.raises(RuntimeError, match="while Play is held"):
        controller.set_note(61)


@pytest.mark.parametrize("number", [47, 73])
def test_note_selection_rejects_values_outside_the_visible_range(number: int) -> None:
    controller, _ = make_controller()
    with pytest.raises(ValueError, match="configured selector range"):
        controller.set_note(number)


def test_audio_commands_validate_pitch_payloads() -> None:
    with pytest.raises(ValueError, match="requires a pitch"):
        AudioCommand(AudioCommandKind.NOTE_ON)
    with pytest.raises(ValueError, match="does not accept"):
        AudioCommand(AudioCommandKind.NOTE_OFF, Pitch.from_midi(60))


def test_force_stop_clears_state_and_is_idempotent() -> None:
    controller, commands = make_controller()
    controller.press_play()
    controller.force_stop()
    controller.force_stop()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not controller.state.gate_active
    assert controller.state.selector_enabled


def test_note_can_change_after_normal_release() -> None:
    controller, _ = make_controller()
    controller.press_play()
    controller.release_play()
    state = controller.set_note(61)
    assert state.note == MidiNote(61)
    assert state.readout.startswith("C♯3 · MIDI 61")
