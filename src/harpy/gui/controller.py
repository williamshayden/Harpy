from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from harpy.config import AppConfig
from harpy.note_names import format_pitch_readout
from harpy.pitch import MidiNote, Pitch


class AudioCommandKind(StrEnum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class AudioCommand:
    kind: AudioCommandKind
    pitch: Pitch | None = None

    def __post_init__(self) -> None:
        if self.kind is AudioCommandKind.NOTE_ON and self.pitch is None:
            raise ValueError("NOTE_ON requires a pitch")
        if self.kind is not AudioCommandKind.NOTE_ON and self.pitch is not None:
            raise ValueError(f"{self.kind} does not accept a pitch")


@dataclass(frozen=True, slots=True)
class ControllerState:
    note: MidiNote
    readout: str
    gate_active: bool
    selector_enabled: bool


class LabController:
    def __init__(
        self,
        config: AppConfig,
        send_command: Callable[[AudioCommand], None],
    ) -> None:
        self._config = config
        self._send_command = send_command
        self._note = config.keyboard.initial_note
        self._gate_active = False
        self._voice_started = False

    @property
    def state(self) -> ControllerState:
        return ControllerState(
            note=self._note,
            readout=format_pitch_readout(
                self._note,
                self._config.tuning,
                self._config.keyboard.middle_c_octave,
            ),
            gate_active=self._gate_active,
            selector_enabled=not self._gate_active,
        )

    def set_note(self, number: int) -> ControllerState:
        if self._gate_active:
            raise RuntimeError("pitch cannot change while Play is held")
        note = MidiNote(number)
        if not (
            self._config.keyboard.minimum_note.number
            <= number
            <= self._config.keyboard.maximum_note.number
        ):
            raise ValueError("note is outside the configured selector range")
        self._note = note
        return self.state

    def press_play(self) -> ControllerState:
        if self._gate_active:
            return self.state
        self._send_command(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(self._note)))
        self._gate_active = True
        self._voice_started = True
        return self.state

    def release_play(self) -> ControllerState:
        if not self._gate_active:
            return self.state
        self._send_command(AudioCommand(AudioCommandKind.NOTE_OFF))
        self._gate_active = False
        return self.state

    def force_stop(self) -> ControllerState:
        self._gate_active = False
        if self._voice_started:
            self._send_command(AudioCommand(AudioCommandKind.RESET))
            self._voice_started = False
        return self.state
