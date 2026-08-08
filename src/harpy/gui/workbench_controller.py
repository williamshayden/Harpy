"""Semantic state transitions for the native synthesis workbench."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass

from harpy.capture import CaptureCoordinator, CaptureState, CaptureView
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import RenderConfig, SynthPatch, validate_renderable_patch


@dataclass(frozen=True, slots=True)
class WorkbenchState:
    """UI-independent workbench state expressed in domain values."""

    selected_frequency_hz: float
    gate_held: bool
    voice_may_be_active: bool
    patch: SynthPatch
    capture: CaptureView
    audio_available: bool
    audio_error: str | None


class WorkbenchController:
    """Coordinate workbench edits, transport, capture, and audio availability."""

    def __init__(
        self,
        spec: WorkbenchSpec,
        render: RenderConfig,
        patch: SynthPatch,
        capture: CaptureCoordinator,
        send_command: Callable[[AudioCommand], None],
    ) -> None:
        if not isinstance(spec, WorkbenchSpec):
            raise ValueError("spec must be a WorkbenchSpec")
        if not isinstance(render, RenderConfig):
            raise ValueError("render must be a RenderConfig")
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        if not isinstance(capture, CaptureCoordinator):
            raise ValueError("capture must be a CaptureCoordinator")
        if not callable(send_command):
            raise ValueError("send_command must be callable")
        self._validate_spec(spec)
        validate_renderable_patch(patch, render)

        self._spec = spec
        self._render = render
        self._patch = patch
        self._capture = capture
        self._send_command = send_command
        self._selected_frequency_hz = spec.center_frequency_hz
        self._gate_held = False
        self._voice_may_be_active = False
        self._voice_generation: int | None = None
        self._capture_view = capture.refresh()
        self._audio_available = True
        self._audio_error: str | None = None

    @property
    def state(self) -> WorkbenchState:
        """Return an immutable snapshot of the current semantic state."""

        return WorkbenchState(
            selected_frequency_hz=self._selected_frequency_hz,
            gate_held=self._gate_held,
            voice_may_be_active=self._voice_may_be_active,
            patch=self._patch,
            capture=self._capture_view,
            audio_available=self._audio_available,
            audio_error=self._audio_error,
        )

    def set_frequency(self, frequency_hz: float) -> WorkbenchState:
        """Select a frequency and retune any voice that may still be sounding."""

        frequency = self._validated_frequency(frequency_hz)
        self._selected_frequency_hz = frequency
        if self._voice_may_be_active:
            self._send_command(AudioCommand(AudioCommandKind.RETUNE, frequency_hz=frequency))
        return self.state

    def press_play(self) -> WorkbenchState:
        """Start or retrigger the selected frequency when audio is available."""

        if self._gate_held or not self._audio_available:
            return self.state

        generation = self._capture.begin()
        command = AudioCommand(
            AudioCommandKind.NOTE_ON,
            frequency_hz=self._selected_frequency_hz,
            generation=generation,
        )
        self._capture_view = self._capture.refresh()
        self._gate_held = True
        self._voice_may_be_active = True
        self._voice_generation = generation
        self._send_command(command)
        return self.state

    def release_play(self) -> WorkbenchState:
        """Release a held gate while retaining possible release-tail activity."""

        if not self._gate_held:
            return self.state
        self._gate_held = False
        self._send_command(AudioCommand(AudioCommandKind.NOTE_OFF))
        return self.state

    def clear_measurement(self) -> WorkbenchState:
        """Clear history, measuring anew only if the current voice may be active."""

        voice_active = self._voice_may_be_active
        generation = self._capture.clear(voice_active=voice_active)
        command = AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=generation)
        self._capture_view = self._capture.refresh()
        self._voice_generation = generation if voice_active else None
        self._send_command(command)
        return self.state

    def replace_patch(self, patch: SynthPatch) -> WorkbenchState:
        """Validate and atomically replace the complete patch and playback state."""

        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        validate_renderable_patch(patch, self._render)

        generation = self._capture.reset()
        command = AudioCommand(
            AudioCommandKind.REPLACE_PATCH,
            patch=patch,
            generation=generation,
        )
        self._capture_view = self._capture.refresh()
        self._patch = patch
        self._gate_held = False
        self._voice_may_be_active = False
        self._voice_generation = None
        self._send_command(command)
        return self.state

    def refresh_capture(self) -> WorkbenchState:
        """Publish the coordinator's latest generation-safe capture view."""

        self._capture_view = self._capture.refresh()
        return self.state

    def mark_voice_idle(self, generation: int) -> WorkbenchState:
        """Accept a natural idle notification only for the current voice token."""

        supplied_generation = self._validated_generation(generation)
        if supplied_generation == self._voice_generation:
            self._voice_may_be_active = False
            self._voice_generation = None
        return self.state

    def force_stop(self) -> WorkbenchState:
        """Reset active playback or retained capture once, regardless of callback order."""

        needs_reset = (
            self._gate_held
            or self._voice_may_be_active
            or self._capture_view.state is not CaptureState.EMPTY
        )
        if not needs_reset:
            return self.state

        generation = self._capture.reset()
        command = AudioCommand(AudioCommandKind.RESET, generation=generation)
        self._capture_view = self._capture.refresh()
        self._gate_held = False
        self._voice_may_be_active = False
        self._voice_generation = None
        self._send_command(command)
        return self.state

    def set_audio_availability(
        self,
        available: bool,
        error: str | None = None,
    ) -> WorkbenchState:
        """Update backend availability without changing transport or capture state."""

        if not isinstance(available, bool):
            raise ValueError("available must be a boolean")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string or None")
        self._audio_available = available
        self._audio_error = None if available else error
        return self.state

    def _validated_frequency(self, frequency_hz: float) -> float:
        if isinstance(frequency_hz, bool):
            raise ValueError("frequency_hz must be finite and within the workbench range")
        try:
            frequency = float(frequency_hz)
        except (TypeError, ValueError) as error:
            message = "frequency_hz must be finite and within the workbench range"
            raise ValueError(message) from error
        if (
            not math.isfinite(frequency)
            or frequency < self._spec.minimum_frequency_hz
            or frequency > self._spec.maximum_frequency_hz
        ):
            raise ValueError("frequency_hz must be finite and within the workbench range")
        return frequency

    @staticmethod
    def _validated_generation(generation: int) -> int:
        if isinstance(generation, bool):
            raise ValueError("generation must be a nonnegative integer")
        try:
            value = operator.index(generation)
        except TypeError as error:
            raise ValueError("generation must be a nonnegative integer") from error
        if value < 0:
            raise ValueError("generation must be a nonnegative integer")
        return value

    @staticmethod
    def _validate_spec(spec: WorkbenchSpec) -> None:
        values = (
            spec.minimum_frequency_hz,
            spec.center_frequency_hz,
            spec.maximum_frequency_hz,
        )
        if any(isinstance(value, bool) for value in values):
            raise ValueError("workbench frequencies must be positive, finite, and ordered")
        try:
            minimum, center, maximum = (float(value) for value in values)
        except (TypeError, ValueError) as error:
            message = "workbench frequencies must be positive, finite, and ordered"
            raise ValueError(message) from error
        if not (
            math.isfinite(minimum)
            and math.isfinite(center)
            and math.isfinite(maximum)
            and 0.0 < minimum <= center <= maximum
        ):
            raise ValueError("workbench frequencies must be positive, finite, and ordered")
