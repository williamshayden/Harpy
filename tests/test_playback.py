from __future__ import annotations

from typing import Any

import pytest

from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import SynthPatch


def test_each_audio_command_accepts_its_exact_payload_shape() -> None:
    patch = SynthPatch(output_gain_dbfs=-18.0)

    assert AudioCommand(
        AudioCommandKind.NOTE_ON,
        frequency_hz=220.0,
        generation=3,
    ) == AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=3)
    assert AudioCommand(AudioCommandKind.RETUNE, frequency_hz=330.0).frequency_hz == 330.0
    assert AudioCommand(AudioCommandKind.NOTE_OFF).kind is AudioCommandKind.NOTE_OFF
    assert AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=4).generation == 4
    assert (
        AudioCommand(
            AudioCommandKind.REPLACE_PATCH,
            patch=patch,
            generation=5,
        ).patch
        is patch
    )
    assert AudioCommand(AudioCommandKind.RESET, generation=6).generation == 6


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (AudioCommandKind.NOTE_ON, {}),
        (AudioCommandKind.NOTE_ON, {"frequency_hz": 220.0}),
        (AudioCommandKind.NOTE_ON, {"generation": 1}),
        (
            AudioCommandKind.NOTE_ON,
            {"frequency_hz": 220.0, "patch": SynthPatch(), "generation": 1},
        ),
        (AudioCommandKind.RETUNE, {}),
        (AudioCommandKind.RETUNE, {"frequency_hz": 220.0, "generation": 1}),
        (AudioCommandKind.RETUNE, {"frequency_hz": 220.0, "patch": SynthPatch()}),
        (AudioCommandKind.NOTE_OFF, {"frequency_hz": 220.0}),
        (AudioCommandKind.NOTE_OFF, {"patch": SynthPatch()}),
        (AudioCommandKind.NOTE_OFF, {"generation": 1}),
        (AudioCommandKind.CLEAR_CAPTURE, {}),
        (AudioCommandKind.CLEAR_CAPTURE, {"frequency_hz": 220.0, "generation": 1}),
        (AudioCommandKind.CLEAR_CAPTURE, {"patch": SynthPatch(), "generation": 1}),
        (AudioCommandKind.REPLACE_PATCH, {}),
        (AudioCommandKind.REPLACE_PATCH, {"patch": SynthPatch()}),
        (AudioCommandKind.REPLACE_PATCH, {"generation": 1}),
        (
            AudioCommandKind.REPLACE_PATCH,
            {"frequency_hz": 220.0, "patch": SynthPatch(), "generation": 1},
        ),
        (AudioCommandKind.RESET, {}),
        (AudioCommandKind.RESET, {"frequency_hz": 220.0, "generation": 1}),
        (AudioCommandKind.RESET, {"patch": SynthPatch(), "generation": 1}),
    ],
)
def test_audio_commands_reject_missing_and_extra_payloads(
    kind: AudioCommandKind,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        AudioCommand(kind, **payload)


@pytest.mark.parametrize("frequency_hz", [True, False, 0.0, -1.0, float("inf"), float("nan")])
@pytest.mark.parametrize("kind", [AudioCommandKind.NOTE_ON, AudioCommandKind.RETUNE])
def test_frequency_payloads_must_be_positive_finite_numbers(
    kind: AudioCommandKind,
    frequency_hz: object,
) -> None:
    payload: dict[str, object] = {"frequency_hz": frequency_hz}
    if kind is AudioCommandKind.NOTE_ON:
        payload["generation"] = 1

    with pytest.raises(ValueError, match="frequency"):
        AudioCommand(kind, **payload)


@pytest.mark.parametrize("generation", [True, False, -1, 1.5])
@pytest.mark.parametrize(
    ("kind", "other_payload"),
    [
        (AudioCommandKind.NOTE_ON, {"frequency_hz": 220.0}),
        (AudioCommandKind.CLEAR_CAPTURE, {}),
        (AudioCommandKind.REPLACE_PATCH, {"patch": SynthPatch()}),
        (AudioCommandKind.RESET, {}),
    ],
)
def test_generations_must_be_nonnegative_integers(
    kind: AudioCommandKind,
    other_payload: dict[str, object],
    generation: object,
) -> None:
    with pytest.raises(ValueError, match="generation"):
        AudioCommand(kind, generation=generation, **other_payload)


def test_command_kind_and_patch_payload_require_their_declared_types() -> None:
    with pytest.raises(ValueError, match="kind"):
        AudioCommand("note_off")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="patch"):
        AudioCommand(
            AudioCommandKind.REPLACE_PATCH,
            patch=object(),  # type: ignore[arg-type]
            generation=1,
        )
