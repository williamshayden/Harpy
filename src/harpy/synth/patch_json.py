from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from harpy.synth.models import EnvelopeConfig, OscillatorConfig, OscillatorType, SynthPatch

_ROOT_KEYS = {"schema_version", "oscillator", "envelope", "output_gain_dbfs"}
_OSCILLATOR_KEYS = {"type"}
_ENVELOPE_KEYS = {
    "attack_seconds",
    "decay_seconds",
    "sustain_db",
    "release_seconds",
    "curve",
}


def loads_patch(text: str) -> SynthPatch:
    if not isinstance(text, str):
        raise ValueError("patch JSON must be text")

    start = _first_non_whitespace_index(text)
    if start is None:
        raise ValueError("patch JSON must contain an object")

    decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_reject_non_finite_constant,
    )
    value, end = decoder.raw_decode(text, start)
    if text[end:].strip():
        raise ValueError("patch JSON contains trailing content")

    root = _object(value, "patch")
    _validate_keys(root, _ROOT_KEYS, "patch")
    _require_schema_version(root["schema_version"])

    oscillator_data = _object(root["oscillator"], "oscillator")
    _validate_keys(oscillator_data, _OSCILLATOR_KEYS, "oscillator")
    oscillator_type = oscillator_data["type"]
    if not isinstance(oscillator_type, str):
        raise ValueError("oscillator.type must be a string")
    try:
        oscillator = OscillatorConfig(type=OscillatorType(oscillator_type))
    except ValueError as error:
        raise ValueError("oscillator.type is unsupported") from error

    envelope_data = _object(root["envelope"], "envelope")
    _validate_keys(envelope_data, _ENVELOPE_KEYS, "envelope")
    for field_name in ("attack_seconds", "decay_seconds", "sustain_db", "release_seconds"):
        _number(envelope_data[field_name], f"envelope.{field_name}")
    if not isinstance(envelope_data["curve"], str):
        raise ValueError("envelope.curve must be a string")
    try:
        envelope = EnvelopeConfig(**envelope_data)
    except ValueError as error:
        raise ValueError(f"envelope is invalid: {error}") from error

    _number(root["output_gain_dbfs"], "output_gain_dbfs")
    try:
        return SynthPatch(
            oscillator=oscillator,
            envelope=envelope,
            output_gain_dbfs=root["output_gain_dbfs"],
        )
    except ValueError as error:
        raise ValueError(f"output_gain_dbfs is invalid: {error}") from error


def dumps_patch(patch: SynthPatch) -> str:
    if not isinstance(patch, SynthPatch):
        raise ValueError("patch must be a SynthPatch")
    document = {
        "schema_version": 1,
        "oscillator": {"type": patch.oscillator.type.value},
        "envelope": {
            "attack_seconds": patch.envelope.attack_seconds,
            "decay_seconds": patch.envelope.decay_seconds,
            "sustain_db": patch.envelope.sustain_db,
            "release_seconds": patch.envelope.release_seconds,
            "curve": patch.envelope.curve,
        },
        "output_gain_dbfs": patch.output_gain_dbfs,
    }
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def load_patch(path: Path) -> SynthPatch:
    return loads_patch(path.read_text(encoding="utf-8"))


def save_patch(path: Path, patch: SynthPatch) -> None:
    path.write_text(dumps_patch(patch), encoding="utf-8")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _first_non_whitespace_index(text: str) -> int | None:
    for index, character in enumerate(text):
        if not character.isspace():
            return index
    return None


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _validate_keys(value: dict[str, Any], expected: set[str], field_name: str) -> None:
    missing = expected - value.keys()
    if missing:
        raise ValueError(f"{field_name}.{sorted(missing)[0]} is required")
    unexpected = value.keys() - expected
    if unexpected:
        raise ValueError(f"{field_name}.{sorted(unexpected)[0]} is not supported")


def _require_schema_version(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("schema_version must be the integer 1")


def _number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")
