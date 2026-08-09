from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from harpy.synth.models import EnvelopeConfig, OscillatorConfig, OscillatorType, SynthPatch

_ROOT_KEYS = {"schema_version", "oscillator", "envelope", "output_gain_dbfs"}
_OSCILLATOR_KEYS = {"type"}
_V1_ENVELOPE_KEYS = {
    "attack_seconds",
    "decay_seconds",
    "sustain_db",
    "release_seconds",
    "curve",
}
_V2_ENVELOPE_KEYS = {
    "attack_seconds",
    "decay_seconds",
    "sustain_db",
    "release_seconds",
    "attack_curve",
    "decay_curve",
    "release_curve",
}
_JSON_WHITESPACE = " \t\r\n"


class _NonFiniteConstant(NamedTuple):
    value: str


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
    try:
        value, end = decoder.raw_decode(text, start)
    except _NonFiniteConstantError as error:
        field_name = _diagnose_non_finite_field(text, start)
        if field_name is not None:
            raise ValueError(f"{field_name} must be a finite JSON number") from error
        raise
    if any(character not in _JSON_WHITESPACE for character in text[end:]):
        raise ValueError("patch JSON contains trailing content")

    root = _object(value, "patch")
    _validate_keys(root, _ROOT_KEYS, "patch")
    schema_version = _require_schema_version(root["schema_version"])

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
    envelope_keys = _V1_ENVELOPE_KEYS if schema_version == 1 else _V2_ENVELOPE_KEYS
    _validate_keys(envelope_data, envelope_keys, "envelope")
    numeric_fields = ["attack_seconds", "decay_seconds", "sustain_db", "release_seconds"]
    if schema_version == 2:
        numeric_fields.extend(["attack_curve", "decay_curve", "release_curve"])
    for field_name in numeric_fields:
        _number(envelope_data[field_name], f"envelope.{field_name}")
    if schema_version == 1:
        if not isinstance(envelope_data["curve"], str):
            raise ValueError("envelope.curve must be a string")
        if envelope_data["curve"] != "linear_amplitude":
            raise ValueError("envelope.curve is unsupported")
        envelope_arguments = {
            field_name: envelope_data[field_name] for field_name in numeric_fields
        }
        envelope_arguments.update(
            attack_curve=0.0,
            decay_curve=0.0,
            release_curve=0.0,
        )
    else:
        envelope_arguments = envelope_data
    try:
        envelope = EnvelopeConfig(**envelope_arguments)
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
        "schema_version": 2,
        "oscillator": {"type": patch.oscillator.type.value},
        "envelope": {
            "attack_seconds": patch.envelope.attack_seconds,
            "decay_seconds": patch.envelope.decay_seconds,
            "sustain_db": patch.envelope.sustain_db,
            "release_seconds": patch.envelope.release_seconds,
            "attack_curve": patch.envelope.attack_curve,
            "decay_curve": patch.envelope.decay_curve,
            "release_curve": patch.envelope.release_curve,
        },
        "output_gain_dbfs": patch.output_gain_dbfs,
    }
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def load_patch(path: Path) -> SynthPatch:
    return loads_patch(path.read_text(encoding="utf-8"))


def save_patch(path: Path, patch: SynthPatch) -> None:
    destination = Path(path)
    text = dumps_patch(patch)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = -1
        with stream:
            written = stream.write(text)
            if written != len(text):
                raise OSError("incomplete patch write")
            stream.flush()
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise _NonFiniteConstantError(value)


def _non_finite_constant(value: str) -> _NonFiniteConstant:
    return _NonFiniteConstant(value)


def _diagnose_non_finite_field(text: str, start: int) -> str | None:
    diagnostic_decoder = json.JSONDecoder(
        object_pairs_hook=_object_without_duplicates,
        parse_constant=_non_finite_constant,
    )
    try:
        document, _ = diagnostic_decoder.raw_decode(text, start)
    except ValueError:
        return None

    if not isinstance(document, dict):
        return None
    if isinstance(document.get("output_gain_dbfs"), _NonFiniteConstant):
        return "output_gain_dbfs"
    envelope = document.get("envelope")
    if not isinstance(envelope, dict):
        return None
    for field_name in (
        "attack_seconds",
        "decay_seconds",
        "sustain_db",
        "release_seconds",
        "attack_curve",
        "decay_curve",
        "release_curve",
    ):
        if isinstance(envelope.get(field_name), _NonFiniteConstant):
            return f"envelope.{field_name}"
    return None


def _first_non_whitespace_index(text: str) -> int | None:
    for index, character in enumerate(text):
        if character not in _JSON_WHITESPACE:
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


def _require_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("schema_version must be an integer")
    if value not in (1, 2):
        raise ValueError(f"schema_version {value} is unsupported")
    return value


def _number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a JSON number")


class _NonFiniteConstantError(ValueError):
    pass
