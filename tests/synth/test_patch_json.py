import os
from pathlib import Path

import pytest

from harpy.synth import patch_json
from harpy.synth.models import EnvelopeConfig, SynthPatch
from harpy.synth.patch_json import dumps_patch, load_patch, loads_patch, save_patch

EXPECTED_DEFAULT = """{
  "schema_version": 1,
  "oscillator": {
    "type": "sine"
  },
  "envelope": {
    "attack_seconds": 0.001,
    "decay_seconds": 0.6,
    "sustain_db": -6.0,
    "release_seconds": 0.6,
    "curve": "linear_amplitude"
  },
  "output_gain_dbfs": -12.0
}
"""
UNKNOWN_OSCILLATOR_KEY = EXPECTED_DEFAULT.replace(
    '"type": "sine"',
    '"type": "sine",\n    "extra": 1',
)
UNKNOWN_ROOT_KEY = EXPECTED_DEFAULT.replace(
    '"output_gain_dbfs": -12.0',
    '"output_gain_dbfs": -12.0,\n  "extra": 1',
)
MISSING_SCHEMA_VERSION = EXPECTED_DEFAULT.replace('  "schema_version": 1,\n', "")
MISSING_OSCILLATOR_TYPE = EXPECTED_DEFAULT.replace(
    '  "oscillator": {\n    "type": "sine"\n  },\n',
    '  "oscillator": {},\n',
)
MISSING_ENVELOPE_CURVE = EXPECTED_DEFAULT.replace(
    '    "release_seconds": 0.6,\n    "curve": "linear_amplitude"\n',
    '    "release_seconds": 0.6\n',
)


def test_default_patch_has_canonical_json() -> None:
    assert dumps_patch(SynthPatch()) == EXPECTED_DEFAULT


def test_patch_json_round_trip_preserves_values() -> None:
    patch = SynthPatch(
        envelope=EnvelopeConfig(attack_seconds=0.125, sustain_db=-9.5),
        output_gain_dbfs=-18.25,
    )

    assert loads_patch(dumps_patch(patch)) == patch


@pytest.mark.parametrize("whitespace", [" ", "\t", "\r", "\n", " \t\r\n"])
def test_loads_patch_accepts_each_json_whitespace_character(whitespace: str) -> None:
    assert loads_patch(whitespace + EXPECTED_DEFAULT + whitespace) == SynthPatch()


@pytest.mark.parametrize("text", ["\u00a0" + EXPECTED_DEFAULT, EXPECTED_DEFAULT + "\u00a0"])
def test_loads_patch_rejects_non_json_unicode_whitespace(text: str) -> None:
    with pytest.raises(ValueError):
        loads_patch(text)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (UNKNOWN_OSCILLATOR_KEY, "extra"),
        (UNKNOWN_ROOT_KEY, "extra"),
    ],
)
def test_loads_patch_rejects_unknown_keys_and_names_them(text: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(text)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (MISSING_SCHEMA_VERSION, "schema_version"),
        (MISSING_OSCILLATOR_TYPE, "type"),
        (MISSING_ENVELOPE_CURVE, "curve"),
    ],
)
def test_loads_patch_rejects_missing_keys_and_names_them(text: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(text)


@pytest.mark.parametrize("version", ['"1"', "2", "1.0", "true", "null"])
def test_loads_patch_requires_schema_version_integer_one(version: str) -> None:
    text = EXPECTED_DEFAULT.replace("1,", f"{version},", 1)

    with pytest.raises(ValueError, match="schema_version"):
        loads_patch(text)


def test_loads_patch_rejects_duplicate_keys_before_object_conversion() -> None:
    text = EXPECTED_DEFAULT.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
    )

    with pytest.raises(ValueError, match="schema_version"):
        loads_patch(text)


@pytest.mark.parametrize("value", ["true", "false"])
def test_loads_patch_rejects_booleans_used_as_numbers(value: str) -> None:
    text = EXPECTED_DEFAULT.replace("-12.0", value)

    with pytest.raises(ValueError, match="output_gain_dbfs"):
        loads_patch(text)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_loads_patch_names_output_gain_and_never_returns_a_patch_for_non_finite_constants(
    value: str,
) -> None:
    text = EXPECTED_DEFAULT.replace("-12.0", value)

    with pytest.raises(ValueError, match="output_gain_dbfs"):
        loads_patch(text)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_loads_patch_names_envelope_field_for_non_finite_json_constants(value: str) -> None:
    text = EXPECTED_DEFAULT.replace("0.001", value)

    with pytest.raises(ValueError, match="attack_seconds"):
        loads_patch(text)


def test_loads_patch_turns_an_unrepresentable_json_integer_into_a_named_value_error() -> None:
    text = EXPECTED_DEFAULT.replace("0.001", "9" * 400)

    with pytest.raises(ValueError, match="attack_seconds"):
        loads_patch(text)


def test_strict_non_finite_callback_always_raises() -> None:
    with pytest.raises(ValueError, match="NaN"):
        patch_json._reject_non_finite_constant("NaN")


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ('"sine"', '"square"', "type"),
        ('"linear_amplitude"', '"exponential"', "curve"),
    ],
)
def test_loads_patch_rejects_unsupported_closed_set_values(old: str, new: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(EXPECTED_DEFAULT.replace(old, new))


def test_loads_patch_rejects_trailing_non_whitespace_content() -> None:
    with pytest.raises(ValueError):
        loads_patch(EXPECTED_DEFAULT + "unexpected")


def test_save_and_load_patch_use_utf8_files(tmp_path: Path) -> None:
    path = tmp_path / "patch.json"
    patch = SynthPatch(output_gain_dbfs=-3.0)

    save_patch(path, patch)

    assert path.read_text(encoding="utf-8") == dumps_patch(patch)
    assert load_patch(path) == patch


def test_failed_partial_save_preserves_existing_patch_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "patch.json"
    path.write_text("existing patch\n", encoding="utf-8")
    real_fdopen = os.fdopen

    class FailingWriter:
        def __init__(self, stream) -> None:  # type: ignore[no-untyped-def]
            self._stream = stream

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            self._stream.close()

        def write(self, text: str) -> int:
            self._stream.write(text[:16])
            self._stream.flush()
            raise OSError("injected partial write")

        def flush(self) -> None:
            self._stream.flush()

    def failing_fdopen(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FailingWriter(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(os, "fdopen", failing_fdopen)

    with pytest.raises(OSError, match="injected partial write"):
        save_patch(path, SynthPatch(output_gain_dbfs=-3.0))

    assert path.read_text(encoding="utf-8") == "existing patch\n"
    assert [item.name for item in tmp_path.iterdir()] == ["patch.json"]
