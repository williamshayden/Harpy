import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from harpy.synth import patch_json
from harpy.synth.models import EnvelopeConfig, SynthPatch
from harpy.synth.patch_json import dumps_patch, load_patch, loads_patch, save_patch

V1_DEFAULT = """{
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
V2_DEFAULT = """{
  "schema_version": 2,
  "oscillator": {
    "type": "sine"
  },
  "envelope": {
    "attack_seconds": 0.001,
    "decay_seconds": 0.6,
    "sustain_db": -6.0,
    "release_seconds": 0.6,
    "attack_curve": 0.0,
    "decay_curve": 0.0,
    "release_curve": 0.0
  },
  "output_gain_dbfs": -12.0
}
"""
V2_NEGATIVE_ZERO = """{
  "schema_version": 2,
  "oscillator": {
    "type": "sine"
  },
  "envelope": {
    "attack_seconds": 0.001,
    "decay_seconds": 0.6,
    "sustain_db": -0.0,
    "release_seconds": 0.6,
    "attack_curve": -0.0,
    "decay_curve": -0.0,
    "release_curve": -0.0
  },
  "output_gain_dbfs": -0.0
}
"""
V2_POSITIVE_ZERO = V2_NEGATIVE_ZERO.replace("-0.0", "0.0")


def replace_json_value(text: str, field: str, replacement: str) -> str:
    document = json.loads(text)
    if field.startswith("envelope."):
        section, name = field.split(".")
        original = json.dumps(document[section][name])
        needle = f'"{name}": {original}'
    else:
        original = json.dumps(document[field])
        needle = f'"{field}": {original}'
    return text.replace(needle, f'"{field.rsplit(".", 1)[-1]}": {replacement}', 1)


def test_v1_migrates_to_zero_curve_runtime_patch() -> None:
    assert loads_patch(V1_DEFAULT) == SynthPatch()


def test_default_patch_writes_canonical_v2() -> None:
    assert dumps_patch(SynthPatch()) == V2_DEFAULT


def test_v1_loaded_then_saved_becomes_canonical_v2() -> None:
    assert dumps_patch(loads_patch(V1_DEFAULT)) == V2_DEFAULT


def test_v2_negative_zero_numbers_rewrite_as_canonical_positive_zero() -> None:
    dumped = dumps_patch(loads_patch(V2_NEGATIVE_ZERO))

    assert "-0.0" not in dumped
    assert dumped == V2_POSITIVE_ZERO


def test_patch_json_round_trip_preserves_values() -> None:
    patch = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.125,
            sustain_db=-9.5,
            attack_curve=-0.25,
            decay_curve=0.5,
            release_curve=1.0,
        ),
        output_gain_dbfs=-18.25,
    )

    assert loads_patch(dumps_patch(patch)) == patch


@pytest.mark.parametrize("whitespace", [" ", "\t", "\r", "\n", " \t\r\n"])
def test_loads_patch_accepts_each_json_whitespace_character(whitespace: str) -> None:
    assert loads_patch(whitespace + V2_DEFAULT + whitespace) == SynthPatch()


@pytest.mark.parametrize("text", ["\u00a0" + V2_DEFAULT, V2_DEFAULT + "\u00a0"])
def test_loads_patch_rejects_non_json_unicode_whitespace(text: str) -> None:
    with pytest.raises(ValueError):
        loads_patch(text)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (V2_DEFAULT.replace('"type": "sine"', '"type": "sine",\n    "extra": 1'), "extra"),
        (
            V2_DEFAULT.replace(
                '"release_curve": 0.0',
                '"release_curve": 0.0,\n    "extra": 1',
            ),
            "extra",
        ),
        (
            V2_DEFAULT.replace(
                '"output_gain_dbfs": -12.0',
                '"output_gain_dbfs": -12.0,\n  "extra": 1',
            ),
            "extra",
        ),
    ],
)
def test_loads_patch_rejects_unknown_keys_and_names_them(text: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(text)


@pytest.mark.parametrize(
    ("text", "field"),
    [
        (V2_DEFAULT.replace('  "schema_version": 2,\n', ""), "schema_version"),
        (
            V2_DEFAULT.replace(
                '  "oscillator": {\n    "type": "sine"\n  },\n',
                '  "oscillator": {},\n',
            ),
            "type",
        ),
        (
            V1_DEFAULT.replace(
                '    "release_seconds": 0.6,\n    "curve": "linear_amplitude"',
                '    "release_seconds": 0.6',
            ),
            "curve",
        ),
        (V2_DEFAULT.replace('    "attack_curve": 0.0,\n', ""), "attack_curve"),
    ],
)
def test_loads_patch_rejects_missing_keys_and_names_them(text: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(text)


def test_v1_rejects_v2_curve_number_keys() -> None:
    text = V1_DEFAULT.replace(
        '    "curve": "linear_amplitude"',
        '    "curve": "linear_amplitude",\n    "attack_curve": 0.0',
    )

    with pytest.raises(ValueError, match="attack_curve"):
        loads_patch(text)


def test_v2_rejects_v1_curve_string() -> None:
    text = V2_DEFAULT.replace(
        '    "release_curve": 0.0',
        '    "release_curve": 0.0,\n    "curve": "linear_amplitude"',
    )

    with pytest.raises(ValueError, match="curve"):
        loads_patch(text)


@pytest.mark.parametrize("version", ["0", "3", "999"])
def test_loads_patch_rejects_unsupported_integer_versions(version: str) -> None:
    text = V2_DEFAULT.replace('"schema_version": 2', f'"schema_version": {version}', 1)

    with pytest.raises(ValueError, match="unsupported"):
        loads_patch(text)


@pytest.mark.parametrize("version", ['"2"', "2.0", "true", "null"])
def test_loads_patch_requires_an_integer_schema_version(version: str) -> None:
    text = V2_DEFAULT.replace('"schema_version": 2', f'"schema_version": {version}', 1)

    with pytest.raises(ValueError, match=r"schema_version.*integer"):
        loads_patch(text)


@pytest.mark.parametrize(
    "text",
    [
        V2_DEFAULT.replace(
            '"schema_version": 2,',
            '"schema_version": 2,\n  "schema_version": 2,',
        ),
        V2_DEFAULT.replace(
            '"attack_curve": 0.0,',
            '"attack_curve": 0.0,\n    "attack_curve": 0.0,',
        ),
    ],
)
def test_loads_patch_rejects_duplicate_keys_before_object_conversion(text: str) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        loads_patch(text)


@pytest.mark.parametrize(
    "field",
    [
        "envelope.attack_seconds",
        "envelope.decay_seconds",
        "envelope.sustain_db",
        "envelope.release_seconds",
        "envelope.attack_curve",
        "envelope.decay_curve",
        "envelope.release_curve",
        "output_gain_dbfs",
    ],
)
@pytest.mark.parametrize("value", ["true", "false"])
def test_loads_patch_rejects_booleans_used_as_numbers(field: str, value: str) -> None:
    text = replace_json_value(V2_DEFAULT, field, value)

    with pytest.raises(ValueError, match=field.rsplit(".", 1)[-1]):
        loads_patch(text)


@pytest.mark.parametrize(
    "field",
    [
        "envelope.attack_seconds",
        "envelope.attack_curve",
        "envelope.decay_curve",
        "envelope.release_curve",
        "output_gain_dbfs",
    ],
)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_loads_patch_names_non_finite_number_fields(
    field: str,
    value: str,
) -> None:
    text = replace_json_value(V2_DEFAULT, field, value)

    with pytest.raises(ValueError, match=field.rsplit(".", 1)[-1]):
        loads_patch(text)


@pytest.mark.parametrize("field", ["envelope.attack_seconds", "envelope.attack_curve"])
def test_loads_patch_turns_huge_json_integer_into_named_value_error(field: str) -> None:
    text = replace_json_value(V2_DEFAULT, field, "9" * 400)

    with pytest.raises(ValueError, match=field.rsplit(".", 1)[-1]):
        loads_patch(text)


@pytest.mark.parametrize("field", ["attack_curve", "decay_curve", "release_curve"])
@pytest.mark.parametrize("value", ["-1.001", "1.001"])
def test_loads_patch_rejects_curve_numbers_outside_closed_range(
    field: str,
    value: str,
) -> None:
    text = replace_json_value(V2_DEFAULT, f"envelope.{field}", value)

    with pytest.raises(ValueError, match=field):
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
def test_loads_patch_rejects_unsupported_closed_set_values(
    old: str,
    new: str,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        loads_patch(V1_DEFAULT.replace(old, new))


def test_loads_patch_rejects_trailing_non_whitespace_content() -> None:
    with pytest.raises(ValueError, match="trailing"):
        loads_patch(V2_DEFAULT + "unexpected")


def test_save_and_load_patch_use_utf8_files(tmp_path: Path) -> None:
    path = tmp_path / "pätch.json"
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


def test_failed_replace_preserves_existing_patch_and_removes_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "patch.json"
    path.write_text("existing patch\n", encoding="utf-8")

    def failing_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        save_patch(path, replace(SynthPatch(), output_gain_dbfs=-3.0))

    assert path.read_text(encoding="utf-8") == "existing patch\n"
    assert [item.name for item in tmp_path.iterdir()] == ["patch.json"]
