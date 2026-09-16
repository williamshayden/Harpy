"""Bundle exports and malformed ledgers must agree with recorded evidence."""

import hashlib
import platform
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from harpy._serialization import canonical_json_bytes, decode_json_bytes
from harpy.cli import main
from harpy.tasks.sound_match import results
from harpy.tasks.sound_match.models import INFORMATION_TRACK, EpisodeSpec
from harpy.tasks.sound_match.objective import score
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import render


@pytest.fixture(scope="module")
def result():
    patch = Patch(gain_db=0)
    episode = EpisodeSpec("integrity", replace(patch, notes_cents=(6100,)), patch)
    target, initial = render(episode.target_patch), render(patch)
    candidate = results.CandidateEvidence(
        1, patch, results.audio_hash(initial), score(target, initial), (100,)
    )
    decision = {
        "index": 1,
        "action": "submit",
        "proposal": None,
        "status": "submitted",
        "reason": None,
        "error_type": None,
        "evaluation": None,
        "candidate": None,
        "model_calls": 0,
    }
    record = results.EpisodeRecord(
        "integrity",
        "fixed",
        results.audio_hash(target),
        candidate,
        (decision,),
        "submitted",
        "explicit_submit",
        None,
        1,
        0,
        0.0,
    )
    files = [{"path": "fixture.py", "sha256": "a" * 64, "size_bytes": 1}]
    source = {
        "kind": "package-python-files",
        "package_version": "1.0.0",
        "files": files,
        "sha256": hashlib.sha256(canonical_json_bytes({"files": files})).hexdigest(),
    }
    return results.ExperimentResult(
        (episode,),
        (
            {
                "name": "fixed",
                "information_track": INFORMATION_TRACK,
                "metadata": {"status": "unknown", "reason": "synthetic fixture"},
                "artifact_files": [],
            },
        ),
        (record,),
        (
            {
                "episode_id": "integrity",
                "audio_sha256": results.audio_hash(target),
                "dsp_renders": 1,
                "cache_hits": 0,
                "elapsed_seconds": 0.0,
            },
        ),
        source,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        {"policy": "lru-waveforms-v1", "max_audio_bytes": 134217728},
        0.0,
    )


@pytest.fixture
def bundle(result, tmp_path):
    output = tmp_path / "result"
    results.save_result(result, output)
    return output


def replace_payload(bundle, name, content):
    """Recompute outer hashes so a semantic defect reaches the strict reader."""
    (bundle / name).write_bytes(content)
    manifest = decode_json_bytes((bundle / "manifest.json").read_bytes())
    for entry in manifest["files"]:
        if entry["path"] == name:
            entry.update(sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content))
    encoded = canonical_json_bytes(manifest)
    (bundle / "manifest.json").write_bytes(encoded)
    completion = decode_json_bytes((bundle / "complete.json").read_bytes())
    completion["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (bundle / "complete.json").write_bytes(canonical_json_bytes(completion))


def test_valid_bundle_reads_each_payload_once_without_rendering(bundle, result, monkeypatch):
    calls = []
    original = Path.read_bytes

    def read(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(results, "render", lambda _: pytest.fail("ordinary loading rendered audio"))
    loaded = results.load_result(bundle)
    assert loaded == result
    assert len(calls) == len(set(calls)) == 6


@pytest.mark.parametrize("role", ("target", "submitted"))
def test_exported_samples_must_match_evidence_even_with_recomputed_file_hashes(bundle, role):
    silent = results._wav_bytes(np.zeros(96000, dtype=np.float32))
    replace_payload(bundle, f"audio/{role}-0000.wav", silent)
    with pytest.raises(ValueError, match=f"{role} WAV samples differ"):
        results.load_result(bundle)


@pytest.mark.parametrize("corruption", ("sample_rate", "format", "truncated", "nonfinite"))
def test_wav_format_and_sample_finiteness_are_validated(bundle, corruption):
    filename = "audio/target-0000.wav"
    content = bytearray((bundle / filename).read_bytes())
    if corruption == "sample_rate":
        content[24:28] = struct.pack("<I", 44100)
    elif corruption == "format":
        content[20:22] = struct.pack("<H", 1)
    elif corruption == "truncated":
        del content[-4:]
    else:
        content[56:60] = struct.pack("<f", float("inf"))
    replace_payload(bundle, filename, bytes(content))
    with pytest.raises(ValueError, match="audio export"):
        results.load_result(bundle)


def test_patch_export_boolean_is_not_equal_to_integer_zero(bundle):
    filename = "patches/submitted-0000.json"
    document = decode_json_bytes((bundle / filename).read_bytes())
    document["patch"]["gain_db"] = False
    replace_payload(bundle, filename, canonical_json_bytes(document))
    with pytest.raises(ValueError, match="submitted patch export"):
        results.load_result(bundle)


@pytest.mark.parametrize("value", (True, 1.0))
def test_recomputed_metrics_preserve_integer_json_types(bundle, value):
    document = decode_json_bytes((bundle / "records.jsonl").read_bytes())
    document["metrics"]["evaluations"] = value
    replace_payload(bundle, "records.jsonl", canonical_json_bytes(document))
    with pytest.raises(ValueError, match="record metrics"):
        results.load_result(bundle)


@pytest.mark.parametrize(
    "field,value",
    (
        ("model_calls", "unknown"),
        ("status", []),
        ("elapsed_seconds", 10**400),
    ),
)
def test_malformed_record_values_raise_validation_errors_without_cli_traceback(
    bundle, field, value, capsys
):
    document = decode_json_bytes((bundle / "records.jsonl").read_bytes())
    if field == "model_calls":
        document["ledger"][0][field] = value
    else:
        document[field] = value
    replace_payload(bundle, "records.jsonl", canonical_json_bytes(document))
    with pytest.raises(ValueError):
        results.load_result(bundle)
    assert main(["summarize", str(bundle)]) == 2
    error = capsys.readouterr().err
    assert "harpy:" in error
    assert "Traceback" not in error


def test_explicit_replay_still_verifies_complete_valid_bundle(bundle):
    results.verify_replay(results.load_result(bundle))
