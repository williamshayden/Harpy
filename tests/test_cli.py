"""The supported command surface executes real workflows and fails create-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harpy.cli import main
from harpy.experiments.results import load_result


@pytest.mark.parametrize("actor", ["spectrum-peak", "waveform-fft"])
def test_run_and_all_readout_formats(tmp_path: Path, capsys, actor: str) -> None:
    path = tmp_path / "episode.json"
    assert main(["run", "--actor", actor, "--source-cents", "6000", "--output", str(path)]) == 0
    result = load_result(path)
    assert result.records[0].terminal.submitted_success
    assert result.records[0].trace
    for format in ("text", "markdown", "csv"):
        assert main(["summarize", str(path), "--format", format]) == 0
        assert actor in capsys.readouterr().out


def test_waveform_and_spectrum_comparison_keeps_tracks_separate(tmp_path: Path) -> None:
    path = tmp_path / "tracks.json"
    assert (
        main(
            [
                "evaluate",
                "--actor",
                "waveform-fft",
                "--actor",
                "spectrum-peak",
                "--suite",
                "smoke",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    result = load_result(path)
    assert len(result.records) == 12
    assert [(actor["name"], actor["observation_mode"]) for actor in result.actors] == [
        ("waveform-fft", "waveform"),
        ("spectrum-peak", "spectrum"),
    ]
    assert all(record.terminal.submitted_success for record in result.records)


def test_smoke_evaluation_retains_protocol_and_track(tmp_path: Path) -> None:
    path = tmp_path / "matched.json"
    assert (
        main(
            [
                "evaluate",
                "--actor",
                "oracle",
                "--suite",
                "smoke",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    result = load_result(path)
    assert len(result.episodes) == 6
    assert result.protocol["id"] == "harpy-clean-engineering-smoke-v1"
    assert all(record.terminal.submitted_success for record in result.records)
    assert result.actors[0]["observation_mode"] == "oracle"


def test_existing_output_fails_before_actor_loading(tmp_path: Path, capsys) -> None:
    path = tmp_path / "keep.json"
    path.write_text("original")
    assert main(["run", "--actor", "nonexistent-model", "--output", str(path)]) == 2
    assert path.read_text() == "original"
    assert "Output already exists" in capsys.readouterr().err


def test_summarize_rejects_nonobject_json_without_traceback(tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2]))
    assert main(["summarize", str(path)]) == 2
    assert "harpy:" in capsys.readouterr().err


def test_training_dispatch_preserves_explicit_request(tmp_path: Path, monkeypatch) -> None:
    from harpy.experiments import training

    calls = []
    monkeypatch.setattr(training, "train", lambda **kwargs: calls.append(kwargs))
    path = tmp_path / "checkpoint"
    assert (
        main(
            [
                "train",
                "pitch",
                "--profile",
                "checkpoint",
                "--seed",
                "2",
                "--device",
                "cuda",
                "--output",
                str(path),
            ]
        )
        == 0
    )
    assert calls == [
        {
            "trainer": "pitch",
            "profile": "checkpoint",
            "seed": 2,
            "device": "cuda",
            "output": path,
        }
    ]


def test_ppo_trust_warning_precedes_model_deserialization(tmp_path: Path, monkeypatch, capsys):
    import hashlib

    from harpy.experiments import artifacts
    from harpy.learning.artifacts import FileRecord, canonical_json_bytes

    artifact = tmp_path / "ppo"
    artifact.mkdir()
    model = artifact / "model.zip"
    model.write_bytes(b"metadata validation must not execute this archive")
    (artifact / "metadata.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_id": artifacts.METADATA_SCHEMA_ID,
                "trainer": "ppo",
                "profile": "smoke",
                "seed": 0,
                "device": "cpu",
                "configuration": {},
                "summary": {},
                "runtime": {},
            }
        )
    )
    files = tuple(
        FileRecord(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in artifact.iterdir()
    )
    manifest = artifacts.ExperimentManifest(
        "ppo",
        "smoke",
        0,
        "cpu",
        files,
        canonical_json_bytes({"status": "unknown", "reason": "test fixture"}),
    )
    (artifact / "manifest.json").write_bytes(canonical_json_bytes(manifest.to_document()))

    def guarded_load(path, *, trainer, device):
        assert path == model and trainer == "ppo" and device == "cpu"
        assert "trusted-local PPO archive" in capsys.readouterr().err
        raise RuntimeError("model execution blocked by test")

    monkeypatch.setattr(artifacts, "_load_model_file", guarded_load)
    output = tmp_path / "episode.json"
    assert main(["run", "--actor", str(artifact), "--output", str(output)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "harpy: model execution blocked by test\n"
    assert not output.exists()
    assert artifacts.load_artifact(artifact).manifest == manifest


@pytest.mark.parametrize("operation", ["train", "evaluate", "run", "summarize"])
def test_operation_help(operation: str) -> None:
    with pytest.raises(SystemExit, match="0"):
        main([operation, "--help"])
