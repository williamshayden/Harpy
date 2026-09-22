"""Exercise CLI restart and compatibility boundaries through real bundles."""

from __future__ import annotations

import json

import pytest

from harpy.cli import main


@pytest.mark.parametrize("task", ["sound-match-v1", "pitch-control-v1"])
def test_interrupted_cli_restores_arguments_and_runs_missing_only(tmp_path, monkeypatch, task):
    from harpy import _cli_bundles
    from harpy._journal import read_configuration

    output = tmp_path / task
    sound = task == "sound-match-v1"
    arguments = (
        ["--task", task, "--suite", "tone-smoke", "--actor", "fixed", "--seed", "2"]
        if sound
        else ["--bundle", "--suite", "smoke", "--actor", "oracle", "--trace"]
    )

    def interrupt(self, update):
        if update["completed"] == 2:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(_cli_bundles.Progress, "__call__", interrupt)
        with pytest.raises(KeyboardInterrupt):
            main(["evaluate", *arguments, "--output", str(output)])
    saved = read_configuration(output)
    committed = (output / "records.jsonl").read_bytes()
    assert main(["summarize", str(output)]) == 2
    assert main(["summarize", str(output), "--allow-incomplete"]) == 0
    assert main(["evaluate", "--resume", "--output", str(output)]) == 0
    assert (output / "records.jsonl").read_bytes().startswith(committed)
    assert read_configuration(output)["configuration"] == saved["configuration"]
    from harpy.experiments import durable as pitch
    from harpy.tasks.sound_match import durable as matching

    report = (matching if sound else pitch).read_bundle(output)
    assert report.complete and report.attempts == report.expected
    assert main(["summarize", str(output), "--verify-replay"]) == 0
    assert main(["evaluate", "--resume", "--suite", "invalid", "--output", str(output)]) == 2


def test_sound_cli_can_read_complete_only_m1_bundle(tmp_path):
    from harpy.tasks.sound_match.actors import builtin_actor_spec
    from harpy.tasks.sound_match.protocols import smoke_episodes
    from harpy.tasks.sound_match.results import save_result
    from harpy.tasks.sound_match.runner import run

    output = tmp_path / "m1"
    save_result(run(smoke_episodes("tone-smoke")[0], builtin_actor_spec("fixed")), output)
    assert main(["summarize", str(output), "--verify-replay"]) == 0


def test_dataset_cli_binds_split_manifest_and_catalog(tmp_path):
    from harpy._journal import read_configuration
    from harpy.tasks.sound_match.datasets import load_manifest

    output = tmp_path / "validation"
    manifest = load_manifest("validation")
    episode = manifest.episodes[0]
    assert (
        main(
            [
                "run",
                "--task",
                "sound-match-v1",
                "--suite",
                "validation",
                "--episode",
                episode.episode_id,
                "--actor",
                "fixed",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    dataset = read_configuration(output)["configuration"]["dataset"]
    assert dataset["manifest_sha256"] == manifest.manifest_sha256
    assert dataset["split"] == "validation"
    assert len(dataset["catalog_sha256"]) == 64
    assert main(["run", "--resume", "--output", str(output)]) == 0


def test_invalid_resume_metadata_is_reported_without_execution(tmp_path, capsys):
    output = tmp_path / "bundle"
    assert main(["run", "--bundle", "--actor", "oracle", "--output", str(output)]) == 0
    path = output / "manifest.json"
    document = json.loads(path.read_text())
    document["configuration"]["options"]["source_cents"] = [1]
    path.write_text(json.dumps(document))
    assert main(["run", "--resume", "--output", str(output)]) == 2
    assert "wrong type" in capsys.readouterr().err


def test_inverse_training_dispatch_and_portable_actor_alias(tmp_path, monkeypatch):
    from harpy._cli_bundles import actor_requests
    from harpy.tasks.sound_match import inverse
    from harpy.tasks.sound_match.actors import builtin_actor_spec

    calls = []
    monkeypatch.setattr(
        inverse, "train_inverse", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    output = tmp_path / "trained"
    assert main(["train", "inverse", "--seed", "2", "--output", str(output)]) == 0
    assert calls == [((output,), {"profile": "quick", "seed": 2, "device": "cpu"})]
    calls.clear()

    def model(path, **kwargs):
        calls.append((path, kwargs))
        return builtin_actor_spec("fixed")

    monkeypatch.setattr(inverse, "inverse_actor_spec", model)
    specs, descriptors, locators = actor_requests(
        [f"seed2=inverse-refine:{output}"], task="sound-match-v1", device="cpu"
    )
    assert specs[0].name == "seed2"
    assert descriptors == [{"name": "seed2", "kind": "artifact", "algorithm": "inverse-refine"}]
    assert locators == {"actors": {"seed2": str(output.resolve())}}
    assert calls == [(output.resolve(), {"refine": True, "device": "cpu"})]


def test_sound_builtin_names_match_public_api():
    from harpy._cli_bundles import SOUND_BUILTINS
    from harpy.tasks.sound_match.actors import BUILTIN_ACTORS

    assert set(BUILTIN_ACTORS) == SOUND_BUILTINS


@pytest.mark.parametrize("task", ["sound-match-v1", "pitch-control-v1"])
def test_saved_builtin_cannot_be_reinterpreted_as_an_artifact(tmp_path, monkeypatch, task):
    from harpy import _cli_bundles

    output = tmp_path / "bundle"
    arguments = (
        ["--task", task, "--actor", "fixed"]
        if task == "sound-match-v1"
        else ["--bundle", "--actor", "oracle"]
    )
    assert main(["run", *arguments, "--output", str(output)]) == 0
    path = output / "manifest.json"
    document = json.loads(path.read_text())
    document["configuration"]["actors"][0]["algorithm"] = str(tmp_path / "model")
    path.write_text(json.dumps(document))

    def unexpected_actor_load(*args, **kwargs):
        pytest.fail("Malformed builtin metadata must fail before actor construction")

    monkeypatch.setattr(_cli_bundles, "actor_requests", unexpected_actor_load)
    assert main(["run", "--resume", "--output", str(output)]) == 2
