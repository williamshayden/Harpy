"""Exercise the source-only pilot with local fixture processes, never paid models."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from harpy.tasks.sound_match.durable import iter_records, read_bundle, verify_replay

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
try:
    pilot = importlib.import_module("research.sound_matching.llm.pilot")
finally:
    sys.path.remove(str(REPOSITORY))

SECRET_MARKER = "PRIVATE_FIXTURE_TOKEN_MUST_NOT_APPEAR"


def command_configuration(tmp_path: Path, mode: str = "valid") -> Path:
    script = tmp_path / "completion_fixture.py"
    script.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        f"mode = {mode!r}\n"
        "if mode == 'error':\n"
        f"    print({SECRET_MARKER!r}, file=sys.stderr)\n"
        "    raise SystemExit(23)\n"
        "if mode == 'invalid':\n"
        "    content = 'not a JSON action'\n"
        "elif any(item['role'] == 'assistant' for item in request['messages']):\n"
        "    content = json.dumps({'action': 'submit'})\n"
        "else:\n"
        "    public = json.loads(request['messages'][-1]['content'])\n"
        "    patch = dict(public['current_patch'])\n"
        "    patch['brightness_hz'] = max(200, patch['brightness_hz'] // 2)\n"
        "    content = json.dumps({'action': 'propose', 'patch': patch})\n"
        "print(json.dumps({'content': content, 'response_model': request['model'],\n"
        "                  'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}))\n",
        encoding="utf-8",
    )
    configuration = tmp_path / "command.json"
    configuration.write_text(json.dumps([sys.executable, str(script)]), encoding="utf-8")
    return configuration


def run_fixture(tmp_path: Path, *, mode: str = "valid", maximum: int = 4):
    command = command_configuration(tmp_path, mode)
    output = tmp_path / "pilot"
    status = pilot.main(
        [
            "--command-json",
            str(command),
            "--model",
            "fixture-only",
            "--max-model-calls",
            str(maximum),
            "--timeout-seconds",
            "10",
            "--output",
            str(output),
        ]
    )
    return status, output


def journal_events(output: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted((output / "requests").glob("*.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_command_pilot_runs_real_renderer_durable_readout_and_replay(tmp_path):
    status, output = run_fixture(tmp_path)
    assert status == 0
    outcome = pilot.verify_pilot(output)
    assert outcome["records"] == 8
    assert outcome["terminal_statuses"] == {"submitted": 8}
    assert outcome["request_costs"]["command_attempts_started"] == 4
    assert outcome["request_costs"]["http_attempts_started"] == 0
    assert outcome["request_costs"]["native_provider_calls"] is None
    assert {item["episode_id"] for item in outcome["request_costs"]["controllers"]} == {
        "tone-001",
        "tone-002",
    }
    rows = list(iter_records(output / "evaluation"))
    llm_rows = [row for row in rows if row["key"][1] == pilot.LLM_NAME]
    assert len(llm_rows) == 2
    assert all(row["record"]["metrics"]["model_calls"] == 2 for row in llm_rows)
    assert all(row["record"]["metrics"]["evaluations"] == 2 for row in llm_rows)
    assert all(row["record"]["ledger"][-1]["action"] == "submit" for row in llm_rows)
    assert all(
        {"target_audio", "submitted_audio", "submitted_patch"}
        == {item["role"] for item in row["artifacts"]}
        for row in llm_rows
    )
    verify_replay(output / "evaluation")
    manifest = json.loads((output / "pilot-manifest.json").read_text())
    assert all(
        episode["render_budget"] == 32 and episode["decision_budget"] == 64
        for episode in manifest["episodes"]
    )
    assert manifest["configuration"]["observation_protocol"] == "text-derived-audio-features-v1"
    assert manifest["direct_audio_input"] is False
    filenames = {Path(item["path"]).name for item in manifest["immutable_files"]}
    assert {"actor.py", "features.py", "transport.py", "pilot.py", "bridge_client.mjs"} <= filenames
    assert "system-prompt.txt" in filenames and "configuration.json" in filenames
    assert "target_patch" not in json.dumps(journal_events(output))
    for event in journal_events(output):
        if event["event"] == "attempt_start":
            assert "tone-001" not in json.dumps(event["request"])
            assert "tone-002" not in json.dumps(event["request"])
    bundle = json.loads((output / "evaluation/manifest.json").read_text())
    llm_actor = next(row for row in bundle["identity"]["actors"] if row["name"] == pilot.LLM_NAME)
    assert all(".jsonl" not in item["path"] for item in llm_actor["artifact_files"])


def test_model_attempt_cap_submits_through_explicit_guard_without_retry(tmp_path):
    status, output = run_fixture(tmp_path, maximum=1)
    assert status == 0
    outcome = pilot.verify_pilot(output)
    assert outcome["request_costs"]["transport_attempts_started"] == 2
    assert outcome["request_costs"]["automatic_retries"] == 0
    assert all(item["attempts_started"] == 1 for item in outcome["request_costs"]["controllers"])
    guards = [event for event in journal_events(output) if event["event"] == "controller_guard"]
    assert guards and all(event["model_calls"] == 0 for event in guards)
    assert all(event["origin"] == "controller" for event in guards)
    verify_replay(output / "evaluation")


def test_invalid_responses_are_retained_and_do_not_get_success_completion(tmp_path):
    status, output = run_fixture(tmp_path, mode="invalid", maximum=1)
    assert status == 1
    assert not (output / "complete.json").exists()
    outcome = json.loads((output / "outcome.json").read_text())
    assert outcome["status"] == "completed-with-errors"
    assert outcome["bundle_complete"] is True
    assert outcome["invalid_actions"] == 2
    assert outcome["request_costs"]["transport_attempts_started"] == 2
    assert sum(item["parse_rejections"] for item in outcome["request_costs"]["controllers"]) == 2
    assert read_bundle(output / "evaluation").complete
    verify_replay(output / "evaluation")


def test_transport_failure_retains_started_attempt_and_unknown_native_calls(tmp_path, capsys):
    status, output = run_fixture(tmp_path, mode="error")
    assert status == 1
    assert not (output / "complete.json").exists()
    outcome = json.loads((output / "outcome.json").read_text())
    assert outcome["status"] == "completed-with-errors"
    assert outcome["terminal_statuses"] == {"submitted": 6, "error": 2}
    assert outcome["request_costs"]["transport_attempts_started"] == 2
    assert outcome["request_costs"]["native_provider_calls"] is None
    events = journal_events(output)
    assert sum(event["event"] == "attempt_start" for event in events) == 2
    assert sum(event["event"] == "transport_failure" for event in events) == 2
    for path in output.rglob("*"):
        if path.is_file():
            assert SECRET_MARKER.encode() not in path.read_bytes()
    captured = capsys.readouterr()
    assert SECRET_MARKER not in captured.out + captured.err
    assert read_bundle(output / "evaluation").complete


@pytest.mark.parametrize("kind", ["prompt", "journal"])
def test_completion_binds_immutable_sources_and_request_transcripts(tmp_path, kind):
    status, output = run_fixture(tmp_path)
    assert status == 0
    target = (
        output / "immutable/system-prompt.txt"
        if kind == "prompt"
        else next((output / "requests").glob("*.jsonl"))
    )
    target.write_bytes(target.read_bytes() + b"tampered\n")
    with pytest.raises(ValueError, match="hash or size changed"):
        pilot.verify_pilot(output)


def test_existing_empty_output_is_rejected_before_any_transport(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    def forbidden():
        raise AssertionError("existing output must fail before transport creation")

    with pytest.raises(FileExistsError):
        pilot.run_pilot(output, transport_factory=forbidden, model="fixture")
    assert list(output.iterdir()) == []


def test_interruption_preserves_partial_bundle_without_completion(tmp_path):
    class InterruptedTransport:
        def complete(self, *, model, messages):
            raise KeyboardInterrupt

    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        pilot.run_pilot(output, transport_factory=InterruptedTransport, model="fixture")
    assert (output / "pilot-manifest.json").is_file()
    assert (output / "interrupted.json").is_file()
    assert not (output / "complete.json").exists()
    assert not (output / "outcome.json").exists()
    partial = read_bundle(output / "evaluation", allow_incomplete=True)
    assert partial.completed == 3 and partial.interrupted_attempts == 1
    events = journal_events(output)
    assert sum(event["event"] == "attempt_start" for event in events) == 1
    assert not any(event["event"] == "response" for event in events)


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--model", "fixture"],
        ["--output", "ignored"],
        ["--model", "fixture", "--output", "ignored"],
    ],
)
def test_no_live_transport_or_model_is_selected_by_default(arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as failure:
        pilot.main(arguments)
    assert failure.value.code == 2
    assert not (tmp_path / "ignored").exists()


def test_http_requires_explicit_token_file_before_output_creation(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(SystemExit) as failure:
        pilot.main(
            ["--model", "fixture", "--base-url", "http://127.0.0.1:1", "--output", str(output)]
        )
    assert failure.value.code == 2
    assert not output.exists()


def test_source_module_cli_is_executable_without_network(tmp_path):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY / "src"), str(REPOSITORY)])
    result = subprocess.run(
        [sys.executable, "-m", "research.sound_matching.llm.pilot", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--command-json" in result.stdout
    assert "--token-file" in result.stdout
    assert "--max-model-calls" in result.stdout
