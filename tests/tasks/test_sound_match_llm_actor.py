"""Offline actor contracts, real runner metering, and public-only text features."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from harpy.tasks.sound_match import EpisodeSpec, evaluate, run
from harpy.tasks.sound_match.models import Observation, Propose, Submit
from harpy.tasks.sound_match.patch import PARAMETER_BOUNDS, WAVEFORMS, default_patch

_SOURCE_ROOT = str(Path(__file__).parents[2])
sys.path.insert(0, _SOURCE_ROOT)
try:
    _actor = importlib.import_module("research.sound_matching.llm.actor")
    _features = importlib.import_module("research.sound_matching.llm.features")
finally:
    sys.path.remove(_SOURCE_ROOT)

HISTORY_TURNS = _actor.HISTORY_TURNS
MAX_RESPONSE_BYTES = _actor.MAX_RESPONSE_BYTES
LLMActor = _actor.LLMActor
ResponseError = _actor.ResponseError
llm_actor_spec = _actor.llm_actor_spec
parse_action = _actor.parse_action
ENVELOPE_BINS = _features.ENVELOPE_BINS
PEAK_COUNT = _features.PEAK_COUNT
public_observation = _features.public_observation
waveform_features = _features.waveform_features


class ScriptedTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, *, model, messages):
        self.requests.append(json.loads(json.dumps({"model": model, "messages": messages})))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {
            "content": response,
            "usage": {"total_tokens": 4, "authorization": "NEVER-LOG-THIS"},
            "response_id": "offline-response",
            "response_model": model,
        }


def response_for(patch):
    return json.dumps({"action": "propose", "patch": patch.to_dict()})


def observation():
    target = np.zeros(96_000, dtype=np.float32)
    target[:48_000] = 0.1 * np.sin(np.arange(48_000) * (2 * np.pi * 440 / 48_000))
    return Observation(
        episode_id="PRIVATE-EPISODE-IDENTITY",
        target_audio=target,
        candidate_audio=np.zeros_like(target),
        patch=default_patch(),
        editable_fields=("notes_cents",),
        render_budget_remaining=31,
        decision_budget_remaining=64,
        last_action_status={"status": "initial", "reason": None, "hidden": "PRIVATE-FEEDBACK"},
        allowed_values={
            "bounds": PARAMETER_BOUNDS,
            "waveforms": WAVEFORMS,
            "voice_count": 1,
            "hidden": "PRIVATE-MANIFEST",
        },
    )


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_whole_clip_features_keep_early_tone_and_silent_tail_without_rendering(monkeypatch):
    def forbidden_render(*_args, **_kwargs):
        raise AssertionError("feature extraction must not synthesize candidates")

    monkeypatch.setattr("harpy.tasks.sound_match.renderer.render", forbidden_render)
    public = observation()
    before = public.target_audio.copy()
    result = waveform_features(public.target_audio, public.sample_rate_hz)
    assert result["spectral_peaks"][0]["frequency_hz"] == 440
    assert len(result["spectral_peaks"]) <= PEAK_COUNT
    assert len(result["rms_envelope"]) == ENVELOPE_BINS
    assert any(value > 0 for value in result["rms_envelope"][:32])
    assert result["rms_envelope"][32:] == [0] * 32
    assert waveform_features(np.zeros_like(public.target_audio), 48_000)["spectral_peaks"] == []
    np.testing.assert_array_equal(before, public.target_audio)
    json.dumps(result, allow_nan=False)


def test_model_request_contains_only_declared_public_fields_and_no_oracle_identity(tmp_path):
    transport = ScriptedTransport(['{"action":"submit"}'])
    journal = tmp_path / "requests.jsonl"
    actor = LLMActor(transport, model="offline-model", journal_path=journal)
    action = actor.decide(observation())
    actor.close()
    assert isinstance(action, Submit) and action.model_calls == 1
    request = transport.requests[0]
    assert set(request) == {"model", "messages"}
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    serialized = json.dumps(request)
    for forbidden in (
        "PRIVATE-EPISODE-IDENTITY",
        "PRIVATE-MANIFEST",
        "PRIVATE-FEEDBACK",
        "episode_id",
        "target_patch",
        "pitch_error_cents",
        "sha256",
        "artifact_paths",
    ):
        assert forbidden not in serialized
    payload = json.loads(request["messages"][-1]["content"])
    assert payload["current_patch"] == default_patch().to_dict()
    assert payload["budgets"]["model_calls_remaining"] == 4
    assert payload["candidate_objective"]["distance"] > 0
    assert set(payload["last_feedback"]) == {"status", "reason"}
    assert "NEVER-LOG-THIS" not in journal.read_text()
    assert [row["event"] for row in events(journal)] == [
        "controller_start",
        "episode_binding",
        "attempt_start",
        "response",
        "model_decision",
        "controller_close",
    ]


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"action":"submit"}\n```',
        '{"action":"submit","action":"propose"}',
        '{"action":"submit","confidence":NaN}',
        '{"action":"submit","confidence":1e999}',
        '{"action":"submit","extra":true}',
        '{"action":"submit"} trailing explanation',
        '[{"action":"submit"}]',
        '{"action":"propose","patch":{"notes_cents":[6000]}}',
        response_for(default_patch()).replace('"attack_ms": 10', '"attack_ms": true'),
        response_for(default_patch()).replace('"attack_ms": 10', '"attack_ms": 10.0'),
        response_for(default_patch()).replace('"attack_ms": 10', '"attack_ms": 0'),
        "x" * (MAX_RESPONSE_BYTES + 1),
        None,
    ],
)
def test_response_parser_rejects_malformed_or_repaired_actions(content):
    with pytest.raises(ResponseError):
        parse_action(content)


def test_response_parser_accepts_full_patch_and_exact_submit():
    action = parse_action(response_for(default_patch()))
    assert isinstance(action, Propose)
    assert action.patch == default_patch() and action.model_calls == 1
    assert parse_action(' {"action":"submit"}\n') == Submit(model_calls=1)


def test_invalid_response_is_one_counted_invalid_decision_without_a_render(tmp_path):
    transport = ScriptedTransport(["not-json", '{"action":"submit"}'])
    spec = llm_actor_spec(
        transport_factory=lambda: transport,
        model="offline-model",
        max_calls=2,
        journal_directory=tmp_path,
    )
    record = run(EpisodeSpec("opaque", default_patch()), spec).records[0]
    assert record.status == "submitted"
    assert record.metrics["model_calls"] == 2
    assert record.metrics["invalid_actions"] == 1
    assert record.metrics["evaluations"] == 1
    assert record.ledger[0]["reason"] == "invalid_patch"
    assert record.ledger[0]["model_calls"] == 1
    assert len(transport.requests) == 2
    assert json.loads(transport.requests[1]["messages"][-1]["content"])["last_feedback"] == {
        "status": "invalid",
        "reason": "invalid_patch",
    }
    rows = events(next(tmp_path.glob("*.jsonl")))
    assert sum(row["event"] == "attempt_start" for row in rows) == 2
    assert sum(row["event"] == "parse_rejection" for row in rows) == 1


def test_transport_failure_keeps_attempt_journal_and_native_cost_unknown(tmp_path):
    transport = ScriptedTransport([TimeoutError("credential=DO-NOT-LOG-SECRET")])
    spec = llm_actor_spec(
        transport_factory=lambda: transport,
        model="offline-model",
        journal_directory=tmp_path,
    )
    record = run(EpisodeSpec("opaque", default_patch()), spec).records[0]
    assert record.status == "error" and record.reason == "actor_exception"
    assert record.metrics["model_calls"] is None
    assert record.submitted_candidate is None
    assert len(transport.requests) == 1
    journal = next(tmp_path.glob("*.jsonl"))
    assert "DO-NOT-LOG-SECRET" not in journal.read_text()
    rows = events(journal)
    assert [row["event"] for row in rows] == [
        "controller_start",
        "episode_binding",
        "attempt_start",
        "transport_failure",
        "controller_close",
    ]
    assert rows[3]["provider_execution"] == "unknown"


def test_call_cap_restores_best_heard_patch_with_metered_zero_call_actions(tmp_path):
    worse = replace(default_patch(), notes_cents=(6100,))
    transport = ScriptedTransport([response_for(worse)])
    spec = llm_actor_spec(
        transport_factory=lambda: transport,
        model="offline-model",
        max_calls=1,
        journal_directory=tmp_path,
    )
    record = run(EpisodeSpec("opaque", default_patch()), spec).records[0]
    assert record.status == "submitted"
    assert record.submitted_candidate.patch == default_patch()
    assert record.metrics["evaluations"] == 3
    assert record.metrics["model_calls"] == 1
    assert [(row["action"], row["model_calls"]) for row in record.ledger] == [
        ("propose", 1),
        ("propose", 0),
        ("submit", 0),
    ]
    assert len(transport.requests) == 1
    guards = [
        row for row in events(next(tmp_path.glob("*.jsonl"))) if row["event"] == "controller_guard"
    ]
    assert [row["action"] for row in guards] == ["restore_best_heard", "submit"]
    assert all(
        row["origin"] == "controller" and row["reason"] == "model_call_cap" for row in guards
    )


def test_model_submit_is_not_replaced_with_controller_best_patch(tmp_path):
    worse = replace(default_patch(), notes_cents=(6100,))
    transport = ScriptedTransport([response_for(worse), '{"action":"submit"}'])
    spec = llm_actor_spec(
        transport_factory=lambda: transport,
        model="offline-model",
        max_calls=2,
        journal_directory=tmp_path,
    )
    record = run(EpisodeSpec("opaque", default_patch()), spec).records[0]
    assert record.submitted_candidate.patch == worse
    assert record.metrics["model_calls"] == 2
    assert record.metrics["evaluations"] == 2
    assert not any(
        row["event"] == "controller_guard" for row in events(next(tmp_path.glob("*.jsonl")))
    )


@pytest.mark.parametrize("changes", [{"render_budget": 1}, {"decision_budget": 2}])
def test_budget_guard_is_explicit_and_makes_no_model_request(changes):
    transport = ScriptedTransport([])
    spec = llm_actor_spec(transport_factory=lambda: transport, model="offline-model")
    record = run(EpisodeSpec("opaque", default_patch(), **changes), spec).records[0]
    assert record.status == "submitted"
    assert record.metrics["model_calls"] == 0
    assert record.metrics["evaluations"] == 1
    assert transport.requests == []


def test_factories_isolate_transports_history_and_journal_identity(tmp_path):
    transports = []

    def factory():
        transport = ScriptedTransport(['{"action":"submit"}'])
        transports.append(transport)
        return transport

    spec = llm_actor_spec(
        transport_factory=factory, model="offline-model", journal_directory=tmp_path
    )
    result = evaluate(
        (EpisodeSpec("first", default_patch()), EpisodeSpec("second", default_patch())), (spec,)
    )
    assert [record.metrics["model_calls"] for record in result.records] == [1, 1]
    assert len(transports) == 2
    assert all(len(transport.requests[0]["messages"]) == 2 for transport in transports)
    journals = sorted(tmp_path.glob("*.jsonl"))
    assert len(journals) == 2
    assert len({events(path)[0]["controller_id"] for path in journals}) == 2
    assert {
        row["episode_id"]
        for path in journals
        for row in events(path)
        if row["event"] == "episode_binding"
    } == {"first", "second"}
    assert not set(journals).intersection(spec.artifact_paths)


def test_history_is_bounded_and_response_truncation_is_declared(tmp_path):
    transport = ScriptedTransport(["x" * (MAX_RESPONSE_BYTES + 10)] * 6)
    journal = tmp_path / "history.jsonl"
    actor = LLMActor(transport, model="offline-model", max_calls=6, journal_path=journal)
    public = observation()
    for _ in range(6):
        action = actor.decide(public)
        assert isinstance(action, Propose) and action.patch == {} and action.model_calls == 1
    assert actor.decide(public) == Submit(model_calls=0)
    actor.close()
    assert max(len(request["messages"]) for request in transport.requests) == 2 + 2 * HISTORY_TURNS
    assert all(
        len(message["content"].encode()) <= MAX_RESPONSE_BYTES
        for request in transport.requests
        for message in request["messages"]
        if message["role"] == "assistant"
    )
    assert all(row["content_truncated"] for row in events(journal) if row["event"] == "response")


def test_public_payload_ignores_identity_change_with_identical_audio():
    first = observation()
    second = replace(first, episode_id="OTHER-PRIVATE-IDENTITY")
    assert public_observation(first) == public_observation(second)


@pytest.mark.parametrize("max_calls", [0, 65, True, 1.5])
def test_invalid_call_cap_is_rejected(max_calls):
    with pytest.raises(ValueError, match="max_calls"):
        LLMActor(ScriptedTransport([]), model="offline-model", max_calls=max_calls)
    with pytest.raises(ValueError, match="max_calls"):
        llm_actor_spec(
            transport_factory=lambda: ScriptedTransport([]),
            model="offline-model",
            max_calls=max_calls,
        )


def test_local_binding_prevents_controller_reuse_without_leaking_model_identity(tmp_path):
    transport = ScriptedTransport(['{"action":"submit"}'])
    journal = tmp_path / "identity.jsonl"
    actor = LLMActor(transport, model="offline-model", journal_path=journal)
    actor.decide(observation())
    with pytest.raises(RuntimeError, match="different episode"):
        actor.decide(replace(observation(), episode_id="different-private-case"))
    actor.close()
    assert len(transport.requests) == 1
    binding = [row for row in events(journal) if row["event"] == "episode_binding"]
    assert binding[0]["episode_id"] == "PRIVATE-EPISODE-IDENTITY"
    assert binding[0]["sent_to_model"] is False
    assert "PRIVATE-EPISODE-IDENTITY" not in json.dumps(transport.requests)


def test_journal_is_create_only_and_close_is_idempotent(tmp_path):
    journal = tmp_path / "existing.jsonl"
    actor = LLMActor(ScriptedTransport([]), model="offline-model", journal_path=journal)
    actor.close()
    actor.close()
    assert len(events(journal)) == 2
    with pytest.raises(FileExistsError):
        LLMActor(ScriptedTransport([]), model="offline-model", journal_path=journal)
