"""Exercise real audio episodes plus bounded failure and ownership boundaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from harpy.tasks.sound_match import (
    ActorSpec,
    EpisodeSpec,
    Propose,
    Submit,
    builtin_actor_spec,
    default_patch,
    evaluate,
    run,
    runner,
)
from harpy.tasks.sound_match.results import audio_hash


class ScriptedActor:
    def __init__(self, actions) -> None:
        self.actions = iter(actions)
        self.closed = False

    def decide(self, observation):
        action = next(self.actions)
        return action(observation) if callable(action) else action

    def close(self):
        self.closed = True


def scripted(actions, *, name="scripted"):
    return ActorSpec(name, lambda: ScriptedActor(actions))


def episode(**changes):
    return EpisodeSpec(
        "opaque-001",
        replace(default_patch(), notes_cents=(6100,)),
        **changes,
    )


def test_explicit_submission_and_monotonic_best_curve():
    spec = episode(render_budget=4, decision_budget=4)
    worse = replace(default_patch(), notes_cents=(5800,))
    result = run(spec, scripted([Propose(spec.target_patch, 0), Propose(worse, 0), Submit(0)]))
    record = result.records[0]
    assert record.status == "submitted"
    assert record.submitted_candidate.patch == worse
    assert record.metrics["best_distance"] == 0
    assert record.metrics["last_distance"] == record.metrics["submitted_distance"] > 0
    assert record.metrics["evaluations"] == 3
    assert record.metrics["decisions"] == 3
    assert record.metrics["model_calls"] == 0
    assert list(record.metrics["best_by_evaluation"].values())[1:] == [0, 0, 0]
    assert record.dsp_renders + record.cache_hits == 3


def test_truncation_does_not_invent_a_submission():
    spec = episode(decision_budget=1)
    record = run(spec, scripted([Propose(spec.target_patch)])).records[0]
    assert record.status == "truncated"
    assert record.last_candidate.distance == 0
    assert record.submitted_candidate is None
    assert record.metrics["submitted_distance"] is None
    assert record.metrics["model_calls"] is None


def test_invalid_actions_consume_decisions_without_rendering():
    spec = episode(editable_fields=("notes_cents",), render_budget=2, decision_budget=7)
    actions = [
        Propose({"notes_cents": [6100]}, 0),
        Propose(default_patch(3), 0),
        Propose(replace(default_patch(), gain_db=-7), 0),
        object(),
        Propose(spec.target_patch, 0),
        Propose(spec.initial_patch, 0),
        Submit(0),
    ]
    record = run(spec, scripted(actions)).records[0]
    assert [row["reason"] for row in record.ledger] == [
        "invalid_patch",
        "voice_count_changed",
        "locked_field_changed",
        "unsupported_action",
        None,
        "render_budget_exhausted",
        None,
    ]
    assert record.status == "submitted"
    assert record.metrics["evaluations"] == 2
    assert record.metrics["invalid_actions"] == 5
    assert record.metrics["decisions"] == 7
    assert record.metrics["model_calls"] is None  # Unsupported action declares no cost.


def test_cached_proposals_are_charged_and_cache_capacity_does_not_change_scores():
    spec = episode(render_budget=4, decision_budget=4)
    actions = [Propose(spec.initial_patch, 0), Propose(spec.target_patch, 0), Submit(0)]
    cached = run(spec, scripted(actions)).records[0]
    uncached = run(spec, scripted(actions), cache_max_audio_bytes=0).records[0]
    assert cached.metrics == uncached.metrics
    assert cached.metrics["evaluations"] == 3
    assert cached.cache_hits == 2
    assert cached.dsp_renders == 1
    assert uncached.cache_hits == 0
    assert uncached.dsp_renders == 3
    assert cached.initial.audio_sha256 == uncached.initial.audio_sha256
    assert cached.submitted_candidate.audio_sha256 == uncached.submitted_candidate.audio_sha256


def test_cache_is_bounded_and_uses_lru_eviction():
    size = runner.FRAME_COUNT * np.dtype(np.float32).itemsize
    cache = runner._WaveCache(size)
    a, b = default_patch(), replace(default_patch(), gain_db=-7)
    cache.get(a)
    cache.get(a)
    cache.get(b)
    cache.get(a)
    assert cache.bytes == size
    assert len(cache.items) == 1
    assert cache.renders == 3
    assert cache.hits == 1


def test_actor_receives_owned_audio_patch_and_public_state_only():
    spec = episode(render_budget=3, decision_budget=3)
    observed = []

    class MutatingActor:
        def decide(self, observation):
            assert not hasattr(observation, "target_patch")
            assert not hasattr(observation, "objective")
            assert not hasattr(observation, "pitch_error_cents")
            observed.append(observation)
            if len(observed) == 1:
                observation.target_audio[:] = 0
                observation.candidate_audio[:] = 0
                object.__setattr__(observation.patch, "gain_db", -18)
                return Propose(spec.initial_patch, 0)
            assert np.any(observation.target_audio)
            assert np.any(observation.candidate_audio)
            assert observation.patch.gain_db == spec.initial_patch.gain_db
            assert observation.last_action_status["status"] == "accepted"
            with pytest.raises(TypeError):
                observation.allowed_values["bounds"]["gain_db"] = (0, 1)
            return Submit(0)

    result = run(spec, ActorSpec("owned", MutatingActor))
    record = result.records[0]
    assert record.initial.distance == record.submitted_candidate.distance > 0
    assert record.target_audio_sha256 == audio_hash(runner.render(spec.target_patch))
    assert observed[0].target_audio is not observed[1].target_audio
    assert spec.initial_patch.gain_db == -6


def test_factory_is_fresh_for_each_episode_and_cleanup_runs():
    instances = []

    def factory():
        actor = ScriptedActor([Submit(0)])
        instances.append(actor)
        return actor

    result = evaluate(
        [episode(), replace(episode(), episode_id="opaque-002")],
        [ActorSpec("fresh", factory)],
    )
    assert len(instances) == 2
    assert instances[0] is not instances[1]
    assert all(actor.closed for actor in instances)
    assert all(record.status == "submitted" for record in result.records)


@pytest.mark.parametrize("failure", ["factory", "decide", "cleanup"])
def test_actor_failures_are_explicit_and_do_not_invent_submission(failure):
    class FailingActor:
        closed = False

        def decide(self, observation):
            if failure == "decide":
                raise RuntimeError("not serialized")
            return Submit(0)

        def close(self):
            self.closed = True
            if failure == "cleanup":
                raise RuntimeError("not serialized")

    actor = FailingActor()

    def factory():
        if failure == "factory":
            raise RuntimeError("not serialized")
        return actor

    record = run(episode(), ActorSpec("failure", factory)).records[0]
    assert record.status == "error"
    assert (
        record.reason
        == {
            "factory": "actor_factory_error",
            "decide": "actor_exception",
            "cleanup": "actor_cleanup_error",
        }[failure]
    )
    assert record.error_type == "RuntimeError"
    assert record.metrics["submitted_distance"] is None
    assert record.metrics["model_calls"] is None
    assert record.metrics["evaluations"] == 1
    assert actor.closed == (failure != "factory")


def test_cancellation_propagates_after_cleanup():
    class CancelledActor:
        closed = False

        def decide(self, observation):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    actor = CancelledActor()
    with pytest.raises(KeyboardInterrupt):
        run(episode(), ActorSpec("cancel", lambda: actor))
    assert actor.closed


@pytest.mark.parametrize("failure", ["render", "objective"])
def test_initial_evaluation_failure_is_charged_and_missing_scores_are_null(monkeypatch, failure):
    spec = episode()
    original_render = runner.render
    called = []

    def broken_render(patch):
        if patch == spec.initial_patch:
            raise RuntimeError("failure")
        return original_render(patch)

    def broken_score(*args):
        raise RuntimeError("failure")

    monkeypatch.setattr(
        runner,
        "render" if failure == "render" else "score",
        broken_render if failure == "render" else broken_score,
    )
    record = run(spec, ActorSpec("unused", lambda: called.append(1))).records[0]
    assert not called
    assert record.reason == f"initial_{failure}_error"
    assert record.initial is None
    assert not record.ledger
    assert record.metrics["evaluations"] == 1
    assert all(
        record.metrics[name] is None
        for name in ("initial_distance", "last_distance", "best_distance", "submitted_distance")
    )


@pytest.mark.parametrize("failure", ["render", "objective"])
def test_failed_proposal_is_charged_and_previous_candidate_is_retained(monkeypatch, failure):
    spec = episode()
    proposed = replace(spec.initial_patch, notes_cents=(6200,))
    original_render, original_score = runner.render, runner.score
    score_calls = 0

    def broken_render(patch):
        if patch == proposed:
            raise RuntimeError("failure")
        return original_render(patch)

    def broken_score(*args):
        nonlocal score_calls
        score_calls += 1
        if score_calls == 2:
            raise RuntimeError("failure")
        return original_score(*args)

    monkeypatch.setattr(
        runner,
        "render" if failure == "render" else "score",
        broken_render if failure == "render" else broken_score,
    )
    record = run(spec, scripted([Propose(proposed, 0)])).records[0]
    assert record.status == "error"
    assert record.reason == f"{failure}_error"
    assert record.metrics["evaluations"] == 2
    assert record.dsp_renders == 2
    assert record.metrics["last_distance"] == record.metrics["initial_distance"]
    assert record.metrics["submitted_distance"] is None
    assert record.ledger[-1]["evaluation"] == 2
    assert record.ledger[-1]["candidate"] is None


def test_target_setup_failure_aborts_before_creating_actor(monkeypatch):
    def fail(patch):
        raise RuntimeError("no audio")

    monkeypatch.setattr(runner, "render", fail)
    created = []
    with pytest.raises(RuntimeError, match="Target setup failed"):
        run(episode(), ActorSpec("unused", lambda: created.append(1)))
    assert not created


@pytest.mark.parametrize("name", ["fixed", "random-search", "pattern-search"])
def test_builtin_actors_submit_within_budget_and_do_not_regress_initial_score(name):
    spec = episode(render_budget=8, decision_budget=10)
    record = run(spec, builtin_actor_spec(name, seed=17)).records[0]
    assert record.status == "submitted"
    assert record.metrics["evaluations"] <= 8
    assert record.metrics["decisions"] <= 10
    assert record.metrics["invalid_actions"] == 0
    assert record.metrics["model_calls"] == 0
    assert record.submitted_candidate.distance <= record.initial.distance
    assert record.submitted_candidate.distance == record.metrics["best_distance"]
    if name == "pattern-search":
        assert record.submitted_candidate.distance == 0


def test_random_actor_episode_stream_is_order_independent():
    first = episode(render_budget=5, decision_budget=6)
    second = replace(first, episode_id="opaque-002")
    actor = builtin_actor_spec("random-search", seed=17)
    forward = evaluate([first, second], [actor])
    reverse = evaluate([second, first], [actor])
    for left, right in zip(forward.records, reversed(reverse.records), strict=True):
        assert left.ledger == right.ledger
        assert left.metrics == right.metrics


def test_artifact_change_during_episode_rejects_result(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"before")

    class ModifyingActor:
        def decide(self, observation):
            artifact.write_bytes(b"after")
            return Submit(0)

    actor = ActorSpec("changing", ModifyingActor, artifact_paths=(artifact,))
    with pytest.raises(RuntimeError, match="Actor artifacts changed"):
        run(episode(), actor)


def test_episode_rejects_unreachable_locked_targets():
    with pytest.raises(ValueError, match="locked field notes_cents"):
        episode(editable_fields=("gain_db",))
    with pytest.raises(ValueError, match="same note count"):
        episode(initial_patch=default_patch(3))
