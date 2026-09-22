"""Opt-in v1 durability preserves legacy scientific semantics and cleanup."""

from __future__ import annotations

from dataclasses import replace

import pytest

from harpy.envs.models import ObservationMode, PitchAction
from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS
from harpy.experiments import (
    ActorSpec,
    EpisodeSpec,
    durable,
    evaluate,
    runner,
    spectrum_peak_actor_spec,
)


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    snapshot = durable.source_identity()
    monkeypatch.setattr(durable, "source_identity", lambda: snapshot)
    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": snapshot["sha256"]})


def episodes():
    return [EpisodeSpec("one", 6064, 12, 7), EpisodeSpec("two", 5971, 12, 8)]


def test_durable_v1_records_equal_legacy_results_and_replay_traces(tmp_path):
    actors = [spectrum_peak_actor_spec()]
    conditions = [CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[5]]
    legacy = evaluate(episodes(), actors, conditions=conditions, trace=True)
    output = tmp_path / "durable"
    result = durable.evaluate_bundle(episodes(), actors, output, conditions=conditions, trace=True)
    assert result.complete and result.completed == 4 and result.interrupted_attempts == 0
    assert [row["record"] for row in durable.iter_records(output)] == [
        record.to_document() for record in legacy.records
    ]
    durable.verify_replay(output)
    for fmt in ("text", "markdown", "csv"):
        assert durable.summarize_bundle(output, fmt)
    with pytest.raises(FileExistsError):
        durable.evaluate_bundle(episodes(), actors, output)


def test_v1_resume_retains_prefix_and_skips_completed_members(tmp_path):
    output = tmp_path / "bundle"
    actors = [spectrum_peak_actor_spec()]

    def stop(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        durable.evaluate_bundle(episodes(), actors, output, progress=stop)
    with pytest.raises(ValueError, match="incomplete"):
        durable.read_bundle(output)
    before = next(durable.iter_records(output, allow_incomplete=True))
    events = []
    result = durable.evaluate_bundle(
        episodes(), actors, output, resume=True, progress=events.append
    )
    assert result.complete and result.attempts == 2
    assert [event["key"] for event in events] == [["clean", "two", actors[0].name]]
    assert next(durable.iter_records(output)) == before


@pytest.mark.parametrize("failure", ["action", "cleanup", "interrupt", "setup"])
def test_v1_closes_actor_and_environment_on_failures_and_cancellation(
    tmp_path, monkeypatch, failure
):
    closed = []
    original = runner.make_robustness_env

    def make(*args, **kwargs):
        if failure == "setup":
            raise RuntimeError("setup failed")
        env = original(*args, **kwargs)
        env.close = lambda: closed.append("environment")
        return env

    monkeypatch.setattr(runner, "make_robustness_env", make)

    class Actor:
        def act(self, observation):
            if failure == "action":
                raise RuntimeError("action failed")
            if failure == "interrupt":
                raise KeyboardInterrupt
            return PitchAction.SUBMIT

        def close(self):
            closed.append("actor")
            if failure in {"interrupt", "cleanup"}:
                raise RuntimeError("cleanup failed")

    actor = ActorSpec("failure", Actor, ObservationMode.SPECTRUM, "none", "none", "test")
    output = tmp_path / "bundle"
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            durable.evaluate_bundle(episodes()[:1], [actor], output)
        result = durable.read_bundle(output, allow_incomplete=True)
        assert result.completed == 0 and result.interrupted_attempts == 1
    else:
        result = durable.evaluate_bundle(episodes()[:1], [actor], output)
        assert result.complete and result.metrics["groups"][0]["error"] == 1
        assert result.metrics["groups"][0]["unknown_rollout_costs"] == 1
        assert next(durable.iter_records(output))["error"]["costs"] == "unknown"
        # Terminal errors are scientifically completed; resuming does not retry.
        assert durable.evaluate_bundle(episodes()[:1], [actor], output, resume=True).attempts == 1
    assert closed == (["actor"] if failure == "setup" else ["environment", "actor"])


@pytest.mark.parametrize("change", ["episode", "actor", "trace", "cache", "weights"])
def test_v1_resume_rejects_changed_scientific_inputs(tmp_path, change):
    output = tmp_path / "bundle"
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"before")
    actor = replace(spectrum_peak_actor_spec(), artifact_paths=(weights,))
    membership = episodes()[:1]
    durable.evaluate_bundle(membership, [actor], output)
    kwargs = {}
    if change == "episode":
        membership = [replace(membership[0], source_pitch_cents=6000)]
    elif change == "actor":
        actor = replace(actor, name="renamed")
    elif change == "weights":
        weights.write_bytes(b"after")
    else:
        kwargs["trace" if change == "trace" else "cache_max_bytes"] = (
            True if change == "trace" else 0
        )
    with pytest.raises(ValueError, match="resume identity"):
        durable.evaluate_bundle(membership, [actor], output, resume=True, **kwargs)
