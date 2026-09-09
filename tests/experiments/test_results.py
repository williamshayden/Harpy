"""Saved experiments bind paired membership and rederive every aggregate."""

import csv
import io
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS
from harpy.experiments import (
    EpisodeSpec,
    ExperimentResult,
    evaluate,
    load_result,
    save_result,
    spectrum_peak_actor_spec,
    summarize,
)


@pytest.fixture
def result(monkeypatch):
    import harpy.experiments.runner as runner

    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": "a" * 64})
    return evaluate(
        (
            EpisodeSpec("a", 6_000, 12, partition="iid"),
            EpisodeSpec("b", 6_001, 12, partition="lower"),
        ),
        (spectrum_peak_actor_spec(),),
        conditions=(CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[1]),
        trace=True,
    )


def test_roundtrip_create_only_and_partition_rows(result, tmp_path):
    path = tmp_path / "result.json"
    save_result(result, path)
    restored = load_result(path)
    assert restored == result
    assert [row["partition"] for row in result.rows] == ["iid", "lower"] * 2
    assert [row["episodes"] for row in result.rows] == [1] * 4
    assert len(result.paired_clean) == 2
    with pytest.raises(FileExistsError):
        save_result(result, path)
    assert load_result(path) == result


@pytest.mark.parametrize(
    "change",
    [
        lambda doc: doc["rows"][0].update(submitted_success_rate=0),
        lambda doc: doc["paired_clean"][0].update(regressed=1),
        lambda doc: doc["records"].reverse(),
        lambda doc: doc["records"][0]["terminal"].update(final_absolute_error_cents=4),
        lambda doc: doc["records"][0].update(inference_counts=[]),
        lambda doc: doc["records"][0]["estimates"][0].update(step=2),
        lambda doc: doc["records"][0]["trace"][0].update(controls=[0, 0, 99]),
        lambda doc: doc["actors"][0].update(observation_mode="unknown"),
        lambda doc: doc.update(extra=True),
    ],
)
def test_rejects_inconsistent_records_and_claimed_metrics(result, change):
    document = result.to_document()
    change(document)
    with pytest.raises(ValueError):
        ExperimentResult.from_document(document)


def test_protocol_requires_exact_episode_membership(result):
    protocol = {"id": "custom", "episodes": [episode.to_document() for episode in result.episodes]}
    matched = replace(result, protocol=protocol)
    assert matched.protocol["id"] == "custom"
    protocol["episodes"].reverse()
    with pytest.raises(ValueError, match="membership"):
        replace(result, protocol=protocol)


def test_metadata_is_owned_and_deeply_immutable(result):
    doc = result.to_document()
    restored = ExperimentResult.from_document(doc)
    doc["actors"][0]["artifact_provenance"]["injected"] = True
    assert "injected" not in restored.actors[0]["artifact_provenance"]
    with pytest.raises(TypeError):
        restored.actors[0]["name"] = "changed"


@pytest.mark.parametrize("format", ["text", "markdown", "csv"])
def test_readout_explains_units_and_does_not_pool_partitions(result, format):
    text = summarize(result, format=format)
    assert (
        "initial_perception.mean_absolute_error_cents"
        if format == "csv"
        else "Initial pitch MAE (c)"
    ) in text
    assert "iid" in text and "lower" in text
    assert ("inferences" if format == "csv" else "Inferences") in text


def test_duplicate_and_nonfinite_json_are_rejected(result, tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"schema_id":"one","schema_id":"two"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_result(path)
    document = result.to_document()
    document["records"][0]["estimates"][0]["estimated_candidate_cents"] = float("nan")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="nonfinite"):
        load_result(path)


def test_saved_result_readout_is_training_dependency_free(result, tmp_path):
    path = tmp_path / "result.json"
    save_result(result, path)
    code = (
        "import sys; from pathlib import Path; "
        "from harpy.experiments import load_result, summarize; "
        "print(summarize(load_result(Path(sys.argv[1])))); assert 'torch' not in sys.modules; "
        "assert 'stable_baselines3' not in sys.modules"
    )
    process = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True)
    assert process.returncode == 0, process.stderr.decode()


@pytest.mark.parametrize(
    "change",
    [
        lambda doc: doc["task"].update(success_tolerance_cents=10),
        lambda doc: doc["records"][0]["initial_evidence"].update(nuisance_seed=999),
        lambda doc: doc["records"][0]["initial_evidence"].update(dry_rms=0),
        lambda doc: doc["records"][0]["trace"][0]["evidence"].update(condition_id="noise-10db"),
        lambda doc: doc["records"][0]["trace"][0].update(controls=[False, 0, 0]),
    ],
)
def test_task_and_view_identity_cannot_drift(result, change):
    document = result.to_document()
    change(document)
    with pytest.raises(ValueError):
        ExperimentResult.from_document(document)


def test_trace_rewards_cannot_be_redistributed_without_changing_total(monkeypatch):
    import harpy.experiments.runner as runner

    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": "a" * 64})
    result = evaluate((EpisodeSpec("moves", 6_006, 12),), (spectrum_peak_actor_spec(),), trace=True)
    document = result.to_document()
    document["records"][0]["trace"][0]["reward"] += 10
    document["records"][0]["trace"][1]["reward"] -= 10
    with pytest.raises(ValueError, match=r"reward.*dynamics"):
        ExperimentResult.from_document(document)


def test_tail_metrics_counts_and_paired_changes_are_derived(result):
    row = result.rows[0]
    assert row["submitted_successes"] == 1
    assert row["invalid_actions"] == row["truncations"] == 0
    assert row["final_p99_absolute_error_cents"] == row["final_max_absolute_error_cents"] == 0
    assert row["initial_perception"]["p99_absolute_error_cents"] == 0
    assert all(row["mean_change_final_error_cents"] == 0 for row in result.paired_clean)


def test_csv_contains_all_scalar_metrics_and_paired_deltas(result):
    rows = list(csv.DictReader(io.StringIO(summarize(result, format="csv"))))
    assert rows[0]["device"] == "cpu"
    assert "seed" in rows[0] and rows[0]["truncations"] == "0"
    assert rows[0]["initial_perception.p99_absolute_error_cents"] == "0.0"
    assert "trajectory_perception.max_absolute_error_cents" in rows[0]
    assert rows[-1]["paired_clean.mean_change_final_error_cents"] == "0.0"


@pytest.mark.parametrize("estimate_steps", [(1, 3, 4, 5), (3, 4, 5), ()])
def test_perception_metrics_follow_sparse_estimates_and_blocked_controls(
    monkeypatch, estimate_steps
):
    import harpy.experiments.runner as runner
    from harpy.envs.models import ObservationMode, PitchAction
    from harpy.experiments import ActorSpec, Decision

    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": "a" * 64})
    actions = (
        PitchAction.OCTAVE_UP,
        PitchAction.OCTAVE_UP,
        PitchAction.OCTAVE_UP,  # Already at the bound; the candidate stays at 8400.
        PitchAction.CENT_DOWN,
        PitchAction.SUBMIT,
    )
    estimates = {1: 6001, 3: 8397, 4: 8402, 5: 8399}

    class Actor:
        def __init__(self):
            self.step = 0

        def decide(self, observation):
            self.step += 1
            estimate = estimates[self.step] if self.step in estimate_steps else None
            return Decision(actions[self.step - 1], estimate, int(estimate is not None))

    actor = ActorSpec("sparse", Actor, ObservationMode.SPECTRUM, "test", "test", "test")
    result = evaluate((EpisodeSpec("blocked", 6000, 12),), (actor,))
    row = result.rows[0]
    initial = row["initial_perception"]
    trajectory = row["trajectory_perception"]
    assert row["invalid_actions"] == 1
    assert initial["estimates"] == int(1 in estimate_steps)
    assert initial["mean_absolute_error_cents"] == (1.0 if 1 in estimate_steps else None)
    errors = {1: 1.0, 3: 3.0, 4: 2.0, 5: 0.0}
    expected = [errors[step] for step in estimate_steps]
    assert trajectory["estimates"] == len(expected)
    assert trajectory["mean_absolute_error_cents"] == (
        sum(expected) / len(expected) if expected else None
    )
    assert trajectory["max_absolute_error_cents"] == (max(expected) if expected else None)
    assert trajectory["within_1_cents_rate"] == (
        sum(error <= 1 for error in expected) / len(expected) if expected else None
    )


def test_reserved_smoke_protocol_rejects_substituted_membership(result):
    from harpy.experiments.protocols import SMOKE_PROTOCOL_ID

    forged = {
        "id": SMOKE_PROTOCOL_ID,
        "episodes": [episode.to_document() for episode in result.episodes],
    }
    with pytest.raises(ValueError, match="frozen manifest"):
        replace(result, protocol=forged)


@pytest.mark.parametrize(
    "field, value",
    [
        ("source", None),
        ("source", {}),
        ("source", {"package_sha256": "invalid"}),
        ("source", {"status": "unknown", "reason": " "}),
        ("source", {"status": "maybe", "package_sha256": "a" * 64}),
        ("source", {"package_sha256": "a" * 64, "identity": "unverified"}),
        ("runtime", None),
        ("runtime", {"elapsed_wall_time_seconds": -1}),
        ("runtime", {"elapsed_wall_time_seconds": True}),
        ("runtime", {"evaluation_devices": None}),
        ("runtime", {"evaluation_devices": {"spectrum-peak": "cuda"}}),
        ("runtime", {"evaluation_devices": {}}),
        ("runtime", {"python": 312}),
        ("runtime", {"torch": False}),
    ],
)
def test_result_rejects_malformed_provenance_claims(result, field, value):
    with pytest.raises(ValueError):
        replace(result, provenance={**dict(result.provenance), field: value})


def test_result_preserves_unknown_source_and_runtime_extensions(result):
    source = {"status": "unknown", "reason": "External producer did not capture source."}
    runtime = {
        **dict(result.provenance["runtime"]),
        "elapsed_wall_time_seconds": 0.0,
        "worker_count": 4,
        "condition_wall_seconds": {"clean": 1.2},
        "torch": None,
    }
    updated = replace(
        result, provenance={**dict(result.provenance), "source": source, "runtime": runtime}
    )
    restored = ExperimentResult.from_document(updated.to_document())
    assert restored.provenance["source"] == source
    assert restored.provenance["runtime"] == runtime
