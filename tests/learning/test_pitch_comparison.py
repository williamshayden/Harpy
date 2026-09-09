"""Paired decoder identity, terminal outcomes, and lightweight report contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import harpy.learning.pitch_comparison as comparison
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import TARGET_MIN_COORDINATE, ObservationMode, TerminalReason
from harpy.envs.planning import minimum_action_plan
from harpy.learning.artifacts import FileRecord, PackageSourceStatus, RuntimeStatus, SourceStatus
from harpy.learning.errors import LearningContractError
from harpy.learning.evaluation import TerminalEpisodeRecord
from harpy.learning.experiment_results import ArtifactProvenance
from harpy.learning.models import ENVIRONMENT_ID, DeviceName, EpisodeSpec, ProfileName
from harpy.learning.pitch_actor import PitchDecoding
from harpy.learning.pitch_comparison import (
    PitchDecoderComparison,
    compare_pitch_decoders,
    comparison_bytes,
    comparison_from_bytes,
)
from harpy.learning.pitch_data import (
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
)
from harpy.learning.pitch_evaluation import build_pitch_evaluation_row
from harpy.learning.pitch_reports import PITCH_ESTIMATOR_PARAMETER_COUNT


def _source():
    return SourceStatus("1" * 40, False, hashlib.sha256(b"").hexdigest(), "d" * 64, True)


def _runtime():
    return RuntimeStatus(
        "3.12", "Linux", "", "2.0", "1.0", "2.0", "2.0", DeviceName.CPU, "CPU", None, None
    )


def _inventory():
    files = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "__init__.py",
            "learning/pitch_actor.py",
            "learning/pitch_comparison.py",
        )
    }
    digest = hashlib.sha256(
        json.dumps(sorted(files.items()), separators=(",", ":")).encode()
    ).hexdigest()
    return PackageSourceStatus("harpy-audio", None, digest), files


def _provenance(profile):
    return ArtifactProvenance(
        2, "a" * 64, "b" * 64, "pitch", 7, profile, DeviceName.CPU, DeviceName.CPU, _source()
    )


def _records(suite, decoding=None):
    records = []
    for index, episode in enumerate(suite.episodes):
        success = index == (1 if decoding is PitchDecoding.FEASIBLE_ARGMAX else 0)
        records.append(
            TerminalEpisodeRecord(
                episode_index=index,
                episode=EpisodeSpec(episode.target_note_index, episode.source_pitch_cents),
                submitted_success=success,
                within_5_cents=success,
                within_1_cent=success,
                final_absolute_error_cents=0
                if success
                else (200 if decoding is PitchDecoding.FEASIBLE_ARGMAX and index == 2 else 100),
                action_count=(
                    len(
                        minimum_action_plan(
                            episode.source_pitch_cents
                            - 100 * (TARGET_MIN_COORDINATE + episode.target_note_index)
                        )
                    )
                    if success
                    else 1
                ),
                excess_actions=0 if success else None,
                invalid_action_count=0,
                total_return=1.0 if success else -1.0,
                terminal_reason=TerminalReason.SUBMITTED_SUCCESS
                if success
                else TerminalReason.SUBMITTED_FAILURE,
            )
        )
    return tuple(records)


def _row(suite, profile, decoding=None):
    learned = decoding is not None
    return build_pitch_evaluation_row(
        actor_id=f"pitch-{decoding.value}-7" if learned else BaselineKind.SPECTRUM_PEAK.value,
        trainer=PitchTrainerKind.PITCH if learned else None,
        seed=7 if learned else None,
        environment_id=ENVIRONMENT_ID,
        observation_mode=ObservationMode.SPECTRUM,
        suite=suite,
        records=_records(suite, decoding),
        parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT if learned else None,
        training_examples=(256 if profile is ProfileName.SMOKE else 1400) if learned else None,
        training_wall_time_seconds=1.5 if learned else None,
    )


def _report(profile=ProfileName.SMOKE):
    ids = (
        (PitchEvaluationSuiteId.SMOKE,)
        if profile is ProfileName.SMOKE
        else (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        )
    )
    suites = tuple(fixed_pitch_evaluation_suite(item) for item in ids)
    snapshot, files = _inventory()
    return PitchDecoderComparison(
        _provenance(profile),
        _source(),
        snapshot,
        files,
        _runtime(),
        tuple(_row(suite, profile, PitchDecoding.GLOBAL_ARGMAX) for suite in suites),
        tuple(_row(suite, profile, PitchDecoding.FEASIBLE_ARGMAX) for suite in suites),
        tuple(_row(suite, profile) for suite in suites),
    )


@pytest.mark.parametrize("profile", list(ProfileName))
def test_roundtrip_preserves_explicit_identity_and_derives_pairs(profile):
    report = _report(profile)
    restored = comparison_from_bytes(comparison_bytes(report))
    assert restored == report
    assert comparison_bytes(restored) == comparison_bytes(report)
    for count, row in zip(restored.paired_counts, report.global_rows, strict=True):
        assert count.suite_id is row.suite_id
        assert count.episodes == len(row.episodes)
        assert (count.rescued, count.regressed, count.changed_outcome) == (1, 1, 3)
    document = report.to_document()
    assert document["criterion"] == {
        "eligible": False,
        "criterion_met": None,
        "status": "ineligible",
        "reason": "paired_decoder_intervention",
    }
    assert (
        document["actor_semantics"]["global-argmax"]
        != document["actor_semantics"]["feasible-argmax"]
    )
    assert "paired_counts" not in document
    with pytest.raises(TypeError):
        report.evaluator_files["extra.py"] = "a" * 64


@pytest.mark.parametrize(
    "change",
    [
        lambda doc: doc.update(extra=True),
        lambda doc: doc.pop("runtime"),
        lambda doc: doc.update(schema_id="unknown"),
        lambda doc: doc["criterion"].update(eligible=True),
        lambda doc: doc["criterion"].update(eligible=0),
        lambda doc: doc["actor_semantics"].update(
            {"feasible-argmax": "harpy-sine-pitch-estimator-planner-v1"}
        ),
        lambda doc: doc["evaluator_files"].update({"learning/pitch_actor.py": "0" * 64}),
        lambda doc: doc["global_rows"][0].update(actor_id="pitch-feasible-argmax-7"),
        lambda doc: doc["global_rows"][0].update(seed=0),
        lambda doc: doc["feasible_rows"][0].update(training_wall_time_seconds=2.0),
        lambda doc: doc["baseline_rows"][0].update(actor_id="oracle"),
        lambda doc: doc.update(global_rows=[]),
        lambda doc: doc["runtime"].update(device="cuda"),
    ],
)
def test_report_rejects_identity_and_provenance_tampering(change):
    document = _report().to_document()
    change(document)
    with pytest.raises(ValueError):
        PitchDecoderComparison.from_document(document)


def test_report_rejects_reordered_episodes_and_claimed_metrics():
    document = _report().to_document()
    document["feasible_rows"][0]["episodes"].reverse()
    with pytest.raises(ValueError):
        PitchDecoderComparison.from_document(document)
    document = _report().to_document()
    document["global_rows"][0]["metrics"]["submitted_success_rate"] = 1.0
    with pytest.raises(ValueError):
        PitchDecoderComparison.from_document(document)


def test_package_evaluator_requires_matching_snapshot():
    report = _report()
    assert (
        replace(report, evaluator_source=report.evaluator_snapshot).evaluator_source
        == report.evaluator_snapshot
    )
    with pytest.raises(ValueError, match="package evaluator"):
        replace(
            report, evaluator_source=replace(report.evaluator_snapshot, package_sha256="0" * 64)
        )


def test_comparison_requires_cpu_before_dependency_resolution(monkeypatch, tmp_path):
    import harpy.learning.dependencies as dependencies

    def unexpected_dependencies():
        raise AssertionError("CPU contract must be checked before resolving dependencies")

    monkeypatch.setattr(dependencies, "require_training_dependencies", unexpected_dependencies)
    with pytest.raises(LearningContractError, match="requires CPU"):
        compare_pitch_decoders(tmp_path, device=DeviceName.CUDA)


def test_report_requires_cpu_evaluation_but_accepts_cuda_training():
    report = _report()
    cuda_trained = replace(
        report, provenance=replace(report.provenance, training_device=DeviceName.CUDA)
    )
    assert comparison_from_bytes(comparison_bytes(cuda_trained)) == cuda_trained
    document = report.to_document()
    document["provenance"]["evaluation_device"] = "cuda"
    document["runtime"]["device"] = "cuda"
    with pytest.raises(LearningContractError, match="requires CPU"):
        PitchDecoderComparison.from_document(document)


def _mock_execution(monkeypatch, profile):
    import harpy.learning.dependencies as dependencies
    import harpy.learning.pitch as pitch
    import harpy.learning.pitch_actor as actor_module
    import harpy.learning.pitch_artifacts as artifacts
    import harpy.learning.pitch_evaluation as evaluation
    import harpy.learning.workflows as workflows

    report = _report(profile)
    capture = (report.evaluator_source, report.evaluator_snapshot, dict(report.evaluator_files))
    monkeypatch.setattr(comparison, "_capture_evaluator", lambda: capture)
    monkeypatch.setattr(artifacts, "_capture_pitch_runtime_status", lambda device: report.runtime)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(
        workflows, "_artifact_readout_provenance", lambda artifact, *, device: report.provenance
    )
    artifact = SimpleNamespace(
        root=Path("."),
        manifest=SimpleNamespace(
            environment_id=ENVIRONMENT_ID, parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT, files=()
        ),
        file=lambda name: Path(name),
    )
    monkeypatch.setattr(artifacts, "load_pitch_artifact", lambda path: artifact)
    monkeypatch.setattr(
        artifacts,
        "read_pitch_training_summary",
        lambda artifact: SimpleNamespace(
            summary=SimpleNamespace(
                training_examples=256 if profile is ProfileName.SMOKE else 1400,
                training_wall_time_seconds=1.5,
            )
        ),
    )
    monkeypatch.setattr(
        dependencies,
        "require_training_dependencies",
        lambda: SimpleNamespace(torch=SimpleNamespace(device=lambda name: name)),
    )
    model = object()
    loads = []

    def load_model(path, *, device):
        loads.append((path, device))
        return model

    monkeypatch.setattr(pitch, "load_pitch_estimator_model", load_model)
    actors, calls, baselines = [], [], []

    def make_actor(loaded, *, device, decoding):
        actor = SimpleNamespace(model=loaded, decoding=decoding, device=device)
        actors.append(actor)
        return actor

    monkeypatch.setattr(actor_module, "PitchPlannerActor", make_actor)

    def evaluate(actor, suite, *, environment_factory):
        calls.append((actor.decoding, suite.suite_id))
        return _records(suite, actor.decoding)

    monkeypatch.setattr(evaluation, "evaluate_pitch_learned_actor", evaluate)

    def baseline(kind, suite, *, cache):
        baselines.append((kind, suite.suite_id, cache))
        return _records(suite)

    monkeypatch.setattr(evaluation, "evaluate_pitch_baseline_suite", baseline)
    return report, model, loads, actors, calls, baselines


@pytest.mark.parametrize("profile", list(ProfileName))
def test_comparison_reuses_one_model_and_fixed_matched_suites(monkeypatch, tmp_path, profile):
    expected, model, loads, actors, calls, baselines = _mock_execution(monkeypatch, profile)
    report = compare_pitch_decoders(tmp_path)
    assert report == expected
    assert len(loads) == 1
    assert len(actors) == 2 and all(actor.model is model for actor in actors)
    suites = tuple(row.suite_id for row in report.global_rows)
    assert calls == [(decoding, suite) for decoding in PitchDecoding for suite in suites]
    assert [(kind, suite) for kind, suite, _ in baselines] == [
        (BaselineKind.SPECTRUM_PEAK, suite) for suite in suites
    ]
    assert len({id(cache) for _, _, cache in baselines}) == 1


def test_comparison_rejects_source_changes_during_run(monkeypatch, tmp_path):
    report, *_ = _mock_execution(monkeypatch, ProfileName.SMOKE)
    values = iter(
        [
            (report.evaluator_source, report.evaluator_snapshot, dict(report.evaluator_files)),
            (
                replace(report.evaluator_source, commit="2" * 40),
                report.evaluator_snapshot,
                dict(report.evaluator_files),
            ),
        ]
    )
    monkeypatch.setattr(comparison, "_capture_evaluator", lambda: next(values))
    with pytest.raises(ValueError, match="source changed during"):
        compare_pitch_decoders(tmp_path)


@pytest.mark.parametrize("payload", ["model.pt", "training-summary.json"])
def test_comparison_rejects_payload_changes_during_run(monkeypatch, tmp_path, payload):
    import harpy.learning.pitch_artifacts as artifacts
    import harpy.learning.pitch_evaluation as evaluation

    _mock_execution(monkeypatch, ProfileName.SMOKE)
    original = b"original"
    path = tmp_path / payload
    path.write_bytes(original)
    artifact = artifacts.load_pitch_artifact(tmp_path)
    artifact.root = tmp_path
    artifact.manifest.files = (
        FileRecord(payload, len(original), hashlib.sha256(original).hexdigest()),
    )
    evaluate_baseline = evaluation.evaluate_pitch_baseline_suite

    def mutate_after_evaluation(kind, suite, *, cache):
        records = evaluate_baseline(kind, suite, cache=cache)
        path.write_bytes(b"modified")  # same length; unchanged manifest must not hide this
        return records

    monkeypatch.setattr(evaluation, "evaluate_pitch_baseline_suite", mutate_after_evaluation)
    with pytest.raises(ValueError, match="hash does not match"):
        compare_pitch_decoders(tmp_path)


def test_capture_includes_untracked_comparator_and_actual_file_hashes():
    _, snapshot, files = comparison._capture_evaluator()
    assert (
        files["learning/pitch_comparison.py"]
        == hashlib.sha256(Path(comparison.__file__).read_bytes()).hexdigest()
    )
    assert (
        snapshot.package_sha256
        == hashlib.sha256(
            json.dumps(sorted(files.items()), separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_saved_report_load_does_not_import_training_dependencies(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_bytes(comparison_bytes(_report()))
    script = """
import importlib.abc
import sys
from pathlib import Path
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3'}:
            raise AssertionError('training dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockTraining())
from harpy.learning.pitch_comparison import comparison_from_bytes
report = comparison_from_bytes(Path(sys.argv[1]).read_bytes())
assert report.paired_counts[0].rescued == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
