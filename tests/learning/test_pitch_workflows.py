"""Schema-v2 pitch evaluation, report, preflight, and run integration."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest

import harpy.learning.workflows as workflows
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode, TerminalReason
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PITCH_ARTIFACT_SCHEMA_VERSION,
    RuntimeStatus,
    SourceStatus,
    canonical_json_bytes,
)
from harpy.learning.errors import (
    ArtifactError,
    LearningContractError,
    PitchArtifactSetError,
)
from harpy.learning.evaluation import TerminalEpisodeRecord
from harpy.learning.models import (
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    SPECTRUM_GRID_ID,
    DeviceName,
    EpisodeSpec,
    PitchCoordinatePartition,
    PitchCoordinateRecord,
    ProfileName,
)
from harpy.learning.pitch_artifacts import (
    PITCH_POLICY_SEMANTICS_ID,
    PITCH_PREPROCESSING_SCHEMA_ID,
    LoadedPitchArtifact,
    pitch_compatibility_sha256,
)
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
    pitch_coordinate_split,
)
from harpy.learning.pitch_e1 import (
    PITCH_E1_PROTOCOL_ID,
    PITCH_E1_REPORT_SCHEMA_VERSION,
    PitchE1ArtifactProvenance,
    PitchE1EvaluationReport,
    PitchE1RawEvidence,
)
from harpy.learning.pitch_evaluation import (
    PITCH_SHUFFLED_SPECTRUM_PROBE,
    PITCH_ZERO_SPECTRUM_PROBE,
    PitchCoordinateEvaluation,
    RegisterOODAggregate,
    build_pitch_evaluation_row,
    evaluate_pitch_criterion,
)
from harpy.learning.pitch_network import (
    PITCH_ESTIMATOR_ARCHITECTURE_ID,
    PITCH_ESTIMATOR_PARAMETER_COUNT,
    PitchEstimatorNetwork,
    pitch_cents_from_index,
    pitch_class_index,
)
from harpy.learning.pitch_reports import PitchEvaluationReport

_BASELINE_ORDER = (
    BaselineKind.RANDOM,
    BaselineKind.REWARD_SEARCH,
    BaselineKind.SPECTRUM_PEAK,
    BaselineKind.ORACLE,
)


def _terminal_records(
    suite_id: PitchEvaluationSuiteId,
) -> tuple[TerminalEpisodeRecord, ...]:
    suite = fixed_pitch_evaluation_suite(suite_id)
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=EpisodeSpec(episode.target_note_index, episode.source_pitch_cents),
            submitted_success=False,
            within_5_cents=False,
            within_1_cent=False,
            final_absolute_error_cents=100,
            action_count=1,
            excess_actions=None,
            invalid_action_count=0,
            total_return=-1.0,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        )
        for index, episode in enumerate(suite.episodes)
    )


def _pitch_row(
    *,
    actor_id: str,
    trainer: PitchTrainerKind | None,
    seed: int | None,
    suite_id: PitchEvaluationSuiteId,
    probe: str | None = None,
):
    suite = fixed_pitch_evaluation_suite(suite_id)
    if trainer is None:
        kind = BaselineKind(actor_id)
        environment_id = kind.environment_id
        observation_mode = kind.observation_mode
    else:
        environment_id = ENVIRONMENT_ID
        observation_mode = ObservationMode.SPECTRUM
    return build_pitch_evaluation_row(
        actor_id=actor_id,
        trainer=trainer,
        seed=seed,
        environment_id=environment_id,
        observation_mode=observation_mode,
        suite=suite,
        records=_terminal_records(suite_id),
        probe=probe,
        parameter_count=None if trainer is None else PITCH_ESTIMATOR_PARAMETER_COUNT,
        training_examples=(
            None
            if trainer is None
            else (256 if suite_id is PitchEvaluationSuiteId.SMOKE else 1_400)
        ),
        training_wall_time_seconds=None if trainer is None else 12.5,
    )


@cache
def _smoke_report(device: DeviceName = DeviceName.CPU) -> PitchEvaluationReport:
    rows = (
        _pitch_row(
            actor_id="pitch-0",
            trainer=PitchTrainerKind.PITCH,
            seed=0,
            suite_id=PitchEvaluationSuiteId.SMOKE,
        ),
        _pitch_row(
            actor_id="pitch-0",
            trainer=PitchTrainerKind.PITCH,
            seed=0,
            suite_id=PitchEvaluationSuiteId.SMOKE,
            probe=PITCH_ZERO_SPECTRUM_PROBE,
        ),
        _pitch_row(
            actor_id="pitch-0",
            trainer=PitchTrainerKind.PITCH,
            seed=0,
            suite_id=PitchEvaluationSuiteId.SMOKE,
            probe=PITCH_SHUFFLED_SPECTRUM_PROBE,
        ),
        *(
            _pitch_row(
                actor_id=kind.value,
                trainer=None,
                seed=None,
                suite_id=PitchEvaluationSuiteId.SMOKE,
            )
            for kind in _BASELINE_ORDER
        ),
    )
    return PitchEvaluationReport(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.SMOKE,
        evaluation_device=device,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
        policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
        coordinate_distribution_id=PITCH_DISTRIBUTION_ID,
        coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        compatibility_sha256=pitch_compatibility_sha256(ProfileName.SMOKE),
        terminal_rows=rows,
        coordinate_evaluations=(),
        register_ood_aggregates=(),
        criterion=evaluate_pitch_criterion(
            coordinate_evaluations=(),
            terminal_rows=(),
            eligible=False,
        ),
    )


@cache
def _coordinate_evaluation(seed: int) -> PitchCoordinateEvaluation:
    split = pitch_coordinate_split()
    members = (
        (PitchCoordinatePartition.IID, tuple(sorted(split.iid_holdout_coordinates))),
        (PitchCoordinatePartition.OOD_LOWER, split.ood_lower_coordinates),
        (PitchCoordinatePartition.OOD_UPPER, split.ood_upper_coordinates),
    )
    records = []
    for partition, coordinates in members:
        for coordinate in coordinates:
            grid_index = pitch_class_index(coordinate)
            predicted_cents = pitch_cents_from_index(grid_index)
            signed_error = predicted_cents - coordinate
            records.append(
                PitchCoordinateRecord(
                    seed=seed,
                    distribution_id=PITCH_DISTRIBUTION_ID,
                    split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
                    partition=partition,
                    true_coordinate_cents=coordinate,
                    predicted_grid_index=grid_index,
                    predicted_cents=predicted_cents,
                    signed_error_cents=signed_error,
                    absolute_error_cents=abs(signed_error),
                )
            )
    return PitchCoordinateEvaluation.from_records(seed=seed, records=tuple(records))


@cache
def _checkpoint_report(device: DeviceName = DeviceName.CPU) -> PitchEvaluationReport:
    rows = []
    for seed in (0, 1, 2):
        rows.extend(
            (
                _pitch_row(
                    actor_id=f"pitch-{seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=seed,
                    suite_id=PitchEvaluationSuiteId.IID,
                ),
                _pitch_row(
                    actor_id=f"pitch-{seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=seed,
                    suite_id=PitchEvaluationSuiteId.OOD_LOWER,
                ),
                _pitch_row(
                    actor_id=f"pitch-{seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=seed,
                    suite_id=PitchEvaluationSuiteId.OOD_UPPER,
                ),
                _pitch_row(
                    actor_id=f"pitch-{seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=seed,
                    suite_id=PitchEvaluationSuiteId.IID,
                    probe=PITCH_ZERO_SPECTRUM_PROBE,
                ),
                _pitch_row(
                    actor_id=f"pitch-{seed}",
                    trainer=PitchTrainerKind.PITCH,
                    seed=seed,
                    suite_id=PitchEvaluationSuiteId.IID,
                    probe=PITCH_SHUFFLED_SPECTRUM_PROBE,
                ),
            )
        )
    for kind in _BASELINE_ORDER:
        rows.extend(
            _pitch_row(
                actor_id=kind.value,
                trainer=None,
                seed=None,
                suite_id=suite_id,
            )
            for suite_id in (
                PitchEvaluationSuiteId.IID,
                PitchEvaluationSuiteId.OOD_LOWER,
                PitchEvaluationSuiteId.OOD_UPPER,
            )
        )
    terminal_rows = tuple(rows)
    coordinate_evaluations = tuple(_coordinate_evaluation(seed) for seed in (0, 1, 2))
    aggregates = tuple(
        RegisterOODAggregate.from_rows(terminal_rows[start], terminal_rows[start + 1])
        for start in (1, 6, 11, 16, 19, 22, 25)
    )
    criterion = evaluate_pitch_criterion(
        coordinate_evaluations=coordinate_evaluations,
        terminal_rows=terminal_rows,
        eligible=device is DeviceName.CPU,
    )
    return PitchEvaluationReport(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.CHECKPOINT,
        evaluation_device=device,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
        policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
        coordinate_distribution_id=PITCH_DISTRIBUTION_ID,
        coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        compatibility_sha256=pitch_compatibility_sha256(ProfileName.CHECKPOINT),
        terminal_rows=terminal_rows,
        coordinate_evaluations=coordinate_evaluations,
        register_ood_aggregates=aggregates,
        criterion=criterion,
    )


def test_pitch_smoke_report_codec_is_canonical_and_exact() -> None:
    report = _smoke_report()

    content = workflows.evaluation_report_bytes(report)

    assert workflows.evaluation_report_from_bytes(content) == report
    assert content == canonical_json_bytes(report.to_document())
    assert len(report.terminal_rows) == 7
    assert tuple(row.actor_id for row in report.terminal_rows) == (
        "pitch-0",
        "pitch-0",
        "pitch-0",
        "random",
        "reward_search",
        "spectrum_peak",
        "oracle",
    )
    assert report.coordinate_evaluations == ()
    assert report.register_ood_aggregates == ()
    assert report.criterion.status == "ineligible"


@pytest.mark.parametrize("profile", [ProfileName.SMOKE, ProfileName.CHECKPOINT])
def test_pitch_report_compatibility_digest_matches_artifact_schema(profile: ProfileName) -> None:
    from harpy.learning.pitch_reports import pitch_report_compatibility_sha256

    assert pitch_report_compatibility_sha256(profile) == pitch_compatibility_sha256(profile)


@pytest.mark.parametrize("mutation", ["actor_id", "parameter_count", "training_examples"])
def test_pitch_report_rejects_consistently_spoofed_learned_metadata(mutation: str) -> None:
    report = _smoke_report()
    document = report.to_document()
    for row in document["terminal_rows"][:3]:
        if mutation == "actor_id":
            row[mutation] = "pitch-spoofed"
        else:
            row[mutation] = 999

    with pytest.raises(ValueError, match=r"profile metadata|row matrix"):
        PitchEvaluationReport.from_document(document)


def test_pitch_report_codec_import_is_training_dependency_free() -> None:
    script = r"""
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"blocked import: {fullname}")
        if fullname == "stable_baselines3" or fullname.startswith("stable_baselines3."):
            raise RuntimeError(f"blocked import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import harpy.learning.pitch_reports
assert "torch" not in sys.modules
assert "stable_baselines3" not in sys.modules
assert "harpy.learning.pitch_artifacts" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("schema_version", [True, 2.0])
def test_pitch_report_rejects_noninteger_schema_ids(schema_version: object) -> None:
    report = _smoke_report()
    with pytest.raises(ValueError, match="schema_version"):
        replace(report, schema_version=schema_version)  # type: ignore[arg-type]
    document = report.to_document()
    document["schema_version"] = schema_version
    with pytest.raises(ValueError, match="schema_version"):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(document))


@pytest.mark.parametrize("mutation", ["terminal", "coordinate", "aggregate", "criterion"])
def test_pitch_checkpoint_report_decoder_rederives_every_raw_section(mutation: str) -> None:
    report = _checkpoint_report()
    document = copy.deepcopy(report.to_document())
    if mutation == "terminal":
        document["terminal_rows"][0]["metrics"]["mean_return"] = 0.5
    elif mutation == "coordinate":
        document["coordinate_evaluations"][0]["records"][0]["predicted_cents"] += 5
    elif mutation == "aggregate":
        document["register_ood_aggregates"][0]["metrics"]["mean_return"] = 0.5
    else:
        document["criterion"]["status"] = "criterion_met"

    with pytest.raises(ValueError):
        PitchEvaluationReport.from_document(document)
    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(document))


def test_pitch_checkpoint_report_pins_complete_order_and_cuda_ineligibility() -> None:
    cpu = _checkpoint_report(DeviceName.CPU)
    cuda = _checkpoint_report(DeviceName.CUDA)

    assert len(cpu.terminal_rows) == 27
    assert tuple(item.seed for item in cpu.coordinate_evaluations) == (0, 1, 2)
    assert all(len(item.records) == 801 for item in cpu.coordinate_evaluations)
    assert len(cpu.register_ood_aggregates) == 7
    assert cpu.criterion.eligible is True
    assert cpu.criterion.status == "criterion_not_met"
    assert cuda.criterion.eligible is False
    assert cuda.criterion.status == "ineligible"


def _e1_source() -> SourceStatus:
    return SourceStatus(
        commit="1" * 40,
        dirty_tree=False,
        tracked_diff_sha256=hashlib.sha256(b"").hexdigest(),
        dependency_lock_sha256="2" * 64,
        required_inputs_committed=True,
    )


def _e1_runtime() -> RuntimeStatus:
    return RuntimeStatus(
        python_version="3.12.3",
        platform="Linux-test",
        processor="x86_64",
        numpy_version="2.5.1",
        gymnasium_version="1.3.0",
        torch_version="2.13.0+cu130",
        stable_baselines3_version="2.9.0",
        device=DeviceName.CUDA,
        device_description="Test CUDA; compute capability 8.9; 16 SMs",
        cuda_runtime_version="13.0",
        cuda_driver_version="596.08",
    )


def _e1_report() -> PitchE1EvaluationReport:
    source = _e1_source()
    runtime = _e1_runtime()
    provenance = tuple(
        PitchE1ArtifactProvenance(
            seed=seed,
            artifact_manifest_sha256=str(seed + 3) * 64,
            source=source,
            runtime=runtime,
            compatibility_sha256=pitch_compatibility_sha256(ProfileName.CHECKPOINT),
        )
        for seed in (0, 1, 2)
    )
    return PitchE1EvaluationReport.create(
        artifacts=provenance,
        evaluator_source=source,
        evidence=PitchE1RawEvidence.from_pitch_report(_checkpoint_report(DeviceName.CPU)),
    )


def test_e1_cuda_report_is_strict_versioned_cpu_evidence() -> None:
    report = _e1_report()
    content = workflows.evaluation_report_bytes(report)

    assert report.schema_version == PITCH_E1_REPORT_SCHEMA_VERSION == 3
    assert report.protocol_id == PITCH_E1_PROTOCOL_ID
    assert report.training_device is DeviceName.CUDA
    assert report.evaluation_device is DeviceName.CPU
    assert report.criterion.eligible is True
    assert workflows.evaluation_report_from_bytes(content) == report
    assert content == canonical_json_bytes(report.to_document())
    with pytest.raises(ValueError, match="schema_version"):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(report.evidence.to_document()))


@pytest.mark.parametrize(
    "mutation",
    ["training_device", "manifest", "source", "evidence", "criterion"],
)
def test_e1_report_decoder_rederives_cohort_identity(mutation: str) -> None:
    document = copy.deepcopy(_e1_report().to_document())
    if mutation == "training_device":
        document["training_device"] = "cpu"
    elif mutation == "manifest":
        document["artifacts"][0]["artifact_manifest_sha256"] = "f" * 64
    elif mutation == "source":
        document["evaluator_source"]["commit"] = "9" * 40
    elif mutation == "evidence":
        document["evidence"]["terminal_rows"][0]["metrics"]["mean_return"] = 0.5
    else:
        document["criterion"]["status"] = "criterion_met"

    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(document))


@pytest.mark.parametrize("mutation", ["reorder", "duplicate", "wrong", "compatibility", "digest"])
def test_e1_report_rejects_cohort_identity_mutations(mutation: str) -> None:
    document = copy.deepcopy(_e1_report().to_document())
    if mutation == "reorder":
        document["artifacts"][0], document["artifacts"][1] = (
            document["artifacts"][1],
            document["artifacts"][0],
        )
    elif mutation == "duplicate":
        document["artifacts"][2]["seed"] = 1
    elif mutation == "wrong":
        document["artifacts"][2]["seed"] = 3
    elif mutation == "compatibility":
        document["artifacts"][2]["compatibility_sha256"] = "f" * 64
    else:
        document["cohort_digest_sha256"] = "f" * 64

    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(document))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligible", 0),
        ("eligible", 1),
        ("criterion_met", 0),
        ("criterion_met", 1),
        ("criterion_met", None),
    ],
)
def test_e1_report_rejects_nonboolean_or_null_criterion_fields(
    field: str,
    value: object,
) -> None:
    document = copy.deepcopy(_e1_report().to_document())
    document["criterion"][field] = value

    with pytest.raises(ValueError, match="criterion"):
        workflows.evaluation_report_from_bytes(canonical_json_bytes(document))


def _write_peek_manifest(
    root: Path,
    *,
    schema_version: int,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    training_device: DeviceName = DeviceName.CPU,
) -> None:
    root.mkdir()
    document: dict[str, object] = {
        "schema_version": schema_version,
        "profile": profile.value,
        "seed": seed,
        "trainer": "pitch" if schema_version == 2 else "bc",
    }
    if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        eligible = profile is ProfileName.CHECKPOINT and training_device is DeviceName.CPU
        document.update(
            {
                "status": "complete",
                "source": {
                    "commit": "test-commit",
                    "dirty_tree": False,
                    "dependency_lock_sha256": "a" * 64,
                    "required_inputs_committed": True,
                },
                "runtime": {"device": training_device.value},
                "evaluation_device": "cpu",
                "eligible_for_aggregate": eligible,
                "criterion_status": ("eligible_for_aggregate" if eligible else "ineligible"),
                "criterion_met": None,
                "compatibility_sha256": "b" * 64,
            }
        )
    (root / "manifest.json").write_bytes(canonical_json_bytes(document))


def _exploratory_artifact(root: Path, profile: ProfileName) -> LoadedPitchArtifact:
    """A strict-loader test double with deliberately non-cohort source/seed/device."""
    from harpy.learning.artifacts import FileRecord, _source_to_document

    _write_peek_manifest(
        root, schema_version=2, profile=profile, seed=17, training_device=DeviceName.CUDA
    )
    source = SourceStatus("a" * 40, True, "b" * 64, "c" * 64, True)
    document = workflows.decode_json_bytes((root / "manifest.json").read_bytes())
    document["source"] = _source_to_document(source)
    document["files"] = [{"relative_path": "model.pt", "size_bytes": 1, "sha256": "d" * 64}]
    (root / "manifest.json").write_bytes(canonical_json_bytes(document))
    artifact = object.__new__(LoadedPitchArtifact)
    object.__setattr__(artifact, "root", root.resolve())
    object.__setattr__(
        artifact,
        "manifest",
        SimpleNamespace(
            schema_version=2,
            profile=profile,
            trainer=PitchTrainerKind.PITCH,
            seed=17,
            runtime=SimpleNamespace(device=DeviceName.CUDA),
            source=source,
            files=(FileRecord("model.pt", 1, "d" * 64),),
            environment_id=ENVIRONMENT_ID,
            parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
            policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
            to_document=lambda: copy.deepcopy(document),
        ),
    )
    return artifact


@pytest.mark.parametrize("profile", [ProfileName.SMOKE, ProfileName.CHECKPOINT])
def test_single_pitch_exploration_keeps_raw_metrics_and_provenance_but_never_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: ProfileName
) -> None:
    from harpy.learning import pitch_artifacts, pitch_evaluation
    from harpy.learning.experiment_results import readout_from_bytes

    root = tmp_path / "single"
    artifact = _exploratory_artifact(root, profile)
    loaded = []
    monkeypatch.setattr(workflows, "load_artifact", lambda path: loaded.append(path) or artifact)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(workflows, "_load_pitch_artifact_actor", lambda *args, **kwargs: object())
    for name in ("_preflight_pitch_artifacts", "_preflight_pitch_e1_artifacts"):
        monkeypatch.setattr(workflows, name, lambda *args: pytest.fail("exploration used a cohort"))
    monkeypatch.setattr(
        pitch_artifacts,
        "read_pitch_training_summary",
        lambda artifact: SimpleNamespace(
            summary=SimpleNamespace(
                training_examples=256 if profile is ProfileName.SMOKE else 1_400,
                training_wall_time_seconds=1.5,
            )
        ),
    )
    evaluated_suites = []

    def evaluate_actor(actor, suite, **kwargs):
        evaluated_suites.append(suite.suite_id)
        return _terminal_records(suite.suite_id)

    monkeypatch.setattr(pitch_evaluation, "evaluate_pitch_learned_actor", evaluate_actor)
    monkeypatch.setattr(
        pitch_evaluation,
        "evaluate_pitch_baseline_suite",
        lambda kind, suite, **kwargs: _terminal_records(suite.suite_id),
    )

    result = workflows.evaluate_artifacts((root,), exploratory=True)
    content = workflows.evaluation_report_bytes(result)
    restored = readout_from_bytes(content)

    assert restored == result
    assert workflows.evaluation_report_from_bytes(content) == result
    assert loaded == [root.resolve()]
    assert evaluated_suites == (
        [PitchEvaluationSuiteId.SMOKE]
        if profile is ProfileName.SMOKE
        else [
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        ]
    )
    assert len(result.terminal_rows) == 5 * len(evaluated_suites)
    assert result.provenance.training_seed == 17
    assert result.provenance.training_device is DeviceName.CUDA
    assert result.provenance.evaluation_device is DeviceName.CPU
    assert result.provenance.source.dirty_tree is True
    assert (
        result.provenance.manifest_sha256
        == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    )
    assert result.provenance.model_sha256 == "d" * 64
    assert result.to_document()["criterion"] == {
        "eligible": False,
        "criterion_met": None,
        "status": "ineligible",
        "reason": "exploratory_single_artifact",
    }
    for mutation in (
        "eligibility",
        "seed",
        "metrics",
        "missing_baseline",
        "parameter_count",
        "training_examples",
        "baseline_trainer",
    ):
        document = result.to_document()
        if mutation == "eligibility":
            document["criterion"]["eligible"] = True
        elif mutation == "seed":
            document["provenance"]["training_seed"] = 99
        elif mutation == "metrics":
            document["terminal_rows"][0]["metrics"]["submitted_success_rate"] = 1.0
        elif mutation in {"parameter_count", "training_examples"}:
            document["terminal_rows"][0][mutation] = 123
        elif mutation == "baseline_trainer":
            spoofed_baseline = copy.deepcopy(document["terminal_rows"][0])
            spoofed_baseline["actor_id"] = "random"
            document["terminal_rows"][len(evaluated_suites)] = spoofed_baseline
        else:
            document["terminal_rows"].pop()
        with pytest.raises(ValueError):
            readout_from_bytes(canonical_json_bytes(document))
    if profile is ProfileName.CHECKPOINT:
        with pytest.raises(LearningContractError, match="exactly three"):
            workflows.evaluate_artifacts((root,))


@pytest.mark.parametrize("suite", ["smoke", "iid", "ood-lower", "ood-upper"])
def test_exploratory_diagnostic_preflight_admits_one_arbitrary_checkpoint_only_explicitly(
    tmp_path: Path, suite: str
) -> None:
    root = tmp_path / "single"
    _exploratory_artifact(root, ProfileName.CHECKPOINT)

    workflows.preflight_diagnostic_request((root,), suite=suite, exploratory=True)

    with pytest.raises(LearningContractError):
        workflows.preflight_diagnostic_request((root,), suite=suite)
    with pytest.raises(LearningContractError, match="bound_mask"):
        workflows.preflight_diagnostic_request(
            (root,), suite=suite, exploratory=True, bound_mask=True
        )


def test_exploration_does_not_bypass_a_corrupt_artifact_or_accept_legacy_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "single"
    _write_peek_manifest(root, schema_version=2, profile=ProfileName.CHECKPOINT, seed=17)

    def corrupt(path):
        raise ValueError("model payload hash mismatch")

    monkeypatch.setattr(workflows, "load_artifact", corrupt)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        workflows.evaluate_artifacts((root,), exploratory=True)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        workflows.diagnose_artifacts((root,), suite="iid", exploratory=True)
    legacy = tmp_path / "legacy"
    _write_peek_manifest(legacy, schema_version=1)
    with pytest.raises(LearningContractError, match="exactly one pitch"):
        workflows.evaluate_artifacts((legacy,), exploratory=True)


@pytest.mark.parametrize("suite", ["smoke", "iid"])
def test_exploratory_diagnostics_execute_single_checkpoint_and_bind_exact_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str
) -> None:
    from harpy.envs.models import PitchAction
    from harpy.learning import evaluation
    from harpy.learning.diagnostics import DiagnosticDecisionInput, diagnose_episode
    from harpy.learning.experiment_results import readout_bytes, readout_from_bytes

    root = tmp_path / "single"
    artifact = _exploratory_artifact(root, ProfileName.CHECKPOINT)
    monkeypatch.setattr(workflows, "load_artifact", lambda path: artifact)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(
        workflows,
        "_load_pitch_artifact_actor",
        lambda *args, **kwargs: SimpleNamespace(decide=lambda observation: None),
    )

    def episodes(*, decide, suite, environment_factory):
        return tuple(
            diagnose_episode(
                episode_index=index,
                source_pitch_cents=episode.source_pitch_cents,
                target_note_index=episode.target_note_index,
                decisions=(
                    DiagnosticDecisionInput(
                        PitchAction.SUBMIT,
                        pitch_cents_from_index(pitch_class_index(episode.source_pitch_cents)),
                    ),
                ),
            )
            for index, episode in enumerate(suite.episodes)
        )

    monkeypatch.setattr(evaluation, "diagnose_actor_suite", episodes)
    result = workflows.diagnose_artifacts((root,), suite=suite, exploratory=True)
    assert readout_from_bytes(readout_bytes(result)) == result
    assert result.diagnostics.seed == 17
    assert len(result.diagnostics.episodes) == (50 if suite == "smoke" else 250)
    assert result.to_document()["criterion"]["eligible"] is False
    document = result.to_document()
    document["provenance"]["manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="provenance"):
        readout_from_bytes(canonical_json_bytes(document))


def test_experiment_readouts_import_without_the_training_stack() -> None:
    script = """
import importlib.abc
import sys

class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3'}:
            raise AssertionError(f'readout imported optional training package: {fullname}')

sys.meta_path.insert(0, BlockTraining())
import harpy.learning.experiment_results
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_evaluate_rejects_mixed_or_unknown_schema_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    unknown = tmp_path / "unknown"
    _write_peek_manifest(v1, schema_version=ARTIFACT_SCHEMA_VERSION)
    _write_peek_manifest(v2, schema_version=PITCH_ARTIFACT_SCHEMA_VERSION)
    _write_peek_manifest(unknown, schema_version=999)
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"schema rejection loaded {path}"),
    )

    with pytest.raises(LearningContractError, match="mixed"):
        workflows.evaluate_artifacts((v1, v2))
    with pytest.raises(LearningContractError, match="unsupported"):
        workflows.evaluate_artifacts((unknown,))


def test_unknown_schema_rejects_before_profile_or_trainer_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "unknown"
    root.mkdir()
    (root / "manifest.json").write_bytes(b'{"schema_version":999}\n')
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"unknown schema reached artifact load: {path}"),
    )

    with pytest.raises(LearningContractError, match=r"unsupported.*999"):
        workflows.evaluate_artifacts((root,))


def test_checkpoint_calls_exact_preflight_once_before_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"seed-{seed}" for seed in (2, 0, 1))
    for path, seed in zip(paths, (2, 0, 1), strict=True):
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
            seed=seed,
        )
    events = []
    artifacts = tuple(object.__new__(LoadedPitchArtifact) for _ in range(3))

    def preflight(supplied):
        events.append(("preflight", tuple(supplied)))
        return artifacts

    def evaluate(supplied, *, device):
        assert events == [("preflight", paths)]
        assert supplied == artifacts
        assert device is DeviceName.CPU
        events.append(("evaluate", supplied))
        return _checkpoint_report()

    monkeypatch.setattr(workflows, "_preflight_pitch_artifacts", preflight)
    monkeypatch.setattr(workflows, "_evaluate_pitch_checkpoint", evaluate)

    report = workflows.evaluate_artifacts(paths)

    assert report is _checkpoint_report()
    assert tuple(event[0] for event in events) == ("preflight", "evaluate")


def test_cuda_checkpoint_routes_through_e1_and_requires_cpu_final_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"seed-{seed}" for seed in (2, 0, 1))
    for path, seed in zip(paths, (2, 0, 1), strict=True):
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
            seed=seed,
            training_device=DeviceName.CUDA,
        )
    cohort = SimpleNamespace(training_device=DeviceName.CUDA, artifacts=(object(),) * 3)
    expected = _e1_report()
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        workflows,
        "_preflight_pitch_e1_artifacts",
        lambda supplied: calls.append(("preflight", tuple(supplied))) or cohort,
    )
    monkeypatch.setattr(
        workflows,
        "_evaluate_pitch_e1_checkpoint",
        lambda supplied: calls.append(("evaluate", supplied)) or expected,
    )

    assert workflows.evaluate_artifacts(paths, device=DeviceName.CPU) is expected
    assert calls == [("preflight", paths), ("evaluate", cohort)]
    with pytest.raises(LearningContractError, match=r"E\.1 final evaluation must use CPU"):
        workflows.evaluate_artifacts(paths, device=DeviceName.CUDA)


def test_stale_e1_source_blocks_evaluation_and_diagnostics_before_evidence_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"stale-seed-{seed}" for seed in (0, 1, 2))
    for path, seed in zip(paths, (0, 1, 2), strict=True):
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
            seed=seed,
            training_device=DeviceName.CUDA,
        )

    def stale_preflight(_paths: object) -> object:
        raise PitchArtifactSetError("evaluator source is stale")

    monkeypatch.setattr(workflows, "_preflight_pitch_e1_artifacts", stale_preflight)
    monkeypatch.setattr(
        workflows,
        "_evaluate_pitch_e1_checkpoint",
        lambda cohort: pytest.fail(f"stale source reached final evidence: {cohort}"),
    )
    monkeypatch.setattr(
        workflows,
        "_diagnose_pitch_artifacts",
        lambda *args, **kwargs: pytest.fail("stale source reached diagnostics evidence"),
    )
    monkeypatch.setattr(
        workflows,
        "_require_evaluation_device",
        lambda device: pytest.fail(f"stale source reached dependency/device access: {device}"),
    )

    with pytest.raises(LearningContractError, match="evaluator source is stale"):
        workflows.evaluate_artifacts(paths)
    with pytest.raises(LearningContractError, match="evaluator source is stale"):
        workflows.diagnose_artifacts(paths, suite="iid")


def test_pitch_checkpoint_reuses_each_reloaded_model_and_one_evidence_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import harpy.learning.pitch as pitch
    import harpy.learning.pitch_actor as pitch_actor
    import harpy.learning.pitch_artifacts as pitch_artifacts
    import harpy.learning.pitch_evaluation as pitch_evaluation

    artifacts = []
    for seed in (0, 1, 2):
        root = tmp_path / f"seed-{seed}"
        root.mkdir()
        artifact = object.__new__(LoadedPitchArtifact)
        object.__setattr__(artifact, "root", root.resolve())
        object.__setattr__(
            artifact,
            "manifest",
            SimpleNamespace(
                seed=seed,
                profile=ProfileName.CHECKPOINT,
                environment_id=ENVIRONMENT_ID,
                environment_contract_id=ENVIRONMENT_CONTRACT_ID,
                spectrum_grid_id=SPECTRUM_GRID_ID,
                preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
                architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
                policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
                train_distribution_id=PITCH_DISTRIBUTION_ID,
                coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
                compatibility_sha256=pitch_compatibility_sha256(ProfileName.CHECKPOINT),
                parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
            ),
        )
        artifacts.append(artifact)

    loaded: list[tuple[Path, object, object]] = []
    actor_inputs: list[tuple[object, object]] = []
    predictor_inputs: list[tuple[object, object]] = []
    direct_inputs: list[tuple[int, object, object]] = []
    environment_cache_inputs: list[object] = []
    baseline_cache_inputs: list[object] = []
    constructed_caches: list[object] = []

    class TrackingCache(workflows.SpectrumEvidenceCache):
        def __init__(self) -> None:
            super().__init__()
            constructed_caches.append(self)

    class FakeActor:
        def __init__(self, model: object) -> None:
            self.model = model

    class FakePredictor:
        def __init__(self, model: object) -> None:
            self.model = model

        def __call__(self, spectrum: object) -> int:
            del spectrum
            return 0

    fake_torch = SimpleNamespace(device=lambda value: f"torch:{value}")

    def artifact_file(artifact: LoadedPitchArtifact, relative_path: str) -> Path:
        assert relative_path == "model.pt"
        return artifact.root / relative_path

    def load_model(path: Path, *, device: object) -> object:
        model = object()
        loaded.append((path, device, model))
        return model

    def make_actor(model: object, *, device: object) -> FakeActor:
        actor_inputs.append((model, device))
        return FakeActor(model)

    def make_predictor(model: object, *, device: object) -> FakePredictor:
        predictor_inputs.append((model, device))
        return FakePredictor(model)

    def make_provider(cache: object) -> object:
        direct_inputs.append((-1, cache, None))
        return SimpleNamespace(cache=cache)

    def evaluate_coordinates(
        *,
        seed: int,
        evidence_provider: object,
        predict_grid_index: object,
    ) -> PitchCoordinateEvaluation:
        direct_inputs.append((seed, evidence_provider, predict_grid_index))
        return _coordinate_evaluation(seed)

    def make_environment(cache: object) -> object:
        environment_cache_inputs.append(cache)
        return SimpleNamespace(cache=cache)

    def evaluate_actor(actor: object, suite: object, *, environment_factory: object):
        assert isinstance(actor, FakeActor)
        environment = environment_factory()
        assert environment.cache is constructed_caches[0]
        return _terminal_records(suite.suite_id)

    def make_probe_factory(environment_factory: object, probe: str):
        assert probe in (PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE)
        return environment_factory

    def evaluate_baseline(kind: BaselineKind, suite: object, *, cache: object):
        del kind
        baseline_cache_inputs.append(cache)
        return _terminal_records(suite.suite_id)

    monkeypatch.setattr(workflows, "SpectrumEvidenceCache", TrackingCache)
    monkeypatch.setattr(
        workflows,
        "require_training_dependencies",
        lambda: SimpleNamespace(torch=fake_torch),
    )
    monkeypatch.setattr(workflows, "make_cached_sine_pitch_env", make_environment)
    monkeypatch.setattr(LoadedPitchArtifact, "file", artifact_file)
    monkeypatch.setattr(pitch, "load_pitch_estimator_model", load_model)
    monkeypatch.setattr(pitch_actor, "PitchPlannerActor", make_actor)
    monkeypatch.setattr(
        pitch_artifacts,
        "read_pitch_training_summary",
        lambda artifact: SimpleNamespace(
            summary=SimpleNamespace(
                training_examples=1_400,
                training_wall_time_seconds=12.5,
            )
        ),
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "default_pitch_evidence_provider",
        make_provider,
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "evaluate_final_pitch_coordinates",
        evaluate_coordinates,
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "evaluate_pitch_learned_actor",
        evaluate_actor,
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "evaluate_pitch_baseline_suite",
        evaluate_baseline,
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "make_pitch_model_grid_predictor",
        make_predictor,
    )
    monkeypatch.setattr(
        pitch_evaluation,
        "make_pitch_spectrum_probe_factory",
        make_probe_factory,
    )

    report = workflows._evaluate_pitch_checkpoint(
        tuple(artifacts),
        device=DeviceName.CPU,
    )

    assert len(constructed_caches) == 1
    shared_cache = constructed_caches[0]
    assert [path for path, _device, _model in loaded] == [
        artifact.root / "model.pt" for artifact in artifacts
    ]
    assert len(loaded) == len(actor_inputs) == len(predictor_inputs) == 3
    for index, (_path, device, model) in enumerate(loaded):
        assert device == "torch:cpu"
        assert actor_inputs[index] == (model, device)
        assert predictor_inputs[index] == (model, device)
    assert direct_inputs[0] == (-1, shared_cache, None)
    provider = direct_inputs[1][1]
    assert provider.cache is shared_cache
    assert tuple(seed for seed, _provider, _predictor in direct_inputs[1:]) == (0, 1, 2)
    for seed, coordinate_provider, predictor in direct_inputs[1:]:
        assert coordinate_provider is provider
        assert predictor.model is loaded[seed][2]
    assert len(environment_cache_inputs) == 15
    assert all(cache is shared_cache for cache in environment_cache_inputs)
    assert len(baseline_cache_inputs) == 12
    assert all(cache is shared_cache for cache in baseline_cache_inputs)
    assert len(report.terminal_rows) == 27
    assert tuple(row.seed for row in report.terminal_rows[:15]) == (
        *(0 for _ in range(5)),
        *(1 for _ in range(5)),
        *(2 for _ in range(5)),
    )
    assert tuple(row.actor_id for row in report.terminal_rows[15:]) == tuple(
        kind.value for kind in _BASELINE_ORDER for _ in range(3)
    )


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (
            PitchArtifactSetError("pitch aggregate seeds must be exactly 0, 1, and 2"),
            LearningContractError,
        ),
        (ValueError("artifact inventory mismatch"), ArtifactError),
        (LearningContractError("pitch estimator payload state is invalid"), ArtifactError),
    ],
)
def test_checkpoint_preflight_preserves_contract_vs_integrity_errors(
    failure: Exception,
    error_type: type[Exception],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"seed-{seed}" for seed in (0, 1, 2))
    for path in paths:
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
        )

    def fail_preflight(supplied):
        del supplied
        raise failure

    monkeypatch.setattr(workflows, "_preflight_pitch_artifacts", fail_preflight)
    monkeypatch.setattr(
        workflows,
        "_evaluate_pitch_checkpoint",
        lambda *args, **kwargs: pytest.fail("failed preflight reached evaluation"),
    )

    with pytest.raises(error_type):
        workflows.evaluate_artifacts(paths)


def test_single_checkpoint_rejects_before_preflight_or_final_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoint"
    _write_peek_manifest(
        root,
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.CHECKPOINT,
    )
    monkeypatch.setattr(
        workflows,
        "_preflight_pitch_artifacts",
        lambda paths: pytest.fail(f"invalid cardinality reached preflight: {paths}"),
    )

    with pytest.raises(LearningContractError, match="exactly three"):
        workflows.evaluate_artifacts((root,))


def test_smoke_reconstruction_rejects_a_requested_device_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "smoke"
    _write_peek_manifest(
        root,
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.SMOKE,
    )
    artifact = object.__new__(LoadedPitchArtifact)
    object.__setattr__(artifact, "root", root.resolve())
    object.__setattr__(
        artifact,
        "manifest",
        SimpleNamespace(
            profile=ProfileName.SMOKE,
            trainer=PitchTrainerKind.PITCH,
            evaluation_device=DeviceName.CPU,
        ),
    )
    monkeypatch.setattr(workflows, "load_artifact", lambda path: artifact)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(
        workflows,
        "_evaluate_pitch_smoke",
        lambda *args, **kwargs: pytest.fail("mismatched device reconstructed smoke rows"),
    )

    with pytest.raises(LearningContractError, match="artifact evaluation device"):
        workflows.evaluate_artifacts((root,), device=DeviceName.CUDA)


def test_report_decoder_dispatches_schema_before_codec() -> None:
    report = _smoke_report()
    content = workflows.evaluation_report_bytes(report)

    assert isinstance(workflows.evaluation_report_from_bytes(content), PitchEvaluationReport)
    with pytest.raises(ValueError, match="unsupported"):
        workflows.evaluation_report_from_bytes(b'{"schema_version":999}\n')


def test_legacy_helpers_reject_unknown_trainer_without_ppo_fallthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = SimpleNamespace(manifest=SimpleNamespace(trainer="unknown", seed=0))
    monkeypatch.setitem(
        __import__("sys").modules,
        "harpy.learning.ppo",
        SimpleNamespace(
            validate_ppo_artifact=lambda value: pytest.fail("unknown trainer validated as PPO"),
            load_ppo_actor=lambda *args, **kwargs: pytest.fail("unknown trainer loaded as PPO"),
        ),
    )

    with pytest.raises(LearningContractError, match="trainer"):
        workflows._validate_artifact_payload(artifact)
    with pytest.raises(LearningContractError, match="trainer"):
        workflows._load_artifact_actor(artifact, device=DeviceName.CPU)
    with pytest.raises(LearningContractError, match="trainer"):
        workflows._artifact_order_key(artifact)
    with pytest.raises(LearningContractError, match="trainer"):
        workflows._manifest_is_scientifically_eligible(
            artifact,
            evaluation_device=DeviceName.CPU,
        )


def test_manifest_peek_rejects_symlink_before_generic_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "manifest-source.json"
    source.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": PITCH_ARTIFACT_SCHEMA_VERSION,
                "profile": ProfileName.SMOKE.value,
                "trainer": PitchTrainerKind.PITCH.value,
            }
        )
    )
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").symlink_to(source)
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"unsafe manifest reached generic loader: {path}"),
    )

    with pytest.raises(Exception, match="manifest"):
        workflows.evaluate_artifacts((root,))


def test_pitch_actor_loader_matches_an_independent_persisted_reload(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from harpy.envs.sine_pitch import SinePitchEnv
    from harpy.learning.pitch import (
        load_pitch_estimator_model,
        save_pitch_estimator_model,
    )
    from harpy.learning.pitch_actor import PitchPlannerActor

    model_path = tmp_path / "model.pt"
    model = PitchEstimatorNetwork()
    save_pitch_estimator_model(model_path, model)
    artifact = object.__new__(LoadedPitchArtifact)
    object.__setattr__(artifact, "root", tmp_path.resolve())
    object.__setattr__(
        artifact,
        "manifest",
        SimpleNamespace(files=(SimpleNamespace(relative_path="model.pt"),)),
    )
    loaded = workflows._load_pitch_artifact_actor(artifact, device=DeviceName.CPU)
    independent = PitchPlannerActor(
        load_pitch_estimator_model(model_path, device=torch.device("cpu")),
        device=torch.device("cpu"),
    )
    observation, _ = SinePitchEnv().reset(seed=123)

    assert loaded.act(observation) == independent.act(observation)


def test_final_pitch_diagnostics_preflight_once_and_consume_seed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"input-{seed}" for seed in (2, 0, 1))
    for path, seed in zip(paths, (2, 0, 1), strict=True):
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
            seed=seed,
        )
    artifacts = tuple(object.__new__(LoadedPitchArtifact) for _ in range(3))
    events: list[tuple[str, object]] = []

    def preflight(supplied):
        events.append(("preflight", tuple(supplied)))
        return artifacts

    sentinel = object()

    def diagnose(supplied, *, suite, device):
        assert events == [("preflight", paths)]
        assert tuple(supplied) == artifacts
        assert suite == "iid" and device is DeviceName.CPU
        events.append(("diagnose", tuple(supplied)))
        return sentinel

    monkeypatch.setattr(workflows, "_preflight_pitch_artifacts", preflight)
    monkeypatch.setattr(workflows, "_diagnose_pitch_artifacts", diagnose)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)

    assert workflows.diagnose_artifacts(paths, suite="iid") is sentinel
    assert tuple(event[0] for event in events) == ("preflight", "diagnose")


def test_final_cuda_pitch_diagnostics_use_e1_preflight_and_cpu_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(tmp_path / f"cuda-input-{seed}" for seed in (2, 0, 1))
    for path, seed in zip(paths, (2, 0, 1), strict=True):
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
            seed=seed,
            training_device=DeviceName.CUDA,
        )
    artifacts = tuple(object.__new__(LoadedPitchArtifact) for _ in range(3))
    cohort = SimpleNamespace(training_device=DeviceName.CUDA, artifacts=artifacts)
    events: list[tuple[str, object]] = []
    sentinel = object()

    monkeypatch.setattr(
        workflows,
        "_preflight_pitch_e1_artifacts",
        lambda supplied: events.append(("preflight", tuple(supplied))) or cohort,
    )
    monkeypatch.setattr(
        workflows,
        "_diagnose_pitch_artifacts",
        lambda supplied, *, suite, device: (
            events.append(("diagnose", (tuple(supplied), suite, device))) or sentinel
        ),
    )
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)

    assert workflows.diagnose_artifacts(paths, suite="iid") is sentinel
    assert tuple(event[0] for event in events) == ("preflight", "diagnose")
    with pytest.raises(LearningContractError, match=r"E\.1 final diagnostics must use CPU"):
        workflows.diagnose_artifacts(paths, suite="iid", device=DeviceName.CUDA)


@pytest.mark.parametrize("suite", ["ood-lower", "ood-upper"])
def test_v1_diagnostics_reject_split_ood_alias_before_loading(
    suite: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "legacy"
    _write_peek_manifest(root, schema_version=ARTIFACT_SCHEMA_VERSION)
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"legacy split alias loaded an artifact: {path}"),
    )

    with pytest.raises(LearningContractError, match="historical"):
        workflows.diagnose_artifacts((root,), suite=suite)


def test_pitch_bound_mask_and_single_final_diagnostics_reject_before_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoint"
    _write_peek_manifest(
        root,
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.CHECKPOINT,
    )
    monkeypatch.setattr(
        workflows,
        "_preflight_pitch_artifacts",
        lambda paths: pytest.fail(f"invalid diagnostic reached preflight: {paths}"),
    )
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"invalid diagnostic loaded an artifact: {path}"),
    )

    with pytest.raises(LearningContractError, match="schema-v1 BC"):
        workflows.diagnose_artifacts((root,), suite="iid", bound_mask=True)
    with pytest.raises(LearningContractError, match="exactly three"):
        workflows.diagnose_artifacts((root,), suite="iid")


def test_diagnostic_manifest_hash_uses_exact_persisted_bytes(tmp_path: Path) -> None:
    content = b'{ "trainer" : "pitch", "schema_version" : 2 }\n'
    (tmp_path / "manifest.json").write_bytes(content)

    assert workflows._artifact_manifest_sha256(tmp_path) == hashlib.sha256(content).hexdigest()


def test_pitch_diagnostic_decider_calls_decide_once_and_never_act() -> None:
    from harpy.envs.models import PitchAction
    from harpy.learning.pitch_actor import PitchDecision

    decision = PitchDecision(PitchAction.SUBMIT, 6_000)

    class Actor:
        decide_calls = 0

        def decide(self, observation):
            self.decide_calls += 1
            return decision

        def act(self, observation):
            pytest.fail(f"pitch diagnostic called act: {observation}")

    actor = Actor()
    adapter = workflows._pitch_diagnostic_decider(actor)

    result = adapter({})

    assert actor.decide_calls == 1
    assert result.action is decision.action
    assert result.estimated_candidate_cents == 6_000


def test_legacy_diagnostic_decider_calls_act_once_without_an_estimate() -> None:
    pitch_action = __import__("harpy.envs.models", fromlist=["PitchAction"]).PitchAction

    class Actor:
        act_calls = 0

        def act(self, observation):
            self.act_calls += 1
            return pitch_action.SUBMIT

        def decide(self, observation):
            pytest.fail(f"legacy diagnostic called decide: {observation}")

    actor = Actor()
    result = workflows._legacy_diagnostic_decider(actor)({})

    assert actor.act_calls == 1
    assert result.action is pitch_action.SUBMIT
    assert result.estimated_candidate_cents is None
