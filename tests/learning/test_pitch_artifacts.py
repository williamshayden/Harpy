"""Strict schema-v2 pitch artifact and compatibility contracts."""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import (
    TARGET_MIN_COORDINATE,
    ObservationMode,
    TerminalReason,
)
from harpy.envs.planning import minimum_action_plan
from harpy.learning import artifacts as learning_artifacts
from harpy.learning import cache as learning_cache
from harpy.learning import pitch_artifacts, pitch_data, pitch_evaluation
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_V1_REGISTRY,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactStatus,
    CriterionStatus,
    RuntimeStatus,
    SourceStatus,
    artifact_manifest_from_document,
    artifact_schema_registries,
    canonical_json_bytes,
    decode_json_bytes,
    load_artifact,
)
from harpy.learning.errors import LearningContractError, PitchArtifactSetError
from harpy.learning.evaluation import TerminalEpisodeRecord
from harpy.learning.models import (
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    SPECTRUM_GRID_ID,
    DeviceName,
    EpisodeSpec,
    ProfileName,
    TrainerKind,
)
from harpy.learning.pitch import (
    PITCH_PROFILE_CONFIGS,
    PitchDatasetMetrics,
    PitchEpochMetrics,
    PitchTrainingResult,
    PitchTrainingSummary,
    load_pitch_estimator_model,
    save_pitch_estimator_model,
)
from harpy.learning.pitch_artifacts import (
    PITCH_ARTIFACT_SCHEMA_V2_REGISTRY,
    PITCH_ARTIFACT_SCHEMA_VERSION,
    PITCH_EVALUATION_SUITE_RECORDS,
    PITCH_POLICY_SEMANTICS_ID,
    PITCH_PREPROCESSING_SCHEMA_ID,
    PITCH_REQUIRED_PAYLOAD_NAMES,
    LoadedPitchArtifact,
    PitchArtifactCompletion,
    PitchArtifactManifest,
    PitchArtifactWriter,
    PitchSmokeEvaluationDocument,
    PitchSmokePayloadKind,
    PitchTrainingConfigDocument,
    PitchTrainingCounts,
    PitchTrainingSummaryDocument,
    capture_pitch_source_status,
    load_pitch_artifact,
    pitch_compatibility_sha256,
    preflight_pitch_artifacts,
    preflight_pitch_e1_artifacts,
    read_pitch_training_config,
    read_pitch_training_summary,
)
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchEpisodeSpec,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
)
from harpy.learning.pitch_evaluation import (
    PITCH_SHUFFLED_SPECTRUM_PROBE,
    PITCH_ZERO_SPECTRUM_PROBE,
    build_pitch_evaluation_row,
)
from harpy.learning.pitch_network import (
    PITCH_ESTIMATOR_ARCHITECTURE_ID,
    PITCH_ESTIMATOR_PARAMETER_COUNT,
    PitchEstimatorNetwork,
)

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_REQUIRED_PITCH_SOURCE_INPUTS = (
    "src/harpy/__init__.py",
    "src/harpy/analysis.py",
    "src/harpy/envs/__init__.py",
    "src/harpy/envs/baselines.py",
    "src/harpy/envs/models.py",
    "src/harpy/envs/planning.py",
    "src/harpy/envs/sine_pitch.py",
    "src/harpy/envs/spectrum.py",
    "src/harpy/learning/__init__.py",
    "src/harpy/learning/action_masks.py",
    "src/harpy/learning/actors.py",
    "src/harpy/learning/artifacts.py",
    "src/harpy/learning/cache.py",
    "src/harpy/learning/dependencies.py",
    "src/harpy/learning/diagnostic_codecs.py",
    "src/harpy/learning/diagnostics.py",
    "src/harpy/learning/envs.py",
    "src/harpy/learning/errors.py",
    "src/harpy/learning/evaluation.py",
    "src/harpy/learning/models.py",
    "src/harpy/learning/observations.py",
    "src/harpy/learning/pitch_data.py",
    "src/harpy/learning/pitch_network.py",
    "src/harpy/learning/pitch_actor.py",
    "src/harpy/learning/pitch.py",
    "src/harpy/learning/pitch_artifacts.py",
    "src/harpy/learning/pitch_evaluation.py",
    "src/harpy/learning/pitch_e1.py",
    "src/harpy/learning/pitch_reports.py",
    "src/harpy/learning/suites.py",
    "src/harpy/learning/workflows.py",
    "src/harpy/synth/__init__.py",
    "src/harpy/synth/curves.py",
    "src/harpy/synth/engine.py",
    "src/harpy/synth/envelope.py",
    "src/harpy/synth/models.py",
    "src/harpy/tuning.py",
    "uv.lock",
)


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    )


def _pitch_source_repo(root: Path, *, missing: str | None = None) -> Path:
    for relative_path in _REQUIRED_PITCH_SOURCE_INPUTS:
        if relative_path == missing:
            continue
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative_path}\n", encoding="utf-8")
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test User")
    _run_git(root, "add", ".")
    _run_git(root, "commit", "-qm", "fixture")
    return root


def _source(
    *,
    clean: bool = True,
    committed: bool = True,
    commit: str = "1" * 40,
    lock: str = _DIGEST_B,
) -> SourceStatus:
    return SourceStatus(
        commit=commit,
        dirty_tree=not clean,
        tracked_diff_sha256=_EMPTY_SHA256 if clean else _DIGEST_A,
        dependency_lock_sha256=lock,
        required_inputs_committed=committed,
    )


def _runtime(device: DeviceName = DeviceName.CPU) -> RuntimeStatus:
    return RuntimeStatus(
        python_version="3.12.11",
        platform="Linux-test",
        processor="x86_64",
        numpy_version="2.3.2",
        gymnasium_version="1.3.0",
        torch_version=torch.__version__,
        stable_baselines3_version="2.9.0",
        device=device,
        device_description=(
            "CPU" if device is DeviceName.CPU else "Test CUDA; compute capability 8.9; 16 SMs"
        ),
        cuda_runtime_version=None if device is DeviceName.CPU else "13.0",
        cuda_driver_version=None if device is DeviceName.CPU else "596.08",
    )


def _dataset_metrics(count: int, *, loss: float) -> PitchDatasetMetrics:
    return PitchDatasetMetrics(
        example_count=count,
        loss=loss,
        mean_absolute_error_cents=1.0,
        median_absolute_error_cents=1.0,
        within_one_count=count,
        within_one_rate=1.0,
        within_five_count=count,
        within_five_rate=1.0,
    )


def _summary(
    profile: ProfileName,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> PitchTrainingSummary:
    configured = PITCH_PROFILE_CONFIGS[profile]
    return PitchTrainingSummary(
        history=(
            PitchEpochMetrics(
                epoch=1,
                training=_dataset_metrics(configured.training_coordinate_count, loss=0.75),
                validation=_dataset_metrics(configured.validation_coordinate_count, loss=0.5),
            ),
        ),
        selected_epoch=1,
        training_examples=configured.training_coordinate_count,
        validation_examples=configured.validation_coordinate_count,
        seed=seed,
        device=device.value,
        deterministic_algorithms=True,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        data_loader_num_workers=0,
        shuffle_generator_device="cpu",
        training_wall_time_seconds=1.25,
    )


def _summary_document(
    profile: ProfileName,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> PitchTrainingSummaryDocument:
    return PitchTrainingSummaryDocument(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        trainer=PitchTrainerKind.PITCH,
        profile=profile,
        seed=seed,
        summary=_summary(profile, seed, device),
    )


def _config(
    profile: ProfileName,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> PitchTrainingConfigDocument:
    return PitchTrainingConfigDocument(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        trainer=PitchTrainerKind.PITCH,
        profile=profile,
        seed=seed,
        device=device,
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=PITCH_DISTRIBUTION_ID,
        coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        evaluation_suites=PITCH_EVALUATION_SUITE_RECORDS,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
        policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
        parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
        profile_config=PITCH_PROFILE_CONFIGS[profile],
        compatibility_sha256=pitch_compatibility_sha256(profile),
    )


def _counts(profile: ProfileName, *, complete: bool) -> PitchTrainingCounts:
    configured = PITCH_PROFILE_CONFIGS[profile]
    return PitchTrainingCounts(
        configured_training_coordinates=configured.training_coordinate_count,
        configured_validation_coordinates=configured.validation_coordinate_count,
        training_examples=(configured.training_coordinate_count if complete else None),
        validation_examples=(configured.validation_coordinate_count if complete else None),
    )


def _manifest(
    profile: ProfileName,
    seed: int,
    *,
    source: SourceStatus | None = None,
    device: DeviceName = DeviceName.CPU,
) -> PitchArtifactManifest:
    return PitchArtifactManifest(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        status=ArtifactStatus.INCOMPLETE,
        trainer=PitchTrainerKind.PITCH,
        profile=profile,
        seed=seed,
        created_at_utc="2026-08-27T12:00:00Z",
        completed_at_utc=None,
        source=_source() if source is None else source,
        runtime=_runtime(device),
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=PITCH_DISTRIBUTION_ID,
        coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        evaluation_suites=PITCH_EVALUATION_SUITE_RECORDS,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
        policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
        parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
        selected_epoch=None,
        training_counts=_counts(profile, complete=False),
        evaluation_device=None,
        eligible_for_aggregate=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        compatibility_sha256=pitch_compatibility_sha256(profile),
        files=(),
    )


def _terminal_records() -> tuple[TerminalEpisodeRecord, ...]:
    suite = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.SMOKE)
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=EpisodeSpec(record.target_note_index, record.source_pitch_cents),
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=True,
            final_absolute_error_cents=0,
            action_count=len(
                minimum_action_plan(
                    record.source_pitch_cents
                    - 100 * (TARGET_MIN_COORDINATE + record.target_note_index)
                )
            ),
            excess_actions=0,
            invalid_action_count=0,
            total_return=1.0,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        )
        for index, record in enumerate(suite.episodes)
    )


def _smoke_documents(
    profile: ProfileName,
    seed: int,
) -> tuple[PitchSmokeEvaluationDocument, PitchSmokeEvaluationDocument]:
    suite = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.SMOKE)
    records = _terminal_records()
    summary = _summary(profile, seed)
    common = {"suite": suite, "records": records}
    learned = build_pitch_evaluation_row(
        actor_id=f"pitch-{seed}",
        trainer=PitchTrainerKind.PITCH,
        seed=seed,
        environment_id=ENVIRONMENT_ID,
        observation_mode=ObservationMode.SPECTRUM,
        parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
        training_examples=summary.training_examples,
        training_wall_time_seconds=summary.training_wall_time_seconds,
        **common,
    )
    probes = tuple(
        build_pitch_evaluation_row(
            actor_id=f"pitch-{seed}",
            trainer=PitchTrainerKind.PITCH,
            seed=seed,
            environment_id=ENVIRONMENT_ID,
            observation_mode=ObservationMode.SPECTRUM,
            probe=probe,
            parameter_count=PITCH_ESTIMATOR_PARAMETER_COUNT,
            training_examples=summary.training_examples,
            training_wall_time_seconds=summary.training_wall_time_seconds,
            **common,
        )
        for probe in (PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE)
    )
    baseline_order = (
        BaselineKind.RANDOM,
        BaselineKind.REWARD_SEARCH,
        BaselineKind.SPECTRUM_PEAK,
        BaselineKind.ORACLE,
    )
    baselines = tuple(
        build_pitch_evaluation_row(
            actor_id=kind.value,
            trainer=None,
            seed=None,
            environment_id=kind.environment_id,
            observation_mode=kind.observation_mode,
            **common,
        )
        for kind in baseline_order
    )
    suite_digest = PITCH_EVALUATION_SUITE_RECORDS[0][1]
    return (
        PitchSmokeEvaluationDocument(
            PITCH_ARTIFACT_SCHEMA_VERSION,
            PitchSmokePayloadKind.BASE,
            PitchEvaluationSuiteId.SMOKE,
            suite_digest,
            (learned, *baselines),
        ),
        PitchSmokeEvaluationDocument(
            PITCH_ARTIFACT_SCHEMA_VERSION,
            PitchSmokePayloadKind.PROBES,
            PitchEvaluationSuiteId.SMOKE,
            suite_digest,
            probes,
        ),
    )


def _publish(
    writer: PitchArtifactWriter,
    profile: ProfileName,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> None:
    base, probes = _smoke_documents(profile, seed)
    writer.publish_json("training-config.json", _config(profile, seed, device).to_document())
    writer.publish_json(
        "training-summary.json", _summary_document(profile, seed, device).to_document()
    )
    writer.publish_model(lambda path: save_pitch_estimator_model(path, PitchEstimatorNetwork()))
    writer.publish_json("evaluation-smoke.json", base.to_document())
    writer.publish_json("evaluation-smoke-probes.json", probes.to_document())


def _complete(
    root: Path,
    profile: ProfileName,
    seed: int,
    *,
    source: SourceStatus | None = None,
    training_device: DeviceName = DeviceName.CPU,
    evaluation_device: DeviceName = DeviceName.CPU,
) -> LoadedPitchArtifact:
    writer = PitchArtifactWriter.begin(
        root,
        _manifest(profile, seed, source=source, device=training_device),
    )
    _publish(writer, profile, seed, training_device)
    return writer.complete(_completion(profile, evaluation_device=evaluation_device))


def _rewrite_payload_and_manifest(root: Path, filename: str, content: bytes) -> None:
    payload = root / filename
    payload.write_bytes(content)
    manifest_path = root / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    for record in document["files"]:  # type: ignore[index]
        if record["relative_path"] == filename:
            record["size_bytes"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()
            break
    else:
        raise AssertionError(f"missing manifest record for {filename}")
    manifest_path.write_bytes(canonical_json_bytes(document))


def _completion(
    profile: ProfileName = ProfileName.SMOKE,
    *,
    evaluation_device: DeviceName = DeviceName.CPU,
) -> PitchArtifactCompletion:
    return PitchArtifactCompletion(
        completed_at_utc="2026-08-27T12:01:00Z",
        training_counts=_counts(profile, complete=True),
        evaluation_device=evaluation_device,
    )


def test_schema_v2_registry_is_closed_and_separate_from_schema_v1() -> None:
    assert ARTIFACT_SCHEMA_VERSION == 1
    assert PITCH_ARTIFACT_SCHEMA_VERSION == 2
    assert PITCH_ARTIFACT_SCHEMA_V2_REGISTRY.trainers == ("pitch",)
    assert tuple(TrainerKind) == (TrainerKind.BC, TrainerKind.PPO)
    assert tuple(PitchTrainerKind) == (PitchTrainerKind.PITCH,)
    registries = artifact_schema_registries()
    assert registries == {
        ARTIFACT_SCHEMA_VERSION: ARTIFACT_SCHEMA_V1_REGISTRY,
        PITCH_ARTIFACT_SCHEMA_VERSION: PITCH_ARTIFACT_SCHEMA_V2_REGISTRY,
    }
    with pytest.raises(TypeError):
        registries[PITCH_ARTIFACT_SCHEMA_VERSION] = ARTIFACT_SCHEMA_V1_REGISTRY  # type: ignore[index]


def test_pitch_inventory_is_exactly_six_files_including_manifest() -> None:
    assert ("manifest.json", *PITCH_REQUIRED_PAYLOAD_NAMES) == (
        "manifest.json",
        "training-config.json",
        "training-summary.json",
        "model.pt",
        "evaluation-smoke.json",
        "evaluation-smoke-probes.json",
    )


def test_pitch_provenance_requires_the_exact_schema_v2_source_inputs() -> None:
    assert pitch_artifacts._PITCH_REQUIRED_SOURCE_INPUTS == _REQUIRED_PITCH_SOURCE_INPUTS


def test_e1_preflight_admits_only_homogeneous_cuda_checkpoint_cohort(tmp_path: Path) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"cuda-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA,
        )
        for seed in (2, 0, 1)
    )

    with pytest.raises(PitchArtifactSetError, match="eligible CPU"):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))

    cohort = preflight_pitch_e1_artifacts(
        tuple(item.root for item in artifacts),
        evaluator_source=_source(),
    )

    assert cohort.training_device is DeviceName.CUDA
    assert tuple(item.manifest.seed for item in cohort.artifacts) == (0, 1, 2)
    assert all(not item.manifest.eligible_for_aggregate for item in cohort.artifacts)


def test_e1_preflight_rejects_mixed_training_devices(tmp_path: Path) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"mixed-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA if seed != 1 else DeviceName.CPU,
        )
        for seed in (0, 1, 2)
    )

    with pytest.raises(PitchArtifactSetError, match="homogeneous training device"):
        preflight_pitch_e1_artifacts(
            tuple(item.root for item in artifacts),
            evaluator_source=_source(),
        )


def test_e1_stale_evaluator_source_fails_before_semantic_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"stale-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA,
        )
        for seed in (0, 1, 2)
    )
    monkeypatch.setattr(
        pitch_artifacts,
        "_semantic_evaluation_documents",
        lambda *args, **kwargs: pytest.fail("stale source reached semantic evidence"),
    )
    monkeypatch.setattr(
        pitch_artifacts,
        "load_pitch_estimator_model",
        lambda *args, **kwargs: pytest.fail("stale source reached model load"),
    )

    with pytest.raises(PitchArtifactSetError, match="evaluator source"):
        preflight_pitch_e1_artifacts(
            tuple(item.root for item in artifacts),
            evaluator_source=_source(commit="2" * 40),
        )


def test_e1_preflight_rejects_runtime_field_disagreement(tmp_path: Path) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"runtime-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA,
        )
        for seed in (0, 1, 2)
    )
    manifest_path = artifacts[2].root / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    document["runtime"]["cuda_driver_version"] = "596.09"  # type: ignore[index]
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(PitchArtifactSetError, match="runtime cohort"):
        preflight_pitch_e1_artifacts(
            tuple(item.root for item in artifacts),
            evaluator_source=_source(),
        )


@pytest.mark.parametrize("seeds", [(0, 1, 1), (0, 1, 3)])
def test_e1_preflight_rejects_duplicate_or_wrong_seed_set(
    tmp_path: Path,
    seeds: tuple[int, int, int],
) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"seed-{index}-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA,
        )
        for index, seed in enumerate(seeds)
    )

    with pytest.raises(PitchArtifactSetError, match="seeds must be exactly"):
        preflight_pitch_e1_artifacts(
            tuple(item.root for item in artifacts),
            evaluator_source=_source(),
        )


def test_e1_preflight_rejects_compatibility_disagreement(tmp_path: Path) -> None:
    artifacts = tuple(
        _complete(
            tmp_path / f"compatibility-{seed}",
            ProfileName.CHECKPOINT,
            seed,
            training_device=DeviceName.CUDA,
        )
        for seed in (0, 1, 2)
    )
    manifest_path = artifacts[2].root / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    document["compatibility_sha256"] = _DIGEST_A
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="compatibility"):
        preflight_pitch_e1_artifacts(
            tuple(item.root for item in artifacts),
            evaluator_source=_source(),
        )


def test_cuda_runtime_capture_records_hardware_and_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Test GPU",
        major=8,
        minor=9,
        multi_processor_count=16,
    )
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            current_device=lambda: 0,
            get_device_properties=lambda index: properties,
        ),
    )
    monkeypatch.setattr(
        pitch_artifacts,
        "require_training_dependencies",
        lambda: SimpleNamespace(
            torch=fake_torch,
            torch_version="2.13.0+cu130",
            stable_baselines3_version="2.9.0",
        ),
    )
    monkeypatch.setattr(pitch_artifacts, "_cuda_driver_version", lambda: "596.08")

    runtime = pitch_artifacts._capture_pitch_runtime_status(DeviceName.CUDA)

    assert runtime.device_description == "Test GPU; compute capability 8.9; 16 SMs"
    assert runtime.cuda_runtime_version == "13.0"
    assert runtime.cuda_driver_version == "596.08"


def test_cuda_availability_probe_observes_workspace_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def is_available() -> bool:
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return True

    monkeypatch.setattr(
        pitch_artifacts,
        "require_training_dependencies",
        lambda: SimpleNamespace(
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=is_available))
        ),
    )

    seed, _stack = pitch_artifacts._validate_pitch_artifact_training_inputs(
        ProfileName.CHECKPOINT,
        0,
        DeviceName.CUDA,
        tmp_path / "new-artifact",
    )

    assert seed == 0


@pytest.mark.parametrize(
    "missing",
    [
        "src/harpy/analysis.py",
        "src/harpy/learning/action_masks.py",
        "src/harpy/learning/artifacts.py",
        "src/harpy/learning/dependencies.py",
        "src/harpy/learning/evaluation.py",
        "src/harpy/learning/pitch_evaluation.py",
        "src/harpy/synth/engine.py",
        "src/harpy/tuning.py",
    ],
)
def test_pitch_source_capture_rejects_uncommitted_required_semantics(
    tmp_path: Path,
    missing: str,
) -> None:
    root = _pitch_source_repo(tmp_path / missing.rsplit("/", 1)[-1], missing=missing)

    status = capture_pitch_source_status(root / "src/harpy/learning/pitch_artifacts.py")

    assert status.required_inputs_committed is False


def test_pitch_source_capture_accepts_all_committed_semantics(tmp_path: Path) -> None:
    root = _pitch_source_repo(tmp_path / "complete")

    status = capture_pitch_source_status(root / "src/harpy/learning/pitch_artifacts.py")

    assert status.required_inputs_committed is True


def test_dirty_required_spectrum_source_makes_checkpoint_ineligible(tmp_path: Path) -> None:
    root = _pitch_source_repo(tmp_path / "dirty-spectrum")
    spectrum = root / "src/harpy/envs/spectrum.py"
    spectrum.write_text("dirty spectrum semantics\n", encoding="utf-8")
    source = capture_pitch_source_status(root / "src/harpy/learning/pitch_artifacts.py")

    artifact = _complete(
        tmp_path / "artifact",
        ProfileName.CHECKPOINT,
        0,
        source=source,
    )

    assert source.dirty_tree is True
    assert source.required_inputs_committed is True
    assert artifact.manifest.eligible_for_aggregate is False


def test_compatibility_digest_is_profile_specific_and_pinned() -> None:
    assert pitch_compatibility_sha256(ProfileName.SMOKE) == (
        "7f08894d0dc4aa41d137d277372fca34bdd2e42567c80aa7c3d1b48e80322aa4"
    )
    assert pitch_compatibility_sha256(ProfileName.CHECKPOINT) == (
        "092f3fa97545e07397685cf7686d972a9a00fe6dffd4151b68d53c8b3b8121cc"
    )


def test_v2_config_and_summary_round_trip_canonical_bytes() -> None:
    config = _config(ProfileName.CHECKPOINT, 2)
    summary = _summary_document(ProfileName.CHECKPOINT, 2)
    assert PitchTrainingConfigDocument.from_document(config.to_document()) == config
    assert PitchTrainingSummaryDocument.from_document(summary.to_document()) == summary
    assert canonical_json_bytes(config.to_document()).endswith(b"\n")


@pytest.mark.parametrize(
    "document_kind",
    ["manifest", "config", "summary", "smoke_base", "smoke_probes"],
)
@pytest.mark.parametrize("invalid_schema_version", [2.0, True])
def test_v2_documents_reject_noninteger_schema_versions(
    document_kind: str,
    invalid_schema_version: object,
) -> None:
    base, probes = _smoke_documents(ProfileName.SMOKE, 0)
    instances = {
        "manifest": _manifest(ProfileName.SMOKE, 0),
        "config": _config(ProfileName.SMOKE, 0),
        "summary": _summary_document(ProfileName.SMOKE, 0),
        "smoke_base": base,
        "smoke_probes": probes,
    }
    instance = instances[document_kind]
    decoder = type(instance).from_document

    with pytest.raises(ValueError, match="schema_version"):
        replace(instance, schema_version=invalid_schema_version)

    document = instance.to_document()
    document["schema_version"] = invalid_schema_version  # type: ignore[assignment]
    with pytest.raises(ValueError, match="schema_version"):
        decoder(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trainer", "bc"),
        ("profile", "unknown"),
        ("coordinate_split_digest_sha256", _DIGEST_A),
        ("preprocessing_schema_id", "legacy"),
        ("architecture_schema_id", "legacy"),
        ("policy_semantics_id", "legacy"),
        ("parameter_count", PITCH_ESTIMATOR_PARAMETER_COUNT + 1),
        ("compatibility_sha256", _DIGEST_A),
    ],
)
def test_v2_config_rejects_cross_schema_and_identity_mutations(
    field: str,
    value: object,
) -> None:
    document = _config(ProfileName.SMOKE, 0).to_document()
    document[field] = value  # type: ignore[literal-required]
    with pytest.raises(ValueError):
        PitchTrainingConfigDocument.from_document(document)


def test_schema_specific_decoders_reject_cross_schema_documents() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        PitchArtifactManifest.from_document({"schema_version": 1})
    with pytest.raises(ValueError, match="schema_version"):
        PitchTrainingConfigDocument.from_document({"schema_version": 1})
    with pytest.raises(ValueError, match="schema_version"):
        ArtifactManifest.from_document({"schema_version": 2})
    assert isinstance(
        artifact_manifest_from_document(_manifest(ProfileName.SMOKE, 0).to_document()),
        PitchArtifactManifest,
    )


def test_manifest_round_trip_and_incomplete_lifecycle_are_strict() -> None:
    manifest = _manifest(ProfileName.CHECKPOINT, 0)
    assert PitchArtifactManifest.from_document(manifest.to_document()) == manifest
    for changes in (
        {"selected_epoch": 1},
        {"completed_at_utc": "2026-08-27T12:01:00Z"},
        {"eligible_for_aggregate": True},
        {"criterion_status": CriterionStatus.ELIGIBLE_FOR_AGGREGATE},
        {"training_counts": _counts(ProfileName.CHECKPOINT, complete=True)},
    ):
        with pytest.raises(ValueError, match="incomplete"):
            replace(manifest, **changes)


def test_smoke_codecs_rederive_rows_and_reject_metric_mutation() -> None:
    base, probes = _smoke_documents(ProfileName.SMOKE, 0)
    assert PitchSmokeEvaluationDocument.from_document(base.to_document()) == base
    assert PitchSmokeEvaluationDocument.from_document(probes.to_document()) == probes
    document = base.to_document()
    document["rows"][0]["metrics"]["submitted_success_rate"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="re-derived"):
        PitchSmokeEvaluationDocument.from_document(document)


def test_writer_bootstrap_is_create_only_and_incomplete(tmp_path: Path) -> None:
    output = tmp_path / "pitch"
    manifest = _manifest(ProfileName.SMOKE, 0)
    writer = PitchArtifactWriter.begin(output, manifest)
    assert (output / "manifest.json").read_bytes() == canonical_json_bytes(manifest.to_document())
    with pytest.raises(FileExistsError):
        PitchArtifactWriter.begin(output, manifest)
    with pytest.raises(ValueError, match="incomplete"):
        load_pitch_artifact(output)
    assert writer.file_records == ()


def test_writer_rejects_unknown_duplicate_and_wrong_kind_payloads(tmp_path: Path) -> None:
    writer = PitchArtifactWriter.begin(tmp_path / "pitch", _manifest(ProfileName.SMOKE, 0))
    with pytest.raises(ValueError, match="inventory"):
        writer.publish_json("unknown.json", {"schema_version": 2})
    writer.publish_json("training-config.json", _config(ProfileName.SMOKE, 0).to_document())
    with pytest.raises(FileExistsError):
        writer.publish_json("training-config.json", _config(ProfileName.SMOKE, 0).to_document())
    with pytest.raises(ValueError, match="kind"):
        writer.publish_json("model.pt", {"schema_version": 2})


def test_complete_artifact_has_exact_hashes_model_and_schema_dispatch(tmp_path: Path) -> None:
    artifact = _complete(tmp_path / "pitch", ProfileName.SMOKE, 0)
    assert {path.name for path in artifact.root.iterdir()} == {
        "manifest.json",
        *PITCH_REQUIRED_PAYLOAD_NAMES,
    }
    for record in artifact.manifest.files:
        content = artifact.file(record.relative_path).read_bytes()
        assert record.size_bytes == len(content)
        assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert isinstance(load_pitch_estimator_model(artifact.file("model.pt")), PitchEstimatorNetwork)
    assert load_pitch_artifact(artifact.root).manifest == artifact.manifest
    assert load_artifact(artifact.root).manifest == artifact.manifest  # type: ignore[union-attr]
    assert read_pitch_training_config(artifact) == _config(ProfileName.SMOKE, 0)
    assert read_pitch_training_summary(artifact) == _summary_document(ProfileName.SMOKE, 0)


@pytest.mark.parametrize("mutation", ["unknown", "model", "json", "manifest"])
def test_loader_rejects_unknown_corrupted_or_cross_schema_payloads(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact = _complete(tmp_path / mutation, ProfileName.SMOKE, 0)
    if mutation == "unknown":
        (artifact.root / "unknown.bin").write_bytes(b"x")
    elif mutation == "model":
        (artifact.root / "model.pt").write_bytes(b"not torch")
    elif mutation == "json":
        (artifact.root / "training-config.json").write_bytes(b'{"schema_version":NaN}\n')
    else:
        document = decode_json_bytes((artifact.root / "manifest.json").read_bytes())
        document["trainer"] = "bc"
        (artifact.root / "manifest.json").write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError):
        load_pitch_artifact(artifact.root)


@pytest.mark.parametrize(
    "filename",
    [
        "training-config.json",
        "training-summary.json",
        "evaluation-smoke.json",
        "evaluation-smoke-probes.json",
    ],
)
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"schema_version":2,"schema_version":2}\n', "duplicate"),
        (b'{"schema_version":2,"value":NaN}\n', "finite"),
    ],
)
def test_loader_rejects_hash_consistent_duplicate_or_nonfinite_json(
    tmp_path: Path,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    artifact = _complete(tmp_path / f"{filename}-{message}", ProfileName.SMOKE, 0)
    _rewrite_payload_and_manifest(artifact.root, filename, content)

    with pytest.raises(ValueError, match=message):
        load_pitch_artifact(artifact.root)


@pytest.mark.parametrize("mutation", ["missing", "extra", "nonfinite"])
def test_loader_strictly_validates_hash_consistent_model_state(
    tmp_path: Path,
    mutation: str,
) -> None:
    artifact = _complete(tmp_path / mutation, ProfileName.SMOKE, 0)
    model_path = artifact.root / "model.pt"
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    first_name = next(iter(state))
    if mutation == "missing":
        state.pop(first_name)
    elif mutation == "extra":
        state["unexpected.weight"] = torch.zeros(1, dtype=torch.float32)
    else:
        tensor = state[first_name].clone()
        tensor.reshape(-1)[0] = torch.nan
        state[first_name] = tensor
    torch.save(state, model_path)
    manifest_path = artifact.root / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    for record in document["files"]:  # type: ignore[index]
        if record["relative_path"] == "model.pt":
            content = model_path.read_bytes()
            record["size_bytes"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(LearningContractError, match="state is invalid"):
        load_pitch_artifact(artifact.root)


@pytest.mark.parametrize(
    ("profile", "seed", "source", "training_device", "evaluation_device", "eligible"),
    [
        (ProfileName.CHECKPOINT, 0, _source(), DeviceName.CPU, DeviceName.CPU, True),
        (ProfileName.CHECKPOINT, 2, _source(), DeviceName.CPU, DeviceName.CPU, True),
        (ProfileName.CHECKPOINT, 3, _source(), DeviceName.CPU, DeviceName.CPU, False),
        (ProfileName.SMOKE, 0, _source(), DeviceName.CPU, DeviceName.CPU, False),
        (ProfileName.CHECKPOINT, 0, _source(clean=False), DeviceName.CPU, DeviceName.CPU, False),
        (
            ProfileName.CHECKPOINT,
            0,
            _source(committed=False),
            DeviceName.CPU,
            DeviceName.CPU,
            False,
        ),
        (ProfileName.CHECKPOINT, 0, _source(), DeviceName.CUDA, DeviceName.CPU, False),
        (ProfileName.CHECKPOINT, 0, _source(), DeviceName.CPU, DeviceName.CUDA, False),
    ],
)
def test_completion_eligibility_is_exactly_derived(
    tmp_path: Path,
    profile: ProfileName,
    seed: int,
    source: SourceStatus,
    training_device: DeviceName,
    evaluation_device: DeviceName,
    eligible: bool,
) -> None:
    artifact = _complete(
        tmp_path / f"{profile.value}-{seed}-{training_device.value}-{evaluation_device.value}",
        profile,
        seed,
        source=source,
        training_device=training_device,
        evaluation_device=evaluation_device,
    )
    assert artifact.manifest.eligible_for_aggregate is eligible
    assert artifact.manifest.criterion_met is None
    assert artifact.manifest.criterion_status is (
        CriterionStatus.ELIGIBLE_FOR_AGGREGATE if eligible else CriterionStatus.INELIGIBLE
    )


def test_exact_three_preflight_returns_seed_order(tmp_path: Path) -> None:
    artifacts = tuple(
        _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed) for seed in (2, 0, 1)
    )
    assert tuple(
        item.manifest.seed
        for item in preflight_pitch_artifacts(tuple(item.root for item in artifacts))
    ) == (0, 1, 2)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "extra",
        "wrong_seed",
        "ineligible",
        "dirty",
        "uncommitted",
        "cuda_training",
        "cuda_evaluation",
        "source",
        "lock",
    ],
)
def test_exact_three_preflight_rejects_every_identity_failure(
    tmp_path: Path,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if failure == "missing":
        artifacts = tuple(
            _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed) for seed in (0, 1)
        )
    elif failure == "duplicate":
        first = _complete(tmp_path / "seed-0", ProfileName.CHECKPOINT, 0)
        artifacts = (first, first, _complete(tmp_path / "seed-2", ProfileName.CHECKPOINT, 2))
    elif failure == "extra":
        artifacts = tuple(
            _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed)
            for seed in (0, 1, 2, 3)
        )
    elif failure == "wrong_seed":
        artifacts = tuple(
            _complete(tmp_path / f"run-{index}", ProfileName.CHECKPOINT, seed)
            for index, seed in enumerate((0, 1, 3))
        )
    elif failure == "ineligible":
        artifacts = (
            _complete(tmp_path / "seed-0", ProfileName.SMOKE, 0),
            _complete(tmp_path / "seed-1", ProfileName.CHECKPOINT, 1),
            _complete(tmp_path / "seed-2", ProfileName.CHECKPOINT, 2),
        )
    elif failure == "dirty":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                source=_source(clean=seed != 0),
            )
            for seed in (0, 1, 2)
        )
    elif failure == "uncommitted":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                source=_source(committed=seed != 0),
            )
            for seed in (0, 1, 2)
        )
    elif failure == "cuda_training":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                training_device=DeviceName.CUDA if seed == 0 else DeviceName.CPU,
            )
            for seed in (0, 1, 2)
        )
    elif failure == "cuda_evaluation":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                evaluation_device=DeviceName.CUDA if seed == 0 else DeviceName.CPU,
            )
            for seed in (0, 1, 2)
        )
    elif failure == "source":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                source=_source(commit=(str(seed + 1) * 40)),
            )
            for seed in (0, 1, 2)
        )
    elif failure == "lock":
        artifacts = tuple(
            _complete(
                tmp_path / f"seed-{seed}",
                ProfileName.CHECKPOINT,
                seed,
                source=_source(lock=(str(seed + 1) * 64)),
            )
            for seed in (0, 1, 2)
        )
    else:
        raise AssertionError(f"unhandled failure case {failure}")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight crossed the construction boundary")

    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", forbidden)
    monkeypatch.setattr(pitch_evaluation, "fixed_pitch_evaluation_suite", forbidden)
    monkeypatch.setattr(learning_cache, "SpectrumEvidenceCache", forbidden)
    with pytest.raises(PitchArtifactSetError):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))


def test_aggregate_compatibility_identity_gate_rejects_a_mixed_set() -> None:
    shared = ("1" * 40, _DIGEST_A, True, _DIGEST_A)
    incompatible = ("1" * 40, _DIGEST_A, True, _DIGEST_B)

    with pytest.raises(PitchArtifactSetError, match="share source, lock, and compatibility"):
        pitch_artifacts._validate_pitch_aggregate_identities((shared, shared, incompatible))


@pytest.mark.parametrize(
    "filename",
    ["evaluation-smoke.json", "evaluation-smoke-probes.json"],
)
@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b'{"schema_version":2,"schema_version":2}\n', "duplicate"),
        (b'{"schema_version":2,"value":NaN}\n', "finite"),
    ],
)
def test_preflight_raw_decodes_all_json_before_identity_or_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    content: bytes,
    message: str,
) -> None:
    artifacts = tuple(
        _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed) for seed in (0, 1, 3)
    )
    _rewrite_payload_and_manifest(artifacts[0].root, filename, content)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight crossed the raw metadata boundary")

    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", forbidden)
    monkeypatch.setattr(pitch_evaluation, "fixed_pitch_evaluation_suite", forbidden)
    monkeypatch.setattr(learning_cache, "SpectrumEvidenceCache", forbidden)
    with pytest.raises(ValueError, match=message):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))


@pytest.mark.parametrize(
    "filename",
    [
        "training-config.json",
        "training-summary.json",
        "evaluation-smoke.json",
        "evaluation-smoke-probes.json",
    ],
)
def test_preflight_semantically_validates_every_json_before_any_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    artifacts = tuple(
        _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed) for seed in (0, 1, 2)
    )
    payload = artifacts[2].root / filename
    document = decode_json_bytes(payload.read_bytes())
    if filename in {"training-config.json", "training-summary.json"}:
        document["seed"] = 99
    else:
        document["rows"][0]["metrics"]["submitted_success_rate"] = 0.0  # type: ignore[index]
    _rewrite_payload_and_manifest(
        artifacts[2].root,
        filename,
        canonical_json_bytes(document),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("model loading started before all JSON was valid")

    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", forbidden)
    with pytest.raises(ValueError):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))


@pytest.mark.parametrize("bad_seed", [0, 1, 2])
def test_preflight_validates_all_json_before_loading_each_model_in_seed_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_seed: int,
) -> None:
    artifacts = tuple(
        _complete(tmp_path / f"seed-{seed}", ProfileName.CHECKPOINT, seed) for seed in (0, 1, 2)
    )
    by_root = {artifact.root: artifact.manifest.seed for artifact in artifacts}
    bad_artifact = next(item for item in artifacts if item.manifest.seed == bad_seed)
    _rewrite_payload_and_manifest(bad_artifact.root, "model.pt", b"not a torch payload\n")

    semantic_seeds: list[int] = []
    model_seeds: list[int] = []
    real_semantic_validation = pitch_artifacts._semantic_evaluation_documents
    real_model_loader = pitch_artifacts.load_pitch_estimator_model

    def tracked_semantic_validation(
        artifact: LoadedPitchArtifact,
        *args: object,
        **kwargs: object,
    ) -> object:
        semantic_seeds.append(artifact.manifest.seed)
        return real_semantic_validation(artifact, *args, **kwargs)  # type: ignore[arg-type]

    def tracked_model_loader(path: Path, *args: object, **kwargs: object) -> object:
        assert semantic_seeds == [0, 1, 2]
        model_seeds.append(by_root[path.parent])
        return real_model_loader(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        pitch_artifacts,
        "_semantic_evaluation_documents",
        tracked_semantic_validation,
    )
    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", tracked_model_loader)

    with pytest.raises(LearningContractError, match="could not be loaded"):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))
    assert semantic_seeds == [0, 1, 2]
    assert model_seeds == list(range(bad_seed + 1))


def test_preflight_rejects_bad_seeds_before_suite_or_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tuple(
        _complete(tmp_path / f"run-{index}", ProfileName.CHECKPOINT, seed)
        for index, seed in enumerate((0, 1, 3))
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("preflight crossed the construction boundary")

    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", forbidden)
    monkeypatch.setattr(pitch_evaluation, "fixed_pitch_evaluation_suite", forbidden)
    with pytest.raises(PitchArtifactSetError, match="seeds"):
        preflight_pitch_artifacts(tuple(item.root for item in artifacts))


@pytest.mark.parametrize("failure", ["temp_open", "fsync", "replace", "interrupt"])
def test_pitch_bootstrap_pre_manifest_failures_remove_temp_only_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    output = tmp_path / "pitch"
    if failure in {"temp_open", "interrupt"}:
        real_open = Path.open

        def fail_manifest_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if path.parent == output and path.name.startswith(".manifest.json."):
                if failure == "interrupt":
                    raise KeyboardInterrupt
                raise OSError("injected manifest temporary open failure")
            return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "open", fail_manifest_open)
    elif failure == "fsync":

        def fail_fsync(_descriptor: int) -> None:
            raise OSError("injected manifest fsync failure")

        monkeypatch.setattr(learning_artifacts.os, "fsync", fail_fsync)
    else:

        def fail_replace(_source: Path, _destination: Path) -> None:
            raise OSError("injected manifest replace failure")

        monkeypatch.setattr(learning_artifacts.os, "replace", fail_replace)

    expected_error = KeyboardInterrupt if failure == "interrupt" else OSError
    with pytest.raises(expected_error):
        PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))

    assert not output.exists()


def test_pitch_bootstrap_interrupt_after_first_manifest_leaves_valid_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    real_replace = os.replace

    def replace_then_interrupt(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(learning_artifacts.os, "replace", replace_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))

    assert {path.name for path in output.iterdir()} == {"manifest.json"}
    persisted = PitchArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE


def test_pitch_json_publication_link_failure_leaves_clean_incomplete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("injected payload link failure")

    monkeypatch.setattr(learning_artifacts.os, "link", fail_link)

    with pytest.raises(OSError, match="payload link"):
        writer.publish_json("training-config.json", _config(ProfileName.SMOKE, 0).to_document())

    assert {path.name for path in output.iterdir()} == {"manifest.json"}
    assert writer.file_records == ()
    assert (
        PitchArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        ).status
        is ArtifactStatus.INCOMPLETE
    )


@pytest.mark.parametrize("payload_kind", ["json", "model"])
def test_pitch_publication_record_failure_keeps_unrecorded_incomplete_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_kind: str,
) -> None:
    output = tmp_path / payload_kind
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))

    def fail_record(_path: Path, _relative_path: str) -> object:
        raise OSError("injected record failure")

    monkeypatch.setattr(pitch_artifacts, "_record_file", fail_record)
    if payload_kind == "json":
        filename = "training-config.json"
        publish = lambda: writer.publish_json(  # noqa: E731
            filename,
            _config(ProfileName.SMOKE, 0).to_document(),
        )
    else:
        filename = "model.pt"
        publish = lambda: writer.publish_model(  # noqa: E731
            lambda path: save_pitch_estimator_model(path, PitchEstimatorNetwork())
        )

    with pytest.raises(OSError, match="record failure"):
        publish()

    assert (output / filename).is_file()
    assert writer.file_records == ()
    assert (
        PitchArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        ).status
        is ArtifactStatus.INCOMPLETE
    )


@pytest.mark.parametrize("payload_kind", ["json", "model"])
def test_pitch_publication_never_overwrites_a_raced_final_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload_kind: str,
) -> None:
    output = tmp_path / payload_kind
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    filename = "training-config.json" if payload_kind == "json" else "model.pt"
    final = output / filename
    raced_content = f"raced-{payload_kind}\n".encode()
    if payload_kind == "json":
        real_write = learning_artifacts._write_fsynced_file

        def stage_then_race(path: Path, content: bytes) -> None:
            real_write(path, content)
            final.write_bytes(raced_content)

        monkeypatch.setattr(learning_artifacts, "_write_fsynced_file", stage_then_race)
        publish = lambda: writer.publish_json(  # noqa: E731
            filename,
            _config(ProfileName.SMOKE, 0).to_document(),
        )
    else:

        def save_then_race(path: Path) -> None:
            save_pitch_estimator_model(path, PitchEstimatorNetwork())
            final.write_bytes(raced_content)

        publish = lambda: writer.publish_model(save_then_race)  # noqa: E731

    with pytest.raises(FileExistsError):
        publish()

    assert final.read_bytes() == raced_content
    assert writer.file_records == ()
    assert {path.name for path in output.iterdir()} == {"manifest.json", filename}


def test_model_saver_failure_preserves_recoverable_incomplete_artifact(tmp_path: Path) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))

    def fail(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        writer.publish_model(fail)
    persisted = PitchArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert persisted.status is ArtifactStatus.INCOMPLETE
    assert not (output / "model.pt").exists()


def test_pending_pitch_view_is_owner_bound_hash_checked_and_invalidated(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)
    pending = writer.pending_view(("training-config.json", "training-summary.json", "model.pt"))

    assert read_pitch_training_config(pending) == _config(ProfileName.SMOKE, 0)
    assert read_pitch_training_summary(pending) == _summary_document(ProfileName.SMOKE, 0)
    with pytest.raises(ValueError, match="published"):
        writer.pending_view(("unknown.json",))

    loaded = writer.complete(_completion())

    assert loaded.manifest.status is ArtifactStatus.COMPLETE
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.file("model.pt")
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")
    with pytest.raises(RuntimeError, match="complete"):
        writer.pending_view(("model.pt",))


def test_pending_pitch_view_rejects_tamper_and_destroyed_owner(tmp_path: Path) -> None:
    tamper_output = tmp_path / "tamper"
    tamper_writer = PitchArtifactWriter.begin(
        tamper_output,
        _manifest(ProfileName.SMOKE, 0),
    )
    tamper_writer.publish_model(
        lambda path: save_pitch_estimator_model(path, PitchEstimatorNetwork())
    )
    (tamper_output / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match=r"size|hash"):
        tamper_writer.pending_view(("model.pt",))

    owner_output = tmp_path / "owner"
    owner = PitchArtifactWriter.begin(owner_output, _manifest(ProfileName.SMOKE, 0))
    owner.publish_json("training-config.json", _config(ProfileName.SMOKE, 0).to_document())
    pending = owner.pending_view(("training-config.json",))
    del owner
    gc.collect()

    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")


@pytest.mark.parametrize("fault", ["missing", "unexpected", "temporary", "tampered"])
def test_pitch_completion_rejects_nonclosed_or_unhashed_inventory(
    tmp_path: Path,
    fault: str,
) -> None:
    output = tmp_path / fault
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)
    if fault == "missing":
        (output / "evaluation-smoke.json").unlink()
    elif fault == "unexpected":
        (output / "notes.txt").write_bytes(b"unexpected\n")
    elif fault == "temporary":
        (output / f".model.{'a' * 32}.tmp.pt").write_bytes(b"temporary")
    else:
        (output / "model.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"inventory|missing|unexpected|size|hash"):
        writer.complete(_completion())

    assert (
        PitchArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        ).status
        is ArtifactStatus.INCOMPLETE
    )


def test_pitch_completion_rejects_persisted_bootstrap_mutation(tmp_path: Path) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)
    manifest_path = output / "manifest.json"
    document = decode_json_bytes(manifest_path.read_bytes())
    document["seed"] = 9
    manifest_path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(ValueError, match="bootstrap"):
        writer.complete(_completion())

    assert decode_json_bytes(manifest_path.read_bytes())["status"] == "incomplete"


def test_pitch_final_manifest_replace_failure_preserves_incomplete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected final manifest replace failure")

    monkeypatch.setattr(pitch_artifacts.os, "replace", fail_replace)

    with pytest.raises(OSError, match="final manifest"):
        writer.complete(_completion())

    assert not any(path.name.startswith(".manifest.json.") for path in output.iterdir())
    assert (
        PitchArtifactManifest.from_document(
            decode_json_bytes((output / "manifest.json").read_bytes())
        ).status
        is ArtifactStatus.INCOMPLETE
    )
    assert writer.pending_view(("model.pt",)).file("model.pt").is_file()


def test_pitch_interrupt_after_consumed_final_replace_returns_committed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)
    pending = writer.pending_view(("training-config.json", "model.pt"))
    real_replace = os.replace

    def replace_then_interrupt(source: Path, destination: Path) -> None:
        real_replace(source, destination)
        raise KeyboardInterrupt

    monkeypatch.setattr(pitch_artifacts.os, "replace", replace_then_interrupt)

    artifact = writer.complete(_completion())

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert load_pitch_artifact(output) == artifact
    assert not any(path.name.startswith(".manifest.json.") for path in output.iterdir())
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.file("model.pt")
    with pytest.raises(RuntimeError, match="complete"):
        writer.pending_view(("model.pt",))


def test_pitch_completion_runs_no_fallible_work_after_atomic_manifest_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    writer = PitchArtifactWriter.begin(output, _manifest(ProfileName.SMOKE, 0))
    _publish(writer, ProfileName.SMOKE, 0)
    pending = writer.pending_view(("training-config.json", "model.pt"))
    committed = False
    real_replace = pitch_artifacts.os.replace
    guarded_functions = {
        "load_pitch_estimator_model": pitch_artifacts.load_pitch_estimator_model,
        "_fsync_directory": pitch_artifacts._fsync_directory,
        "_validate_config_manifest": pitch_artifacts._validate_config_manifest,
        "_validate_summary_manifest": pitch_artifacts._validate_summary_manifest,
        "canonical_json_bytes": pitch_artifacts.canonical_json_bytes,
    }

    for name, function in guarded_functions.items():

        def guarded(*args: object, _function: object = function, **kwargs: object) -> object:
            if committed:
                raise AssertionError("fallible work ran after final manifest replacement")
            assert callable(_function)
            return _function(*args, **kwargs)

        monkeypatch.setattr(pitch_artifacts, name, guarded)

    def forbidden_public_reload(_root: Path) -> LoadedPitchArtifact:
        raise AssertionError("completion must return its prevalidated object")

    monkeypatch.setattr(pitch_artifacts, "load_pitch_artifact", forbidden_public_reload)

    def tracked_replace(source: Path, destination: Path) -> None:
        nonlocal committed
        real_replace(source, destination)
        committed = True

    monkeypatch.setattr(pitch_artifacts.os, "replace", tracked_replace)

    artifact = writer.complete(_completion())

    assert committed
    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert decode_json_bytes((output / "manifest.json").read_bytes())["status"] == "complete"
    with pytest.raises(RuntimeError, match="no longer live"):
        pending.document("training-config.json")


def _install_fast_pitch_artifact_workflow(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    events: list[str],
    *,
    fail_stage: str | None = None,
    real_probe_factory: bool = False,
) -> dict[str, object]:
    """Install deterministic workflow seams while retaining real artifact codecs."""

    training_model = PitchEstimatorNetwork()
    with torch.no_grad():
        for parameter in training_model.parameters():
            parameter.fill_(0.125)
    selected_state = {
        name: tensor.detach().cpu().clone() for name, tensor in training_model.state_dict().items()
    }
    actor_models: list[object] = []
    persisted_models: list[object] = []
    cache_ids: list[int] = []
    real_load = pitch_artifacts.load_pitch_estimator_model
    real_publish_json = PitchArtifactWriter.publish_json

    def capture_source(path: Path) -> SourceStatus:
        assert path.name == "pitch_artifacts.py"
        events.append("source")
        return _source()

    def capture_runtime(device: DeviceName) -> RuntimeStatus:
        events.append(f"runtime:{device.value}")
        return _runtime(device)

    def build_datasets(
        profile: ProfileName,
        *,
        cache: learning_cache.SpectrumEvidenceCache | None = None,
        evidence_provider: object | None = None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        assert evidence_provider is None
        assert isinstance(cache, learning_cache.SpectrumEvidenceCache)
        assert decode_json_bytes((output / "manifest.json").read_bytes())["status"] == (
            "incomplete"
        )
        assert {path.name for path in output.iterdir()} == {"manifest.json"}
        events.append("build")
        cache_ids.append(id(cache))
        configured = PITCH_PROFILE_CONFIGS[profile]
        return (
            tuple(range(configured.training_coordinate_count)),
            tuple(range(configured.validation_coordinate_count)),
        )

    def train(
        training_dataset: object,
        validation_dataset: object,
        *,
        profile: object,
        seed: int,
        device: torch.device,
    ) -> PitchTrainingResult:
        events.append(f"train:{device.type}")
        if fail_stage == "train":
            raise RuntimeError("injected pitch training failure")
        profile_name = next(
            name for name, configured in PITCH_PROFILE_CONFIGS.items() if configured == profile
        )
        assert len(training_dataset) == (  # type: ignore[arg-type]
            PITCH_PROFILE_CONFIGS[profile_name].training_coordinate_count
        )
        assert len(validation_dataset) == (  # type: ignore[arg-type]
            PITCH_PROFILE_CONFIGS[profile_name].validation_coordinate_count
        )
        return PitchTrainingResult(
            model=training_model,
            selected_state=selected_state,
            summary=_summary(profile_name, seed, DeviceName(device.type)),
        )

    def tracked_load(
        path: Path,
        *,
        device: torch.device | None = None,
    ) -> PitchEstimatorNetwork:
        events.append(f"load:{path.name}:{'default' if device is None else device.type}")
        model = real_load(path, device=device)
        if path.name == "model.pt":
            persisted_models.append(model)
        return model

    class Actor:
        def __init__(self, model: object, *, device: object | None = None) -> None:
            assert device == torch.device("cpu")
            actor_models.append(model)
            events.append("actor:cpu")

    def make_environment(cache: learning_cache.SpectrumEvidenceCache) -> object:
        cache_ids.append(id(cache))
        events.append("env")
        return object()

    def probe_factory(environment_factory: object, probe: str) -> object:
        assert callable(environment_factory)
        events.append(f"probe-factory:{probe}")

        def factory() -> object:
            return environment_factory()

        factory._pitch_test_probe = probe  # type: ignore[attr-defined]
        return factory

    def evaluate_learned(
        actor: object,
        suite: object,
        *,
        environment_factory: object,
    ) -> tuple[TerminalEpisodeRecord, ...]:
        del actor
        assert suite.suite_id is PitchEvaluationSuiteId.SMOKE  # type: ignore[union-attr]
        probe = getattr(environment_factory, "_pitch_test_probe", None)
        lane = "base" if probe is None else probe
        events.append(f"evaluate:{lane}")
        if fail_stage == "evaluate" and probe is None:
            raise RuntimeError("injected pitch evaluation failure")
        if not real_probe_factory:
            assert callable(environment_factory)
            environment_factory()
        assert (output / "model.pt").is_file()
        assert not (output / "evaluation-smoke.json").exists()
        return _terminal_records()

    def evaluate_baseline(
        kind: BaselineKind,
        suite: object,
        *,
        cache: learning_cache.SpectrumEvidenceCache,
    ) -> tuple[TerminalEpisodeRecord, ...]:
        assert suite.suite_id is PitchEvaluationSuiteId.SMOKE  # type: ignore[union-attr]
        cache_ids.append(id(cache))
        events.append(f"baseline:{kind.value}")
        return _terminal_records()

    def tracked_publish_json(
        self: PitchArtifactWriter,
        filename: str,
        document: object,
    ) -> object:
        events.append(f"publish:{filename}")
        return real_publish_json(self, filename, document)  # type: ignore[arg-type]

    monkeypatch.setattr(pitch_artifacts, "capture_pitch_source_status", capture_source)
    monkeypatch.setattr(pitch_artifacts, "_capture_pitch_runtime_status", capture_runtime)
    monkeypatch.setattr(pitch_artifacts, "build_pitch_coordinate_datasets", build_datasets)
    monkeypatch.setattr(pitch_artifacts, "train_pitch_estimator", train)
    monkeypatch.setattr(pitch_artifacts, "load_pitch_estimator_model", tracked_load)
    monkeypatch.setattr(pitch_artifacts, "PitchPlannerActor", Actor)
    monkeypatch.setattr(pitch_artifacts, "make_cached_sine_pitch_env", make_environment)
    if not real_probe_factory:
        monkeypatch.setattr(pitch_artifacts, "_pitch_probe_factory", probe_factory)
    monkeypatch.setattr(pitch_artifacts, "evaluate_pitch_learned_actor", evaluate_learned)
    monkeypatch.setattr(pitch_artifacts, "evaluate_pitch_baseline_suite", evaluate_baseline)
    monkeypatch.setattr(PitchArtifactWriter, "publish_json", tracked_publish_json)
    return {
        "training_model": training_model,
        "actor_models": actor_models,
        "persisted_models": persisted_models,
        "cache_ids": cache_ids,
    }


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("profile", "smoke"),
        ("seed", True),
        ("seed", 1.0),
        ("seed", -1),
        ("device", "cpu"),
        ("output", "pitch"),
    ],
)
def test_train_pitch_artifact_rejects_invalid_inputs_before_creating_output(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    output = tmp_path / "pitch"
    arguments: dict[str, object] = {
        "profile": ProfileName.SMOKE,
        "seed": 0,
        "device": DeviceName.CPU,
        "output": output,
    }
    arguments[field] = invalid

    with pytest.raises(ValueError):
        pitch_artifacts.train_pitch_artifact(**arguments)  # type: ignore[arg-type]

    assert not output.exists()


def test_train_pitch_artifact_is_create_only_and_rejects_unavailable_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        pitch_artifacts.train_pitch_artifact(
            profile=ProfileName.SMOKE,
            seed=0,
            device=DeviceName.CPU,
            output=existing,
        )

    unavailable = SimpleNamespace(
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setattr(pitch_artifacts, "require_training_dependencies", lambda: unavailable)
    output = tmp_path / "cuda"
    with pytest.raises(ValueError, match=r"CUDA.*unavailable"):
        pitch_artifacts.train_pitch_artifact(
            profile=ProfileName.SMOKE,
            seed=0,
            device=DeviceName.CUDA,
            output=output,
        )
    assert not output.exists()


def test_train_pitch_artifact_publishes_strict_lifecycle_from_persisted_cpu_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    events: list[str] = []
    observed = _install_fast_pitch_artifact_workflow(monkeypatch, output, events)

    artifact = pitch_artifacts.train_pitch_artifact(
        profile=ProfileName.SMOKE,
        seed=7,
        device=DeviceName.CPU,
        output=output,
    )

    assert isinstance(artifact, LoadedPitchArtifact)
    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact.manifest.eligible_for_aggregate is False
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        *PITCH_REQUIRED_PAYLOAD_NAMES,
    }
    assert events[:6] == [
        "source",
        "runtime:cpu",
        "build",
        "train:cpu",
        "publish:training-config.json",
        "publish:training-summary.json",
    ]
    assert events.index("load:model.pt:cpu") < events.index("actor:cpu")
    assert events.index("actor:cpu") < events.index("evaluate:base")
    assert events.index("evaluate:base") < events.index(f"evaluate:{PITCH_ZERO_SPECTRUM_PROBE}")
    assert events.index(f"evaluate:{PITCH_SHUFFLED_SPECTRUM_PROBE}") < events.index(
        "baseline:random"
    )
    assert [event for event in events if event.startswith("baseline:")] == [
        "baseline:random",
        "baseline:reward_search",
        "baseline:spectrum_peak",
        "baseline:oracle",
    ]
    assert events.index("publish:evaluation-smoke.json") < events.index(
        "publish:evaluation-smoke-probes.json"
    )
    actor_models = observed["actor_models"]
    persisted_models = observed["persisted_models"]
    assert isinstance(actor_models, list) and isinstance(persisted_models, list)
    assert len(actor_models) == 1
    assert actor_models[0] is persisted_models[0]
    assert actor_models[0] is not observed["training_model"]
    cache_ids = observed["cache_ids"]
    assert isinstance(cache_ids, list) and len(set(cache_ids)) == 1


def test_cuda_training_still_reloads_and_evaluates_on_cpu_and_is_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch-cuda"
    events: list[str] = []
    observed = _install_fast_pitch_artifact_workflow(monkeypatch, output, events)
    available = SimpleNamespace(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            device=torch.device,
        )
    )
    monkeypatch.setattr(pitch_artifacts, "require_training_dependencies", lambda: available)

    artifact = pitch_artifacts.train_pitch_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=0,
        device=DeviceName.CUDA,
        output=output,
    )

    assert "train:cuda" in events
    assert "load:model.pt:cpu" in events
    assert "actor:cpu" in events
    assert artifact.manifest.runtime.device is DeviceName.CUDA
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact.manifest.eligible_for_aggregate is False
    assert artifact.manifest.criterion_status is CriterionStatus.INELIGIBLE
    actor_models = observed["actor_models"]
    persisted_models = observed["persisted_models"]
    assert isinstance(actor_models, list) and isinstance(persisted_models, list)
    assert actor_models == persisted_models[:1]


def test_checkpoint_training_never_constructs_final_suites_or_direct_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "checkpoint"
    events: list[str] = []
    _install_fast_pitch_artifact_workflow(
        monkeypatch,
        output,
        events,
        real_probe_factory=True,
    )
    pitch_artifacts._pitch_shuffle_permutation.cache_clear()
    pitch_data._fixed_pitch_evaluation_suite.cache_clear()
    real_fixed = pitch_data.fixed_pitch_evaluation_suite
    accessed: list[PitchEvaluationSuiteId] = []
    constructed_suite_codes: list[int] = []
    real_suite_episodes = pitch_data._suite_episodes

    def record_suite_construction(
        *, suite_code: int, source_pool: tuple[int, ...], per_target: int
    ) -> tuple[PitchEpisodeSpec, ...]:
        constructed_suite_codes.append(suite_code)
        return real_suite_episodes(
            suite_code=suite_code,
            source_pool=source_pool,
            per_target=per_target,
        )

    def smoke_only(suite_id: PitchEvaluationSuiteId) -> object:
        accessed.append(suite_id)
        if suite_id is not PitchEvaluationSuiteId.SMOKE:
            raise AssertionError("training accessed a final evaluation suite")
        return real_fixed(suite_id)

    def forbidden_direct(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("training accessed final direct-coordinate evaluation")

    monkeypatch.setattr(pitch_artifacts, "fixed_pitch_evaluation_suite", smoke_only)
    monkeypatch.setattr(pitch_data, "fixed_pitch_evaluation_suite", smoke_only)
    monkeypatch.setattr(pitch_data, "_suite_episodes", record_suite_construction)
    monkeypatch.setattr(pitch_evaluation, "fixed_pitch_evaluation_suite", smoke_only)
    monkeypatch.setattr(
        pitch_evaluation,
        "evaluate_final_pitch_coordinates",
        forbidden_direct,
    )

    artifact = pitch_artifacts.train_pitch_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=0,
        device=DeviceName.CPU,
        output=output,
    )

    assert artifact.manifest.profile is ProfileName.CHECKPOINT
    assert set(accessed) == {PitchEvaluationSuiteId.SMOKE}
    assert constructed_suite_codes == [401]
    assert artifact.manifest.eligible_for_aggregate is True


def test_artifact_smoke_shuffle_matches_the_pinned_v2_probe_without_suite_access() -> None:
    pitch_artifacts._pitch_shuffle_permutation.cache_clear()

    assert (
        pitch_artifacts._pitch_shuffle_permutation()
        == pitch_data.iid_shuffled_spectrum_permutation()
    )


@pytest.mark.parametrize(
    ("fail_stage", "expected_payloads"),
    [
        ("train", set()),
        ("evaluate", {"training-config.json", "training-summary.json", "model.pt"}),
    ],
)
def test_train_pitch_artifact_failure_preserves_recoverable_incomplete_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_stage: str,
    expected_payloads: set[str],
) -> None:
    output = tmp_path / fail_stage
    events: list[str] = []
    _install_fast_pitch_artifact_workflow(
        monkeypatch,
        output,
        events,
        fail_stage=fail_stage,
    )

    with pytest.raises(RuntimeError, match="injected pitch"):
        pitch_artifacts.train_pitch_artifact(
            profile=ProfileName.SMOKE,
            seed=0,
            device=DeviceName.CPU,
            output=output,
        )

    manifest = PitchArtifactManifest.from_document(
        decode_json_bytes((output / "manifest.json").read_bytes())
    )
    assert manifest.status is ArtifactStatus.INCOMPLETE
    assert {path.name for path in output.iterdir()} == {"manifest.json", *expected_payloads}


def test_train_pitch_artifact_returns_completion_immediately_as_last_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "pitch"
    events: list[str] = []
    _install_fast_pitch_artifact_workflow(monkeypatch, output, events)
    sentinel = object()

    def complete(
        self: PitchArtifactWriter,
        completion: PitchArtifactCompletion,
    ) -> object:
        assert completion.evaluation_device is DeviceName.CPU
        assert set(self.file_records)  # type: ignore[arg-type]
        events.append("complete")
        return sentinel

    monkeypatch.setattr(PitchArtifactWriter, "complete", complete)

    result = pitch_artifacts.train_pitch_artifact(
        profile=ProfileName.SMOKE,
        seed=0,
        device=DeviceName.CPU,
        output=output,
    )

    assert result is sentinel
    assert events[-1] == "complete"
