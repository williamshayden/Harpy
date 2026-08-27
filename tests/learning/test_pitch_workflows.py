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


def _write_peek_manifest(
    root: Path,
    *,
    schema_version: int,
    profile: ProfileName = ProfileName.SMOKE,
) -> None:
    root.mkdir()
    (root / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": schema_version,
                "profile": profile.value,
                "trainer": "pitch" if schema_version == 2 else "bc",
            }
        )
    )


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
    for path in paths:
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
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
    for path in paths:
        _write_peek_manifest(
            path,
            schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
            profile=ProfileName.CHECKPOINT,
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
