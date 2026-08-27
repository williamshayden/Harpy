"""Deterministic behavior-cloning training and artifact workflow contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from harpy.envs.models import MAX_STEPS, ControlState, ObservationMode, PitchAction, TerminalReason
from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning import bc
from harpy.learning.actors import BCActor, MaskedBCActor
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCompletion,
    ArtifactManifest,
    ArtifactStatus,
    ArtifactWriter,
    BCTrainingCounts,
    CriterionStatus,
    LoadedArtifact,
    PendingArtifactView,
    RuntimeStatus,
    SourceStatus,
    TrainingConfigDocument,
    TrainingSummaryDocument,
    load_artifact,
    read_json_document,
    required_payload_names,
)
from harpy.learning.bc import (
    BCExample,
    OracleTrajectoryDataset,
    load_bc_actor,
    load_masked_bc_actor,
    next_action_accuracy,
    save_bc_model,
    train_bc_artifact,
    train_behavior_cloning,
    validate_bc_artifact,
)
from harpy.learning.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    EvaluationFile,
    TerminalEpisodeRecord,
    build_evaluation_rows,
)
from harpy.learning.models import (
    ARCHITECTURE_SCHEMA_ID,
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    PREPROCESSING_SCHEMA_ID,
    PROFILE_CONFIGS,
    SPECTRUM_GRID_ID,
    BCEpochMetrics,
    BCProfile,
    BCTrainingSummary,
    DeviceName,
    EpisodeSpec,
    EvaluationSuiteId,
    ProfileName,
    TrainerKind,
)
from harpy.learning.network import BCPolicyNetwork
from harpy.learning.suites import (
    TRAIN_DISTRIBUTION_ID,
    build_bc_episode_splits,
    fixed_evaluation_suite,
)


class _ConstantEvidence:
    def spectrum_for_cents(self, effective_pitch_cents: int) -> np.ndarray:
        del effective_pitch_cents
        return np.linspace(0.0, 1.0, LOG_SPECTRUM_SIZE, dtype=np.float32)


class _FixedLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(observations["state"].shape[0], -1)


def _dataset(actions: Sequence[PitchAction]) -> OracleTrajectoryDataset:
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=5_151)
    examples = tuple(
        BCExample(
            episode=episode,
            controls=ControlState(cents=(index % 21) - 10),
            steps_remaining=MAX_STEPS - index,
            action=action,
        )
        for index, action in enumerate(actions)
    )
    return OracleTrajectoryDataset(examples, _ConstantEvidence())  # type: ignore[arg-type]


def _profile(
    *,
    max_epochs: int = 2,
    patience: int | None = None,
    batch_size: int = 4,
) -> BCProfile:
    return BCProfile(
        train_episodes=1,
        validation_episodes=1,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        batch_size=batch_size,
        optimizer="AdamW",
        learning_rate=3e-4,
        weight_decay=1e-4,
    )


def _all_classes(repetitions: int = 1) -> tuple[PitchAction, ...]:
    return tuple(action for _ in range(repetitions) for action in PitchAction)


def _suite_records(profile: ProfileName) -> tuple[tuple[EvaluationSuiteId, str], ...]:
    return tuple(
        (suite_id, fixed_evaluation_suite(suite_id).digest_sha256)
        for suite_id in PROFILE_CONFIGS[profile].evaluation_suites
    )


def _source_status() -> SourceStatus:
    return SourceStatus(
        commit="1" * 40,
        dirty_tree=True,
        tracked_diff_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
        required_inputs_committed=True,
    )


def _runtime_status(device: DeviceName = DeviceName.CPU) -> RuntimeStatus:
    return RuntimeStatus(
        python_version="3.12.11",
        platform="Linux-test",
        processor="x86_64",
        numpy_version=np.__version__,
        gymnasium_version="1.3.0",
        torch_version=torch.__version__,
        stable_baselines3_version="2.9.0",
        device=device,
        device_description="CPU" if device is DeviceName.CPU else "Test CUDA",
        cuda_runtime_version=None if device is DeviceName.CPU else "12.8",
        cuda_driver_version=None,
    )


def _training_summary(profile: ProfileName) -> BCTrainingSummary:
    configured = PROFILE_CONFIGS[profile].bc
    return BCTrainingSummary(
        history=(
            BCEpochMetrics(
                epoch=1,
                training_loss=0.75,
                validation_loss=0.5,
                validation_accuracy=0.625,
            ),
        ),
        selected_epoch=1,
        training_examples=configured.train_episodes + 100,
        validation_examples=configured.validation_episodes + 50,
        training_wall_time_seconds=1.25,
    )


def _incomplete_manifest(
    profile: ProfileName,
    *,
    seed: int = 0,
    parameter_count: int | None = None,
    device: DeviceName = DeviceName.CPU,
) -> ArtifactManifest:
    configured = PROFILE_CONFIGS[profile].bc
    count = sum(parameter.numel() for parameter in BCPolicyNetwork().parameters())
    return ArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        status=ArtifactStatus.INCOMPLETE,
        trainer=TrainerKind.BC,
        profile=profile,
        seed=seed,
        created_at_utc="2026-08-10T12:00:00Z",
        completed_at_utc=None,
        source=_source_status(),
        runtime=_runtime_status(device),
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=tuple(
            (suite_id.value, digest) for suite_id, digest in _suite_records(profile)
        ),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        parameter_count=count if parameter_count is None else parameter_count,
        training_counts=BCTrainingCounts(
            configured_training_episodes=configured.train_episodes,
            configured_validation_episodes=configured.validation_episodes,
            training_examples=None,
            validation_examples=None,
        ),
        evaluation_device=None,
        criterion_eligible=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        files=(),
    )


def _evaluation_file(
    suite_id: EvaluationSuiteId,
    *,
    include_probes: bool = True,
) -> EvaluationFile:
    suite = fixed_evaluation_suite(suite_id)
    records = tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=True,
            final_absolute_error_cents=0,
            action_count=1,
            excess_actions=0,
            invalid_action_count=0,
            total_return=1.0,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        )
        for index, episode in enumerate(suite.episodes)
    )
    row_arguments = {
        "actor_id": "bc-0",
        "trainer": TrainerKind.BC,
        "seed": 0,
        "environment_id": ENVIRONMENT_ID,
        "observation_mode": ObservationMode.SPECTRUM,
        "suite": suite,
        "records": records,
        "parameter_count": sum(parameter.numel() for parameter in BCPolicyNetwork().parameters()),
        "training_examples": _training_summary(
            ProfileName.SMOKE if suite_id is EvaluationSuiteId.SMOKE else ProfileName.CHECKPOINT
        ).training_examples,
        "training_wall_time_seconds": 1.25,
    }
    rows = list(build_evaluation_rows(**row_arguments))  # type: ignore[arg-type]
    if include_probes and suite_id is not EvaluationSuiteId.REGISTER_OOD:
        for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
            rows.extend(
                build_evaluation_rows(
                    **row_arguments,  # type: ignore[arg-type]
                    probe=probe,
                )
            )
    return EvaluationFile(
        schema_version=EVALUATION_SCHEMA_VERSION,
        suite_id=suite_id,
        suite_digest_sha256=suite.digest_sha256,
        rows=tuple(rows),
        next_action_accuracy=0.625 if suite_id is not EvaluationSuiteId.REGISTER_OOD else None,
    )


def _single_lane_evaluation_file(suite_id: EvaluationSuiteId) -> EvaluationFile:
    """Build a Task 5-valid file that is incomplete for the Task 7 workflow."""
    return _evaluation_file(suite_id, include_probes=False)


def _writer_with_core(
    output: Path,
    *,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    model_state: dict[str, torch.Tensor] | None = None,
    parameter_count: int | None = None,
) -> tuple[ArtifactWriter, PendingArtifactView]:
    writer = ArtifactWriter.begin(
        output,
        _incomplete_manifest(profile, seed=seed, parameter_count=parameter_count),
    )
    splits = build_bc_episode_splits(profile=profile, run_seed=seed)
    writer.publish_json(
        "training-config.json",
        TrainingConfigDocument(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            trainer=TrainerKind.BC,
            profile=profile,
            seed=seed,
            device=DeviceName.CPU,
            environment_id=ENVIRONMENT_ID,
            environment_contract_id=ENVIRONMENT_CONTRACT_ID,
            train_distribution_id=TRAIN_DISTRIBUTION_ID,
            evaluation_suites=_suite_records(profile),
            spectrum_grid_id=SPECTRUM_GRID_ID,
            preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
            architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
            profile_config=PROFILE_CONFIGS[profile].bc,
            bc_training_digest_sha256=splits.training_digest_sha256,
            bc_validation_digest_sha256=splits.validation_digest_sha256,
        ).to_document(),
    )
    writer.publish_json(
        "training-summary.json",
        TrainingSummaryDocument(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            trainer=TrainerKind.BC,
            profile=profile,
            seed=seed,
            summary=_training_summary(profile),
        ).to_document(),
    )
    if model_state is None:
        model = BCPolicyNetwork()
        writer.publish_model("model.pt", lambda path: save_bc_model(path, model))
    else:
        writer.publish_model("model.pt", lambda path: torch.save(model_state, path))
    return writer, writer.pending_view(
        ("training-config.json", "training-summary.json", "model.pt")
    )


def _completion(profile: ProfileName) -> ArtifactCompletion:
    summary = _training_summary(profile)
    configured = PROFILE_CONFIGS[profile].bc
    return ArtifactCompletion(
        completed_at_utc="2026-08-10T12:01:00Z",
        training_counts=BCTrainingCounts(
            configured_training_episodes=configured.train_episodes,
            configured_validation_episodes=configured.validation_episodes,
            training_examples=summary.training_examples,
            validation_examples=summary.validation_examples,
        ),
        evaluation_device=DeviceName.CPU,
        bc_criterion_met=None,
    )


def _clean_source_status() -> SourceStatus:
    return SourceStatus(
        commit="2" * 40,
        dirty_tree=False,
        tracked_diff_sha256=("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        dependency_lock_sha256="b" * 64,
        required_inputs_committed=True,
    )


def _compact_examples(episodes: Sequence[EpisodeSpec]) -> tuple[BCExample, ...]:
    return tuple(
        BCExample(
            episode=episode,
            controls=ControlState(),
            steps_remaining=MAX_STEPS,
            action=tuple(PitchAction)[index % len(PitchAction)],
        )
        for index, episode in enumerate(episodes)
    )


def _terminal_records(
    suite: object,
    *,
    success: bool,
) -> tuple[TerminalEpisodeRecord, ...]:
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
            submitted_success=success,
            within_5_cents=success,
            within_1_cent=success,
            final_absolute_error_cents=0 if success else 100,
            action_count=1,
            excess_actions=0 if success else None,
            invalid_action_count=0,
            total_return=1.0 if success else -1.0,
            terminal_reason=(
                TerminalReason.SUBMITTED_SUCCESS if success else TerminalReason.SUBMITTED_FAILURE
            ),
        )
        for index, episode in enumerate(suite.episodes)  # type: ignore[union-attr]
    )


def _install_fast_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    evaluation_success: bool = True,
) -> None:
    def capture_source(path: Path) -> SourceStatus:
        assert path.name == "bc.py"
        events.append("source")
        return _clean_source_status()

    def capture_runtime(device: DeviceName) -> RuntimeStatus:
        events.append(f"runtime:{device.value}")
        return _runtime_status(device)

    def examples(
        episodes: Sequence[EpisodeSpec],
        *,
        env_factory: object,
    ) -> tuple[BCExample, ...]:
        del env_factory
        events.append(f"examples:{len(episodes)}")
        return _compact_examples(episodes)

    def train(
        training_dataset: OracleTrajectoryDataset,
        validation_dataset: OracleTrajectoryDataset,
        *,
        config: BCProfile,
        seed: int,
        device: torch.device,
    ) -> tuple[BCPolicyNetwork, BCTrainingSummary]:
        assert (
            config
            is PROFILE_CONFIGS[
                ProfileName.SMOKE
                if config.train_episodes == PROFILE_CONFIGS[ProfileName.SMOKE].bc.train_episodes
                else ProfileName.CHECKPOINT
            ].bc
        )
        events.append(f"train:{device.type}")
        model = BCPolicyNetwork()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01 * (seed + 1))
        return model, BCTrainingSummary(
            history=(
                BCEpochMetrics(
                    epoch=1,
                    training_loss=0.4,
                    validation_loss=0.3,
                    validation_accuracy=0.9,
                ),
            ),
            selected_epoch=1,
            training_examples=len(training_dataset),
            validation_examples=len(validation_dataset),
            training_wall_time_seconds=1.25,
        )

    def accuracy(
        model: BCPolicyNetwork,
        dataset: OracleTrajectoryDataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> float:
        del model, dataset, batch_size
        events.append(f"accuracy:{device.type}")
        return 0.95

    def evaluate(
        actor: object,
        suite: object,
        *,
        environment_factory: object,
    ) -> tuple[TerminalEpisodeRecord, ...]:
        del actor, environment_factory
        events.append(f"evaluate:{suite.suite_id.value}")  # type: ignore[union-attr]
        return _terminal_records(suite, success=evaluation_success)

    monkeypatch.setattr(bc, "capture_source_status", capture_source)
    monkeypatch.setattr(bc, "_capture_runtime_status", capture_runtime)
    monkeypatch.setattr(bc, "build_oracle_examples", examples)
    monkeypatch.setattr(bc, "train_behavior_cloning", train)
    monkeypatch.setattr(bc, "next_action_accuracy", accuracy)
    monkeypatch.setattr(bc, "evaluate_learned_actor", evaluate)


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_cpu_training_is_seeded_deterministic_and_uses_pinned_loader_optimizer_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _dataset(_all_classes(2))
    validation = _dataset(tuple(reversed(_all_classes(1))))
    dataloader_calls: list[dict[str, object]] = []
    optimizer_calls: list[dict[str, object]] = []
    real_dataloader = torch.utils.data.DataLoader
    real_adamw = torch.optim.AdamW

    def recording_dataloader(*args: object, **kwargs: object) -> object:
        dataloader_calls.append(dict(kwargs))
        return real_dataloader(*args, **kwargs)

    def recording_adamw(*args: object, **kwargs: object) -> object:
        optimizer_calls.append(dict(kwargs))
        return real_adamw(*args, **kwargs)

    monkeypatch.setattr(bc.torch.utils.data, "DataLoader", recording_dataloader)
    monkeypatch.setattr(bc.torch.optim, "AdamW", recording_adamw)

    first_model, first_summary = train_behavior_cloning(
        training,
        validation,
        config=_profile(),
        seed=73,
        device=torch.device("cpu"),
    )
    second_model, second_summary = train_behavior_cloning(
        training,
        validation,
        config=_profile(),
        seed=73,
        device=torch.device("cpu"),
    )

    assert first_summary.history == second_summary.history
    assert first_summary.selected_epoch == second_summary.selected_epoch
    assert first_summary.training_examples == second_summary.training_examples == 14
    assert first_summary.validation_examples == second_summary.validation_examples == 7
    assert all(
        torch.equal(first_model.state_dict()[name], second_model.state_dict()[name])
        for name in first_model.state_dict()
    )
    assert len(dataloader_calls) == 4
    for offset in (0, 2):
        training_loader = dataloader_calls[offset]
        validation_loader = dataloader_calls[offset + 1]
        assert training_loader["shuffle"] is True
        assert training_loader["num_workers"] == 0
        assert isinstance(training_loader["generator"], torch.Generator)
        assert training_loader["generator"].initial_seed() == 73
        assert validation_loader["shuffle"] is False
        assert validation_loader["num_workers"] == 0
        assert "generator" not in validation_loader
    assert optimizer_calls == [
        {"lr": 3e-4, "weight_decay": 1e-4},
        {"lr": 3e-4, "weight_decay": 1e-4},
    ]
    for metrics in (*first_summary.history, *second_summary.history):
        assert math.isfinite(metrics.training_loss)
        assert math.isfinite(metrics.validation_loss)
        assert math.isfinite(metrics.validation_accuracy)


@pytest.mark.parametrize(
    ("losses", "patience", "expected_epochs", "selected_epoch", "selected_fill"),
    [
        ((0.3, 0.2, 0.2, 0.1), 5, 4, 4, 4.0),
        ((0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01), 5, 6, 1, 1.0),
        ((0.2, 0.1, 0.3), 5, 3, 2, 2.0),
        ((0.1, 0.1), None, 2, 1, 1.0),
    ],
)
def test_checkpoint_selection_is_strict_patience_is_consecutive_and_state_is_restored(
    monkeypatch: pytest.MonkeyPatch,
    losses: tuple[float, ...],
    patience: int | None,
    expected_epochs: int,
    selected_epoch: int,
    selected_fill: float,
) -> None:
    training = _dataset(_all_classes())
    validation = _dataset((PitchAction.SUBMIT,) * 3)
    train_calls = 0
    evaluation_calls = 0
    seen_class_weights: list[torch.Tensor] = []

    def fake_train_epoch(
        model: torch.nn.Module,
        loader: object,
        optimizer: object,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> float:
        nonlocal train_calls
        del loader, optimizer, device
        train_calls += 1
        seen_class_weights.append(class_weights.detach().cpu().clone())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(float(train_calls))
        return float(train_calls)

    def fake_evaluate_epoch(
        model: torch.nn.Module,
        loader: object,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> tuple[float, float]:
        nonlocal evaluation_calls
        del model, loader, device
        seen_class_weights.append(class_weights.detach().cpu().clone())
        result = (losses[evaluation_calls], 0.5)
        evaluation_calls += 1
        return result

    real_weights = bc.training_class_weights
    weight_inputs: list[tuple[PitchAction, ...]] = []

    def recording_weights(actions: Sequence[PitchAction]) -> torch.Tensor:
        normalized = tuple(actions)
        weight_inputs.append(normalized)
        return real_weights(normalized)

    monkeypatch.setattr(bc, "_train_epoch", fake_train_epoch)
    monkeypatch.setattr(bc, "_evaluate_epoch", fake_evaluate_epoch)
    monkeypatch.setattr(bc, "training_class_weights", recording_weights)

    model, summary = train_behavior_cloning(
        training,
        validation,
        config=_profile(max_epochs=len(losses), patience=patience),
        seed=9,
        device=torch.device("cpu"),
    )

    assert len(summary.history) == expected_epochs
    assert summary.selected_epoch == selected_epoch
    assert train_calls == evaluation_calls == expected_epochs
    assert weight_inputs == [_all_classes()]
    assert all(torch.equal(weights, torch.ones(7)) for weights in seen_class_weights)
    assert all(
        torch.equal(parameter, torch.full_like(parameter, selected_fill))
        for parameter in model.parameters()
    )


def test_weighted_validation_metrics_use_dataset_denominators_with_unequal_final_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_actions = tuple(
        action
        for action, count in zip(PitchAction, (1, 2, 3, 4, 5, 6, 7), strict=True)
        for _ in range(count)
    )
    validation_actions = (
        PitchAction.OCTAVE_DOWN,
        PitchAction.OCTAVE_DOWN,
        PitchAction.SEMITONE_DOWN,
        PitchAction.CENT_DOWN,
        PitchAction.CENT_DOWN,
        PitchAction.SUBMIT,
        PitchAction.CENT_UP,
        PitchAction.OCTAVE_UP,
        PitchAction.OCTAVE_UP,
    )
    training = _dataset(training_actions)
    validation = _dataset(validation_actions)

    def no_training(
        model: torch.nn.Module,
        loader: object,
        optimizer: object,
        class_weights: torch.Tensor,
        device: torch.device,
    ) -> float:
        del model, loader, optimizer, class_weights, device
        return 1.0

    monkeypatch.setattr(bc, "BCPolicyNetwork", _FixedLogits)
    monkeypatch.setattr(bc, "_train_epoch", no_training)

    results = tuple(
        train_behavior_cloning(
            training,
            validation,
            config=_profile(max_epochs=1, batch_size=batch_size),
            seed=1,
            device=torch.device("cpu"),
        )
        for batch_size in (4, 5)
    )

    weights = [4 / count for count in (1, 2, 3, 4, 5, 6, 7)]
    log_normalizer = math.log(sum(math.exp(value) for value in range(7)))
    numerator = sum(
        weights[int(action)] * (log_normalizer - int(action)) for action in validation_actions
    )
    denominator = sum(weights[int(action)] for action in validation_actions)
    expected_loss = numerator / denominator
    observed_losses = tuple(summary.history[0].validation_loss for _, summary in results)
    assert observed_losses[0] == observed_losses[1]
    assert observed_losses[0] == pytest.approx(expected_loss, abs=1e-6)
    assert tuple(summary.selected_epoch for _, summary in results) == (1, 1)
    assert tuple(summary.history[0].validation_accuracy for _, summary in results) == (
        2 / 9,
        2 / 9,
    )
    assert tuple(
        next_action_accuracy(
            model,
            validation,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )
        for (model, _), batch_size in zip(results, (4, 5), strict=True)
    ) == (2 / 9, 2 / 9)


def test_model_persistence_is_state_dict_only_and_actor_reload_validates_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, pending = _writer_with_core(tmp_path / "artifact")
    persisted = torch.load(pending.file("model.pt"), map_location="cpu", weights_only=True)
    assert isinstance(persisted, dict)
    assert tuple(persisted) == tuple(BCPolicyNetwork().state_dict())
    assert all(type(value) is torch.Tensor for value in persisted.values())

    events: list[str] = []
    real_validate = bc.validate_bc_artifact
    real_network = bc.BCPolicyNetwork
    real_load = torch.load
    load_calls: list[dict[str, object]] = []

    def recording_validate(artifact: LoadedArtifact | PendingArtifactView) -> None:
        events.append("validate")
        real_validate(artifact)

    def recording_network() -> BCPolicyNetwork:
        events.append("construct")
        return real_network()

    def recording_load(*args: object, **kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(bc, "validate_bc_artifact", recording_validate)
    monkeypatch.setattr(bc, "BCPolicyNetwork", recording_network)
    monkeypatch.setattr(bc.torch, "load", recording_load)

    actor = load_bc_actor(pending)

    assert isinstance(actor, BCActor)
    assert events[:2] == ["validate", "construct"]
    assert load_calls
    assert all(call == {"map_location": "cpu", "weights_only": True} for call in load_calls)
    assert writer.file_records[-1].relative_path == "model.pt"


def test_masked_actor_loader_strictly_reloads_persisted_state_without_legacy_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = _writer_with_core(tmp_path / "artifact")
    events: list[str] = []
    real_validate = bc.validate_bc_artifact
    real_network = bc.BCPolicyNetwork
    real_masked_actor = bc.MaskedBCActor

    def recording_validate(artifact: LoadedArtifact | PendingArtifactView) -> None:
        events.append("validate")
        real_validate(artifact)

    def recording_network() -> BCPolicyNetwork:
        events.append("construct")
        return real_network()

    def recording_masked_actor(model: object, *, device: object) -> MaskedBCActor:
        events.append("masked")
        return real_masked_actor(model, device=device)

    monkeypatch.setattr(bc, "validate_bc_artifact", recording_validate)
    monkeypatch.setattr(bc, "BCPolicyNetwork", recording_network)
    monkeypatch.setattr(bc, "MaskedBCActor", recording_masked_actor)
    monkeypatch.setattr(
        bc,
        "BCActor",
        lambda *args, **kwargs: pytest.fail("masked loader constructed a legacy BCActor"),
    )

    actor = load_masked_bc_actor(pending)

    assert isinstance(actor, MaskedBCActor)
    assert events[:3] == ["validate", "construct", "masked"]


def test_masked_actor_loader_rejects_wrong_artifact_and_device_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, pending = _writer_with_core(tmp_path / "artifact")
    monkeypatch.setattr(bc.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        bc,
        "MaskedBCActor",
        lambda *args, **kwargs: pytest.fail("invalid input constructed a masked actor"),
    )

    with pytest.raises(ValueError, match="DeviceName"):
        load_masked_bc_actor(pending, device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="CUDA"):
        load_masked_bc_actor(pending, device=DeviceName.CUDA)
    with pytest.raises(ValueError, match="artifact"):
        load_masked_bc_actor(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "names"),
        ("extra", "names"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("nonfinite", "finite"),
    ],
)
def test_artifact_validation_rejects_unsafe_or_inexact_model_state(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in BCPolicyNetwork().state_dict().items()
    }
    first = next(iter(state))
    if mutation == "missing":
        state.pop(first)
    elif mutation == "extra":
        state["unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[first] = state[first].reshape(-1)
    elif mutation == "dtype":
        state[first] = state[first].to(torch.float64)
    else:
        state[first].reshape(-1)[0] = float("nan")
    _, pending = _writer_with_core(tmp_path / mutation, model_state=state)

    with pytest.raises(ValueError, match=message):
        validate_bc_artifact(pending)


def test_artifact_validation_cross_checks_parameter_count_and_pending_inventory(
    tmp_path: Path,
) -> None:
    expected_count = sum(parameter.numel() for parameter in BCPolicyNetwork().parameters())
    writer, pending = _writer_with_core(
        tmp_path / "wrong-count",
        parameter_count=expected_count + 1,
    )

    with pytest.raises(ValueError, match="parameter_count"):
        validate_bc_artifact(pending)
    with pytest.raises(ValueError, match=r"summary|inventory"):
        validate_bc_artifact(writer.pending_view(("training-config.json", "model.pt")))


@pytest.mark.parametrize("profile", list(ProfileName))
def test_full_pending_and_complete_validation_decode_every_evaluation_payload(
    tmp_path: Path,
    profile: ProfileName,
) -> None:
    writer, _ = _writer_with_core(tmp_path / profile.value, profile=profile)
    for filename in required_payload_names(TrainerKind.BC, profile):
        if filename == "evaluation-smoke.json":
            document = _evaluation_file(EvaluationSuiteId.SMOKE).to_document()
        elif filename == "evaluation-iid.json":
            document = _evaluation_file(EvaluationSuiteId.IID).to_document()
        elif filename == "evaluation-ood.json":
            document = _evaluation_file(EvaluationSuiteId.REGISTER_OOD).to_document()
        else:
            continue
        writer.publish_json(filename, document)
    full_pending = writer.pending_view(required_payload_names(TrainerKind.BC, profile))

    validate_bc_artifact(full_pending)
    completed = writer.complete(_completion(profile))
    loaded = load_artifact(completed.root)
    validate_bc_artifact(loaded)

    assert loaded is not completed


def test_evaluation_payloads_are_strictly_decoded_in_pending_and_complete_views(
    tmp_path: Path,
) -> None:
    profile = ProfileName.SMOKE
    writer, _ = _writer_with_core(tmp_path / "malformed", profile=profile)
    writer.publish_json("evaluation-smoke.json", {"schema_version": 1, "rows": []})
    full_pending = writer.pending_view(required_payload_names(TrainerKind.BC, profile))

    with pytest.raises(ValueError, match="evaluation file"):
        validate_bc_artifact(full_pending)

    loaded = writer.complete(_completion(profile))
    with pytest.raises(ValueError, match="evaluation file"):
        validate_bc_artifact(loaded)


def test_artifact_validation_requires_the_exact_bc_probe_lane_inventory(tmp_path: Path) -> None:
    profile = ProfileName.SMOKE
    writer, _ = _writer_with_core(tmp_path / "missing-probes", profile=profile)
    writer.publish_json(
        "evaluation-smoke.json",
        _single_lane_evaluation_file(EvaluationSuiteId.SMOKE).to_document(),
    )
    full_pending = writer.pending_view(required_payload_names(TrainerKind.BC, profile))

    with pytest.raises(ValueError, match="probe inventory"):
        validate_bc_artifact(full_pending)


@pytest.mark.parametrize(
    ("profile", "expected_evaluations", "expected_rollouts", "accuracy_calls"),
    [
        (ProfileName.SMOKE, ("evaluation-smoke.json",), 3, 0),
        (
            ProfileName.CHECKPOINT,
            ("evaluation-iid.json", "evaluation-ood.json"),
            4,
            1,
        ),
    ],
)
def test_train_bc_artifact_publishes_validates_and_completes_in_exact_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: ProfileName,
    expected_evaluations: tuple[str, ...],
    expected_rollouts: int,
    accuracy_calls: int,
) -> None:
    output = tmp_path / profile.value
    events: list[str] = []
    _install_fast_workflow_fakes(monkeypatch, events)
    completed_views: list[LoadedArtifact] = []

    real_publish_json = ArtifactWriter.publish_json
    real_pending_view = ArtifactWriter.pending_view
    real_complete = ArtifactWriter.complete
    real_save = bc.save_bc_model
    real_load_actor = bc.load_bc_actor
    real_validate = bc.validate_bc_artifact
    real_criterion = bc.evaluate_bc_criterion

    def publish_json(
        writer: ArtifactWriter,
        filename: str,
        document: object,
    ) -> object:
        events.append(f"publish:{filename}")
        return real_publish_json(writer, filename, document)  # type: ignore[arg-type]

    def pending_view(writer: ArtifactWriter, names: Sequence[str]) -> PendingArtifactView:
        events.append(f"pending:{','.join(names)}")
        return real_pending_view(writer, names)

    def complete(writer: ArtifactWriter, completion: ArtifactCompletion) -> LoadedArtifact:
        loaded = real_complete(writer, completion)
        completed_views.append(loaded)
        events.append("complete")
        return loaded

    def save(path: Path, model: BCPolicyNetwork) -> None:
        events.append("save")
        real_save(path, model)

    def load_actor(
        artifact: LoadedArtifact | PendingArtifactView,
        *,
        device: DeviceName = DeviceName.CPU,
    ) -> BCActor:
        events.append(f"load-actor:{device.value}")
        return real_load_actor(artifact, device=device)

    def validate(artifact: LoadedArtifact | PendingArtifactView) -> None:
        view_size: int | str = (
            len(artifact.files) if isinstance(artifact, PendingArtifactView) else "complete"
        )
        events.append(f"validate:{view_size}")
        real_validate(artifact)

    def criterion(**kwargs: object) -> object:
        events.append("criterion")
        return real_criterion(**kwargs)  # type: ignore[arg-type]

    def forbidden_post_completion_load(path: Path) -> LoadedArtifact:
        del path
        events.append("post-completion-load")
        raise AssertionError("no fallible workflow operation may run after completion")

    monkeypatch.setattr(ArtifactWriter, "publish_json", publish_json)
    monkeypatch.setattr(ArtifactWriter, "pending_view", pending_view)
    monkeypatch.setattr(ArtifactWriter, "complete", complete)
    monkeypatch.setattr(bc, "save_bc_model", save)
    monkeypatch.setattr(bc, "load_bc_actor", load_actor)
    monkeypatch.setattr(bc, "validate_bc_artifact", validate)
    monkeypatch.setattr(bc, "evaluate_bc_criterion", criterion)
    monkeypatch.setattr(
        bc,
        "load_artifact",
        forbidden_post_completion_load,
        raising=False,
    )

    artifact = train_bc_artifact(
        profile=profile,
        seed=0,
        device=DeviceName.CPU,
        output=output,
    )

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact.manifest.runtime.device is DeviceName.CPU
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact is completed_views[0]
    assert events.count("accuracy:cpu") == accuracy_calls
    assert events.count("examples:256") == accuracy_calls
    assert sum(event.startswith("evaluate:") for event in events) == expected_rollouts
    example_event = "examples:128" if profile is ProfileName.SMOKE else "examples:4096"
    assert events.index("source") < events.index(example_event)
    assert events.index("train:cpu") < events.index("publish:training-config.json")
    assert events.index("publish:training-config.json") < events.index(
        "publish:training-summary.json"
    )
    assert events.index("publish:training-summary.json") < events.index("save")
    assert events.index("save") < events.index("load-actor:cpu")
    assert events.index("load-actor:cpu") < next(
        index for index, event in enumerate(events) if event.startswith("evaluate:")
    )
    assert max(
        index for index, event in enumerate(events) if event.startswith("evaluate:")
    ) < events.index("criterion")
    assert events.index("criterion") < events.index(f"publish:{expected_evaluations[0]}")
    full_inventory_size = 4 if profile is ProfileName.SMOKE else 5
    assert events.index(f"validate:{full_inventory_size}") < events.index("complete")
    assert events[-1] == "complete"

    assert tuple(
        record.relative_path for record in artifact.manifest.files
    ) == required_payload_names(TrainerKind.BC, profile)
    for filename in expected_evaluations:
        evaluation = EvaluationFile.from_document(artifact.document(filename))
        if evaluation.suite_id is EvaluationSuiteId.REGISTER_OOD:
            assert tuple(row.probe for row in evaluation.rows) == (None, None, None)
        else:
            assert tuple(row.probe for row in evaluation.rows) == (
                None,
                ZERO_SPECTRUM_PROBE,
                SHUFFLED_SPECTRUM_PROBE,
            )
    assert artifact.manifest.criterion_eligible is (profile is ProfileName.CHECKPOINT)


def test_checkpoint_accuracy_uses_fixed_iid_oracle_data_after_persisted_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    real_build_examples = bc.build_oracle_examples
    real_save = bc.save_bc_model
    real_load_actor = bc.load_bc_actor
    _install_fast_workflow_fakes(monkeypatch, events)
    fake_build_examples = bc.build_oracle_examples
    fake_train = bc.train_behavior_cloning
    fake_evaluate = bc.evaluate_learned_actor

    splits = build_bc_episode_splits(profile=ProfileName.CHECKPOINT, run_seed=0)
    iid_suite = fixed_evaluation_suite(EvaluationSuiteId.IID)
    build_inputs: list[tuple[EpisodeSpec, ...]] = []
    selection_datasets: list[tuple[OracleTrajectoryDataset, OracleTrajectoryDataset]] = []
    accuracy_datasets: list[OracleTrajectoryDataset] = []
    evaluated_suites: list[object] = []

    def build_examples(
        episodes: Sequence[EpisodeSpec],
        *,
        env_factory: object,
    ) -> tuple[BCExample, ...]:
        normalized = tuple(episodes)
        build_inputs.append(normalized)
        if normalized == iid_suite.episodes:
            events.append("build-fixed-iid")
            return real_build_examples(
                normalized,
                env_factory=env_factory,  # type: ignore[arg-type]
            )
        return fake_build_examples(
            normalized,
            env_factory=env_factory,  # type: ignore[arg-type]
        )

    def train(
        training_dataset: OracleTrajectoryDataset,
        validation_dataset: OracleTrajectoryDataset,
        *,
        config: BCProfile,
        seed: int,
        device: torch.device,
    ) -> tuple[BCPolicyNetwork, BCTrainingSummary]:
        selection_datasets.append((training_dataset, validation_dataset))
        return fake_train(
            training_dataset,
            validation_dataset,
            config=config,
            seed=seed,
            device=device,
        )

    def save(path: Path, model: BCPolicyNetwork) -> None:
        events.append("save-selected")
        real_save(path, model)

    def load_actor(
        artifact: LoadedArtifact | PendingArtifactView,
        *,
        device: DeviceName = DeviceName.CPU,
    ) -> BCActor:
        events.append(f"reload-selected:{device.value}")
        return real_load_actor(artifact, device=device)

    def accuracy(
        model: BCPolicyNetwork,
        dataset: OracleTrajectoryDataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> float:
        del model, batch_size
        events.append(f"accuracy:{device.type}")
        accuracy_datasets.append(dataset)
        assert dataset is not selection_datasets[0][1]
        observation, label = dataset[0]
        assert observation["spectrum"].shape == (LOG_SPECTRUM_SIZE,)
        assert observation["spectrum"].dtype == np.dtype(np.float32)
        assert observation["state"].shape == (5,)
        assert observation["state"].dtype == np.dtype(np.float32)
        assert type(label) is np.int64
        return 0.95

    def evaluate(
        actor: object,
        suite: object,
        *,
        environment_factory: object,
    ) -> tuple[TerminalEpisodeRecord, ...]:
        evaluated_suites.append(suite)
        return fake_evaluate(
            actor,
            suite,
            environment_factory=environment_factory,
        )

    monkeypatch.setattr(bc, "build_oracle_examples", build_examples)
    monkeypatch.setattr(bc, "train_behavior_cloning", train)
    monkeypatch.setattr(bc, "save_bc_model", save)
    monkeypatch.setattr(bc, "load_bc_actor", load_actor)
    monkeypatch.setattr(bc, "next_action_accuracy", accuracy)
    monkeypatch.setattr(bc, "evaluate_learned_actor", evaluate)

    artifact = train_bc_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=0,
        device=DeviceName.CPU,
        output=tmp_path / "heldout-iid",
    )

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert build_inputs == [splits.training, splits.validation, iid_suite.episodes]
    assert len(iid_suite.episodes) == 256
    assert set(iid_suite.episodes).isdisjoint(splits.validation)
    assert len(selection_datasets) == 1
    assert len(accuracy_datasets) == 1
    assert accuracy_datasets[0] not in selection_datasets[0]
    assert events.index("train:cpu") < events.index("save-selected")
    assert events.index("save-selected") < events.index("reload-selected:cpu")
    assert events.index("reload-selected:cpu") < events.index("build-fixed-iid")
    assert events.index("build-fixed-iid") < events.index("accuracy:cpu")
    assert tuple(suite for suite in evaluated_suites if suite is iid_suite) == (
        iid_suite,
        iid_suite,
        iid_suite,
    )


def test_exploratory_cuda_training_reloads_and_evaluates_the_persisted_actor_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_fast_workflow_fakes(monkeypatch, events)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    real_load_actor = bc.load_bc_actor

    def load_actor(
        artifact: LoadedArtifact | PendingArtifactView,
        *,
        device: DeviceName = DeviceName.CPU,
    ) -> BCActor:
        events.append(f"load-actor:{device.value}")
        return real_load_actor(artifact, device=device)

    monkeypatch.setattr(bc, "load_bc_actor", load_actor)

    artifact = train_bc_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=0,
        device=DeviceName.CUDA,
        output=tmp_path / "cuda-exploratory",
    )

    assert "train:cuda" in events
    assert "load-actor:cpu" in events
    assert "accuracy:cpu" in events
    assert artifact.manifest.runtime.device is DeviceName.CUDA
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact.manifest.criterion_eligible is False
    assert artifact.manifest.criterion_status is CriterionStatus.INELIGIBLE


@pytest.mark.parametrize(
    "boundary",
    [
        "examples",
        "train",
        "publish-core",
        "save",
        "pending-core",
        "load",
        "accuracy",
        "evaluate",
        "criterion",
        "publish-evaluation",
        "pending-full",
        "validate-full",
        "complete",
    ],
)
def test_post_bootstrap_failures_leave_the_persisted_manifest_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    output = tmp_path / boundary
    events: list[str] = []
    _install_fast_workflow_fakes(monkeypatch, events)

    def injected(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError(f"injected {boundary}")

    if boundary == "examples":
        monkeypatch.setattr(bc, "build_oracle_examples", injected)
    elif boundary == "train":
        monkeypatch.setattr(bc, "train_behavior_cloning", injected)
    elif boundary == "publish-core":
        real = ArtifactWriter.publish_json

        def fail_core(writer: ArtifactWriter, filename: str, document: object) -> object:
            if filename == "training-config.json":
                return injected()
            return real(writer, filename, document)  # type: ignore[arg-type]

        monkeypatch.setattr(ArtifactWriter, "publish_json", fail_core)
    elif boundary == "save":
        monkeypatch.setattr(bc, "save_bc_model", injected)
    elif boundary in {"pending-core", "pending-full"}:
        real = ArtifactWriter.pending_view

        def fail_pending(writer: ArtifactWriter, names: Sequence[str]) -> PendingArtifactView:
            is_full = len(tuple(names)) > 3
            if is_full == (boundary == "pending-full"):
                return injected()  # type: ignore[return-value]
            return real(writer, names)

        monkeypatch.setattr(ArtifactWriter, "pending_view", fail_pending)
    elif boundary == "load":
        monkeypatch.setattr(bc, "load_bc_actor", injected)
    elif boundary == "accuracy":
        monkeypatch.setattr(bc, "next_action_accuracy", injected)
    elif boundary == "evaluate":
        monkeypatch.setattr(bc, "evaluate_learned_actor", injected)
    elif boundary == "criterion":
        monkeypatch.setattr(bc, "evaluate_bc_criterion", injected)
    elif boundary == "publish-evaluation":
        real = ArtifactWriter.publish_json

        def fail_evaluation(writer: ArtifactWriter, filename: str, document: object) -> object:
            if filename.startswith("evaluation-"):
                return injected()
            return real(writer, filename, document)  # type: ignore[arg-type]

        monkeypatch.setattr(ArtifactWriter, "publish_json", fail_evaluation)
    elif boundary == "validate-full":
        real = bc.validate_bc_artifact

        def fail_full(artifact: LoadedArtifact | PendingArtifactView) -> None:
            if isinstance(artifact, PendingArtifactView) and len(artifact.files) > 3:
                injected()
            real(artifact)

        monkeypatch.setattr(bc, "validate_bc_artifact", fail_full)
    else:
        monkeypatch.setattr(ArtifactWriter, "complete", injected)

    with pytest.raises(RuntimeError, match=f"injected {boundary}"):
        train_bc_artifact(
            profile=ProfileName.CHECKPOINT,
            seed=0,
            device=DeviceName.CPU,
            output=output,
        )

    manifest = ArtifactManifest.from_document(read_json_document(output / "manifest.json"))
    assert manifest.status is ArtifactStatus.INCOMPLETE
    assert manifest.completed_at_utc is None
    assert manifest.files == ()
    with pytest.raises(ValueError, match="incomplete"):
        load_artifact(output)


def test_nonzero_seed_is_repeatable_distinct_ineligible_and_final_iid_cannot_feed_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_events: list[str] = []
    _install_fast_workflow_fakes(monkeypatch, first_events, evaluation_success=False)
    first = train_bc_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=3,
        device=DeviceName.CPU,
        output=tmp_path / "failed-finals",
    )

    second_events: list[str] = []
    _install_fast_workflow_fakes(monkeypatch, second_events, evaluation_success=True)
    second = train_bc_artifact(
        profile=ProfileName.CHECKPOINT,
        seed=3,
        device=DeviceName.CPU,
        output=tmp_path / "successful-finals",
    )

    first_config = TrainingConfigDocument.from_document(first.document("training-config.json"))
    second_config = TrainingConfigDocument.from_document(second.document("training-config.json"))
    seed_zero = build_bc_episode_splits(profile=ProfileName.CHECKPOINT, run_seed=0)
    assert first_config == second_config
    assert (
        first_config.bc_training_digest_sha256,
        first_config.bc_validation_digest_sha256,
    ) != (seed_zero.training_digest_sha256, seed_zero.validation_digest_sha256)
    assert first.manifest.criterion_eligible is False
    assert second.manifest.criterion_eligible is False
    assert first.manifest.criterion_met is None
    assert second.manifest.criterion_met is None

    first_summary = TrainingSummaryDocument.from_document(first.document("training-summary.json"))
    second_summary = TrainingSummaryDocument.from_document(second.document("training-summary.json"))
    assert first_summary == second_summary
    first_state = torch.load(first.file("model.pt"), map_location="cpu", weights_only=True)
    second_state = torch.load(second.file("model.pt"), map_location="cpu", weights_only=True)
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)

    failed_iid = EvaluationFile.from_document(first.document("evaluation-iid.json"))
    successful_iid = EvaluationFile.from_document(second.document("evaluation-iid.json"))
    assert failed_iid.rows[0].metrics.submitted_success_rate == 0.0
    assert successful_iid.rows[0].metrics.submitted_success_rate == 1.0
    assert first_events.count("accuracy:cpu") == second_events.count("accuracy:cpu") == 1


def test_train_bc_artifact_rejects_inputs_before_source_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = False

    def capture(path: Path) -> SourceStatus:
        nonlocal captured
        del path
        captured = True
        return _clean_source_status()

    monkeypatch.setattr(bc, "capture_source_status", capture)
    invalid_calls = (
        {"profile": "smoke", "seed": 0, "device": DeviceName.CPU, "output": tmp_path / "a"},
        {
            "profile": ProfileName.SMOKE,
            "seed": True,
            "device": DeviceName.CPU,
            "output": tmp_path / "b",
        },
        {
            "profile": ProfileName.SMOKE,
            "seed": -1,
            "device": DeviceName.CPU,
            "output": tmp_path / "c",
        },
        {"profile": ProfileName.SMOKE, "seed": 0, "device": "cpu", "output": tmp_path / "d"},
        {"profile": ProfileName.SMOKE, "seed": 0, "device": DeviceName.CPU, "output": "artifact"},
    )
    for arguments in invalid_calls:
        with pytest.raises(ValueError):
            train_bc_artifact(**arguments)  # type: ignore[arg-type]
    assert captured is False
