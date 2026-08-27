"""Deterministic Milestone E pitch-coordinate training contracts."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning import pitch
from harpy.learning.errors import LearningContractError
from harpy.learning.models import ProfileName
from harpy.learning.pitch import (
    PITCH_PROFILE_CONFIGS,
    PitchCoordinateDataset,
    PitchDatasetMetrics,
    PitchEpochMetrics,
    PitchPredictionRecord,
    PitchTrainingProfile,
    PitchTrainingSummary,
    build_pitch_coordinate_datasets,
    load_pitch_estimator_model,
    pitch_epoch_rank,
    save_pitch_estimator_model,
    summarize_pitch_predictions,
    train_pitch_estimator,
)
from harpy.learning.pitch_data import pitch_coordinate_split
from harpy.learning.pitch_network import PitchEstimatorNetwork, pitch_class_index


class _EvidenceSpy:
    def __init__(self) -> None:
        self.requested: list[int] = []
        self.returned: dict[int, np.ndarray] = {}

    def spectrum_for_cents(self, candidate_cents: int) -> np.ndarray:
        self.requested.append(candidate_cents)
        spectrum = np.linspace(0.0, 0.5, LOG_SPECTRUM_SIZE, dtype=np.float32)
        spectrum[(candidate_cents - 1_100) // 5] = np.float32(1.0)
        self.returned[candidate_cents] = spectrum
        return spectrum


def _small_profile(
    *,
    max_epochs: int = 2,
    patience: int | None = None,
) -> PitchTrainingProfile:
    return PitchTrainingProfile(
        training_coordinate_count=8,
        validation_coordinate_count=4,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        batch_size=64,
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.0001,
        eligible_for_aggregate=False,
    )


def _small_datasets() -> tuple[PitchCoordinateDataset, PitchCoordinateDataset]:
    split = pitch_coordinate_split()
    evidence = _EvidenceSpy()
    return (
        PitchCoordinateDataset(split.training_coordinates[:8], evidence),
        PitchCoordinateDataset(split.validation_coordinates[:4], evidence),
    )


def _metrics(
    *,
    loss: float,
    mae: float,
    within_one: int,
    within_five: int,
    count: int = 10,
) -> PitchDatasetMetrics:
    return PitchDatasetMetrics(
        example_count=count,
        loss=loss,
        mean_absolute_error_cents=mae,
        median_absolute_error_cents=mae,
        within_one_count=within_one,
        within_one_rate=within_one / count,
        within_five_count=within_five,
        within_five_rate=within_five / count,
    )


def test_profiles_pin_smoke_checkpoint_and_optimizer_contracts() -> None:
    assert isinstance(PITCH_PROFILE_CONFIGS, MappingProxyType)
    assert tuple(PITCH_PROFILE_CONFIGS) == (ProfileName.SMOKE, ProfileName.CHECKPOINT)

    smoke = PITCH_PROFILE_CONFIGS[ProfileName.SMOKE]
    assert (
        smoke.training_coordinate_count,
        smoke.validation_coordinate_count,
        smoke.max_epochs,
        smoke.early_stopping_patience,
        smoke.eligible_for_aggregate,
    ) == (256, 64, 2, None, False)

    checkpoint = PITCH_PROFILE_CONFIGS[ProfileName.CHECKPOINT]
    assert (
        checkpoint.training_coordinate_count,
        checkpoint.validation_coordinate_count,
        checkpoint.max_epochs,
        checkpoint.early_stopping_patience,
        checkpoint.eligible_for_aggregate,
    ) == (1_400, 200, 50, 8, True)

    for profile in PITCH_PROFILE_CONFIGS.values():
        assert profile.optimizer == "AdamW"
        assert profile.learning_rate == 0.001
        assert profile.weight_decay == 0.0001
        assert profile.batch_size == 64

    with pytest.raises(TypeError):
        PITCH_PROFILE_CONFIGS[ProfileName.SMOKE] = checkpoint  # type: ignore[index]


def test_dataset_factory_renders_only_ordered_profile_coordinate_prefixes() -> None:
    split = pitch_coordinate_split()
    evidence = _EvidenceSpy()

    training, validation = build_pitch_coordinate_datasets(
        ProfileName.SMOKE,
        evidence_provider=evidence,
    )

    expected_training = split.training_coordinates[:256]
    expected_validation = split.validation_coordinates[:64]
    assert training.coordinates == expected_training
    assert validation.coordinates == expected_validation
    assert evidence.requested == [*expected_training, *expected_validation]
    assert set(evidence.requested).isdisjoint(split.iid_holdout_coordinates)
    assert set(evidence.requested).isdisjoint(split.ood_lower_coordinates)
    assert set(evidence.requested).isdisjoint(split.ood_upper_coordinates)


def test_dataset_owns_provider_spectra_and_returns_fresh_copies() -> None:
    coordinate = pitch_coordinate_split().training_coordinates[0]
    evidence = _EvidenceSpy()
    dataset = PitchCoordinateDataset((coordinate,), evidence)

    evidence.returned[coordinate].fill(np.float32(99.0))
    spectrum, label, returned_coordinate = dataset[0]
    original = spectrum.copy()
    spectrum.fill(np.float32(-5.0))
    second_spectrum, second_label, second_coordinate = dataset[0]

    assert second_spectrum is not spectrum
    assert np.array_equal(second_spectrum, original)
    assert second_spectrum.flags.writeable
    assert label == second_label == np.int64(pitch_class_index(coordinate))
    assert returned_coordinate == second_coordinate == np.int64(coordinate)


def test_prediction_metrics_are_recomputed_from_ordered_raw_records() -> None:
    records = (
        PitchPredictionRecord(5_000, pitch_class_index(5_000), 0.25),
        PitchPredictionRecord(5_001, pitch_class_index(5_000), 0.50),
        PitchPredictionRecord(5_004, pitch_class_index(5_010), 1.00),
        PitchPredictionRecord(5_008, pitch_class_index(5_015), 2.25),
    )

    metrics = summarize_pitch_predictions(records)

    assert metrics.example_count == 4
    assert metrics.loss == pytest.approx(math.fsum((0.25, 0.5, 1.0, 2.25)) / 4)
    assert metrics.mean_absolute_error_cents == pytest.approx((0 + 1 + 6 + 7) / 4)
    assert metrics.median_absolute_error_cents == pytest.approx(3.5)
    assert (metrics.within_one_count, metrics.within_one_rate) == (2, 0.5)
    assert (metrics.within_five_count, metrics.within_five_rate) == (2, 0.5)

    with pytest.raises(ValueError, match="derived"):
        replace(metrics, within_five_rate=0.75)


def test_epoch_rank_is_the_exact_full_lexicographic_order() -> None:
    base = PitchEpochMetrics(
        epoch=3,
        training=_metrics(loss=0.7, mae=4.0, within_one=7, within_five=8),
        validation=_metrics(loss=0.6, mae=3.0, within_one=7, within_five=8),
    )

    assert pitch_epoch_rank(base) == (-8, -7, 3.0, 0.6, 3)
    assert pitch_epoch_rank(
        replace(base, validation=_metrics(loss=999.0, mae=999.0, within_one=0, within_five=9))
    ) < pitch_epoch_rank(base)
    assert pitch_epoch_rank(
        replace(base, validation=_metrics(loss=999.0, mae=999.0, within_one=8, within_five=8))
    ) < pitch_epoch_rank(base)
    assert pitch_epoch_rank(
        replace(base, validation=_metrics(loss=999.0, mae=2.0, within_one=7, within_five=8))
    ) < pitch_epoch_rank(base)
    assert pitch_epoch_rank(
        replace(base, validation=_metrics(loss=0.5, mae=3.0, within_one=7, within_five=8))
    ) < pitch_epoch_rank(base)
    assert pitch_epoch_rank(replace(base, epoch=2)) < pitch_epoch_rank(base)


def test_training_uses_seeded_cpu_shuffle_and_ordered_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, validation = _small_datasets()
    calls: list[dict[str, object]] = []
    seeds: list[tuple[str, int]] = []
    real_loader = torch.utils.data.DataLoader
    real_python_seed = pitch.random.seed
    real_numpy_seed = pitch.np.random.seed
    real_torch_seed = pitch.torch.manual_seed

    def tracking_loader(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return real_loader(*args, **kwargs)

    def tracking_python_seed(seed: int) -> None:
        seeds.append(("python", seed))
        real_python_seed(seed)

    def tracking_numpy_seed(seed: int) -> None:
        seeds.append(("numpy", seed))
        real_numpy_seed(seed)

    def tracking_torch_seed(seed: int) -> torch.Generator:
        seeds.append(("torch", seed))
        return real_torch_seed(seed)

    monkeypatch.setattr(pitch.torch.utils.data, "DataLoader", tracking_loader)
    monkeypatch.setattr(pitch.random, "seed", tracking_python_seed)
    monkeypatch.setattr(pitch.np.random, "seed", tracking_numpy_seed)
    monkeypatch.setattr(pitch.torch, "manual_seed", tracking_torch_seed)

    train_pitch_estimator(
        training,
        validation,
        profile=_small_profile(max_epochs=1),
        seed=17,
        device=torch.device("cpu"),
    )

    assert calls[0]["shuffle"] is True
    assert calls[0]["num_workers"] == 0
    generator = calls[0]["generator"]
    assert isinstance(generator, torch.Generator)
    assert generator.device.type == "cpu"
    assert generator.initial_seed() == 17
    assert seeds == [("python", 17), ("numpy", 17), ("torch", 17)]
    assert all(call["shuffle"] is False for call in calls[1:])
    assert all(call["num_workers"] == 0 for call in calls)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_selection_uses_strict_rank_patience_and_restores_deep_cpu_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    training, validation = _small_datasets()
    validation_metrics = iter(
        (
            _metrics(loss=1.0, mae=4.0, within_one=1, within_five=2, count=4),
            _metrics(loss=0.5, mae=4.0, within_one=1, within_five=2, count=4),
            _metrics(loss=0.5, mae=4.0, within_one=1, within_five=2, count=4),
            _metrics(loss=0.5, mae=4.0, within_one=1, within_five=2, count=4),
            _metrics(loss=0.1, mae=0.0, within_one=4, within_five=4, count=4),
        )
    )
    epoch = 0

    def fake_train_one_epoch(*args: object, **kwargs: object) -> None:
        nonlocal epoch
        del kwargs
        model = args[0]
        assert isinstance(model, PitchEstimatorNetwork)
        epoch += 1
        with torch.no_grad():
            model.scorer[0].weight.fill_(float(epoch))

    def fake_evaluate(
        model: PitchEstimatorNetwork,
        dataset: PitchCoordinateDataset,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[PitchDatasetMetrics, tuple[PitchPredictionRecord, ...]]:
        del model, batch_size, device
        if dataset is training:
            return _metrics(loss=0.9, mae=9.0, within_one=1, within_five=2, count=8), ()
        return next(validation_metrics), ()

    monkeypatch.setattr(pitch, "_train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(pitch, "evaluate_pitch_dataset", fake_evaluate)

    result = train_pitch_estimator(
        training,
        validation,
        profile=_small_profile(max_epochs=10, patience=2),
        seed=0,
        device=torch.device("cpu"),
    )

    assert len(result.summary.history) == 4
    assert result.summary.selected_epoch == 2
    assert torch.all(result.selected_state["scorer.0.weight"] == 2.0)
    assert torch.all(result.model.state_dict()["scorer.0.weight"] == 2.0)
    assert all(value.device.type == "cpu" for value in result.selected_state.values())
    result.selected_state["scorer.0.weight"].fill_(123.0)
    assert torch.all(result.model.state_dict()["scorer.0.weight"] == 2.0)

    persisted_path = tmp_path / "selected-model.pt"
    save_pitch_estimator_model(persisted_path, result.model)
    persisted = load_pitch_estimator_model(persisted_path)
    assert all(
        torch.equal(persisted.state_dict()[name], result.model.state_dict()[name])
        for name in result.model.state_dict()
    )


def test_same_seed_cpu_training_is_bit_identical_for_state_and_metrics() -> None:
    training, validation = build_pitch_coordinate_datasets(
        ProfileName.SMOKE,
        evidence_provider=_EvidenceSpy(),
    )
    profile = PITCH_PROFILE_CONFIGS[ProfileName.SMOKE]

    first = train_pitch_estimator(
        training,
        validation,
        profile=profile,
        seed=23,
        device=torch.device("cpu"),
    )
    second = train_pitch_estimator(
        training,
        validation,
        profile=profile,
        seed=23,
        device=torch.device("cpu"),
    )

    assert first.summary.history == second.summary.history
    assert first.summary.selected_epoch == second.summary.selected_epoch
    assert first.selected_state.keys() == second.selected_state.keys()
    assert all(
        torch.equal(first.selected_state[name], second.selected_state[name])
        for name in first.selected_state
    )


def test_model_state_round_trip_is_exact_and_rejects_malformed_payloads(tmp_path: Path) -> None:
    model = PitchEstimatorNetwork()
    path = tmp_path / "model.pt"

    save_pitch_estimator_model(path, model)
    reloaded = load_pitch_estimator_model(path)

    assert all(
        torch.equal(model.state_dict()[name], reloaded.state_dict()[name])
        for name in model.state_dict()
    )
    assert all(value.device.type == "cpu" for value in reloaded.state_dict().values())

    malformed_states: dict[str, dict[str, torch.Tensor]] = {}

    nonfinite = dict(model.state_dict())
    nonfinite["scorer.0.weight"] = nonfinite["scorer.0.weight"].clone()
    nonfinite["scorer.0.weight"][0, 0, 0] = float("nan")
    malformed_states["nonfinite"] = nonfinite

    wrong_names = dict(model.state_dict())
    wrong_names["other.weight"] = wrong_names.pop("scorer.0.weight")
    malformed_states["wrong-names"] = wrong_names

    wrong_shape = dict(model.state_dict())
    wrong_shape["scorer.0.weight"] = wrong_shape["scorer.0.weight"][:, :, :-1]
    malformed_states["wrong-shape"] = wrong_shape

    wrong_dtype = dict(model.state_dict())
    wrong_dtype["scorer.0.weight"] = wrong_dtype["scorer.0.weight"].to(torch.float64)
    malformed_states["wrong-dtype"] = wrong_dtype

    for name, malformed in malformed_states.items():
        bad_path = tmp_path / f"{name}.pt"
        torch.save(malformed, bad_path)
        with pytest.raises(LearningContractError, match="state is invalid"):
            load_pitch_estimator_model(bad_path)


def test_training_summary_rejects_contradictory_counts_and_device_names() -> None:
    epoch = PitchEpochMetrics(
        epoch=1,
        training=_metrics(loss=0.5, mae=2.0, within_one=7, within_five=8),
        validation=_metrics(loss=0.4, mae=1.0, within_one=8, within_five=9),
    )
    summary = PitchTrainingSummary(
        history=(epoch,),
        selected_epoch=1,
        training_examples=10,
        validation_examples=10,
        seed=0,
        device="cpu",
        deterministic_algorithms=True,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        data_loader_num_workers=0,
        shuffle_generator_device="cpu",
        training_wall_time_seconds=0.0,
    )

    with pytest.raises(ValueError, match="training_examples"):
        replace(summary, training_examples=9)
    with pytest.raises(ValueError, match="validation_examples"):
        replace(summary, validation_examples=9)
    with pytest.raises(ValueError, match="device"):
        replace(summary, device="cudafoo")


def test_profile_and_dataset_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        PitchTrainingProfile(
            training_coordinate_count=0,
            validation_coordinate_count=1,
            max_epochs=1,
            early_stopping_patience=None,
            batch_size=64,
            optimizer="AdamW",
            learning_rate=0.001,
            weight_decay=0.0001,
            eligible_for_aggregate=False,
        )

    split = pitch_coordinate_split()
    evidence = _EvidenceSpy()
    with pytest.raises(ValueError, match="duplicate"):
        PitchCoordinateDataset((split.training_coordinates[0],) * 2, evidence)
    with pytest.raises(ValueError, match="profile"):
        build_pitch_coordinate_datasets("smoke", evidence_provider=evidence)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _restore_rng_state() -> Iterator[None]:
    python_state = pitch.random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    yield
    pitch.random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)
