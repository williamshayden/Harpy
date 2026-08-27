"""Cross-artifact evaluation orchestration and strict report contracts."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import harpy.learning.workflows as workflows
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode, TerminalReason
from harpy.learning.errors import LearningContractError
from harpy.learning.evaluation import (
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    BCScientificCriterion,
    PPOScientificCriterion,
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
    DeviceName,
    EvaluationSuiteId,
    ProfileName,
    TrainerKind,
)
from harpy.learning.suites import fixed_evaluation_suite

_INELIGIBLE_BC_CRITERION = BCScientificCriterion(False, None, None, None, "ineligible")


def _rows(
    *,
    actor_id: str,
    trainer: TrainerKind | None,
    seed: int | None,
    suite_id: EvaluationSuiteId = EvaluationSuiteId.SMOKE,
    probe: str | None = None,
):
    suite = fixed_evaluation_suite(suite_id)
    records = tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
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
    if trainer is None:
        kind = BaselineKind(actor_id)
        environment_id = kind.environment_id
        observation_mode = kind.observation_mode
    else:
        environment_id = ENVIRONMENT_ID
        observation_mode = ObservationMode.SPECTRUM
    return build_evaluation_rows(
        actor_id=actor_id,
        trainer=trainer,
        seed=seed,
        environment_id=environment_id,
        observation_mode=observation_mode,
        suite=suite,
        records=records,
        probe=probe,
        parameter_count=None if trainer is None else 75_000,
        training_environment_steps=(2_048 if trainer is TrainerKind.PPO else None),
        training_examples=(1_000 if trainer is TrainerKind.BC else None),
        training_wall_time_seconds=(12.5 if trainer is not None else None),
    )


def _row(
    *,
    actor_id: str,
    trainer: TrainerKind | None,
    seed: int | None,
    suite_id: EvaluationSuiteId = EvaluationSuiteId.SMOKE,
    probe: str | None = None,
):
    return _rows(
        actor_id=actor_id,
        trainer=trainer,
        seed=seed,
        suite_id=suite_id,
        probe=probe,
    )[0]


def _actor_rows(
    *,
    profile: ProfileName,
    actor_id: str,
    trainer: TrainerKind | None,
    seed: int | None,
):
    rows = []
    for suite_id in PROFILE_CONFIGS[profile].evaluation_suites:
        rows.extend(
            _rows(
                actor_id=actor_id,
                trainer=trainer,
                seed=seed,
                suite_id=suite_id,
            )
        )
        if trainer is not None and suite_id is not EvaluationSuiteId.REGISTER_OOD:
            for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
                rows.extend(
                    _rows(
                        actor_id=actor_id,
                        trainer=trainer,
                        seed=seed,
                        suite_id=suite_id,
                        probe=probe,
                    )
                )
    return tuple(rows)


def _report_rows(
    *,
    profile: ProfileName,
    learned: tuple[tuple[TrainerKind, int], ...],
):
    rows = []
    for trainer, seed in learned:
        rows.extend(
            _actor_rows(
                profile=profile,
                actor_id=f"{trainer.value}-{seed}",
                trainer=trainer,
                seed=seed,
            )
        )
    for kind in (
        BaselineKind.RANDOM,
        BaselineKind.REWARD_SEARCH,
        BaselineKind.SPECTRUM_PEAK,
        BaselineKind.ORACLE,
    ):
        rows.extend(
            _actor_rows(
                profile=profile,
                actor_id=kind.value,
                trainer=None,
                seed=None,
            )
        )
    return tuple(rows)


def _report(
    *,
    profile: ProfileName = ProfileName.SMOKE,
    device: DeviceName = DeviceName.CPU,
    learned: tuple[tuple[TrainerKind, int], ...] = ((TrainerKind.BC, 0),),
    bc_criterion: BCScientificCriterion | None = _INELIGIBLE_BC_CRITERION,
    ppo_criterion: PPOScientificCriterion | None = None,
) -> workflows.EvaluationReport:
    return _report_from_rows(
        profile=profile,
        device=device,
        rows=_report_rows(profile=profile, learned=learned),
        bc_criterion=bc_criterion,
        ppo_criterion=ppo_criterion,
    )


def _report_from_rows(
    *,
    profile: ProfileName,
    device: DeviceName,
    rows: tuple,
    bc_criterion: BCScientificCriterion | None,
    ppo_criterion: PPOScientificCriterion | None,
) -> workflows.EvaluationReport:
    return workflows.EvaluationReport(
        schema_version=1,
        profile=profile,
        evaluation_device=device,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        rows=rows,
        bc_criterion=bc_criterion,
        ppo_criterion=ppo_criterion,
    )


def _base_iid_row(rows: tuple, *, actor_id: str):
    return next(
        row
        for row in rows
        if row.actor_id == actor_id
        and row.suite_id is EvaluationSuiteId.IID
        and row.subset == "combined"
        and row.probe is None
    )


def _manifest(
    *,
    trainer: TrainerKind,
    seed: int,
    profile: ProfileName = ProfileName.SMOKE,
    training_device: DeviceName = DeviceName.CPU,
    evaluation_device: DeviceName = DeviceName.CPU,
    dirty_tree: bool = False,
    required_inputs_committed: bool = True,
    criterion_eligible: bool | None = None,
):
    declared_seed = seed == 0 if trainer is TrainerKind.BC else seed in range(5)
    derived_eligible = (
        profile is ProfileName.CHECKPOINT
        and declared_seed
        and training_device is DeviceName.CPU
        and evaluation_device is DeviceName.CPU
        and not dirty_tree
        and required_inputs_committed
    )
    return SimpleNamespace(
        trainer=trainer,
        seed=seed,
        profile=profile,
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id="harpy-sine-policy-train-v1",
        evaluation_suites=tuple(
            (
                suite_id.value,
                fixed_evaluation_suite(suite_id).digest_sha256,
            )
            for suite_id in PROFILE_CONFIGS[profile].evaluation_suites
        ),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        source=SimpleNamespace(
            dirty_tree=dirty_tree,
            required_inputs_committed=required_inputs_committed,
        ),
        runtime=SimpleNamespace(device=training_device),
        evaluation_device=evaluation_device,
        criterion_eligible=(derived_eligible if criterion_eligible is None else criterion_eligible),
    )


def _artifact(root: Path, manifest: object):
    return SimpleNamespace(root=root.resolve(), manifest=manifest)


def _install_fast_workflow(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: dict[Path, object],
    events: list[str],
) -> None:
    def load(path: Path):
        events.append(f"load:{path.name}")
        return artifacts[path.resolve()]

    def peek(path: Path):
        manifest = artifacts[path.resolve()].manifest
        return 1, manifest.profile, manifest.trainer.value

    def validate(artifact: object) -> None:
        events.append(f"validate:{artifact.root.name}")

    def load_actor(artifact: object, *, device: DeviceName):
        assert device in DeviceName
        assert sum(item.startswith("validate:") for item in events) == len(artifacts)
        events.append(f"actor:{artifact.root.name}")
        return object()

    def learned_rows(actor: object, artifact: object):
        del actor
        manifest = artifact.manifest
        return _actor_rows(
            profile=manifest.profile,
            actor_id=f"{manifest.trainer.value}-{manifest.seed}",
            trainer=manifest.trainer,
            seed=manifest.seed,
        )

    def baseline_rows(kind: BaselineKind, *, profile: ProfileName):
        return _actor_rows(
            profile=profile,
            actor_id=kind.value,
            trainer=None,
            seed=None,
        )

    monkeypatch.setattr(workflows, "load_artifact", load)
    monkeypatch.setattr(workflows, "_peek_artifact_identity", peek)
    monkeypatch.setattr(workflows, "_validate_artifact_payload", validate)
    monkeypatch.setattr(workflows, "_load_artifact_actor", load_actor)
    monkeypatch.setattr(workflows, "_evaluate_learned_rows", learned_rows)
    monkeypatch.setattr(workflows, "_evaluate_baseline_rows", baseline_rows)
    monkeypatch.setattr(workflows, "_bc_heldout_accuracy", lambda artifact: 0.75)


def test_report_codec_round_trips_strict_canonical_json() -> None:
    report = _report(
        learned=((TrainerKind.BC, 0), (TrainerKind.PPO, 2)),
        ppo_criterion=PPOScientificCriterion(False, None, None, None, "ineligible"),
    )

    content = workflows.evaluation_report_bytes(report)

    assert content.endswith(b"\n") and not content.endswith(b"\n\n")
    assert b"NaN" not in content and b"Infinity" not in content
    assert workflows.evaluation_report_from_bytes(content) == report
    assert content == workflows.evaluation_report_bytes(report)


@pytest.mark.parametrize("missing", tuple(BaselineKind))
def test_report_constructor_requires_every_baseline_exactly_once(
    missing: BaselineKind,
) -> None:
    rows = tuple(
        row
        for row in _report_rows(profile=ProfileName.SMOKE, learned=((TrainerKind.BC, 0),))
        if row.actor_id != missing.value
    )

    with pytest.raises(ValueError):
        _report_from_rows(
            profile=ProfileName.SMOKE,
            device=DeviceName.CPU,
            rows=rows,
            bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
            ppo_criterion=None,
        )


@pytest.mark.parametrize("probe", [None, ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE])
def test_report_constructor_requires_every_learned_profile_lane(
    probe: str | None,
) -> None:
    rows = tuple(
        row
        for row in _report_rows(
            profile=ProfileName.SMOKE,
            learned=((TrainerKind.BC, 0),),
        )
        if not (row.actor_id == "bc-0" and row.probe == probe)
    )

    with pytest.raises(ValueError):
        _report_from_rows(
            profile=ProfileName.SMOKE,
            device=DeviceName.CPU,
            rows=rows,
            bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
            ppo_criterion=None,
        )


@pytest.mark.parametrize(
    ("actor_id", "suite_id", "subset", "probe"),
    [
        ("bc-0", EvaluationSuiteId.IID, "combined", ZERO_SPECTRUM_PROBE),
        ("bc-0", EvaluationSuiteId.REGISTER_OOD, "lower", None),
        (BaselineKind.RANDOM.value, EvaluationSuiteId.REGISTER_OOD, "upper", None),
    ],
)
def test_report_constructor_requires_every_checkpoint_profile_lane(
    actor_id: str,
    suite_id: EvaluationSuiteId,
    subset: str,
    probe: str | None,
) -> None:
    rows = tuple(
        row
        for row in _report_rows(
            profile=ProfileName.CHECKPOINT,
            learned=((TrainerKind.BC, 0),),
        )
        if (row.actor_id, row.suite_id, row.subset, row.probe)
        != (actor_id, suite_id, subset, probe)
    )

    with pytest.raises(ValueError):
        _report_from_rows(
            profile=ProfileName.CHECKPOINT,
            device=DeviceName.CPU,
            rows=rows,
            bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
            ppo_criterion=None,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "extra"])
def test_report_constructor_rejects_duplicate_and_extra_lanes(mutation: str) -> None:
    rows = _report_rows(profile=ProfileName.SMOKE, learned=((TrainerKind.BC, 0),))
    added = rows[1] if mutation == "duplicate" else replace(rows[0], subset="lower")
    mutated = (*rows[:3], added, *rows[3:])

    with pytest.raises(ValueError):
        _report_from_rows(
            profile=ProfileName.SMOKE,
            device=DeviceName.CPU,
            rows=mutated,
            bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
            ppo_criterion=None,
        )


def test_report_document_and_bytes_reject_incomplete_lane_matrix() -> None:
    report = _report()
    document = copy.deepcopy(report.to_document())
    document["rows"] = [
        row for row in document["rows"] if row["actor_id"] != BaselineKind.ORACLE.value
    ]

    with pytest.raises(ValueError):
        workflows.EvaluationReport.from_document(document)
    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(workflows.canonical_json_bytes(document))


def test_report_recomputes_eligible_bc_criterion_from_retained_iid_row() -> None:
    rows = _report_rows(
        profile=ProfileName.CHECKPOINT,
        learned=((TrainerKind.BC, 0),),
    )
    criterion = workflows.evaluate_bc_criterion(
        heldout_next_action_accuracy=0.95,
        iid_row=_base_iid_row(rows, actor_id="bc-0"),
        eligible=True,
    )
    report = _report_from_rows(
        profile=ProfileName.CHECKPOINT,
        device=DeviceName.CPU,
        rows=rows,
        bc_criterion=criterion,
        ppo_criterion=None,
    )
    document = copy.deepcopy(report.to_document())
    document["bc_criterion"]["iid_submitted_success_rate"] = 0.5

    with pytest.raises(ValueError):
        workflows.EvaluationReport.from_document(document)
    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(workflows.canonical_json_bytes(document))


@pytest.mark.parametrize("mutation", ["status", "profile", "device"])
def test_report_rejects_visible_bc_eligibility_contradictions(mutation: str) -> None:
    rows = _report_rows(
        profile=ProfileName.CHECKPOINT,
        learned=((TrainerKind.BC, 0),),
    )
    criterion = workflows.evaluate_bc_criterion(
        heldout_next_action_accuracy=0.95,
        iid_row=_base_iid_row(rows, actor_id="bc-0"),
        eligible=True,
    )
    report = _report_from_rows(
        profile=ProfileName.CHECKPOINT,
        device=DeviceName.CPU,
        rows=rows,
        bc_criterion=criterion,
        ppo_criterion=None,
    )
    document = copy.deepcopy(report.to_document())
    if mutation == "status":
        document["bc_criterion"].update(criterion_met=True, status="criterion_met")
    elif mutation == "profile":
        document["profile"] = ProfileName.SMOKE.value
    else:
        document["evaluation_device"] = DeviceName.CUDA.value

    with pytest.raises(ValueError):
        workflows.EvaluationReport.from_document(document)


@pytest.mark.parametrize(
    ("profile", "device"),
    [
        (ProfileName.SMOKE, DeviceName.CPU),
        (ProfileName.CHECKPOINT, DeviceName.CUDA),
    ],
)
def test_eligible_criteria_require_visible_checkpoint_cpu_evaluation(
    profile: ProfileName,
    device: DeviceName,
) -> None:
    with pytest.raises(ValueError):
        _report(
            profile=profile,
            device=device,
            bc_criterion=BCScientificCriterion(True, 0.95, 0.0, False, "criterion_not_met"),
        )


def test_report_recomputes_eligible_ppo_criterion_from_retained_paired_rows() -> None:
    learned = tuple((TrainerKind.PPO, seed) for seed in range(5))
    rows = _report_rows(profile=ProfileName.CHECKPOINT, learned=learned)
    criterion = workflows.evaluate_ppo_criterion(
        iid_rows=tuple(_base_iid_row(rows, actor_id=f"ppo-{seed}") for seed in range(5)),
        random_iid_row=_base_iid_row(rows, actor_id=BaselineKind.RANDOM.value),
        eligible=True,
    )
    report = _report_from_rows(
        profile=ProfileName.CHECKPOINT,
        device=DeviceName.CPU,
        rows=rows,
        bc_criterion=None,
        ppo_criterion=criterion,
    )
    document = copy.deepcopy(report.to_document())
    document["ppo_criterion"].update(
        median_iid_submitted_success_rate=0.5,
        seeds_strictly_beating_random=5,
        criterion_met=True,
        status="criterion_met",
    )

    with pytest.raises(ValueError):
        workflows.EvaluationReport.from_document(document)
    with pytest.raises(ValueError):
        workflows.evaluation_report_from_bytes(workflows.canonical_json_bytes(document))


@pytest.mark.parametrize(
    "learned",
    [
        ((TrainerKind.BC, 1),),
        ((TrainerKind.BC, 0), (TrainerKind.BC, 1)),
    ],
)
def test_eligible_bc_criterion_requires_exact_visible_seed_zero_cardinality(
    learned: tuple[tuple[TrainerKind, int], ...],
) -> None:
    with pytest.raises(ValueError):
        _report(
            profile=ProfileName.CHECKPOINT,
            learned=learned,
            bc_criterion=BCScientificCriterion(True, 0.95, 0.0, False, "criterion_not_met"),
        )


@pytest.mark.parametrize("seeds", [(0, 1, 2, 3), (0, 1, 2, 3, 4, 5)])
def test_eligible_ppo_criterion_requires_exact_visible_seed_set(
    seeds: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _report(
            profile=ProfileName.CHECKPOINT,
            learned=tuple((TrainerKind.PPO, seed) for seed in seeds),
            bc_criterion=None,
            ppo_criterion=PPOScientificCriterion(True, 0.0, 0, False, "criterion_not_met"),
        )


def test_eligible_ppo_criterion_rejects_ineligible_visible_bc_companion() -> None:
    learned = ((TrainerKind.BC, 0), *((TrainerKind.PPO, seed) for seed in range(5)))

    with pytest.raises(ValueError):
        _report(
            profile=ProfileName.CHECKPOINT,
            learned=learned,
            bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
            ppo_criterion=PPOScientificCriterion(True, 0.0, 0, False, "criterion_not_met"),
        )


def test_checkpoint_cpu_report_allows_ineligible_exploratory_actor_seeds() -> None:
    report = _report(
        profile=ProfileName.CHECKPOINT,
        learned=((TrainerKind.BC, 2), (TrainerKind.PPO, 7)),
        bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
        ppo_criterion=PPOScientificCriterion(False, None, None, None, "ineligible"),
    )

    assert report.bc_criterion is not None and not report.bc_criterion.eligible
    assert report.ppo_criterion is not None and not report.ppo_criterion.eligible


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update(extra=True),
        lambda document: document.pop("rows"),
        lambda document: document["rows"][0]["metrics"].update(mean_return=float("nan")),
        lambda document: document["bc_criterion"].update(eligible="true"),
    ],
)
def test_report_codec_rejects_extra_missing_nonfinite_and_wrong_scalar_types(mutation) -> None:
    report = _report()
    document = copy.deepcopy(report.to_document())
    mutation(document)

    with pytest.raises(ValueError):
        workflows.EvaluationReport.from_document(document)


def test_evaluate_validates_every_artifact_before_actor_construction_and_canonicalizes_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bc_root = tmp_path / "bc"
    ppo_root = tmp_path / "ppo"
    bc_root.mkdir()
    ppo_root.mkdir()
    artifacts = {
        bc_root.resolve(): _artifact(
            bc_root,
            _manifest(trainer=TrainerKind.BC, seed=2),
        ),
        ppo_root.resolve(): _artifact(
            ppo_root,
            _manifest(trainer=TrainerKind.PPO, seed=3),
        ),
    }
    events: list[str] = []
    _install_fast_workflow(monkeypatch, artifacts, events)

    report = workflows.evaluate_artifacts((ppo_root, bc_root), device=DeviceName.CPU)

    assert max(index for index, item in enumerate(events) if item.startswith("validate:")) < min(
        index for index, item in enumerate(events) if item.startswith("actor:")
    )
    assert tuple(dict.fromkeys(row.actor_id for row in report.rows)) == (
        "bc-2",
        "ppo-3",
        "random",
        "reward_search",
        "spectrum_peak",
        "oracle",
    )


def test_evaluate_rejects_resolved_alias_duplicate_before_artifact_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: pytest.fail(f"duplicate path reached artifact loading: {path}"),
    )

    with pytest.raises(LearningContractError, match="duplicate artifact path"):
        workflows.evaluate_artifacts((root, alias))


@pytest.mark.parametrize("mismatch", ["identity", "profile", "architecture"])
def test_evaluate_rejects_duplicate_identity_and_compatibility_before_actors(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_manifest = _manifest(trainer=TrainerKind.PPO, seed=0)
    second_manifest = _manifest(
        trainer=(TrainerKind.PPO if mismatch == "identity" else TrainerKind.BC),
        seed=0,
        profile=(ProfileName.CHECKPOINT if mismatch == "profile" else ProfileName.SMOKE),
    )
    if mismatch == "architecture":
        second_manifest.architecture_schema_id = "other-architecture"
    artifacts = {
        first_root.resolve(): _artifact(first_root, first_manifest),
        second_root.resolve(): _artifact(second_root, second_manifest),
    }
    monkeypatch.setattr(workflows, "load_artifact", lambda path: artifacts[path.resolve()])
    monkeypatch.setattr(
        workflows,
        "_peek_artifact_identity",
        lambda path: (
            1,
            artifacts[path.resolve()].manifest.profile,
            artifacts[path.resolve()].manifest.trainer.value,
        ),
    )
    monkeypatch.setattr(
        workflows,
        "_load_artifact_actor",
        lambda *args, **kwargs: pytest.fail("invalid set constructed an actor"),
    )

    with pytest.raises(LearningContractError):
        workflows.evaluate_artifacts((first_root, second_root))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("none", True),
        ("smoke", False),
        ("cuda_training", False),
        ("cuda_evaluation", False),
        ("dirty", False),
        ("uncommitted", False),
        ("undeclared_seed", False),
        ("extra_seed", False),
    ],
)
def test_ppo_criterion_eligibility_is_derived_only_from_the_complete_manifest_set(
    mutation: str,
    expected: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds = list(range(5))
    if mutation == "extra_seed":
        seeds.append(5)
    profile = ProfileName.SMOKE if mutation == "smoke" else ProfileName.CHECKPOINT
    artifacts: dict[Path, object] = {}
    paths: list[Path] = []
    for seed in seeds:
        root = tmp_path / f"ppo-{seed}"
        root.mkdir()
        paths.append(root)
        manifest = _manifest(
            trainer=TrainerKind.PPO,
            seed=(7 if mutation == "undeclared_seed" and seed == 4 else seed),
            profile=profile,
            training_device=(
                DeviceName.CUDA if mutation == "cuda_training" and seed == 0 else DeviceName.CPU
            ),
            evaluation_device=(
                DeviceName.CUDA if mutation == "cuda_evaluation" and seed == 0 else DeviceName.CPU
            ),
            dirty_tree=mutation == "dirty" and seed == 0,
            required_inputs_committed=not (mutation == "uncommitted" and seed == 0),
        )
        artifacts[root.resolve()] = _artifact(root, manifest)
    events: list[str] = []
    _install_fast_workflow(monkeypatch, artifacts, events)
    eligibility: list[bool] = []

    def ppo_criterion(*, iid_rows, random_iid_row, eligible: bool):
        del iid_rows, random_iid_row
        eligibility.append(eligible)
        return (
            PPOScientificCriterion(True, 0.5, 5, True, "criterion_met")
            if eligible
            else PPOScientificCriterion(False, None, None, None, "ineligible")
        )

    monkeypatch.setattr(workflows, "evaluate_ppo_criterion", ppo_criterion)

    report = workflows.evaluate_artifacts(paths)

    assert eligibility == ([expected, True] if expected else [False])
    assert report.ppo_criterion is not None
    assert report.ppo_criterion.eligible is expected


def test_evaluation_report_bytes_excludes_artifact_paths_and_fresh_timing(
    tmp_path: Path,
) -> None:
    report = _report(
        learned=((TrainerKind.PPO, 0),),
        bc_criterion=None,
        ppo_criterion=PPOScientificCriterion(False, None, None, None, "ineligible"),
    )

    content = workflows.evaluation_report_bytes(report)

    assert str(tmp_path).encode() not in content
    assert b"evaluated_at" not in content
    assert b"evaluation_wall_time" not in content
    assert b'"training_wall_time_seconds":12.5' in content


def test_evaluation_report_decoder_uses_task6_strict_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bytes] = []

    def decode(content: bytes):
        calls.append(content)
        raise ValueError("strict decoder sentinel")

    monkeypatch.setattr(workflows, "decode_json_bytes", decode)

    with pytest.raises(ValueError, match="strict decoder sentinel"):
        workflows.evaluation_report_from_bytes(b"{}")
    assert calls == [b"{}"]
