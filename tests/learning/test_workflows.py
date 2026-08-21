"""Cross-artifact evaluation orchestration and strict report contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import harpy.learning.workflows as workflows
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode, TerminalReason
from harpy.learning.errors import LearningContractError
from harpy.learning.evaluation import (
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


def _row(
    *,
    actor_id: str,
    trainer: TrainerKind | None,
    seed: int | None,
    suite_id: EvaluationSuiteId = EvaluationSuiteId.SMOKE,
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
        parameter_count=None if trainer is None else 75_000,
        training_environment_steps=(2_048 if trainer is TrainerKind.PPO else None),
        training_examples=(1_000 if trainer is TrainerKind.BC else None),
        training_wall_time_seconds=(12.5 if trainer is not None else None),
    )[0]


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
        suite_id = PROFILE_CONFIGS[manifest.profile].evaluation_suites[0]
        return (
            _row(
                actor_id=f"{manifest.trainer.value}-{manifest.seed}",
                trainer=manifest.trainer,
                seed=manifest.seed,
                suite_id=suite_id,
            ),
        )

    def baseline_rows(kind: BaselineKind, *, profile: ProfileName):
        suite_id = PROFILE_CONFIGS[profile].evaluation_suites[0]
        return (
            _row(
                actor_id=kind.value,
                trainer=None,
                seed=None,
                suite_id=suite_id,
            ),
        )

    monkeypatch.setattr(workflows, "load_artifact", load)
    monkeypatch.setattr(workflows, "_validate_artifact_payload", validate)
    monkeypatch.setattr(workflows, "_load_artifact_actor", load_actor)
    monkeypatch.setattr(workflows, "_evaluate_learned_rows", learned_rows)
    monkeypatch.setattr(workflows, "_evaluate_baseline_rows", baseline_rows)
    monkeypatch.setattr(workflows, "_bc_heldout_accuracy", lambda artifact: 0.75)


def test_report_codec_round_trips_strict_canonical_json() -> None:
    report = workflows.EvaluationReport(
        schema_version=1,
        profile=ProfileName.SMOKE,
        evaluation_device=DeviceName.CPU,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        rows=(
            _row(actor_id="bc-0", trainer=TrainerKind.BC, seed=0),
            _row(actor_id=BaselineKind.RANDOM.value, trainer=None, seed=None),
        ),
        bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
        ppo_criterion=None,
    )

    content = workflows.evaluation_report_bytes(report)

    assert content.endswith(b"\n") and not content.endswith(b"\n\n")
    assert b"NaN" not in content and b"Infinity" not in content
    assert workflows.evaluation_report_from_bytes(content) == report
    assert content == workflows.evaluation_report_bytes(report)


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
    report = workflows.EvaluationReport(
        schema_version=1,
        profile=ProfileName.SMOKE,
        evaluation_device=DeviceName.CPU,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        rows=(_row(actor_id="bc-0", trainer=TrainerKind.BC, seed=0),),
        bc_criterion=BCScientificCriterion(False, None, None, None, "ineligible"),
        ppo_criterion=None,
    )
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
    assert tuple(row.actor_id for row in report.rows) == (
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

    assert eligibility == [expected]
    assert report.ppo_criterion is not None
    assert report.ppo_criterion.eligible is expected


def test_evaluation_report_bytes_excludes_artifact_paths_and_fresh_timing(
    tmp_path: Path,
) -> None:
    row = _row(actor_id="ppo-0", trainer=TrainerKind.PPO, seed=0)
    report = workflows.EvaluationReport(
        schema_version=1,
        profile=ProfileName.SMOKE,
        evaluation_device=DeviceName.CPU,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        rows=(row,),
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
