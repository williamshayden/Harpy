"""Real optional-stack smoke coverage for learned-policy user workflows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_STACK_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("torch", "stable_baselines3")
)
_REQUIRES_TRAIN_STACK = pytest.mark.skipif(
    not _TRAIN_STACK_AVAILABLE,
    reason="optional train group is unavailable",
)


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _assert_strict_json(content: bytes) -> dict[str, object]:
    document = json.loads(
        content.decode("utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    assert isinstance(document, dict)
    return document


def _assert_canonical_trace(content: bytes, *, expected_seed: int) -> tuple[int, ...]:
    from harpy.envs.models import (
        MAX_STEPS,
        SUCCESS_TOLERANCE_CENTS,
        PitchAction,
        TerminalReason,
    )
    from harpy.learning.artifacts import canonical_json_bytes, decode_json_bytes
    from harpy.learning.models import ENVIRONMENT_ID
    from harpy.learning.trace import (
        FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID,
        EpisodeTrace,
        TraceStep,
        trace_json_bytes,
    )

    document = decode_json_bytes(content)
    assert canonical_json_bytes(document) == content
    assert set(document) == {
        "environment_id",
        "distribution_id",
        "seed",
        "target_note_index",
        "target_note",
        "steps",
        "terminal_reason",
        "final_absolute_error_cents",
        "submitted_success",
        "action_count",
        "excess_actions",
        "total_return",
    }
    assert document["environment_id"] == ENVIRONMENT_ID
    assert document["distribution_id"] == FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID
    assert type(document["seed"]) is int
    assert document["seed"] == expected_seed
    assert type(document["target_note_index"]) is int
    assert type(document["target_note"]) is str
    assert type(document["terminal_reason"]) is str
    assert type(document["final_absolute_error_cents"]) is int
    assert type(document["submitted_success"]) is bool
    assert type(document["action_count"]) is int
    assert document["excess_actions"] is None or type(document["excess_actions"]) is int
    assert type(document["total_return"]) is float

    raw_steps = document["steps"]
    assert type(raw_steps) is list
    assert raw_steps
    steps: list[TraceStep] = []
    for expected_step, raw_step in enumerate(raw_steps, start=1):
        assert type(raw_step) is dict
        assert set(raw_step) == {"step", "action", "reward"}
        assert type(raw_step["step"]) is int
        assert raw_step["step"] == expected_step
        assert type(raw_step["action"]) is int
        assert type(raw_step["reward"]) is float
        steps.append(
            TraceStep(
                step=raw_step["step"],
                action=PitchAction(raw_step["action"]),
                reward=raw_step["reward"],
            )
        )

    episode = EpisodeTrace(
        environment_id=document["environment_id"],
        distribution_id=document["distribution_id"],
        seed=document["seed"],
        target_note_index=document["target_note_index"],
        target_note=document["target_note"],
        steps=tuple(steps),
        terminal_reason=TerminalReason(document["terminal_reason"]),
        final_absolute_error_cents=document["final_absolute_error_cents"],
        submitted_success=document["submitted_success"],
        action_count=document["action_count"],
        excess_actions=document["excess_actions"],
        total_return=document["total_return"],
    )
    assert trace_json_bytes(episode) == content
    assert episode.action_count == len(episode.steps)

    actions = tuple(step.action for step in episode.steps)
    if episode.terminal_reason is TerminalReason.SUBMITTED_SUCCESS:
        assert actions[-1] is PitchAction.SUBMIT
        assert episode.submitted_success is True
        assert episode.final_absolute_error_cents <= SUCCESS_TOLERANCE_CENTS
        assert episode.excess_actions is not None
    elif episode.terminal_reason is TerminalReason.SUBMITTED_FAILURE:
        assert actions[-1] is PitchAction.SUBMIT
        assert episode.submitted_success is False
        assert episode.final_absolute_error_cents > SUCCESS_TOLERANCE_CENTS
        assert episode.excess_actions is None
    else:
        assert episode.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
        assert actions[-1] is not PitchAction.SUBMIT
        assert episode.submitted_success is False
        assert episode.action_count == MAX_STEPS
        assert episode.excess_actions is None
    return tuple(int(action) for action in actions)


def _assert_closed_hashed_inventory(artifact: object) -> None:
    from harpy.learning.artifacts import LoadedArtifact, required_payload_names

    assert isinstance(artifact, LoadedArtifact)
    expected_payloads = required_payload_names(
        artifact.manifest.trainer,
        artifact.manifest.profile,
    )
    assert tuple(record.relative_path for record in artifact.manifest.files) == expected_payloads
    assert {path.name for path in artifact.root.iterdir()} == {"manifest.json", *expected_payloads}
    for record in artifact.manifest.files:
        payload = artifact.file(record.relative_path)
        content = payload.read_bytes()
        assert record.size_bytes == len(content)
        assert record.sha256 == hashlib.sha256(content).hexdigest()
        if payload.suffix == ".json":
            _assert_strict_json(content)


def _assert_pitch_closed_hashed_inventory(artifact: object) -> None:
    from harpy.learning.pitch_artifacts import (
        PITCH_REQUIRED_PAYLOAD_NAMES,
        LoadedPitchArtifact,
    )

    assert isinstance(artifact, LoadedPitchArtifact)
    assert (
        tuple(record.relative_path for record in artifact.manifest.files)
        == PITCH_REQUIRED_PAYLOAD_NAMES
    )
    assert {path.name for path in artifact.root.iterdir()} == {
        "manifest.json",
        *PITCH_REQUIRED_PAYLOAD_NAMES,
    }
    _assert_strict_json((artifact.root / "manifest.json").read_bytes())
    for record in artifact.manifest.files:
        payload = artifact.file(record.relative_path)
        content = payload.read_bytes()
        assert record.size_bytes == len(content)
        assert record.sha256 == hashlib.sha256(content).hexdigest()
        if payload.suffix == ".json":
            _assert_strict_json(content)


def _run_learning_cli(*arguments: str, timeout: int = 600) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-m", "harpy.learning.cli", *arguments],
        cwd=_REPOSITORY_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _assert_one_trusted_local_warning(result: subprocess.CompletedProcess[bytes]) -> None:
    from harpy.learning.cli import TRUSTED_LOCAL_MODEL_WARNING

    assert result.stderr.decode("utf-8").splitlines() == [TRUSTED_LOCAL_MODEL_WARNING]


def test_isolated_no_train_command_fails_before_output_or_qt_import(tmp_path: Path) -> None:
    """A default install must fail with one actionable dependency diagnostic."""

    poison_directory = tmp_path / "poison"
    poison_directory.mkdir()
    marker = poison_directory / "sitecustomize-loaded"
    (poison_directory / "sitecustomize.py").write_text(
        "\n".join(
            (
                "import importlib.abc",
                "import pathlib",
                "import sys",
                f"pathlib.Path({str(marker)!r}).write_text('loaded', encoding='utf-8')",
                "class _PoisonFinder(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        del path, target",
                "        blocked = ('torch', 'stable_baselines3', 'PySide6')",
                "        if any(",
                "            fullname == name or fullname.startswith(name + '.')",
                "            for name in blocked",
                "        ):",
                "            raise ModuleNotFoundError(",
                '                f"No module named {fullname!r}", name=fullname',
                "            )",
                "        return None",
                "sys.meta_path.insert(0, _PoisonFinder())",
                "",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "new-artifact"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(poison_directory)

    result = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-group",
            "train",
            "--locked",
            "harpy-sine-learn",
            "train-bc",
            "--profile",
            "smoke",
            "--seed",
            "0",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        check=False,
        timeout=180,
    )

    assert marker.read_text(encoding="utf-8") == "loaded"
    assert result.returncode == 1
    assert result.stdout == b""
    assert result.stderr.decode("utf-8").splitlines()[-1] == (
        "error: The optional training stack is unavailable. Install it with "
        "`uv sync --group train`."
    )
    assert result.stderr.count(b"uv sync --group train") == 1
    assert b"PySide6" not in result.stderr
    assert not output.exists()


@_REQUIRES_TRAIN_STACK
def test_real_pitch_smoke_cli_trains_reloads_diagnoses_evaluates_and_traces(
    tmp_path: Path,
) -> None:
    """The public pitch CLI must cross every persisted Milestone E smoke boundary."""

    from harpy.learning.artifacts import (
        PITCH_ARTIFACT_SCHEMA_VERSION,
        ArtifactStatus,
        CriterionStatus,
    )
    from harpy.learning.diagnostic_codecs import (
        diagnostic_bundle_bytes,
        diagnostic_bundle_from_bytes,
    )
    from harpy.learning.models import DeviceName, ProfileName
    from harpy.learning.pitch_artifacts import load_pitch_artifact
    from harpy.learning.pitch_data import PitchEvaluationSuiteId, PitchTrainerKind
    from harpy.learning.pitch_reports import PitchEvaluationReport
    from harpy.learning.workflows import (
        evaluation_report_bytes,
        evaluation_report_from_bytes,
    )

    artifact_path = tmp_path / "pitch-smoke"
    diagnostic_path = tmp_path / "pitch-smoke-diagnostics.json"
    report_path = tmp_path / "pitch-smoke-report.json"
    artifact_resolved = artifact_path.resolve(strict=False)
    assert not diagnostic_path.resolve(strict=False).is_relative_to(artifact_resolved)
    assert not report_path.resolve(strict=False).is_relative_to(artifact_resolved)
    assert not artifact_path.exists()
    assert not diagnostic_path.exists()
    assert not report_path.exists()

    training = _run_learning_cli(
        "train-pitch",
        "--profile",
        "smoke",
        "--seed",
        "0",
        "--output",
        str(artifact_path),
        "--device",
        "cpu",
    )
    assert training.returncode == 0, training.stderr.decode("utf-8")
    assert training.stdout == b""

    artifact = load_pitch_artifact(artifact_path)
    assert artifact.manifest.schema_version == PITCH_ARTIFACT_SCHEMA_VERSION
    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact.manifest.trainer is PitchTrainerKind.PITCH
    assert artifact.manifest.profile is ProfileName.SMOKE
    assert artifact.manifest.seed == 0
    assert artifact.manifest.runtime.device is DeviceName.CPU
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact.manifest.eligible_for_aggregate is False
    assert artifact.manifest.criterion_status is CriterionStatus.INELIGIBLE
    assert artifact.manifest.criterion_met is None
    _assert_pitch_closed_hashed_inventory(artifact)

    diagnosis = _run_learning_cli(
        "diagnose",
        str(artifact_path),
        "--suite",
        "smoke",
        "--output",
        str(diagnostic_path),
        "--device",
        "cpu",
    )
    assert diagnosis.returncode == 0, diagnosis.stderr.decode("utf-8")
    assert diagnosis.stdout == diagnostic_path.read_bytes()
    _assert_one_trusted_local_warning(diagnosis)
    diagnostic_bundle = diagnostic_bundle_from_bytes(diagnosis.stdout)
    assert diagnostic_bundle_bytes(diagnostic_bundle) == diagnosis.stdout
    assert diagnostic_bundle.suite_id == PitchEvaluationSuiteId.SMOKE.value
    assert diagnostic_bundle.device == DeviceName.CPU.value
    assert diagnostic_bundle.bound_mask is False
    assert len(diagnostic_bundle.reports) == 1
    assert diagnostic_bundle.reports[0].seed == 0
    assert len(diagnostic_bundle.reports[0].episodes) == 50

    evaluation = _run_learning_cli(
        "evaluate",
        str(artifact_path),
        "--output",
        str(report_path),
        "--device",
        "cpu",
    )
    assert evaluation.returncode == 0, evaluation.stderr.decode("utf-8")
    assert evaluation.stdout == report_path.read_bytes()
    _assert_one_trusted_local_warning(evaluation)
    report = evaluation_report_from_bytes(evaluation.stdout)
    assert isinstance(report, PitchEvaluationReport)
    assert evaluation_report_bytes(report) == evaluation.stdout
    assert report.profile is ProfileName.SMOKE
    assert report.evaluation_device is DeviceName.CPU
    assert len(report.terminal_rows) == 7
    assert report.coordinate_evaluations == ()
    assert report.register_ood_aggregates == ()
    assert report.criterion.status == "ineligible"

    run_arguments = (
        "run",
        str(artifact_path),
        "--seed",
        "123",
        "--json",
        "--device",
        "cpu",
    )
    first_trace = _run_learning_cli(*run_arguments)
    second_trace = _run_learning_cli(*run_arguments)
    assert first_trace.returncode == 0, first_trace.stderr.decode("utf-8")
    assert second_trace.returncode == 0, second_trace.stderr.decode("utf-8")
    _assert_one_trusted_local_warning(first_trace)
    _assert_one_trusted_local_warning(second_trace)
    assert first_trace.stdout == second_trace.stdout
    first_actions = _assert_canonical_trace(first_trace.stdout, expected_seed=123)
    second_actions = _assert_canonical_trace(second_trace.stdout, expected_seed=123)
    assert first_actions == second_actions


@_REQUIRES_TRAIN_STACK
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_real_smoke_trains_reloads_evaluates_and_traces_both_policies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checked-in smoke profile must cross every persisted production boundary."""

    from harpy.envs.models import PitchAction
    from harpy.envs.sine_pitch import SinePitchEnv
    from harpy.learning import bc as bc_module
    from harpy.learning import ppo as ppo_module
    from harpy.learning.artifacts import (
        ArtifactStatus,
        BCTrainingCounts,
        LoadedArtifact,
        PendingArtifactView,
        PPOTrainingCounts,
        load_artifact,
        read_training_config,
        read_training_summary,
    )
    from harpy.learning.evaluation import (
        SHUFFLED_SPECTRUM_PROBE,
        ZERO_SPECTRUM_PROBE,
        EvaluationFile,
    )
    from harpy.learning.models import (
        PROFILE_CONFIGS,
        DeviceName,
        ProfileName,
        TrainerKind,
    )
    from harpy.learning.trace import (
        FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID,
        trace_json_bytes,
    )
    from harpy.learning.workflows import (
        evaluate_artifacts,
        evaluation_report_bytes,
        evaluation_report_from_bytes,
        run_artifact,
    )

    persisted_reloads: list[tuple[TrainerKind, ArtifactStatus, tuple[str, ...]]] = []
    real_bc_loader = bc_module.load_bc_actor
    real_ppo_loader = ppo_module.load_ppo_actor

    def tracked_bc_loader(
        artifact: LoadedArtifact | PendingArtifactView,
        *,
        device: DeviceName = DeviceName.CPU,
    ) -> object:
        persisted_reloads.append(
            (
                TrainerKind.BC,
                artifact.manifest.status,
                tuple(
                    record.relative_path
                    for record in (
                        artifact.files
                        if isinstance(artifact, PendingArtifactView)
                        else artifact.manifest.files
                    )
                ),
            )
        )
        assert artifact.file("model.pt").is_file()
        return real_bc_loader(artifact, device=device)

    def tracked_ppo_loader(
        artifact: LoadedArtifact | PendingArtifactView,
        *,
        device: DeviceName = DeviceName.CPU,
    ) -> object:
        persisted_reloads.append(
            (
                TrainerKind.PPO,
                artifact.manifest.status,
                tuple(
                    record.relative_path
                    for record in (
                        artifact.files
                        if isinstance(artifact, PendingArtifactView)
                        else artifact.manifest.files
                    )
                ),
            )
        )
        assert artifact.file("model.zip").is_file()
        return real_ppo_loader(artifact, device=device)

    monkeypatch.setattr(bc_module, "load_bc_actor", tracked_bc_loader)
    monkeypatch.setattr(ppo_module, "load_ppo_actor", tracked_ppo_loader)

    bc_path = tmp_path / "bc-smoke"
    ppo_path = tmp_path / "ppo-smoke"
    bc_artifact = bc_module.train_bc_artifact(
        profile=ProfileName.SMOKE,
        seed=0,
        device=DeviceName.CPU,
        output=bc_path,
    )
    ppo_artifact = ppo_module.train_ppo_artifact(
        profile=ProfileName.SMOKE,
        seed=0,
        device=DeviceName.CPU,
        output=ppo_path,
    )

    assert persisted_reloads[:2] == [
        (
            TrainerKind.BC,
            ArtifactStatus.INCOMPLETE,
            ("training-config.json", "training-summary.json", "model.pt"),
        ),
        (
            TrainerKind.PPO,
            ArtifactStatus.INCOMPLETE,
            ("training-config.json", "training-summary.json", "model.zip"),
        ),
    ]
    for artifact in (bc_artifact, ppo_artifact):
        assert artifact.manifest.status is ArtifactStatus.COMPLETE
        assert artifact.manifest.completed_at_utc is not None
        assert artifact.manifest.runtime.device is DeviceName.CPU
        assert artifact.manifest.evaluation_device is DeviceName.CPU
        assert artifact.manifest.criterion_eligible is False
        assert artifact.manifest.criterion_met is None
        _assert_closed_hashed_inventory(artifact)
        assert load_artifact(artifact.root).manifest == artifact.manifest

    smoke_profile = PROFILE_CONFIGS[ProfileName.SMOKE]
    bc_config = read_training_config(bc_artifact)
    bc_summary = read_training_summary(bc_artifact).summary
    assert bc_config.profile_config == smoke_profile.bc
    assert isinstance(bc_artifact.manifest.training_counts, BCTrainingCounts)
    assert (
        bc_artifact.manifest.training_counts.configured_training_episodes,
        bc_artifact.manifest.training_counts.configured_validation_episodes,
        bc_artifact.manifest.training_counts.training_examples,
        bc_artifact.manifest.training_counts.validation_examples,
    ) == (128, 64, 2_947, 1_476)
    assert (bc_summary.training_examples, bc_summary.validation_examples) == (2_947, 1_476)

    ppo_config = read_training_config(ppo_artifact)
    ppo_summary = read_training_summary(ppo_artifact).summary
    assert ppo_config.profile_config == smoke_profile.ppo
    assert isinstance(ppo_artifact.manifest.training_counts, PPOTrainingCounts)
    assert (
        ppo_artifact.manifest.training_counts.requested_environment_steps,
        ppo_artifact.manifest.training_counts.completed_environment_steps,
        ppo_summary.requested_environment_steps,
        ppo_summary.completed_environment_steps,
    ) == (2_048, 2_048, 2_048, 2_048)

    for artifact, trainer in (
        (bc_artifact, TrainerKind.BC),
        (ppo_artifact, TrainerKind.PPO),
    ):
        evaluation = EvaluationFile.from_document(artifact.document("evaluation-smoke.json"))
        assert tuple((row.subset, row.probe) for row in evaluation.rows) == (
            ("combined", None),
            ("combined", ZERO_SPECTRUM_PROBE),
            ("combined", SHUFFLED_SPECTRUM_PROBE),
        )
        assert all(row.trainer is trainer and row.metrics.episodes == 32 for row in evaluation.rows)

    environment = SinePitchEnv()
    try:
        observation, _ = environment.reset(
            options={"target_note_index": 12, "source_pitch_cents": 6_137}
        )
    finally:
        environment.close()
    for artifact, loader in (
        (bc_artifact, tracked_bc_loader),
        (ppo_artifact, tracked_ppo_loader),
    ):
        first_actor = loader(artifact)
        second_actor = loader(artifact)
        actions = (
            first_actor.act(observation),
            first_actor.act(observation),
            second_actor.act(observation),
        )
        assert all(isinstance(action, PitchAction) for action in actions)
        assert actions[0] is actions[1] is actions[2]

    for artifact in (bc_artifact, ppo_artifact):
        first_trace = run_artifact(artifact.root, seed=123)
        second_trace = run_artifact(artifact.root, seed=123)
        assert first_trace.distribution_id == FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID
        assert first_trace.seed == 123
        assert first_trace.action_count == len(first_trace.steps)
        first_trace_bytes = trace_json_bytes(first_trace)
        assert first_trace_bytes == trace_json_bytes(second_trace)
        _assert_strict_json(first_trace_bytes)

    report = evaluate_artifacts((bc_artifact.root, ppo_artifact.root))
    report_bytes = evaluation_report_bytes(report)
    assert evaluation_report_from_bytes(report_bytes) == report
    report_document = _assert_strict_json(report_bytes)
    assert len(report_document["rows"]) == 10
    assert tuple((row.actor_id, row.probe) for row in report.rows) == (
        ("bc-0", None),
        ("bc-0", ZERO_SPECTRUM_PROBE),
        ("bc-0", SHUFFLED_SPECTRUM_PROBE),
        ("ppo-0", None),
        ("ppo-0", ZERO_SPECTRUM_PROBE),
        ("ppo-0", SHUFFLED_SPECTRUM_PROBE),
        ("random", None),
        ("reward_search", None),
        ("spectrum_peak", None),
        ("oracle", None),
    )
    assert len(
        {(row.actor_id, row.trainer, row.seed, row.subset, row.probe) for row in report.rows}
    ) == len(report.rows)
    assert report.bc_criterion is not None and report.bc_criterion.status == "ineligible"
    assert report.ppo_criterion is not None and report.ppo_criterion.status == "ineligible"
