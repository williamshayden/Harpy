"""Real optional-stack smoke coverage for learned-policy user workflows."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
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


@pytest.fixture(scope="module")
def installed_wheel_cli() -> Iterator[Callable[..., subprocess.CompletedProcess[bytes]]]:
    """Exercise installed code outside Git with the current optional dependencies."""
    with tempfile.TemporaryDirectory(prefix="harpy-installed-smoke-") as temporary:
        root = Path(temporary)
        built = root / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(built)],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            timeout=180,
        )
        (wheel,) = built.glob("*.whl")
        target = root / "site-packages"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(target),
                "--no-deps",
                str(wheel),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=180,
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(target)
        probe = subprocess.run(
            [sys.executable, "-c", "import harpy; print(harpy.__file__)"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        )
        assert Path(probe.stdout.decode().strip()) == target / "harpy" / "__init__.py"

        def run(*arguments: str, timeout: int = 600) -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                [sys.executable, "-m", "harpy", *arguments],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                timeout=timeout,
            )

        yield run


def test_isolated_no_train_command_fails_before_output(tmp_path: Path) -> None:
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
                "        blocked = ('torch', 'stable_baselines3')",
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
            "--locked",
            "harpy",
            "train",
            "ppo",
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
    assert result.returncode == 2
    assert result.stdout == b""
    assert result.stderr.decode("utf-8").splitlines()[-1] == (
        "harpy: The optional training stack is unavailable. Install it with "
        "`python -m pip install 'harpy-audio[train]'`."
    )
    assert result.stderr.count(b"harpy-audio[train]") == 1
    assert not output.exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional pitch group unavailable"
)
def test_installed_pitch_smoke_cli_trains_reloads_evaluates_and_traces(
    tmp_path: Path,
    installed_wheel_cli: Callable[..., subprocess.CompletedProcess[bytes]],
) -> None:
    """The supported CLI crosses persisted boundaries outside the source checkout."""
    from harpy.experiments.artifacts import load_artifact
    from harpy.experiments.results import load_result

    artifact_path = tmp_path / "pitch-smoke"
    training = installed_wheel_cli(
        "train",
        "pitch",
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
    artifact = load_artifact(artifact_path)
    assert artifact.manifest.trainer == "pitch"
    assert artifact.manifest.profile == "smoke"
    assert artifact.manifest.seed == 0
    assert artifact.manifest.device == "cpu"
    source = json.loads(artifact.manifest.source_bytes)
    assert source["status"] == "known"
    assert source["identity"]["source_kind"] == "package_snapshot"
    assert source["identity"]["distribution_version"] is not None
    metadata = _assert_strict_json((artifact_path / "metadata.json").read_bytes())
    assert metadata["runtime"]["stable_baselines3"] is None
    assert artifact.load_model() is not None
    artifact.verify_unchanged()

    report_path = tmp_path / "pitch-smoke-report.json"
    evaluation = installed_wheel_cli(
        "evaluate",
        "--actor",
        str(artifact_path),
        "--suite",
        "smoke",
        "--output",
        str(report_path),
    )
    assert evaluation.returncode == 0, evaluation.stderr.decode("utf-8")
    report = load_result(report_path)
    assert len(report.records) == 6
    assert report.actors[0]["observation_mode"] == "spectrum"
    assert report.protocol["id"] == "harpy-clean-engineering-smoke-v1"
    summary = installed_wheel_cli("summarize", str(report_path), "--format", "markdown")
    assert summary.returncode == 0, summary.stderr.decode("utf-8")
    assert b"pitch-0" in summary.stdout

    first_path, second_path = tmp_path / "first.json", tmp_path / "second.json"
    for path in (first_path, second_path):
        result = installed_wheel_cli(
            "run",
            "--actor",
            str(artifact_path),
            "--source-cents",
            "6137",
            "--target-note-index",
            "12",
            "--output",
            str(path),
        )
        assert result.returncode == 0, result.stderr.decode("utf-8")
    first, second = load_result(first_path), load_result(second_path)
    assert first.records == second.records
    assert first.records[0].trace
    assert len(first.records[0].trace) == len(first.records[0].terminal.actions)
    artifact.verify_unchanged()


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
