"""Six-command learned sine-policy CLI and stable exit contracts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path

import pytest

import harpy.learning.cli as cli
from harpy.learning.errors import (
    ArtifactError,
    DependencyUnavailableError,
    LearningContractError,
    LearningExecutionError,
)
from harpy.learning.models import DeviceName, ProfileName

_V1_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures/learning/schema-v1"


def _write_pitch_preflight_manifest(
    root: Path,
    *,
    seed: int,
    eligible: bool = True,
    training_device: str = "cpu",
    evaluation_device: str = "cpu",
    dirty_tree: bool = False,
    required_inputs_committed: bool = True,
    commit: str = "test-commit",
    dependency_lock_sha256: str = "a" * 64,
    compatibility_sha256: str = "b" * 64,
) -> None:
    root.mkdir()
    document = {
        "schema_version": 2,
        "status": "complete",
        "trainer": "pitch",
        "profile": "checkpoint",
        "seed": seed,
        "source": {
            "commit": commit,
            "dirty_tree": dirty_tree,
            "dependency_lock_sha256": dependency_lock_sha256,
            "required_inputs_committed": required_inputs_committed,
        },
        "runtime": {"device": training_device},
        "evaluation_device": evaluation_device,
        "eligible_for_aggregate": eligible,
        "criterion_status": "eligible_for_aggregate" if eligible else "ineligible",
        "criterion_met": None,
        "compatibility_sha256": compatibility_sha256,
    }
    (root / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_parser_exposes_exact_six_command_grammar(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    captured = capsys.readouterr()
    assert raised.value.code == 0
    assert captured.err == ""
    assert "{train-bc,train-ppo,train-pitch,diagnose,evaluate,run}" in captured.out
    for command in (
        "train-bc",
        "train-ppo",
        "train-pitch",
        "diagnose",
        "evaluate",
        "run",
    ):
        assert command in captured.out


def test_diagnose_help_pins_exact_options_and_order(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["diagnose", "--help"])

    captured = capsys.readouterr()
    normalized = " ".join(captured.out.split())
    assert raised.value.code == 0
    assert captured.err == ""
    assert (
        "--suite {smoke,iid,ood-lower,ood-upper} --output FILE [--device {cpu,cuda}] [--bound-mask]"
    ) in normalized
    assert "ARTIFACT [ARTIFACT ...]" in normalized


@pytest.mark.parametrize(
    "argv",
    [
        ["train-bc", "--profile", "smoke", "--seed", "-1", "--output", "new"],
        ["train-ppo", "--profile", "smoke", "--seed", "1.5", "--output", "new"],
        ["train-bc", "--profile", "smoke", "--seed", "true", "--output", "new"],
        ["train-pitch", "--profile", "smoke", "--seed", "-1", "--output", "new"],
        ["train-pitch", "--profile", "smoke", "--seed", "nan", "--output", "new"],
        ["run", "artifact", "--seed", "-1"],
        ["run", "artifact", "--seed", "false"],
        ["run", "artifact", "--seed", "2.0"],
    ],
)
def test_invalid_seeds_exit_two_before_dependency_or_workflow(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid seed reached dependency check: {device}"),
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert "seed" in captured.err.lower()


def test_train_commands_validate_and_create_only_parents_before_lazy_trainer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, ProfileName, int, DeviceName, Path]] = []
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)

    def train_bc(*, profile: ProfileName, seed: int, device: DeviceName, output: Path):
        assert output.parent.is_dir()
        assert not output.exists()
        calls.append(("bc", profile, seed, device, output))

    def train_ppo(*, profile: ProfileName, seed: int, device: DeviceName, output: Path):
        assert output.parent.is_dir()
        assert not output.exists()
        calls.append(("ppo", profile, seed, device, output))

    def train_pitch(*, profile: ProfileName, seed: int, device: DeviceName, output: Path):
        assert output.parent.is_dir()
        assert not output.exists()
        calls.append(("pitch", profile, seed, device, output))

    monkeypatch.setattr(cli, "_train_bc_artifact", train_bc)
    monkeypatch.setattr(cli, "_train_ppo_artifact", train_ppo)
    monkeypatch.setattr(cli, "_train_pitch_artifact", train_pitch)
    bc_output = tmp_path / "missing" / "bc"
    ppo_output = tmp_path / "other" / "ppo"
    pitch_output = tmp_path / "pitch-parent" / "pitch"

    assert (
        cli.main(
            [
                "train-bc",
                "--profile",
                "checkpoint",
                "--seed",
                "3",
                "--output",
                str(bc_output),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "train-ppo",
                "--profile",
                "smoke",
                "--seed",
                "4",
                "--output",
                str(ppo_output),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "train-pitch",
                "--profile",
                "checkpoint",
                "--seed",
                "5",
                "--output",
                str(pitch_output),
                "--device",
                "cuda",
            ]
        )
        == 0
    )

    assert calls == [
        ("bc", ProfileName.CHECKPOINT, 3, DeviceName.CPU, bc_output),
        ("ppo", ProfileName.SMOKE, 4, DeviceName.CUDA, ppo_output),
        ("pitch", ProfileName.CHECKPOINT, 5, DeviceName.CUDA, pitch_output),
    ]
    assert capsys.readouterr() == ("", "")


def test_train_pitch_lazy_adapter_forwards_exact_public_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harpy.learning import pitch_artifacts

    output = tmp_path / "pitch"
    sentinel = object()
    calls: list[tuple[ProfileName, int, DeviceName, Path]] = []

    def train_pitch_artifact(
        *,
        profile: ProfileName,
        seed: int,
        device: DeviceName,
        output: Path,
    ) -> object:
        calls.append((profile, seed, device, output))
        return sentinel

    monkeypatch.setattr(pitch_artifacts, "train_pitch_artifact", train_pitch_artifact)

    assert (
        cli._train_pitch_artifact(
            profile=ProfileName.SMOKE,
            seed=7,
            device=DeviceName.CPU,
            output=output,
        )
        is sentinel
    )
    assert calls == [(ProfileName.SMOKE, 7, DeviceName.CPU, output)]


@pytest.mark.parametrize("command", ["train-bc", "train-ppo", "train-pitch"])
def test_existing_train_output_is_contract_exit_two_before_dependencies(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"existing output reached dependency check: {device}"),
    )

    assert (
        cli.main(
            [
                command,
                "--profile",
                "smoke",
                "--seed",
                "0",
                "--output",
                str(output),
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "already exists" in captured.err


@pytest.mark.parametrize("case", ["existing", "equal", "nested", "duplicate", "unwritable"])
def test_evaluate_path_contract_failures_exit_two_before_workflow(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    output = tmp_path / "report.json"
    arguments = ["evaluate", str(artifact)]
    if case == "existing":
        output.write_text("occupied", encoding="utf-8")
        arguments.extend(["--output", str(output)])
    elif case == "equal":
        arguments.extend(["--output", str(artifact)])
    elif case == "nested":
        arguments.extend(["--output", str(artifact / "reports" / "result.json")])
    elif case == "duplicate":
        alias = tmp_path / "alias"
        alias.symlink_to(artifact, target_is_directory=True)
        arguments.append(str(alias))
    else:
        unwritable = tmp_path / "unwritable"
        unwritable.mkdir(mode=0o500)
        arguments.extend(["--output", str(unwritable / "report.json")])
    monkeypatch.setattr(
        cli,
        "evaluate_artifacts",
        lambda *args, **kwargs: pytest.fail("invalid path reached evaluation workflow"),
    )
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid path reached dependency check: {device}"),
    )

    try:
        assert cli.main(arguments) == 2
    finally:
        if case == "unwritable":
            (tmp_path / "unwritable").chmod(0o700)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("artifact_value", ["", "missing", "regular-file"])
def test_invalid_artifact_input_is_contract_exit_two_before_workflow(
    artifact_value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if artifact_value == "regular-file":
        artifact = tmp_path / artifact_value
        artifact.write_text("not a directory", encoding="utf-8")
        supplied = str(artifact)
    elif artifact_value == "missing":
        supplied = str(tmp_path / artifact_value)
    else:
        supplied = artifact_value
    monkeypatch.setattr(
        cli,
        "evaluate_artifacts",
        lambda *args, **kwargs: pytest.fail("invalid input reached workflow"),
    )

    if artifact_value == "":
        with pytest.raises(SystemExit) as raised:
            cli.main(["evaluate", supplied])
        assert raised.value.code == 2
    else:
        assert cli.main(["evaluate", supplied]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err != ""


@pytest.mark.parametrize(
    "argv",
    [
        ["diagnose", "artifact", "--suite", "smoke"],
        ["diagnose", "artifact", "--output", "diagnostics.json"],
        [
            "diagnose",
            "artifact",
            "--suite",
            "register-ood",
            "--output",
            "diagnostics.json",
        ],
    ],
)
def test_diagnose_requires_closed_suite_and_output_grammar(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid grammar reached dependency check: {device}"),
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(argv)

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert captured.err != ""


def test_diagnose_valid_v1_bc_bound_mask_canonicalizes_and_publishes_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _V1_FIXTURE_ROOT / "bc-smoke"
    output = tmp_path / "missing" / "diagnostics.json"
    bundle = object()
    content = b'{"schema_id":"harpy-sine-diagnostic-bundle-v1"}\n'
    calls: list[tuple[tuple[Path, ...], str, DeviceName, bool]] = []
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)

    def diagnose(
        paths: tuple[Path, ...],
        *,
        suite: str,
        device: DeviceName,
        bound_mask: bool,
    ) -> object:
        calls.append((tuple(paths), suite, device, bound_mask))
        assert output.parent.is_dir()
        assert not output.exists()
        return bundle

    monkeypatch.setattr(cli, "_diagnose_artifacts", diagnose)
    monkeypatch.setattr(
        cli,
        "_diagnostic_bundle_bytes",
        lambda value: content if value is bundle else pytest.fail("wrong diagnostic bundle"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                str(artifact / "."),
                "--suite",
                "smoke",
                "--output",
                str(output),
                "--device",
                "cuda",
                "--bound-mask",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert calls == [
        (
            (artifact.resolve(),),
            "smoke",
            DeviceName.CUDA,
            True,
        )
    ]
    assert output.read_bytes() == content
    assert captured.out.encode() == content
    assert captured.err == f"{cli.TRUSTED_LOCAL_MODEL_WARNING}\n"


def test_diagnose_adapter_forwards_exact_workflow_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harpy.learning import workflows

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    sentinel = object()
    calls: list[tuple[tuple[Path, ...], str, DeviceName, bool]] = []

    def diagnose_artifacts(
        paths: tuple[Path, ...],
        *,
        suite: str,
        device: DeviceName,
        bound_mask: bool,
    ) -> object:
        calls.append((tuple(paths), suite, device, bound_mask))
        return sentinel

    monkeypatch.setattr(workflows, "diagnose_artifacts", diagnose_artifacts)

    assert (
        cli._diagnose_artifacts(
            (artifact,),
            suite="iid",
            device=DeviceName.CPU,
            bound_mask=True,
        )
        is sentinel
    )
    assert calls == [((artifact,), "iid", DeviceName.CPU, True)]


def test_diagnose_preflight_adapter_forwards_exact_workflow_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harpy.learning import workflows

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    calls: list[tuple[tuple[Path, ...], str, DeviceName, bool]] = []

    def preflight_diagnostic_request(
        paths: tuple[Path, ...],
        *,
        suite: str,
        device: DeviceName,
        bound_mask: bool,
    ) -> None:
        calls.append((tuple(paths), suite, device, bound_mask))

    monkeypatch.setattr(
        workflows,
        "preflight_diagnostic_request",
        preflight_diagnostic_request,
    )

    cli._preflight_diagnostic_request(
        (artifact,),
        suite="iid",
        device=DeviceName.CUDA,
        bound_mask=True,
    )

    assert calls == [((artifact,), "iid", DeviceName.CUDA, True)]


@pytest.mark.parametrize(
    "case",
    ["existing", "equal", "nested", "duplicate", "symlink", "unwritable"],
)
def test_diagnose_path_contract_failures_exit_two_before_dependencies_or_workflow(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "diagnostics.json"
    arguments = [
        "diagnose",
        str(artifact),
        "--suite",
        "smoke",
        "--output",
        str(output),
    ]
    if case == "existing":
        output.write_text("occupied", encoding="utf-8")
    elif case == "equal":
        arguments[-1] = str(artifact)
    elif case == "nested":
        arguments[-1] = str(artifact / "reports" / "diagnostics.json")
    elif case == "duplicate":
        alias = tmp_path / "alias"
        alias.symlink_to(artifact, target_is_directory=True)
        arguments.insert(2, str(alias))
    elif case == "symlink":
        occupied = tmp_path / "occupied.json"
        occupied.write_text("occupied", encoding="utf-8")
        output.symlink_to(occupied)
    else:
        unwritable = tmp_path / "unwritable"
        unwritable.mkdir(mode=0o500)
        arguments[-1] = str(unwritable / "diagnostics.json")
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid path reached dependency check: {device}"),
    )
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("invalid path reached diagnostics workflow"),
    )

    try:
        assert cli.main(arguments) == 2
    finally:
        if case == "unwritable":
            (tmp_path / "unwritable").chmod(0o700)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


def test_diagnostic_path_validator_value_error_is_contract_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from harpy.learning import diagnostic_codecs

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(
        diagnostic_codecs,
        "validate_diagnostic_output_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("diagnostic path rejected")),
    )
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid path reached dependency check: {device}"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                str(artifact),
                "--suite",
                "smoke",
                "--output",
                str(output),
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert "diagnostic path rejected" in captured.err


def test_diagnose_closed_contract_failure_does_not_publish_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = _V1_FIXTURE_ROOT / "ppo-smoke"
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid mask reached dependency check: {device}"),
    )
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("invalid mask reached diagnostics workflow"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                str(artifact),
                "--suite",
                "smoke",
                "--output",
                str(output),
                "--device",
                "cuda",
                "--bound-mask",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert "bound_mask" in captured.err


def test_diagnose_wrong_pitch_seed_set_precedes_cuda_and_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = []
    for seed in (0, 1, 3):
        artifact = tmp_path / f"pitch-{seed}"
        _write_pitch_preflight_manifest(artifact, seed=seed)
        artifacts.append(artifact)
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"wrong seeds reached dependency check: {device}"),
    )
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("wrong seeds reached diagnostics workflow"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                *(str(artifact) for artifact in artifacts),
                "--suite",
                "iid",
                "--output",
                str(output),
                "--device",
                "cuda",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert "exact seeds" in captured.err


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("ineligible", "valid CPU checkpoints or prospective E.1 CUDA checkpoints"),
        ("cuda-training", "homogeneous training device"),
        ("cuda-evaluation", "valid CPU checkpoints or prospective E.1 CUDA checkpoints"),
        ("dirty", "valid CPU checkpoints or prospective E.1 CUDA checkpoints"),
        ("uncommitted", "valid CPU checkpoints or prospective E.1 CUDA checkpoints"),
        ("commit", "share source"),
        ("lock", "share source"),
        ("compatibility", "share source"),
    ],
)
def test_diagnose_ineligible_or_incompatible_pitch_triple_precedes_cuda_and_output(
    mutation: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = []
    for seed in (0, 1, 2):
        overrides: dict[str, object] = {}
        if seed == 2:
            if mutation == "ineligible":
                overrides["eligible"] = False
            elif mutation == "cuda-training":
                overrides["training_device"] = "cuda"
            elif mutation == "cuda-evaluation":
                overrides["evaluation_device"] = "cuda"
            elif mutation == "dirty":
                overrides["dirty_tree"] = True
            elif mutation == "uncommitted":
                overrides["required_inputs_committed"] = False
            elif mutation == "commit":
                overrides["commit"] = "other-commit"
            elif mutation == "lock":
                overrides["dependency_lock_sha256"] = "c" * 64
            elif mutation == "compatibility":
                overrides["compatibility_sha256"] = "d" * 64
        artifact = tmp_path / f"pitch-{seed}"
        _write_pitch_preflight_manifest(artifact, seed=seed, **overrides)  # type: ignore[arg-type]
        artifacts.append(artifact)
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"invalid triple reached dependency check: {device}"),
    )
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("invalid triple reached diagnostics workflow"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                *(str(artifact) for artifact in artifacts),
                "--suite",
                "iid",
                "--output",
                str(output),
                "--device",
                "cuda",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert expected in captured.err


def test_diagnose_exact_e1_cuda_triple_reaches_cpu_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifacts = []
    for seed in (2, 0, 1):
        artifact = tmp_path / f"pitch-{seed}"
        _write_pitch_preflight_manifest(
            artifact,
            seed=seed,
            eligible=False,
            training_device="cuda",
        )
        artifacts.append(artifact)
    output = tmp_path / "missing" / "diagnostics.json"
    bundle = object()
    content = b'{"schema_id":"harpy-sine-diagnostic-bundle-v1"}\n'
    calls: list[tuple[tuple[Path, ...], str, DeviceName, bool]] = []
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda paths, *, suite, device, bound_mask: (
            calls.append((tuple(paths), suite, device, bound_mask)) or bundle
        ),
    )
    monkeypatch.setattr(
        cli,
        "_diagnostic_bundle_bytes",
        lambda value: content if value is bundle else pytest.fail("wrong diagnostic bundle"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                *(str(item) for item in artifacts),
                "--suite",
                "iid",
                "--output",
                str(output),
                "--device",
                "cpu",
            ]
        )
        == 0
    )

    assert calls == [((tuple(item.resolve() for item in artifacts)), "iid", DeviceName.CPU, False)]
    assert output.read_bytes() == content
    captured = capsys.readouterr()
    assert captured.out.encode() == content
    assert captured.err == f"{cli.TRUSTED_LOCAL_MODEL_WARNING}\n"


def test_diagnose_multi_v1_bound_mask_precedes_device_and_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = shutil.copytree(_V1_FIXTURE_ROOT / "bc-smoke", tmp_path / "bc-first")
    second = shutil.copytree(_V1_FIXTURE_ROOT / "bc-smoke", tmp_path / "bc-second")
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(
        cli,
        "_require_requested_device",
        lambda device: pytest.fail(f"multiple v1 artifacts reached device check: {device}"),
    )
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("multiple v1 artifacts reached workflow"),
    )

    assert (
        cli.main(
            [
                "diagnose",
                str(first),
                str(second),
                "--suite",
                "smoke",
                "--output",
                str(output),
                "--bound-mask",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert "exactly one" in captured.err


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [
        (DependencyUnavailableError("uv sync --group train"), 1),
        (ArtifactError("pitch artifact hash mismatch"), 1),
        (LearningExecutionError("diagnostics failed"), 1),
        (LearningContractError("unsupported diagnostic mask"), 2),
        (KeyboardInterrupt(), 130),
    ],
)
def test_diagnose_failures_have_stable_exit_and_never_publish(
    failure: BaseException,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "diagnostics.json"
    monkeypatch.setattr(cli, "_preflight_diagnostic_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)

    def fail(*args, **kwargs):
        del args, kwargs
        raise failure

    monkeypatch.setattr(cli, "_diagnose_artifacts", fail)

    assert (
        cli.main(
            [
                "diagnose",
                str(artifact),
                "--suite",
                "smoke",
                "--output",
                str(output),
            ]
        )
        == exit_code
    )

    captured = capsys.readouterr()
    assert not output.exists()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("device", "message"),
    [
        (DeviceName.CPU, "uv sync --group train"),
        (DeviceName.CUDA, "CUDA was requested but is unavailable"),
    ],
)
def test_diagnose_missing_dependency_or_unavailable_cuda_fails_before_parent_or_workflow(
    device: DeviceName,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "missing" / "diagnostics.json"
    monkeypatch.setattr(cli, "_preflight_diagnostic_request", lambda *args, **kwargs: None)

    def unavailable(requested: DeviceName) -> None:
        assert requested is device
        raise DependencyUnavailableError(message)

    monkeypatch.setattr(cli, "_require_requested_device", unavailable)
    monkeypatch.setattr(
        cli,
        "_diagnose_artifacts",
        lambda *args, **kwargs: pytest.fail("dependency failure reached diagnostics workflow"),
    )

    arguments = [
        "diagnose",
        str(artifact),
        "--suite",
        "smoke",
        "--output",
        str(output),
        "--device",
        device.value,
    ]
    assert cli.main(arguments) == 1

    captured = capsys.readouterr()
    assert not output.parent.exists()
    assert captured.out == ""
    assert message in captured.err


def test_evaluate_publishes_create_only_identical_stdout_and_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "missing" / "report.json"
    report = object()
    content = b'{"criterion":"criterion_not_met"}\n'
    calls: list[tuple[tuple[Path, ...], DeviceName]] = []
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)
    monkeypatch.setattr(
        cli,
        "evaluate_artifacts",
        lambda paths, *, device: calls.append((tuple(paths), device)) or report,
    )
    monkeypatch.setattr(
        cli,
        "evaluation_report_bytes",
        lambda value: content if value is report else pytest.fail("wrong report"),
    )

    assert (
        cli.main(
            [
                "evaluate",
                str(artifact),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out.encode() == content
    assert output.read_bytes() == content
    assert calls == [((artifact.resolve(),), DeviceName.CPU)]
    assert captured.err.count(cli.TRUSTED_LOCAL_MODEL_WARNING) == 1


@pytest.mark.parametrize("json_mode", [False, True])
def test_run_emits_exact_human_or_json_trace_and_one_trusted_warning(
    json_mode: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    episode = object()
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)
    monkeypatch.setattr(
        cli,
        "run_artifact",
        lambda path, *, seed, device: (
            episode
            if (path, seed, device) == (artifact.resolve(), 8, DeviceName.CPU)
            else pytest.fail("run arguments changed")
        ),
    )
    monkeypatch.setattr(cli, "trace_json_bytes", lambda value: b'{"trace":true}\n')
    monkeypatch.setattr(cli, "format_human_trace", lambda value: "Target: C3\nTerminal: done\n")
    arguments = ["run", str(artifact), "--seed", "8"]
    if json_mode:
        arguments.append("--json")

    assert cli.main(arguments) == 0

    captured = capsys.readouterr()
    assert captured.out == ('{"trace":true}\n' if json_mode else "Target: C3\nTerminal: done\n")
    assert captured.err.count(cli.TRUSTED_LOCAL_MODEL_WARNING) == 1


@pytest.mark.parametrize(
    ("failure", "exit_code"),
    [
        (LearningContractError("duplicate artifact seed"), 2),
        (ArtifactError("hash mismatch"), 1),
        (DependencyUnavailableError("uv sync --group train"), 1),
        (LearningExecutionError("evaluation failed"), 1),
        (OSError("publication failed"), 1),
        (RuntimeError("unexpected evaluator failure"), 1),
        (KeyboardInterrupt(), 130),
    ],
)
def test_operational_failures_have_stable_exit_empty_stdout_and_one_diagnostic(
    failure: BaseException,
    exit_code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(cli, "_require_requested_device", lambda device: None)

    def fail(*args, **kwargs):
        del args, kwargs
        raise failure

    monkeypatch.setattr(cli, "evaluate_artifacts", fail)

    assert cli.main(["evaluate", str(artifact)]) == exit_code

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err


def test_explicit_unavailable_cuda_is_exit_one_after_all_paths_validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    calls: list[DeviceName] = []

    def unavailable(device: DeviceName) -> None:
        calls.append(device)
        raise DependencyUnavailableError("CUDA was requested but is unavailable")

    monkeypatch.setattr(cli, "_require_requested_device", unavailable)
    monkeypatch.setattr(
        cli,
        "run_artifact",
        lambda *args, **kwargs: pytest.fail("unavailable CUDA reached actor workflow"),
    )

    assert cli.main(["run", str(artifact), "--seed", "0", "--device", "cuda"]) == 1

    captured = capsys.readouterr()
    assert calls == [DeviceName.CUDA]
    assert captured.out == ""
    assert "CUDA" in captured.err


def test_console_script_metadata_preserves_existing_scripts() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["harpy"] == "harpy.gui.app:main"
    assert scripts["harpy-sine-gym"] == "harpy.envs.checkpoint:main"
    assert scripts["harpy-sine-learn"] == "harpy.learning.cli:main"


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_help_is_torch_sb3_and_qt_free(
    entry_point: str,
    tmp_path: Path,
) -> None:
    environment = _heavy_import_poisoned_environment(tmp_path)
    if entry_point == "module":
        command = [sys.executable, "-m", "harpy.learning.cli", "--help"]
    else:
        executable = shutil.which("harpy-sine-learn")
        assert executable is not None
        command = [executable, "--help"]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "{train-bc,train-ppo,train-pitch,diagnose,evaluate,run}" in completed.stdout


def _heavy_import_poisoned_environment(tmp_path: Path) -> dict[str, str]:
    poison = tmp_path / "import-poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        """
import builtins

_original_import = builtins.__import__

def _guarded_import(name, *args, **kwargs):
    blocked = ("torch", "stable_baselines3", "PySide6", "pyqtgraph")
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise RuntimeError("help imported optional heavy dependency: " + name)
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import
""".lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(poison), *(value for value in [python_path] if value)]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment
