"""Four-command learned sine-policy CLI and stable exit contracts."""

from __future__ import annotations

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


def test_parser_exposes_exact_four_command_grammar(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    captured = capsys.readouterr()
    assert raised.value.code == 0
    assert captured.err == ""
    assert "{train-bc,train-ppo,evaluate,run}" in captured.out
    for command in ("train-bc", "train-ppo", "evaluate", "run"):
        assert command in captured.out


@pytest.mark.parametrize(
    "argv",
    [
        ["train-bc", "--profile", "smoke", "--seed", "-1", "--output", "new"],
        ["train-ppo", "--profile", "smoke", "--seed", "1.5", "--output", "new"],
        ["train-bc", "--profile", "smoke", "--seed", "true", "--output", "new"],
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

    monkeypatch.setattr(cli, "_train_bc_artifact", train_bc)
    monkeypatch.setattr(cli, "_train_ppo_artifact", train_ppo)
    bc_output = tmp_path / "missing" / "bc"
    ppo_output = tmp_path / "other" / "ppo"

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

    assert calls == [
        ("bc", ProfileName.CHECKPOINT, 3, DeviceName.CPU, bc_output),
        ("ppo", ProfileName.SMOKE, 4, DeviceName.CUDA, ppo_output),
    ]
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("command", ["train-bc", "train-ppo"])
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
    assert "{train-bc,train-ppo,evaluate,run}" in completed.stdout


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
