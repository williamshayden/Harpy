"""Six-command CLI for training, diagnosing, evaluating, and running sine policies."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from harpy.learning.artifacts import write_new_bytes
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import DependencyUnavailableError, LearningContractError
from harpy.learning.models import DeviceName, ProfileName
from harpy.learning.trace import format_human_trace, trace_json_bytes
from harpy.learning.workflows import (
    evaluate_artifacts,
    evaluation_report_bytes,
    run_artifact,
)

if TYPE_CHECKING:
    from harpy.learning.diagnostics import DiagnosticBundle

TRUSTED_LOCAL_MODEL_WARNING = (
    "warning: learned model archives are trusted-local artifacts; "
    "do not load files from untrusted sources"
)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one narrow command and return its stable process exit status."""

    arguments = _argument_parser().parse_args(argv)
    try:
        if arguments.command == "train-bc":
            return _run_train(arguments, trainer="bc")
        if arguments.command == "train-ppo":
            return _run_train(arguments, trainer="ppo")
        if arguments.command == "train-pitch":
            return _run_train(arguments, trainer="pitch")
        if arguments.command == "diagnose":
            return _run_diagnose(arguments)
        if arguments.command == "evaluate":
            return _run_evaluate(arguments)
        if arguments.command == "run":
            return _run_episode(arguments)
        raise LearningContractError(f"unsupported command: {arguments.command}")
    except KeyboardInterrupt:
        _write_diagnostic("interrupted")
        return 130
    except LearningContractError as error:
        _write_diagnostic(error)
        return 2
    except Exception as error:
        _write_diagnostic(error)
        return 1


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harpy-sine-learn")
    commands = parser.add_subparsers(dest="command", required=True)

    train_bc = commands.add_parser("train-bc")
    _add_training_arguments(train_bc)

    train_ppo = commands.add_parser("train-ppo")
    _add_training_arguments(train_ppo)

    train_pitch = commands.add_parser("train-pitch")
    _add_training_arguments(train_pitch)

    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("artifacts", nargs="+", type=_nonempty_path, metavar="ARTIFACT")
    diagnose.add_argument(
        "--suite",
        choices=("smoke", "iid", "ood-lower", "ood-upper"),
        required=True,
    )
    diagnose.add_argument("--output", type=_nonempty_path, required=True, metavar="FILE")
    _add_device_argument(diagnose)
    diagnose.add_argument("--bound-mask", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("artifacts", nargs="+", type=_nonempty_path, metavar="ARTIFACT")
    evaluate.add_argument("--output", type=_nonempty_path, metavar="FILE")
    _add_device_argument(evaluate)

    run = commands.add_parser("run")
    run.add_argument("artifact", type=_nonempty_path, metavar="ARTIFACT")
    run.add_argument("--seed", type=_nonnegative_integer, required=True, metavar="N")
    run.add_argument("--json", action="store_true")
    _add_device_argument(run)
    return parser


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=ProfileName,
        choices=tuple(ProfileName),
        required=True,
    )
    parser.add_argument("--seed", type=_nonnegative_integer, required=True, metavar="N")
    parser.add_argument("--output", type=_nonempty_path, required=True, metavar="PATH")
    _add_device_argument(parser)


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=DeviceName,
        choices=tuple(DeviceName),
        default=DeviceName.CPU,
    )


def _nonempty_path(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("path must not be empty")
    return Path(value)


def _nonnegative_integer(value: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return parsed


def _run_train(arguments: argparse.Namespace, *, trainer: str) -> int:
    output = _preflight_new_output(arguments.output, role="artifact output")
    _create_parent(output)
    _require_requested_device(arguments.device)
    if trainer == "bc":
        _train_bc_artifact(
            profile=arguments.profile,
            seed=arguments.seed,
            device=arguments.device,
            output=output,
        )
    elif trainer == "ppo":
        _train_ppo_artifact(
            profile=arguments.profile,
            seed=arguments.seed,
            device=arguments.device,
            output=output,
        )
    elif trainer == "pitch":
        _train_pitch_artifact(
            profile=arguments.profile,
            seed=arguments.seed,
            device=arguments.device,
            output=output,
        )
    else:
        raise LearningContractError(f"unsupported trainer: {trainer}")
    return 0


def _run_diagnose(arguments: argparse.Namespace) -> int:
    artifacts = _canonical_input_directories(arguments.artifacts)
    output = _preflight_new_output(arguments.output, role="diagnostic output")
    output = _validate_diagnostic_output(output, artifacts=artifacts)
    _preflight_diagnostic_request(
        artifacts,
        suite=arguments.suite,
        device=arguments.device,
        bound_mask=arguments.bound_mask,
    )
    _require_requested_device(arguments.device)
    _create_parent(output)
    bundle = _diagnose_artifacts(
        artifacts,
        suite=arguments.suite,
        device=arguments.device,
        bound_mask=arguments.bound_mask,
    )
    content = _diagnostic_bundle_bytes(bundle)
    write_new_bytes(output, content)
    _write_trusted_local_warning()
    _write_stdout_bytes(content)
    return 0


def _run_evaluate(arguments: argparse.Namespace) -> int:
    artifacts = _canonical_input_directories(arguments.artifacts)
    output = None
    if arguments.output is not None:
        output = _preflight_new_output(arguments.output, role="evaluation output")
        resolved_output = output.resolve(strict=False)
        if any(
            resolved_output == artifact or resolved_output.is_relative_to(artifact)
            for artifact in artifacts
        ):
            raise LearningContractError(
                "evaluation output must be outside every input artifact directory"
            )
        _create_parent(output)
    _require_requested_device(arguments.device)
    report = evaluate_artifacts(artifacts, device=arguments.device)
    content = evaluation_report_bytes(report)
    if output is not None:
        write_new_bytes(output, content)
    _write_trusted_local_warning()
    _write_stdout_bytes(content)
    return 0


def _run_episode(arguments: argparse.Namespace) -> int:
    artifact = _canonical_input_directories((arguments.artifact,))[0]
    _require_requested_device(arguments.device)
    episode = run_artifact(
        artifact,
        seed=arguments.seed,
        device=arguments.device,
    )
    if arguments.json:
        content = trace_json_bytes(episode)
    else:
        content = format_human_trace(episode).encode("utf-8")
    _write_trusted_local_warning()
    _write_stdout_bytes(content)
    return 0


def _canonical_input_directories(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for path in paths:
        try:
            canonical = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise LearningContractError(f"artifact path does not exist: {path}") from error
        if not canonical.is_dir():
            raise LearningContractError(f"artifact path must be a directory: {path}")
        resolved.append(canonical)
    if len(set(resolved)) != len(resolved):
        raise LearningContractError("duplicate artifact path after alias resolution")
    return tuple(resolved)


def _preflight_new_output(path: Path, *, role: str) -> Path:
    if path.exists() or path.is_symlink():
        raise LearningContractError(f"{role} already exists: {path}")
    ancestor = path.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        parent = ancestor.parent
        if parent == ancestor:
            break
        ancestor = parent
    try:
        resolved_ancestor = ancestor.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise LearningContractError(
            f"nearest existing output ancestor is unavailable: {ancestor}"
        ) from error
    if not resolved_ancestor.is_dir():
        raise LearningContractError(
            f"nearest existing output ancestor must be a directory: {resolved_ancestor}"
        )
    mode = stat.S_IMODE(resolved_ancestor.stat().st_mode)
    if mode & 0o222 == 0 or not os.access(resolved_ancestor, os.W_OK | os.X_OK):
        raise LearningContractError(
            f"nearest existing output ancestor is not writable: {resolved_ancestor}"
        )
    return path


def _create_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _require_requested_device(device: DeviceName) -> None:
    stack = require_training_dependencies()
    if device is DeviceName.CUDA and not stack.torch.cuda.is_available():
        raise DependencyUnavailableError("CUDA was requested but is unavailable")


def _train_bc_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> object:
    from harpy.learning.bc import train_bc_artifact

    return train_bc_artifact(
        profile=profile,
        seed=seed,
        device=device,
        output=output,
    )


def _train_ppo_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> object:
    from harpy.learning.ppo import train_ppo_artifact

    return train_ppo_artifact(
        profile=profile,
        seed=seed,
        device=device,
        output=output,
    )


def _train_pitch_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> object:
    from harpy.learning.pitch_artifacts import train_pitch_artifact

    return train_pitch_artifact(
        profile=profile,
        seed=seed,
        device=device,
        output=output,
    )


def _validate_diagnostic_output(output: Path, *, artifacts: Sequence[Path]) -> Path:
    from harpy.learning.diagnostic_codecs import validate_diagnostic_output_path

    try:
        return validate_diagnostic_output_path(
            output,
            artifact_directories=artifacts,
        )
    except ValueError as error:
        raise LearningContractError(str(error)) from error


def _diagnose_artifacts(
    artifacts: Sequence[Path],
    *,
    suite: str,
    device: DeviceName,
    bound_mask: bool,
) -> DiagnosticBundle:
    from harpy.learning.workflows import diagnose_artifacts

    return diagnose_artifacts(
        artifacts,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
    )


def _preflight_diagnostic_request(
    artifacts: Sequence[Path],
    *,
    suite: str,
    device: DeviceName,
    bound_mask: bool,
) -> None:
    from harpy.learning.workflows import preflight_diagnostic_request

    preflight_diagnostic_request(
        artifacts,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
    )


def _diagnostic_bundle_bytes(bundle: DiagnosticBundle) -> bytes:
    from harpy.learning.diagnostic_codecs import diagnostic_bundle_bytes

    return diagnostic_bundle_bytes(bundle)


def _write_trusted_local_warning() -> None:
    sys.stderr.write(f"{TRUSTED_LOCAL_MODEL_WARNING}\n")


def _write_stdout_bytes(content: bytes) -> None:
    stream = getattr(sys.stdout, "buffer", None)
    if stream is None:
        sys.stdout.write(content.decode("utf-8"))
    else:
        stream.write(content)


def _write_diagnostic(error: object) -> None:
    message = " ".join(str(error).strip().splitlines()) or type(error).__name__
    sys.stderr.write(f"error: {message}\n")


if __name__ == "__main__":
    raise SystemExit(main())
