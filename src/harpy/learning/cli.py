"""Headless CLI for training, evaluating, diagnosing, and reading research results."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from harpy.learning.artifacts import write_new_bytes
from harpy.learning.dependencies import (
    configure_deterministic_cuda_environment,
    require_training_dependencies,
)
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
        if arguments.command == "summarize":
            return _run_summarize(arguments)
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
    parser = argparse.ArgumentParser(
        prog="harpy-sine-learn",
        description="Train sine-control models and inspect reproducible experiment results.",
        epilog="Use COMMAND --help for examples. "
        "JSON results go to stdout; progress goes to stderr.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_bc = commands.add_parser(
        "train-bc",
        help="train the supervised oracle-action diagnostic",
        description="Train behavior cloning from oracle action labels.",
    )
    _add_training_arguments(train_bc)

    train_ppo = commands.add_parser(
        "train-ppo",
        help="train a freshly initialized PPO policy",
        description="Train PPO from scratch on the frozen sine-pitch environment.",
    )
    _add_training_arguments(train_ppo)

    train_pitch = commands.add_parser(
        "train-pitch",
        help="train a spectrum pitch estimator with symbolic control",
        description="Train the spectrum-only pitch estimator used by the symbolic planner.",
    )
    _add_training_arguments(train_pitch)

    diagnose = commands.add_parser(
        "diagnose",
        help="explain action errors, loops, and failed episodes",
        description="Write strict per-decision diagnostic JSON from complete artifacts.",
        epilog="Example: harpy-sine-learn diagnose runs/pitch-smoke --suite smoke "
        "--output runs/pitch-diagnostics.json",
    )
    diagnose.add_argument(
        "artifacts",
        nargs="+",
        type=_nonempty_path,
        metavar="ARTIFACT",
        help="one BC/PPO artifact; one pitch smoke artifact "
        "or pitch checkpoints with seeds 0, 1, 2",
    )
    diagnose.add_argument(
        "--suite",
        choices=("smoke", "iid", "ood-lower", "ood-upper"),
        required=True,
        help="smoke checks plumbing; iid and ood suites diagnose checkpoint behavior "
        "(ood-lower/ood-upper require pitch artifacts)",
    )
    diagnose.add_argument(
        "--output",
        type=_nonempty_path,
        required=True,
        metavar="FILE",
        help="new JSON file outside every input artifact directory",
    )
    _add_device_argument(diagnose)
    diagnose.add_argument(
        "--bound-mask",
        action="store_true",
        help="mask actions blocked by control bounds; "
        "diagnostic intervention for one BC artifact only",
    )
    diagnose.add_argument(
        "--exploratory",
        action="store_true",
        help="inspect one pitch artifact independently; always scientifically ineligible",
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="compare policies with matched baselines and scientific criteria",
        description="Write canonical evaluation JSON with separately labeled observation tracks.",
        epilog="Example: harpy-sine-learn evaluate runs/pitch-smoke --output runs/report.json",
    )
    evaluate.add_argument(
        "artifacts",
        nargs="+",
        type=_nonempty_path,
        metavar="ARTIFACT",
        help="compatible complete artifacts; final pitch evaluation requires exact seeds 0, 1, 2",
    )
    evaluate.add_argument(
        "--output",
        type=_nonempty_path,
        metavar="FILE",
        help="also save stdout JSON to a new file outside every input artifact directory",
    )
    _add_device_argument(evaluate)
    evaluate.add_argument(
        "--exploratory",
        action="store_true",
        help="evaluate one pitch artifact independently; always scientifically ineligible",
    )

    run = commands.add_parser(
        "run",
        help="trace one seeded demonstration episode",
        description="Run a model on one full-range demonstration episode, "
        "separate from test suites.",
        epilog="Example: harpy-sine-learn run runs/pitch-smoke --seed 123 --json",
    )
    run.add_argument(
        "artifact",
        type=_nonempty_path,
        metavar="ARTIFACT",
        help="complete model artifact directory",
    )
    run.add_argument(
        "--seed",
        type=_nonnegative_integer,
        required=True,
        metavar="N",
        help="non-negative demonstration episode seed (independent of the model training seed)",
    )
    run.add_argument(
        "--json", action="store_true", help="emit canonical JSON instead of a text trace"
    )
    run.add_argument(
        "--with-provenance",
        action="store_true",
        help="emit a versioned JSON run result identifying its model and runtime (implies --json)",
    )
    _add_device_argument(run)

    summarize = commands.add_parser(
        "summarize",
        help="read an existing evaluation or diagnostic report without loading models",
        description="Strictly validate saved report JSON and print a concise research readout.",
        epilog="Example: harpy-sine-learn summarize runs/report.json --format markdown > report.md",
    )
    summarize.add_argument(
        "report", type=_nonempty_path, metavar="REPORT", help="saved report JSON"
    )
    summarize.add_argument(
        "--format",
        choices=("text", "markdown"),
        default="text",
        help="readout format (default: text); the source report is never modified",
    )
    return parser


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=ProfileName,
        choices=tuple(ProfileName),
        required=True,
        help="smoke verifies the workflow and is scientifically ineligible; checkpoint uses "
        "the frozen experiment settings",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        required=True,
        metavar="N",
        help="non-negative training seed; authoritative pitch cohorts use exact seeds 0, 1, 2",
    )
    parser.add_argument(
        "--output",
        type=_nonempty_path,
        required=True,
        metavar="PATH",
        help="new artifact directory; existing paths are never overwritten",
    )
    parser.epilog = (
        f"Example: {parser.prog} --profile smoke --seed 0 --output runs/model-smoke --device cpu"
    )
    _add_device_argument(parser)


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=DeviceName,
        choices=tuple(DeviceName),
        default=DeviceName.CPU,
        help="execution device (default: cpu); cuda must be available and is not authoritative "
        "except for a qualifying E.1 training cohort",
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
    _write_progress(
        f"Training {trainer}: profile={arguments.profile}, seed={arguments.seed}, "
        f"device={arguments.device}; artifact={output.resolve(strict=False)}. "
        "Preparing data, fitting the model, and validating the saved result; this may take minutes."
    )
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
    _write_progress(f"Training complete; artifact={output.resolve(strict=False)}")
    return 0


def _run_diagnose(arguments: argparse.Namespace) -> int:
    artifacts = _canonical_input_directories(arguments.artifacts)
    output = _preflight_new_output(arguments.output, role="diagnostic output")
    output = _validate_diagnostic_output(output, artifacts=artifacts)
    exploratory = {"exploratory": True} if arguments.exploratory else {}
    _preflight_diagnostic_request(
        artifacts,
        suite=arguments.suite,
        device=arguments.device,
        bound_mask=arguments.bound_mask,
        **exploratory,
    )
    _require_requested_device(arguments.device)
    _create_parent(output)
    _write_trusted_local_warning()
    _write_progress(f"Diagnosing {len(artifacts)} artifact(s); suite={arguments.suite}")
    bundle = _diagnose_artifacts(
        artifacts,
        suite=arguments.suite,
        device=arguments.device,
        bound_mask=arguments.bound_mask,
        **exploratory,
    )
    if arguments.exploratory:
        from harpy.learning.experiment_results import readout_bytes

        content = readout_bytes(bundle)
    else:
        content = _diagnostic_bundle_bytes(bundle)
    write_new_bytes(output, content)
    _write_stdout_bytes(content)
    _write_progress(f"Diagnostics complete; report={output.resolve(strict=False)}")
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
    _write_trusted_local_warning()
    _write_progress(f"Evaluating {len(artifacts)} artifact(s); device={arguments.device}")
    if arguments.exploratory:
        from harpy.learning.experiment_results import readout_bytes

        report = evaluate_artifacts(artifacts, device=arguments.device, exploratory=True)
        content = readout_bytes(report)
    else:
        report = evaluate_artifacts(artifacts, device=arguments.device)
        content = evaluation_report_bytes(report)
    if output is not None:
        write_new_bytes(output, content)
    _write_stdout_bytes(content)
    destination = "stdout" if output is None else str(output.resolve(strict=False))
    _write_progress(f"Evaluation complete; report={destination}")
    return 0


def _run_episode(arguments: argparse.Namespace) -> int:
    artifact = _canonical_input_directories((arguments.artifact,))[0]
    _require_requested_device(arguments.device)
    _write_trusted_local_warning()
    _write_progress(f"Running demonstration; seed={arguments.seed}, device={arguments.device}")
    if arguments.with_provenance:
        from harpy.learning.experiment_results import readout_bytes
        from harpy.learning.workflows import run_artifact_result

        result = run_artifact_result(artifact, seed=arguments.seed, device=arguments.device)
        _write_stdout_bytes(readout_bytes(result))
        _write_progress("Demonstration complete; trace=stdout")
        return 0
    episode = run_artifact(
        artifact,
        seed=arguments.seed,
        device=arguments.device,
    )
    if arguments.json:
        content = trace_json_bytes(episode)
    else:
        content = format_human_trace(episode).encode("utf-8")
    _write_stdout_bytes(content)
    _write_progress("Demonstration complete; trace=stdout")
    return 0


def _run_summarize(arguments: argparse.Namespace) -> int:
    from harpy.learning.readout import summarize_report_bytes

    content = arguments.report.read_bytes()
    summary = summarize_report_bytes(content, format=arguments.format)
    _write_stdout_bytes(summary.encode("utf-8"))
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
    if device is DeviceName.CUDA:
        configure_deterministic_cuda_environment()
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
    exploratory: bool = False,
) -> DiagnosticBundle:
    from harpy.learning.workflows import diagnose_artifacts

    options = {"exploratory": True} if exploratory else {}
    return diagnose_artifacts(
        artifacts,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
        **options,
    )


def _preflight_diagnostic_request(
    artifacts: Sequence[Path],
    *,
    suite: str,
    device: DeviceName,
    bound_mask: bool,
    exploratory: bool = False,
) -> None:
    from harpy.learning.workflows import preflight_diagnostic_request

    options = {"exploratory": True} if exploratory else {}
    preflight_diagnostic_request(
        artifacts,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
        **options,
    )


def _diagnostic_bundle_bytes(bundle: DiagnosticBundle) -> bytes:
    from harpy.learning.diagnostic_codecs import diagnostic_bundle_bytes

    return diagnostic_bundle_bytes(bundle)


def _write_trusted_local_warning() -> None:
    sys.stderr.write(f"{TRUSTED_LOCAL_MODEL_WARNING}\n")


def _write_progress(message: str) -> None:
    sys.stderr.write(f"{' '.join(message.splitlines())}\n")
    sys.stderr.flush()


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
