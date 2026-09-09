"""The four supported headless research operations; optional ML imports are lazy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harpy", description="Reproducible single-sine audio-control experiments."
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    train = operations.add_parser(
        "train", help="Train a pitch reference or experimental PPO policy"
    )
    train.add_argument("trainer", choices=("pitch", "ppo"))
    train.add_argument("--profile", choices=("smoke", "checkpoint"), default="smoke")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--output", type=Path, required=True, help="New artifact directory")

    evaluate = operations.add_parser(
        "evaluate", help="Compare actors on matched episode membership"
    )
    evaluate.add_argument(
        "--actor",
        action="append",
        dest="actors",
        metavar="NAME_OR_PATH",
        help="Repeat for comparisons: spectrum-peak, spectrum-peak-replan, waveform-fft, oracle, "
        "reward-search, random, reference, a saved artifact directory, or LABEL=PATH",
    )
    evaluate.add_argument(
        "--suite",
        choices=("clean", "robustness", "confirmation", "smoke"),
        default="clean",
        help="650 clean, 8 x 650 robustness, 1000 confirmation, or a small workflow smoke",
    )
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--trace", action="store_true", help="Include every public decision")
    evaluate.add_argument("--output", type=Path, required=True, help="New result JSON file")

    run = operations.add_parser("run", help="Inspect one episode and its action trace")
    run.add_argument("--actor", default="spectrum-peak", metavar="NAME_OR_PATH")
    run.add_argument("--source-cents", type=int, default=6064)
    run.add_argument(
        "--target-note-index", type=int, default=12, help="0..24 represents MIDI 48..72"
    )
    run.add_argument("--nuisance-seed", type=int, default=0)
    run.add_argument("--condition", default="clean", help="A frozen robustness condition name")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument(
        "--output", type=Path, required=True, help="New result JSON with a decision trace"
    )

    summarize = operations.add_parser(
        "summarize", help="Validate and read a saved result without ML"
    )
    summarize.add_argument("result", type=Path)
    summarize.add_argument("--format", choices=("text", "markdown", "csv"), default="text")
    return parser


def _actor(name: str, *, device: str):
    from harpy.experiments import actors

    if "=" in name:
        label, path = name.split("=", 1)
        if not label or not path:
            raise ValueError("An actor alias must use LABEL=PATH with both parts nonempty")
        return replace(_actor(path, device=device), name=label)

    builtins = {
        "spectrum-peak": lambda: actors.spectrum_peak_actor_spec(),
        "spectrum-peak-replan": lambda: actors.spectrum_peak_actor_spec(replan=True),
        "waveform-fft": actors.waveform_fft_actor_spec,
        "oracle": actors.oracle_actor_spec,
        "reward-search": actors.reward_search_actor_spec,
        "random": actors.random_actor_spec,
    }
    if name in builtins:
        return builtins[name]()
    path = Path(str(files("harpy").joinpath("reference"))) if name == "reference" else Path(name)
    if not path.is_dir():
        raise ValueError(f"Unknown actor or missing artifact directory: {name}")
    from harpy.experiments.artifacts import load_artifact

    artifact = load_artifact(path)
    trainer = artifact.manifest.trainer
    if trainer == "pitch":
        return actors.pitch_actor_spec(path, device=device)
    if trainer == "ppo":
        print(
            "Loading a trusted-local PPO archive; hashes establish identity, not trust.",
            file=sys.stderr,
        )
        return actors.ppo_actor_spec(path, device=device)
    raise ValueError(f"Unsupported artifact trainer: {trainer}")


def _summarize_path(path: Path, *, format: str) -> str:
    from harpy.experiments.results import RESULT_SCHEMA_ID, load_result, summarize

    content = path.read_bytes()
    document = json.loads(content)
    if not isinstance(document, dict):
        raise ValueError("A saved result must be a JSON object")
    if document.get("schema_id") == RESULT_SCHEMA_ID:
        return summarize(load_result(path), format=format)
    from harpy.learning.readout import summarize_report_bytes

    if format == "csv":
        raise ValueError(
            "Historical reports support text/markdown; CSV is available for new results"
        )
    return summarize_report_bytes(content, format=format)


def _execute(args: argparse.Namespace) -> int:
    if args.operation == "summarize":
        print(_summarize_path(args.result, format=args.format), end="")
        return 0
    if args.output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {args.output}")
    if args.operation == "train":
        from harpy.experiments.training import train

        train(
            trainer=args.trainer,
            profile=args.profile,
            seed=args.seed,
            device=args.device,
            output=args.output,
        )
        print(f"Saved {args.trainer} artifact: {args.output}", file=sys.stderr)
        return 0

    from harpy.envs.robustness import ROBUSTNESS_CONDITIONS
    from harpy.experiments.models import EpisodeSpec
    from harpy.experiments.protocols import (
        benchmark_episodes,
        confirmation_episodes,
        protocol_document,
        smoke_episodes,
    )
    from harpy.experiments.results import save_result, summarize
    from harpy.experiments.runner import evaluate

    if args.operation == "run":
        condition = next((c for c in ROBUSTNESS_CONDITIONS if c.id == args.condition), None)
        if condition is None:
            raise ValueError(
                "Unknown condition. Choose: " + ", ".join(c.id for c in ROBUSTNESS_CONDITIONS)
            )
        episodes = (
            EpisodeSpec("single", args.source_cents, args.target_note_index, args.nuisance_seed),
        )
        actor_specs = (_actor(args.actor, device=args.device),)
        conditions = (condition,)
        trace = True
        protocol = None
    else:
        episodes = confirmation_episodes() if args.suite == "confirmation" else benchmark_episodes()
        if args.suite == "smoke":
            episodes = smoke_episodes()
        conditions = (
            ROBUSTNESS_CONDITIONS if args.suite == "robustness" else (ROBUSTNESS_CONDITIONS[0],)
        )
        actor_specs = tuple(
            _actor(name, device=args.device) for name in (args.actors or ["spectrum-peak"])
        )
        trace = args.trace
        protocol = protocol_document(args.suite)
    result = evaluate(episodes, actor_specs, conditions=conditions, trace=trace, protocol=protocol)
    save_result(result, args.output)
    print(summarize(result), end="")
    print(f"Saved validated result: {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Execute a command, returning a shell status without importing ML for help/readout."""
    args = _parser().parse_args(argv)
    try:
        return _execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"harpy: {error}", file=sys.stderr)
        return 2
