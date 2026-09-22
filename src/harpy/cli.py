"""The four supported headless research operations; optional ML imports are lazy."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harpy", description="Reproducible headless audio-control experiments."
    )
    operations = parser.add_subparsers(dest="operation", required=True)
    train = operations.add_parser(
        "train", help="Train an inverse sound model, pitch reference, or experimental PPO policy"
    )
    train.add_argument("trainer", choices=("inverse", "pitch", "ppo"))
    train.add_argument("--profile", choices=("quick", "reference", "smoke", "checkpoint"))
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--output", type=Path, required=True, help="New artifact directory")

    evaluate = operations.add_parser(
        "evaluate", help="Compare actors on matched episode membership"
    )
    evaluate.add_argument(
        "--task", choices=("sound-match-v1",), help="Omit for the released v1 pitch task"
    )
    evaluate.add_argument(
        "--actor",
        action="append",
        dest="actors",
        metavar="NAME_OR_PATH",
        help="Repeat for comparisons: spectrum-peak, spectrum-peak-replan, waveform-fft, oracle, "
        "reward-search, random, reference, a saved artifact directory, or LABEL=PATH. "
        "Sound matching: fixed, random-search, pattern-search, inverse=PATH, inverse-refine=PATH, "
        "or LABEL=inverse:PATH",
    )
    evaluate.add_argument(
        "--suite",
        help="Pitch: clean (default), robustness, confirmation, smoke; "
        "sound matching: tone-smoke (default), chord-smoke, shaping-smoke, calibration, "
        "validation, iid-test, compositional-test",
    )
    evaluate.add_argument("--seed", type=int, help="Sound-matching search seed (default: 0)")
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.add_argument(
        "--trace",
        action="store_true",
        default=None,
        help="Include every public decision (always included for sound matching)",
    )
    evaluate.add_argument(
        "--output", type=Path, required=True, help="New pitch JSON file or sound-matching bundle"
    )

    run = operations.add_parser("run", help="Inspect one episode and its action trace")
    run.add_argument(
        "--task", choices=("sound-match-v1",), help="Omit for the released v1 pitch task"
    )
    run.add_argument("--actor", metavar="NAME_OR_PATH")
    run.add_argument("--suite", help="Sound-matching smoke suite (default: tone-smoke)")
    run.add_argument("--episode", help="Sound-matching episode ID (default: first in suite)")
    run.add_argument("--seed", type=int, help="Sound-matching search seed (default: 0)")
    run.add_argument("--source-cents", type=int)
    run.add_argument(
        "--target-note-index", type=int, help="0..24 represents MIDI 48..72 (default: 12)"
    )
    run.add_argument("--nuisance-seed", type=int)
    run.add_argument("--condition", help="A frozen pitch robustness condition (default: clean)")
    run.add_argument("--device", choices=("cpu", "cuda"))
    run.add_argument(
        "--output", type=Path, required=True, help="New pitch JSON trace or sound-matching bundle"
    )
    for operation in (evaluate, run):
        operation.add_argument(
            "--bundle",
            action="store_true",
            help="Use a resumable pitch bundle (default for sound matching)",
        )
        operation.add_argument(
            "--resume",
            action="store_true",
            help="Resume this bundle; omitted options are restored from it",
        )

    summarize = operations.add_parser(
        "summarize", help="Validate and read a saved result without ML"
    )
    summarize.add_argument("result", type=Path)
    summarize.add_argument("--format", choices=("text", "markdown", "csv"), default="text")
    summarize.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Read committed evidence from an interrupted bundle",
    )
    summarize.add_argument(
        "--verify-replay",
        action="store_true",
        help="Replay recorded sound candidates or pitch actions without loading models",
    )
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


def _summarize_path(
    path: Path, *, format: str, verify_replay: bool = False, allow_incomplete: bool = False
) -> str:
    if path.is_dir():
        from harpy._serialization import read_json_document

        if (path / "manifest.json").is_file():
            manifest = read_json_document(path / "manifest.json")
            from harpy._journal import BUNDLE_SCHEMA_ID

            if manifest.get("schema_id") == BUNDLE_SCHEMA_ID:
                if manifest.get("task_id") == "sound-match-v1":
                    from harpy.tasks.sound_match import durable

                    if verify_replay:
                        durable.verify_replay(path, allow_incomplete=allow_incomplete)
                elif manifest.get("task_id") == "pitch-control-v1":
                    from harpy.experiments import durable

                    if verify_replay:
                        durable.verify_replay(path, allow_incomplete=allow_incomplete)
                else:
                    raise ValueError("Unknown durable bundle task")
                return durable.summarize_bundle(
                    path, format=format, allow_incomplete=allow_incomplete
                )
        from harpy.tasks.sound_match.results import load_result, summarize

        result = load_result(path)
        if verify_replay:
            from harpy.tasks.sound_match.results import verify_replay as verify_sound_replay

            verify_sound_replay(result)
        return summarize(result, format=format)
    if verify_replay:
        raise ValueError("--verify-replay requires a sound-matching bundle directory")

    from harpy._serialization import decode_json_bytes
    from harpy.experiments.results import RESULT_SCHEMA_ID, load_result_document, summarize

    content = path.read_bytes()
    document = decode_json_bytes(content)
    if document.get("schema_id") == RESULT_SCHEMA_ID:
        return summarize(load_result_document(document), format=format)
    from harpy.learning.readout import summarize_report_bytes

    if format == "csv":
        raise ValueError(
            "Historical reports support text/markdown; CSV is available for new results"
        )
    return summarize_report_bytes(content, format=format)


def _execute_sound_match(args: argparse.Namespace) -> int:
    from harpy._cli_bundles import Progress, actor_requests, configuration
    from harpy.tasks.sound_match.durable import evaluate_bundle, summarize_bundle
    from harpy.tasks.sound_match.protocols import smoke_episodes

    if args.device != "cpu":
        raise ValueError("sound-match-v1 currently supports CPU execution only")
    if args.operation == "run" and any(
        getattr(args, name) is not None
        for name in ("source_cents", "target_note_index", "nuisance_seed", "condition")
    ):
        raise ValueError(
            "Pitch source/target/nuisance/condition options do not apply to sound matching"
        )
    args.seed = 0 if args.seed is None else args.seed
    if args.seed < 0:
        raise ValueError("Sound-matching search seed must be nonnegative")
    args.suite = args.suite or "tone-smoke"
    dataset = None
    if args.suite in ("calibration", "validation", "iid-test", "compositional-test"):
        import hashlib

        from harpy.tasks.sound_match.datasets import DATA_DIRECTORY, load_manifest

        manifest = load_manifest(args.suite)
        episodes = manifest.episodes
        dataset = {
            "split": manifest.split,
            "manifest_sha256": manifest.manifest_sha256,
            "catalog_sha256": hashlib.sha256(
                (DATA_DIRECTORY / "catalog.json").read_bytes()
            ).hexdigest(),
        }
    else:
        episodes = smoke_episodes(args.suite)
    if args.operation == "run":
        episode = (
            episodes[0]
            if args.episode is None
            else next((item for item in episodes if item.episode_id == args.episode), None)
        )
        if episode is None:
            raise ValueError(
                "Unknown episode. Choose: " + ", ".join(e.episode_id for e in episodes)
            )
        args.episode = episode.episode_id
        episodes = (episode,)
        requests = [args.actor or "pattern-search"]
    else:
        args.trace = True
        requests = args.actors or ["pattern-search"]
    actors, descriptors, locators = actor_requests(
        requests, task=args.task, device=args.device, seed=args.seed
    )
    evaluate_bundle(
        episodes,
        actors,
        args.output,
        resume=args.resume,
        progress=Progress(),
        configuration=configuration(args, descriptors, dataset=dataset),
        locators=locators,
    )
    print(summarize_bundle(args.output), end="")
    print(f"Saved sound-matching bundle: {args.output}", file=sys.stderr)
    return 0


def _execute(args: argparse.Namespace) -> int:
    if args.operation == "summarize":
        print(
            _summarize_path(
                args.result,
                format=args.format,
                verify_replay=args.verify_replay,
                allow_incomplete=args.allow_incomplete,
            ),
            end="",
        )
        if args.verify_replay:
            print("Verified recorded synthesis and decision replay.", file=sys.stderr)
        return 0
    if args.output.exists() and not getattr(args, "resume", False):
        raise FileExistsError(f"Output already exists; choose a new path: {args.output}")
    if args.operation == "train":
        if args.trainer == "inverse":
            from harpy.tasks.sound_match.inverse import train_inverse

            train_inverse(
                args.output, profile=args.profile or "quick", seed=args.seed, device=args.device
            )
            print(f"Saved inverse artifact: {args.output}", file=sys.stderr)
            return 0
        from harpy.experiments.training import train

        train(
            trainer=args.trainer,
            profile=args.profile or "smoke",
            seed=args.seed,
            device=args.device,
            output=args.output,
        )
        print(f"Saved {args.trainer} artifact: {args.output}", file=sys.stderr)
        return 0

    from harpy._cli_bundles import restore_arguments

    restore_arguments(args)
    args.device = args.device or "cpu"
    if args.task == "sound-match-v1":
        return _execute_sound_match(args)
    if args.seed is not None:
        raise ValueError("--seed applies to sound-match-v1; use --nuisance-seed for a pitch run")
    if args.operation == "run" and (args.suite is not None or args.episode is not None):
        raise ValueError("--suite and --episode on run require --task sound-match-v1")

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
        condition_name = args.condition or "clean"
        condition = next((c for c in ROBUSTNESS_CONDITIONS if c.id == condition_name), None)
        if condition is None:
            raise ValueError(
                "Unknown condition. Choose: " + ", ".join(c.id for c in ROBUSTNESS_CONDITIONS)
            )
        episodes = (
            EpisodeSpec(
                "single",
                6064 if args.source_cents is None else args.source_cents,
                12 if args.target_note_index is None else args.target_note_index,
                0 if args.nuisance_seed is None else args.nuisance_seed,
            ),
        )
        requests = [args.actor or "spectrum-peak"]
        conditions = (condition,)
        trace = True
        protocol = None
    else:
        args.suite = args.suite or "clean"
        if args.suite not in ("clean", "robustness", "confirmation", "smoke"):
            raise ValueError("Unknown pitch suite. Choose: clean, robustness, confirmation, smoke")
        episodes = confirmation_episodes() if args.suite == "confirmation" else benchmark_episodes()
        if args.suite == "smoke":
            episodes = smoke_episodes()
        conditions = (
            ROBUSTNESS_CONDITIONS if args.suite == "robustness" else (ROBUSTNESS_CONDITIONS[0],)
        )
        requests = args.actors or ["spectrum-peak"]
        args.trace = bool(args.trace)
        trace = args.trace
        protocol = protocol_document(args.suite)
    if args.bundle:
        from harpy._cli_bundles import Progress, actor_requests, configuration
        from harpy.experiments.durable import evaluate_bundle, summarize_bundle

        actor_specs, descriptors, locators = actor_requests(
            requests, task=args.task, device=args.device
        )
        evaluate_bundle(
            episodes,
            actor_specs,
            args.output,
            conditions=conditions,
            trace=trace,
            protocol=protocol,
            resume=args.resume,
            progress=Progress(),
            configuration=configuration(args, descriptors),
            locators=locators,
        )
        print(summarize_bundle(args.output), end="")
        print(f"Saved pitch bundle: {args.output}", file=sys.stderr)
        return 0
    actor_specs = tuple(_actor(name, device=args.device) for name in requests)
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
