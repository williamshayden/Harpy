"""Portable CLI reconstruction metadata; local model paths are kept separately."""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

CLI_SCHEMA = "harpy.cli-bundle/1"
PITCH_BUILTINS = {
    "spectrum-peak",
    "spectrum-peak-replan",
    "waveform-fft",
    "oracle",
    "reward-search",
    "random",
    "reference",
}
SOUND_BUILTINS = {"fixed", "random-search", "pattern-search"}
OPTION_NAMES = (
    "task",
    "suite",
    "seed",
    "device",
    "trace",
    "episode",
    "source_cents",
    "target_note_index",
    "nuisance_seed",
    "condition",
)


def restore_arguments(args) -> None:
    """Infer omitted arguments on resume without executing metadata as code."""
    if not args.resume:
        return
    from harpy._journal import read_configuration

    saved = read_configuration(args.output)
    config = saved["configuration"]
    if (
        config.get("schema_id") != CLI_SCHEMA
        or config.get("operation") != args.operation
        or not isinstance(config.get("options"), dict)
        or not isinstance(config.get("actors"), list)
    ):
        raise ValueError("Resume requires a bundle created by this CLI operation")
    options = config["options"]
    if set(options) != {name for name in OPTION_NAMES if hasattr(args, name)}:
        raise ValueError("Saved CLI options are malformed")
    for name, value in options.items():
        expected = (
            bool
            if name == "trace"
            else int
            if name in ("seed", "source_cents", "target_note_index", "nuisance_seed")
            else str
        )
        if value is not None and type(value) is not expected:
            raise ValueError(f"Saved CLI option {name} has the wrong type")
        supplied = getattr(args, name)
        if supplied is not None and supplied != value:
            raise ValueError(f"Resume cannot change --{name.replace('_', '-')}")
        setattr(args, name, value)
    expected_task = args.task or "pitch-control-v1"
    if expected_task != saved["task_id"]:
        raise ValueError("Saved CLI task differs from the evaluation task")
    args.bundle = True
    supplied_actors = args.actors if args.operation == "evaluate" else args.actor
    if supplied_actors is not None:
        return  # evaluate_bundle will check portable actor identities and file hashes.
    requests = []
    for descriptor in config["actors"]:
        if not isinstance(descriptor, dict) or set(descriptor) != {"name", "kind", "algorithm"}:
            raise ValueError("Saved actor descriptor is malformed")
        name, kind, algorithm = (descriptor[key] for key in ("name", "kind", "algorithm"))
        if not all(isinstance(value, str) and value for value in (name, kind, algorithm)):
            raise ValueError("Saved actor descriptor fields must be strings")
        if kind == "builtin":
            builtins = SOUND_BUILTINS if args.task == "sound-match-v1" else PITCH_BUILTINS
            if algorithm not in builtins:
                raise ValueError("Unknown saved builtin actor algorithm")
            body = algorithm
        elif kind == "artifact":
            if args.task != "sound-match-v1" and algorithm != "saved":
                raise ValueError("Unknown saved pitch artifact algorithm")
            try:
                body = saved["locators"]["actors"][name]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Supply --actor for relocated model {name}") from exc
            if not isinstance(body, str) or not body:
                raise ValueError("Saved actor locator must be a path string")
            if args.task == "sound-match-v1":
                if algorithm not in ("inverse", "inverse-refine"):
                    raise ValueError("Unknown saved sound actor algorithm")
                body = f"{algorithm}:{body}"
        else:
            raise ValueError("Unknown saved actor kind")
        requests.append(f"{name}={body}")
    if not requests or (args.operation == "run" and len(requests) != 1):
        raise ValueError("Saved actor membership is malformed")
    if args.operation == "evaluate":
        args.actors = requests
    else:
        args.actor = requests[0]


def actor_requests(requests, *, task, device, seed=0):
    """Return executable specs, portable descriptors, and inert local locators."""
    specs, descriptors, locators = [], [], {}
    sound = task == "sound-match-v1"
    for request in requests:
        label, body = request.split("=", 1) if "=" in request else (None, request)
        if not body or label == "":
            raise ValueError("An actor alias must use LABEL=PATH with both parts nonempty")
        if (
            sound
            and label in ("inverse", "inverse-refine")
            and not body.startswith(("inverse:", "inverse-refine:"))
        ):
            body = f"{label}:{body}"
        builtins = SOUND_BUILTINS if sound else PITCH_BUILTINS
        if body in builtins:
            kind, algorithm, locator = "builtin", body, None
            if sound:
                from harpy.tasks.sound_match.actors import builtin_actor_spec

                spec = builtin_actor_spec(body, seed=seed)
            else:
                from harpy.cli import _actor

                spec = _actor(body, device=device)
        else:
            kind = "artifact"
            algorithm = "inverse" if sound else "saved"
            if sound and body.startswith(("inverse:", "inverse-refine:")):
                algorithm, body = body.split(":", 1)
            if not body:
                raise ValueError("An artifact actor requires a directory path")
            locator = str(Path(body).resolve())
            if sound:
                from harpy.tasks.sound_match.inverse import inverse_actor_spec

                spec = inverse_actor_spec(
                    Path(locator), refine=algorithm == "inverse-refine", device=device
                )
            else:
                from harpy.cli import _actor

                spec = _actor(locator, device=device)
        if label is not None:
            spec = replace(spec, name=label)
        if any(item.name == spec.name for item in specs):
            raise ValueError("Actor names must be unique; use LABEL=... for comparisons")
        specs.append(spec)
        descriptors.append({"name": spec.name, "kind": kind, "algorithm": algorithm})
        if locator is not None:
            locators[spec.name] = locator
    return tuple(specs), descriptors, {"actors": locators}


def configuration(args, descriptors, *, dataset=None):
    return {
        "schema_id": CLI_SCHEMA,
        "operation": args.operation,
        "options": {name: getattr(args, name) for name in OPTION_NAMES if hasattr(args, name)},
        "actors": descriptors,
        "dataset": dataset,
    }


class Progress:
    """At most one progress line per five seconds, plus first and final commits."""

    def __init__(self):
        self.last = 0.0

    def __call__(self, update):
        now = time.monotonic()
        if self.last == 0 or now - self.last >= 5 or update["completed"] == update["expected"]:
            print(
                f"Committed {update['completed']}/{update['expected']}: "
                f"{' / '.join(update['key'])} ({update['status']})",
                file=sys.stderr,
                flush=True,
            )
            self.last = now
