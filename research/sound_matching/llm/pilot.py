"""Run an explicit, bounded text-feature LLM smoke comparison.

No provider is selected or contacted by default. Invoke from a source checkout
with an explicit model and either a trusted command JSON file or a loopback
HTTP bridge plus local token file. Smoke membership is public and consumed;
these runs never qualify a model or reproduce the frozen v2 study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    read_json_document,
    write_new_bytes,
)
from harpy.tasks.sound_match import builtin_actor_spec
from harpy.tasks.sound_match.durable import evaluate_bundle, iter_records, read_bundle
from harpy.tasks.sound_match.protocols import smoke_episodes

from .actor import PROTOCOL_ID, SYSTEM_PROMPT, llm_actor_spec
from .features import FEATURE_ID, feature_configuration
from .transport import CommandTransport, HttpTransport

PILOT_SCHEMA = "harpy.llm-smoke-pilot/1"
COMPLETION_SCHEMA = "harpy.llm-smoke-completion/1"
LLM_NAME = "harnessbridge-text"
SUITES = ("tone-smoke", "chord-smoke")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path: Path, document: dict) -> None:
    write_new_bytes(path, canonical_json_bytes(document))


def _file_record(root: Path, path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("pilot evidence must contain regular files")
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> list[dict]:
    return [
        _file_record(root, path)
        for path in sorted(root.rglob("*"))
        if (path.is_file() or path.is_symlink())
        and path.relative_to(root).as_posix() not in exclude
    ]


def _verify_inventory(root: Path, inventory: list[dict]) -> None:
    seen = set()
    for record in inventory:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or record["path"] in seen:
            raise ValueError("invalid pilot inventory path")
        seen.add(record["path"])
        candidate = root / relative
        if not candidate.resolve().is_relative_to(root.resolve()):
            raise ValueError("pilot inventory escapes output")
        if _file_record(root, candidate) != record:
            raise ValueError("pilot evidence hash or size changed")


def _transport_metadata(value: dict | None) -> dict:
    value = dict(value or {"kind": "injected"})
    fields = {
        "injected": {"kind"},
        "command": {"kind", "argv_sha256", "argument_count", "executable"},
        "http": {"kind", "base_url", "authorization"},
    }
    if value.get("kind") not in fields or set(value) != fields[value["kind"]]:
        raise ValueError("unexpected transport metadata; credentials are never recorded")
    # Callers cannot smuggle arbitrary authentication/header/configuration fields
    # into the persisted configuration. CLI values are derived below, not copied.
    return value


def _request_costs(directory: Path, kind: str, maximum: int) -> dict:
    controllers = []
    episode_ids, controller_ids = set(), set()
    for path in sorted(directory.glob("*.jsonl")):
        events = []
        with path.open("rb") as stream:
            for line in stream:
                events.append(decode_json_bytes(line))
        counts = Counter(event["event"] for event in events)
        bindings = [event for event in events if event["event"] == "episode_binding"]
        if len(bindings) != 1 or bindings[0].get("sent_to_model") is not False:
            raise ValueError("request journal requires one local-only episode binding")
        episode_id = bindings[0]["episode_id"]
        controller_id = bindings[0]["controller_id"]
        if episode_id in episode_ids or controller_id in controller_ids:
            raise ValueError("duplicate request journal controller or episode")
        if any(event.get("controller_id") != controller_id for event in events):
            raise ValueError("request journal controller identity changed")
        episode_ids.add(episode_id)
        controller_ids.add(controller_id)
        attempts = counts["attempt_start"]
        if attempts > maximum:
            raise ValueError("LLM transport attempts exceeded the episode cap")
        controllers.append(
            {
                "journal": path.name,
                "episode_id": episode_id,
                "controller_id": controller_id,
                "attempts_started": attempts,
                "responses": counts["response"],
                "transport_failures": counts["transport_failure"],
                "parse_rejections": counts["parse_rejection"],
                "controller_closed": bool(counts["controller_close"]),
            }
        )
    total = sum(row["attempts_started"] for row in controllers)
    return {
        "transport_kind": kind,
        "controllers": controllers,
        "transport_attempts_started": total,
        "http_attempts_started": total if kind == "http" else 0,
        "command_attempts_started": total if kind == "command" else 0,
        "attempt_scope": "transport invocations; wire delivery and native execution are unknown",
        "native_provider_calls": None,
        "native_provider_calls_status": "unknown; bridge/command attempts are not native calls",
        "automatic_retries": 0,
        "max_attempts_per_episode": maximum,
    }


def run_pilot(
    output: Path,
    *,
    transport_factory: Callable[[], object],
    model: str,
    suite: str = "tone-smoke",
    max_model_calls: int = 4,
    timeout_seconds: float = 120,
    transport_metadata: dict | None = None,
) -> dict:
    """Evaluate explicit transport against ordinary classical smoke baselines.

    The caller owns authentication and the transport; no credential value or
    token path is accepted as provenance. Existing output directories, including
    empty directories, are rejected. Interrupted pilots have no resume shortcut.
    """
    if suite not in SUITES:
        raise ValueError("pilot suite must be tone-smoke or chord-smoke")
    if type(max_model_calls) is not int or not 1 <= max_model_calls <= 64:
        raise ValueError("max_model_calls must be an integer from 1 through 64")
    if not isinstance(model, str) or not model.strip() or any(c in model for c in "\r\n\x00"):
        raise ValueError("an explicit model name is required")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    transport = _transport_metadata(transport_metadata)
    episodes = smoke_episodes(suite)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    immutable = output / "immutable"
    immutable.mkdir()
    (output / "requests").mkdir()
    configuration = {
        "model": model,
        "suite": suite,
        "max_model_calls": max_model_calls,
        "timeout_seconds": timeout_seconds,
        "transport": transport,
        "observation_protocol": "text-derived-audio-features-v1",
        "feature_id": FEATURE_ID,
        "feature_configuration": feature_configuration(),
        "response_protocol": PROTOCOL_ID,
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "native_provider_calls": "unknown",
        "backend_tool_isolation": "operator-owned; not enforced by this pilot",
        "qualification": "public engineering smoke only; no frozen-study or model-quality claim",
    }
    _write(immutable / "configuration.json", configuration)
    write_new_bytes(immutable / "system-prompt.txt", SYSTEM_PROMPT.encode("utf-8"))
    adapter = Path(__file__).resolve().parent
    source_pairs = []
    for source in sorted(adapter.iterdir()):
        if source.suffix not in (".py", ".mjs"):
            continue
        if source.is_symlink() or not source.is_file():
            raise ValueError("adapter source snapshots require regular files")
        target = immutable / source.name
        write_new_bytes(target, source.read_bytes())
        source_pairs.append((source, target))
    artifacts = tuple(sorted(path for path in immutable.iterdir() if path.is_file()))
    pilot_manifest = {
        "schema_id": PILOT_SCHEMA,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "episodes": [episode.to_dict() for episode in episodes],
        "actors": ["fixed", "random-search", "pattern-search", LLM_NAME],
        "random_search_seed": 0,
        "immutable_files": [_file_record(output, path) for path in artifacts],
        "journal_policy": "requests are mutable operational evidence, never actor artifacts",
        "direct_audio_input": False,
        "external_command_implementation": "operator-trusted; native internals are not verified",
    }
    _write(output / "pilot-manifest.json", pilot_manifest)
    actors = (
        builtin_actor_spec("fixed"),
        builtin_actor_spec("random-search", seed=0),
        builtin_actor_spec("pattern-search"),
        llm_actor_spec(
            transport_factory=transport_factory,
            model=model,
            max_calls=max_model_calls,
            name=LLM_NAME,
            journal_directory=output / "requests",
            artifact_paths=(*artifacts, output / "pilot-manifest.json"),
        ),
    )
    try:
        result = evaluate_bundle(
            episodes,
            actors,
            output / "evaluation",
            configuration={"pilot_schema_id": PILOT_SCHEMA, **configuration},
        )
        for original, snapshot in source_pairs:
            if _sha256(original) != _sha256(snapshot):
                raise ValueError("adapter source changed while pilot was running")
        _verify_inventory(output, pilot_manifest["immutable_files"])
        rows = list(iter_records(output / "evaluation"))
        statuses = Counter(row["status"] for row in rows)
        invalid_actions = sum(
            row["record"]["metrics"]["invalid_actions"] for row in rows if row["record"] is not None
        )
        costs = _request_costs(output / "requests", transport["kind"], max_model_calls)
        succeeded = statuses["submitted"] == len(rows) and invalid_actions == 0
        expected_episodes = {row["key"][0] for row in rows if row["key"][1] == LLM_NAME}
        journal_episodes = {row["episode_id"] for row in costs["controllers"]}
        if not journal_episodes <= expected_episodes:
            raise ValueError("request journal identifies an unknown pilot episode")
        if succeeded and (
            journal_episodes != expected_episodes
            or not all(row["controller_closed"] for row in costs["controllers"])
        ):
            raise ValueError("completed pilot requires closed journals for every LLM episode")
        outcome = {
            "schema_id": "harpy.llm-smoke-outcome/1",
            "status": "complete" if succeeded else "completed-with-errors",
            "bundle_complete": result.complete,
            "records": len(rows),
            "terminal_statuses": dict(statuses),
            "invalid_actions": invalid_actions,
            "metrics": result.metrics,
            "request_costs": costs,
            "interpretation": "workflow evidence only; no model advantage or perceptual claim",
        }
        _write(output / "outcome.json", outcome)
        if succeeded:
            _write(
                output / "complete.json",
                {
                    "schema_id": COMPLETION_SCHEMA,
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "files": _inventory(output),
                    "scope": "Protocol complete; model quality unqualified",
                },
            )
        return outcome
    except BaseException as exc:
        # Keep started attempts and partial durable records; never persist raw
        # exception messages that might contain command stderr or credentials.
        _write(
            output / "interrupted.json",
            {
                "schema_id": "harpy.llm-smoke-interruption/1",
                "error_type": type(exc).__name__,
                "complete": False,
                "native_provider_calls": None,
            },
        )
        raise


def verify_pilot(output: Path) -> dict:
    """Check exact completed-pilot bytes and strictly read its ordinary bundle."""
    output = Path(output).resolve()
    completion = read_json_document(output / "complete.json")
    if completion.get("schema_id") != COMPLETION_SCHEMA:
        raise ValueError("unsupported pilot completion schema")
    _verify_inventory(output, completion["files"])
    if _inventory(output, exclude=("complete.json",)) != completion["files"]:
        raise ValueError("pilot file membership changed")
    manifest = read_json_document(output / "pilot-manifest.json")
    if manifest.get("schema_id") != PILOT_SCHEMA:
        raise ValueError("unsupported pilot schema")
    _verify_inventory(output, manifest["immutable_files"])
    outcome = read_json_document(output / "outcome.json")
    result = read_bundle(output / "evaluation")
    if not result.complete or outcome["status"] != "complete":
        raise ValueError("pilot did not complete successfully")
    if result.metrics != outcome["metrics"]:
        raise ValueError("pilot metrics differ from strict bundle readout")
    configuration = manifest["configuration"]
    costs = _request_costs(
        output / "requests",
        configuration["transport"]["kind"],
        configuration["max_model_calls"],
    )
    if costs != outcome["request_costs"]:
        raise ValueError("pilot request accounting differs from retained journals")
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default="tone-smoke")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-calls", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--command-json", type=Path)
    choice.add_argument("--base-url")
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args(argv)
    if bool(args.base_url) != bool(args.token_file):
        parser.error("--token-file is required only with --base-url")
    try:
        if args.command_json is not None:
            command_bytes = args.command_json.read_bytes()
            if len(command_bytes) > 65536:
                raise ValueError("command configuration exceeds the 64 KiB limit")
            command = json.loads(command_bytes)
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(argument, str) for argument in command)
            ):
                raise ValueError("command JSON must be a nonempty argv array")
            command = tuple(command)
            # Validate configuration without starting a process or request.
            CommandTransport(command, timeout_seconds=args.timeout_seconds)

            def factory():
                return CommandTransport(command, timeout_seconds=args.timeout_seconds)

            transport = {
                "kind": "command",
                "argv_sha256": hashlib.sha256(
                    json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "argument_count": len(command),
                "executable": Path(command[0]).name,
            }
        else:
            HttpTransport(
                args.base_url,
                token_file=args.token_file,
                timeout_seconds=args.timeout_seconds,
            )

            def factory():
                return HttpTransport(
                    args.base_url,
                    token_file=args.token_file,
                    timeout_seconds=args.timeout_seconds,
                )

            transport = {
                "kind": "http",
                "base_url": args.base_url,
                "authorization": "local-token-file",
            }
        outcome = run_pilot(
            args.output,
            transport_factory=factory,
            model=args.model,
            suite=args.suite,
            max_model_calls=args.max_model_calls,
            timeout_seconds=args.timeout_seconds,
            transport_metadata=transport,
        )
    except KeyboardInterrupt:
        print("Pilot interrupted; partial evidence has no completion marker.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            f"Pilot did not complete ({type(exc).__name__}); inspect retained evidence.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Pilot {outcome['status']}: {outcome['records']} retained records; "
        f"{outcome['request_costs']['transport_attempts_started']} transport attempts. "
        "Native provider calls are unknown."
    )
    return 0 if outcome["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
