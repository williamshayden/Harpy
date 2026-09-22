"""Opt-in durable envelopes around the frozen ordinary v1 scientific records.

Legacy JSON serialization, task physics, and scientific membership stay intact.
The shared journal adds explicit errors, interrupted attempts, portable artifact
identities, locking, streaming readout, and resume of missing members only.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from harpy._journal import (
    BundleReport,
    Journal,
    MetricStore,
    ScanState,
    artifact_inventory,
    callable_identity,
    exact,
    integer,
    lock_bundle,
    manifest_for,
    object_fields,
    read_configuration,
    report,
    scan,
    source_identity,
    validate_callable_identity,
    validate_inventory,
    validate_source_identity,
)
from harpy._serialization import read_json_document
from harpy.envs.models import ControlState, ObservationMode, TerminalReason
from harpy.envs.robustness import (
    CLEAN_CONDITION,
    ROBUSTNESS_RENDERER_ID,
    ConditionSpec,
    RobustnessEvidenceCache,
)
from harpy.experiments.models import ActorSpec, Decision, EpisodeSpec, _plain
from harpy.experiments.results import (
    TASK_CONFIGURATION,
    EpisodeRecord,
    ExperimentResult,
    _validate_protocol,
)
from harpy.experiments.runner import _run_member, _runtime

TASK_ID = "pitch-control-v1"
DEFAULT_CACHE_BYTES = 16 * 1024 * 1024


def _actor(spec: ActorSpec) -> dict:
    if spec._verify_artifact is not None:
        spec._verify_artifact()
    return {
        **spec.to_document(),
        "artifact_provenance": {
            **_plain(spec.artifact_provenance),
            "factory_identity": callable_identity(spec.factory),
        },
        "artifact_files": artifact_inventory(spec.artifact_paths),
    }


def _identity(episodes, actors, conditions, trace, protocol, cache_bytes) -> dict:
    runtime = _runtime()
    # Optional actors may import these lazily. Freeze installed versions without
    # importing a model runtime, so first construction cannot change identity.
    for package in ("torch", "stable_baselines3"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = None
    return {
        "task_configuration": _plain(TASK_CONFIGURATION),
        "episodes": [episode.to_document() for episode in episodes],
        "actors": [_actor(actor) for actor in actors],
        "conditions": [condition.to_document() for condition in conditions],
        "trace": trace,
        "protocol": None if protocol is None else _plain(protocol),
        "source": source_identity(),
        "runtime": runtime,
        "renderer_id": ROBUSTNESS_RENDERER_ID,
        "cache_max_bytes": cache_bytes,
    }


class _Contract:
    def __init__(self, path: Path, manifest: dict):
        self.path, self.manifest = path, manifest
        if manifest["task_id"] != TASK_ID:
            raise ValueError("bundle is not a pitch-control task")
        identity = manifest["identity"]
        object_fields(
            identity,
            {
                "task_configuration",
                "episodes",
                "actors",
                "conditions",
                "trace",
                "protocol",
                "source",
                "runtime",
                "renderer_id",
                "cache_max_bytes",
            },
        )
        exact(identity["task_configuration"], _plain(TASK_CONFIGURATION), "frozen v1 task")
        if identity["renderer_id"] != ROBUSTNESS_RENDERER_ID:
            raise ValueError("unsupported v1 renderer")
        if type(identity["trace"]) is not bool:
            raise ValueError("trace must be a bool")
        integer(identity["cache_max_bytes"], "cache_max_bytes")
        for name in ("episodes", "actors", "conditions"):
            if not isinstance(identity[name], list) or not identity[name]:
                raise ValueError(f"{name} must be a nonempty array")
        self.episodes = tuple(EpisodeSpec.from_document(row) for row in identity["episodes"])
        self.actors = identity["actors"]
        self.conditions = tuple(ConditionSpec.from_document(row) for row in identity["conditions"])
        for values, key in (
            (self.episodes, lambda item: item.id),
            (self.actors, lambda item: item["name"]),
            (self.conditions, lambda item: item.id),
        ):
            if len({key(item) for item in values}) != len(values):
                raise ValueError("v1 membership identities must be unique")
        expected = len(self.episodes) * len(self.actors) * len(self.conditions)
        if expected != manifest["expected"]:
            raise ValueError("expected count differs from ordered v1 membership")
        if identity["protocol"] is not None:
            _validate_protocol(identity["protocol"], self.episodes)
        # The source document is the same portable package inventory used by
        # both durable adapters, independent of legacy checkout-status labels.
        validate_source_identity(identity["source"])
        for actor in self.actors:
            validate_inventory(actor["artifact_files"])
            validate_callable_identity(actor["artifact_provenance"]["factory_identity"])
        self.provenance = {
            "source": {"package_sha256": identity["source"]["sha256"]},
            "runtime": identity["runtime"],
            "renderer": ROBUSTNESS_RENDERER_ID,
        }
        # Validate actor/provenance schemas even when no scientific record has
        # yet committed, using their existing strict v1 validation helpers.
        from harpy.experiments.results import _ACTOR_FIELDS, _validate_provenance

        for actor in self.actors:
            object_fields(actor, _ACTOR_FIELDS)
            for name in ("name", "estimator", "decoder", "controller"):
                if not isinstance(actor[name], str) or not actor[name]:
                    raise ValueError("actor identity must contain nonempty strings")
            ObservationMode(actor["observation_mode"])
            if actor["device"] not in {"cpu", "cuda"}:
                raise ValueError("unsupported actor device")
            if actor["seed"] is not None:
                integer(actor["seed"], "actor seed")
            if not isinstance(actor["artifact_provenance"], dict):
                raise ValueError("actor provenance must be an object")
        _validate_provenance(self.provenance, tuple(self.actors))
        self._paired_member = None
        self._paired_evidence = None

    def member(self, index):
        per_condition = len(self.episodes) * len(self.actors)
        condition_index, inner = divmod(index, per_condition)
        episode_index, actor_index = divmod(inner, len(self.actors))
        return (
            self.episodes[episode_index],
            self.actors[actor_index],
            self.conditions[condition_index],
        )

    def key(self, index):
        episode, actor, condition = self.member(index)
        return [condition.id, episode.id, actor["name"]]

    def validate(self, payload: dict, index: int) -> None:
        object_fields(payload, {"key", "status", "error", "record", "artifacts"})
        if payload["key"] != self.key(index) or payload["artifacts"] != []:
            raise ValueError("v1 envelope differs from membership or artifact contract")
        if payload["record"] is None:
            if payload["status"] != "error":
                raise ValueError("missing terminal v1 record must be an explicit error")
            object_fields(payload["error"], {"phase", "error_type", "costs"})
            if payload["error"]["phase"] != "rollout" or payload["error"]["costs"] != "unknown":
                raise ValueError("failed v1 rollout must preserve unknown partial costs")
            if (
                not isinstance(payload["error"]["error_type"], str)
                or not payload["error"]["error_type"]
            ):
                raise ValueError("failed v1 rollout requires an exception type")
            return
        if payload["error"] is not None:
            raise ValueError("completed v1 records cannot claim rollout failure")
        episode, actor, condition = self.member(index)
        record = EpisodeRecord.from_document(payload["record"])
        ExperimentResult(
            (episode,),
            (actor,),
            (condition.to_document(),),
            (record,),
            self.provenance,
        )
        wanted = (
            "truncated"
            if record.terminal.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
            else "submitted"
        )
        if payload["status"] != wanted:
            raise ValueError("v1 terminal status differs from envelope")
        if (record.trace is not None) != self.manifest["identity"]["trace"]:
            raise ValueError("v1 trace availability differs from frozen configuration")
        pair = (condition.id, episode.id)
        evidence = _plain(record.initial_evidence)
        if index == 0 or self._paired_member != pair:
            self._paired_member, self._paired_evidence = pair, evidence
        else:
            exact(evidence, self._paired_evidence, "paired v1 initial evidence")


def evaluate_bundle(
    episodes: Sequence[EpisodeSpec],
    actors: Sequence[ActorSpec],
    output: Path,
    *,
    conditions: Sequence[ConditionSpec] = (CLEAN_CONDITION,),
    trace: bool = False,
    protocol: Mapping | None = None,
    resume: bool = False,
    progress: Callable[[dict], None] | None = None,
    cache_max_bytes: int = DEFAULT_CACHE_BYTES,
    configuration: Mapping | None = None,
    locators: Mapping | None = None,
) -> BundleReport:
    episodes, actors, conditions = tuple(episodes), tuple(actors), tuple(conditions)
    for values, expected in (
        (episodes, EpisodeSpec),
        (actors, ActorSpec),
        (conditions, ConditionSpec),
    ):
        if not values or any(not isinstance(value, expected) for value in values):
            raise ValueError("durable v1 evaluation requires explicit nonempty typed membership")
    if type(trace) is not bool:
        raise ValueError("trace must be a bool")
    integer(cache_max_bytes, "cache_max_bytes")
    identity = _identity(episodes, actors, conditions, trace, protocol, cache_max_bytes)
    manifest = manifest_for(
        TASK_ID, identity, len(episodes) * len(actors) * len(conditions), configuration
    )
    output = Path(output)
    contract = _Contract(output, manifest)
    with Journal(output, manifest, resume=resume, locators=locators) as journal:
        journal.recover(key_at=contract.key, validate=contract.validate)
        cache = RobustnessEvidenceCache(max_bytes=cache_max_bytes)
        for index in range(journal.state.completed, manifest["expected"]):
            condition_index, inner = divmod(index, len(episodes) * len(actors))
            episode_index, actor_index = divmod(inner, len(actors))
            key = contract.key(index)
            journal.begin(key)
            try:
                record = _run_member(
                    episodes[episode_index],
                    actors[actor_index],
                    conditions[condition_index],
                    cache,
                    trace,
                )
            except Exception as exc:
                payload = {
                    "key": key,
                    "status": "error",
                    "record": None,
                    "artifacts": [],
                    "error": {
                        "phase": "rollout",
                        "error_type": type(exc).__name__,
                        "costs": "unknown",
                    },
                }
            else:
                status = (
                    "truncated"
                    if record.terminal.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
                    else "submitted"
                )
                payload = {
                    "key": key,
                    "status": status,
                    "record": record.to_document(),
                    "error": None,
                    "artifacts": [],
                }
            exact(
                _identity(episodes, actors, conditions, trace, protocol, cache_max_bytes),
                identity,
                "running evaluation identity",
            )
            contract.validate(payload, index)
            journal.commit(payload)
            if progress is not None:
                progress(
                    {
                        "completed": journal.state.completed,
                        "expected": manifest["expected"],
                        "key": key,
                        "status": payload["status"],
                    }
                )
        journal.finish(key_at=contract.key, validate=contract.validate)
    return read_bundle(output)


def _manifest(path: Path):
    read_configuration(path)
    manifest = read_json_document(path / "manifest.json")
    try:
        return manifest, _Contract(path, manifest)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("v1 bundle manifest contains malformed fields") from exc


def iter_records(path: Path, *, allow_incomplete: bool = False) -> Iterator[dict]:
    path = Path(path)
    manifest, contract = _manifest(path)
    if not allow_incomplete and not (path / "complete.json").is_file():
        raise ValueError("bundle is incomplete; opt in to partial readout")
    with lock_bundle(path, exclusive=False), MetricStore() as refs:
        state = ScanState()
        yield from scan(
            path, manifest, state, key_at=contract.key, validate=contract.validate, references=refs
        )
        if not allow_incomplete and not state.is_complete:
            raise ValueError("bundle is incomplete")


def read_bundle(path: Path, *, allow_incomplete: bool = False) -> BundleReport:
    path = Path(path)
    manifest, contract = _manifest(path)
    if not allow_incomplete and not (path / "complete.json").is_file():
        raise ValueError("bundle is incomplete; opt in to partial readout")
    groups = {}
    with lock_bundle(path, exclusive=False), MetricStore() as store:
        state = ScanState()
        for index, payload in enumerate(
            scan(
                path,
                manifest,
                state,
                key_at=contract.key,
                validate=contract.validate,
                references=store,
            )
        ):
            _, actor, condition = contract.member(index)
            key = (actor["name"], condition.id)
            group = groups.setdefault(
                key,
                {
                    "actor": actor["name"],
                    "condition": condition.id,
                    "episodes": 0,
                    "submitted": 0,
                    "truncated": 0,
                    "error": 0,
                    "submitted_successes": 0,
                    "actions": 0,
                    "inferences": 0,
                    "invalid_actions": 0,
                    "unknown_rollout_costs": 0,
                },
            )
            group["episodes"] += 1
            group[payload["status"]] += 1
            if payload["record"] is None:
                group["unknown_rollout_costs"] += 1
                continue
            record = EpisodeRecord.from_document(payload["record"])
            terminal = record.terminal
            group["submitted_successes"] += terminal.submitted_success
            group["actions"] += len(terminal.actions)
            group["inferences"] += sum(record.inference_counts)
            group["invalid_actions"] += terminal.invalid_action_count
            group_id = repr(key)
            store.add(group_id, "final_absolute_error_cents", terminal.final_absolute_error_cents)
            store.add(group_id, "actions", len(terminal.actions))
            store.add(group_id, "inferences", sum(record.inference_counts))
            controls, cursor = ControlState(), 0
            for estimate in record.estimates:
                while cursor < estimate["step"] - 1:
                    controls, _ = controls.apply(terminal.actions[cursor])
                    cursor += 1
                error = abs(
                    estimate["estimated_candidate_cents"]
                    - terminal.source_pitch_cents
                    - controls.offset_cents
                )
                store.add(group_id, "trajectory_perception_error_cents", error)
                if estimate["step"] == 1:
                    store.add(group_id, "initial_perception_error_cents", error)
        if not allow_incomplete and not state.is_complete:
            raise ValueError("bundle is incomplete")
        for key, group in groups.items():
            group["submitted_success_rate"] = group["submitted_successes"] / group["episodes"]
            for name in (
                "final_absolute_error_cents",
                "actions",
                "inferences",
                "initial_perception_error_cents",
                "trajectory_perception_error_cents",
            ):
                group[f"{name}_distribution"] = store.distribution(repr(key), name)
        return report(path, manifest, state, {"groups": list(groups.values())})


def summarize_bundle(path: Path, format: str = "text", *, allow_incomplete: bool = False) -> str:
    result = read_bundle(path, allow_incomplete=allow_incomplete)
    if format not in {"text", "markdown", "csv"}:
        raise ValueError("format must be text, markdown, or csv")
    columns = [
        "actor",
        "condition",
        "episodes",
        "submitted",
        "truncated",
        "error",
        "submitted_successes",
        "actions",
        "inferences",
        "invalid_actions",
        "unknown_rollout_costs",
        "bundle_complete",
        "completed",
        "expected",
        "interrupted_attempts",
    ]
    rows = [
        [
            *[group[name] for name in columns[:11]],
            result.complete,
            result.completed,
            result.expected,
            result.interrupted_attempts,
        ]
        for group in result.metrics["groups"]
    ]
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
        return stream.getvalue()
    heading = (
        f"Pitch control v1: {'COMPLETE' if result.complete else 'INCOMPLETE'}; "
        f"committed {result.completed}/{result.expected}, missing={result.missing}."
    )
    if result.interrupted_attempts:
        heading += f" Interrupted attempts={result.interrupted_attempts}; their costs are unknown."
    if format == "markdown":
        lines = [
            heading,
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        lines += [
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in rows
        ]
    else:
        lines = [heading]
        lines += [
            f"{row[0]} / {row[1]}: submitted={row[3]}/{row[2]}, "
            f"truncated={row[4]}, error={row[5]}, "
            f"successes={row[6]}, known actions={row[7]}, known inferences={row[8]}, "
            f"unknown rollout costs={row[10]}"
            for row in rows
        ]
    return "\n".join(lines) + "\n"


def verify_replay(path: Path, *, allow_incomplete: bool = False) -> None:
    """Replay committed actions through the renderer without loading any model.

    Estimates and inference counts remain recorded actor evidence; this verifies
    environment state, terminal physics, rendered views, and optional step traces.
    Failed rollouts without a terminal record have no invented replay trajectory.
    """
    path = Path(path)
    _, contract = _manifest(path)
    cache = RobustnessEvidenceCache(max_bytes=DEFAULT_CACHE_BYTES)

    class ReplayActor:
        def __init__(self, decisions):
            self.decisions = decisions

        def decide(self, observation):
            return next(self.decisions)

    for index, payload in enumerate(iter_records(path, allow_incomplete=allow_incomplete)):
        if payload["record"] is None:
            continue
        episode, actor, condition = contract.member(index)
        record = EpisodeRecord.from_document(payload["record"])
        estimates = {row["step"]: row["estimated_candidate_cents"] for row in record.estimates}
        decisions = iter(
            Decision(action, estimates.get(step), count)
            for step, (action, count) in enumerate(
                zip(record.terminal.actions, record.inference_counts, strict=True), 1
            )
        )

        spec = ActorSpec(
            actor["name"],
            partial(ReplayActor, decisions),
            ObservationMode(actor["observation_mode"]),
            actor["estimator"],
            actor["decoder"],
            actor["controller"],
        )
        replayed = _run_member(episode, spec, condition, cache, record.trace is not None)
        exact(replayed.to_document(), record.to_document(), "replayed v1 trajectory")


__all__ = [
    "BundleReport",
    "evaluate_bundle",
    "iter_records",
    "read_bundle",
    "read_configuration",
    "summarize_bundle",
    "verify_replay",
]
