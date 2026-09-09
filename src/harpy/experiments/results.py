"""One strict ordinary result format; metrics are always derived from records."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from harpy.envs.models import (
    MAX_ABSOLUTE_ERROR_CENTS,
    MAX_STEPS,
    ControlState,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.experiments.models import (
    EpisodeSpec,
    _finite,
    _freeze,
    _integer,
    _json_bytes,
    _json_loads,
    _name,
    _object,
    _plain,
)
from harpy.learning.pitch_data import PITCH_SPLIT_DIGEST_SHA256

RESULT_SCHEMA_ID = "harpy-experiment-result-v1"
TASK_CONFIGURATION = _freeze(
    {
        "id": "harpy-static-single-sine-task-v1",
        "source_pitch_cents": [4800, 7200],
        "target_note_indices": [0, 24],
        "control_bounds": {"octaves": [-2, 2], "semitones": [-12, 12], "cents": [-100, 100]},
        "actions": [{"id": int(action), "name": action.name} for action in PitchAction],
        "max_steps": 64,
        "success_tolerance_cents": 5,
        "success_tolerance_inclusive": True,
        "sample_rate_hz": 48000,
        "waveform_frames": 262144,
        "spectrum_grid": {"min_cents": 1100, "max_cents": 10900, "step_cents": 5, "size": 1961},
        "training_coordinate_split_sha256": PITCH_SPLIT_DIGEST_SHA256,
    }
)
_ACTOR_FIELDS = {
    "name",
    "observation_mode",
    "estimator",
    "decoder",
    "controller",
    "seed",
    "device",
    "artifact_provenance",
    "artifact_files",
}


def _action(value: object) -> PitchAction:
    return PitchAction(_integer(value, "action"))


def _terminal_document(result: EpisodeResult) -> dict[str, object]:
    return {
        item.name: (
            [int(action) for action in getattr(result, item.name)]
            if item.name in {"actions", "optimal_actions"}
            else result.terminal_reason.value
            if item.name == "terminal_reason"
            else getattr(result, item.name)
        )
        for item in fields(result)
    }


def _terminal_from_document(value: object) -> EpisodeResult:
    document = dict(_object(value, {item.name for item in fields(EpisodeResult)}))
    for name in ("actions", "optimal_actions"):
        if not isinstance(document[name], list):
            raise ValueError("actions must be JSON arrays")
        document[name] = tuple(_action(action) for action in document[name])
    document["terminal_reason"] = TerminalReason(document["terminal_reason"])
    return EpisodeResult(**document)


def _control_trajectory(terminal: EpisodeResult) -> tuple[ControlState, ...]:
    controls = ControlState()
    before = []
    for action in terminal.actions:
        before.append(controls)
        if action is not PitchAction.SUBMIT:
            controls, _ = controls.apply(action)
    return tuple(before)


def _step_reward(terminal: EpisodeResult, controls: ControlState, index: int) -> float:
    action = terminal.actions[index]
    error = abs(terminal.source_pitch_cents + controls.offset_cents - terminal.target_pitch_cents)
    if action is PitchAction.SUBMIT:
        return 1.0 if error <= 5 else -1.0
    updated, applied = controls.apply(action)
    after = abs(terminal.source_pitch_cents + updated.offset_cents - terminal.target_pitch_cents)
    reward = (error - after) / MAX_ABSOLUTE_ERROR_CENTS - 0.00001 if applied else -0.01
    return reward - float(index + 1 == MAX_STEPS)


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("SHA-256 must be a lowercase digest")
    return value


def _evidence(value: object, *, condition, episode: EpisodeSpec, renderer: str) -> None:
    document = _object(
        value,
        {
            "renderer_id",
            "condition_id",
            "nuisance_seed",
            "dry_rms",
            "requested_snr_db",
            "realized_snr_db",
            "waveform_sha256",
            "spectrum_sha256",
        },
    )
    if (
        document["renderer_id"] != renderer
        or document["condition_id"] != condition["id"]
        or _integer(document["nuisance_seed"], "nuisance_seed") != episode.nuisance_seed
        or document["requested_snr_db"] != condition["snr_db"]
    ):
        raise ValueError("evidence must match renderer, condition, and episode nuisance identity")
    if _finite(document["dry_rms"], "dry_rms") <= 0:
        raise ValueError("dry_rms must be positive")
    if condition["snr_db"] is None:
        if document["realized_snr_db"] is not None:
            raise ValueError("noise-free evidence must not claim realized SNR")
    else:
        _finite(document["requested_snr_db"], "requested_snr_db")
        _finite(document["realized_snr_db"], "realized_snr_db")
    _digest(document["waveform_sha256"])
    _digest(document["spectrum_sha256"])


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    actor_name: str
    condition_id: str
    episode_id: str
    terminal: EpisodeResult
    inference_counts: tuple[int, ...]
    estimates: tuple[Mapping[str, object], ...] = ()
    initial_evidence: Mapping[str, object] = None
    trace: tuple[Mapping[str, object], ...] | None = None

    def __post_init__(self) -> None:
        for name in ("actor_name", "condition_id", "episode_id"):
            _name(getattr(self, name), name)
        if not isinstance(self.terminal, EpisodeResult):
            raise ValueError("terminal must be a validated EpisodeResult")
        counts = tuple(self.inference_counts)
        if len(counts) != len(self.terminal.actions):
            raise ValueError("inference_counts must correspond to every recorded action")
        for count in counts:
            _integer(count, "inference_count")
        estimates = tuple(self.estimates)
        previous = 0
        for estimate in estimates:
            _object(estimate, {"step", "estimated_candidate_cents"})
            step = _integer(estimate["step"], "estimate step", minimum=1)
            if not previous < step <= len(counts) or counts[step - 1] == 0:
                raise ValueError("estimates require ordered unique steps with actual inferences")
            _finite(estimate["estimated_candidate_cents"], "estimated_candidate_cents")
            previous = step
        object.__setattr__(self, "inference_counts", counts)
        object.__setattr__(self, "estimates", _freeze(estimates))
        evidence = {} if self.initial_evidence is None else self.initial_evidence
        object.__setattr__(self, "initial_evidence", _freeze(_object(evidence)))
        if self.trace is not None:
            trace = tuple(self.trace)
            if len(trace) != len(counts):
                raise ValueError("trace must cover every recorded action")
            trajectory = _control_trajectory(self.terminal)
            estimates_by_step = {
                item["step"]: item["estimated_candidate_cents"] for item in estimates
            }
            total_reward = 0.0
            for index, step in enumerate(trace):
                _object(
                    step,
                    {
                        "step",
                        "action",
                        "controls",
                        "reward",
                        "inference_count",
                        "estimated_candidate_cents",
                        "evidence",
                    },
                )
                control = trajectory[index]
                if not isinstance(step["controls"], (list, tuple)) or (
                    len(step["controls"]) != 3
                    or any(type(value) is not int for value in step["controls"])
                ):
                    raise ValueError("trace controls must be three integers")
                if (
                    _integer(step["step"], "trace step", minimum=1) != index + 1
                    or _action(step["action"]) is not self.terminal.actions[index]
                    or _plain(step["controls"])
                    != [control.octaves, control.semitones, control.cents]
                    or _integer(step["inference_count"], "inference_count") != counts[index]
                    or step["estimated_candidate_cents"] != estimates_by_step.get(index + 1)
                ):
                    raise ValueError(
                        "trace must match actions, controls, estimates, and inferences"
                    )
                _object(step["evidence"])
                reward = _finite(step["reward"], "reward")
                if not math.isclose(
                    reward, _step_reward(self.terminal, control, index), rel_tol=0, abs_tol=1e-12
                ):
                    raise ValueError("trace reward must match the action dynamics")
                total_reward += reward
            if not math.isclose(total_reward, self.terminal.total_return, rel_tol=0, abs_tol=1e-12):
                raise ValueError("trace rewards must sum to the terminal return")
            object.__setattr__(self, "trace", _freeze(trace))

    def to_document(self) -> dict[str, object]:
        return {
            "actor_name": self.actor_name,
            "condition_id": self.condition_id,
            "episode_id": self.episode_id,
            "terminal": _terminal_document(self.terminal),
            "inference_counts": list(self.inference_counts),
            "estimates": _plain(self.estimates),
            "initial_evidence": _plain(self.initial_evidence),
            "trace": _plain(self.trace),
        }

    @classmethod
    def from_document(cls, value: object) -> EpisodeRecord:
        document = dict(_object(value, set(cls.__dataclass_fields__)))
        for name in ("inference_counts", "estimates"):
            if not isinstance(document[name], list):
                raise ValueError(f"{name} must be a JSON array")
        if document["trace"] is not None and not isinstance(document["trace"], list):
            raise ValueError("trace must be null or a JSON array")
        document["terminal"] = _terminal_from_document(document["terminal"])
        return cls(**document)


def _p99(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = 0.99 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _initial_error(record: EpisodeRecord) -> float | None:
    if record.estimates and record.estimates[0]["step"] == 1:
        return abs(
            float(record.estimates[0]["estimated_candidate_cents"])
            - record.terminal.source_pitch_cents
        )
    return None


def _perception_metrics(records: tuple[EpisodeRecord, ...], *, initial_only: bool) -> dict:
    errors = []
    for record in records:
        if initial_only:
            error = _initial_error(record)
            if error is not None:
                errors.append(error)
            continue
        controls = ControlState()
        cursor = 0
        for estimate in record.estimates:
            # Estimates describe the view before this action. Replay only the
            # prefix needed to reach it, including any bound-blocked actions.
            while cursor < estimate["step"] - 1:
                controls, _ = controls.apply(record.terminal.actions[cursor])
                cursor += 1
            truth = record.terminal.source_pitch_cents + controls.offset_cents
            errors.append(abs(float(estimate["estimated_candidate_cents"]) - truth))
    return {
        "estimates": len(errors),
        "mean_absolute_error_cents": sum(errors) / len(errors) if errors else None,
        "p99_absolute_error_cents": _p99(errors),
        "max_absolute_error_cents": max(errors) if errors else None,
        **{
            f"within_{tolerance}_cents_rate": (
                sum(error <= tolerance for error in errors) / len(errors) if errors else None
            )
            for tolerance in (1, 2, 5)
        },
    }


def _metrics(records: tuple[EpisodeRecord, ...]) -> dict:
    count = len(records)
    terminal = tuple(record.terminal for record in records)
    actions = sum(len(item.actions) for item in terminal)
    inferences = sum(sum(record.inference_counts) for record in records)
    errors = [item.final_absolute_error_cents for item in terminal]
    successes = sum(item.submitted_success for item in terminal)
    truncations = sum(item.terminal_reason is TerminalReason.BUDGET_EXHAUSTED for item in terminal)
    invalid = sum(item.invalid_action_count for item in terminal)
    return {
        "episodes": count,
        "submitted_successes": successes,
        "truncations": truncations,
        "invalid_actions": invalid,
        "actions": actions,
        "inferences": inferences,
        "submitted_success_rate": successes / count,
        "submitted_within_1_cent_rate": sum(
            item.submitted_success and item.within_1_cent for item in terminal
        )
        / count,
        "final_within_1_cent_rate": sum(item.within_1_cent for item in terminal) / count,
        "final_within_5_cents_rate": sum(item.within_5_cents for item in terminal) / count,
        "final_mean_absolute_error_cents": sum(item.final_absolute_error_cents for item in terminal)
        / count,
        "final_p99_absolute_error_cents": _p99(errors),
        "final_max_absolute_error_cents": max(errors),
        "mean_actions": actions / count,
        "mean_inferences": inferences / count,
        "truncation_rate": truncations / count,
        "invalid_action_rate": invalid / actions,
        "initial_perception": _perception_metrics(records, initial_only=True),
        "trajectory_perception": _perception_metrics(records, initial_only=False),
    }


def _validate_protocol(protocol: Mapping[str, object], episodes: tuple[EpisodeSpec, ...]) -> None:
    if "episodes" not in protocol or _plain(protocol["episodes"]) != [
        item.to_document() for item in episodes
    ]:
        raise ValueError("protocol must retain the exact ordered episode membership")
    if "digest_sha256" in protocol:
        preimage = {key: value for key, value in protocol.items() if key != "digest_sha256"}
        if (
            hashlib.sha256(_json_bytes(preimage).rstrip(b"\n")).hexdigest()
            != protocol["digest_sha256"]
        ):
            raise ValueError("protocol digest must match its document")
    if protocol.get("id") in {
        "harpy-clean-development-v1",
        "harpy-clean-confirmation-v1",
        "harpy-clean-engineering-smoke-v1",
    }:
        from harpy.experiments.protocols import validate_protocol_document

        validate_protocol_document(_plain(protocol))


def _validate_provenance(provenance: Mapping[str, object], actors: tuple[Mapping, ...]) -> None:
    """Validate recorded identities and runtime claims, permitting producer extensions."""
    source = _object(provenance["source"])
    status = source.get("status")
    if "status" in source and status not in ("known", "unknown"):
        raise ValueError("source status must be known or unknown")
    if status == "unknown":
        reason = source.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("unknown source must explain its missing provenance")
    elif "package_sha256" not in source:
        raise ValueError(
            "source must identify its package or explicitly explain unknown provenance"
        )
    if "package_sha256" in source:
        _digest(source["package_sha256"])
    if "identity" in source:
        from harpy.learning.artifacts import _source_from_document

        _source_from_document(source["identity"])

    runtime = _object(provenance["runtime"])
    if "elapsed_wall_time_seconds" in runtime:
        elapsed = _finite(runtime["elapsed_wall_time_seconds"], "elapsed_wall_time_seconds")
        if elapsed < 0:
            raise ValueError("elapsed_wall_time_seconds must be nonnegative")
    if "evaluation_devices" in runtime:
        devices = _object(runtime["evaluation_devices"])
        if devices != {actor["name"]: actor["device"] for actor in actors}:
            raise ValueError("runtime evaluation_devices must match the declared actors")
    for field in ("python", "platform", "numpy", "gymnasium", "timing_scope"):
        if field in runtime:
            _name(runtime[field], f"runtime {field}")
    for field in ("torch", "stable_baselines3"):
        if field in runtime and runtime[field] is not None:
            _name(runtime[field], f"runtime {field}")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Paired ordinary measurements, without an implicit scientific eligibility claim."""

    episodes: tuple[EpisodeSpec, ...]
    actors: tuple[Mapping[str, object], ...]
    conditions: tuple[Mapping[str, object], ...]
    records: tuple[EpisodeRecord, ...]
    provenance: Mapping[str, object]
    protocol: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        from harpy.envs.robustness import ROBUSTNESS_RENDERER_ID, ConditionSpec

        episodes = tuple(self.episodes)
        actors = tuple(self.actors)
        conditions = tuple(self.conditions)
        records = tuple(self.records)
        if not episodes or not all(isinstance(item, EpisodeSpec) for item in episodes):
            raise ValueError("episodes must be nonempty EpisodeSpec values")
        if len({item.id for item in episodes}) != len(episodes):
            raise ValueError("episode ids must be unique")
        for actor in actors:
            _object(actor, _ACTOR_FIELDS)
            for name in ("name", "estimator", "decoder", "controller"):
                _name(actor[name], name)
            ObservationMode(actor["observation_mode"])
            if actor["device"] not in {"cpu", "cuda"}:
                raise ValueError("actor device must be cpu or cuda")
            if actor["seed"] is not None:
                _integer(actor["seed"], "actor seed")
            _object(actor["artifact_provenance"])
            if not isinstance(actor["artifact_files"], (list, tuple)):
                raise ValueError("artifact_files must be an array")
            for item in actor["artifact_files"]:
                _object(item, {"path", "sha256", "size_bytes"})
                _name(item["path"], "artifact path")
                digest = item["sha256"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                ):
                    raise ValueError("artifact file SHA-256 must be a lowercase digest")
                _integer(item["size_bytes"], "artifact size")
        if not actors or len({item["name"] for item in actors}) != len(actors):
            raise ValueError("actor names must be nonempty and unique")
        for condition in conditions:
            ConditionSpec.from_document(condition)
        if not conditions or len({item["id"] for item in conditions}) != len(conditions):
            raise ValueError("condition ids must be nonempty and unique")
        expected = tuple(
            (actor["name"], condition["id"], episode.id)
            for condition in conditions
            for episode in episodes
            for actor in actors
        )
        if (
            not all(isinstance(record, EpisodeRecord) for record in records)
            or tuple(
                (record.actor_name, record.condition_id, record.episode_id) for record in records
            )
            != expected
        ):
            raise ValueError("records must cover the exact ordered condition/episode/actor product")
        by_id = {episode.id: episode for episode in episodes}
        by_condition = {condition["id"]: condition for condition in conditions}
        paired_evidence = {}
        for record in records:
            episode = by_id[record.episode_id]
            if (record.terminal.source_pitch_cents, record.terminal.target_note_index) != (
                episode.source_pitch_cents,
                episode.target_note_index,
            ):
                raise ValueError("terminal truth must match its declared episode")
            condition = by_condition[record.condition_id]
            _evidence(
                record.initial_evidence,
                condition=condition,
                episode=episode,
                renderer=ROBUSTNESS_RENDERER_ID,
            )
            pair = (record.condition_id, record.episode_id)
            if pair in paired_evidence and paired_evidence[pair] != record.initial_evidence:
                raise ValueError("paired actors must receive the same initial evidence")
            paired_evidence[pair] = record.initial_evidence
            if record.trace is not None:
                if record.trace[0]["evidence"] != record.initial_evidence:
                    raise ValueError("first trace view must match initial evidence")
                for step in record.trace:
                    _evidence(
                        step["evidence"],
                        condition=condition,
                        episode=episode,
                        renderer=ROBUSTNESS_RENDERER_ID,
                    )
        _object(self.provenance, {"source", "runtime", "renderer"})
        _validate_provenance(self.provenance, actors)
        if self.provenance["renderer"] != ROBUSTNESS_RENDERER_ID:
            raise ValueError("renderer provenance must match the task")
        if self.protocol is not None:
            _validate_protocol(_object(self.protocol), episodes)
        for name, value in (("episodes", episodes), ("records", records)):
            object.__setattr__(self, name, value)
        for name, value in (
            ("actors", actors),
            ("conditions", conditions),
            ("provenance", self.provenance),
            ("protocol", self.protocol),
        ):
            object.__setattr__(self, name, _freeze(value))

    @property
    def rows(self) -> tuple[dict, ...]:
        partitions = tuple(dict.fromkeys(episode.partition for episode in self.episodes))
        partition_by_id = {episode.id: episode.partition for episode in self.episodes}
        grouped = {}
        for record in self.records:
            key = (record.actor_name, record.condition_id, partition_by_id[record.episode_id])
            grouped.setdefault(key, []).append(record)
        return tuple(
            {
                "actor": actor["name"],
                "condition": condition["id"],
                "partition": partition,
                "observation_mode": actor["observation_mode"],
                "estimator": actor["estimator"],
                "decoder": actor["decoder"],
                "controller": actor["controller"],
                "seed": actor["seed"],
                "device": actor["device"],
                **_metrics(tuple(grouped[(actor["name"], condition["id"], partition)])),
            }
            for actor in self.actors
            for condition in self.conditions
            for partition in partitions
        )

    @property
    def paired_clean(self) -> tuple[dict, ...]:
        if "clean" not in {condition["id"] for condition in self.conditions}:
            return ()
        lookup = {(r.actor_name, r.condition_id, r.episode_id): r for r in self.records}
        partitions = tuple(dict.fromkeys(episode.partition for episode in self.episodes))
        groups = (
            (actor, condition, partition)
            for actor in self.actors
            for condition in self.conditions
            for partition in partitions
        )
        rows = []
        for actor, condition, partition in groups:
            if condition["id"] == "clean":
                continue
            pairs = [
                (
                    lookup[(actor["name"], "clean", ep.id)],
                    lookup[(actor["name"], condition["id"], ep.id)],
                )
                for ep in self.episodes
                if ep.partition == partition
            ]
            perception_changes = [
                _initial_error(after) - _initial_error(before)
                for before, after in pairs
                if _initial_error(before) is not None and _initial_error(after) is not None
            ]
            rows.append(
                {
                    "actor": actor["name"],
                    "condition": condition["id"],
                    "partition": partition,
                    "seed": actor["seed"],
                    "episodes": len(pairs),
                    "rescued": sum(
                        not a.terminal.submitted_success and b.terminal.submitted_success
                        for a, b in pairs
                    ),
                    "regressed": sum(
                        a.terminal.submitted_success and not b.terminal.submitted_success
                        for a, b in pairs
                    ),
                    "changed_outcome": sum(a.terminal != b.terminal for a, b in pairs),
                    "mean_change_final_error_cents": sum(
                        b.terminal.final_absolute_error_cents
                        - a.terminal.final_absolute_error_cents
                        for a, b in pairs
                    )
                    / len(pairs),
                    "mean_change_actions": sum(
                        len(b.terminal.actions) - len(a.terminal.actions) for a, b in pairs
                    )
                    / len(pairs),
                    "mean_change_inferences": sum(
                        sum(b.inference_counts) - sum(a.inference_counts) for a, b in pairs
                    )
                    / len(pairs),
                    "paired_initial_estimates": len(perception_changes),
                    "mean_change_initial_perception_error_cents": (
                        sum(perception_changes) / len(perception_changes)
                        if perception_changes
                        else None
                    ),
                }
            )
        return tuple(rows)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_id": RESULT_SCHEMA_ID,
            "task": _plain(TASK_CONFIGURATION),
            "episodes": [episode.to_document() for episode in self.episodes],
            "actors": _plain(self.actors),
            "conditions": _plain(self.conditions),
            "records": [record.to_document() for record in self.records],
            "provenance": _plain(self.provenance),
            "protocol": _plain(self.protocol),
            "rows": list(self.rows),
            "paired_clean": list(self.paired_clean),
        }

    @classmethod
    def from_document(cls, value: object) -> ExperimentResult:
        doc = _object(
            value,
            {
                "schema_id",
                "task",
                "episodes",
                "actors",
                "conditions",
                "records",
                "provenance",
                "protocol",
                "rows",
                "paired_clean",
            },
        )
        if doc["schema_id"] != RESULT_SCHEMA_ID:
            raise ValueError("unsupported experiment result schema")
        if _json_bytes(doc["task"]) != _json_bytes(TASK_CONFIGURATION):
            raise ValueError("task configuration must match the declared experiment task")
        for name in ("episodes", "actors", "conditions", "records", "rows", "paired_clean"):
            if not isinstance(doc[name], list):
                raise ValueError(f"{name} must be a JSON array")
        result = cls(
            tuple(EpisodeSpec.from_document(item) for item in doc["episodes"]),
            tuple(doc["actors"]),
            tuple(doc["conditions"]),
            tuple(EpisodeRecord.from_document(item) for item in doc["records"]),
            doc["provenance"],
            doc["protocol"],
        )
        if _json_bytes(doc["rows"]) != _json_bytes(result.rows) or (
            _json_bytes(doc["paired_clean"]) != _json_bytes(result.paired_clean)
        ):
            raise ValueError("metrics and paired outcomes must be re-derived from records")
        return result


def save_result(result: ExperimentResult, path: Path) -> None:
    """Write a new result atomically; an existing destination is never replaced."""
    from harpy.learning.artifacts import write_new_bytes

    if not isinstance(result, ExperimentResult) or not isinstance(path, Path):
        raise ValueError("save_result requires an ExperimentResult and Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new_bytes(path, _json_bytes(result.to_document()))


def load_result(path: Path) -> ExperimentResult:
    if not isinstance(path, Path):
        raise ValueError("path must be a Path")
    return ExperimentResult.from_document(_json_loads(path.read_bytes()))


def summarize(result: ExperimentResult, *, format: str = "text") -> str:
    """Readable measurements grouped by actor, condition, and source partition."""
    if not isinstance(result, ExperimentResult) or format not in {"text", "markdown", "csv"}:
        raise ValueError("summarize requires a result and text, markdown, or csv format")
    metric_rows = result.rows
    paired_rows = result.paired_clean
    headings = [
        "Actor",
        "Track",
        "Condition",
        "Partition",
        "N",
        "Success",
        "Within 1c",
        "Final MAE (c)",
        "Actions",
        "Inferences",
        "Initial pitch MAE (c)",
    ]
    rows = [
        [
            row["actor"],
            row["observation_mode"],
            row["condition"],
            row["partition"],
            row["episodes"],
            row["submitted_success_rate"],
            row["submitted_within_1_cent_rate"],
            row["final_mean_absolute_error_cents"],
            row["mean_actions"],
            row["mean_inferences"],
            row["initial_perception"]["mean_absolute_error_cents"],
        ]
        for row in metric_rows
    ]
    if format == "csv":

        def flatten(document, prefix=""):
            flattened = {}
            for key, value in document.items():
                name = f"{prefix}.{key}" if prefix else key
                if isinstance(value, Mapping):
                    flattened.update(flatten(value, name))
                else:
                    flattened[name] = value
            return flattened

        paired = {(row["actor"], row["condition"], row["partition"]): row for row in paired_rows}
        documents = []
        for row in metric_rows:
            document = flatten(row)
            changes = paired.get((row["actor"], row["condition"], row["partition"]))
            if changes is not None:
                document.update(flatten(changes, "paired_clean"))
            documents.append(document)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=list(dict.fromkeys(key for doc in documents for key in doc))
        )
        writer.writeheader()
        writer.writerows(documents)
        return output.getvalue()

    def cell(value):
        value = "—" if value is None else f"{value:.4f}" if type(value) is float else str(value)
        return value.replace("|", "\\|") if format == "markdown" else value

    lines = [
        "Harpy experiment measurements",
        "",
        "Success is explicit submission within 5 cents.",
        "Initial perception uses each episode's first view; trajectory estimates are separate.",
        "Inference counts count estimator/policy evaluations, including classical estimators.",
        "Observation tracks, conditions, seeds, and partitions are not pooled.",
        "",
    ]
    delimiter = " | " if format == "markdown" else "\t"
    lines.append(delimiter.join(headings))
    if format == "markdown":
        lines.append(delimiter.join(["---"] * len(headings)))
    lines.extend(delimiter.join(cell(value) for value in row) for row in rows)
    lines.extend(["", "Tail errors and control failures (cents):", ""])
    tails = [
        "Actor",
        "Condition",
        "Partition",
        "Seed",
        "Device",
        "Truncated",
        "Invalid actions",
        "Final p99",
        "Final max",
        "Initial pitch p99",
        "Initial pitch max",
    ]
    lines.append(delimiter.join(tails))
    if format == "markdown":
        lines.append(delimiter.join(["---"] * len(tails)))
    for row in metric_rows:
        lines.append(
            delimiter.join(
                cell(value)
                for value in [
                    row["actor"],
                    row["condition"],
                    row["partition"],
                    row["seed"],
                    row["device"],
                    row["truncations"],
                    row["invalid_actions"],
                    row["final_p99_absolute_error_cents"],
                    row["final_max_absolute_error_cents"],
                    row["initial_perception"]["p99_absolute_error_cents"],
                    row["initial_perception"]["max_absolute_error_cents"],
                ]
            )
        )
    if paired_rows:
        lines.extend(["", "Paired with the same actor and episodes under clean conditions:"])
        for row in paired_rows:
            lines.append(
                f"{row['actor']} / {row['condition']} / {row['partition']}: "
                f"{row['regressed']} regressed, {row['rescued']} rescued of {row['episodes']}."
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "RESULT_SCHEMA_ID",
    "TASK_CONFIGURATION",
    "EpisodeRecord",
    "ExperimentResult",
    "load_result",
    "save_result",
    "summarize",
]
