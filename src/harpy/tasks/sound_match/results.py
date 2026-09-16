"""Complete M1 bundles, strict ledger readout, and explicit expensive DSP replay.

Ordinary loading validates recorded objective components and derives metrics from
the decision ledger. ``verify_replay`` additionally rerenders every candidate.
This format does not yet implement resumable journals or incomplete-run readout.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harpy._serialization import canonical_json_bytes, decode_json_bytes, write_new_bytes
from harpy.tasks.sound_match.models import (
    INFORMATION_TRACK,
    TASK_ID,
    EpisodeSpec,
    _finite,
    _freeze,
    _integer,
    _name,
    _object,
    _patch,
    _plain,
    proposal_error,
)
from harpy.tasks.sound_match.objective import OBJECTIVE_ID, score
from harpy.tasks.sound_match.patch import PATCH_SCHEMA_ID, Patch
from harpy.tasks.sound_match.renderer import FRAME_COUNT, RENDERER_ID, SAMPLE_RATE_HZ, render

RESULT_SCHEMA_ID = "harpy.sound-match.result/1"
COMPLETION_SCHEMA_ID = "harpy.sound-match.complete/1"
CURVE_BUDGETS = (1, 4, 16, 32)


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def audio_hash(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes(order="C")).hexdigest()


def _equal(left: object, right: object, name: str) -> None:
    # Python equality conflates True with 1 and False with 0. Evidence must
    # preserve JSON types as well as values, including nested metric fields.
    encoded_left = canonical_json_bytes({"value": _plain(left)})
    encoded_right = canonical_json_bytes({"value": _plain(right)})
    if encoded_left != encoded_right:
        raise ValueError(f"{name} does not match recomputed evidence")


def _objective(value: object) -> Mapping:
    document = _object(value, {"distance", "spectral", "envelope_rmse", "level_error_db"})
    spectral = document["spectral"]
    if not isinstance(spectral, (list, tuple)) or len(spectral) != 3:
        raise ValueError("objective must contain all three spectral resolutions")
    terms = []
    for item, size in zip(spectral, (512, 2048, 8192), strict=True):
        row = _object(item, {"fft_size", "convergence", "log_distance"})
        if _integer(row["fft_size"], "fft_size") != size:
            raise ValueError("unexpected spectral resolution")
        terms.append(
            _finite(row["convergence"], "convergence", 0)
            + _finite(row["log_distance"], "log_distance", 0)
        )
    distance = _finite(document["distance"], "distance", 0)
    if not math.isclose(distance, sum(terms) / 3, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("distance does not match its spectral components")
    _finite(document["envelope_rmse"], "envelope_rmse", 0)
    _finite(document["level_error_db"], "level_error_db")
    return _freeze(document)


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    evaluation: int
    patch: Patch
    audio_sha256: str
    objective: Mapping
    pitch_error_cents: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer(self.evaluation, "evaluation", 1)
        object.__setattr__(self, "patch", _patch(self.patch))
        _digest(self.audio_sha256)
        object.__setattr__(self, "objective", _objective(self.objective))
        if not isinstance(self.pitch_error_cents, (list, tuple)):
            raise ValueError("pitch_error_cents must be an array")
        errors = tuple(_integer(error, "pitch error") for error in self.pitch_error_cents)
        if len(errors) != len(self.patch.notes_cents):
            raise ValueError("pitch errors must match note count")
        object.__setattr__(self, "pitch_error_cents", errors)

    @property
    def distance(self) -> float:
        return self.objective["distance"]

    def to_dict(self) -> dict:
        return {
            "evaluation": self.evaluation,
            "patch": self.patch.to_dict(),
            "audio_sha256": self.audio_sha256,
            "objective": _plain(self.objective),
            "pitch_error_cents": list(self.pitch_error_cents),
        }

    @classmethod
    def from_dict(cls, value: object) -> CandidateEvidence:
        document = dict(_object(value, set(cls.__dataclass_fields__)))
        document["patch"] = _patch(document["patch"])
        return cls(**document)


_DECISION_FIELDS = {
    "index",
    "action",
    "proposal",
    "status",
    "reason",
    "error_type",
    "evaluation",
    "candidate",
    "model_calls",
}


def _decision_structure(value: object) -> Mapping:
    """Validate scalar types before metrics inspect or add ledger values.

    Budget and terminal-state validation still needs the enclosing EpisodeSpec.
    """
    row = _object(value, _DECISION_FIELDS)
    _integer(row["index"], "decision index", 1)
    for field in ("action", "status"):
        _name(row[field], f"decision {field}")
    for field in ("reason", "error_type"):
        if row[field] is not None:
            _name(row[field], f"decision {field}")
    for field, minimum in (("evaluation", 1), ("model_calls", 0)):
        if row[field] is not None:
            _integer(row[field], field, minimum)
    if row["proposal"] is not None:
        _patch(row["proposal"])
    if row["candidate"] is not None:
        CandidateEvidence.from_dict(row["candidate"])
    return row


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    actor_name: str
    target_audio_sha256: str
    initial: CandidateEvidence | None
    ledger: tuple[Mapping, ...]
    status: str
    reason: str
    error_type: str | None
    dsp_renders: int
    cache_hits: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        _name(self.actor_name, "actor_name")
        _digest(self.target_audio_sha256)
        if self.initial is not None and not isinstance(self.initial, CandidateEvidence):
            raise ValueError("initial must be candidate evidence or null")
        if not isinstance(self.ledger, (list, tuple)):
            raise ValueError("ledger must be an array")
        object.__setattr__(
            self, "ledger", tuple(_freeze(_decision_structure(row)) for row in self.ledger)
        )
        _name(self.status, "terminal status")
        if self.status not in {"submitted", "truncated", "error"}:
            raise ValueError("unknown terminal status")
        _name(self.reason, "terminal reason")
        if self.error_type is not None:
            _name(self.error_type, "error_type")
        _integer(self.dsp_renders, "dsp_renders")
        _integer(self.cache_hits, "cache_hits")
        _finite(self.elapsed_seconds, "elapsed_seconds", 0)

    @property
    def candidates(self) -> tuple[CandidateEvidence, ...]:
        initial = () if self.initial is None else (self.initial,)
        return initial + tuple(
            CandidateEvidence.from_dict(row["candidate"])
            for row in self.ledger
            if row.get("candidate") is not None
        )

    @property
    def last_candidate(self) -> CandidateEvidence | None:
        candidates = self.candidates
        return candidates[-1] if candidates else None

    @property
    def submitted_candidate(self) -> CandidateEvidence | None:
        return self.last_candidate if self.status == "submitted" else None

    @property
    def metrics(self) -> dict:
        candidates = self.candidates
        distances = [candidate.distance for candidate in candidates]
        submitted = self.submitted_candidate
        counts = [row["model_calls"] for row in self.ledger]
        calls = None if any(value is None for value in counts) else sum(counts)
        if self.reason in {"actor_factory_error", "actor_cleanup_error"}:
            calls = None
        return {
            "evaluations": 1 + sum(row["evaluation"] is not None for row in self.ledger),
            "decisions": len(self.ledger),
            "invalid_actions": sum(row["status"] == "invalid" for row in self.ledger),
            "model_calls": calls,
            "initial_distance": None if self.initial is None else self.initial.distance,
            "submitted_distance": None if submitted is None else submitted.distance,
            "last_distance": distances[-1] if distances else None,
            "best_distance": min(distances) if distances else None,
            "best_by_evaluation": {
                str(budget): min(
                    (
                        candidate.distance
                        for candidate in candidates
                        if candidate.evaluation <= budget
                    ),
                    default=None,
                )
                for budget in CURVE_BUDGETS
            },
        }

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "actor_name": self.actor_name,
            "target_audio_sha256": self.target_audio_sha256,
            "initial": None if self.initial is None else self.initial.to_dict(),
            "ledger": _plain(self.ledger),
            "status": self.status,
            "reason": self.reason,
            "error_type": self.error_type,
            "dsp_renders": self.dsp_renders,
            "cache_hits": self.cache_hits,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, value: object) -> EpisodeRecord:
        document = dict(_object(value, set(cls.__dataclass_fields__) | {"metrics"}))
        metrics = document.pop("metrics")
        if document["initial"] is not None:
            document["initial"] = CandidateEvidence.from_dict(document["initial"])
        record = cls(**document)
        # Scalar structure was checked before deriving metrics. Episode-specific
        # budget and state checks run in the enclosing ExperimentResult.
        _equal(metrics, record.metrics, "record metrics")
        return record


def _validate_record(record: EpisodeRecord, episode: EpisodeSpec) -> None:
    evaluations = 1
    previous = record.initial
    if previous is None:
        if (
            record.status != "error"
            or record.reason not in {"initial_render_error", "initial_objective_error"}
            or record.error_type is None
            or record.ledger
        ):
            raise ValueError("missing initial candidate must record an initial evaluation failure")
    elif previous.evaluation != 1 or previous.patch != episode.initial_patch:
        raise ValueError("initial candidate differs from the episode")
    for index, raw in enumerate(record.ledger, 1):
        row = _object(raw, _DECISION_FIELDS)
        if _integer(row["index"], "decision index", 1) != index or index > episode.decision_budget:
            raise ValueError("decision ledger indices exceed or differ from the budget")
        if previous is None:
            raise ValueError("decisions require an initial rendered candidate")
        calls = row["model_calls"]
        if calls is not None:
            _integer(calls, "model_calls")
        if row["error_type"] is not None:
            _name(row["error_type"], "error_type")
        if row["action"] == "submit":
            if (
                row["status"] != "submitted"
                or row["reason"] is not None
                or any(
                    row[field] is not None
                    for field in ("proposal", "evaluation", "candidate", "error_type")
                )
                or index != len(record.ledger)
            ):
                raise ValueError("invalid submit evidence")
        elif row["action"] == "error":
            if (
                row["status"] != "error"
                or row["reason"] != "actor_exception"
                or row["error_type"] is None
                or index != len(record.ledger)
                or any(row[field] is not None for field in ("proposal", "evaluation", "candidate"))
            ):
                raise ValueError("invalid actor exception evidence")
        elif row["action"] == "invalid":
            if (
                row["status"] != "invalid"
                or row["reason"] != "unsupported_action"
                or any(
                    row[field] is not None
                    for field in ("proposal", "evaluation", "candidate", "error_type")
                )
            ):
                raise ValueError("invalid unsupported-action evidence")
        elif row["action"] == "propose":
            if row["proposal"] is None:
                expected = "invalid_patch"
                patch = None
            else:
                patch = _patch(row["proposal"])
                expected = proposal_error(patch, episode, evaluations)
            if expected is not None:
                if (
                    row["status"] != "invalid"
                    or row["reason"] != expected
                    or any(
                        row[field] is not None
                        for field in ("evaluation", "candidate", "error_type")
                    )
                ):
                    raise ValueError("invalid proposal rejection or budget charge")
            else:
                evaluations += 1
                if _integer(row["evaluation"], "evaluation", 1) != evaluations:
                    raise ValueError("candidate evaluation indices must be contiguous")
                if row["status"] == "error":
                    if (
                        row["reason"] not in {"render_error", "objective_error"}
                        or row["candidate"] is not None
                        or row["error_type"] is None
                        or index != len(record.ledger)
                    ):
                        raise ValueError("invalid failed-evaluation evidence")
                elif row["status"] == "accepted":
                    if row["reason"] is not None or row["error_type"] is not None:
                        raise ValueError("accepted proposal cannot contain an error")
                    candidate = CandidateEvidence.from_dict(row["candidate"])
                    if candidate.evaluation != evaluations or candidate.patch != patch:
                        raise ValueError("rendered candidate differs from its proposal")
                    previous = candidate
                else:
                    raise ValueError("legal proposal must be evaluated")
        else:
            raise ValueError("unknown ledger action")
    if evaluations > episode.render_budget:
        raise ValueError("candidate evaluation budget exceeded")
    if record.dsp_renders + record.cache_hits != evaluations:
        raise ValueError("physical render/cache counts do not match logical evaluations")
    last = record.ledger[-1] if record.ledger else None
    if record.status == "submitted":
        if (
            last is None
            or last["status"] != "submitted"
            or record.reason != "explicit_submit"
            or record.error_type is not None
        ):
            raise ValueError("submitted records require an explicit final Submit")
    elif record.status == "truncated":
        if (
            len(record.ledger) != episode.decision_budget
            or last["status"] not in {"accepted", "invalid"}
            or record.reason != "decision_budget_exhausted"
            or record.error_type is not None
        ):
            raise ValueError("truncated records must exhaust decisions without submitting")
    else:
        if record.error_type is None:
            raise ValueError("error records require their exception type")
        if record.reason in {"initial_render_error", "initial_objective_error"}:
            if record.initial is not None or record.ledger:
                raise ValueError("invalid initial failure state")
        elif record.reason == "actor_factory_error":
            if record.initial is None or record.ledger:
                raise ValueError("invalid factory failure state")
        elif record.reason == "actor_cleanup_error":
            if record.initial is None:
                raise ValueError("cleanup requires an initialized episode")
        elif (
            last is None
            or last["status"] != "error"
            or last["reason"] != record.reason
            or last["error_type"] != record.error_type
        ):
            raise ValueError("terminal error differs from its ledger")
    for candidate in record.candidates:
        expected = tuple(
            abs(a - b)
            for a, b in zip(
                candidate.patch.notes_cents, episode.target_patch.notes_cents, strict=True
            )
        )
        if candidate.pitch_error_cents != expected:
            raise ValueError("pitch diagnostics differ from the matched note sets")


def _distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p90": float(np.percentile(values, 90)) if values else None,
    }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    episodes: tuple[EpisodeSpec, ...]
    actors: tuple[Mapping, ...]
    records: tuple[EpisodeRecord, ...]
    targets: tuple[Mapping, ...]
    source: Mapping
    runtime: Mapping
    cache: Mapping
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name in ("episodes", "records", "actors", "targets"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty array")
            object.__setattr__(self, name, tuple(values))
        if any(not isinstance(item, EpisodeSpec) for item in self.episodes):
            raise ValueError("episodes must contain EpisodeSpec values")
        if any(not isinstance(item, EpisodeRecord) for item in self.records):
            raise ValueError("records must contain EpisodeRecord values")
        for name in ("actors", "targets"):
            object.__setattr__(
                self, name, tuple(_freeze(_object(row)) for row in getattr(self, name))
            )
        for name in ("source", "runtime", "cache"):
            object.__setattr__(self, name, _freeze(_object(getattr(self, name))))
        _finite(self.elapsed_seconds, "elapsed_seconds", 0)
        episodes = {episode.episode_id: episode for episode in self.episodes}
        if len(episodes) != len(self.episodes):
            raise ValueError("episode identities must be unique")
        actor_names = []
        for actor in self.actors:
            _object(actor, {"name", "information_track", "metadata", "artifact_files"})
            actor_names.append(_name(actor["name"], "actor name"))
            if actor["information_track"] != INFORMATION_TRACK:
                raise ValueError("unsupported information track")
            _object(actor["metadata"])
            _validate_inventory(actor["artifact_files"])
        if len(set(actor_names)) != len(actor_names):
            raise ValueError("actor identities must be unique")
        if len(self.targets) != len(episodes):
            raise ValueError("target evidence must cover membership exactly")
        for target, episode in zip(self.targets, self.episodes, strict=True):
            _object(
                target,
                {"episode_id", "audio_sha256", "dsp_renders", "cache_hits", "elapsed_seconds"},
            )
            if target["episode_id"] != episode.episode_id:
                raise ValueError("target evidence order differs from membership")
            _digest(target["audio_sha256"])
            if (
                _integer(target["dsp_renders"], "setup renders")
                + _integer(target["cache_hits"], "setup cache hits")
                != 1
            ):
                raise ValueError("target setup must record one render or cache hit")
            _finite(target["elapsed_seconds"], "target setup time", 0)
        expected = [(episode.episode_id, name) for episode in self.episodes for name in actor_names]
        if [(record.episode_id, record.actor_name) for record in self.records] != expected:
            raise ValueError("records must cover ordered episode/actor membership exactly")
        target_hashes = {row["episode_id"]: row["audio_sha256"] for row in self.targets}
        for record in self.records:
            if record.target_audio_sha256 != target_hashes[record.episode_id]:
                raise ValueError("actor target hashes differ")
            _validate_record(record, episodes[record.episode_id])
        _validate_source(self.source)
        _object(self.runtime, {"python", "platform", "numpy"})
        for value in self.runtime.values():
            _name(value, "runtime identity")
        _object(self.cache, {"policy", "max_audio_bytes"})
        if self.cache["policy"] != "lru-waveforms-v1":
            raise ValueError("unknown cache policy")
        _integer(self.cache["max_audio_bytes"], "max_audio_bytes")

    @property
    def metrics(self) -> dict:
        rows = []
        cohorts = list(
            dict.fromkeys(
                (episode.voice_count, episode.editable_fields) for episode in self.episodes
            )
        )
        episode_map = {episode.episode_id: episode for episode in self.episodes}
        for actor in self.actors:
            for voices, mask in cohorts:
                records = [
                    record
                    for record in self.records
                    if record.actor_name == actor["name"]
                    and episode_map[record.episode_id].voice_count == voices
                    and episode_map[record.episode_id].editable_fields == mask
                ]
                metrics = [record.metrics for record in records]
                row = {
                    "actor": actor["name"],
                    "voice_count": voices,
                    "editable_fields": list(mask),
                    "episodes": len(records),
                    **{
                        status: sum(record.status == status for record in records)
                        for status in ("submitted", "truncated", "error")
                    },
                    **{
                        field: sum(item[field] for item in metrics)
                        for field in ("evaluations", "decisions", "invalid_actions")
                    },
                    "no_candidate": sum(record.last_candidate is None for record in records),
                    "model_calls": None
                    if any(item["model_calls"] is None for item in metrics)
                    else sum(item["model_calls"] for item in metrics),
                    "dsp_renders": sum(record.dsp_renders for record in records),
                    "cache_hits": sum(record.cache_hits for record in records),
                    "elapsed_seconds": sum(record.elapsed_seconds for record in records),
                }
                for field in (
                    "initial_distance",
                    "submitted_distance",
                    "last_distance",
                    "best_distance",
                ):
                    row[field] = _distribution(
                        [item[field] for item in metrics if item[field] is not None]
                    )
                row["best_by_evaluation"] = {
                    str(budget): _distribution(
                        [
                            item["best_by_evaluation"][str(budget)]
                            for item in metrics
                            if item["best_by_evaluation"][str(budget)] is not None
                        ]
                    )
                    for budget in CURVE_BUDGETS
                }
                submitted = [
                    record.submitted_candidate
                    for record in records
                    if record.submitted_candidate is not None
                ]
                for field in ("envelope_rmse", "level_error_db"):
                    row[f"submitted_{field}"] = _distribution(
                        [candidate.objective[field] for candidate in submitted]
                    )
                row["submitted_pitch_mae_cents"] = _distribution(
                    [
                        sum(candidate.pitch_error_cents) / len(candidate.pitch_error_cents)
                        for candidate in submitted
                    ]
                )
                rows.append(row)
        return {"groups": rows}

    def manifest(self) -> dict:
        return {
            "schema_id": RESULT_SCHEMA_ID,
            "task_id": TASK_ID,
            "renderer_id": RENDERER_ID,
            "objective_id": OBJECTIVE_ID,
            "patch_schema_id": PATCH_SCHEMA_ID,
            "calibration_status": "uncalibrated",
            "persistence": "complete-bundle-only",
            "episodes": [episode.to_dict() for episode in self.episodes],
            "actors": _plain(self.actors),
            "targets": _plain(self.targets),
            "source": _plain(self.source),
            "runtime": _plain(self.runtime),
            "cache": _plain(self.cache),
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
        }


def _validate_inventory(value: object) -> None:
    if not isinstance(value, (list, tuple)):
        raise ValueError("file inventory must be an array")
    seen = set()
    for item in value:
        row = _object(item, {"path", "sha256", "size_bytes"})
        name = _name(row["path"], "relative artifact path")
        if (
            name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise ValueError("artifact paths must be portable relative paths")
        if name in seen:
            raise ValueError("duplicate artifact path")
        seen.add(name)
        _digest(row["sha256"])
        _integer(row["size_bytes"], "artifact size")


def _validate_source(value: Mapping) -> None:
    _object(value, {"kind", "package_version", "files", "sha256"})
    if value["kind"] != "package-python-files":
        raise ValueError("unknown source identity kind")
    _name(value["package_version"], "package version")
    _validate_inventory(value["files"])
    if not value["files"]:
        raise ValueError("source inventory cannot be empty")
    digest = hashlib.sha256(canonical_json_bytes({"files": _plain(value["files"])})).hexdigest()
    if _digest(value["sha256"]) != digest:
        raise ValueError("source digest differs from its inventory")


def verify_replay(result: ExperimentResult) -> None:
    """Rerender targets and every accepted candidate; verify hashes and objectives.

    This is intentionally separate from ordinary report loading and invokes no actor.
    It requires the result's qualified Python/NumPy renderer environment.
    """
    episodes = {episode.episode_id: episode for episode in result.episodes}
    for record in result.records:
        episode = episodes[record.episode_id]
        target = render(episode.target_patch)
        if audio_hash(target) != record.target_audio_sha256:
            raise ValueError("target replay hash differs; use the recorded renderer/runtime")
        for candidate in record.candidates:
            audio = render(candidate.patch)
            if audio_hash(audio) != candidate.audio_sha256:
                raise ValueError("candidate replay hash differs; use the recorded renderer/runtime")
            _equal(score(target, audio), candidate.objective, "replayed objective")


def _wav_header() -> bytes:
    """The canonical M1 mono IEEE-float WAV header, including its sample count."""
    data_size = FRAME_COUNT * 4
    fmt = struct.pack("<HHIIHH", 3, 1, SAMPLE_RATE_HZ, SAMPLE_RATE_HZ * 4, 4, 32)
    chunks = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"fact" + struct.pack("<II", 4, FRAME_COUNT)
    chunks += b"data" + struct.pack("<I", data_size)
    return b"RIFF" + struct.pack("<I", len(chunks) + data_size) + chunks


def _wav_bytes(audio: np.ndarray) -> bytes:
    """Lossless IEEE-float WAV: fixed PCM timing, no normalization or quantization."""
    samples = np.asarray(audio, dtype="<f4")
    if samples.shape != (FRAME_COUNT,) or not np.isfinite(samples).all():
        raise ValueError("export requires a finite fixed-duration waveform")
    return _wav_header() + samples.tobytes(order="C")


def _wav_payload_hash(content: bytes) -> str:
    """Validate the fixed export format and hash its actual audio without DSP.

    Bundles contain the writer's canonical WAV form, not arbitrary imported WAVs.
    Inspect the already-read bytes and retain only a digest, not every waveform.
    """
    header = _wav_header()
    if len(content) != len(header) + FRAME_COUNT * 4 or not content.startswith(header):
        raise ValueError("audio export must be canonical 96000-frame mono 48 kHz float32 WAV")
    payload = memoryview(content)[len(header) :]
    if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise ValueError("audio export must contain only finite samples")
    return hashlib.sha256(payload).hexdigest()


def _export_audio(patch: Patch, expected_hash: str) -> bytes:
    audio = render(patch)
    if audio_hash(audio) != expected_hash:
        raise ValueError("exported audio differs from recorded renderer evidence")
    return _wav_bytes(audio)


def save_result(result: ExperimentResult, output: Path) -> Path:
    """Create a complete result bundle, including target/submitted WAVs and patches.

    A failed write leaves an incomplete directory without a completion marker.
    Existing directories are never replaced or resumed by this M1 writer.
    """
    if not isinstance(result, ExperimentResult):
        raise ValueError("result must be an ExperimentResult")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inventory = []

    def publish(relative: str, content: bytes) -> None:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new_bytes(path, content)
        inventory.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )

    records = b"".join(canonical_json_bytes(record.to_dict()) for record in result.records)
    publish("records.jsonl", records)
    for index, (episode, target) in enumerate(zip(result.episodes, result.targets, strict=True)):
        publish(
            f"audio/target-{index:04d}.wav",
            _export_audio(episode.target_patch, target["audio_sha256"]),
        )
    for index, record in enumerate(result.records):
        candidate = record.submitted_candidate
        if candidate is not None:
            publish(
                f"audio/submitted-{index:04d}.wav",
                _export_audio(candidate.patch, candidate.audio_sha256),
            )
            publish(
                f"patches/submitted-{index:04d}.json",
                canonical_json_bytes(
                    {
                        "schema_id": PATCH_SCHEMA_ID,
                        "patch": candidate.patch.to_dict(),
                    }
                ),
            )
    manifest = {**result.manifest(), "files": sorted(inventory, key=lambda item: item["path"])}
    encoded = canonical_json_bytes(manifest)
    write_new_bytes(output / "manifest.json", encoded)
    write_new_bytes(
        output / "complete.json",
        canonical_json_bytes(
            {
                "schema_id": COMPLETION_SCHEMA_ID,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        ),
    )
    return output


def load_result(output: Path) -> ExperimentResult:
    """Validate a completed bundle and recompute costs, curves, and aggregate metrics."""
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("result bundle must be a real directory")
    entries = list(output.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("result bundle must not contain symbolic links")
    if not (output / "complete.json").is_file():
        raise ValueError("incomplete result bundle; M1 does not support resume")
    completion = decode_json_bytes((output / "complete.json").read_bytes())
    _object(completion, {"schema_id", "manifest_sha256"})
    if completion["schema_id"] != COMPLETION_SCHEMA_ID:
        raise ValueError("unknown completion schema")
    encoded = (output / "manifest.json").read_bytes()
    if hashlib.sha256(encoded).hexdigest() != _digest(completion["manifest_sha256"]):
        raise ValueError("result manifest was changed")
    manifest = decode_json_bytes(encoded)
    _object(
        manifest,
        {
            "schema_id",
            "task_id",
            "renderer_id",
            "objective_id",
            "patch_schema_id",
            "calibration_status",
            "persistence",
            "episodes",
            "actors",
            "targets",
            "source",
            "runtime",
            "cache",
            "elapsed_seconds",
            "metrics",
            "files",
        },
    )
    for name, expected in {
        "schema_id": RESULT_SCHEMA_ID,
        "task_id": TASK_ID,
        "renderer_id": RENDERER_ID,
        "objective_id": OBJECTIVE_ID,
        "patch_schema_id": PATCH_SCHEMA_ID,
        "calibration_status": "uncalibrated",
        "persistence": "complete-bundle-only",
    }.items():
        if manifest[name] != expected:
            raise ValueError(f"unsupported {name}")
    _validate_inventory(manifest["files"])
    payloads = {}
    audio_hashes = {}
    patch_exports = {}
    for entry in manifest["files"]:
        path = output / entry["path"]
        if not path.is_file():
            raise ValueError("a declared result artifact is missing")
        content = path.read_bytes()
        if (
            len(content) != entry["size_bytes"]
            or hashlib.sha256(content).hexdigest() != entry["sha256"]
        ):
            raise ValueError(f"result artifact was changed: {entry['path']}")
        if entry["path"] == "records.jsonl":
            payloads["records.jsonl"] = content
        elif entry["path"].endswith(".wav"):
            audio_hashes[entry["path"]] = _wav_payload_hash(content)
        elif entry["path"].startswith("patches/") and entry["path"].endswith(".json"):
            patch_exports[entry["path"]] = decode_json_bytes(content)
    names = {path.relative_to(output).as_posix() for path in entries if path.is_file()}
    expected_names = {entry["path"] for entry in manifest["files"]} | {
        "manifest.json",
        "complete.json",
    }
    if names != expected_names or "records.jsonl" not in payloads:
        raise ValueError("bundle files differ from their inventory")
    for name in ("episodes", "actors", "targets"):
        if not isinstance(manifest[name], list):
            raise ValueError(f"{name} must be an array")
    records = tuple(
        EpisodeRecord.from_dict(decode_json_bytes(line))
        for line in payloads["records.jsonl"].splitlines()
    )
    result = ExperimentResult(
        episodes=tuple(EpisodeSpec.from_dict(row) for row in manifest["episodes"]),
        actors=tuple(manifest["actors"]),
        records=records,
        targets=tuple(manifest["targets"]),
        source=manifest["source"],
        runtime=manifest["runtime"],
        cache=manifest["cache"],
        elapsed_seconds=manifest["elapsed_seconds"],
    )
    _equal(manifest["metrics"], result.metrics, "aggregate metrics")
    for index, target in enumerate(result.targets):
        if audio_hashes.get(f"audio/target-{index:04d}.wav") != target["audio_sha256"]:
            raise ValueError("target WAV samples differ from recorded target evidence")
    expected_exports = {"records.jsonl"} | {
        f"audio/target-{index:04d}.wav" for index in range(len(result.episodes))
    }
    for index, record in enumerate(result.records):
        candidate = record.submitted_candidate
        if candidate is not None:
            expected_exports |= {
                f"audio/submitted-{index:04d}.wav",
                f"patches/submitted-{index:04d}.json",
            }
            if audio_hashes.get(f"audio/submitted-{index:04d}.wav") != candidate.audio_sha256:
                raise ValueError("submitted WAV samples differ from recorded candidate evidence")
            patch = patch_exports.get(f"patches/submitted-{index:04d}.json")
            _equal(
                patch,
                {"schema_id": PATCH_SCHEMA_ID, "patch": candidate.patch.to_dict()},
                "submitted patch export",
            )
    if expected_names - {"manifest.json", "complete.json"} != expected_exports:
        raise ValueError("export inventory differs from submitted episode membership")
    return result


def summarize(result: ExperimentResult | Path, format: str = "text") -> str:
    if not isinstance(result, ExperimentResult):
        result = load_result(Path(result))
    if format not in {"text", "markdown", "csv"}:
        raise ValueError("format must be text, markdown, or csv")
    groups = result.metrics["groups"]

    def display(value: object, missing: str = "unavailable") -> str:
        if value is None:
            return missing
        return f"{value:.8g}" if isinstance(value, float) else str(value)

    rows = []
    for group in groups:
        row = {
            "actor": group["actor"],
            "voices": group["voice_count"],
            "editable_fields": ",".join(group["editable_fields"]),
        }
        for field in (
            "episodes",
            "submitted",
            "truncated",
            "error",
            "no_candidate",
            "evaluations",
            "decisions",
            "invalid_actions",
            "model_calls",
            "dsp_renders",
            "cache_hits",
            "elapsed_seconds",
        ):
            row[field] = group[field]
        for field in (
            "initial_distance",
            "submitted_distance",
            "last_distance",
            "best_distance",
            "submitted_envelope_rmse",
            "submitted_level_error_db",
            "submitted_pitch_mae_cents",
        ):
            row[f"{field}_mean"] = group[field]["mean"]
        for field in ("median", "p90"):
            row[f"submitted_distance_{field}"] = group["submitted_distance"][field]
        rows.append(row)
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: display(value, "") for key, value in row.items()} for row in rows)
        return stream.getvalue()
    if format == "markdown":

        def escaped(value: object) -> str:
            return display(value).replace("|", "\\|")

        lines = ["Sound matching — uncalibrated distance; explicit submissions only.", ""]
        for row in rows:
            lines.append(
                f"**{escaped(row['actor'])}** — {row['voices']} voice(s), "
                f"editable fields: {escaped(row['editable_fields'])}. "
                f"Submitted {row['submitted']}/{row['episodes']}; "
                f"truncated {row['truncated']}; errors {row['error']}."
            )
            lines += [
                "",
                "| Distance | Initial mean | Submitted mean | Submitted median | "
                "Submitted p90 | Last mean | Best audition mean |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| Value | "
                + " | ".join(
                    escaped(row[field])
                    for field in (
                        "initial_distance_mean",
                        "submitted_distance_mean",
                        "submitted_distance_median",
                        "submitted_distance_p90",
                        "last_distance_mean",
                        "best_distance_mean",
                    )
                )
                + " |",
                "",
                f"Evaluations: {row['evaluations']}; decisions: {row['decisions']}; "
                f"invalid: {row['invalid_actions']}; model calls: {escaped(row['model_calls'])}.",
                "",
                "Submitted diagnostics (means): envelope RMSE "
                f"{escaped(row['submitted_envelope_rmse_mean'])}; level error "
                f"{escaped(row['submitted_level_error_db_mean'])} dB; pitch MAE "
                f"{escaped(row['submitted_pitch_mae_cents_mean'])} cents.",
                "",
            ]
        return "\n".join(lines) + "\n"
    lines = ["Sound matching (uncalibrated objective; no perceptual success threshold)"]
    for row in rows:
        lines += [
            f"{row['actor']}: {row['voices']} voice(s), fields={row['editable_fields']}; "
            f"submitted {row['submitted']}/{row['episodes']}, truncated={row['truncated']}, "
            f"error={row['error']}, no candidate={row['no_candidate']}",
            f"  Distance: initial mean={display(row['initial_distance_mean'])}, "
            f"submitted mean={display(row['submitted_distance_mean'])}, "
            f"median={display(row['submitted_distance_median'])}, "
            f"p90={display(row['submitted_distance_p90'])}, "
            f"last mean={display(row['last_distance_mean'])}, "
            f"best audition mean={display(row['best_distance_mean'])}",
            f"  Cost: evaluations={row['evaluations']}, decisions={row['decisions']}, "
            f"invalid={row['invalid_actions']}, "
            f"model calls={display(row['model_calls'], 'unknown')}, "
            f"DSP renders={row['dsp_renders']}, cache hits={row['cache_hits']}",
            "  Submitted diagnostics (means): envelope RMSE="
            f"{display(row['submitted_envelope_rmse_mean'])}, "
            f"level error={display(row['submitted_level_error_db_mean'])} dB, "
            f"pitch MAE={display(row['submitted_pitch_mae_cents_mean'])} cents",
        ]
    return "\n".join(lines) + "\n"
