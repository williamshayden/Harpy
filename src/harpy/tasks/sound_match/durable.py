"""Resumable sound-match bundles with streaming validation and readout.

This is a distinct format from the M1 complete-only result bundle. M1's reader
and writer remain unchanged. Scientific records are committed once per actor
and episode; restarting an interrupted member creates a separately visible
attempt and never retries an already committed error or truncation.
"""

from __future__ import annotations

import csv
import io
import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

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
from harpy._serialization import canonical_json_bytes, decode_json_bytes, read_json_document
from harpy.tasks.sound_match.models import (
    INFORMATION_TRACK,
    TASK_ID,
    ActorSpec,
    EpisodeSpec,
    _plain,
)
from harpy.tasks.sound_match.objective import OBJECTIVE_ID
from harpy.tasks.sound_match.patch import PATCH_SCHEMA_ID
from harpy.tasks.sound_match.renderer import FRAME_COUNT, RENDERER_ID
from harpy.tasks.sound_match.results import (
    CURVE_BUDGETS,
    EpisodeRecord,
    _validate_record,
    _wav_bytes,
    _wav_header,
    _wav_payload_hash,
    audio_hash,
)
from harpy.tasks.sound_match.runner import DEFAULT_CACHE_BYTES, _episode, _WaveCache

CALIBRATION_STATUS = "continuous-distance-frozen; perceptual-acceptance-unqualified"


def _actor(spec: ActorSpec) -> dict:
    return {
        "name": spec.name,
        "information_track": INFORMATION_TRACK,
        "metadata": {**_plain(spec.metadata), "factory_identity": callable_identity(spec.factory)},
        "artifact_files": artifact_inventory(spec.artifact_paths),
    }


def _identity(episodes, actors, cache_bytes) -> dict:
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = None
    return {
        "episodes": [episode.to_dict() for episode in episodes],
        "actors": [_actor(actor) for actor in actors],
        "source": source_identity(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch_version,
        },
        "cache": {"policy": "lru-waveforms-v1", "max_audio_bytes": cache_bytes},
        "renderer_id": RENDERER_ID,
        "objective_id": OBJECTIVE_ID,
        "patch_schema_id": PATCH_SCHEMA_ID,
        "calibration_status": CALIBRATION_STATUS,
    }


class _Contract:
    def __init__(self, path: Path, manifest: dict):
        self.path, self.manifest = path, manifest
        if manifest["task_id"] != TASK_ID:
            raise ValueError("bundle is not a sound-match task")
        identity = manifest["identity"]
        object_fields(
            identity,
            {
                "episodes",
                "actors",
                "source",
                "runtime",
                "cache",
                "renderer_id",
                "objective_id",
                "patch_schema_id",
                "calibration_status",
            },
        )
        if any(
            identity[key] != expected
            for key, expected in {
                "renderer_id": RENDERER_ID,
                "objective_id": OBJECTIVE_ID,
                "patch_schema_id": PATCH_SCHEMA_ID,
                "calibration_status": CALIBRATION_STATUS,
            }.items()
        ):
            raise ValueError("unsupported sound-match task configuration")
        if not isinstance(identity["episodes"], list) or not isinstance(identity["actors"], list):
            raise ValueError("episode and actor membership must be arrays")
        self.episodes = tuple(EpisodeSpec.from_dict(item) for item in identity["episodes"])
        self.actors = identity["actors"]
        if not self.episodes or len({ep.episode_id for ep in self.episodes}) != len(self.episodes):
            raise ValueError("episodes must be nonempty and unique")
        names = []
        for actor in self.actors:
            object_fields(actor, {"name", "information_track", "metadata", "artifact_files"})
            if not isinstance(actor["name"], str) or not actor["name"]:
                raise ValueError("actor name is required")
            if actor["information_track"] != INFORMATION_TRACK or not isinstance(
                actor["metadata"], dict
            ):
                raise ValueError("unsupported actor information track or metadata")
            validate_inventory(actor["artifact_files"])
            validate_callable_identity(actor["metadata"]["factory_identity"])
            names.append(actor["name"])
        if not names or len(set(names)) != len(names):
            raise ValueError("actors must be nonempty and unique")
        if len(self.episodes) * len(self.actors) != manifest["expected"]:
            raise ValueError("expected count differs from ordered membership")
        validate_source_identity(identity["source"])
        object_fields(identity["runtime"], {"python", "platform", "numpy", "torch"})
        if any(
            not isinstance(identity["runtime"][name], str) or not identity["runtime"][name]
            for name in ("python", "platform", "numpy")
        ):
            raise ValueError("runtime versions must be declared")
        torch_version = identity["runtime"]["torch"]
        if torch_version is not None and (not isinstance(torch_version, str) or not torch_version):
            raise ValueError("optional Torch version must be a nonempty string or null")
        object_fields(identity["cache"], {"policy", "max_audio_bytes"})
        if identity["cache"]["policy"] != "lru-waveforms-v1":
            raise ValueError("unknown cache policy")
        integer(identity["cache"]["max_audio_bytes"], "cache size")
        self._paired_episode = None
        self._paired_target = None

    def member(self, index):
        return self.episodes[index // len(self.actors)], self.actors[index % len(self.actors)]

    def key(self, index):
        episode, actor = self.member(index)
        return [episode.episode_id, actor["name"]]

    def validate(self, payload: dict, index: int) -> None:
        object_fields(
            payload, {"key", "status", "error", "record", "target", "artifacts", "operational"}
        )
        if payload["key"] != self.key(index):
            raise ValueError("sound record membership differs from the manifest")
        episode, actor = self.member(index)
        operational = payload["operational"]
        object_fields(
            operational,
            {"target_dsp_renders", "target_cache_hits", "export_dsp_renders", "export_cache_hits"},
        )
        for value in operational.values():
            integer(value, "operational render count")
        if operational["target_dsp_renders"] + operational["target_cache_hits"] != 1:
            raise ValueError("target setup must account for one render attempt or cache lookup")
        exports = {item["role"]: item for item in payload["artifacts"]}
        if payload["record"] is None:
            if payload["status"] != "error" or payload["target"] is not None or exports:
                raise ValueError("missing sound record requires a target setup failure")
            object_fields(payload["error"], {"phase", "error_type"})
            if (
                payload["error"]["phase"] != "target_setup"
                or not isinstance(payload["error"]["error_type"], str)
                or not payload["error"]["error_type"]
            ):
                raise ValueError("invalid target setup failure evidence")
            if operational["export_dsp_renders"] or operational["export_cache_hits"]:
                raise ValueError("target setup failure cannot export audio")
            return
        if payload["error"] is not None:
            raise ValueError("ordinary sound records carry their own failure evidence")
        record = EpisodeRecord.from_dict(payload["record"])
        if record.episode_id != episode.episode_id or record.actor_name != actor["name"]:
            raise ValueError("record identities differ from membership")
        _validate_record(record, episode)
        if payload["status"] != record.status:
            raise ValueError("record terminal status differs from envelope")
        object_fields(payload["target"], {"audio_sha256"})
        if payload["target"]["audio_sha256"] != record.target_audio_sha256:
            raise ValueError("target hash differs from record")
        if index == 0 or self._paired_episode != episode.episode_id:
            self._paired_episode, self._paired_target = (
                episode.episode_id,
                record.target_audio_sha256,
            )
        elif self._paired_target != record.target_audio_sha256:
            raise ValueError("paired sound actors received different target evidence")
        expected_roles = {"target_audio"}
        if record.submitted_candidate is not None:
            expected_roles |= {"submitted_audio", "submitted_patch"}
        if set(exports) != expected_roles:
            raise ValueError("exports differ from terminal record membership")
        for role in ("target_audio", "submitted_audio"):
            if (
                role in exports
                and exports[role]["size_bytes"] != len(_wav_header()) + FRAME_COUNT * 4
            ):
                raise ValueError("audio export size differs from the fixed task duration")
        target_bytes = (self.path / exports["target_audio"]["path"]).read_bytes()
        if _wav_payload_hash(target_bytes) != record.target_audio_sha256:
            raise ValueError("target WAV payload differs from scientific evidence")
        candidate = record.submitted_candidate
        if candidate is not None:
            audio = (self.path / exports["submitted_audio"]["path"]).read_bytes()
            if _wav_payload_hash(audio) != candidate.audio_sha256:
                raise ValueError("submitted WAV payload differs from scientific evidence")
            patch_export = exports["submitted_patch"]
            expected_patch = {"schema_id": PATCH_SCHEMA_ID, "patch": candidate.patch.to_dict()}
            if patch_export["size_bytes"] != len(canonical_json_bytes(expected_patch)):
                raise ValueError("patch export size differs from the submitted candidate")
            patch = decode_json_bytes((self.path / patch_export["path"]).read_bytes())
            exact(
                patch,
                expected_patch,
                "submitted patch",
            )
            if operational["export_dsp_renders"] + operational["export_cache_hits"] != 1:
                raise ValueError("submitted export must account for its waveform lookup")
        elif operational["export_dsp_renders"] or operational["export_cache_hits"]:
            raise ValueError("unsubmitted records must not have submitted export costs")


def evaluate_bundle(
    episodes: Sequence[EpisodeSpec],
    actors: Sequence[ActorSpec],
    output: Path,
    *,
    resume: bool = False,
    progress: Callable[[dict], None] | None = None,
    cache_max_audio_bytes: int = DEFAULT_CACHE_BYTES,
    configuration: Mapping | None = None,
    locators: Mapping | None = None,
) -> BundleReport:
    episodes, actors = tuple(episodes), tuple(actors)
    if not episodes or any(not isinstance(item, EpisodeSpec) for item in episodes):
        raise ValueError("episodes must contain EpisodeSpec values")
    if not actors or any(not isinstance(item, ActorSpec) for item in actors):
        raise ValueError("actors must contain ActorSpec values")
    integer(cache_max_audio_bytes, "cache_max_audio_bytes")
    identity = _identity(episodes, actors, cache_max_audio_bytes)
    manifest = manifest_for(TASK_ID, identity, len(episodes) * len(actors), configuration)
    output = Path(output)
    contract = _Contract(output, manifest)
    with Journal(output, manifest, resume=resume, locators=locators) as journal:
        journal.recover(key_at=contract.key, validate=contract.validate)
        cache = _WaveCache(cache_max_audio_bytes)
        for index in range(journal.state.completed, manifest["expected"]):
            episode = episodes[index // len(actors)]
            actor = actors[index % len(actors)]
            key = contract.key(index)
            journal.begin(key)
            before_renders, before_hits = cache.renders, cache.hits
            operational = {
                "target_dsp_renders": 0,
                "target_cache_hits": 0,
                "export_dsp_renders": 0,
                "export_cache_hits": 0,
            }
            try:
                target = cache.get(episode.target_patch)
            except Exception as exc:
                payload = {
                    "key": key,
                    "status": "error",
                    "error": {"phase": "target_setup", "error_type": type(exc).__name__},
                    "record": None,
                    "target": None,
                    "artifacts": [],
                    "operational": operational,
                }
            else:
                record = _episode(episode, actor, target, cache)
                exports = [journal.artifact("target_audio", _wav_bytes(target), "wav")]
                candidate = record.submitted_candidate
                if candidate is not None:
                    export_renders, export_hits = cache.renders, cache.hits
                    audio = cache.get(candidate.patch)
                    if audio_hash(audio) != candidate.audio_sha256:
                        raise ValueError("submitted export differs from candidate evidence")
                    exports += [
                        journal.artifact("submitted_audio", _wav_bytes(audio), "wav"),
                        journal.artifact(
                            "submitted_patch",
                            canonical_json_bytes(
                                {"schema_id": PATCH_SCHEMA_ID, "patch": candidate.patch.to_dict()}
                            ),
                            "json",
                        ),
                    ]
                    operational["export_dsp_renders"] = cache.renders - export_renders
                    operational["export_cache_hits"] = cache.hits - export_hits
                payload = {
                    "key": key,
                    "status": record.status,
                    "error": None,
                    "record": record.to_dict(),
                    "target": {"audio_sha256": audio_hash(target)},
                    "artifacts": exports,
                    "operational": operational,
                }
                # Exclude candidate evaluations and export lookups from target setup.
                before_renders += record.dsp_renders + operational["export_dsp_renders"]
                before_hits += record.cache_hits + operational["export_cache_hits"]
            operational["target_dsp_renders"] = cache.renders - before_renders
            operational["target_cache_hits"] = cache.hits - before_hits
            # Do not commit results produced while evaluator or actor files change.
            exact(
                _identity(episodes, actors, cache_max_audio_bytes),
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


def _manifest(path: Path) -> tuple[dict, _Contract]:
    read_configuration(path)
    manifest = read_json_document(path / "manifest.json")
    try:
        return manifest, _Contract(path, manifest)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError("sound bundle manifest contains malformed fields") from exc


def iter_records(path: Path, *, allow_incomplete: bool = False) -> Iterator[dict]:
    """Yield validated committed envelopes; terminal errors are retained as members."""
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
            episode, actor = contract.member(index)
            group_key = (actor["name"], episode.voice_count, episode.editable_fields)
            group_id = repr(group_key)
            if group_key not in groups:
                groups[group_key] = {
                    "actor": actor["name"],
                    "voice_count": episode.voice_count,
                    "editable_fields": list(episode.editable_fields),
                    "episodes": 0,
                    "submitted": 0,
                    "truncated": 0,
                    "error": 0,
                    "evaluations": 0,
                    "decisions": 0,
                    "invalid_actions": 0,
                    "no_candidate": 0,
                    "model_calls": 0,
                    "dsp_renders": 0,
                    "cache_hits": 0,
                    "target_dsp_renders": 0,
                    "target_cache_hits": 0,
                    "export_dsp_renders": 0,
                    "export_cache_hits": 0,
                }
            group = groups[group_key]
            group["episodes"] += 1
            group[payload["status"]] += 1
            for name, value in payload["operational"].items():
                group[name] += value
            if payload["record"] is None:
                group["no_candidate"] += 1
                continue
            record = EpisodeRecord.from_dict(payload["record"])
            metrics = record.metrics
            for name in ("evaluations", "decisions", "invalid_actions"):
                group[name] += metrics[name]
            group["dsp_renders"] += record.dsp_renders
            group["cache_hits"] += record.cache_hits
            group["no_candidate"] += record.last_candidate is None
            if metrics["model_calls"] is None:
                group["model_calls"] = None
            elif group["model_calls"] is not None:
                group["model_calls"] += metrics["model_calls"]
            for name in (
                "initial_distance",
                "submitted_distance",
                "last_distance",
                "best_distance",
            ):
                store.add(group_id, name, metrics[name])
            for budget in CURVE_BUDGETS:
                store.add(group_id, f"best@{budget}", metrics["best_by_evaluation"][str(budget)])
            for name, candidate in (
                ("initial", record.initial),
                ("submitted", record.submitted_candidate),
                ("last", record.last_candidate),
            ):
                if candidate is not None:
                    store.add(
                        group_id,
                        f"{name}_pitch_error_cents",
                        sum(candidate.pitch_error_cents) / len(candidate.pitch_error_cents),
                    )
                    store.add(
                        group_id, f"{name}_envelope_rmse", candidate.objective["envelope_rmse"]
                    )
                    store.add(
                        group_id, f"{name}_level_error_db", candidate.objective["level_error_db"]
                    )
        if not allow_incomplete and not state.is_complete:
            raise ValueError("bundle is incomplete")
        for key, group in groups.items():
            group_id = repr(key)
            for name in (
                "initial_distance",
                "submitted_distance",
                "last_distance",
                "best_distance",
            ):
                group[name] = store.distribution(group_id, name)
            group["best_by_evaluation"] = {
                str(budget): store.distribution(group_id, f"best@{budget}")
                for budget in CURVE_BUDGETS
            }
            group["diagnostics"] = {
                f"{stage}_{metric}": store.distribution(group_id, f"{stage}_{metric}")
                for stage in ("initial", "submitted", "last")
                for metric in ("pitch_error_cents", "envelope_rmse", "level_error_db")
            }
        return report(
            path,
            manifest,
            state,
            {"groups": list(groups.values()), "calibration_status": CALIBRATION_STATUS},
        )


def summarize_bundle(path: Path, format: str = "text", *, allow_incomplete: bool = False) -> str:
    result = read_bundle(path, allow_incomplete=allow_incomplete)
    if format not in {"text", "markdown", "csv"}:
        raise ValueError("format must be text, markdown, or csv")
    columns = [
        "actor",
        "voices",
        "fields",
        "episodes",
        "submitted",
        "truncated",
        "error",
        "evaluations",
        "decisions",
        "model_calls",
        "initial_distance_mean",
        "submitted_distance_mean",
        "best_distance_mean",
        "bundle_complete",
        "completed",
        "expected",
        "interrupted_attempts",
    ]
    rows = []
    for group in result.metrics["groups"]:
        rows.append(
            [
                group["actor"],
                group["voice_count"],
                ",".join(group["editable_fields"]),
                *[
                    group[name]
                    for name in (
                        "episodes",
                        "submitted",
                        "truncated",
                        "error",
                        "evaluations",
                        "decisions",
                        "model_calls",
                    )
                ],
                *[
                    group[name]["mean"]
                    for name in ("initial_distance", "submitted_distance", "best_distance")
                ],
                result.complete,
                result.completed,
                result.expected,
                result.interrupted_attempts,
            ]
        )
    if format == "csv":
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
        return stream.getvalue()
    heading = (
        f"Sound matching: {'COMPLETE' if result.complete else 'INCOMPLETE'}; "
        f"committed {result.completed}/{result.expected}, missing={result.missing}; "
        "continuous distance frozen; perceptual acceptance unqualified."
    )
    if result.interrupted_attempts:
        heading += (
            f" Interrupted attempts={result.interrupted_attempts}; "
            "their elapsed/render/model costs are unknown."
        )
    if format == "markdown":

        def display(value):
            return ("unknown" if value is None else str(value)).replace("|", "\\|")

        lines = [
            heading,
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        lines += ["| " + " | ".join(display(value) for value in row) + " |" for row in rows]
    else:
        lines = [heading]
        for row in rows:
            lines.append(
                f"{row[0]}: voices={row[1]}, fields={row[2]}; submitted={row[4]}/{row[3]}, "
                f"truncated={row[5]}, error={row[6]}, evaluations={row[7]}, decisions={row[8]}, "
                f"model_calls={row[9] if row[9] is not None else 'unknown'}; "
                f"initial={row[10]}, submitted={row[11]}, best={row[12]}"
            )
    return "\n".join(lines) + "\n"


def verify_replay(path: Path, *, allow_incomplete: bool = False) -> None:
    """Explicitly rerender committed scientific evidence, streaming one member."""
    from harpy.tasks.sound_match.objective import score
    from harpy.tasks.sound_match.renderer import render

    path = Path(path)
    _, contract = _manifest(path)
    for index, payload in enumerate(iter_records(path, allow_incomplete=allow_incomplete)):
        if payload["record"] is None:
            continue
        episode, _ = contract.member(index)
        record = EpisodeRecord.from_dict(payload["record"])
        target = render(episode.target_patch)
        if audio_hash(target) != record.target_audio_sha256:
            raise ValueError("replayed target differs from committed evidence")
        for candidate in record.candidates:
            audio = render(candidate.patch)
            if audio_hash(audio) != candidate.audio_sha256:
                raise ValueError("replayed candidate differs from committed evidence")
            exact(score(target, audio), _plain(candidate.objective), "replayed objective")


__all__ = [
    "BundleReport",
    "evaluate_bundle",
    "iter_records",
    "read_bundle",
    "read_configuration",
    "summarize_bundle",
    "verify_replay",
]
