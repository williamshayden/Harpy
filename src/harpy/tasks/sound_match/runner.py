"""Bounded sound-match episodes with owned observations and complete cost ledgers."""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

import harpy
from harpy._serialization import canonical_json_bytes, sha256_file
from harpy.tasks.sound_match.models import (
    INFORMATION_TRACK,
    ActorSpec,
    EpisodeSpec,
    Observation,
    Propose,
    Submit,
    _freeze,
    _integer,
    _patch,
    _plain,
    proposal_error,
)
from harpy.tasks.sound_match.objective import score
from harpy.tasks.sound_match.patch import PARAMETER_BOUNDS, WAVEFORMS, Patch
from harpy.tasks.sound_match.renderer import FRAME_COUNT, RENDERER_ID, render
from harpy.tasks.sound_match.results import (
    CandidateEvidence,
    EpisodeRecord,
    ExperimentResult,
    audio_hash,
)

DEFAULT_CACHE_BYTES = 16 * 1024 * 1024


class _WaveCache:
    """Shared bounded waveform cache; logical evaluations are charged by callers."""

    def __init__(self, max_audio_bytes: int) -> None:
        self.max_audio_bytes = max_audio_bytes
        self.bytes = 0
        self.items: OrderedDict[tuple[str, Patch], np.ndarray] = OrderedDict()
        self.renders = 0
        self.hits = 0

    def get(self, patch: Patch) -> np.ndarray:
        key = (RENDERER_ID, patch)
        if key in self.items:
            self.hits += 1
            self.items.move_to_end(key)
            return self.items[key]
        # Failed DSP attempts still consume the candidate evaluation.
        self.renders += 1
        raw = render(patch)
        if (
            not isinstance(raw, np.ndarray)
            or raw.dtype != np.float32
            or raw.shape != (FRAME_COUNT,)
            or not np.isfinite(raw).all()
        ):
            raise ValueError("renderer returned an invalid fixed-duration waveform")
        audio = np.array(raw, dtype=np.float32, copy=True)
        audio.flags.writeable = False
        if audio.nbytes <= self.max_audio_bytes:
            while self.items and self.bytes + audio.nbytes > self.max_audio_bytes:
                _, removed = self.items.popitem(last=False)
                self.bytes -= removed.nbytes
            self.items[key] = audio
            self.bytes += audio.nbytes
        return audio


def _inventory_file(path: Path, name: str) -> dict:
    if not path.is_file():
        raise ValueError("source and actor artifacts must be files")
    size = path.stat().st_size
    digest = sha256_file(path)
    if path.stat().st_size != size:
        raise RuntimeError("Source or actor artifact changed while hashing")
    return {"path": name, "sha256": digest, "size_bytes": size}


def _source_identity() -> dict:
    package = Path(harpy.__file__).parent
    files = [
        _inventory_file(path, path.relative_to(package).as_posix())
        for path in sorted(package.rglob("*.py"))
    ]
    return {
        "kind": "package-python-files",
        "package_version": harpy.__version__,
        "files": files,
        "sha256": hashlib.sha256(canonical_json_bytes({"files": files})).hexdigest(),
    }


def _actor_identity(actor: ActorSpec) -> dict:
    files = [
        _inventory_file(path, f"artifact-{index:04d}/{path.name}")
        for index, path in enumerate(actor.artifact_paths)
    ]
    return {
        "name": actor.name,
        "information_track": INFORMATION_TRACK,
        "metadata": _plain(actor.metadata),
        "artifact_files": files,
    }


def _evidence(
    episode: EpisodeSpec,
    evaluation: int,
    patch: Patch,
    target: np.ndarray,
    audio: np.ndarray,
) -> CandidateEvidence:
    return CandidateEvidence(
        evaluation=evaluation,
        patch=patch,
        audio_sha256=audio_hash(audio),
        objective=score(target, audio),
        pitch_error_cents=tuple(
            abs(actual - wanted)
            for actual, wanted in zip(
                patch.notes_cents, episode.target_patch.notes_cents, strict=True
            )
        ),
    )


def _decision(index: int, action: str, model_calls: int | None = None, **values) -> dict:
    return {
        "index": index,
        "action": action,
        "proposal": None,
        "status": None,
        "reason": None,
        "error_type": None,
        "evaluation": None,
        "candidate": None,
        "model_calls": model_calls,
        **values,
    }


def _episode(
    episode: EpisodeSpec,
    actor_spec: ActorSpec,
    target: np.ndarray,
    cache: _WaveCache,
) -> EpisodeRecord:
    started = time.perf_counter()
    renders_before, hits_before = cache.renders, cache.hits
    target_hash = audio_hash(target)
    initial = None
    ledger = []
    status, reason, error_type = "truncated", "decision_budget_exhausted", None
    try:
        current_audio = cache.get(episode.initial_patch)
    except Exception as exc:
        status, reason, error_type = "error", "initial_render_error", type(exc).__name__
    else:
        try:
            initial = _evidence(episode, 1, episode.initial_patch, target, current_audio)
        except Exception as exc:
            status, reason, error_type = "error", "initial_objective_error", type(exc).__name__

    if initial is not None:
        actor = None
        cancelled = False
        try:
            try:
                actor = actor_spec.factory()
            except Exception as exc:
                status, reason, error_type = "error", "actor_factory_error", type(exc).__name__
            else:
                current = initial
                evaluations = 1
                last_status = {"status": "initial", "reason": None}
                allowed = _freeze(
                    {
                        "bounds": dict(PARAMETER_BOUNDS),
                        "waveforms": WAVEFORMS,
                        "voice_count": episode.voice_count,
                    }
                )
                for index in range(1, episode.decision_budget + 1):
                    observation = Observation(
                        episode_id=episode.episode_id,
                        target_audio=target.copy(),
                        candidate_audio=current_audio.copy(),
                        patch=_patch(current.patch),
                        editable_fields=episode.editable_fields,
                        render_budget_remaining=episode.render_budget - evaluations,
                        decision_budget_remaining=episode.decision_budget - index + 1,
                        last_action_status=_freeze(last_status),
                        allowed_values=allowed,
                    )
                    try:
                        action = actor.decide(observation)
                    except Exception as exc:
                        status, reason, error_type = "error", "actor_exception", type(exc).__name__
                        ledger.append(
                            _decision(
                                index, "error", status="error", reason=reason, error_type=error_type
                            )
                        )
                        break
                    if isinstance(action, Submit):
                        ledger.append(
                            _decision(index, "submit", action.model_calls, status="submitted")
                        )
                        status, reason = "submitted", "explicit_submit"
                        break
                    if not isinstance(action, Propose):
                        row = _decision(
                            index, "invalid", status="invalid", reason="unsupported_action"
                        )
                    else:
                        row = _decision(index, "propose", action.model_calls)
                        try:
                            proposal = _patch(action.patch)
                        except Exception:
                            row.update(status="invalid", reason="invalid_patch")
                        else:
                            row["proposal"] = proposal.to_dict()
                            rejection = proposal_error(proposal, episode, evaluations)
                            if rejection is not None:
                                row.update(status="invalid", reason=rejection)
                            else:
                                evaluations += 1
                                row["evaluation"] = evaluations
                                try:
                                    proposed_audio = cache.get(proposal)
                                except Exception as exc:
                                    status, reason, error_type = (
                                        "error",
                                        "render_error",
                                        type(exc).__name__,
                                    )
                                    row.update(status="error", reason=reason, error_type=error_type)
                                else:
                                    try:
                                        candidate = _evidence(
                                            episode, evaluations, proposal, target, proposed_audio
                                        )
                                    except Exception as exc:
                                        status, reason, error_type = (
                                            "error",
                                            "objective_error",
                                            type(exc).__name__,
                                        )
                                        row.update(
                                            status="error", reason=reason, error_type=error_type
                                        )
                                    else:
                                        current, current_audio = candidate, proposed_audio
                                        row.update(status="accepted", candidate=current.to_dict())
                    ledger.append(row)
                    last_status = {"status": row["status"], "reason": row["reason"]}
                    if row["status"] == "error":
                        break
        except BaseException:
            cancelled = True
            raise
        finally:
            if actor is not None:
                try:
                    close = getattr(actor, "close", None)
                    if close is not None:
                        close()
                except Exception as exc:
                    # Preserve cancellation; an ordinary cleanup failure prevents a
                    # successful episode being counted as submitted.
                    if not cancelled:
                        status, reason, error_type = (
                            "error",
                            "actor_cleanup_error",
                            type(exc).__name__,
                        )

    return EpisodeRecord(
        episode_id=episode.episode_id,
        actor_name=actor_spec.name,
        target_audio_sha256=target_hash,
        initial=initial,
        ledger=tuple(ledger),
        status=status,
        reason=reason,
        error_type=error_type,
        dsp_renders=cache.renders - renders_before,
        cache_hits=cache.hits - hits_before,
        elapsed_seconds=time.perf_counter() - started,
    )


def evaluate(
    episodes: Sequence[EpisodeSpec],
    actors: Sequence[ActorSpec],
    *,
    cache_max_audio_bytes: int = DEFAULT_CACHE_BYTES,
) -> ExperimentResult:
    """Run ordered episode/actor membership with fresh factories and bounded cache.

    Target setup failures abort evaluation. Actor/candidate failures become explicit
    error records. Cooperative in-process actors receive no evaluator target patch;
    this API does not claim process isolation against malicious Python extensions.
    """
    episodes, actors = tuple(episodes), tuple(actors)
    if not episodes or any(not isinstance(item, EpisodeSpec) for item in episodes):
        raise ValueError("episodes must contain EpisodeSpec values")
    if not actors or any(not isinstance(item, ActorSpec) for item in actors):
        raise ValueError("actors must contain ActorSpec values")
    if len({item.episode_id for item in episodes}) != len(episodes):
        raise ValueError("episode identities must be unique")
    if len({item.name for item in actors}) != len(actors):
        raise ValueError("actor identities must be unique")
    _integer(cache_max_audio_bytes, "cache_max_audio_bytes")
    started = time.perf_counter()
    source = _source_identity()
    actor_documents = tuple(_actor_identity(actor) for actor in actors)
    cache = _WaveCache(cache_max_audio_bytes)
    records, targets = [], []
    for episode in episodes:
        setup_started = time.perf_counter()
        renders_before, hits_before = cache.renders, cache.hits
        try:
            target = cache.get(episode.target_patch)
        except Exception as exc:
            raise RuntimeError(
                f"Target setup failed for {episode.episode_id}: {type(exc).__name__}"
            ) from exc
        targets.append(
            {
                "episode_id": episode.episode_id,
                "audio_sha256": audio_hash(target),
                "dsp_renders": cache.renders - renders_before,
                "cache_hits": cache.hits - hits_before,
                "elapsed_seconds": time.perf_counter() - setup_started,
            }
        )
        for actor in actors:
            records.append(_episode(episode, actor, target, cache))
    if source != _source_identity():
        raise RuntimeError("Package source changed during evaluation")
    if actor_documents != tuple(_actor_identity(actor) for actor in actors):
        raise RuntimeError("Actor artifacts changed during evaluation")
    return ExperimentResult(
        episodes=episodes,
        actors=actor_documents,
        records=tuple(records),
        targets=tuple(targets),
        source=source,
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        cache={"policy": "lru-waveforms-v1", "max_audio_bytes": cache_max_audio_bytes},
        elapsed_seconds=time.perf_counter() - started,
    )


def run(
    episode: EpisodeSpec,
    actor: ActorSpec,
    *,
    cache_max_audio_bytes: int = DEFAULT_CACHE_BYTES,
) -> ExperimentResult:
    """Run one actor on one episode using the same evidence contract as evaluate."""
    return evaluate((episode,), (actor,), cache_max_audio_bytes=cache_max_audio_bytes)
