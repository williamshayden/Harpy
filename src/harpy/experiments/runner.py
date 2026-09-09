"""Ordinary paired rollouts with explicit inputs and fresh episode controllers."""

from __future__ import annotations

import hashlib
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import gymnasium
import numpy as np

from harpy.envs.models import EpisodeResult, ObservationMode, PitchAction
from harpy.envs.observations import owned_observation
from harpy.envs.robustness import (
    CLEAN_CONDITION,
    ROBUSTNESS_RENDERER_ID,
    ConditionSpec,
    RobustnessEvidenceCache,
    make_robustness_env,
)
from harpy.experiments.models import ActorSpec, Decision, EpisodeSpec, _plain
from harpy.experiments.results import EpisodeRecord, ExperimentResult, _validate_protocol


def _artifact_files(paths: tuple[Path, ...]) -> list[dict]:
    files = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("artifact paths must not be symbolic links")
        root = path.resolve(strict=True)
        entries = (root,) if root.is_file() else tuple(sorted(root.rglob("*")))
        for entry in entries:
            if entry.is_symlink():
                raise ValueError("artifact inventories must not contain symbolic links")
            if entry.is_file():
                content = entry.read_bytes()
                files[str(entry)] = {
                    "path": str(entry),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
    return [files[name] for name in sorted(files)]


def _source() -> dict:
    from harpy.learning.artifacts import (
        _capture_package_source,
        _source_to_document,
        capture_source_status,
    )

    anchor = Path(__file__).resolve()
    return {
        "identity": _source_to_document(capture_source_status(anchor)),
        "package_sha256": _capture_package_source(anchor).package_sha256,
    }


def _runtime() -> dict:
    def version(name):
        value = getattr(sys.modules.get(name), "__version__", None)
        return None if value is None else str(value)

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": version("torch"),
        "stable_baselines3": version("stable_baselines3"),
    }


def _decision(actor: object, observation: Mapping[str, object]) -> Decision:
    decide = getattr(actor, "decide", None)
    if callable(decide):
        decision = decide(observation)
        if not isinstance(decision, Decision):
            raise ValueError("decide must return an experiments.Decision")
        return decision
    act = getattr(actor, "act", None)
    if not callable(act):
        raise ValueError("actor must provide decide(observation) or act(observation)")
    action = act(observation)
    if not isinstance(action, PitchAction):
        raise ValueError("act must return a PitchAction")
    return Decision(action)


def evaluate(
    episodes: Sequence[EpisodeSpec],
    actors: Sequence[ActorSpec],
    *,
    conditions: Sequence[ConditionSpec] = (CLEAN_CONDITION,),
    trace: bool = False,
    protocol: Mapping[str, object] | None = None,
) -> ExperimentResult:
    """Run each actor on each explicit paired episode and condition.

    Factories receive no evaluator truth and must return a fresh controller.
    Only reward-only actors receive optional ``feedback(reward, info)`` calls;
    feedback contains the public action-applied flag, never hidden error or source.
    Compact actions, estimates, and inference counts are always retained. ``trace``
    additionally retains rewards, public controls, and rendered-view metadata.
    """
    episodes, actors, conditions = tuple(episodes), tuple(actors), tuple(conditions)
    if type(trace) is not bool:
        raise ValueError("trace must be a bool")
    for values, expected, name, key in (
        (episodes, EpisodeSpec, "episodes", "id"),
        (actors, ActorSpec, "actors", "name"),
        (conditions, ConditionSpec, "conditions", "id"),
    ):
        if not values or not all(isinstance(value, expected) for value in values):
            raise ValueError(f"{name} must contain nonempty {expected.__name__} values")
        if len({getattr(value, key) for value in values}) != len(values):
            raise ValueError(f"{name} identities must be unique")
    if protocol is not None:
        _validate_protocol(protocol, episodes)
    started = time.perf_counter()
    source = _source()
    actor_documents = []
    for spec in actors:
        if spec._verify_artifact is not None:
            spec._verify_artifact()
        actor_documents.append(
            {**spec.to_document(), "artifact_files": _artifact_files(spec.artifact_paths)}
        )
    cache = RobustnessEvidenceCache()
    records = []
    # Pair nearby work so the byte-bounded cache reuses evidence across actors.
    for condition in conditions:
        for episode in episodes:
            for spec in actors:
                actor = spec.factory()
                env = make_robustness_env(
                    condition,
                    nuisance_seed=episode.nuisance_seed,
                    observation_mode=spec.observation_mode,
                    cache=cache,
                )
                try:
                    observation, _ = env.reset(
                        seed=episode.nuisance_seed,
                        options={
                            "source_pitch_cents": episode.source_pitch_cents,
                            "target_note_index": episode.target_note_index,
                        },
                    )
                    initial_evidence = env.evidence_metadata
                    actions, counts, estimates, steps = [], [], [], []
                    while True:
                        snapshot = owned_observation(observation, spec.observation_mode)
                        evidence = env.evidence_metadata if trace else None
                        controls = [int(value) for value in snapshot["controls"]]
                        decision = _decision(actor, snapshot)
                        observation, reward, terminated, truncated, info = env.step(decision.action)
                        actions.append(decision.action)
                        counts.append(decision.inference_count)
                        if decision.estimated_candidate_cents is not None:
                            estimates.append(
                                {
                                    "step": len(actions),
                                    "estimated_candidate_cents": decision.estimated_candidate_cents,
                                }
                            )
                        if trace:
                            steps.append(
                                {
                                    "step": len(actions),
                                    "action": int(decision.action),
                                    "controls": controls,
                                    "reward": float(reward),
                                    "inference_count": decision.inference_count,
                                    "estimated_candidate_cents": decision.estimated_candidate_cents,
                                    "evidence": evidence,
                                }
                            )
                        if terminated or truncated:
                            terminal = env.unwrapped.episode_result
                            if not isinstance(terminal, EpisodeResult) or terminal.actions != tuple(
                                actions
                            ):
                                raise RuntimeError(
                                    "environment terminal result must match the action trace"
                                )
                            break
                        feedback = getattr(actor, "feedback", None)
                        if spec.observation_mode is ObservationMode.REWARD_ONLY and callable(
                            feedback
                        ):
                            feedback(
                                float(reward), {"action_applied": bool(info["action_applied"])}
                            )
                    records.append(
                        EpisodeRecord(
                            spec.name,
                            condition.id,
                            episode.id,
                            terminal,
                            tuple(counts),
                            tuple(estimates),
                            initial_evidence,
                            tuple(steps) if trace else None,
                        )
                    )
                finally:
                    env.close()
    for spec, document in zip(actors, actor_documents, strict=True):
        if spec._verify_artifact is not None:
            spec._verify_artifact()
        if _artifact_files(spec.artifact_paths) != document["artifact_files"]:
            raise ValueError("actor artifact files changed during evaluation")
    if _source() != source:
        raise ValueError("evaluator source changed during evaluation")
    return ExperimentResult(
        episodes,
        tuple(actor_documents),
        tuple(condition.to_document() for condition in conditions),
        tuple(records),
        {
            "source": source,
            "runtime": {
                **_runtime(),
                "elapsed_wall_time_seconds": time.perf_counter() - started,
                "evaluation_devices": {spec.name: spec.device for spec in actors},
                "timing_scope": "whole evaluation including rendering and shared-cache work",
            },
            "renderer": ROBUSTNESS_RENDERER_ID,
        },
        None if protocol is None else _plain(protocol),
    )


def run(
    episode: EpisodeSpec,
    actor: ActorSpec,
    *,
    condition: ConditionSpec = CLEAN_CONDITION,
) -> ExperimentResult:
    """Run one episode with its complete decision trace through the shared runner."""
    return evaluate((episode,), (actor,), conditions=(condition,), trace=True)


__all__ = ["evaluate", "run"]
