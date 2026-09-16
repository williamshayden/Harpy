"""Deterministic, untrained audio-only baselines; every proposed render is metered."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from harpy.tasks.sound_match.models import ActorSpec, Observation, Propose, Submit, _integer
from harpy.tasks.sound_match.objective import score
from harpy.tasks.sound_match.patch import PARAMETER_BOUNDS, WAVEFORMS, Patch

BUILTIN_ACTORS = ("fixed", "random-search", "pattern-search")


class FixedActor:
    def decide(self, observation: Observation) -> Submit:
        return Submit(model_calls=0)


class _SearchActor:
    def __init__(self) -> None:
        self.best_patch: Patch | None = None
        self.best_distance = float("inf")

    def _observe(self, observation: Observation) -> Propose | Submit | None:
        distance = score(observation.target_audio, observation.candidate_audio)["distance"]
        if distance < self.best_distance:
            self.best_distance, self.best_patch = distance, observation.patch
        if (
            distance == 0
            or not observation.editable_fields
            or observation.render_budget_remaining == 0
            or observation.decision_budget_remaining == 1
        ):
            return Submit(model_calls=0)
        # Retain a charged evaluation and a final decision for restoring the best
        # candidate; submission always means the actual current waveform.
        if observation.render_budget_remaining == 1 or observation.decision_budget_remaining == 2:
            if observation.patch != self.best_patch:
                return Propose(self.best_patch, model_calls=0)
            return Submit(model_calls=0)
        return None


class RandomSearchActor(_SearchActor):
    def __init__(self, seed: int) -> None:
        super().__init__()
        self.seed = seed
        self.rng = None

    def decide(self, observation: Observation) -> Propose | Submit:
        terminal = self._observe(observation)
        if terminal is not None:
            return terminal
        if self.rng is None:
            # Episode-local streams preserve a result when suite order changes.
            identity = f"{self.seed}\0{observation.episode_id}".encode()
            self.rng = np.random.default_rng(
                int.from_bytes(hashlib.sha256(identity).digest()[:16], "big")
            )
        values = observation.patch.to_dict()
        for name in observation.editable_fields:
            if name == "waveform":
                values[name] = WAVEFORMS[int(self.rng.integers(len(WAVEFORMS)))]
            elif name == "notes_cents":
                low, high = PARAMETER_BOUNDS[name]
                values[name] = sorted(
                    int(value) + low
                    for value in self.rng.choice(
                        high - low + 1, len(observation.patch.notes_cents), replace=False
                    )
                )
            else:
                low, high = PARAMETER_BOUNDS[name]
                values[name] = int(self.rng.integers(low, high + 1))
        return Propose(Patch.from_dict(values), model_calls=0)


class PatternSearchActor(_SearchActor):
    def __init__(self) -> None:
        super().__init__()
        self.queue: deque[Patch] = deque()
        self.base = None
        self.steps = {
            "notes_cents": 100,
            "brightness_hz": 1000,
            "attack_ms": 25,
            "decay_ms": 50,
            "sustain_db": 3,
            "release_ms": 75,
            "gain_db": 3,
        }
        self.seen: set[Patch] = set()

    def _neighbors(self, observation: Observation) -> deque[Patch]:
        candidates = []
        base = self.best_patch
        for name in observation.editable_fields:
            if name == "waveform":
                candidates.extend(
                    replace(base, waveform=value) for value in WAVEFORMS if value != base.waveform
                )
                continue
            low, high = PARAMETER_BOUNDS[name]
            for index in range(len(base.notes_cents) if name == "notes_cents" else 1):
                for sign in (-1, 1):
                    if name == "notes_cents":
                        notes = list(base.notes_cents)
                        notes[index] = max(low, min(high, notes[index] + sign * self.steps[name]))
                        if len(set(notes)) != len(notes):
                            continue
                        candidate = replace(base, notes_cents=tuple(notes))
                    else:
                        value = max(low, min(high, getattr(base, name) + sign * self.steps[name]))
                        candidate = replace(base, **{name: value})
                    candidates.append(candidate)
        return deque(
            candidate
            for candidate in candidates
            if candidate not in self.seen and candidate != base
        )

    def decide(self, observation: Observation) -> Propose | Submit:
        terminal = self._observe(observation)
        if terminal is not None:
            return terminal
        self.seen.add(observation.patch)
        if self.base != self.best_patch:
            self.base = self.best_patch
            self.queue = self._neighbors(observation)
        while not self.queue:
            if all(value == 1 for value in self.steps.values()):
                if observation.patch != self.best_patch:
                    return Propose(self.best_patch, model_calls=0)
                return Submit(model_calls=0)
            self.steps = {name: max(1, value // 2) for name, value in self.steps.items()}
            self.queue = self._neighbors(observation)
        return Propose(self.queue.popleft(), model_calls=0)


def builtin_actor_spec(name: str, seed: int = 0) -> ActorSpec:
    """Return a fresh-controller factory and explicit untrained baseline identity."""
    _integer(seed, "seed")
    if name not in BUILTIN_ACTORS:
        raise ValueError("Unknown sound-matching actor. Choose: " + ", ".join(BUILTIN_ACTORS))
    factory = {
        "fixed": FixedActor,
        "random-search": lambda: RandomSearchActor(seed),
        "pattern-search": PatternSearchActor,
    }[name]
    return ActorSpec(
        name=name,
        factory=factory,
        metadata={
            "kind": "untrained-baseline",
            "algorithm": name,
            "version": 1,
            "seed": seed if name == "random-search" else None,
            "training": "none",
            "model_calls": 0,
        },
        artifact_paths=(Path(__file__),),
    )
