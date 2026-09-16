"""Bounded known-synth audio matching, independent of the frozen v1 benchmark."""

from harpy.tasks.sound_match.actors import BUILTIN_ACTORS, builtin_actor_spec
from harpy.tasks.sound_match.models import ActorSpec, EpisodeSpec, Observation, Propose, Submit
from harpy.tasks.sound_match.patch import Patch, default_patch
from harpy.tasks.sound_match.results import (
    ExperimentResult,
    load_result,
    save_result,
    summarize,
    verify_replay,
)
from harpy.tasks.sound_match.runner import evaluate, run

__all__ = [
    "BUILTIN_ACTORS",
    "ActorSpec",
    "EpisodeSpec",
    "ExperimentResult",
    "Observation",
    "Patch",
    "Propose",
    "Submit",
    "builtin_actor_spec",
    "default_patch",
    "evaluate",
    "load_result",
    "run",
    "save_result",
    "summarize",
    "verify_replay",
]
