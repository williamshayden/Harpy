"""Bounded known-synth audio matching, independent of the frozen v1 benchmark."""

from harpy.tasks.sound_match.actors import BUILTIN_ACTORS, builtin_actor_spec
from harpy.tasks.sound_match.durable import (
    evaluate_bundle,
    iter_records,
    read_bundle,
    summarize_bundle,
)
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
    "evaluate_bundle",
    "iter_records",
    "load_result",
    "read_bundle",
    "run",
    "save_result",
    "summarize",
    "summarize_bundle",
    "verify_replay",
]
