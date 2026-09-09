"""Ordinary paired research experiments, with lazy optional model dependencies."""

from harpy.experiments.actors import (
    oracle_actor_spec,
    pitch_actor_spec,
    ppo_actor_spec,
    random_actor_spec,
    reward_search_actor_spec,
    spectrum_peak_actor_spec,
    waveform_fft_actor_spec,
)
from harpy.experiments.models import ActorSpec, Decision, EpisodeSpec
from harpy.experiments.results import ExperimentResult, load_result, save_result, summarize
from harpy.experiments.runner import evaluate, run
from harpy.experiments.training import train

__all__ = [
    "ActorSpec",
    "Decision",
    "EpisodeSpec",
    "ExperimentResult",
    "evaluate",
    "load_result",
    "oracle_actor_spec",
    "pitch_actor_spec",
    "ppo_actor_spec",
    "random_actor_spec",
    "reward_search_actor_spec",
    "run",
    "save_result",
    "spectrum_peak_actor_spec",
    "summarize",
    "train",
    "waveform_fft_actor_spec",
]
