"""Learning-only cached evidence and deterministic episode scheduling environments."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium
import numpy as np

from harpy.envs.models import ObservationMode
from harpy.envs.sine_pitch import Observation, SinePitchEnv, _CandidateEvidence
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.models import EpisodeSpec
from harpy.learning.suites import ppo_training_episode


class SpectrumEvidenceProvider:
    """Populate caller-owned spectrum evidence through the direct renderer seam."""

    def __init__(self, cache: SpectrumEvidenceCache) -> None:
        self._cache = cache
        self._renderer = SinePitchEnv()

    def spectrum_for_cents(self, effective_pitch_cents: int) -> np.ndarray:
        """Return an owned immutable cached spectrum for an effective pitch."""
        return self._cache.get_or_create(
            effective_pitch_cents,
            on_miss=lambda: self._renderer._candidate_evidence(effective_pitch_cents).spectrum,
        )


class CachedSinePitchEnv(SinePitchEnv):
    """Non-registered direct environment variant that retains spectra but never audio."""

    def __init__(
        self,
        evidence_provider: SpectrumEvidenceProvider,
        observation_mode: ObservationMode = ObservationMode.SPECTRUM,
    ) -> None:
        super().__init__(observation_mode=observation_mode)
        self._evidence_provider = evidence_provider

    def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
        return _CandidateEvidence(
            candidate_audio=None,
            spectrum=self._evidence_provider.spectrum_for_cents(candidate_cents),
        )


class ScheduledEpisodeEnv(gymnasium.Wrapper):
    """Inject an indexed deterministic episode through the real environment reset path."""

    def __init__(self, env: gymnasium.Env, episode_at: Callable[[int], EpisodeSpec]) -> None:
        super().__init__(env)
        self._episode_at = episode_at
        self._next_episode_index = 0

    @property
    def next_episode_index(self) -> int:
        """Return the absolute index scheduled for the next successful reset."""
        return self._next_episode_index

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[Observation, dict[str, object]]:
        """Reset the wrapped environment using the next internal episode only."""
        if options not in (None, {}):
            raise ValueError("external options are not permitted for scheduled episodes")
        episode = self._episode_at(self._next_episode_index)
        observation, info = self.env.reset(seed=seed, options=episode.reset_options())
        self._next_episode_index += 1
        return observation, info


def make_cached_sine_pitch_env(
    cache: SpectrumEvidenceCache,
    *,
    observation_mode: ObservationMode = ObservationMode.SPECTRUM,
) -> CachedSinePitchEnv:
    """Build one unregistered cached environment backed by caller-owned evidence."""
    return CachedSinePitchEnv(
        SpectrumEvidenceProvider(cache),
        observation_mode=observation_mode,
    )


def make_ppo_training_env(
    *,
    run_seed: int,
    cache: SpectrumEvidenceCache,
) -> ScheduledEpisodeEnv:
    """Build the unregistered cache-backed PPO training environment sequence."""
    return ScheduledEpisodeEnv(
        make_cached_sine_pitch_env(cache),
        episode_at=lambda episode_index: ppo_training_episode(
            run_seed=run_seed,
            episode_index=episode_index,
        ),
    )


__all__ = [
    "CachedSinePitchEnv",
    "ScheduledEpisodeEnv",
    "SpectrumEvidenceProvider",
    "make_cached_sine_pitch_env",
    "make_ppo_training_env",
]
