"""Contract tests for learning-only cached and scheduled environments."""

from __future__ import annotations

import numpy as np
import pytest

from harpy.envs import PitchAction, SinePitchEnv
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.envs import (
    CachedSinePitchEnv,
    ScheduledEpisodeEnv,
    SpectrumEvidenceProvider,
    make_cached_sine_pitch_env,
    make_ppo_training_env,
)
from harpy.learning.models import EpisodeSpec
from harpy.learning.suites import ppo_training_episode


def _assert_observations_identical(
    direct: dict[str, np.ndarray | np.int64], cached: dict[str, np.ndarray | np.int64]
) -> None:
    assert direct.keys() == cached.keys()
    for key in direct:
        np.testing.assert_array_equal(direct[key], cached[key])


def _assert_equal_transition(
    direct: tuple[dict[str, np.ndarray | np.int64], float, bool, bool, dict[str, object]],
    cached: tuple[dict[str, np.ndarray | np.int64], float, bool, bool, dict[str, object]],
) -> None:
    direct_observation, direct_reward, direct_terminated, direct_truncated, direct_info = direct
    cached_observation, cached_reward, cached_terminated, cached_truncated, cached_info = cached
    _assert_observations_identical(direct_observation, cached_observation)
    assert direct_reward == cached_reward
    assert (direct_terminated, direct_truncated) == (cached_terminated, cached_truncated)
    assert direct_info == cached_info


@pytest.mark.parametrize("candidate_cents", [1_100, 4_800, 6_000, 10_900])
def test_cached_miss_and_hit_are_bit_identical_to_direct(candidate_cents: int) -> None:
    """Cached evidence must reproduce an independently rendered direct spectrum byte for byte."""
    options = {"target_note_index": 12, "source_pitch_cents": 6_000}
    direct = SinePitchEnv()
    cached = CachedSinePitchEnv(SpectrumEvidenceProvider(SpectrumEvidenceCache()))
    direct.reset(options=options)
    cached.reset(options=options)

    direct_evidence = direct._candidate_evidence(candidate_cents).spectrum
    miss = cached._candidate_evidence(candidate_cents).spectrum
    hit = cached._candidate_evidence(candidate_cents).spectrum

    assert np.array_equal(direct_evidence.view(np.uint32), miss.view(np.uint32))
    assert np.array_equal(miss.view(np.uint32), hit.view(np.uint32))


def test_cached_trajectory_matches_direct_through_inverse_actions_and_terminal_result() -> None:
    """Caching cannot alter observations, transitions, terminal flags, or evaluator truth."""
    options = {"target_note_index": 8, "source_pitch_cents": 5_432}
    direct = SinePitchEnv()
    cached = make_cached_sine_pitch_env(SpectrumEvidenceCache())
    direct_reset, direct_info = direct.reset(seed=37, options=options)
    cached_reset, cached_info = cached.reset(seed=37, options=options)
    _assert_observations_identical(direct_reset, cached_reset)
    assert direct_info == cached_info
    assert direct.np_random.bit_generator.state == cached.np_random.bit_generator.state

    for action in (
        PitchAction.OCTAVE_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_DOWN,
        PitchAction.SEMITONE_DOWN,
        PitchAction.OCTAVE_DOWN,
        PitchAction.SUBMIT,
    ):
        _assert_equal_transition(direct.step(action), cached.step(action))
        assert direct.np_random.bit_generator.state == cached.np_random.bit_generator.state

    assert direct.episode_result == cached.episode_result


def test_cached_action_orders_reaching_the_same_controls_match_direct_bytes() -> None:
    """Evidence is a function of the final controls, not the order used to reach them."""
    options = {"target_note_index": 8, "source_pitch_cents": 5_432}
    direct_first = SinePitchEnv()
    cached_first = make_cached_sine_pitch_env(SpectrumEvidenceCache())
    direct_second = SinePitchEnv()
    cached_second = make_cached_sine_pitch_env(SpectrumEvidenceCache())
    for direct, cached in ((direct_first, cached_first), (direct_second, cached_second)):
        direct_reset, direct_info = direct.reset(options=options)
        cached_reset, cached_info = cached.reset(options=options)
        _assert_observations_identical(direct_reset, cached_reset)
        assert direct_info == cached_info

    for action in (PitchAction.OCTAVE_UP, PitchAction.SEMITONE_DOWN, PitchAction.CENT_UP):
        _assert_equal_transition(direct_first.step(action), cached_first.step(action))
    for action in (PitchAction.CENT_UP, PitchAction.OCTAVE_UP, PitchAction.SEMITONE_DOWN):
        _assert_equal_transition(direct_second.step(action), cached_second.step(action))

    assert np.array_equal(
        direct_first._spectrum.view(np.uint32), direct_second._spectrum.view(np.uint32)
    )
    assert np.array_equal(
        cached_first._spectrum.view(np.uint32), cached_second._spectrum.view(np.uint32)
    )
    assert np.array_equal(
        direct_first._spectrum.view(np.uint32), cached_first._spectrum.view(np.uint32)
    )


def test_cached_subclass_omits_audio_and_isolates_cache_env_and_observation_storage() -> None:
    """The learning-only variant retains no audio and exposes no shared spectrum memory."""
    cache = SpectrumEvidenceCache()
    direct = SinePitchEnv()
    cached = CachedSinePitchEnv(SpectrumEvidenceProvider(cache))
    options = {"target_note_index": 12, "source_pitch_cents": 6_000}
    direct.reset(options=options)
    observation, _ = cached.reset(options=options)

    stored = cache._entries[6_000].spectrum
    assert direct._candidate_audio is not None
    assert cached._candidate_audio is None
    assert cached._spectrum is not None
    assert not np.shares_memory(stored, cached._spectrum)
    assert not np.shares_memory(stored, observation["spectrum"])
    assert not np.shares_memory(cached._spectrum, observation["spectrum"])


def test_cached_factory_shares_evidence_but_never_environment_state() -> None:
    """Independent factory products reuse evidence without coupling state or lifecycle."""
    cache = SpectrumEvidenceCache()

    def factory():
        return make_cached_sine_pitch_env(cache)

    first = factory()
    second = factory()
    options = {"target_note_index": 12, "source_pitch_cents": 6_000}

    first_observation, _ = first.reset(options=options)
    first_spectrum = first._spectrum
    second_observation, _ = second.reset(options=options)

    assert len(cache) == 1
    assert first is not second
    assert first._controls == second._controls
    assert first_spectrum is not None and second._spectrum is not None
    assert not np.shares_memory(first_spectrum, second._spectrum)
    assert not np.shares_memory(first_observation["spectrum"], second_observation["spectrum"])

    first.step(PitchAction.CENT_UP)
    assert first._controls != second._controls
    first.close()
    second_again, _ = second.reset(options=options)
    _assert_observations_identical(second_observation, second_again)


def test_scheduled_env_uses_episode_indices_without_exposing_schedule_metadata() -> None:
    """The wrapper injects only the scheduled reset values in exactly indexed order."""
    episodes = (
        EpisodeSpec(target_note_index=2, source_pitch_cents=5_111),
        EpisodeSpec(target_note_index=18, source_pitch_cents=6_789),
    )
    base = SinePitchEnv()
    scheduled = ScheduledEpisodeEnv(base, episode_at=episodes.__getitem__)

    first_observation, first_info = scheduled.reset(seed=37, options={})
    second_observation, second_info = scheduled.reset()

    assert int(first_observation["target_note"]) == 2
    assert int(second_observation["target_note"]) == 18
    assert base._source_pitch_cents == 6_789
    assert base.np_random_seed == 37
    assert first_info == second_info == {"step_count": 0, "steps_remaining": 64}
    assert set(first_observation) == {"target_note", "controls", "steps_remaining", "spectrum"}
    assert set(first_info) == {"step_count", "steps_remaining"}
    assert scheduled.next_episode_index == 2


def test_scheduled_env_rejects_external_overrides_and_rolls_back_its_index_on_reset_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only successful internal resets consume an episode index."""
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=6_000)
    base = SinePitchEnv()
    scheduled = ScheduledEpisodeEnv(base, episode_at=lambda _index: episode)

    with pytest.raises(ValueError, match="external options"):
        scheduled.reset(options={"target_note_index": 0})
    assert scheduled.next_episode_index == 0

    monkeypatch.setattr(
        base,
        "_candidate_evidence",
        lambda _candidate_cents: (_ for _ in ()).throw(RuntimeError("render failed")),
    )
    with pytest.raises(RuntimeError, match="render failed"):
        scheduled.reset()
    assert scheduled.next_episode_index == 0


def test_ppo_training_factory_uses_the_canonical_indexed_episode_sequence() -> None:
    """The PPO factory is an unregistered cached wrapper over the Task 1 sequence."""
    cache = SpectrumEvidenceCache()
    env = make_ppo_training_env(run_seed=7, cache=cache)
    expected = ppo_training_episode(run_seed=7, episode_index=0)

    observation, info = env.reset()

    assert isinstance(env, ScheduledEpisodeEnv)
    assert int(observation["target_note"]) == expected.target_note_index
    assert env.unwrapped._source_pitch_cents == expected.source_pitch_cents
    assert info == {"step_count": 0, "steps_remaining": 64}
    assert env.next_episode_index == 1


@pytest.mark.integration
def test_real_renderer_populates_every_reachable_cache_coordinate() -> None:
    """The direct renderer produces valid bounded evidence for the complete v0 range."""
    cache = SpectrumEvidenceCache()
    provider = SpectrumEvidenceProvider(cache)

    for candidate_cents in range(1_100, 10_901):
        spectrum = provider.spectrum_for_cents(candidate_cents)
        assert spectrum.shape == (1_961,)
        assert spectrum.dtype == np.float32
        assert np.isfinite(spectrum).all()

    assert len(cache) == 9_801
    assert cache.spectrum_bytes == 76_879_044
