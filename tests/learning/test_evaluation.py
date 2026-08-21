"""Matched learned and baseline terminal-evaluation contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace
from typing import Any

import gymnasium
import numpy as np
import pytest

import harpy.learning.evaluation as evaluation
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import EpisodeResult, ObservationMode, PitchAction, TerminalReason
from harpy.envs.sine_pitch import SinePitchEnv, _CandidateEvidence
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.evaluation import (
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    TerminalEpisodeRecord,
    aggregate_episode_records,
    build_evaluation_rows,
    evaluate_baseline_suite,
    evaluate_learned_actor,
    make_spectrum_probe_factory,
)
from harpy.learning.models import EpisodeSpec, EpisodeSuite, EvaluationSuiteId
from harpy.learning.suites import suite_digest


class FastSinePitchEnv(SinePitchEnv):
    """Real environment state machine with cheap, distinctive spectrum evidence."""

    def __init__(self, observation_mode: ObservationMode = ObservationMode.SPECTRUM) -> None:
        super().__init__(observation_mode=observation_mode)
        self.closed = False
        self.result_reads = 0
        self.reset_pairs: list[tuple[int, int]] = []

    @property
    def episode_result(self) -> EpisodeResult:
        self.result_reads += 1
        if self._episode_result is None:
            raise AssertionError("episode_result was read before done")
        return self._episode_result

    def reset(self, **kwargs: Any):
        observation, info = super().reset(**kwargs)
        assert self._target_note_index is not None
        assert self._source_pitch_cents is not None
        self.reset_pairs.append((self._target_note_index, self._source_pitch_cents))
        return observation, info

    def close(self) -> None:
        self.closed = True

    def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
        spectrum = np.linspace(0.0, 1.0, 1_961, dtype=np.float32)
        spectrum = np.roll(spectrum, candidate_cents % spectrum.size)
        spectrum.setflags(write=False)
        return _CandidateEvidence(candidate_audio=None, spectrum=spectrum)


class ObservationOnlySpy:
    def __init__(self, action: PitchAction = PitchAction.SUBMIT) -> None:
        self.action = action
        self.keys: list[tuple[str, ...]] = []

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        self.keys.append(tuple(sorted(observation)))
        return self.action


def _suite(*episodes: EpisodeSpec) -> EpisodeSuite:
    suite_id = EvaluationSuiteId.SMOKE
    episode_tuple = tuple(episodes)
    return EpisodeSuite(
        schema_version=1,
        suite_id=suite_id,
        suite_seed=91,
        episodes=episode_tuple,
        digest_sha256=suite_digest(
            suite_id=suite_id,
            suite_seed=91,
            episodes=episode_tuple,
        ),
    )


def _two_episode_suite() -> EpisodeSuite:
    return _suite(
        EpisodeSpec(target_note_index=12, source_pitch_cents=6_100),
        EpisodeSpec(target_note_index=12, source_pitch_cents=5_900),
    )


def test_learned_rollout_passes_only_raw_actor_observation_and_reads_result_after_done() -> None:
    actor = ObservationOnlySpy()
    environments: list[FastSinePitchEnv] = []

    def environment_factory() -> FastSinePitchEnv:
        env = FastSinePitchEnv()
        environments.append(env)
        return env

    records = evaluate_learned_actor(
        actor,
        _suite(EpisodeSpec(target_note_index=12, source_pitch_cents=6_100)),
        environment_factory=environment_factory,
    )

    assert actor.keys == [("controls", "spectrum", "steps_remaining", "target_note")]
    assert records[0].terminal_reason is TerminalReason.SUBMITTED_FAILURE
    assert environments[0].result_reads == 1
    assert environments[0].closed is True


def test_learned_rollout_preserves_order_and_copies_immutable_terminal_values() -> None:
    suite = _two_episode_suite()
    actor = ObservationOnlySpy()
    env = FastSinePitchEnv()

    records = evaluate_learned_actor(actor, suite, environment_factory=lambda: env)

    assert tuple(record.episode for record in records) == suite.episodes
    assert tuple(record.episode_index for record in records) == (0, 1)
    assert env.reset_pairs == [episode.pair for episode in suite.episodes]
    assert records[0] == TerminalEpisodeRecord(
        episode_index=0,
        episode=suite.episodes[0],
        submitted_success=False,
        within_5_cents=False,
        within_1_cent=False,
        final_absolute_error_cents=100,
        action_count=1,
        excess_actions=None,
        invalid_action_count=0,
        total_return=-1.0,
        terminal_reason=TerminalReason.SUBMITTED_FAILURE,
    )
    with pytest.raises(FrozenInstanceError):
        records[0].action_count = 3  # type: ignore[misc]


def test_learned_rollout_closes_environment_when_actor_fails() -> None:
    class FailingActor:
        def act(self, observation: Mapping[str, object]) -> PitchAction:
            del observation
            raise RuntimeError("actor failed")

    env = FastSinePitchEnv()
    with pytest.raises(RuntimeError, match="actor failed"):
        evaluate_learned_actor(
            FailingActor(),
            _suite(EpisodeSpec(target_note_index=12, source_pitch_cents=6_100)),
            environment_factory=lambda: env,
        )
    assert env.closed is True


def test_rollouts_reject_suite_digest_mismatch_before_environment_construction() -> None:
    suite = replace(_two_episode_suite(), digest_sha256="0" * 64)

    def forbidden_factory() -> FastSinePitchEnv:
        raise AssertionError("invalid suite must be rejected before environment construction")

    with pytest.raises(ValueError, match="suite digest"):
        evaluate_learned_actor(
            ObservationOnlySpy(),
            suite,
            environment_factory=forbidden_factory,
        )


def test_learned_rollout_rejects_terminal_truth_mismatching_injected_episode() -> None:
    class MismatchedResultEnv(FastSinePitchEnv):
        def step(self, action: int):
            transition = super().step(action)
            if transition[2] or transition[3]:
                other = FastSinePitchEnv()
                other.reset(options=EpisodeSpec(12, 5_900).reset_options())
                other.step(PitchAction.SUBMIT)
                self._episode_result = other.episode_result
            return transition

    env = MismatchedResultEnv()
    with pytest.raises(RuntimeError, match="injected EpisodeSpec"):
        evaluate_learned_actor(
            ObservationOnlySpy(),
            _suite(EpisodeSpec(target_note_index=12, source_pitch_cents=6_100)),
            environment_factory=lambda: env,
        )
    assert env.closed is True


@pytest.mark.parametrize(
    ("kind", "mode", "environment_id"),
    [
        (BaselineKind.RANDOM, ObservationMode.SPECTRUM, "Harpy/SinePitch-v0"),
        (BaselineKind.REWARD_SEARCH, ObservationMode.REWARD_ONLY, "Harpy/SinePitchRewardOnly-v0"),
        (BaselineKind.SPECTRUM_PEAK, ObservationMode.SPECTRUM, "Harpy/SinePitch-v0"),
        (BaselineKind.ORACLE, ObservationMode.ORACLE, "Harpy/SinePitchOracle-v0"),
    ],
)
def test_baselines_use_matched_suite_and_keep_declared_capability_rows(
    kind: BaselineKind,
    mode: ObservationMode,
    environment_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _two_episode_suite()
    environments: list[FastSinePitchEnv] = []

    def factory(cache: SpectrumEvidenceCache, *, observation_mode: ObservationMode):
        del cache
        env = FastSinePitchEnv(observation_mode=observation_mode)
        environments.append(env)
        return env

    monkeypatch.setattr(evaluation, "make_cached_sine_pitch_env", factory)

    records = evaluate_baseline_suite(kind, suite, cache=SpectrumEvidenceCache())
    rows = build_evaluation_rows(
        actor_id=kind.value,
        trainer=None,
        seed=None,
        environment_id=kind.environment_id,
        observation_mode=kind.observation_mode,
        suite=suite,
        records=records,
    )

    assert environments[0].reset_pairs == [episode.pair for episode in suite.episodes]
    assert environments[0].observation_mode is mode
    assert environments[0].closed is True
    assert (rows[0].environment_id, rows[0].observation_mode) == (environment_id, mode)


def test_random_uses_exact_per_episode_seed_and_fresh_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _two_episode_suite()
    original_seed_sequence = np.random.SeedSequence
    entropies: list[tuple[int, ...]] = []
    policies: list[object] = []

    def recording_seed_sequence(entropy: list[int] | None = None):
        if entropy is not None:
            entropies.append(tuple(entropy))
        return original_seed_sequence(entropy)

    class SubmitRandomPolicy:
        def __init__(self, rng: np.random.Generator) -> None:
            assert isinstance(rng, np.random.Generator)
            policies.append(self)

        def act(
            self,
            observation: Mapping[str, object],
            previous_reward: float | None,
            previous_info: Mapping[str, object] | None,
        ) -> PitchAction:
            del observation
            assert previous_reward is None
            assert previous_info is None
            return PitchAction.SUBMIT

    monkeypatch.setattr(evaluation.np.random, "SeedSequence", recording_seed_sequence)
    monkeypatch.setattr(evaluation, "RandomPolicy", SubmitRandomPolicy)
    monkeypatch.setattr(
        evaluation,
        "make_cached_sine_pitch_env",
        lambda cache, *, observation_mode: FastSinePitchEnv(observation_mode),
    )

    evaluate_baseline_suite(BaselineKind.RANDOM, suite, cache=SpectrumEvidenceCache())

    assert entropies == [(suite.suite_seed, 1, 0), (suite.suite_seed, 1, 1)]
    assert len(policies) == 2
    assert policies[0] is not policies[1]


def test_only_reward_search_receives_prior_transition_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _suite(EpisodeSpec(target_note_index=12, source_pitch_cents=6_100))
    feedback: list[tuple[float | None, Mapping[str, object] | None]] = []

    class RecordingRewardSearch:
        def act(
            self,
            observation: Mapping[str, object],
            previous_reward: float | None,
            previous_info: Mapping[str, object] | None,
        ) -> PitchAction:
            feedback.append((previous_reward, previous_info))
            return PitchAction.CENT_DOWN if previous_reward is None else PitchAction.SUBMIT

    monkeypatch.setattr(evaluation, "RewardSearchPolicy", RecordingRewardSearch)
    monkeypatch.setattr(
        evaluation,
        "make_cached_sine_pitch_env",
        lambda cache, *, observation_mode: FastSinePitchEnv(observation_mode),
    )

    evaluate_baseline_suite(BaselineKind.REWARD_SEARCH, suite, cache=SpectrumEvidenceCache())

    assert feedback[0] == (None, None)
    assert feedback[1][0] is not None
    assert feedback[1][1] is not None


def test_aggregation_pins_episode_and_action_denominators_median_and_none() -> None:
    episodes = _two_episode_suite().episodes
    records = (
        TerminalEpisodeRecord(
            0,
            episodes[0],
            True,
            True,
            True,
            1,
            3,
            1,
            1,
            0.75,
            TerminalReason.SUBMITTED_SUCCESS,
        ),
        TerminalEpisodeRecord(
            1,
            episodes[1],
            False,
            False,
            False,
            9,
            1,
            None,
            0,
            -1.0,
            TerminalReason.SUBMITTED_FAILURE,
        ),
    )

    metrics = aggregate_episode_records(records)

    assert metrics.submitted_success_rate == 0.5
    assert metrics.submitted_within_1_cent_rate == 0.5
    assert metrics.final_within_5_cents_rate == 0.5
    assert metrics.final_within_1_cent_rate == 0.5
    assert metrics.mean_absolute_final_error_cents == 5.0
    assert metrics.median_absolute_final_error_cents == 5.0
    assert metrics.mean_actions == 2.0
    assert metrics.mean_successful_excess_actions == 1.0
    assert metrics.mean_return == -0.125
    assert metrics.truncation_rate == 0.0
    assert metrics.invalid_action_rate == 0.25

    no_success = tuple(
        TerminalEpisodeRecord(
            index,
            episode,
            False,
            False,
            False,
            100,
            1,
            None,
            0,
            -1.0,
            TerminalReason.SUBMITTED_FAILURE,
        )
        for index, episode in enumerate(episodes)
    )
    assert aggregate_episode_records(no_success).mean_successful_excess_actions is None


def test_ood_rows_are_lower_upper_combined_in_exact_order() -> None:
    episodes = (
        EpisodeSpec(target_note_index=12, source_pitch_cents=4_900),
        EpisodeSpec(target_note_index=12, source_pitch_cents=7_100),
    )
    suite = EpisodeSuite(
        schema_version=1,
        suite_id=EvaluationSuiteId.REGISTER_OOD,
        suite_seed=92,
        episodes=episodes,
        digest_sha256=suite_digest(
            suite_id=EvaluationSuiteId.REGISTER_OOD,
            suite_seed=92,
            episodes=episodes,
        ),
    )
    records = tuple(
        TerminalEpisodeRecord(
            index,
            episode,
            False,
            False,
            False,
            100,
            1,
            None,
            0,
            -1.0,
            TerminalReason.SUBMITTED_FAILURE,
        )
        for index, episode in enumerate(episodes)
    )

    rows = build_evaluation_rows(
        actor_id="bc",
        trainer=evaluation.TrainerKind.BC,
        seed=0,
        environment_id="Harpy/SinePitch-v0",
        observation_mode=ObservationMode.SPECTRUM,
        suite=suite,
        records=records,
    )

    assert tuple(row.subset for row in rows) == ("lower", "upper", "combined")
    assert tuple(record.episode_index for record in rows[0].episodes) == (0,)
    assert tuple(record.episode_index for record in rows[1].episodes) == (1,)


def test_spectrum_probes_copy_only_spectrum_and_shuffle_is_globally_repeatable() -> None:
    original_factories: list[Callable[[], gymnasium.Env]] = [lambda: FastSinePitchEnv()]
    snapshots: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}

    for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):

        class SnapshotActor:
            def __init__(self) -> None:
                self.observation: Mapping[str, object] | None = None

            def act(self, observation: Mapping[str, object]) -> PitchAction:
                self.observation = observation
                return PitchAction.SUBMIT

        plain_actor = SnapshotActor()
        probe_actor = SnapshotActor()
        suite = _suite(EpisodeSpec(target_note_index=12, source_pitch_cents=6_100))
        evaluate_learned_actor(plain_actor, suite, environment_factory=original_factories[0])
        evaluate_learned_actor(
            probe_actor,
            suite,
            environment_factory=make_spectrum_probe_factory(original_factories[0], probe),
        )
        assert plain_actor.observation is not None
        assert probe_actor.observation is not None
        snapshots[probe] = plain_actor.observation, probe_actor.observation

    plain, zeroed = snapshots[ZERO_SPECTRUM_PROBE]
    _, shuffled = snapshots[SHUFFLED_SPECTRUM_PROBE]
    for key in ("target_note", "controls", "steps_remaining"):
        np.testing.assert_array_equal(plain[key], zeroed[key])
        np.testing.assert_array_equal(plain[key], shuffled[key])
    plain_spectrum = np.asarray(plain["spectrum"])
    zero_spectrum = np.asarray(zeroed["spectrum"])
    shuffled_spectrum = np.asarray(shuffled["spectrum"])
    assert np.count_nonzero(zero_spectrum) == 0
    np.testing.assert_array_equal(np.sort(shuffled_spectrum), np.sort(plain_spectrum))
    assert not np.array_equal(shuffled_spectrum, plain_spectrum)

    assert evaluation.SPECTRUM_SHUFFLE_ID == "harpy-sine-spectrum-shuffle-v1"
    assert evaluation.SPECTRUM_SHUFFLE_PERMUTATION.flags.writeable is False


def test_shuffled_spectrum_uses_pinned_permutation_across_actor_evaluation_orders() -> None:
    class SpectrumCaptureActor:
        def __init__(self) -> None:
            self.spectra: list[np.ndarray] = []

        def act(self, observation: Mapping[str, object]) -> PitchAction:
            self.spectra.append(np.array(observation["spectrum"], copy=True))
            return PitchAction.SUBMIT

    suite = _two_episode_suite()
    expected_permutation = np.random.default_rng(
        np.random.SeedSequence([202_608_103, 1])
    ).permutation(1_961)

    def capture(*, shuffled: bool) -> tuple[np.ndarray, ...]:
        actor = SpectrumCaptureActor()

        def base_factory() -> gymnasium.Env:
            return FastSinePitchEnv()

        factory: Callable[[], gymnasium.Env] = base_factory
        if shuffled:
            factory = make_spectrum_probe_factory(factory, SHUFFLED_SPECTRUM_PROBE)
        evaluate_learned_actor(actor, suite, environment_factory=factory)
        return tuple(actor.spectra)

    plain = capture(shuffled=False)
    expected = tuple(spectrum[expected_permutation] for spectrum in plain)
    actor_a_then_b = (capture(shuffled=True), capture(shuffled=True))
    actor_b_then_a = (capture(shuffled=True), capture(shuffled=True))

    np.testing.assert_array_equal(
        evaluation.SPECTRUM_SHUFFLE_PERMUTATION,
        expected_permutation,
    )
    for evaluation_order in (actor_a_then_b, actor_b_then_a):
        for actor_spectra in evaluation_order:
            assert len(actor_spectra) == len(suite.episodes)
            for observed, pinned in zip(actor_spectra, expected, strict=True):
                np.testing.assert_array_equal(observed, pinned)
