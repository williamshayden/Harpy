import copy
from collections import UserDict

import gymnasium
import numpy as np
import pytest

import harpy.envs.sine_pitch as sine_pitch_module
from harpy.envs.models import ControlState, ObservationMode, PitchAction, TerminalReason
from harpy.envs.sine_pitch import SinePitchEnv, _CandidateEvidence


def test_spectrum_reset_is_seeded_fixed_and_actor_safe() -> None:
    env = SinePitchEnv()
    assert env.action_space == gymnasium.spaces.Discrete(7)

    observation, info = env.reset(seed=17)

    assert set(observation) == {"spectrum", "target_note", "controls", "steps_remaining"}
    assert env.observation_space.contains(observation)
    assert observation["spectrum"].shape == (1_961,)
    assert observation["spectrum"].dtype == np.float32
    assert observation["controls"].tolist() == [0, 0, 0]
    assert int(observation["steps_remaining"]) == 64
    assert info == {"step_count": 0, "steps_remaining": 64}


def test_oracle_replaces_spectrum_and_reward_only_has_neither() -> None:
    oracle, _ = SinePitchEnv(ObservationMode.ORACLE).reset(seed=4)
    reward_only, _ = SinePitchEnv(ObservationMode.REWARD_ONLY).reset(seed=4)

    assert set(oracle) == {
        "current_pitch_coordinate",
        "target_note",
        "controls",
        "steps_remaining",
    }
    assert oracle["current_pitch_coordinate"].shape == (1,)
    assert "spectrum" not in oracle
    assert set(reward_only) == {"target_note", "controls", "steps_remaining"}
    assert "spectrum" not in reward_only
    assert "current_pitch_coordinate" not in reward_only


@pytest.mark.parametrize(
    ("mode", "evidence_key", "evidence_space"),
    [
        (
            ObservationMode.SPECTRUM,
            "spectrum",
            gymnasium.spaces.Box(0.0, 1.0, shape=(1_961,), dtype=np.float32),
        ),
        (
            ObservationMode.ORACLE,
            "current_pitch_coordinate",
            gymnasium.spaces.Box(11.0, 109.0, shape=(1,), dtype=np.float32),
        ),
        (ObservationMode.REWARD_ONLY, None, None),
    ],
)
def test_each_mode_has_one_exact_fixed_dict_space(
    mode: ObservationMode,
    evidence_key: str | None,
    evidence_space: gymnasium.Space | None,
) -> None:
    env = SinePitchEnv(mode)
    spaces = env.observation_space.spaces
    expected_keys = {"target_note", "controls", "steps_remaining"}
    if evidence_key is not None:
        expected_keys.add(evidence_key)

    assert set(spaces) == expected_keys
    assert spaces["target_note"] == gymnasium.spaces.Discrete(25, dtype=np.int64)
    assert spaces["target_note"].dtype == np.dtype(np.int64)
    assert spaces["controls"] == gymnasium.spaces.Box(
        low=np.array([-2, -12, -100], dtype=np.int16),
        high=np.array([2, 12, 100], dtype=np.int16),
        dtype=np.int16,
    )
    assert spaces["controls"].dtype == np.dtype(np.int16)
    assert spaces["steps_remaining"] == gymnasium.spaces.Discrete(65, dtype=np.int64)
    assert spaces["steps_remaining"].dtype == np.dtype(np.int64)
    if evidence_key is not None:
        assert spaces[evidence_key] == evidence_space

    observation, _ = env.reset(seed=81)
    assert env.observation_space.contains(observation)
    assert observation["target_note"].dtype == np.int64
    assert observation["controls"].dtype == np.int16
    assert observation["steps_remaining"].dtype == np.int64
    if evidence_key is not None:
        assert observation[evidence_key].dtype == np.float32


@pytest.mark.parametrize("render_mode", ["human", "rgb_array", object()])
def test_constructor_rejects_every_non_null_render_mode(render_mode: object) -> None:
    with pytest.raises(ValueError, match="render_mode"):
        SinePitchEnv(render_mode=render_mode)  # type: ignore[arg-type]


@pytest.mark.parametrize("observation_mode", ["spectrum", "oracle", "reward_only", object()])
def test_constructor_requires_the_observation_mode_enum(observation_mode: object) -> None:
    with pytest.raises(ValueError, match="observation_mode"):
        SinePitchEnv(observation_mode)  # type: ignore[arg-type]


def test_injected_reset_is_exact_and_does_not_draw_from_the_seeded_stream() -> None:
    injected = SinePitchEnv()
    sibling = SinePitchEnv()

    observation, info = injected.reset(
        seed=np.int64(23),
        options={"target_note_index": np.int32(12), "source_pitch_cents": np.int64(5_995)},
    )
    after_injection, _ = injected.reset()
    expected, _ = sibling.reset(seed=23)

    assert int(observation["target_note"]) == 12
    assert info == {"step_count": 0, "steps_remaining": 64}
    assert int(after_injection["target_note"]) == int(expected["target_note"])
    np.testing.assert_array_equal(after_injection["spectrum"], expected["spectrum"])


def test_first_unseeded_injected_reset_initializes_gym_rng_without_drawing() -> None:
    env = SinePitchEnv()
    options = {"target_note_index": 12, "source_pitch_cents": 6_000}

    env.reset(options=options)

    assert isinstance(env._np_random, np.random.Generator)
    assert isinstance(env._np_random_seed, int)
    assert env._np_random_seed >= 0
    state_after_initialization = copy.deepcopy(env._np_random.bit_generator.state)
    env.reset(options=options)
    assert env._np_random.bit_generator.state == state_after_initialization


def test_injected_reset_render_failure_restores_prior_rng_and_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SinePitchEnv()
    env.reset(
        seed=103,
        options={"target_note_index": 9, "source_pitch_cents": 5_821},
    )
    env.step(PitchAction.SEMITONE_UP)
    state_before = _episode_state_snapshot(env)

    with monkeypatch.context() as patch:
        patch.setattr(
            sine_pitch_module,
            "encode_log_spectrum",
            lambda _samples: (_ for _ in ()).throw(RuntimeError("analysis failed")),
        )
        with pytest.raises(RuntimeError, match="analysis failed"):
            env.reset(options={"target_note_index": 4, "source_pitch_cents": 5_432})

    assert _episode_state_snapshot(env) == state_before


@pytest.mark.parametrize(
    "options",
    [
        {"target_note_index": 12},
        {"source_pitch_cents": 6_000},
        {"target_note_index": 12, "source_pitch_cents": 6_000, "extra": 1},
        UserDict({"target_note_index": 12, "source_pitch_cents": 6_000}),
        "target_note_index=12",
        object(),
        {"target_note_index": True, "source_pitch_cents": 6_000},
        {"target_note_index": 12.0, "source_pitch_cents": 6_000},
        {"target_note_index": np.array(12), "source_pitch_cents": 6_000},
        {"target_note_index": -1, "source_pitch_cents": 6_000},
        {"target_note_index": 25, "source_pitch_cents": 6_000},
        {"target_note_index": 12, "source_pitch_cents": False},
        {"target_note_index": 12, "source_pitch_cents": 6_000.0},
        {"target_note_index": 12, "source_pitch_cents": 4_799},
        {"target_note_index": 12, "source_pitch_cents": 7_201},
    ],
)
def test_invalid_options_are_atomic_for_episode_and_rng(options: object) -> None:
    env = SinePitchEnv()
    sibling = SinePitchEnv()
    first, _ = env.reset(seed=31)
    sibling_first, _ = sibling.reset(seed=31)
    env.step(PitchAction.CENT_UP)
    sibling.step(PitchAction.CENT_UP)
    state_before = _episode_state_snapshot(env)

    with pytest.raises(ValueError):
        env.reset(options=options)  # type: ignore[arg-type]

    assert _episode_state_snapshot(env) == state_before
    np.testing.assert_array_equal(first["spectrum"], sibling_first["spectrum"])
    continued, reward, terminated, truncated, info = env.step(PitchAction.CENT_DOWN)
    sibling_continued, sibling_reward, _, _, sibling_info = sibling.step(PitchAction.CENT_DOWN)
    np.testing.assert_array_equal(continued["spectrum"], sibling_continued["spectrum"])
    assert reward == sibling_reward
    assert (terminated, truncated) == (False, False)
    assert info == sibling_info
    after_rejection, _ = env.reset()
    sibling_next, _ = sibling.reset()
    assert int(after_rejection["target_note"]) == int(sibling_next["target_note"])
    np.testing.assert_array_equal(after_rejection["spectrum"], sibling_next["spectrum"])


def test_empty_framework_options_mean_an_ordinary_sampled_reset() -> None:
    env = SinePitchEnv()
    sibling = SinePitchEnv()

    observation, info = env.reset(seed=44, options={})
    expected, expected_info = sibling.reset(seed=44)

    assert int(observation["target_note"]) == int(expected["target_note"])
    np.testing.assert_array_equal(observation["spectrum"], expected["spectrum"])
    assert info == expected_info == {"step_count": 0, "steps_remaining": 64}


@pytest.mark.parametrize("seed", [True, False, -1, 1.0, "1", np.array(1), object()])
def test_invalid_seeds_are_atomic_for_episode_and_rng(seed: object) -> None:
    env = SinePitchEnv()
    sibling = SinePitchEnv()
    env.reset(seed=92)
    sibling.reset(seed=92)
    env.step(PitchAction.SEMITONE_UP)
    sibling.step(PitchAction.SEMITONE_UP)
    state_before = _episode_state_snapshot(env)

    with pytest.raises(ValueError, match="seed"):
        env.reset(seed=seed)  # type: ignore[arg-type]

    assert _episode_state_snapshot(env) == state_before
    continued, reward, terminated, truncated, info = env.step(PitchAction.SEMITONE_DOWN)
    sibling_continued, sibling_reward, _, _, sibling_info = sibling.step(PitchAction.SEMITONE_DOWN)
    np.testing.assert_array_equal(continued["spectrum"], sibling_continued["spectrum"])
    assert reward == sibling_reward
    assert (terminated, truncated) == (False, False)
    assert info == sibling_info
    after_rejection, _ = env.reset()
    sibling_next, _ = sibling.reset()
    assert int(after_rejection["target_note"]) == int(sibling_next["target_note"])
    np.testing.assert_array_equal(after_rejection["spectrum"], sibling_next["spectrum"])


def test_repeated_injected_reset_owns_read_only_reproducible_evidence() -> None:
    options = {"target_note_index": 11, "source_pitch_cents": 5_678}
    env = SinePitchEnv()

    first, _ = env.reset(options=options)
    first_audio = env._candidate_audio
    first_internal_spectrum = env._spectrum
    second, _ = env.reset(options=options)

    np.testing.assert_array_equal(env._candidate_audio, first_audio)
    np.testing.assert_array_equal(env._spectrum, first_internal_spectrum)
    np.testing.assert_array_equal(second["spectrum"], first["spectrum"])
    assert env._candidate_audio.flags.owndata
    assert not env._candidate_audio.flags.writeable
    assert env._spectrum.flags.owndata
    assert not env._spectrum.flags.writeable
    assert first["spectrum"].flags.owndata
    assert second["spectrum"].flags.owndata
    assert not np.shares_memory(first["spectrum"], env._spectrum)
    assert not np.shares_memory(second["spectrum"], env._spectrum)


def test_registered_base_candidate_evidence_retains_owned_audio() -> None:
    env = SinePitchEnv()

    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})

    audio = env._candidate_audio
    spectrum = env._spectrum
    assert audio is not None and spectrum is not None
    assert audio.dtype == spectrum.dtype == np.float32
    assert audio.flags.owndata and not audio.flags.writeable
    assert spectrum.flags.owndata and not spectrum.flags.writeable
    assert not np.shares_memory(audio, observation["spectrum"])
    assert not np.shares_memory(spectrum, observation["spectrum"])


def test_candidate_evidence_hook_can_omit_private_audio_without_changing_observation() -> None:
    class SpectrumOnlyEnv(SinePitchEnv):
        def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
            evidence = super()._candidate_evidence(candidate_cents)
            return _CandidateEvidence(None, evidence.spectrum)

    env = SpectrumOnlyEnv()
    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})

    assert env._candidate_audio is None
    assert observation["spectrum"].shape == (1_961,)


def test_candidate_evidence_hook_runs_only_for_reset_and_applied_actions() -> None:
    class RecordingEvidenceEnv(SinePitchEnv):
        def __init__(self) -> None:
            super().__init__()
            self.evidence_candidates: list[int] = []

        def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
            self.evidence_candidates.append(candidate_cents)
            return super()._candidate_evidence(candidate_cents)

    env = RecordingEvidenceEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.SUBMIT)

    assert env.evidence_candidates == [6_000, 7_200, 8_400]


def test_sampled_reset_render_failure_restores_progressed_episode_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SinePitchEnv()
    sibling = SinePitchEnv()
    options = {"target_note_index": 9, "source_pitch_cents": 5_821}
    env.reset(seed=57, options=options)
    sibling.reset(seed=57, options=options)
    env.step(PitchAction.SEMITONE_UP)
    sibling.step(PitchAction.SEMITONE_UP)
    state_before = _episode_state_snapshot(env)

    with monkeypatch.context() as patch:
        patch.setattr(
            sine_pitch_module,
            "encode_log_spectrum",
            lambda _samples: (_ for _ in ()).throw(RuntimeError("analysis failed")),
        )
        with pytest.raises(RuntimeError, match="analysis failed"):
            env.reset()

    assert _episode_state_snapshot(env) == state_before
    continued, reward, terminated, truncated, info = env.step(PitchAction.SEMITONE_DOWN)
    sibling_continued, sibling_reward, _, _, sibling_info = sibling.step(PitchAction.SEMITONE_DOWN)
    np.testing.assert_array_equal(continued["spectrum"], sibling_continued["spectrum"])
    assert reward == sibling_reward
    assert (terminated, truncated) == (False, False)
    assert info == sibling_info
    after_failure, _ = env.reset()
    sibling_next, _ = sibling.reset()
    assert int(after_failure["target_note"]) == int(sibling_next["target_note"])
    np.testing.assert_array_equal(after_failure["spectrum"], sibling_next["spectrum"])


def test_target_changes_never_change_injected_candidate_evidence() -> None:
    low_target = SinePitchEnv()
    high_target = SinePitchEnv()

    low_observation, _ = low_target.reset(
        options={"target_note_index": 0, "source_pitch_cents": 6_123}
    )
    high_observation, _ = high_target.reset(
        options={"target_note_index": 24, "source_pitch_cents": 6_123}
    )

    assert int(low_observation["target_note"]) == 0
    assert int(high_observation["target_note"]) == 24
    np.testing.assert_array_equal(low_target._candidate_audio, high_target._candidate_audio)
    np.testing.assert_array_equal(low_observation["spectrum"], high_observation["spectrum"])


def test_in_tolerance_injected_pair_is_permitted() -> None:
    env = SinePitchEnv()

    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})

    assert int(observation["target_note"]) == 12
    assert env.observation_space.contains(observation)


def test_null_render_mode_has_no_external_rendering() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})

    assert env.render() is None


def test_submit_is_required_and_five_cents_is_success() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_995})

    _, reward, terminated, truncated, info = env.step(PitchAction.SUBMIT)

    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["submitted_success"] is True
    assert env.episode_result.final_absolute_error_cents == 5


def test_six_cents_fails_only_when_submitted() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_994})

    _, reward, terminated, truncated, _ = env.step(PitchAction.CENT_UP)

    assert reward == pytest.approx(1 / 6_100 - 0.00001)
    assert (terminated, truncated) == (False, False)


@pytest.mark.parametrize(
    ("source_pitch_cents", "action", "expected_reward"),
    [
        (5_900, PitchAction.SEMITONE_UP, 100 / 6_100 - 0.00001),
        (6_100, PitchAction.SEMITONE_DOWN, 100 / 6_100 - 0.00001),
        (5_900, PitchAction.SEMITONE_DOWN, -100 / 6_100 - 0.00001),
        (6_100, PitchAction.SEMITONE_UP, -100 / 6_100 - 0.00001),
    ],
)
def test_applied_pitch_reward_is_normalized_error_improvement_minus_step_cost(
    source_pitch_cents: int,
    action: PitchAction,
    expected_reward: float,
) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": source_pitch_cents})

    observation, reward, terminated, truncated, info = env.step(action)

    assert reward == pytest.approx(expected_reward)
    assert (terminated, truncated) == (False, False)
    assert info == {
        "step_count": 1,
        "steps_remaining": 63,
        "action": action.label,
        "action_applied": True,
        "submitted": False,
        "submitted_success": False,
    }
    assert env.observation_space.contains(observation)


def test_passing_through_success_tolerance_does_not_end_the_episode() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_994})

    _, first_reward, first_terminated, first_truncated, _ = env.step(PitchAction.CENT_UP)
    _, second_reward, second_terminated, second_truncated, _ = env.step(PitchAction.CENT_UP)

    assert first_reward == pytest.approx(1 / 6_100 - 0.00001)
    assert second_reward == pytest.approx(1 / 6_100 - 0.00001)
    assert (first_terminated, first_truncated) == (False, False)
    assert (second_terminated, second_truncated) == (False, False)
    with pytest.raises(RuntimeError, match="done"):
        _ = env.episode_result


@pytest.mark.parametrize("source_pitch_cents", [5_994, 6_006])
def test_submit_outside_five_cents_is_a_failed_termination(source_pitch_cents: int) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": source_pitch_cents})

    _, reward, terminated, truncated, info = env.step(PitchAction.SUBMIT)

    assert (reward, terminated, truncated) == (-1.0, True, False)
    assert info == {
        "step_count": 1,
        "steps_remaining": 63,
        "action": "Submit",
        "action_applied": False,
        "submitted": True,
        "submitted_success": False,
    }
    result = env.episode_result
    assert result.terminal_reason is TerminalReason.SUBMITTED_FAILURE
    assert result.submitted_success is False
    assert result.excess_actions is None


@pytest.mark.parametrize(
    ("source_pitch_cents", "within_five", "within_one"),
    [
        (5_995, True, False),
        (6_005, True, False),
        (5_999, True, True),
        (6_001, True, True),
        (5_998, True, False),
        (6_002, True, False),
        (5_994, False, False),
        (6_006, False, False),
    ],
)
def test_result_truth_uses_inclusive_five_and_one_cent_boundaries(
    source_pitch_cents: int,
    within_five: bool,
    within_one: bool,
) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": source_pitch_cents})
    env.step(PitchAction.SUBMIT)

    result = env.episode_result

    assert result.within_5_cents is within_five
    assert result.within_1_cent is within_one
    assert result.final_signed_error_cents == source_pitch_cents - 6_000
    assert result.final_absolute_error_cents == abs(source_pitch_cents - 6_000)


def test_success_result_is_complete_and_uses_the_initial_optimal_plan() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_994})
    _, move_reward, _, _, _ = env.step(PitchAction.CENT_UP)
    env.step(PitchAction.SUBMIT)

    result = env.episode_result

    assert result.source_pitch_cents == 5_994
    assert result.target_note_index == 12
    assert result.target_pitch_cents == 6_000
    assert result.final_pitch_cents == 5_995
    assert result.initial_signed_error_cents == -6
    assert result.initial_absolute_error_cents == 6
    assert result.final_signed_error_cents == -5
    assert result.final_absolute_error_cents == 5
    assert result.actions == (PitchAction.CENT_UP, PitchAction.SUBMIT)
    assert result.invalid_action_count == 0
    assert result.optimal_actions == (PitchAction.CENT_UP, PitchAction.SUBMIT)
    assert result.optimal_total_actions == 2
    assert result.excess_actions == 0
    assert result.total_return == pytest.approx(move_reward + 1.0)
    assert result.terminal_reason is TerminalReason.SUBMITTED_SUCCESS


@pytest.mark.parametrize(
    ("move", "controls", "control_index", "boundary"),
    [
        (PitchAction.OCTAVE_UP, ControlState(octaves=2), 0, 2),
        (PitchAction.OCTAVE_DOWN, ControlState(octaves=-2), 0, -2),
        (PitchAction.SEMITONE_UP, ControlState(semitones=12), 1, 12),
        (PitchAction.SEMITONE_DOWN, ControlState(semitones=-12), 1, -12),
        (PitchAction.CENT_UP, ControlState(cents=100), 2, 100),
        (PitchAction.CENT_DOWN, ControlState(cents=-100), 2, -100),
    ],
)
def test_each_control_bound_blocks_without_carry_or_rerender(
    move: PitchAction,
    controls: ControlState,
    control_index: int,
    boundary: int,
) -> None:
    env = SinePitchEnv(ObservationMode.ORACLE)
    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env._controls = controls
    observation = env._observation()
    audio_at_bound = env._candidate_audio
    spectrum_at_bound = env._spectrum
    coordinate_at_bound = observation["current_pitch_coordinate"].copy()

    blocked, reward, terminated, truncated, info = env.step(move)

    assert blocked["controls"][control_index] == boundary
    assert sum(int(value != 0) for value in blocked["controls"]) == 1
    assert reward == -0.01
    assert (terminated, truncated) == (False, False)
    assert info["action_applied"] is False
    assert env._candidate_audio is audio_at_bound
    assert env._spectrum is spectrum_at_bound
    np.testing.assert_array_equal(blocked["current_pitch_coordinate"], coordinate_at_bound)


def test_submit_does_not_rerender_candidate_evidence() -> None:
    env = SinePitchEnv()
    observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    audio_before = env._candidate_audio
    spectrum_before = env._spectrum

    submitted, _, _, _, _ = env.step(PitchAction.SUBMIT)

    assert env._candidate_audio is audio_before
    assert env._spectrum is spectrum_before
    np.testing.assert_array_equal(submitted["spectrum"], observation["spectrum"])


def test_every_applied_action_fresh_renders_from_immutable_source() -> None:
    env = SinePitchEnv()
    initial, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    initial_audio = env._candidate_audio
    initial_spectrum = env._spectrum

    env.step(PitchAction.CENT_UP)
    moved_audio = env._candidate_audio
    moved_spectrum = env._spectrum
    returned, _, _, _, _ = env.step(PitchAction.CENT_DOWN)

    assert moved_audio is not initial_audio
    assert moved_spectrum is not initial_spectrum
    assert env._candidate_audio is not moved_audio
    assert env._spectrum is not moved_spectrum
    np.testing.assert_array_equal(env._candidate_audio, initial_audio)
    np.testing.assert_array_equal(returned["spectrum"], initial["spectrum"])


def test_reset_and_applied_moves_each_render_the_exact_candidate_before_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_renderer = sine_pitch_module._render_settled_sine
    real_encoder = sine_pitch_module.encode_log_spectrum
    events: list[tuple[object, ...]] = []

    def recording_renderer(candidate_cents: int) -> np.ndarray:
        events.append(("render", candidate_cents))
        return real_renderer(candidate_cents)

    def recording_encoder(samples: np.ndarray) -> np.ndarray:
        events.append(("analyze",))
        return real_encoder(samples)

    monkeypatch.setattr(sine_pitch_module, "_render_settled_sine", recording_renderer)
    monkeypatch.setattr(sine_pitch_module, "encode_log_spectrum", recording_encoder)
    env = SinePitchEnv()

    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)

    assert events == [
        ("render", 6_000),
        ("analyze",),
        ("render", 7_200),
        ("analyze",),
        ("render", 8_400),
        ("analyze",),
    ]
    events.clear()

    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.SUBMIT)

    assert events == []


def test_applied_action_analysis_failure_is_atomic_and_episode_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_880})
    env.step(PitchAction.SEMITONE_UP)
    state_before = _episode_state_snapshot(env)

    with monkeypatch.context() as patch:
        patch.setattr(
            sine_pitch_module,
            "encode_log_spectrum",
            lambda _samples: (_ for _ in ()).throw(RuntimeError("analysis failed")),
        )
        with pytest.raises(RuntimeError, match="analysis failed"):
            env.step(PitchAction.CENT_UP)

    assert _episode_state_snapshot(env) == state_before
    observation, _, terminated, truncated, info = env.step(PitchAction.CENT_UP)
    assert observation["controls"].tolist() == [0, 1, 1]
    assert (terminated, truncated) == (False, False)
    assert info["step_count"] == 2


def test_terminal_result_construction_failure_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    state_before = _episode_state_snapshot(env)

    with monkeypatch.context() as patch:
        patch.setattr(
            sine_pitch_module,
            "EpisodeResult",
            lambda **_values: (_ for _ in ()).throw(RuntimeError("result failed")),
        )
        with pytest.raises(RuntimeError, match="result failed"):
            env.step(PitchAction.SUBMIT)

    assert _episode_state_snapshot(env) == state_before
    _, reward, terminated, truncated, _ = env.step(PitchAction.SUBMIT)
    assert (reward, terminated, truncated) == (1.0, True, False)


def test_control_equivalent_orders_have_bit_identical_evidence() -> None:
    first = SinePitchEnv()
    second = SinePitchEnv()
    options = {"target_note_index": 8, "source_pitch_cents": 5_432}
    first.reset(options=options)
    second.reset(options=options)

    for action in (PitchAction.OCTAVE_UP, PitchAction.SEMITONE_DOWN, PitchAction.CENT_UP):
        first_observation, _, _, _, _ = first.step(action)
    for action in (PitchAction.CENT_UP, PitchAction.OCTAVE_UP, PitchAction.SEMITONE_DOWN):
        second_observation, _, _, _, _ = second.step(action)

    np.testing.assert_array_equal(first._candidate_audio, second._candidate_audio)
    np.testing.assert_array_equal(first_observation["spectrum"], second_observation["spectrum"])


def test_inverse_sequence_returns_bit_identical_initial_evidence() -> None:
    env = SinePitchEnv()
    initial, _ = env.reset(options={"target_note_index": 8, "source_pitch_cents": 5_432})
    initial_audio = env._candidate_audio

    for action in (
        PitchAction.OCTAVE_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_DOWN,
        PitchAction.SEMITONE_DOWN,
        PitchAction.OCTAVE_DOWN,
    ):
        returned, _, _, _, _ = env.step(action)

    np.testing.assert_array_equal(env._candidate_audio, initial_audio)
    np.testing.assert_array_equal(returned["spectrum"], initial["spectrum"])


def test_target_changes_never_change_stepped_candidate_evidence() -> None:
    first = SinePitchEnv()
    second = SinePitchEnv()
    first.reset(options={"target_note_index": 0, "source_pitch_cents": 6_123})
    second.reset(options={"target_note_index": 24, "source_pitch_cents": 6_123})

    first_observation, _, _, _, _ = first.step(PitchAction.SEMITONE_UP)
    second_observation, _, _, _, _ = second.step(PitchAction.SEMITONE_UP)

    np.testing.assert_array_equal(first._candidate_audio, second._candidate_audio)
    np.testing.assert_array_equal(first_observation["spectrum"], second_observation["spectrum"])
    assert int(first_observation["target_note"]) == 0
    assert int(second_observation["target_note"]) == 24


@pytest.mark.parametrize(
    "action",
    [True, False, 1.0, "1", np.array(1), np.array([1]), -1, 7, object()],
)
def test_invalid_actions_are_atomic(action: object) -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    state_before = (
        env._controls,
        env._step_count,
        env._actions,
        env._invalid_action_count,
        env._total_return,
        env._candidate_audio,
        env._spectrum,
        env._episode_result,
    )

    with pytest.raises(ValueError, match="action"):
        env.step(action)  # type: ignore[arg-type]

    assert (
        env._controls,
        env._step_count,
        env._actions,
        env._invalid_action_count,
        env._total_return,
        env._candidate_audio,
        env._spectrum,
        env._episode_result,
    ) == state_before


def test_numpy_and_python_integer_actions_are_accepted() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})

    first, _, _, _, first_info = env.step(np.int64(PitchAction.CENT_UP))
    second, _, _, _, second_info = env.step(int(PitchAction.CENT_DOWN))

    assert first["controls"].tolist() == [0, 0, 1]
    assert second["controls"].tolist() == [0, 0, 0]
    assert first_info["action"] == "Cent Up"
    assert second_info["action"] == "Cent Down"


def test_sixty_fourth_valid_move_applies_then_truncates_with_surcharge() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)
    for _ in range(61):
        env.step(PitchAction.OCTAVE_UP)

    observation, reward, terminated, truncated, info = env.step(PitchAction.OCTAVE_DOWN)

    assert reward == pytest.approx(1_200 / 6_100 - 0.00001 - 1.0)
    assert (terminated, truncated) == (False, True)
    assert observation["controls"].tolist() == [1, 0, 0]
    assert info == {
        "step_count": 64,
        "steps_remaining": 0,
        "action": "Octave Down",
        "action_applied": True,
        "submitted": False,
        "submitted_success": False,
    }
    result = env.episode_result
    assert result.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
    assert len(result.actions) == 64
    assert result.invalid_action_count == 61
    assert result.excess_actions is None


def test_sixty_fourth_blocked_move_is_negative_one_point_zero_one() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)
    for _ in range(61):
        env.step(PitchAction.OCTAVE_UP)
    audio_before = env._candidate_audio
    spectrum_before = env._spectrum

    _, reward, terminated, truncated, info = env.step(PitchAction.OCTAVE_UP)

    assert reward == -1.01
    assert (terminated, truncated) == (False, True)
    assert info["action_applied"] is False
    assert env._candidate_audio is audio_before
    assert env._spectrum is spectrum_before
    assert env.episode_result.invalid_action_count == 62


def test_submit_on_step_sixty_four_terminates_without_truncation_surcharge() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.OCTAVE_UP)
    env.step(PitchAction.OCTAVE_UP)
    for _ in range(61):
        env.step(PitchAction.OCTAVE_UP)

    _, reward, terminated, truncated, info = env.step(PitchAction.SUBMIT)

    assert (reward, terminated, truncated) == (-1.0, True, False)
    assert info["step_count"] == 64
    assert info["steps_remaining"] == 0
    assert info["submitted"] is True
    assert env.episode_result.terminal_reason is TerminalReason.SUBMITTED_FAILURE


def test_post_done_step_is_rejected_atomically() -> None:
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_000})
    env.step(PitchAction.SUBMIT)
    result_before = env.episode_result
    state_before = (
        env._controls,
        env._step_count,
        env._actions,
        env._invalid_action_count,
        env._total_return,
        env._candidate_audio,
        env._spectrum,
        env._episode_result,
    )

    with pytest.raises(RuntimeError, match="done"):
        env.step(PitchAction.CENT_UP)

    assert env.episode_result is result_before
    assert (
        env._controls,
        env._step_count,
        env._actions,
        env._invalid_action_count,
        env._total_return,
        env._candidate_audio,
        env._spectrum,
        env._episode_result,
    ) == state_before


def test_step_before_reset_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="reset"):
        SinePitchEnv().step(PitchAction.CENT_UP)


@pytest.mark.parametrize("mode", list(ObservationMode))
def test_observations_and_info_expose_only_actor_safe_allowlisted_keys(
    mode: ObservationMode,
) -> None:
    env = SinePitchEnv(mode)
    reset_observation, reset_info = env.reset(
        options={"target_note_index": 12, "source_pitch_cents": 5_994}
    )
    step_observation, _, _, _, step_info = env.step(PitchAction.CENT_UP)

    base_keys = {"target_note", "controls", "steps_remaining"}
    if mode is ObservationMode.SPECTRUM:
        base_keys.add("spectrum")
    elif mode is ObservationMode.WAVEFORM:
        base_keys.add("waveform")
    elif mode is ObservationMode.ORACLE:
        base_keys.add("current_pitch_coordinate")
    assert set(reset_observation) == base_keys
    assert set(step_observation) == base_keys
    assert set(reset_info) == {"step_count", "steps_remaining"}
    assert set(step_info) == {
        "step_count",
        "steps_remaining",
        "action",
        "action_applied",
        "submitted",
        "submitted_success",
    }
    forbidden_truth = {
        "source_pitch_cents",
        "target_pitch_cents",
        "initial_signed_error_cents",
        "initial_absolute_error_cents",
        "final_signed_error_cents",
        "final_absolute_error_cents",
        "within_5_cents",
        "within_1_cent",
        "optimal_actions",
        "excess_actions",
        "terminal_reason",
    }
    assert forbidden_truth.isdisjoint(_recursive_keys(reset_observation))
    assert forbidden_truth.isdisjoint(_recursive_keys(reset_info))
    assert forbidden_truth.isdisjoint(_recursive_keys(step_observation))
    assert forbidden_truth.isdisjoint(_recursive_keys(step_info))


def _recursive_keys(value: object) -> set[str]:
    if not isinstance(value, dict):
        return set()
    keys = {str(key) for key in value}
    for child in value.values():
        keys.update(_recursive_keys(child))
    return keys


def _episode_state_snapshot(env: SinePitchEnv) -> dict[str, object]:
    return {
        "source_pitch_cents": env._source_pitch_cents,
        "target_note_index": env._target_note_index,
        "controls": env._controls,
        "step_count": env._step_count,
        "actions": env._actions,
        "invalid_action_count": env._invalid_action_count,
        "total_return": env._total_return,
        "episode_result": env._episode_result,
        "candidate_audio_id": id(env._candidate_audio),
        "candidate_audio": (
            None if env._candidate_audio is None else env._candidate_audio.tobytes()
        ),
        "spectrum_id": id(env._spectrum),
        "spectrum": None if env._spectrum is None else env._spectrum.tobytes(),
        "np_random_seed": env.np_random_seed,
        "np_random_state": copy.deepcopy(env.np_random.bit_generator.state),
    }
