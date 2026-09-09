import math

import numpy as np
import pytest

import harpy.envs.robustness as robustness
from harpy.envs.models import ObservationMode, PitchAction
from harpy.envs.observations import owned_observation
from harpy.envs.robustness import (
    CLEAN_CONDITION,
    ROBUSTNESS_CONDITIONS,
    ConditionSpec,
    RobustnessEvidenceCache,
    make_robustness_env,
)
from harpy.envs.sine_pitch import SinePitchEnv, _render_settled_sine
from harpy.envs.spectrum import encode_log_spectrum
from harpy.synth import RenderConfig, SynthEngine, SynthPatch
from harpy.tuning import Tuning


def condition(name):
    return next(item for item in ROBUSTNESS_CONDITIONS if item.id == name)


@pytest.mark.parametrize("coordinate", [1100, 6000, 10900])
def test_clean_anchor_preserves_legacy_waveform_and_spectrum_bytes(coordinate):
    engine = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(Tuning().frequency_hz_for_midi_coordinate(coordinate / 100))
    expected_audio = engine.render(48 + 28800 + 262144)[48 + 28800 :]
    expected_spectrum = encode_log_spectrum(expected_audio)
    cache = RobustnessEvidenceCache()
    actual = cache.evidence(coordinate, CLEAN_CONDITION, 27)
    assert actual.waveform.tobytes() == expected_audio.tobytes()
    assert actual.spectrum.tobytes() == expected_spectrum.tobytes()
    assert actual.waveform.shape == (262144,) and actual.waveform.dtype == np.float32
    assert actual.realized_snr_db is None
    # Nuisance identity cannot perturb the clean anchor or duplicate its cache entry.
    second = cache.evidence(coordinate, CLEAN_CONDITION, 999)
    assert second.waveform.tobytes() == actual.waveform.tobytes()
    assert len(cache) == 1


@pytest.mark.parametrize("mode", [ObservationMode.SPECTRUM, ObservationMode.WAVEFORM])
def test_observations_are_owned_fixed_and_hide_truth_and_nuisance(mode):
    env = make_robustness_env(condition("noise-10db"), nuisance_seed=13, observation_mode=mode)
    observation, info = env.reset(options={"source_pitch_cents": 6000, "target_note_index": 12})
    evidence_key = mode.value
    assert set(observation) == {evidence_key, "controls", "target_note", "steps_remaining"}
    assert set(info) == {"step_count", "steps_remaining"}
    assert env.observation_space.contains(observation)
    copy = owned_observation(observation, mode)
    observation[evidence_key][:] = 0
    assert np.any(copy[evidence_key])
    restored, _, done, truncated, _ = env.step(PitchAction.SUBMIT)
    assert done and not truncated and env.episode_result.submitted_success
    np.testing.assert_array_equal(restored[evidence_key], copy[evidence_key])
    metadata = env.evidence_metadata
    metadata["nuisance_seed"] = 444
    assert env.evidence_metadata["nuisance_seed"] == 13
    assert abs(env.evidence_metadata["realized_snr_db"] - 10) < 1e-5


def test_noise_is_paired_across_conditions_candidates_and_order():
    cache = RobustnessEvidenceCache()
    dry = cache.evidence(6000, CLEAN_CONDITION, 7).waveform.astype(np.float64)
    at30 = cache.evidence(6000, condition("noise-30db"), 7)
    cache.evidence(6100, condition("noise-10db"), 998)
    at10 = cache.evidence(6000, condition("noise-10db"), 7)
    noise30 = at30.waveform.astype(np.float64) - dry
    noise10 = at10.waveform.astype(np.float64) - dry
    np.testing.assert_allclose(noise30 * 10, noise10, atol=1e-7, rtol=1e-5)
    for evidence, snr in ((at30, 30), (at10, 10)):
        assert abs(evidence.realized_snr_db - snr) < 1e-5
    later = RobustnessEvidenceCache().evidence(6000, condition("noise-30db"), 7)
    assert later.waveform.tobytes() == at30.waveform.tobytes()
    assert later.spectrum.tobytes() == at30.spectrum.tobytes()
    alternate = cache.evidence(6100, condition("noise-10db"), 7)
    alternate_dry = cache.evidence(6100, CLEAN_CONDITION, 7).waveform.astype(np.float64)
    alternate_noise = alternate.waveform.astype(np.float64) - alternate_dry
    np.testing.assert_allclose(
        alternate_noise / alternate.dry_rms, noise10 / at10.dry_rms, atol=2e-7, rtol=1e-4
    )


def test_gain_phase_and_combined_condition_apply_to_waveform_before_noise():
    cache = RobustnessEvidenceCache()
    clean = cache.evidence(6123, CLEAN_CONDITION, 19)
    quieter = cache.evidence(6123, condition("level-minus-24db"), 19)
    np.testing.assert_array_equal(
        quieter.waveform, (clean.waveform.astype(np.float64) * 10 ** (-24 / 20)).astype(np.float32)
    )
    phase = cache.evidence(6123, condition("phase-90"), 19)
    assert not np.array_equal(phase.waveform, clean.waveform)
    assert abs(phase.dry_rms / clean.dry_rms - 1) < 0.002
    combined = cache.evidence(6123, condition("combined"), 19)
    np.testing.assert_allclose(combined.dry_rms, phase.dry_rms * 10 ** (-24 / 20), rtol=1e-7)
    assert abs(combined.realized_snr_db - 10) < 1e-5


def test_revisited_pitch_has_identical_static_observation_and_truth_is_unchanged():
    env = make_robustness_env(condition("combined"), nuisance_seed=52)
    clean = SinePitchEnv()
    options = {"source_pitch_cents": 5980, "target_note_index": 12}
    initial, _ = env.reset(options=options)
    clean.reset(options=options)
    env.step(PitchAction.CENT_UP)
    revisited, *_ = env.step(PitchAction.CENT_DOWN)
    np.testing.assert_array_equal(initial["spectrum"], revisited["spectrum"])
    for action in (PitchAction.CENT_UP, PitchAction.CENT_DOWN):
        clean.step(action)
    noisy_terminal = env.step(PitchAction.SUBMIT)
    clean_terminal = clean.step(PitchAction.SUBMIT)
    assert noisy_terminal[1:] == clean_terminal[1:]
    assert env.episode_result == clean.episode_result


def test_lru_is_byte_bounded_and_cache_returns_owned_arrays(monkeypatch):
    calls = []

    def render(pitch, spec, seed):
        calls.append((pitch, spec.id, seed))
        return robustness._Evidence(
            np.full(8, pitch, np.float32), np.ones(4, np.float32), 1.0, None
        )

    monkeypatch.setattr(robustness, "_render", render)
    pilot = RobustnessEvidenceCache()
    pilot.evidence(6000, CLEAN_CONDITION, 1)
    cache = RobustnessEvidenceCache(max_bytes=2 * pilot.retained_bytes)
    calls.clear()
    first = cache.evidence(6000, CLEAN_CONDITION, 1)
    first.waveform[:] = 0
    assert cache.evidence(6000, CLEAN_CONDITION, 2).waveform[0] == 6000
    cache.evidence(6001, CLEAN_CONDITION, 1)
    cache.evidence(6000, CLEAN_CONDITION, 1)  # Refresh oldest.
    cache.evidence(6002, CLEAN_CONDITION, 1)  # Evict 6001.
    assert len(cache) == 2 and 96 < cache.retained_bytes <= cache.max_bytes
    cache.evidence(6001, CLEAN_CONDITION, 1)
    assert [entry[0] for entry in calls] == [6000, 6001, 6002, 6001]
    cache.clear()
    assert cache.retained_bytes == 0 and len(cache) == 0
    too_small = RobustnessEvidenceCache(max_bytes=1)
    too_small.evidence(6000, CLEAN_CONDITION, 1)
    assert len(too_small) == 0 and too_small.retained_bytes == 0


def test_spectrum_cache_is_compact_and_waveform_upgrades_preserve_capture():
    cache = RobustnessEvidenceCache()
    compact = cache.evidence(6013, condition("combined"), 42, include_waveform=False)
    assert compact.waveform is None
    assert 1961 * 4 < cache.retained_bytes < 1961 * 4 + 2048
    full = cache.evidence(6013, condition("combined"), 42)
    assert (1961 + 262144) * 4 < cache.retained_bytes < (1961 + 262144) * 4 + 2048
    assert compact.waveform_sha256 == full.waveform_sha256
    assert compact.spectrum_sha256 == full.spectrum_sha256
    np.testing.assert_array_equal(compact.spectrum, full.spectrum)
    assert cache.evidence(6013, condition("combined"), 42, include_waveform=False).waveform is None


@pytest.mark.parametrize("coordinate", [1100, 1101, 4800, 5003, 5983, 6983, 7200, 10899, 10900])
@pytest.mark.parametrize("phase", [0.0, math.pi / 4, math.pi / 2])
def test_settled_render_matches_full_engine_exactly(coordinate, phase):
    engine = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(Tuning().frequency_hz_for_midi_coordinate(coordinate / 100), phase_radians=phase)
    expected = engine.render(48 + 28800 + 262144)[48 + 28800 :]
    actual = _render_settled_sine(coordinate, phase_radians=phase)
    assert actual.tobytes() == expected.tobytes()
    assert encode_log_spectrum(actual).tobytes() == encode_log_spectrum(expected).tobytes()


def test_failed_render_does_not_change_evaluator_metadata(monkeypatch):
    env = make_robustness_env(condition("noise-10db"), nuisance_seed=1)
    env.reset(options={"source_pitch_cents": 6000, "target_note_index": 12})
    previous = env.evidence_metadata
    monkeypatch.setattr(
        env._evidence_cache,
        "evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("failed")),
    )
    with pytest.raises(ValueError, match="failed"):
        env.step(PitchAction.CENT_UP)
    assert env.evidence_metadata == previous


def test_condition_codec_rejects_mislabeled_and_nonfinite_conditions():
    assert len(ROBUSTNESS_CONDITIONS) == 8
    for spec in ROBUSTNESS_CONDITIONS:
        assert ConditionSpec.from_document(spec.to_document()) == spec
    with pytest.raises(ValueError):
        ConditionSpec("clean", gain_db=-12)
    with pytest.raises(ValueError):
        ConditionSpec("noise-10db", snr_db=float("nan"))
    with pytest.raises(ValueError):
        ConditionSpec.from_document({**CLEAN_CONDITION.to_document(), "secret": 3})


@pytest.mark.parametrize("phase", [True, math.nan, math.inf, -1, math.tau])
def test_phase_validation_is_atomic(phase):
    actual = SynthEngine(RenderConfig(), SynthPatch())
    expected = SynthEngine(RenderConfig(), SynthPatch())
    actual.note_on(440)
    expected.note_on(440)
    with pytest.raises(ValueError, match="phase_radians"):
        actual.note_on(880, phase_radians=phase)
    np.testing.assert_array_equal(actual.render(100), expected.render(100))


def test_owned_observation_rejects_truth_extra_wrong_dtype_and_nonfinite():
    observation, _ = SinePitchEnv(ObservationMode.WAVEFORM).reset(seed=9)
    for changed in (
        {**observation, "source_pitch_cents": 6000},
        {**observation, "waveform": observation["waveform"].astype(np.float64)},
        {**observation, "waveform": np.full(262144, np.nan, np.float32)},
        {**observation, "target_note": 12},
    ):
        with pytest.raises(ValueError):
            owned_observation(changed, ObservationMode.WAVEFORM)
