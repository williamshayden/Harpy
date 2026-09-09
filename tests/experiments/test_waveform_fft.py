"""The optional waveform baseline preserves the frozen research estimator math."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from harpy.envs.models import ControlState, ObservationMode, PitchAction
from harpy.envs.robustness import (
    ROBUSTNESS_CONDITIONS,
    RobustnessEvidenceCache,
    make_robustness_env,
)
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG
from harpy.experiments.actors import (
    COMMITTED_CONTROLLER,
    EstimatorPlanner,
    waveform_fft_actor_spec,
)


@pytest.fixture(scope="module")
def prototype():
    path = Path(__file__).resolve().parents[2] / "research/spectrum_preservation/encoders.py"
    spec = importlib.util.spec_from_file_location("frozen_spectrum_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observation(samples, controls=None):
    controls = ControlState() if controls is None else controls
    return {
        "waveform": samples,
        "controls": np.array(
            [controls.octaves, controls.semitones, controls.cents], dtype=np.int16
        ),
        "target_note": np.int64(12),
        "steps_remaining": np.int64(64),
    }


@pytest.mark.parametrize("condition", ROBUSTNESS_CONDITIONS, ids=lambda item: item.id)
@pytest.mark.parametrize(
    ("source", "controls"),
    [
        (4800, ControlState(-2, -12, -100)),
        (7200, ControlState(2, 12, 100)),
        (4800, ControlState(0, 0, -2)),
        (7200, ControlState(0, 0, 2)),
        (6857, ControlState()),
    ],
)
def test_real_estimates_match_frozen_prototype_at_bounds_and_under_nuisance(
    prototype, source, controls, condition
):
    evidence = RobustnessEvidenceCache(max_bytes=0).evidence(
        source + controls.offset_cents, condition, 14708078973187757233
    )
    obs = observation(evidence.waveform, controls)
    before = obs["waveform"].copy()
    direct = waveform_fft_actor_spec().factory().decide(obs)
    frozen = prototype.WaveformActor("quadratic-fft").decide(obs)
    assert direct == frozen
    assert direct.estimated_candidate_cents % 5 == 0
    assert direct.inference_count == 1
    assert np.array_equal(obs["waveform"], before)


@pytest.mark.parametrize(
    "controls", [ControlState(), ControlState(-2, -12, -100), ControlState(2, 12, 100)]
)
def test_silence_selects_first_feasible_five_cent_cell(prototype, controls):
    obs = observation(np.zeros(GYM_ANALYSIS_CONFIG.fft_frames, dtype=np.float32), controls)
    decision = waveform_fft_actor_spec().factory().decide(obs)
    assert decision == prototype.WaveformActor("quadratic-fft").decide(obs)
    assert decision.estimated_candidate_cents == 4800 + controls.offset_cents


def test_exact_nearest_cell_tie_keeps_the_lower_coordinate(monkeypatch, prototype):
    import harpy.experiments.actors as actors

    frequency = 440.0 * 2.0 ** ((6002.5 - 6900.0) / 1200.0)
    assert 6900.0 + 1200.0 * np.log2(frequency / 440.0) == 6002.5
    analysis = SimpleNamespace(
        has_signal=True,
        spectrum_frequency_hz=np.array([frequency - 1, frequency, frequency + 1]),
        spectrum_level_dbfs=np.array([-30.0, -10.0, -30.0]),
    )
    monkeypatch.setattr(actors, "analyze", lambda *args: analysis)
    monkeypatch.setattr(prototype, "analyze", lambda *args: analysis)
    obs = observation(np.zeros(GYM_ANALYSIS_CONFIG.fft_frames, dtype=np.float32))
    decision = waveform_fft_actor_spec().factory().decide(obs)
    assert decision.estimated_candidate_cents == 6000.0
    assert decision == prototype.WaveformActor("quadratic-fft").decide(obs)


def test_fresh_committed_controllers_match_complete_prototype_trajectories(prototype):
    spec = waveform_fft_actor_spec()
    assert spec.name == "waveform-fft"
    assert spec.observation_mode is ObservationMode.WAVEFORM
    assert spec.estimator == "harpy-hann-quadratic-fft-v1"
    assert spec.decoder == "harpy-feasible-fft-peak-nearest-five-cent-v1"
    assert spec.controller == COMMITTED_CONTROLLER
    first = spec.factory()
    second = spec.factory()
    assert first is not second
    for actor, source in ((first, 6064), (second, 6857)):
        frozen = prototype.WaveformActor("quadratic-fft")
        with make_robustness_env(
            ROBUSTNESS_CONDITIONS[6], nuisance_seed=17, observation_mode=ObservationMode.WAVEFORM
        ) as env:
            obs, _ = env.reset(options={"source_pitch_cents": source, "target_note_index": 12})
            decisions = []
            while True:
                decision = actor.decide(obs)
                assert decision == frozen.decide(obs)
                decisions.append(decision)
                obs, _, done, truncated, _ = env.step(decision.action)
                if done or truncated:
                    break
            assert decisions[-1].action is PitchAction.SUBMIT
            assert sum(item.inference_count for item in decisions) == 1
            assert all(item.estimated_candidate_cents is None for item in decisions[1:])
            with pytest.raises(RuntimeError, match="plan ended"):
                actor.decide(obs)


@pytest.mark.parametrize("bad", ["dtype", "shape", "nan", "inf", "hidden", "controls"])
def test_waveform_actor_validates_the_public_observation_before_estimation(bad):
    obs = observation(np.zeros(GYM_ANALYSIS_CONFIG.fft_frames, dtype=np.float32))
    if bad == "dtype":
        obs["waveform"] = obs["waveform"].astype(np.float64)
    elif bad == "shape":
        obs["waveform"] = obs["waveform"][:-1]
    elif bad in {"nan", "inf"}:
        obs["waveform"][0] = float(bad)
    elif bad == "hidden":
        obs["source_pitch_cents"] = 6000
    else:
        obs["controls"][0] = 3
    with pytest.raises(ValueError):
        waveform_fft_actor_spec().factory().decide(obs)


def test_planner_requires_an_explicit_observation_mode_enum():
    with pytest.raises(ValueError, match="ObservationMode"):
        EstimatorPlanner(object(), observation_mode="waveform")


def test_waveform_factory_loads_without_optional_learning_or_research_modules():
    script = (
        "import sys; from harpy.experiments.actors import waveform_fft_actor_spec; "
        "waveform_fft_actor_spec().factory(); "
        "assert not any(name == 'torch' or name == 'stable_baselines3' "
        "or name.startswith('research.') for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
