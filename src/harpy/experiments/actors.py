"""Explicit built-in estimators and episode-local control policies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from harpy.analysis import analyze
from harpy.envs.baselines import RandomPolicy, RewardSearchPolicy, oracle_plan
from harpy.envs.models import (
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    ControlState,
    ObservationMode,
    PitchAction,
)
from harpy.envs.observations import owned_observation
from harpy.envs.planning import minimum_action_plan
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, LOG_FREQUENCY_GRID_HZ
from harpy.experiments.models import ActorSpec, Decision

COMMITTED_CONTROLLER = "harpy-committed-zero-tolerance-plan-v1"
REPLANNING_CONTROLLER = "harpy-replanned-zero-tolerance-plan-v1"
FEASIBLE_DECODER = "harpy-public-source-feasible-five-cent-argmax-v1"
_FFT_GRID_CENTS = np.arange(1_100, 10_901, 5, dtype=np.float64)
_FFT_CELL_EDGES_HZ = np.append(
    LOG_FREQUENCY_GRID_HZ / 2.0 ** (2.5 / 1_200.0),
    LOG_FREQUENCY_GRID_HZ[-1] * 2.0 ** (2.5 / 1_200.0),
)
_FFT_GRID_CENTS.setflags(write=False)
_FFT_CELL_EDGES_HZ.setflags(write=False)


def _controls(observation: Mapping[str, object]) -> ControlState:
    return ControlState(*(int(value) for value in observation["controls"]))


def _plan(estimate: float, observation: Mapping[str, object]) -> tuple[PitchAction, ...]:
    controls = _controls(observation)
    target = 4_800 + 100 * int(observation["target_note"])
    base_error = round(estimate) - controls.offset_cents - target
    return minimum_action_plan(min(2_400, max(-2_400, base_error)), controls, tolerance_cents=0)


def _feasible_interval(observation: Mapping[str, object]) -> tuple[int, int]:
    offset = _controls(observation).offset_cents
    return (
        (SOURCE_MIN_CENTS + offset - 1_100 + 2) // 5,
        (SOURCE_MAX_CENTS + offset - 1_100 + 2) // 5 + 1,
    )


class _SpectrumEstimator:
    def estimate(self, observation: Mapping[str, object]) -> float:
        first, stop = _feasible_interval(observation)
        spectrum = observation["spectrum"]
        index = first + int(np.argmax(spectrum[first:stop]))
        return float(1_100 + 5 * index)


class _WaveformFFTEstimator:
    def estimate(self, observation: Mapping[str, object]) -> float:
        """Fit a feasible Hann-FFT peak, retaining explicit five-cent quantization."""
        first, stop = _feasible_interval(observation)
        analysis = analyze(observation["waveform"], 48_000, GYM_ANALYSIS_CONFIG)
        if not analysis.has_signal:
            return float(_FFT_GRID_CENTS[first])
        frequencies = analysis.spectrum_frequency_hz
        levels = analysis.spectrum_level_dbfs
        eligible = np.flatnonzero(
            (frequencies >= _FFT_CELL_EDGES_HZ[first]) & (frequencies <= _FFT_CELL_EDGES_HZ[stop])
        )
        winning = int(eligible[int(np.argmax(levels[eligible]))])
        offset = 0.0
        if 0 < winning < levels.size - 1:
            left, center, right = levels[winning - 1 : winning + 2]
            curvature = left - 2.0 * center + right
            if curvature < 0.0:
                offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
        frequency = frequencies[winning] + offset * 48_000 / GYM_ANALYSIS_CONFIG.fft_frames
        coordinate = 6_900.0 + 1_200.0 * np.log2(frequency / 440.0)
        # np.argmin selects the lower coordinate when two five-cent classes tie.
        index = first + int(np.argmin(np.abs(_FFT_GRID_CENTS[first:stop] - coordinate)))
        return float(_FFT_GRID_CENTS[index])


class _PitchEstimator:
    def __init__(self, model: object, device: str) -> None:
        from harpy.learning.dependencies import require_pitch_dependencies
        from harpy.learning.pitch_network import validate_pitch_estimator_model

        self._torch = require_pitch_dependencies().torch
        validate_pitch_estimator_model(model)
        self._model = model.to(self._torch.device(device))
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def estimate(self, observation: Mapping[str, object]) -> float:
        spectrum = self._torch.from_numpy(observation["spectrum"]).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(spectrum)
        if (
            not isinstance(logits, self._torch.Tensor)
            or logits.shape != (1, 1_961)
            or logits.dtype != self._torch.float32
            or logits.device != self._device
            or not self._torch.isfinite(logits).all()
        ):
            raise ValueError("pitch estimator must return finite float32 logits of shape (1, 1961)")
        first, stop = _feasible_interval(observation)
        index = first + int(self._torch.argmax(logits[:, first:stop], dim=1).item())
        return float(1_100 + 5 * index)


class EstimatorPlanner:
    """Estimate once and commit, or explicitly re-estimate before every action."""

    def __init__(
        self,
        estimator: object,
        *,
        replan: bool = False,
        observation_mode: ObservationMode = ObservationMode.SPECTRUM,
    ) -> None:
        if not isinstance(observation_mode, ObservationMode):
            raise ValueError("observation_mode must be an ObservationMode")
        self._estimator = estimator
        self._replan = replan
        self._observation_mode = observation_mode
        self._actions: tuple[PitchAction, ...] | None = None
        self._cursor = 0

    def decide(self, observation: Mapping[str, object]) -> Decision:
        observation = owned_observation(observation, self._observation_mode)
        estimate = None
        count = 0
        if self._actions is None or self._replan:
            estimate = self._estimator.estimate(observation)
            self._actions = _plan(estimate, observation)
            self._cursor = 0
            count = 1
        if self._cursor >= len(self._actions):
            raise RuntimeError("episode controller was reused after its plan ended")
        action = self._actions[self._cursor]
        self._cursor += 1
        return Decision(action, estimate, count)

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        return self.decide(observation).action


def spectrum_peak_actor_spec(*, replan: bool = False) -> ActorSpec:
    if type(replan) is not bool:
        raise ValueError("replan must be a bool")
    return ActorSpec(
        name="spectrum-peak-replan" if replan else "spectrum-peak",
        factory=lambda: EstimatorPlanner(_SpectrumEstimator(), replan=replan),
        observation_mode=ObservationMode.SPECTRUM,
        estimator="harpy-spectrum-peak-v1",
        decoder=FEASIBLE_DECODER,
        controller=REPLANNING_CONTROLLER if replan else COMMITTED_CONTROLLER,
    )


def waveform_fft_actor_spec() -> ActorSpec:
    """Estimate once from the waveform FFT and execute the shared committed plan."""
    return ActorSpec(
        name="waveform-fft",
        factory=lambda: EstimatorPlanner(
            _WaveformFFTEstimator(), observation_mode=ObservationMode.WAVEFORM
        ),
        observation_mode=ObservationMode.WAVEFORM,
        estimator="harpy-hann-quadratic-fft-v1",
        decoder="harpy-feasible-fft-peak-nearest-five-cent-v1",
        controller=COMMITTED_CONTROLLER,
    )


class _Oracle:
    def __init__(self) -> None:
        self._actions = None
        self._cursor = 0

    def decide(self, observation: Mapping[str, object]) -> Decision:
        count = int(self._actions is None)
        if self._actions is None:
            self._actions = oracle_plan(observation)
        action = self._actions[self._cursor]
        self._cursor += 1
        return Decision(action, inference_count=count)


def oracle_actor_spec() -> ActorSpec:
    return ActorSpec(
        "oracle",
        _Oracle,
        ObservationMode.ORACLE,
        "public-exact-coordinate",
        "identity",
        "harpy-committed-five-cent-tolerance-plan-v1",
    )


class _RewardSearch:
    def __init__(self) -> None:
        self._policy = RewardSearchPolicy()
        self._reward = None
        self._info = None

    def feedback(self, reward: float, info: Mapping[str, object]) -> None:
        self._reward = reward
        self._info = dict(info)

    def decide(self, observation: Mapping[str, object]) -> Decision:
        return Decision(self._policy.act(observation, self._reward, self._info))


def reward_search_actor_spec() -> ActorSpec:
    return ActorSpec(
        "reward-search",
        _RewardSearch,
        ObservationMode.REWARD_ONLY,
        "none",
        "none",
        "harpy-reward-search-v1",
    )


class _Random:
    def __init__(self, seed: int) -> None:
        self._policy = RandomPolicy(np.random.default_rng(seed))

    def decide(self, observation: Mapping[str, object]) -> Decision:
        return Decision(self._policy.act(observation, None, None))


def random_actor_spec(*, seed: int = 0) -> ActorSpec:
    return ActorSpec(
        "random",
        lambda: _Random(seed),
        ObservationMode.SPECTRUM,
        "none",
        "none",
        "harpy-public-legal-random-v1",
        seed=seed,
    )


def _model_actor_spec(
    artifact_path: Path, *, device: str, trainer: str, name: str | None
) -> ActorSpec:
    from harpy.experiments.artifacts import load_artifact

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    artifact = load_artifact(artifact_path)
    if artifact.manifest.trainer != trainer:
        raise ValueError(f"artifact must contain a {trainer} model")
    artifact.verify_unchanged()
    model = artifact.load_model(device=device)
    artifact.verify_unchanged()
    if trainer == "pitch":
        pitch_estimator = _PitchEstimator(model, device)

        def factory():
            return EstimatorPlanner(pitch_estimator)

        estimator = "harpy-sine-pitch-estimator-v1"
        decoder = FEASIBLE_DECODER
        controller = COMMITTED_CONTROLLER
    else:
        from harpy.learning.actors import PPOActor

        def factory():
            return PPOActor(model)

        estimator = "harpy-ppo-policy-v1"
        decoder = "deterministic-policy"
        controller = "harpy-direct-action-policy-v1"
    return ActorSpec(
        name=f"{trainer}-{artifact.manifest.seed}" if name is None else name,
        factory=factory,
        observation_mode=ObservationMode.SPECTRUM,
        estimator=estimator,
        decoder=decoder,
        controller=controller,
        artifact_paths=(artifact_path,),
        seed=artifact.manifest.seed,
        device=device,
        artifact_provenance={**artifact.provenance(), "evaluation_device": device},
        _verify_artifact=artifact.verify_unchanged,
    )


def pitch_actor_spec(
    artifact_path: Path, *, device: str = "cpu", name: str | None = None
) -> ActorSpec:
    """Load one model, then create fresh committed controllers over shared weights."""
    return _model_actor_spec(artifact_path, device=device, trainer="pitch", name=name)


def ppo_actor_spec(
    artifact_path: Path, *, device: str = "cpu", name: str | None = None
) -> ActorSpec:
    """Load one PPO policy and adapt its deterministic inference per episode."""
    return _model_actor_spec(artifact_path, device=device, trainer="ppo", name=name)


__all__ = [
    "EstimatorPlanner",
    "oracle_actor_spec",
    "pitch_actor_spec",
    "ppo_actor_spec",
    "random_actor_spec",
    "reward_search_actor_spec",
    "spectrum_peak_actor_spec",
    "waveform_fft_actor_spec",
]
