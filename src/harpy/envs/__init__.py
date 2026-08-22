"""Headless Gymnasium environments for Harpy."""

from gymnasium.envs.registration import register as _register
from gymnasium.envs.registration import registry as _registry

from harpy.envs.models import ControlState, EpisodeResult, ObservationMode, PitchAction
from harpy.envs.sine_pitch import SinePitchEnv

_ENTRY_POINT = "harpy.envs.sine_pitch:SinePitchEnv"
_ENVIRONMENTS = {
    "Harpy/SinePitch-v0": ObservationMode.SPECTRUM,
    "Harpy/SinePitchOracle-v0": ObservationMode.ORACLE,
    "Harpy/SinePitchRewardOnly-v0": ObservationMode.REWARD_ONLY,
}


def register_envs() -> None:
    """Register Harpy's Gymnasium environments without overwriting other packages."""
    for env_id, observation_mode in _ENVIRONMENTS.items():
        existing_spec = _registry.get(env_id)
        if existing_spec is not None and not (
            existing_spec.entry_point == _ENTRY_POINT
            and set(existing_spec.kwargs) == {"observation_mode"}
            and isinstance(existing_spec.kwargs["observation_mode"], ObservationMode)
            and existing_spec.kwargs["observation_mode"] is observation_mode
            and existing_spec.max_episode_steps is None
        ):
            raise RuntimeError(
                f"{env_id} is already registered with a different entry point or kwargs"
            )

    for env_id, observation_mode in _ENVIRONMENTS.items():
        if env_id not in _registry:
            _register(
                env_id,
                entry_point=_ENTRY_POINT,
                kwargs={"observation_mode": observation_mode},
            )


register_envs()


__all__ = [
    "ControlState",
    "EpisodeResult",
    "ObservationMode",
    "PitchAction",
    "SinePitchEnv",
    "register_envs",
]
