"""Public Gymnasium registration for the sine-pitch environments."""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Iterator

import gymnasium
import pytest
from gymnasium.envs.registration import registry
from gymnasium.utils.env_checker import check_env

from harpy.envs.models import ObservationMode

ENTRY_POINT = "harpy.envs.sine_pitch:SinePitchEnv"
EXPECTED_ENVIRONMENTS = {
    "Harpy/SinePitch-v0": "spectrum",
    "Harpy/SinePitchOracle-v0": "oracle",
    "Harpy/SinePitchRewardOnly-v0": "reward_only",
}


@pytest.mark.parametrize(("env_id", "raw_mode"), EXPECTED_ENVIRONMENTS.items())
def test_raw_string_mode_registration_is_rejected_atomically(
    isolated_harpy_registrations: None,
    env_id: str,
    raw_mode: str,
) -> None:
    """StrEnum/string equality must not accept a spec the constructor will reject."""
    gymnasium.register(
        env_id,
        entry_point=ENTRY_POINT,
        kwargs={"observation_mode": raw_mode},
    )
    registry_before = {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS}

    with pytest.raises(RuntimeError, match=rf"{env_id}.*already registered"):
        importlib.import_module("harpy.envs")

    assert {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS} == (
        registry_before
    )


@pytest.mark.parametrize(("env_id", "mode_name"), EXPECTED_ENVIRONMENTS.items())
def test_time_limited_registration_is_rejected_atomically(
    isolated_harpy_registrations: None,
    env_id: str,
    mode_name: str,
) -> None:
    """A matching mode with a TimeLimit still changes the frozen 64-call contract."""
    gymnasium.register(
        env_id,
        entry_point=ENTRY_POINT,
        kwargs={"observation_mode": ObservationMode(mode_name)},
        max_episode_steps=7,
    )
    registry_before = {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS}

    with pytest.raises(RuntimeError, match=rf"{env_id}.*already registered"):
        importlib.import_module("harpy.envs")

    assert {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS} == (
        registry_before
    )


def test_exact_enum_registration_without_time_limit_is_idempotent(
    isolated_harpy_registrations: None,
) -> None:
    """Only the exact constructor-ready spec is compatible and must retain identity."""
    for env_id, mode_name in EXPECTED_ENVIRONMENTS.items():
        gymnasium.register(
            env_id,
            entry_point=ENTRY_POINT,
            kwargs={"observation_mode": ObservationMode(mode_name)},
            max_episode_steps=None,
        )
    original_specs = {env_id: registry[env_id] for env_id in EXPECTED_ENVIRONMENTS}

    envs = importlib.import_module("harpy.envs")
    envs.register_envs()

    assert all(registry[env_id] is original_specs[env_id] for env_id in EXPECTED_ENVIRONMENTS)


@pytest.fixture
def isolated_harpy_registrations() -> Iterator[None]:
    """Restore the process-global Gymnasium registry after each test."""
    prior_specs = {env_id: registry.get(env_id) for env_id in EXPECTED_ENVIRONMENTS}
    _remove_harpy_envs_module()
    for env_id in EXPECTED_ENVIRONMENTS:
        registry.pop(env_id, None)

    yield

    _remove_harpy_envs_module()
    for env_id, prior_spec in prior_specs.items():
        registry.pop(env_id, None)
        if prior_spec is not None:
            registry[env_id] = prior_spec


def test_import_registers_each_public_environment_once(
    isolated_harpy_registrations: None,
) -> None:
    """Deleting registration from package import would leave Gym users unable to make an env."""
    envs = importlib.import_module("harpy.envs")

    assert set(envs.__all__) == {
        "ControlState",
        "EpisodeResult",
        "ObservationMode",
        "PitchAction",
        "SinePitchEnv",
        "register_envs",
    }
    assert set(EXPECTED_ENVIRONMENTS).issubset(registry)

    for env_id, observation_mode in EXPECTED_ENVIRONMENTS.items():
        spec = registry[env_id]
        assert spec.entry_point == ENTRY_POINT
        assert spec.kwargs == {"observation_mode": envs.ObservationMode(observation_mode)}
        assert spec.max_episode_steps is None


def test_registration_and_module_reload_are_idempotent(
    isolated_harpy_registrations: None,
) -> None:
    """Re-registering matching specs must not replace or warn about existing registrations."""
    envs = importlib.import_module("harpy.envs")
    original_specs = {env_id: registry[env_id] for env_id in EXPECTED_ENVIRONMENTS}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        envs.register_envs()
        reloaded_envs = importlib.reload(envs)

    assert reloaded_envs is envs
    assert {env_id: registry[env_id] for env_id in EXPECTED_ENVIRONMENTS} == original_specs
    assert all(registry[env_id] is original_specs[env_id] for env_id in EXPECTED_ENVIRONMENTS)


def test_made_environments_use_the_registered_observation_mode(
    isolated_harpy_registrations: None,
) -> None:
    """A wrong registration kwarg would produce the wrong agent observation contract."""
    importlib.import_module("harpy.envs")

    for env_id, expected_mode in EXPECTED_ENVIRONMENTS.items():
        env = gymnasium.make(env_id, disable_env_checker=True)
        try:
            assert env.unwrapped.observation_mode.value == expected_mode
            assert env.spec is not None
            assert env.spec.max_episode_steps is None
        finally:
            env.close()


def test_registered_environments_pass_the_gymnasium_checker_without_warnings(
    isolated_harpy_registrations: None,
) -> None:
    """A registration that wraps or constructs an invalid environment breaks Gym tooling."""
    importlib.import_module("harpy.envs")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for env_id in EXPECTED_ENVIRONMENTS:
            env = gymnasium.make(env_id, disable_env_checker=True).unwrapped
            check_env(env, skip_render_check=True)


def test_foreign_registration_collision_raises_without_overwriting(
    isolated_harpy_registrations: None,
) -> None:
    """Silently replacing another package's environment would corrupt the shared registry."""
    env_id = "Harpy/SinePitch-v0"
    gymnasium.register(env_id, entry_point="foreign_package:ForeignEnvironment")

    with pytest.raises(RuntimeError, match=r"Harpy/SinePitch-v0.*already registered"):
        importlib.import_module("harpy.envs")

    assert registry[env_id].entry_point == "foreign_package:ForeignEnvironment"


def test_late_foreign_collision_does_not_partially_register(
    isolated_harpy_registrations: None,
) -> None:
    """A later collision must not leave earlier Harpy IDs registered as partial state."""
    env_id = "Harpy/SinePitchRewardOnly-v0"
    gymnasium.register(env_id, entry_point="foreign_package:ForeignEnvironment")
    registry_before = {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS}

    with pytest.raises(RuntimeError, match=r"Harpy/SinePitchRewardOnly-v0.*already registered"):
        importlib.import_module("harpy.envs")

    assert {target_id: registry.get(target_id) for target_id in EXPECTED_ENVIRONMENTS} == (
        registry_before
    )


def _remove_harpy_envs_module() -> None:
    sys.modules.pop("harpy.envs", None)
    harpy = sys.modules.get("harpy")
    if harpy is not None and hasattr(harpy, "envs"):
        delattr(harpy, "envs")
