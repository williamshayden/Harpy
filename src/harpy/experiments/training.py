"""Training entry points with explicit data boundaries and create-only output."""

from __future__ import annotations

import hashlib
import platform
from dataclasses import asdict
from pathlib import Path

import gymnasium
import numpy as np

from harpy.experiments.artifacts import (
    METADATA_SCHEMA_ID,
    ExperimentManifest,
    LoadedExperimentArtifact,
    _load_model_file,
    load_artifact,
    validate_training_request,
)
from harpy.learning.artifacts import FileRecord, canonical_json_bytes
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.models import EpisodeSpec, ProfileName
from harpy.learning.pitch_data import PITCH_SPLIT_DIGEST_SHA256, pitch_coordinate_split


def ppo_training_episode(*, seed: int, index: int) -> EpisodeSpec:
    """Draw reset sources only from the pitch training partition.

    Subsequent action trajectories can visit validation or holdout coordinates;
    this constrains initial conditions, not all audio encountered by PPO.
    """
    from harpy.envs.models import TARGET_NOTE_COUNT

    validate_training_request("ppo", "smoke", seed, "cpu")
    if type(index) is not int or index < 0:
        raise ValueError("episode index must be a non-negative integer")
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 917]))
    return EpisodeSpec(
        target_note_index=int(rng.integers(TARGET_NOTE_COUNT)),
        source_pitch_cents=int(rng.choice(pitch_coordinate_split().training_coordinates)),
    )


def _source() -> dict:
    from harpy.learning.artifacts import (
        _capture_package_source,
        _source_to_document,
        capture_source_status,
    )

    try:
        anchor = Path(__file__).resolve()
        return {
            "status": "known",
            "identity": _source_to_document(capture_source_status(anchor)),
            "package_sha256": _capture_package_source(anchor).package_sha256,
        }
    except (OSError, ValueError) as error:
        return {"status": "unknown", "reason": str(error) or type(error).__name__}


def _train_pitch(profile: ProfileName, seed: int, device: str, output: Path) -> tuple[dict, dict]:
    from harpy.learning.dependencies import require_pitch_dependencies
    from harpy.learning.pitch import (
        PITCH_PROFILE_CONFIGS,
        build_pitch_coordinate_datasets,
        save_pitch_estimator_model,
        train_pitch_estimator,
    )

    config = PITCH_PROFILE_CONFIGS[profile]
    training, validation = build_pitch_coordinate_datasets(profile)
    result = train_pitch_estimator(
        training,
        validation,
        profile=config,
        seed=seed,
        device=require_pitch_dependencies().torch.device(device),
    )
    save_pitch_estimator_model(output / "model.pt", result.model)
    configuration = {
        "optimizer": asdict(config),
        "objective": "cross_entropy_nearest_5_cent_class",
        "architecture": "location_preserving_conv_2497_parameters",
        "coordinate_split_sha256": PITCH_SPLIT_DIGEST_SHA256,
        "training_coordinates": list(training.coordinates),
        "validation_coordinates": list(validation.coordinates),
        "checkpoint_selection": "validation_within_5c_then_1c_then_mae_then_loss_then_epoch",
    }
    # Historical cohort eligibility is not a property of this independent workflow.
    configuration["optimizer"].pop("eligible_for_aggregate")
    summary = asdict(result.summary)
    summary["history"] = list(summary["history"])
    return configuration, summary


def _train_ppo(profile: ProfileName, seed: int, device: str, output: Path) -> tuple[dict, dict]:
    from harpy.learning.envs import ScheduledEpisodeEnv, make_cached_sine_pitch_env
    from harpy.learning.models import PROFILE_CONFIGS
    from harpy.learning.ppo import make_ppo_vec_env, save_ppo_model, train_ppo

    config = PROFILE_CONFIGS[profile].ppo
    cache = SpectrumEvidenceCache()
    env = make_ppo_vec_env(
        lambda: ScheduledEpisodeEnv(
            make_cached_sine_pitch_env(cache),
            episode_at=lambda index: ppo_training_episode(seed=seed, index=index),
        )
    )
    try:
        model, summary = train_ppo(env, config=config, seed=seed, device=device)
        save_ppo_model(output / "model.zip", model)
    finally:
        env.close()
    configuration = {
        "optimizer": asdict(config),
        "architecture": "harpy_sine_policy_75816_parameters",
        "coordinate_split_sha256": PITCH_SPLIT_DIGEST_SHA256,
        "initial_source_coordinates": list(pitch_coordinate_split().training_coordinates),
        "initial_source_sampling": "indexed_uniform_training_partition_v1",
        "trajectory_scope": "actions may visit validation and holdout coordinates",
        "checkpoint_selection": "final_training_step_no_performance_gate",
        "budget_exhaustion_bootstrap": False,
    }
    return configuration, asdict(summary)


def train(
    *,
    trainer: str = "pitch",
    profile: str = "smoke",
    seed: int = 0,
    device: str = "cpu",
    output: Path,
) -> LoadedExperimentArtifact:
    """Fit and save a new artifact; existing output paths are never overwritten.

    A failed run may leave an incomplete directory without a manifest. Model
    quality is measured separately by evaluation and never gates PPO persistence.
    """
    from harpy.learning.dependencies import (
        configure_deterministic_cuda_environment,
        require_pitch_dependencies,
        require_training_dependencies,
    )

    validate_training_request(trainer, profile, seed, device)
    if not isinstance(output, Path):
        raise ValueError("output must be a Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if device == "cuda":
        configure_deterministic_cuda_environment()
    stack = require_pitch_dependencies() if trainer == "pitch" else require_training_dependencies()
    if device == "cuda" and not stack.torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    source = _source()
    output.mkdir(parents=True, exist_ok=False)
    configuration, summary = (_train_pitch if trainer == "pitch" else _train_ppo)(
        ProfileName(profile), seed, device, output
    )
    _load_model_file(
        output / ("model.pt" if trainer == "pitch" else "model.zip"), trainer=trainer, device=device
    )
    if _source() != source:
        source = {"status": "unknown", "reason": "package source changed during training"}
    metadata = {
        "schema_id": METADATA_SCHEMA_ID,
        "trainer": trainer,
        "profile": profile,
        "seed": seed,
        "device": device,
        "configuration": configuration,
        "summary": summary,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "gymnasium": gymnasium.__version__,
            "torch": stack.torch_version,
            "stable_baselines3": getattr(stack, "stable_baselines3_version", None),
            "cuda_runtime": stack.torch.version.cuda if device == "cuda" else None,
            "device_description": stack.torch.cuda.get_device_name(0)
            if device == "cuda"
            else platform.processor() or "CPU",
        },
    }
    with (output / "metadata.json").open("xb") as stream:
        stream.write(canonical_json_bytes(metadata))
    files = tuple(
        FileRecord(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(output.iterdir())
    )
    manifest = ExperimentManifest(
        trainer, profile, seed, device, files, canonical_json_bytes(source)
    )
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical_json_bytes(manifest.to_document()))
    return load_artifact(output)


__all__ = ["ppo_training_episode", "train"]
