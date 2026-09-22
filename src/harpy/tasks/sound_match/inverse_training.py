"""Streaming CPU training, with checkpoint selection through ordinary audio episodes."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import numpy as np

from harpy.tasks.sound_match.inverse_artifacts import (
    SUMMARY_SCHEMA_ID,
    TRAINING_SCHEMA_ID,
    dataset_binding,
    publish_inverse_artifact,
    require_torch,
    state_sha256,
    training_configuration,
    training_membership,
)
from harpy.tasks.sound_match.inverse_features import (
    architecture_configuration,
    conditioning,
    decoder_configuration,
    feature_configuration,
    supervision_configuration,
    supervision_targets,
)
from harpy.tasks.sound_match.models import ActorSpec, EpisodeSpec, _integer, _plain


def _validation(model, entries: tuple, *, seed: int, epoch: int) -> dict:
    from harpy.tasks.sound_match.datasets import CELL_NAMES
    from harpy.tasks.sound_match.inverse import InverseActor
    from harpy.tasks.sound_match.runner import evaluate

    episodes = tuple(EpisodeSpec.from_dict(entry["episode"]) for entry in entries)
    actor = ActorSpec(
        "inverse-validation",
        lambda: InverseActor(model),
        metadata={
            "kind": "training-epoch-validation",
            "seed": seed,
            "epoch": epoch,
            "state_sha256": state_sha256(model.state_dict()),
        },
    )
    result = evaluate(episodes, (actor,), cache_max_audio_bytes=0)
    expected_targets = {
        entry["episode"]["episode_id"]: entry["target_audio_sha256"] for entry in entries
    }
    observed_targets = {target["episode_id"]: target["audio_sha256"] for target in result.targets}
    if len(result.targets) != len(entries) or observed_targets != expected_targets:
        raise ValueError("validation target audio differs from its frozen dataset identity")
    cells: dict[str, list[float]] = {name: [] for name in CELL_NAMES}
    by_id = {entry["episode"]["episode_id"]: entry["cell"] for entry in entries}
    for record in result.records:
        if record.status != "submitted" or record.metrics["submitted_distance"] is None:
            raise ValueError(
                "a missing or unsubmitted validation episode makes a checkpoint ineligible"
            )
        cells[by_id[record.episode_id]].append(record.metrics["submitted_distance"])
    if any(not values for values in cells.values()):
        raise ValueError("validation must retain every declared voice-count/task-mask cell")
    rows = {
        name: {"examples": len(values), "mean_distance": sum(values) / len(values)}
        for name, values in cells.items()
    }
    return {
        "macro_distance": sum(rows[name]["mean_distance"] for name in CELL_NAMES) / len(CELL_NAMES),
        "cells": rows,
        "examples": len(episodes),
    }


def _peak_rss_mib() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def train_inverse(
    output: Path,
    *,
    seed: int = 0,
    profile: str = "quick",
    device: str = "cpu",
    progress=None,
    cache_directory: Path | None = None,
):
    if not isinstance(output, Path):
        raise ValueError("output must be a Path")
    _integer(seed, "training seed")
    if seed >= 2**32:
        raise ValueError("training seed must be below 2**32")
    if device != "cpu":
        raise ValueError("inverse training currently supports CPU only; no device fallback")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    configuration = training_configuration(profile)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"inverse output already exists: {output}")
    torch = require_torch()
    from harpy.tasks.sound_match.datasets import SELECTION_CRITERION
    from harpy.tasks.sound_match.inverse_cache import cache_identity, prepare_feature_cache
    from harpy.tasks.sound_match.inverse_network import InverseNetwork, supervised_loss
    from harpy.tasks.sound_match.runner import _source_identity

    train, validation, train_entries, validation_entries = training_membership(profile)
    training_episodes = tuple(EpisodeSpec.from_dict(entry["episode"]) for entry in train_entries)
    source = _source_identity()
    started = time.perf_counter()
    training = {
        "schema_id": TRAINING_SCHEMA_ID,
        "seed": seed,
        "configuration": configuration,
        "architecture": architecture_configuration(),
        "features": feature_configuration(),
        "decoder": decoder_configuration(),
        "supervision": supervision_configuration(),
        "datasets": {
            "train": dataset_binding(train, train_entries),
            "validation": dataset_binding(validation, validation_entries),
        },
        "selection": _plain(SELECTION_CRITERION),
        "source": source,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "device": "cpu",
            "torch_threads": 1,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    cache = prepare_feature_cache(
        output.parent / ".inverse-features" if cache_directory is None else cache_directory,
        train_entries,
        cache_identity(training["datasets"]["train"], source, training["runtime"]),
        progress=progress,
    )
    training["feature_cache"] = cache.identity
    cached_features = cache.open()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = InverseNetwork()
            initial_hash = state_sha256(model.state_dict())
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=configuration["learning_rate"],
                weight_decay=configuration["weight_decay"],
            )
            rng = np.random.default_rng(seed)
            best_state, best_distance, best_epoch = None, float("inf"), None
            history = []
            total_updates = 0
            for epoch in range(1, configuration["epochs"] + 1):
                model.train()
                order = rng.permutation(len(training_episodes))
                total_loss, examples, updates = 0.0, 0, 0
                for start in range(0, len(order), configuration["batch_size"]):
                    batch = tuple(
                        training_episodes[int(index)]
                        for index in order[start : start + configuration["batch_size"]]
                    )
                    features = np.asarray(
                        cached_features[order[start : start + configuration["batch_size"]]]
                    )
                    conditions = np.stack(
                        [conditioning(item.initial_patch, item.editable_fields) for item in batch]
                    )
                    targets = {
                        name: torch.from_numpy(values)
                        for name, values in supervision_targets(batch).items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    predictions = model(
                        torch.from_numpy(features)[:, None], torch.from_numpy(conditions)
                    )
                    loss = supervised_loss(predictions, targets)
                    if not torch.isfinite(loss):
                        raise ValueError("inverse optimization produced a nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        configuration["gradient_norm_limit"],
                        error_if_nonfinite=True,
                    )
                    optimizer.step()
                    total_loss += float(loss.detach()) * len(batch)
                    examples += len(batch)
                    updates += 1
                model.eval()
                validation_scores = _validation(model, validation_entries, seed=seed, epoch=epoch)
                if _source_identity() != source:
                    raise RuntimeError("package source changed during inverse training")
                row = {
                    "epoch": epoch,
                    "mean_training_loss": total_loss / examples,
                    "optimizer_updates": updates,
                    "validation": validation_scores,
                }
                history.append(row)
                total_updates += updates
                if validation_scores["macro_distance"] < best_distance:
                    best_distance = validation_scores["macro_distance"]
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                if progress is not None:
                    progress(_plain(row))
            summary = {
                "schema_id": SUMMARY_SCHEMA_ID,
                "epochs": history,
                "selected_epoch": best_epoch,
                "selected_validation_macro_distance": best_distance,
                "initial_state_sha256": initial_hash,
                "selected_state_sha256": state_sha256(best_state),
                "total_optimizer_updates": total_updates,
                "training_examples": len(training_episodes),
                "validation_examples": len(validation_entries),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_rss_mib": _peak_rss_mib(),
            }
            return publish_inverse_artifact(output, best_state, training, summary)
    finally:
        del cached_features
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
