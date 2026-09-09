"""Named release gates applied to ordinary results, without changing their schema."""

from __future__ import annotations

from pathlib import PurePath

from harpy.envs.models import TerminalReason
from harpy.envs.robustness import ROBUSTNESS_CONDITIONS
from harpy.experiments.actors import COMMITTED_CONTROLLER, FEASIBLE_DECODER
from harpy.experiments.protocols import (
    benchmark_episodes,
    confirmation_episodes,
    validate_protocol_document,
)
from harpy.experiments.results import ExperimentResult

QUALIFICATION_ID = "harpy-v1-supervised-reference-qualification-v1"


def _digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _references(result: ExperimentResult) -> dict[int, object]:
    actors = {}
    for actor in result.actors:
        if actor["estimator"] != "harpy-sine-pitch-estimator-v1":
            continue
        seed = actor["seed"]
        provenance = actor["artifact_provenance"]
        if (
            seed not in (0, 1, 2)
            or seed in actors
            or actor["observation_mode"] != "spectrum"
            or actor["device"] != "cpu"
            or actor["decoder"] != FEASIBLE_DECODER
            or actor["controller"] != COMMITTED_CONTROLLER
            or provenance.get("trainer") != "pitch"
            or provenance.get("profile") != "checkpoint"
            or provenance.get("seed") != seed
            or provenance.get("evaluation_device") != "cpu"
            or not _digest(provenance.get("manifest_sha256"))
            or not _digest(provenance.get("payload_sha256", {}).get("model.pt"))
        ):
            raise ValueError(
                "reference identity must bind a checkpoint seed and CPU committed actor"
            )
        inventory = {
            PurePath(item["path"]).name: item["sha256"] for item in actor["artifact_files"]
        }
        expected = {"manifest.json": provenance["manifest_sha256"], **provenance["payload_sha256"]}
        if len(actor["artifact_files"]) != len(expected) or inventory != expected:
            raise ValueError(
                "qualification artifact inventory must match its manifest and payloads"
            )
        actors[seed] = actor
    if set(actors) != {0, 1, 2}:
        raise ValueError("qualification requires every reference seed 0, 1, and 2 exactly once")
    return actors


def qualify_reference(
    robustness: ExperimentResult, confirmation: ExperimentResult
) -> dict[str, object]:
    """Check fixed clean gates and complete robustness reporting for all seeds.

    This establishes the declared model-performance gates only. Engineering,
    packaging, CUDA training evidence, and publication are separate decisions.
    Robustness measurements have no score threshold.
    """
    if not isinstance(robustness, ExperimentResult) or not isinstance(
        confirmation, ExperimentResult
    ):
        raise ValueError("qualification accepts validated ordinary ExperimentResult values")
    if any(
        robustness.provenance[key] != confirmation.provenance[key] for key in ("source", "renderer")
    ):
        raise ValueError("qualification requires matching evaluator source and rendering identity")
    for result, expected in (
        (robustness, benchmark_episodes()),
        (confirmation, confirmation_episodes()),
    ):
        if result.episodes != expected or result.protocol is None:
            raise ValueError("qualification membership must match its complete frozen protocol")
        validate_protocol_document(result.to_document()["protocol"])
    expected_conditions = tuple(condition.to_document() for condition in ROBUSTNESS_CONDITIONS)
    if tuple(dict(condition) for condition in robustness.conditions) != expected_conditions:
        raise ValueError("qualification requires all eight robustness conditions in protocol order")
    if tuple(dict(condition) for condition in confirmation.conditions) != expected_conditions[:1]:
        raise ValueError("confirmation must use the exact clean condition")
    benchmark_actors, confirmation_actors = _references(robustness), _references(confirmation)
    classical = [
        actor
        for actor in robustness.actors
        if actor["estimator"] == "harpy-spectrum-peak-v1"
        and actor["decoder"] == FEASIBLE_DECODER
        and actor["controller"] == COMMITTED_CONTROLLER
        and actor["observation_mode"] == "spectrum"
    ]
    if len(classical) != 1:
        raise ValueError("qualification requires one matched committed Spectrum Peak baseline")
    rows = []
    for seed in (0, 1, 2):
        left, right = benchmark_actors[seed], confirmation_actors[seed]
        if left["artifact_provenance"] != right["artifact_provenance"]:
            raise ValueError(
                "reference artifact identity changed between benchmark and confirmation"
            )
        for label, result, actor in (
            ("benchmark", robustness, left),
            ("confirmation", confirmation, right),
        ):
            for partition in ("iid", "lower", "upper"):
                members = {
                    episode.id for episode in result.episodes if episode.partition == partition
                }
                records = [
                    record.terminal
                    for record in result.records
                    if record.actor_name == actor["name"]
                    and record.condition_id == "clean"
                    and record.episode_id in members
                ]
                successes = sum(record.submitted_success for record in records)
                invalid = sum(record.invalid_action_count for record in records)
                truncations = sum(
                    record.terminal_reason is TerminalReason.BUDGET_EXHAUSTED for record in records
                )
                rows.append(
                    {
                        "seed": seed,
                        "suite": label,
                        "partition": partition,
                        "episodes": len(records),
                        "successes": successes,
                        "invalid_actions": invalid,
                        "truncations": truncations,
                        "passed": len(records) == successes and invalid == 0 and truncations == 0,
                    }
                )
    return {
        "protocol": QUALIFICATION_ID,
        "performance_qualified": all(row["passed"] for row in rows),
        "shipped_seed": 0,
        "rows": rows,
        "robustness_complete": True,
        "robustness_score_gate": None,
        "engineering_qualification": "separate",
    }
