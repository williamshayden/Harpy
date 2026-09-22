"""Optional inverse estimation and matched-budget refinement for sound matching."""

from __future__ import annotations

from pathlib import Path

from harpy.tasks.sound_match.actors import PatternSearchActor
from harpy.tasks.sound_match.inverse_artifacts import (
    LoadedInverseArtifact,
    load_inverse_artifact,
    require_torch,
)
from harpy.tasks.sound_match.models import ActorSpec, Observation, Propose, Submit
from harpy.tasks.sound_match.objective import score


class InverseActor:
    """One inference, then either explicit submission or the ordinary pattern search."""

    def __init__(self, model, *, refine: bool = False) -> None:
        self.model = model
        self.refine = refine
        self.proposed = False
        self.search = PatternSearchActor() if refine else None

    def decide(self, observation: Observation) -> Propose | Submit:
        if not self.proposed:
            if (
                not observation.editable_fields
                or observation.render_budget_remaining < 1
                or observation.decision_budget_remaining < 2
            ):
                return Submit(model_calls=0)
            from harpy.tasks.sound_match.inverse_network import predict_patch

            if self.search is not None:
                self.search.best_patch = observation.patch
                self.search.best_distance = score(
                    observation.target_audio, observation.candidate_audio
                )["distance"]
                self.search.seen.add(observation.patch)
            patch = predict_patch(
                self.model, observation.target_audio, observation.patch, observation.editable_fields
            )
            self.proposed = True
            return Propose(patch, model_calls=1)
        if self.search is not None:
            return self.search.decide(observation)
        return Submit(model_calls=0)


def inverse_actor_spec(
    path: Path,
    *,
    refine: bool = False,
    name: str | None = None,
    device: str = "cpu",
) -> ActorSpec:
    """Load weights once; fresh controllers share only immutable trained parameters."""
    if type(refine) is not bool:
        raise ValueError("refine must be a boolean")
    artifact = load_inverse_artifact(path)
    model = artifact.load_model(device=device)
    manifest = artifact.manifest

    def factory() -> InverseActor:
        artifact.verify_unchanged()
        return InverseActor(model, refine=refine)

    return ActorSpec(
        name=name or ("inverse-refine" if refine else "inverse"),
        factory=factory,
        metadata={
            "kind": "supervised-inverse",
            "algorithm": "inverse-refine" if refine else "inverse",
            "architecture_id": manifest["architecture_id"],
            "observation_id": manifest["observation_id"],
            "decoder_id": manifest["decoder_id"],
            "manifest_sha256": artifact.manifest_sha256,
            "training_manifest_sha256": manifest["training_manifest_sha256"],
            "model_state_sha256": artifact.summary["selected_state_sha256"],
            "seed": manifest["seed"],
            "profile": manifest["profile"],
            "training_device": artifact.training["runtime"]["device"],
            "inference_device": device,
            "inference_torch_version": str(require_torch().__version__),
            "quality_status": "experimental",
            "initial_proposal": "one-forward-call-if-propose-and-submit-fit",
            "refinement": "ordinary-pattern-search-v1" if refine else "none",
            "refinement_budget": "remaining-public-budget-after-inverse-proposal",
            "model_calls": "one-first-proposal-zero-subsequent-decisions",
        },
        artifact_paths=artifact.artifact_paths,
    )


def train_inverse(
    output: Path,
    *,
    seed: int = 0,
    profile: str = "quick",
    device: str = "cpu",
    progress=None,
    cache_directory: Path | None = None,
) -> LoadedInverseArtifact:
    """Train from frozen train/validation memberships with bounded CPU memory."""
    from harpy.tasks.sound_match.inverse_training import train_inverse as train

    return train(
        output,
        seed=seed,
        profile=profile,
        device=device,
        progress=progress,
        cache_directory=cache_directory,
    )


__all__ = [
    "InverseActor",
    "LoadedInverseArtifact",
    "inverse_actor_spec",
    "load_inverse_artifact",
    "train_inverse",
]
