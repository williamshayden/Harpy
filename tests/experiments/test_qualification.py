"""Release identity gates reject invalid cohorts before expensive score readout."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from harpy.envs.models import EpisodeResult, PitchAction, TerminalReason
from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_RENDERER_ID
from harpy.experiments.actors import COMMITTED_CONTROLLER, FEASIBLE_DECODER
from harpy.experiments.models import EpisodeSpec
from harpy.experiments.qualification import _references, qualify_reference
from harpy.experiments.results import EpisodeRecord, ExperimentResult


def _actors():
    actors = []
    for seed in range(3):
        manifest_hash, model_hash, metadata_hash = (str(seed + offset) * 64 for offset in (1, 4, 7))
        payloads = {"model.pt": model_hash, "metadata.json": metadata_hash}
        actors.append(
            {
                "name": f"pitch-{seed}",
                "observation_mode": "spectrum",
                "seed": seed,
                "device": "cpu",
                "estimator": "harpy-sine-pitch-estimator-v1",
                "decoder": FEASIBLE_DECODER,
                "controller": COMMITTED_CONTROLLER,
                "artifact_provenance": {
                    "trainer": "pitch",
                    "profile": "checkpoint",
                    "seed": seed,
                    "evaluation_device": "cpu",
                    "manifest_sha256": manifest_hash,
                    "payload_sha256": payloads,
                },
                "artifact_files": [
                    {"path": f"/models/pitch-{seed}/{name}", "sha256": digest, "size_bytes": 1}
                    for name, digest in {"manifest.json": manifest_hash, **payloads}.items()
                ],
            }
        )
    return actors


def _result(actors=None):
    """A small valid ordinary result; it intentionally is not a release protocol."""
    actors = _actors() if actors is None else actors
    episode = EpisodeSpec("minimal", 6000, 12, partition="iid")
    terminal = EpisodeResult(
        source_pitch_cents=6000,
        target_note_index=12,
        target_pitch_cents=6000,
        final_pitch_cents=6000,
        initial_signed_error_cents=0,
        initial_absolute_error_cents=0,
        final_signed_error_cents=0,
        final_absolute_error_cents=0,
        submitted_success=True,
        within_5_cents=True,
        within_1_cent=True,
        actions=(PitchAction.SUBMIT,),
        invalid_action_count=0,
        optimal_actions=(PitchAction.SUBMIT,),
        excess_actions=0,
        total_return=1.0,
        terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
    )
    evidence = {
        "renderer_id": ROBUSTNESS_RENDERER_ID,
        "condition_id": "clean",
        "nuisance_seed": 0,
        "dry_rms": 1.0,
        "requested_snr_db": None,
        "realized_snr_db": None,
        "waveform_sha256": "a" * 64,
        "spectrum_sha256": "b" * 64,
    }
    return ExperimentResult(
        (episode,),
        tuple(actors),
        (CLEAN_CONDITION.to_document(),),
        tuple(
            EpisodeRecord(
                actor["name"], "clean", episode.id, terminal, (1,), initial_evidence=evidence
            )
            for actor in actors
        ),
        {"source": {"package_sha256": "c" * 64}, "runtime": {}, "renderer": ROBUSTNESS_RENDERER_ID},
        {"id": "minimal-custom", "episodes": [episode.to_document()]},
    )


def test_reference_identity_accepts_exactly_three_matched_checkpoint_seeds():
    result = _result()
    assert set(_references(result)) == {0, 1, 2}
    # Generic actor names do not select the shipped seed or change its identity.
    actors = _actors()
    actors[0]["name"] = "renamed-reference"
    assert _references(_result(actors))[0]["name"] == "renamed-reference"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda actor: actor.update(device="cuda"),
        lambda actor: actor["artifact_provenance"].update(evaluation_device="cuda"),
        lambda actor: actor["artifact_provenance"].update(manifest_sha256="forged"),
        lambda actor: actor["artifact_provenance"]["payload_sha256"].update({"model.pt": "forged"}),
        lambda actor: actor["artifact_provenance"]["payload_sha256"].update({"model.pt": "d" * 64}),
        lambda actor: actor["artifact_files"][0].update(sha256="e" * 64),
        lambda actor: actor["artifact_files"].pop(),
        lambda actor: actor.update(controller="different-controller"),
        lambda actor: actor["artifact_provenance"].update(profile="smoke"),
    ],
)
def test_reference_identity_rejects_device_and_artifact_mismatch(mutate):
    actors = _actors()
    mutate(actors[0])
    with pytest.raises(ValueError, match=r"identity|inventory"):
        _references(_result(actors))


@pytest.mark.parametrize("selection", [(0, 1), (1, 2), (0, 1, 1)])
def test_reference_cohort_rejects_missing_or_duplicated_seeds(selection):
    originals = _actors()
    actors = [deepcopy(originals[seed]) for seed in selection]
    for index, actor in enumerate(actors):
        actor["name"] = f"selected-{index}"
    with pytest.raises(ValueError, match=r"exactly once|identity"):
        _references(_result(actors))


def test_reference_cohort_rejects_substituted_seed():
    actors = _actors()
    actors[2]["seed"] = 3
    actors[2]["artifact_provenance"]["seed"] = 3
    with pytest.raises(ValueError, match="identity"):
        _references(_result(actors))


def test_qualification_rejects_changed_evaluator_source_before_scoring():
    result = _result()
    changed = replace(
        result,
        provenance={
            **dict(result.provenance),
            "source": {"package_sha256": "d" * 64},
        },
    )
    with pytest.raises(ValueError, match="matching evaluator source"):
        qualify_reference(result, changed)


def test_changed_renderer_is_rejected_at_the_result_boundary():
    result = _result()
    with pytest.raises(ValueError, match="renderer"):
        changed = replace(result, provenance={**dict(result.provenance), "renderer": "different"})
        qualify_reference(result, changed)


@pytest.mark.parametrize("protocol", [None, "custom"])
def test_qualification_rejects_incomplete_or_custom_episode_protocol(protocol):
    result = _result()
    if protocol is None:
        result = replace(result, protocol=None)
    with pytest.raises(ValueError, match="complete frozen protocol"):
        qualify_reference(result, result)
