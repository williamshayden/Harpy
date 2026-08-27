from __future__ import annotations

import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

from harpy.envs.models import PitchAction
from harpy.learning.actors import BC_ACTOR_SEMANTICS_ID
from harpy.learning.diagnostic_codecs import (
    diagnostic_bundle_bytes,
    diagnostic_bundle_from_bytes,
    diagnostic_report_bytes,
    diagnostic_report_from_bytes,
    validate_diagnostic_output_path,
)
from harpy.learning.diagnostics import (
    DiagnosticDecisionInput,
    DiagnosticReportInventory,
    EpisodeDiagnostic,
    build_diagnostic_bundle,
    build_diagnostic_report,
    diagnose_episode,
)
from harpy.learning.models import EvaluationSuiteId
from harpy.learning.pitch_data import (
    PitchEvaluationSuiteId,
    fixed_pitch_evaluation_suite,
)
from harpy.learning.suites import fixed_evaluation_suite

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _episode(index: int = 0, *, source_pitch_cents: int = 4_800):
    return diagnose_episode(
        episode_index=index,
        source_pitch_cents=source_pitch_cents,
        target_note_index=0,
        decisions=(DiagnosticDecisionInput(PitchAction.SUBMIT),),
    )


def _report(
    seed: int = 0,
    *,
    final: bool = False,
    source_pitch_cents: int = 4_800,
):
    if final:
        suite = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.IID)
        return build_diagnostic_report(
            seed=seed,
            artifact_manifest_sha256=f"{seed + 1:064x}",
            actor_semantics="harpy-sine-pitch-estimator-planner-v1",
            bound_mask=False,
            suite_id=suite.suite_id.value,
            suite_digest_sha256=suite.digest_sha256,
            episodes=_pitch_episodes(PitchEvaluationSuiteId.IID),
        )
    return build_diagnostic_report(
        seed=seed,
        artifact_manifest_sha256=f"{seed + 1:064x}",
        actor_semantics=BC_ACTOR_SEMANTICS_ID,
        bound_mask=False,
        suite_id="legacy-smoke-v1",
        suite_digest_sha256=_DIGEST_A,
        episodes=(_episode(source_pitch_cents=source_pitch_cents),),
    )


@cache
def _pitch_episodes(
    suite_id: PitchEvaluationSuiteId,
) -> tuple[EpisodeDiagnostic, ...]:
    suite = fixed_pitch_evaluation_suite(suite_id)
    return tuple(
        diagnose_episode(
            episode_index=index,
            source_pitch_cents=spec.source_pitch_cents,
            target_note_index=spec.target_note_index,
            decisions=(
                DiagnosticDecisionInput(
                    PitchAction.SUBMIT,
                    1_100 + 5 * ((spec.source_pitch_cents - 1_100 + 2) // 5),
                ),
            ),
        )
        for index, spec in enumerate(suite.episodes)
    )


@cache
def _legacy_episodes(suite_id: EvaluationSuiteId) -> tuple[EpisodeDiagnostic, ...]:
    suite = fixed_evaluation_suite(suite_id)
    return tuple(
        diagnose_episode(
            episode_index=index,
            source_pitch_cents=spec.source_pitch_cents,
            target_note_index=spec.target_note_index,
            decisions=(DiagnosticDecisionInput(PitchAction.SUBMIT),),
        )
        for index, spec in enumerate(suite.episodes)
    )


def _canonical(document: object) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        + b"\n"
    )


def test_report_codec_is_canonical_strict_and_round_trips() -> None:
    report = _report()

    content = diagnostic_report_bytes(report)

    assert content.endswith(b"\n") and not content.endswith(b"\n\n")
    assert content == diagnostic_report_bytes(diagnostic_report_from_bytes(content))
    assert diagnostic_report_from_bytes(content) == report


def test_registered_legacy_report_round_trips_and_rejects_identity_spoofs() -> None:
    suite = fixed_evaluation_suite(EvaluationSuiteId.SMOKE)
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=f"{1:064x}",
        actor_semantics=BC_ACTOR_SEMANTICS_ID,
        bound_mask=False,
        suite_id=suite.suite_id.value,
        suite_digest_sha256=suite.digest_sha256,
        episodes=_legacy_episodes(EvaluationSuiteId.SMOKE),
    )
    content = diagnostic_report_bytes(report)
    assert diagnostic_report_from_bytes(content) == report
    assert diagnostic_report_bytes(diagnostic_report_from_bytes(content)) == content

    with pytest.raises(ValueError, match="pinned registered suite"):
        replace(report, suite_digest_sha256=_DIGEST_B)

    second = suite.episodes[1]
    mutated_first = diagnose_episode(
        episode_index=0,
        source_pitch_cents=second.source_pitch_cents,
        target_note_index=second.target_note_index,
        decisions=(DiagnosticDecisionInput(PitchAction.SUBMIT),),
    )
    with pytest.raises(ValueError, match="pinned ordered suite membership"):
        build_diagnostic_report(
            seed=report.seed,
            artifact_manifest_sha256=report.artifact_manifest_sha256,
            actor_semantics=report.actor_semantics,
            bound_mask=report.bound_mask,
            suite_id=report.suite_id,
            suite_digest_sha256=report.suite_digest_sha256,
            episodes=(mutated_first, *report.episodes[1:]),
        )


@pytest.mark.parametrize("suite_id", tuple(EvaluationSuiteId))
def test_every_registered_legacy_suite_closes_digest_and_membership(
    suite_id: EvaluationSuiteId,
) -> None:
    suite = fixed_evaluation_suite(suite_id)
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=f"{1:064x}",
        actor_semantics=BC_ACTOR_SEMANTICS_ID,
        bound_mask=False,
        suite_id=suite.suite_id.value,
        suite_digest_sha256=suite.digest_sha256,
        episodes=_legacy_episodes(suite_id),
    )
    assert report.suite_digest_sha256 == suite.digest_sha256
    with pytest.raises(ValueError, match="pinned registered suite"):
        replace(report, suite_digest_sha256=_DIGEST_B)

    second = suite.episodes[1]
    mutated_first = diagnose_episode(
        episode_index=0,
        source_pitch_cents=second.source_pitch_cents,
        target_note_index=second.target_note_index,
        decisions=(DiagnosticDecisionInput(PitchAction.SUBMIT),),
    )
    with pytest.raises(ValueError, match="pinned ordered suite membership"):
        replace(report, episodes=(mutated_first, *report.episodes[1:]))


@pytest.mark.parametrize("suite_id", tuple(PitchEvaluationSuiteId))
def test_every_registered_pitch_suite_closes_digest_and_membership(
    suite_id: PitchEvaluationSuiteId,
) -> None:
    suite = fixed_pitch_evaluation_suite(suite_id)
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=f"{1:064x}",
        actor_semantics="harpy-sine-pitch-estimator-planner-v1",
        bound_mask=False,
        suite_id=suite.suite_id.value,
        suite_digest_sha256=suite.digest_sha256,
        episodes=_pitch_episodes(suite_id),
    )
    assert report.suite_digest_sha256 == suite.digest_sha256
    with pytest.raises(ValueError, match="pinned registered suite"):
        replace(report, suite_digest_sha256=_DIGEST_B)

    second = suite.episodes[1]
    mutated_first = diagnose_episode(
        episode_index=0,
        source_pitch_cents=second.source_pitch_cents,
        target_note_index=second.target_note_index,
        decisions=(
            DiagnosticDecisionInput(
                PitchAction.SUBMIT,
                1_100 + 5 * ((second.source_pitch_cents - 1_100 + 2) // 5),
            ),
        ),
    )
    with pytest.raises(ValueError, match="pinned ordered suite membership"):
        replace(report, episodes=(mutated_first, *report.episodes[1:]))


@pytest.mark.parametrize(
    "content, message",
    [
        (b'{"schema_id":"x","schema_id":"x"}\n', "duplicate"),
        (b'{"value":NaN}\n', "finite"),
        (b'{"value":Infinity}\n', "finite"),
        (b"[]\n", "object"),
        (b"{broken}\n", "JSON"),
    ],
)
def test_report_codec_rejects_duplicate_nonfinite_and_malformed_json(
    content: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnostic_report_from_bytes(content)


def test_report_codec_rejects_derived_field_mismatch_and_noncanonical_bytes() -> None:
    content = diagnostic_report_bytes(_report())
    document = json.loads(content)
    document["shortest_action_correct"] = 0
    with pytest.raises(ValueError, match=r"derived|match"):
        diagnostic_report_from_bytes(_canonical(document))

    with pytest.raises(ValueError, match="canonical"):
        diagnostic_report_from_bytes(content[:-1] + b"  \n")


def test_report_codec_rejects_actor_estimate_and_mask_linkage_mutations() -> None:
    document = json.loads(diagnostic_report_bytes(_report()))
    document["actor_semantics"] = "invented-policy-v1"
    with pytest.raises(ValueError, match="actor_semantics"):
        diagnostic_report_from_bytes(_canonical(document))

    document = json.loads(diagnostic_report_bytes(_report()))
    document["bound_mask"] = True
    with pytest.raises(ValueError, match="bound_mask"):
        diagnostic_report_from_bytes(_canonical(document))

    document = json.loads(diagnostic_report_bytes(_report()))
    document["episodes"][0]["decisions"][0]["estimate"] = {
        "estimated_candidate_cents": 4_800,
        "signed_error_cents": 0,
        "absolute_error_cents": 0,
    }
    with pytest.raises(ValueError, match=r"legacy.*null estimates"):
        diagnostic_report_from_bytes(_canonical(document))

    pitch_report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=f"{1:064x}",
        actor_semantics="harpy-sine-pitch-estimator-planner-v1",
        bound_mask=False,
        suite_id=fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.SMOKE).suite_id.value,
        suite_digest_sha256=fixed_pitch_evaluation_suite(
            PitchEvaluationSuiteId.SMOKE
        ).digest_sha256,
        episodes=_pitch_episodes(PitchEvaluationSuiteId.SMOKE),
    )
    document = json.loads(diagnostic_report_bytes(pitch_report))
    document["episodes"][0]["decisions"][0]["estimate"] = None
    with pytest.raises(ValueError, match=r"pitch.*estimate"):
        diagnostic_report_from_bytes(_canonical(document))


def test_bundle_requires_final_seeds_and_exact_hashed_inventory() -> None:
    reports = tuple(_report(seed, final=True) for seed in (0, 1, 2))
    bundle = build_diagnostic_bundle(
        device="cpu",
        bound_mask=False,
        reports=reports,
    )

    content = diagnostic_bundle_bytes(bundle)
    assert diagnostic_bundle_from_bytes(content) == bundle
    assert tuple(item.seed for item in bundle.inventory) == (0, 1, 2)

    with pytest.raises(ValueError, match="seeds 0, 1, and 2"):
        build_diagnostic_bundle(device="cpu", bound_mask=False, reports=reports[:2])
    with pytest.raises(ValueError, match=r"duplicate.*seed"):
        build_diagnostic_bundle(
            device="cpu",
            bound_mask=False,
            reports=(reports[0], reports[0], reports[2]),
        )

    bad_inventory = (
        DiagnosticReportInventory(
            seed=0,
            artifact_manifest_sha256=reports[0].artifact_manifest_sha256,
            report_sha256=_DIGEST_B,
        ),
        *bundle.inventory[1:],
    )
    with pytest.raises(ValueError, match="inventory"):
        replace(bundle, inventory=bad_inventory)


def test_bundle_codec_rejects_missing_inventory_and_wrong_suite_identity() -> None:
    bundle = build_diagnostic_bundle(
        device="cpu",
        bound_mask=False,
        reports=tuple(_report(seed, final=True) for seed in (0, 1, 2)),
    )
    document = json.loads(diagnostic_bundle_bytes(bundle))
    document["inventory"].pop()
    with pytest.raises(ValueError, match="inventory"):
        diagnostic_bundle_from_bytes(_canonical(document))

    document = json.loads(diagnostic_bundle_bytes(bundle))
    document["reports"][1]["suite_digest_sha256"] = _DIGEST_B
    with pytest.raises(ValueError, match=r"suite|derived|inventory"):
        diagnostic_bundle_from_bytes(_canonical(document))


def test_registered_suite_rejects_fake_digest_and_ordered_membership() -> None:
    report = _report(0, final=True)
    with pytest.raises(ValueError, match="pinned registered suite"):
        replace(report, suite_digest_sha256=_DIGEST_B)

    _, second, *remaining = report.episodes
    mutated_first = diagnose_episode(
        episode_index=0,
        source_pitch_cents=second.source_pitch_cents,
        target_note_index=second.target_note_index,
        decisions=(
            DiagnosticDecisionInput(
                PitchAction.SUBMIT,
                1_100 + 5 * ((second.source_pitch_cents - 1_100 + 2) // 5),
            ),
        ),
    )
    with pytest.raises(ValueError, match="pinned ordered suite membership"):
        build_diagnostic_report(
            seed=report.seed,
            artifact_manifest_sha256=report.artifact_manifest_sha256,
            actor_semantics=report.actor_semantics,
            bound_mask=report.bound_mask,
            suite_id=report.suite_id,
            suite_digest_sha256=report.suite_digest_sha256,
            episodes=(mutated_first, second, *remaining),
        )


def test_single_legacy_bundle_is_allowed_but_multi_seed_legacy_is_not() -> None:
    assert (
        build_diagnostic_bundle(device="cpu", bound_mask=False, reports=(_report(seed=9),))
        .reports[0]
        .seed
        == 9
    )

    with pytest.raises(ValueError, match="exactly one"):
        build_diagnostic_bundle(device="cpu", bound_mask=False, reports=(_report(0), _report(1)))


def test_output_path_preflight_rejects_existing_and_artifact_aliases(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    existing = tmp_path / "existing.json"
    existing.write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="exist"):
        validate_diagnostic_output_path(existing, artifact_directories=(artifact,))
    with pytest.raises(ValueError, match="outside"):
        validate_diagnostic_output_path(
            artifact / "diagnostics.json", artifact_directories=(artifact,)
        )

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    assert not dangling.exists() and dangling.is_symlink()
    with pytest.raises(ValueError, match=r"symlink|exist"):
        validate_diagnostic_output_path(dangling, artifact_directories=(artifact,))

    output = validate_diagnostic_output_path(
        tmp_path / "diagnostics.json", artifact_directories=(artifact,)
    )
    assert output == (tmp_path / "diagnostics.json").resolve()
