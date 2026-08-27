"""Dependency-light, evaluator-owned trajectory diagnostics.

This module deliberately contains no environment runner.  Callers supply the actor's
ordered actions and optional typed pitch estimates after a real rollout; the builders
reconstruct all musical truth from the frozen public task contract.
"""

from __future__ import annotations

import hashlib
import math
import operator
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from harpy.envs.models import (
    MAX_ABSOLUTE_ERROR_CENTS,
    MAX_STEPS,
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    SUCCESS_TOLERANCE_CENTS,
    TARGET_MIN_COORDINATE,
    TARGET_NOTE_COUNT,
    ZERO_CONTROLS,
    ControlState,
    PitchAction,
    TerminalReason,
)
from harpy.envs.planning import minimum_action_plan

DIAGNOSTIC_REPORT_SCHEMA_ID = "harpy-sine-diagnostic-report-v1"
DIAGNOSTIC_BUNDLE_SCHEMA_ID = "harpy-sine-diagnostic-bundle-v1"

# These identities form the dependency-light diagnostic registry.  Keep the
# schema-v2 value here instead of importing ``pitch_artifacts``: that module owns
# Torch model persistence and must remain lazy for report decoding.
_BC_ACTOR_SEMANTICS_ID = "harpy-sine-policy-bc-v1"
_MASKED_BC_ACTOR_SEMANTICS_ID = "harpy-sine-policy-bc-public-bound-mask-v1"
_PPO_ACTOR_SEMANTICS_ID = "harpy-sine-policy-ppo-v1"
_PITCH_ACTOR_SEMANTICS_ID = "harpy-sine-pitch-estimator-planner-v1"
_LEGACY_ACTOR_SEMANTICS_IDS = frozenset(
    {
        _BC_ACTOR_SEMANTICS_ID,
        _MASKED_BC_ACTOR_SEMANTICS_ID,
        _PPO_ACTOR_SEMANTICS_ID,
    }
)
_KNOWN_ACTOR_SEMANTICS_IDS = _LEGACY_ACTOR_SEMANTICS_IDS | {_PITCH_ACTOR_SEMANTICS_ID}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ESTIMATE_MIN_CENTS = 1_100
_ESTIMATE_MAX_CENTS = 10_900
_FINAL_SUITE_IDS = frozenset(
    {
        "harpy-sine-pitch-e-iid-v1",
        "harpy-sine-pitch-e-ood-lower-v1",
        "harpy-sine-pitch-e-ood-upper-v1",
    }
)
_SUBMIT_BANDS = (
    ("0..1", 0, 1),
    ("2..5", 2, 5),
    ("6..25", 6, 25),
    ("26..99", 26, 99),
    ("100+", 100, None),
)


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _finite_rate(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite rate")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite rate") from error
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be a finite rate within 0..1")
    return normalized


def _optional_rate(value: object, field: str) -> float | None:
    return None if value is None else _finite_rate(value, field)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _action(value: object, field: str) -> PitchAction:
    if not isinstance(value, PitchAction):
        raise ValueError(f"{field} must be a PitchAction")
    return value


def _optional_step(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, minimum=1, maximum=MAX_STEPS)


def _optional_action(value: object, field: str) -> PitchAction | None:
    return None if value is None else _action(value, field)


def _typed_tuple[T](
    value: Iterable[T], expected_type: type[T], field: str, *, nonempty: bool = False
) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    try:
        normalized = tuple(value)
    except TypeError as error:
        raise ValueError(f"{field} must contain {expected_type.__name__} values") from error
    if (nonempty and not normalized) or not all(
        isinstance(item, expected_type) for item in normalized
    ):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    return normalized


@dataclass(frozen=True, slots=True)
class DiagnosticDecisionInput:
    """One actor decision plus an optional estimate from a typed decision API."""

    action: PitchAction
    estimated_candidate_cents: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _action(self.action, "action"))
        if self.estimated_candidate_cents is not None:
            estimate = _integer(
                self.estimated_candidate_cents,
                "estimated_candidate_cents",
                minimum=_ESTIMATE_MIN_CENTS,
                maximum=_ESTIMATE_MAX_CENTS,
            )
            if (estimate - _ESTIMATE_MIN_CENTS) % 5 != 0:
                raise ValueError("estimated_candidate_cents must lie on the five-cent grid")
            object.__setattr__(self, "estimated_candidate_cents", estimate)


@dataclass(frozen=True, slots=True)
class EstimateDiagnostic:
    """Typed estimator output scored against evaluator-owned current pitch truth."""

    estimated_candidate_cents: int
    signed_error_cents: int
    absolute_error_cents: int

    def __post_init__(self) -> None:
        estimate = _integer(
            self.estimated_candidate_cents,
            "estimated_candidate_cents",
            minimum=_ESTIMATE_MIN_CENTS,
            maximum=_ESTIMATE_MAX_CENTS,
        )
        if (estimate - _ESTIMATE_MIN_CENTS) % 5 != 0:
            raise ValueError("estimated_candidate_cents must lie on the five-cent grid")
        signed = _integer(self.signed_error_cents, "signed_error_cents")
        absolute = _integer(self.absolute_error_cents, "absolute_error_cents", minimum=0)
        if absolute != abs(signed):
            raise ValueError("absolute_error_cents must equal the absolute signed error")
        object.__setattr__(self, "estimated_candidate_cents", estimate)
        object.__setattr__(self, "signed_error_cents", signed)
        object.__setattr__(self, "absolute_error_cents", absolute)


@dataclass(frozen=True, slots=True)
class DecisionDiagnostic:
    """All evaluator-derived facts for one one-based pre-action decision."""

    step: int
    controls: ControlState
    steps_remaining: int
    true_candidate_cents: int
    true_absolute_error_cents: int
    teacher_action: PitchAction
    predicted_action: PitchAction
    canonical_match: bool
    in_shortest_action_set: bool
    legal: bool
    bound_blocked: bool
    repeated_state_visit: bool
    loop_transition: bool
    shortest_plan_length: int
    recoverable: bool
    estimate: EstimateDiagnostic | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _integer(self.step, "step", minimum=1, maximum=MAX_STEPS))
        if not isinstance(self.controls, ControlState):
            raise ValueError("controls must be a ControlState")
        object.__setattr__(
            self,
            "steps_remaining",
            _integer(self.steps_remaining, "steps_remaining", minimum=1, maximum=MAX_STEPS),
        )
        object.__setattr__(
            self,
            "true_candidate_cents",
            _integer(self.true_candidate_cents, "true_candidate_cents"),
        )
        object.__setattr__(
            self,
            "true_absolute_error_cents",
            _integer(
                self.true_absolute_error_cents,
                "true_absolute_error_cents",
                minimum=0,
                maximum=MAX_ABSOLUTE_ERROR_CENTS,
            ),
        )
        object.__setattr__(self, "teacher_action", _action(self.teacher_action, "teacher_action"))
        object.__setattr__(
            self, "predicted_action", _action(self.predicted_action, "predicted_action")
        )
        for field in (
            "canonical_match",
            "in_shortest_action_set",
            "legal",
            "bound_blocked",
            "repeated_state_visit",
            "loop_transition",
            "recoverable",
        ):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        object.__setattr__(
            self,
            "shortest_plan_length",
            _integer(
                self.shortest_plan_length,
                "shortest_plan_length",
                minimum=1,
                maximum=MAX_STEPS,
            ),
        )
        if self.estimate is not None and not isinstance(self.estimate, EstimateDiagnostic):
            raise ValueError("estimate must be an EstimateDiagnostic or None")


@dataclass(frozen=True, slots=True)
class EpisodeDiagnostic:
    """A complete trajectory record with exact event/count null semantics."""

    episode_index: int
    source_pitch_cents: int
    target_note_index: int
    target_pitch_cents: int
    decisions: tuple[DecisionDiagnostic, ...]
    first_canonical_mismatch_step: int | None
    first_canonical_mismatch_action: PitchAction | None
    first_shortest_divergence_step: int | None
    first_shortest_divergence_action: PitchAction | None
    first_loop_step: int | None
    first_unrecoverable_step: int | None
    legal_action_count: int
    bound_blocked_action_count: int
    repeated_state_visit_count: int
    loop_transition_count: int
    terminal_reason: TerminalReason
    final_absolute_error_cents: int
    action_count: int
    excess_actions: int | None

    def __post_init__(self) -> None:
        episode_index = _integer(self.episode_index, "episode_index", minimum=0)
        source = _integer(
            self.source_pitch_cents,
            "source_pitch_cents",
            minimum=SOURCE_MIN_CENTS,
            maximum=SOURCE_MAX_CENTS,
        )
        target_index = _integer(
            self.target_note_index,
            "target_note_index",
            minimum=0,
            maximum=TARGET_NOTE_COUNT - 1,
        )
        target = _integer(self.target_pitch_cents, "target_pitch_cents")
        expected_target = 100 * (TARGET_MIN_COORDINATE + target_index)
        if target != expected_target:
            raise ValueError("target_pitch_cents must match target_note_index")
        decisions = _typed_tuple(self.decisions, DecisionDiagnostic, "decisions", nonempty=True)
        inputs = tuple(
            DiagnosticDecisionInput(
                decision.predicted_action,
                None if decision.estimate is None else decision.estimate.estimated_candidate_cents,
            )
            for decision in decisions
        )
        expected_decisions, facts = _derive_episode(source, target_index, inputs)
        if decisions != expected_decisions:
            raise ValueError("decision diagnostics must match the derived trajectory")

        expected_fields: dict[str, object] = {
            "first_canonical_mismatch_step": facts.first_canonical_mismatch_step,
            "first_canonical_mismatch_action": facts.first_canonical_mismatch_action,
            "first_shortest_divergence_step": facts.first_shortest_divergence_step,
            "first_shortest_divergence_action": facts.first_shortest_divergence_action,
            "first_loop_step": facts.first_loop_step,
            "first_unrecoverable_step": facts.first_unrecoverable_step,
            "legal_action_count": facts.legal_action_count,
            "bound_blocked_action_count": facts.bound_blocked_action_count,
            "repeated_state_visit_count": facts.repeated_state_visit_count,
            "loop_transition_count": facts.loop_transition_count,
            "terminal_reason": facts.terminal_reason,
            "final_absolute_error_cents": facts.final_absolute_error_cents,
            "action_count": len(decisions),
            "excess_actions": facts.excess_actions,
        }
        for field, expected in expected_fields.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} must match the derived trajectory")

        # Normalize types even though equality above rejects their bool/int aliases.
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "source_pitch_cents", source)
        object.__setattr__(self, "target_note_index", target_index)
        object.__setattr__(self, "target_pitch_cents", target)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self,
            "first_canonical_mismatch_step",
            _optional_step(self.first_canonical_mismatch_step, "first_canonical_mismatch_step"),
        )
        object.__setattr__(
            self,
            "first_canonical_mismatch_action",
            _optional_action(
                self.first_canonical_mismatch_action, "first_canonical_mismatch_action"
            ),
        )
        object.__setattr__(
            self,
            "first_shortest_divergence_step",
            _optional_step(self.first_shortest_divergence_step, "first_shortest_divergence_step"),
        )
        object.__setattr__(
            self,
            "first_shortest_divergence_action",
            _optional_action(
                self.first_shortest_divergence_action, "first_shortest_divergence_action"
            ),
        )
        object.__setattr__(
            self, "first_loop_step", _optional_step(self.first_loop_step, "first_loop_step")
        )
        object.__setattr__(
            self,
            "first_unrecoverable_step",
            _optional_step(self.first_unrecoverable_step, "first_unrecoverable_step"),
        )
        for field in (
            "legal_action_count",
            "bound_blocked_action_count",
            "repeated_state_visit_count",
            "loop_transition_count",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field, minimum=0))
        if not isinstance(self.terminal_reason, TerminalReason):
            raise ValueError("terminal_reason must be a TerminalReason")
        object.__setattr__(
            self,
            "final_absolute_error_cents",
            _integer(
                self.final_absolute_error_cents,
                "final_absolute_error_cents",
                minimum=0,
                maximum=MAX_ABSOLUTE_ERROR_CENTS,
            ),
        )
        object.__setattr__(
            self,
            "action_count",
            _integer(self.action_count, "action_count", minimum=1, maximum=MAX_STEPS),
        )
        if self.excess_actions is not None:
            object.__setattr__(
                self,
                "excess_actions",
                _integer(self.excess_actions, "excess_actions", minimum=0),
            )


@dataclass(frozen=True, slots=True)
class PerActionMetrics:
    """One confusion-derived action row with explicit raw denominators."""

    action: PitchAction
    precision_numerator: int
    precision_denominator: int
    precision: float | None
    recall_numerator: int
    recall_denominator: int
    recall: float | None
    support: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _action(self.action, "action"))
        precision_numerator = _integer(self.precision_numerator, "precision_numerator", minimum=0)
        precision_denominator = _integer(
            self.precision_denominator, "precision_denominator", minimum=0
        )
        recall_numerator = _integer(self.recall_numerator, "recall_numerator", minimum=0)
        recall_denominator = _integer(self.recall_denominator, "recall_denominator", minimum=0)
        support = _integer(self.support, "support", minimum=0)
        if precision_numerator > precision_denominator:
            raise ValueError("precision numerator must not exceed its denominator")
        if recall_numerator > recall_denominator:
            raise ValueError("recall numerator must not exceed its denominator")
        if recall_numerator != precision_numerator:
            raise ValueError("precision and recall true-positive numerators must match")
        if support != recall_denominator:
            raise ValueError("support must equal the recall denominator")
        precision = _optional_rate(self.precision, "precision")
        recall = _optional_rate(self.recall, "recall")
        expected_precision = (
            None if precision_denominator == 0 else precision_numerator / precision_denominator
        )
        expected_recall = None if recall_denominator == 0 else recall_numerator / recall_denominator
        if precision != expected_precision:
            raise ValueError("precision must match its numerator and denominator")
        if recall != expected_recall:
            raise ValueError("recall must match its numerator and denominator")
        object.__setattr__(self, "precision_numerator", precision_numerator)
        object.__setattr__(self, "precision_denominator", precision_denominator)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "recall_numerator", recall_numerator)
        object.__setattr__(self, "recall_denominator", recall_denominator)
        object.__setattr__(self, "recall", recall)
        object.__setattr__(self, "support", support)


@dataclass(frozen=True, slots=True)
class SurvivalEntry:
    """Canonical shortest-set survival at one one-based decision index."""

    step: int
    at_risk: int
    survivors: int
    rate: float

    def __post_init__(self) -> None:
        step = _integer(self.step, "step", minimum=1, maximum=MAX_STEPS)
        at_risk = _integer(self.at_risk, "at_risk", minimum=1)
        survivors = _integer(self.survivors, "survivors", minimum=0)
        if survivors > at_risk:
            raise ValueError("survivors must not exceed at_risk")
        rate = _finite_rate(self.rate, "rate")
        if rate != survivors / at_risk:
            raise ValueError("survival rate must equal survivors / at_risk")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "at_risk", at_risk)
        object.__setattr__(self, "survivors", survivors)
        object.__setattr__(self, "rate", rate)


@dataclass(frozen=True, slots=True)
class SubmitErrorBand:
    """Count of Submit decisions within one fixed true-error band."""

    label: str
    minimum_cents: int
    maximum_cents: int | None
    count: int

    def __post_init__(self) -> None:
        label = _string(self.label, "label")
        minimum = _integer(self.minimum_cents, "minimum_cents", minimum=0)
        maximum = (
            None
            if self.maximum_cents is None
            else _integer(self.maximum_cents, "maximum_cents", minimum=minimum)
        )
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "minimum_cents", minimum)
        object.__setattr__(self, "maximum_cents", maximum)
        object.__setattr__(self, "count", _integer(self.count, "count", minimum=0))


def _registered_suite_contract(
    suite_id: str,
) -> tuple[bool, str, tuple[tuple[int, int, int], ...]] | None:
    """Resolve one workflow suite without importing optional training dependencies."""

    from harpy.learning.models import EvaluationSuiteId

    try:
        legacy_suite_id = EvaluationSuiteId(suite_id)
    except ValueError:
        legacy_suite_id = None
    if legacy_suite_id is not None:
        from harpy.learning.suites import fixed_evaluation_suite

        suite = fixed_evaluation_suite(legacy_suite_id)
        return (
            False,
            suite.digest_sha256,
            tuple(
                (episode_index, episode.target_note_index, episode.source_pitch_cents)
                for episode_index, episode in enumerate(suite.episodes)
            ),
        )

    from harpy.learning.pitch_data import (
        PitchEvaluationSuiteId,
        fixed_pitch_evaluation_suite,
    )

    try:
        pitch_suite_id = PitchEvaluationSuiteId(suite_id)
    except ValueError:
        return None
    suite = fixed_pitch_evaluation_suite(pitch_suite_id)
    return (
        True,
        suite.digest_sha256,
        tuple(
            (episode_index, episode.target_note_index, episode.source_pitch_cents)
            for episode_index, episode in enumerate(suite.episodes)
        ),
    )


def _validate_report_identity(
    *,
    actor_semantics: str,
    bound_mask: bool,
    suite_id: str,
    suite_digest_sha256: str,
    episodes: tuple[EpisodeDiagnostic, ...],
) -> None:
    if actor_semantics not in _KNOWN_ACTOR_SEMANTICS_IDS:
        raise ValueError("actor_semantics must be a registered diagnostic actor identity")
    expected_bound_mask = actor_semantics == _MASKED_BC_ACTOR_SEMANTICS_ID
    if bound_mask is not expected_bound_mask:
        raise ValueError("bound_mask must be true exactly for masked-BC actor semantics")

    is_pitch_actor = actor_semantics == _PITCH_ACTOR_SEMANTICS_ID
    estimates = tuple(decision.estimate for episode in episodes for decision in episode.decisions)
    if is_pitch_actor and any(estimate is None for estimate in estimates):
        raise ValueError("pitch actor diagnostics require an estimate on every decision")
    if not is_pitch_actor and any(estimate is not None for estimate in estimates):
        raise ValueError("legacy actor diagnostics require null estimates on every decision")

    contract = _registered_suite_contract(suite_id)
    if contract is None:
        return
    is_pitch_suite, expected_digest, expected_membership = contract
    if suite_digest_sha256 != expected_digest:
        raise ValueError("suite_digest_sha256 must match the pinned registered suite")
    if is_pitch_suite != is_pitch_actor:
        expected_lane = "pitch" if is_pitch_suite else "legacy BC/PPO"
        raise ValueError(f"registered suite requires {expected_lane} actor semantics")
    membership = tuple(
        (episode.episode_index, episode.target_note_index, episode.source_pitch_cents)
        for episode in episodes
    )
    if membership != expected_membership:
        raise ValueError("episodes must match the exact pinned ordered suite membership")


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """One strict report for one artifact seed and one fixed suite.

    The seven workflow-exposed suite identities are closed over their pinned digest
    and ordered membership.  Unregistered suite identities remain available only to
    low-level diagnostic-algorithm fixtures; they still obey the closed actor,
    estimator, and mask semantics below.
    """

    schema_id: str
    seed: int
    artifact_manifest_sha256: str
    actor_semantics: str
    bound_mask: bool
    suite_id: str
    suite_digest_sha256: str
    episodes: tuple[EpisodeDiagnostic, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_action_metrics: tuple[PerActionMetrics, ...]
    shortest_action_correct: int
    shortest_action_decisions: int
    shortest_action_accuracy: float
    survival_curve: tuple[SurvivalEntry, ...]
    submit_error_bands: tuple[SubmitErrorBand, ...]

    def __post_init__(self) -> None:
        if self.schema_id != DIAGNOSTIC_REPORT_SCHEMA_ID:
            raise ValueError(f"schema_id must be {DIAGNOSTIC_REPORT_SCHEMA_ID!r}")
        seed = _integer(self.seed, "seed", minimum=0)
        manifest = _digest(self.artifact_manifest_sha256, "artifact_manifest_sha256")
        actor_semantics = _string(self.actor_semantics, "actor_semantics")
        bound_mask = _boolean(self.bound_mask, "bound_mask")
        suite_id = _string(self.suite_id, "suite_id")
        suite_digest = _digest(self.suite_digest_sha256, "suite_digest_sha256")
        episodes = _typed_tuple(self.episodes, EpisodeDiagnostic, "episodes", nonempty=True)
        if tuple(episode.episode_index for episode in episodes) != tuple(range(len(episodes))):
            raise ValueError("episodes must use consecutive zero-based indices")
        _validate_report_identity(
            actor_semantics=actor_semantics,
            bound_mask=bound_mask,
            suite_id=suite_id,
            suite_digest_sha256=suite_digest,
            episodes=episodes,
        )
        expected = _derive_report_fields(episodes)
        actual_matrix = _confusion_matrix(self.confusion_matrix)
        metrics = _typed_tuple(self.per_action_metrics, PerActionMetrics, "per_action_metrics")
        survival = _typed_tuple(self.survival_curve, SurvivalEntry, "survival_curve")
        bands = _typed_tuple(self.submit_error_bands, SubmitErrorBand, "submit_error_bands")
        if actual_matrix != expected.confusion_matrix:
            raise ValueError("confusion_matrix must match the derived episode decisions")
        if metrics != expected.per_action_metrics:
            raise ValueError("per_action_metrics must match the derived confusion matrix")
        if self.shortest_action_correct != expected.shortest_action_correct:
            raise ValueError("shortest_action_correct must match the derived episode decisions")
        if self.shortest_action_decisions != expected.shortest_action_decisions:
            raise ValueError("shortest_action_decisions must match the derived episode decisions")
        accuracy = _finite_rate(self.shortest_action_accuracy, "shortest_action_accuracy")
        if accuracy != expected.shortest_action_accuracy:
            raise ValueError("shortest_action_accuracy must match its derived counts")
        if survival != expected.survival_curve:
            raise ValueError("survival_curve must match the derived episode decisions")
        if bands != expected.submit_error_bands:
            raise ValueError("submit_error_bands must match the derived Submit decisions")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "artifact_manifest_sha256", manifest)
        object.__setattr__(self, "actor_semantics", actor_semantics)
        object.__setattr__(self, "bound_mask", bound_mask)
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "suite_digest_sha256", suite_digest)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "confusion_matrix", actual_matrix)
        object.__setattr__(self, "per_action_metrics", metrics)
        object.__setattr__(
            self,
            "shortest_action_correct",
            _integer(self.shortest_action_correct, "shortest_action_correct", minimum=0),
        )
        object.__setattr__(
            self,
            "shortest_action_decisions",
            _integer(self.shortest_action_decisions, "shortest_action_decisions", minimum=1),
        )
        object.__setattr__(self, "shortest_action_accuracy", accuracy)
        object.__setattr__(self, "survival_curve", survival)
        object.__setattr__(self, "submit_error_bands", bands)


@dataclass(frozen=True, slots=True)
class DiagnosticReportInventory:
    """Hash-closed identity for one report embedded in a bundle."""

    seed: int
    artifact_manifest_sha256: str
    report_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        object.__setattr__(
            self,
            "artifact_manifest_sha256",
            _digest(self.artifact_manifest_sha256, "artifact_manifest_sha256"),
        )
        object.__setattr__(self, "report_sha256", _digest(self.report_sha256, "report_sha256"))


@dataclass(frozen=True, slots=True)
class DiagnosticBundle:
    """One invocation's exact one- or three-report diagnostic payload."""

    schema_id: str
    suite_id: str
    suite_digest_sha256: str
    device: str
    bound_mask: bool
    reports: tuple[DiagnosticReport, ...]
    inventory: tuple[DiagnosticReportInventory, ...]

    def __post_init__(self) -> None:
        if self.schema_id != DIAGNOSTIC_BUNDLE_SCHEMA_ID:
            raise ValueError(f"schema_id must be {DIAGNOSTIC_BUNDLE_SCHEMA_ID!r}")
        suite_id = _string(self.suite_id, "suite_id")
        suite_digest = _digest(self.suite_digest_sha256, "suite_digest_sha256")
        device = _string(self.device, "device")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        bound_mask = _boolean(self.bound_mask, "bound_mask")
        reports = _typed_tuple(self.reports, DiagnosticReport, "reports", nonempty=True)
        seeds = tuple(report.seed for report in reports)
        if len(set(seeds)) != len(seeds):
            raise ValueError("reports must not contain duplicate artifact seeds")
        if suite_id in _FINAL_SUITE_IDS:
            if seeds != (0, 1, 2):
                raise ValueError("final diagnostic bundles require ordered seeds 0, 1, and 2")
        elif len(reports) != 1:
            raise ValueError("smoke and legacy diagnostic bundles require exactly one report")
        if len({report.artifact_manifest_sha256 for report in reports}) != len(reports):
            raise ValueError("reports must not contain duplicate artifact manifests")
        memberships = tuple(
            tuple(
                (
                    episode.episode_index,
                    episode.target_note_index,
                    episode.source_pitch_cents,
                )
                for episode in report.episodes
            )
            for report in reports
        )
        if any(membership != memberships[0] for membership in memberships[1:]):
            raise ValueError("bundle reports must share identical ordered episode membership")
        for report in reports:
            if (
                report.suite_id != suite_id
                or report.suite_digest_sha256 != suite_digest
                or report.bound_mask != bound_mask
            ):
                raise ValueError("bundle suite and mask identity must match every report")
        if len({report.actor_semantics for report in reports}) != 1:
            raise ValueError("bundle reports must share actor semantics")
        inventory = _typed_tuple(
            self.inventory, DiagnosticReportInventory, "inventory", nonempty=True
        )
        expected_inventory = _report_inventory(reports)
        if inventory != expected_inventory:
            raise ValueError(
                "bundle inventory must exactly match ordered report identities and hashes"
            )
        object.__setattr__(self, "suite_id", suite_id)
        object.__setattr__(self, "suite_digest_sha256", suite_digest)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "bound_mask", bound_mask)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "inventory", inventory)


@dataclass(frozen=True, slots=True)
class _EpisodeFacts:
    first_canonical_mismatch_step: int | None
    first_canonical_mismatch_action: PitchAction | None
    first_shortest_divergence_step: int | None
    first_shortest_divergence_action: PitchAction | None
    first_loop_step: int | None
    first_unrecoverable_step: int | None
    legal_action_count: int
    bound_blocked_action_count: int
    repeated_state_visit_count: int
    loop_transition_count: int
    terminal_reason: TerminalReason
    final_absolute_error_cents: int
    excess_actions: int | None


@dataclass(frozen=True, slots=True)
class _ReportFields:
    confusion_matrix: tuple[tuple[int, ...], ...]
    per_action_metrics: tuple[PerActionMetrics, ...]
    shortest_action_correct: int
    shortest_action_decisions: int
    shortest_action_accuracy: float
    survival_curve: tuple[SurvivalEntry, ...]
    submit_error_bands: tuple[SubmitErrorBand, ...]


def diagnose_episode(
    *,
    episode_index: int,
    source_pitch_cents: int,
    target_note_index: int,
    decisions: Sequence[DiagnosticDecisionInput],
) -> EpisodeDiagnostic:
    """Derive one exact diagnostic episode from actor actions and typed estimates."""

    normalized_inputs = _typed_tuple(decisions, DiagnosticDecisionInput, "decisions", nonempty=True)
    source = _integer(
        source_pitch_cents,
        "source_pitch_cents",
        minimum=SOURCE_MIN_CENTS,
        maximum=SOURCE_MAX_CENTS,
    )
    target_index = _integer(
        target_note_index,
        "target_note_index",
        minimum=0,
        maximum=TARGET_NOTE_COUNT - 1,
    )
    derived, facts = _derive_episode(source, target_index, normalized_inputs)
    return EpisodeDiagnostic(
        episode_index=_integer(episode_index, "episode_index", minimum=0),
        source_pitch_cents=source,
        target_note_index=target_index,
        target_pitch_cents=100 * (TARGET_MIN_COORDINATE + target_index),
        decisions=derived,
        first_canonical_mismatch_step=facts.first_canonical_mismatch_step,
        first_canonical_mismatch_action=facts.first_canonical_mismatch_action,
        first_shortest_divergence_step=facts.first_shortest_divergence_step,
        first_shortest_divergence_action=facts.first_shortest_divergence_action,
        first_loop_step=facts.first_loop_step,
        first_unrecoverable_step=facts.first_unrecoverable_step,
        legal_action_count=facts.legal_action_count,
        bound_blocked_action_count=facts.bound_blocked_action_count,
        repeated_state_visit_count=facts.repeated_state_visit_count,
        loop_transition_count=facts.loop_transition_count,
        terminal_reason=facts.terminal_reason,
        final_absolute_error_cents=facts.final_absolute_error_cents,
        action_count=len(derived),
        excess_actions=facts.excess_actions,
    )


def build_diagnostic_report(
    *,
    seed: int,
    artifact_manifest_sha256: str,
    actor_semantics: str,
    bound_mask: bool,
    suite_id: str,
    suite_digest_sha256: str,
    episodes: Sequence[EpisodeDiagnostic],
) -> DiagnosticReport:
    """Build and internally revalidate one report's complete derived surface."""

    normalized_episodes = _typed_tuple(episodes, EpisodeDiagnostic, "episodes", nonempty=True)
    fields = _derive_report_fields(normalized_episodes)
    return DiagnosticReport(
        schema_id=DIAGNOSTIC_REPORT_SCHEMA_ID,
        seed=seed,
        artifact_manifest_sha256=artifact_manifest_sha256,
        actor_semantics=actor_semantics,
        bound_mask=bound_mask,
        suite_id=suite_id,
        suite_digest_sha256=suite_digest_sha256,
        episodes=normalized_episodes,
        confusion_matrix=fields.confusion_matrix,
        per_action_metrics=fields.per_action_metrics,
        shortest_action_correct=fields.shortest_action_correct,
        shortest_action_decisions=fields.shortest_action_decisions,
        shortest_action_accuracy=fields.shortest_action_accuracy,
        survival_curve=fields.survival_curve,
        submit_error_bands=fields.submit_error_bands,
    )


def build_diagnostic_bundle(
    *,
    device: str,
    bound_mask: bool,
    reports: Sequence[DiagnosticReport],
) -> DiagnosticBundle:
    """Build a closed diagnostic bundle from one legacy/smoke or three final reports."""

    normalized_reports = _typed_tuple(reports, DiagnosticReport, "reports", nonempty=True)
    first = normalized_reports[0]
    return DiagnosticBundle(
        schema_id=DIAGNOSTIC_BUNDLE_SCHEMA_ID,
        suite_id=first.suite_id,
        suite_digest_sha256=first.suite_digest_sha256,
        device=device,
        bound_mask=bound_mask,
        reports=normalized_reports,
        inventory=_report_inventory(normalized_reports),
    )


def _derive_episode(
    source_pitch_cents: int,
    target_note_index: int,
    inputs: tuple[DiagnosticDecisionInput, ...],
) -> tuple[tuple[DecisionDiagnostic, ...], _EpisodeFacts]:
    if len(inputs) > MAX_STEPS:
        raise ValueError(f"decisions must contain at most {MAX_STEPS} decisions")
    submit_positions = tuple(
        index for index, item in enumerate(inputs) if item.action is PitchAction.SUBMIT
    )
    if submit_positions and submit_positions != (len(inputs) - 1,):
        raise ValueError("Submit may appear only as the final decision")
    if not submit_positions and len(inputs) != MAX_STEPS:
        raise ValueError("decision sequence must terminate with Submit or exhaust the budget")

    target_pitch_cents = 100 * (TARGET_MIN_COORDINATE + target_note_index)
    base_error_cents = source_pitch_cents - target_pitch_cents
    controls = ZERO_CONTROLS
    seen_states: set[ControlState] = set()
    decisions: list[DecisionDiagnostic] = []
    for step, decision_input in enumerate(inputs, start=1):
        action = decision_input.action
        steps_remaining = MAX_STEPS - step + 1
        plan = minimum_action_plan(
            base_error_cents, controls, tolerance_cents=SUCCESS_TOLERANCE_CENTS
        )
        teacher_action = plan[0]
        true_candidate_cents = source_pitch_cents + controls.offset_cents
        true_absolute_error = abs(true_candidate_cents - target_pitch_cents)
        repeated = controls in seen_states
        seen_states.add(controls)

        if action is PitchAction.SUBMIT:
            successor = controls
            legal = True
            bound_blocked = False
            loop = False
            in_shortest_set = true_absolute_error <= SUCCESS_TOLERANCE_CENTS
        else:
            successor, applied = controls.apply(action)
            legal = applied
            bound_blocked = not applied
            loop = successor in seen_states
            in_shortest_set = (
                legal
                and len(
                    minimum_action_plan(
                        base_error_cents,
                        successor,
                        tolerance_cents=SUCCESS_TOLERANCE_CENTS,
                    )
                )
                == len(plan) - 1
            )

        estimate = None
        if decision_input.estimated_candidate_cents is not None:
            signed_estimate_error = decision_input.estimated_candidate_cents - true_candidate_cents
            estimate = EstimateDiagnostic(
                estimated_candidate_cents=decision_input.estimated_candidate_cents,
                signed_error_cents=signed_estimate_error,
                absolute_error_cents=abs(signed_estimate_error),
            )
        decisions.append(
            DecisionDiagnostic(
                step=step,
                controls=controls,
                steps_remaining=steps_remaining,
                true_candidate_cents=true_candidate_cents,
                true_absolute_error_cents=true_absolute_error,
                teacher_action=teacher_action,
                predicted_action=action,
                canonical_match=action is teacher_action,
                in_shortest_action_set=in_shortest_set,
                legal=legal,
                bound_blocked=bound_blocked,
                repeated_state_visit=repeated,
                loop_transition=loop,
                shortest_plan_length=len(plan),
                recoverable=len(plan) <= steps_remaining,
                estimate=estimate,
            )
        )
        controls = successor

    normalized = tuple(decisions)
    final_absolute_error = abs(source_pitch_cents + controls.offset_cents - target_pitch_cents)
    if inputs[-1].action is PitchAction.SUBMIT:
        if final_absolute_error <= SUCCESS_TOLERANCE_CENTS:
            terminal_reason = TerminalReason.SUBMITTED_SUCCESS
            excess_actions = len(inputs) - len(
                minimum_action_plan(
                    base_error_cents,
                    ZERO_CONTROLS,
                    tolerance_cents=SUCCESS_TOLERANCE_CENTS,
                )
            )
            if excess_actions < 0:
                raise RuntimeError("successful trajectory cannot be shorter than the minimum plan")
        else:
            terminal_reason = TerminalReason.SUBMITTED_FAILURE
            excess_actions = None
    else:
        terminal_reason = TerminalReason.BUDGET_EXHAUSTED
        excess_actions = None

    first_mismatch = next((item for item in normalized if not item.canonical_match), None)
    first_divergence = next((item for item in normalized if not item.in_shortest_action_set), None)
    first_loop = next((item for item in normalized if item.loop_transition), None)
    first_unrecoverable = next((item for item in normalized if not item.recoverable), None)
    return normalized, _EpisodeFacts(
        first_canonical_mismatch_step=None if first_mismatch is None else first_mismatch.step,
        first_canonical_mismatch_action=(
            None if first_mismatch is None else first_mismatch.predicted_action
        ),
        first_shortest_divergence_step=(
            None if first_divergence is None else first_divergence.step
        ),
        first_shortest_divergence_action=(
            None if first_divergence is None else first_divergence.predicted_action
        ),
        first_loop_step=None if first_loop is None else first_loop.step,
        first_unrecoverable_step=(
            None if first_unrecoverable is None else first_unrecoverable.step
        ),
        legal_action_count=sum(item.legal for item in normalized),
        bound_blocked_action_count=sum(item.bound_blocked for item in normalized),
        repeated_state_visit_count=sum(item.repeated_state_visit for item in normalized),
        loop_transition_count=sum(item.loop_transition for item in normalized),
        terminal_reason=terminal_reason,
        final_absolute_error_cents=final_absolute_error,
        excess_actions=excess_actions,
    )


def _derive_report_fields(episodes: tuple[EpisodeDiagnostic, ...]) -> _ReportFields:
    matrix = [[0 for _ in PitchAction] for _ in PitchAction]
    all_decisions = tuple(decision for episode in episodes for decision in episode.decisions)
    for decision in all_decisions:
        matrix[int(decision.teacher_action)][int(decision.predicted_action)] += 1
    confusion = tuple(tuple(row) for row in matrix)
    metrics: list[PerActionMetrics] = []
    for action in PitchAction:
        index = int(action)
        true_positive = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        support = sum(confusion[index])
        metrics.append(
            PerActionMetrics(
                action=action,
                precision_numerator=true_positive,
                precision_denominator=predicted,
                precision=None if predicted == 0 else true_positive / predicted,
                recall_numerator=true_positive,
                recall_denominator=support,
                recall=None if support == 0 else true_positive / support,
                support=support,
            )
        )
    shortest_correct = sum(decision.in_shortest_action_set for decision in all_decisions)
    shortest_total = len(all_decisions)
    survival: list[SurvivalEntry] = []
    for step in range(1, max(len(episode.decisions) for episode in episodes) + 1):
        at_risk_episodes = tuple(episode for episode in episodes if len(episode.decisions) >= step)
        if not at_risk_episodes:
            break
        survivors = sum(
            episode.first_shortest_divergence_step is None
            or episode.first_shortest_divergence_step > step
            for episode in at_risk_episodes
        )
        survival.append(
            SurvivalEntry(
                step=step,
                at_risk=len(at_risk_episodes),
                survivors=survivors,
                rate=survivors / len(at_risk_episodes),
            )
        )
    bands = tuple(
        SubmitErrorBand(
            label=label,
            minimum_cents=minimum,
            maximum_cents=maximum,
            count=sum(
                decision.predicted_action is PitchAction.SUBMIT
                and decision.true_absolute_error_cents >= minimum
                and (maximum is None or decision.true_absolute_error_cents <= maximum)
                for decision in all_decisions
            ),
        )
        for label, minimum, maximum in _SUBMIT_BANDS
    )
    return _ReportFields(
        confusion_matrix=confusion,
        per_action_metrics=tuple(metrics),
        shortest_action_correct=shortest_correct,
        shortest_action_decisions=shortest_total,
        shortest_action_accuracy=shortest_correct / shortest_total,
        survival_curve=tuple(survival),
        submit_error_bands=bands,
    )


def _confusion_matrix(value: object) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("confusion_matrix must be a 7-by-7 integer matrix")
    try:
        rows = tuple(tuple(row) for row in value)  # type: ignore[union-attr]
    except TypeError as error:
        raise ValueError("confusion_matrix must be a 7-by-7 integer matrix") from error
    if len(rows) != len(PitchAction) or any(len(row) != len(PitchAction) for row in rows):
        raise ValueError("confusion_matrix must be a 7-by-7 integer matrix")
    return tuple(
        tuple(_integer(item, "confusion count", minimum=0) for item in row) for row in rows
    )


def _report_inventory(
    reports: tuple[DiagnosticReport, ...],
) -> tuple[DiagnosticReportInventory, ...]:
    # Local import avoids making the pure trajectory layer own JSON mechanics.
    from harpy.learning.diagnostic_codecs import diagnostic_report_bytes

    return tuple(
        DiagnosticReportInventory(
            seed=report.seed,
            artifact_manifest_sha256=report.artifact_manifest_sha256,
            report_sha256=hashlib.sha256(diagnostic_report_bytes(report)).hexdigest(),
        )
        for report in reports
    )


__all__ = [
    "DIAGNOSTIC_BUNDLE_SCHEMA_ID",
    "DIAGNOSTIC_REPORT_SCHEMA_ID",
    "DecisionDiagnostic",
    "DiagnosticBundle",
    "DiagnosticDecisionInput",
    "DiagnosticReport",
    "DiagnosticReportInventory",
    "EpisodeDiagnostic",
    "EstimateDiagnostic",
    "PerActionMetrics",
    "SubmitErrorBand",
    "SurvivalEntry",
    "build_diagnostic_bundle",
    "build_diagnostic_report",
    "diagnose_episode",
]
