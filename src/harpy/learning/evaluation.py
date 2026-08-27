"""Matched terminal evaluation and scientific criteria for learned sine policies."""

from __future__ import annotations

import math
import operator
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np

from harpy.envs.baselines import (
    BaselineKind,
    RandomPolicy,
    RewardSearchPolicy,
    oracle_plan,
    spectrum_peak_plan,
)
from harpy.envs.models import (
    MAX_ABSOLUTE_ERROR_CENTS,
    MAX_STEPS,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.learning.actors import Actor
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.envs import make_cached_sine_pitch_env
from harpy.learning.models import (
    ENVIRONMENT_ID,
    EpisodeSpec,
    EpisodeSuite,
    EvaluationSuiteId,
    JSONValue,
    TrainerKind,
)

EVALUATION_SCHEMA_VERSION = 1
ZERO_SPECTRUM_PROBE = "zero_spectrum"
SHUFFLED_SPECTRUM_PROBE = "shuffled_spectrum"
SPECTRUM_SHUFFLE_ID = "harpy-sine-spectrum-shuffle-v1"

SPECTRUM_SHUFFLE_PERMUTATION = np.random.default_rng(
    np.random.SeedSequence([202_608_103, 1])
).permutation(1_961)
SPECTRUM_SHUFFLE_PERMUTATION.setflags(write=False)

_SUBSETS = frozenset({"combined", "lower", "upper"})
_PROBES = frozenset({ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE})


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _optional_nonnegative_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field)


def _optional_nonnegative_float(value: object, field: str) -> float | None:
    return None if value is None else _finite_float(value, field, minimum=0.0)


@dataclass(frozen=True, slots=True)
class TerminalEpisodeRecord:
    """Immutable evaluator-owned copy of one environment terminal result."""

    episode_index: int
    episode: EpisodeSpec
    submitted_success: bool
    within_5_cents: bool
    within_1_cent: bool
    final_absolute_error_cents: int
    action_count: int
    excess_actions: int | None
    invalid_action_count: int
    total_return: float
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_index", _integer(self.episode_index, "episode_index"))
        if not isinstance(self.episode, EpisodeSpec):
            raise ValueError("episode must be an EpisodeSpec")
        for field in ("submitted_success", "within_5_cents", "within_1_cent"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a bool")
        final_error = _integer(self.final_absolute_error_cents, "final_absolute_error_cents")
        if final_error > MAX_ABSOLUTE_ERROR_CENTS:
            raise ValueError(
                f"final_absolute_error_cents must be at most {MAX_ABSOLUTE_ERROR_CENTS}"
            )
        action_count = _integer(self.action_count, "action_count", minimum=1)
        if action_count > MAX_STEPS:
            raise ValueError(f"action_count must be at most {MAX_STEPS}")
        invalid_count = _integer(self.invalid_action_count, "invalid_action_count")
        if invalid_count > action_count:
            raise ValueError("invalid_action_count must not exceed action_count")
        excess_actions = _optional_nonnegative_integer(self.excess_actions, "excess_actions")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise ValueError("terminal_reason must be a TerminalReason")
        if self.within_1_cent and not self.within_5_cents:
            raise ValueError("within_1_cents requires within_5_cents")
        if self.within_5_cents != (final_error <= 5):
            raise ValueError("within_5_cents must match final_absolute_error_cents")
        if self.within_1_cent != (final_error <= 1):
            raise ValueError("within_1_cent must match final_absolute_error_cents")
        if self.submitted_success:
            if self.terminal_reason is not TerminalReason.SUBMITTED_SUCCESS:
                raise ValueError("submitted_success requires submitted_success terminal reason")
            if not self.within_5_cents or excess_actions is None:
                raise ValueError("submitted_success requires tolerance and excess actions")
        elif excess_actions is not None:
            raise ValueError("only submitted successes may report excess_actions")
        if excess_actions is not None and excess_actions >= action_count:
            raise ValueError("excess_actions must be less than action_count")
        if self.terminal_reason is TerminalReason.SUBMITTED_SUCCESS and not self.submitted_success:
            raise ValueError("submitted_success terminal reason requires submitted_success")
        if self.terminal_reason is TerminalReason.SUBMITTED_FAILURE and self.within_5_cents:
            raise ValueError("failed submissions must finish outside tolerance")
        if self.terminal_reason is TerminalReason.BUDGET_EXHAUSTED and action_count != MAX_STEPS:
            raise ValueError("budget exhaustion requires MAX_STEPS actions")
        object.__setattr__(self, "final_absolute_error_cents", final_error)
        object.__setattr__(self, "action_count", action_count)
        object.__setattr__(self, "excess_actions", excess_actions)
        object.__setattr__(self, "invalid_action_count", invalid_count)
        object.__setattr__(self, "total_return", _finite_float(self.total_return, "total_return"))


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Exact aggregate metrics over a nonempty terminal-record denominator."""

    episodes: int
    submitted_success_rate: float
    submitted_within_1_cent_rate: float
    final_within_5_cents_rate: float
    final_within_1_cent_rate: float
    mean_absolute_final_error_cents: float
    median_absolute_final_error_cents: float
    mean_actions: float
    mean_successful_excess_actions: float | None
    mean_return: float
    truncation_rate: float
    invalid_action_rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "episodes", _integer(self.episodes, "episodes", minimum=1))
        for field in (
            "submitted_success_rate",
            "submitted_within_1_cent_rate",
            "final_within_5_cents_rate",
            "final_within_1_cent_rate",
            "truncation_rate",
            "invalid_action_rate",
        ):
            object.__setattr__(
                self,
                field,
                _finite_float(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )
        for field in (
            "mean_absolute_final_error_cents",
            "median_absolute_final_error_cents",
            "mean_actions",
        ):
            object.__setattr__(
                self,
                field,
                _finite_float(getattr(self, field), field, minimum=0.0),
            )
        object.__setattr__(
            self,
            "mean_successful_excess_actions",
            _optional_nonnegative_float(
                self.mean_successful_excess_actions,
                "mean_successful_excess_actions",
            ),
        )
        object.__setattr__(self, "mean_return", _finite_float(self.mean_return, "mean_return"))


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One capability- and subset-specific evaluation row."""

    actor_id: str
    trainer: TrainerKind | None
    seed: int | None
    environment_id: str
    observation_mode: ObservationMode
    suite_id: EvaluationSuiteId
    suite_digest_sha256: str
    subset: str
    probe: str | None
    parameter_count: int | None
    training_environment_steps: int | None
    training_examples: int | None
    training_wall_time_seconds: float | None
    metrics: AggregateMetrics
    episodes: tuple[TerminalEpisodeRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("actor_id must be a nonempty string")
        if self.trainer is not None and not isinstance(self.trainer, TrainerKind):
            raise ValueError("trainer must be a TrainerKind or None")
        object.__setattr__(self, "seed", _optional_nonnegative_integer(self.seed, "seed"))
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError("environment_id must be a nonempty string")
        if not isinstance(self.observation_mode, ObservationMode):
            raise ValueError("observation_mode must be an ObservationMode")
        if self.trainer is None:
            try:
                baseline_kind = BaselineKind(self.actor_id)
            except ValueError as error:
                raise ValueError("baseline actor_id must name a declared baseline") from error
            if (
                self.environment_id != baseline_kind.environment_id
                or self.observation_mode is not baseline_kind.observation_mode
            ):
                raise ValueError("row must retain its declared baseline capability lane")
            if self.seed is not None:
                raise ValueError("baseline rows must not set a trainer seed")
        elif (
            self.environment_id != ENVIRONMENT_ID
            or self.observation_mode is not ObservationMode.SPECTRUM
        ):
            raise ValueError("learned rows must use the declared spectrum environment lane")
        if not isinstance(self.suite_id, EvaluationSuiteId):
            raise ValueError("suite_id must be an EvaluationSuiteId")
        if (
            not isinstance(self.suite_digest_sha256, str)
            or len(self.suite_digest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.suite_digest_sha256)
        ):
            raise ValueError("suite_digest_sha256 must be a lowercase SHA-256 hex digest")
        if self.subset not in _SUBSETS:
            raise ValueError("subset must be combined, lower, or upper")
        if self.probe is not None and self.probe not in _PROBES:
            raise ValueError("probe must be zero_spectrum, shuffled_spectrum, or None")
        if self.probe is not None and self.observation_mode is not ObservationMode.SPECTRUM:
            raise ValueError("spectrum probes require spectrum observation mode")
        if not isinstance(self.metrics, AggregateMetrics):
            raise ValueError("metrics must be AggregateMetrics")
        try:
            episodes = tuple(self.episodes)
        except TypeError as error:
            raise ValueError("episodes must contain TerminalEpisodeRecord values") from error
        if not episodes or not all(isinstance(item, TerminalEpisodeRecord) for item in episodes):
            raise ValueError("episodes must contain TerminalEpisodeRecord values")
        if self.metrics.episodes != len(episodes):
            raise ValueError("metrics episode denominator must match episodes")
        if self.metrics != aggregate_episode_records(episodes):
            raise ValueError("metrics must exactly aggregate episodes")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(
            self,
            "parameter_count",
            _optional_nonnegative_integer(self.parameter_count, "parameter_count"),
        )
        object.__setattr__(
            self,
            "training_environment_steps",
            _optional_nonnegative_integer(
                self.training_environment_steps, "training_environment_steps"
            ),
        )
        object.__setattr__(
            self,
            "training_examples",
            _optional_nonnegative_integer(self.training_examples, "training_examples"),
        )
        object.__setattr__(
            self,
            "training_wall_time_seconds",
            _optional_nonnegative_float(
                self.training_wall_time_seconds, "training_wall_time_seconds"
            ),
        )


@dataclass(frozen=True, slots=True)
class BCScientificCriterion:
    """The two-threshold BC scientific result, distinct from engineering status."""

    eligible: bool
    heldout_next_action_accuracy: float | None
    iid_submitted_success_rate: float | None
    criterion_met: bool | None
    status: str

    def __post_init__(self) -> None:
        _validate_criterion_state(
            self.eligible,
            self.criterion_met,
            self.status,
            values=(
                self.heldout_next_action_accuracy,
                self.iid_submitted_success_rate,
            ),
        )
        if self.eligible:
            object.__setattr__(
                self,
                "heldout_next_action_accuracy",
                _finite_float(
                    self.heldout_next_action_accuracy,
                    "heldout_next_action_accuracy",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
            object.__setattr__(
                self,
                "iid_submitted_success_rate",
                _finite_float(
                    self.iid_submitted_success_rate,
                    "iid_submitted_success_rate",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )


@dataclass(frozen=True, slots=True)
class PPOScientificCriterion:
    """The exact five-seed PPO scientific result."""

    eligible: bool
    median_iid_submitted_success_rate: float | None
    seeds_strictly_beating_random: int | None
    criterion_met: bool | None
    status: str

    def __post_init__(self) -> None:
        _validate_criterion_state(
            self.eligible,
            self.criterion_met,
            self.status,
            values=(
                self.median_iid_submitted_success_rate,
                self.seeds_strictly_beating_random,
            ),
        )
        if self.eligible:
            object.__setattr__(
                self,
                "median_iid_submitted_success_rate",
                _finite_float(
                    self.median_iid_submitted_success_rate,
                    "median_iid_submitted_success_rate",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
            beating = _integer(
                self.seeds_strictly_beating_random,
                "seeds_strictly_beating_random",
            )
            if beating > 5:
                raise ValueError("seeds_strictly_beating_random must be at most 5")
            object.__setattr__(self, "seeds_strictly_beating_random", beating)


def _validate_criterion_state(
    eligible: object,
    criterion_met: object,
    status: object,
    *,
    values: tuple[object, object],
) -> None:
    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a bool")
    if eligible:
        if not isinstance(criterion_met, bool):
            raise ValueError("eligible criteria require a bool criterion_met")
        expected = "criterion_met" if criterion_met else "criterion_not_met"
        if status != expected:
            raise ValueError(f"eligible criterion status must be {expected}")
        if any(value is None for value in values):
            raise ValueError("eligible criteria require computed values")
    else:
        if (
            criterion_met is not None
            or status != "ineligible"
            or any(value is not None for value in values)
        ):
            raise ValueError("ineligible criteria must not contain computed values")


@dataclass(frozen=True, slots=True)
class EvaluationFile:
    """Strict schema-v1 trainer evaluation payload."""

    schema_version: int
    suite_id: EvaluationSuiteId
    suite_digest_sha256: str
    rows: tuple[EvaluationRow, ...]
    next_action_accuracy: float | None

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVALUATION_SCHEMA_VERSION}")
        if not isinstance(self.suite_id, EvaluationSuiteId):
            raise ValueError("suite_id must be an EvaluationSuiteId")
        if (
            not isinstance(self.suite_digest_sha256, str)
            or len(self.suite_digest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.suite_digest_sha256)
        ):
            raise ValueError("suite_digest_sha256 must be a lowercase SHA-256 hex digest")
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise ValueError("rows must contain EvaluationRow values") from error
        if not rows or not all(isinstance(row, EvaluationRow) for row in rows):
            raise ValueError("rows must contain EvaluationRow values")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "rows", rows)
        if self.next_action_accuracy is not None:
            object.__setattr__(
                self,
                "next_action_accuracy",
                _finite_float(
                    self.next_action_accuracy,
                    "next_action_accuracy",
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        self._validate_rows()

    def _validate_rows(self) -> None:
        from harpy.learning.suites import fixed_evaluation_suite

        suite = fixed_evaluation_suite(self.suite_id)
        if self.suite_digest_sha256 != suite.digest_sha256:
            raise ValueError("suite digest must match the fixed suite")
        identities: set[tuple[object, ...]] = set()
        for row in self.rows:
            if (
                row.suite_id is not self.suite_id
                or row.suite_digest_sha256 != self.suite_digest_sha256
            ):
                raise ValueError("row suite identity must match the evaluation file suite")
            identity = (
                row.actor_id,
                row.trainer,
                row.seed,
                row.environment_id,
                row.observation_mode,
                row.subset,
                row.probe,
            )
            if identity in identities:
                raise ValueError("rows must not contain duplicate identities")
            identities.add(identity)
            expected_episodes = _expected_subset_membership(suite, row.subset)
            actual_episodes = tuple(
                (record.episode_index, record.episode) for record in row.episodes
            )
            if actual_episodes != expected_episodes:
                raise ValueError("rows must preserve the fixed suite's ordered episode membership")
        if self.suite_id is EvaluationSuiteId.REGISTER_OOD:
            if any(row.probe is not None for row in self.rows):
                raise ValueError("register-OOD rows must remain unperturbed")
            if len(self.rows) % 3 != 0:
                raise ValueError("register-OOD rows must contain lower, upper, combined groups")
            for start in range(0, len(self.rows), 3):
                group = self.rows[start : start + 3]
                if tuple(row.subset for row in group) != ("lower", "upper", "combined"):
                    raise ValueError("register-OOD rows must be ordered lower, upper, combined")
                if len({_row_group_identity(row) for row in group}) != 1:
                    raise ValueError("register-OOD subset rows must describe the same actor")
        elif any(row.subset != "combined" for row in self.rows):
            raise ValueError("smoke and IID rows must use the combined subset")
        if self.next_action_accuracy is not None and (
            self.suite_id is EvaluationSuiteId.REGISTER_OOD
            or any(row.trainer is not TrainerKind.BC for row in self.rows)
        ):
            raise ValueError("next_action_accuracy is allowed only for BC smoke or IID files")

    def to_document(self) -> dict[str, JSONValue]:
        """Return a freshly owned strict JSON-compatible document."""

        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id.value,
            "suite_digest_sha256": self.suite_digest_sha256,
            "rows": [_row_to_document(row) for row in self.rows],
            "next_action_accuracy": self.next_action_accuracy,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> EvaluationFile:
        """Decode and validate an exact schema-v1 evaluation document."""

        mapping = _mapping(document, "evaluation file")
        _exact_fields(
            mapping,
            {
                "schema_version",
                "suite_id",
                "suite_digest_sha256",
                "rows",
                "next_action_accuracy",
            },
            "evaluation file",
        )
        rows_value = mapping["rows"]
        if not isinstance(rows_value, list):
            raise ValueError("rows must be a list")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            suite_id=_enum(EvaluationSuiteId, mapping["suite_id"], "suite_id"),
            suite_digest_sha256=_string(mapping["suite_digest_sha256"], "suite_digest_sha256"),
            rows=tuple(_row_from_document(value) for value in rows_value),
            next_action_accuracy=(
                None
                if mapping["next_action_accuracy"] is None
                else _document_float(
                    mapping["next_action_accuracy"],
                    "next_action_accuracy",
                    minimum=0.0,
                    maximum=1.0,
                )
            ),
        )


def _record_episode_result(
    episode_index: int,
    episode: EpisodeSpec,
    result: EpisodeResult,
) -> TerminalEpisodeRecord:
    if (result.target_note_index, result.source_pitch_cents) != episode.pair:
        raise RuntimeError("terminal EpisodeResult does not match the injected EpisodeSpec")
    return TerminalEpisodeRecord(
        episode_index=episode_index,
        episode=episode,
        submitted_success=result.submitted_success,
        within_5_cents=result.within_5_cents,
        within_1_cent=result.within_1_cent,
        final_absolute_error_cents=result.final_absolute_error_cents,
        action_count=len(result.actions),
        excess_actions=result.excess_actions,
        invalid_action_count=result.invalid_action_count,
        total_return=result.total_return,
        terminal_reason=result.terminal_reason,
    )


def _terminal_result(env: gymnasium.Env) -> EpisodeResult:
    result = env.unwrapped.episode_result
    if not isinstance(result, EpisodeResult):
        raise RuntimeError("environment produced no EpisodeResult after done")
    return result


def _validate_suite(suite: object) -> EpisodeSuite:
    if not isinstance(suite, EpisodeSuite):
        raise ValueError("suite must be an EpisodeSuite")
    from harpy.learning.suites import suite_digest

    expected = suite_digest(
        suite_id=suite.suite_id,
        suite_seed=suite.suite_seed,
        episodes=suite.episodes,
    )
    if suite.digest_sha256 != expected:
        raise ValueError("suite digest must match its ordered episode membership")
    return suite


def evaluate_learned_actor(
    actor: Actor,
    suite: EpisodeSuite,
    *,
    environment_factory: Callable[[], gymnasium.Env],
) -> tuple[TerminalEpisodeRecord, ...]:
    """Roll out a narrow actor on the suite without exposing transition feedback."""

    suite = _validate_suite(suite)
    if not callable(environment_factory):
        raise ValueError("environment_factory must be callable")
    env = environment_factory()
    records: list[TerminalEpisodeRecord] = []
    try:
        for episode_index, episode in enumerate(suite.episodes):
            observation, _ = env.reset(options=episode.reset_options())
            while True:
                action = actor.act(observation)
                if not isinstance(action, PitchAction):
                    raise RuntimeError("actor must return a PitchAction")
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    result = _terminal_result(env)
                    records.append(_record_episode_result(episode_index, episode, result))
                    break
    finally:
        env.close()
    return tuple(records)


class _PlannerActor:
    """Episode-local adapter whose private cursor cannot cross episode boundaries."""

    def __init__(self, planner: Callable[[Mapping[str, object]], tuple[PitchAction, ...]]) -> None:
        self._planner = planner
        self._actions: tuple[PitchAction, ...] | None = None
        self._cursor = 0

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        if self._actions is None:
            self._actions = self._planner(observation)
        action = self._actions[self._cursor]
        self._cursor += 1
        return action


def evaluate_baseline_suite(
    kind: BaselineKind,
    suite: EpisodeSuite,
    *,
    cache: SpectrumEvidenceCache,
) -> tuple[TerminalEpisodeRecord, ...]:
    """Roll out one baseline lane on the exact injected suite membership."""

    if not isinstance(kind, BaselineKind):
        raise ValueError("kind must be a BaselineKind")
    suite = _validate_suite(suite)
    if not isinstance(cache, SpectrumEvidenceCache):
        raise ValueError("cache must be a SpectrumEvidenceCache")
    env = make_cached_sine_pitch_env(cache, observation_mode=kind.observation_mode)
    records: list[TerminalEpisodeRecord] = []
    try:
        for episode_index, episode in enumerate(suite.episodes):
            observation, _ = env.reset(options=episode.reset_options())
            if kind is BaselineKind.RANDOM:
                seed_sequence = np.random.SeedSequence([suite.suite_seed, 1, episode_index])
                policy: Any = RandomPolicy(np.random.default_rng(seed_sequence))
            elif kind is BaselineKind.REWARD_SEARCH:
                policy = RewardSearchPolicy()
            elif kind is BaselineKind.SPECTRUM_PEAK:
                policy = _PlannerActor(spectrum_peak_plan)
            else:
                policy = _PlannerActor(oracle_plan)
            previous_reward: float | None = None
            previous_info: Mapping[str, object] | None = None
            while True:
                if kind is BaselineKind.REWARD_SEARCH:
                    action = policy.act(observation, previous_reward, previous_info)
                elif kind is BaselineKind.RANDOM:
                    action = policy.act(observation, None, None)
                else:
                    action = policy.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    records.append(
                        _record_episode_result(episode_index, episode, _terminal_result(env))
                    )
                    break
                if kind is BaselineKind.REWARD_SEARCH:
                    previous_reward = reward
                    previous_info = info
    finally:
        env.close()
    return tuple(records)


def aggregate_episode_records(
    records: Sequence[TerminalEpisodeRecord],
) -> AggregateMetrics:
    """Aggregate terminal records using their exact episode/action denominators."""

    try:
        normalized = tuple(records)
    except TypeError as error:
        raise ValueError("records must contain TerminalEpisodeRecord values") from error
    if not normalized or not all(
        isinstance(record, TerminalEpisodeRecord) for record in normalized
    ):
        raise ValueError("records must contain TerminalEpisodeRecord values")
    episodes = len(normalized)
    actions = sum(record.action_count for record in normalized)
    successful_excess = tuple(
        record.excess_actions
        for record in normalized
        if record.submitted_success and record.excess_actions is not None
    )
    return AggregateMetrics(
        episodes=episodes,
        submitted_success_rate=sum(record.submitted_success for record in normalized) / episodes,
        submitted_within_1_cent_rate=sum(
            record.submitted_success and record.within_1_cent for record in normalized
        )
        / episodes,
        final_within_5_cents_rate=sum(record.within_5_cents for record in normalized) / episodes,
        final_within_1_cent_rate=sum(record.within_1_cent for record in normalized) / episodes,
        mean_absolute_final_error_cents=sum(
            record.final_absolute_error_cents for record in normalized
        )
        / episodes,
        median_absolute_final_error_cents=float(
            statistics.median(record.final_absolute_error_cents for record in normalized)
        ),
        mean_actions=actions / episodes,
        mean_successful_excess_actions=(
            sum(successful_excess) / len(successful_excess) if successful_excess else None
        ),
        mean_return=sum(record.total_return for record in normalized) / episodes,
        truncation_rate=sum(
            record.terminal_reason is TerminalReason.BUDGET_EXHAUSTED for record in normalized
        )
        / episodes,
        invalid_action_rate=sum(record.invalid_action_count for record in normalized) / actions,
    )


def build_evaluation_rows(
    *,
    actor_id: str,
    trainer: TrainerKind | None,
    seed: int | None,
    environment_id: str,
    observation_mode: ObservationMode,
    suite: EpisodeSuite,
    records: Sequence[TerminalEpisodeRecord],
    probe: str | None = None,
    parameter_count: int | None = None,
    training_environment_steps: int | None = None,
    training_examples: int | None = None,
    training_wall_time_seconds: float | None = None,
) -> tuple[EvaluationRow, ...]:
    """Build combined or exact lower/upper/combined OOD rows in report order."""

    suite = _validate_suite(suite)
    normalized = tuple(records)
    if tuple((record.episode_index, record.episode) for record in normalized) != tuple(
        enumerate(suite.episodes)
    ):
        raise ValueError("records must preserve the suite's ordered episode membership")
    common = {
        "actor_id": actor_id,
        "trainer": trainer,
        "seed": seed,
        "environment_id": environment_id,
        "observation_mode": observation_mode,
        "suite_id": suite.suite_id,
        "suite_digest_sha256": suite.digest_sha256,
        "probe": probe,
        "parameter_count": parameter_count,
        "training_environment_steps": training_environment_steps,
        "training_examples": training_examples,
        "training_wall_time_seconds": training_wall_time_seconds,
    }

    def row(subset: str, members: tuple[TerminalEpisodeRecord, ...]) -> EvaluationRow:
        return EvaluationRow(
            **common,
            subset=subset,
            metrics=aggregate_episode_records(members),
            episodes=members,
        )

    if suite.suite_id is not EvaluationSuiteId.REGISTER_OOD:
        return (row("combined", normalized),)
    if probe is not None:
        raise ValueError("register-OOD rows must remain unperturbed")
    lower = tuple(record for record in normalized if record.episode.source_pitch_cents <= 4_999)
    upper = tuple(record for record in normalized if record.episode.source_pitch_cents >= 7_001)
    if len(lower) + len(upper) != len(normalized) or not lower or not upper:
        raise ValueError("register-OOD episodes must belong to lower or upper source bands")
    return row("lower", lower), row("upper", upper), row("combined", normalized)


class _SpectrumProbeWrapper(gymnasium.ObservationWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        probe: str,
        *,
        shuffle_permutation: np.ndarray[Any, np.dtype[np.int64]] | None = None,
    ) -> None:
        super().__init__(env)
        if probe not in _PROBES:
            raise ValueError("probe must be zero_spectrum or shuffled_spectrum")
        self._probe = probe
        self._shuffle_permutation = (
            SPECTRUM_SHUFFLE_PERMUTATION if shuffle_permutation is None else shuffle_permutation
        )

    def observation(self, observation: Mapping[str, object]) -> dict[str, object]:
        if "spectrum" not in observation:
            raise RuntimeError("spectrum probe requires a spectrum observation")
        transformed = dict(observation)
        spectrum = np.array(observation["spectrum"], dtype=np.float32, copy=True, order="C")
        if spectrum.shape != (1_961,):
            raise RuntimeError("spectrum probe requires shape (1961,)")
        if self._probe == ZERO_SPECTRUM_PROBE:
            spectrum.fill(0.0)
        else:
            spectrum = np.array(
                spectrum[self._shuffle_permutation],
                dtype=np.float32,
                copy=True,
                order="C",
            )
        transformed["spectrum"] = spectrum
        return transformed


def make_spectrum_probe_factory(
    environment_factory: Callable[[], gymnasium.Env],
    probe: str,
) -> Callable[[], gymnasium.Env]:
    """Wrap each fresh environment with one deterministic spectrum-only probe."""

    if not callable(environment_factory):
        raise ValueError("environment_factory must be callable")
    if probe not in _PROBES:
        raise ValueError("probe must be zero_spectrum or shuffled_spectrum")

    def factory() -> gymnasium.Env:
        return _SpectrumProbeWrapper(environment_factory(), probe)

    return factory


def make_indexed_spectrum_probe_factory(
    environment_factory: Callable[[], gymnasium.Env],
    probe: str,
    *,
    shuffle_permutation: Sequence[int],
) -> Callable[[], gymnasium.Env]:
    """Build a probe wrapper with a caller-pinned complete spectrum permutation.

    Schema v1 continues to use :func:`make_spectrum_probe_factory` and its historical
    global permutation.  This explicit seam lets schema v2 supply its digest-derived
    permutation without changing legacy behavior.
    """

    if not callable(environment_factory):
        raise ValueError("environment_factory must be callable")
    if probe not in _PROBES:
        raise ValueError("probe must be zero_spectrum or shuffled_spectrum")
    try:
        raw_permutation = tuple(shuffle_permutation)
        permutation = np.asarray(
            tuple(_integer(index, "shuffle_permutation", minimum=0) for index in raw_permutation),
            dtype=np.int64,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("shuffle_permutation must contain every spectrum index once") from error
    if permutation.shape != (1_961,) or not np.array_equal(
        np.sort(permutation), np.arange(1_961, dtype=np.int64)
    ):
        raise ValueError("shuffle_permutation must contain every spectrum index once")
    owned_permutation = np.array(permutation, dtype=np.int64, copy=True, order="C")
    owned_permutation.setflags(write=False)

    def factory() -> gymnasium.Env:
        return _SpectrumProbeWrapper(
            environment_factory(),
            probe,
            shuffle_permutation=owned_permutation,
        )

    return factory


def _validate_iid_row(
    row: object,
    *,
    trainer: TrainerKind | None,
    actor_id: str | None = None,
) -> EvaluationRow:
    if not isinstance(row, EvaluationRow):
        raise ValueError("iid_row must be an EvaluationRow")
    if row.suite_id is not EvaluationSuiteId.IID:
        raise ValueError("scientific criteria require an IID row")
    if row.subset != "combined":
        raise ValueError("scientific criteria require the combined IID subset")
    if row.probe is not None:
        raise ValueError("scientific criteria exclude probe rows")
    if row.trainer is not trainer:
        expected = "baseline" if trainer is None else trainer.value
        raise ValueError(f"IID row must be a {expected} row")
    if actor_id is not None and row.actor_id != actor_id:
        raise ValueError(f"IID row actor_id must be {actor_id}")
    if row.observation_mode is not ObservationMode.SPECTRUM:
        raise ValueError("scientific criteria require spectrum IID rows")
    if row.environment_id != ENVIRONMENT_ID:
        raise ValueError("scientific criteria require the declared spectrum environment")
    from harpy.learning.suites import fixed_evaluation_suite

    canonical_suite = fixed_evaluation_suite(EvaluationSuiteId.IID)
    canonical_membership = tuple(enumerate(canonical_suite.episodes))
    row_membership = tuple((record.episode_index, record.episode) for record in row.episodes)
    if (
        row.suite_digest_sha256 != canonical_suite.digest_sha256
        or row_membership != canonical_membership
    ):
        raise ValueError("scientific criteria require canonical IID digest and membership")
    return row


def evaluate_bc_criterion(
    *,
    heldout_next_action_accuracy: float,
    iid_row: EvaluationRow,
    eligible: bool,
) -> BCScientificCriterion:
    """Evaluate the exact BC thresholds after workflow-derived eligibility."""

    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a bool")
    if not eligible:
        return BCScientificCriterion(False, None, None, None, "ineligible")
    accuracy = _finite_float(
        heldout_next_action_accuracy,
        "heldout_next_action_accuracy",
        minimum=0.0,
        maximum=1.0,
    )
    row = _validate_iid_row(iid_row, trainer=TrainerKind.BC)
    rate = row.metrics.submitted_success_rate
    met = accuracy >= 0.90 and rate >= 0.75
    return BCScientificCriterion(
        eligible=True,
        heldout_next_action_accuracy=accuracy,
        iid_submitted_success_rate=rate,
        criterion_met=met,
        status="criterion_met" if met else "criterion_not_met",
    )


def evaluate_ppo_criterion(
    *,
    iid_rows: Sequence[EvaluationRow],
    random_iid_row: EvaluationRow,
    eligible: bool,
) -> PPOScientificCriterion:
    """Evaluate the exact five-seed PPO median and paired Random thresholds."""

    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a bool")
    if not eligible:
        return PPOScientificCriterion(False, None, None, None, "ineligible")
    try:
        rows = tuple(iid_rows)
    except TypeError as error:
        raise ValueError("iid_rows must contain the exact seeds 0..4") from error
    seeds = tuple(row.seed if isinstance(row, EvaluationRow) else None for row in rows)
    if len(rows) != 5 or len(set(seeds)) != 5 or set(seeds) != set(range(5)):
        raise ValueError("iid_rows must contain exactly the distinct seeds 0..4")
    validated = tuple(_validate_iid_row(row, trainer=TrainerKind.PPO) for row in rows)
    random_row = _validate_iid_row(
        random_iid_row,
        trainer=None,
        actor_id=BaselineKind.RANDOM.value,
    )
    reference_identity = (
        random_row.suite_id,
        random_row.suite_digest_sha256,
        tuple((record.episode_index, record.episode) for record in random_row.episodes),
    )
    if any(
        (
            row.suite_id,
            row.suite_digest_sha256,
            tuple((record.episode_index, record.episode) for record in row.episodes),
        )
        != reference_identity
        for row in validated
    ):
        raise ValueError("PPO and Random rows must use the same IID suite and digest")
    rates = tuple(row.metrics.submitted_success_rate for row in validated)
    median_rate = float(statistics.median(rates))
    beating = sum(rate > random_row.metrics.submitted_success_rate for rate in rates)
    met = median_rate >= 0.50 and beating >= 4
    return PPOScientificCriterion(
        eligible=True,
        median_iid_submitted_success_rate=median_rate,
        seeds_strictly_beating_random=beating,
        criterion_met=met,
        status="criterion_met" if met else "criterion_not_met",
    )


def _expected_subset_membership(
    suite: EpisodeSuite,
    subset: str,
) -> tuple[tuple[int, EpisodeSpec], ...]:
    indexed = tuple(enumerate(suite.episodes))
    if subset == "combined":
        return indexed
    if subset == "lower":
        return tuple(item for item in indexed if item[1].source_pitch_cents <= 4_999)
    return tuple(item for item in indexed if item[1].source_pitch_cents >= 7_001)


def _row_group_identity(row: EvaluationRow) -> tuple[object, ...]:
    return (
        row.actor_id,
        row.trainer,
        row.seed,
        row.environment_id,
        row.observation_mode,
        row.probe,
        row.parameter_count,
        row.training_environment_steps,
        row.training_examples,
        row.training_wall_time_seconds,
    )


def _row_to_document(row: EvaluationRow) -> dict[str, JSONValue]:
    return {
        "actor_id": row.actor_id,
        "trainer": None if row.trainer is None else row.trainer.value,
        "seed": row.seed,
        "environment_id": row.environment_id,
        "observation_mode": row.observation_mode.value,
        "suite_id": row.suite_id.value,
        "suite_digest_sha256": row.suite_digest_sha256,
        "subset": row.subset,
        "probe": row.probe,
        "parameter_count": row.parameter_count,
        "training_environment_steps": row.training_environment_steps,
        "training_examples": row.training_examples,
        "training_wall_time_seconds": row.training_wall_time_seconds,
        "metrics": _metrics_to_document(row.metrics),
        "episodes": [_record_to_document(record) for record in row.episodes],
    }


def _metrics_to_document(metrics: AggregateMetrics) -> dict[str, JSONValue]:
    return {
        "episodes": metrics.episodes,
        "submitted_success_rate": metrics.submitted_success_rate,
        "submitted_within_1_cent_rate": metrics.submitted_within_1_cent_rate,
        "final_within_5_cents_rate": metrics.final_within_5_cents_rate,
        "final_within_1_cent_rate": metrics.final_within_1_cent_rate,
        "mean_absolute_final_error_cents": metrics.mean_absolute_final_error_cents,
        "median_absolute_final_error_cents": metrics.median_absolute_final_error_cents,
        "mean_actions": metrics.mean_actions,
        "mean_successful_excess_actions": metrics.mean_successful_excess_actions,
        "mean_return": metrics.mean_return,
        "truncation_rate": metrics.truncation_rate,
        "invalid_action_rate": metrics.invalid_action_rate,
    }


def _record_to_document(record: TerminalEpisodeRecord) -> dict[str, JSONValue]:
    return {
        "episode_index": record.episode_index,
        "episode": {
            "target_note_index": record.episode.target_note_index,
            "source_pitch_cents": record.episode.source_pitch_cents,
        },
        "submitted_success": record.submitted_success,
        "within_5_cents": record.within_5_cents,
        "within_1_cent": record.within_1_cent,
        "final_absolute_error_cents": record.final_absolute_error_cents,
        "action_count": record.action_count,
        "excess_actions": record.excess_actions,
        "invalid_action_count": record.invalid_action_count,
        "total_return": record.total_return,
        "terminal_reason": record.terminal_reason.value,
    }


def _row_from_document(value: object) -> EvaluationRow:
    mapping = _mapping(value, "evaluation row")
    _exact_fields(
        mapping,
        {
            "actor_id",
            "trainer",
            "seed",
            "environment_id",
            "observation_mode",
            "suite_id",
            "suite_digest_sha256",
            "subset",
            "probe",
            "parameter_count",
            "training_environment_steps",
            "training_examples",
            "training_wall_time_seconds",
            "metrics",
            "episodes",
        },
        "evaluation row",
    )
    episodes_value = mapping["episodes"]
    if not isinstance(episodes_value, list):
        raise ValueError("episodes must be a list")
    return EvaluationRow(
        actor_id=_string(mapping["actor_id"], "actor_id"),
        trainer=(
            None
            if mapping["trainer"] is None
            else _enum(TrainerKind, mapping["trainer"], "trainer")
        ),
        seed=_optional_document_integer(mapping["seed"], "seed"),
        environment_id=_string(mapping["environment_id"], "environment_id"),
        observation_mode=_enum(
            ObservationMode,
            mapping["observation_mode"],
            "observation_mode",
        ),
        suite_id=_enum(EvaluationSuiteId, mapping["suite_id"], "suite_id"),
        suite_digest_sha256=_string(mapping["suite_digest_sha256"], "suite_digest_sha256"),
        subset=_string(mapping["subset"], "subset"),
        probe=None if mapping["probe"] is None else _string(mapping["probe"], "probe"),
        parameter_count=_optional_document_integer(mapping["parameter_count"], "parameter_count"),
        training_environment_steps=_optional_document_integer(
            mapping["training_environment_steps"],
            "training_environment_steps",
        ),
        training_examples=_optional_document_integer(
            mapping["training_examples"], "training_examples"
        ),
        training_wall_time_seconds=(
            None
            if mapping["training_wall_time_seconds"] is None
            else _document_float(
                mapping["training_wall_time_seconds"],
                "training_wall_time_seconds",
                minimum=0.0,
            )
        ),
        metrics=_metrics_from_document(mapping["metrics"]),
        episodes=tuple(_record_from_document(item) for item in episodes_value),
    )


def _metrics_from_document(value: object) -> AggregateMetrics:
    mapping = _mapping(value, "aggregate metrics")
    fields = {
        "episodes",
        "submitted_success_rate",
        "submitted_within_1_cent_rate",
        "final_within_5_cents_rate",
        "final_within_1_cent_rate",
        "mean_absolute_final_error_cents",
        "median_absolute_final_error_cents",
        "mean_actions",
        "mean_successful_excess_actions",
        "mean_return",
        "truncation_rate",
        "invalid_action_rate",
    }
    _exact_fields(mapping, fields, "aggregate metrics")
    arguments: dict[str, object] = {}
    for field in fields:
        if field == "episodes":
            arguments[field] = _integer(mapping[field], field, minimum=1)
        elif field == "mean_successful_excess_actions" and mapping[field] is None:
            arguments[field] = None
        else:
            arguments[field] = _document_float(mapping[field], field)
    return AggregateMetrics(**arguments)  # type: ignore[arg-type]


def _record_from_document(value: object) -> TerminalEpisodeRecord:
    mapping = _mapping(value, "terminal episode record")
    _exact_fields(
        mapping,
        {
            "episode_index",
            "episode",
            "submitted_success",
            "within_5_cents",
            "within_1_cent",
            "final_absolute_error_cents",
            "action_count",
            "excess_actions",
            "invalid_action_count",
            "total_return",
            "terminal_reason",
        },
        "terminal episode record",
    )
    episode = _mapping(mapping["episode"], "episode")
    _exact_fields(episode, {"target_note_index", "source_pitch_cents"}, "episode")
    return TerminalEpisodeRecord(
        episode_index=_integer(mapping["episode_index"], "episode_index"),
        episode=EpisodeSpec(
            target_note_index=_integer(episode["target_note_index"], "target_note_index"),
            source_pitch_cents=_integer(episode["source_pitch_cents"], "source_pitch_cents"),
        ),
        submitted_success=_boolean(mapping["submitted_success"], "submitted_success"),
        within_5_cents=_boolean(mapping["within_5_cents"], "within_5_cents"),
        within_1_cent=_boolean(mapping["within_1_cent"], "within_1_cent"),
        final_absolute_error_cents=_integer(
            mapping["final_absolute_error_cents"], "final_absolute_error_cents"
        ),
        action_count=_integer(mapping["action_count"], "action_count", minimum=1),
        excess_actions=_optional_document_integer(mapping["excess_actions"], "excess_actions"),
        invalid_action_count=_integer(mapping["invalid_action_count"], "invalid_action_count"),
        total_return=_document_float(mapping["total_return"], "total_return"),
        terminal_reason=_enum(TerminalReason, mapping["terminal_reason"], "terminal_reason"),
    )


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(mapping: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _enum(enum_type: type[Any], value: object, field: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a valid {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid {enum_type.__name__}") from error


def _optional_document_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field)


def _document_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number")
    return _finite_float(value, field, minimum=minimum, maximum=maximum)


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "SHUFFLED_SPECTRUM_PROBE",
    "SPECTRUM_SHUFFLE_ID",
    "SPECTRUM_SHUFFLE_PERMUTATION",
    "ZERO_SPECTRUM_PROBE",
    "AggregateMetrics",
    "BCScientificCriterion",
    "EvaluationFile",
    "EvaluationRow",
    "PPOScientificCriterion",
    "TerminalEpisodeRecord",
    "aggregate_episode_records",
    "build_evaluation_rows",
    "evaluate_baseline_suite",
    "evaluate_bc_criterion",
    "evaluate_learned_actor",
    "evaluate_ppo_criterion",
    "make_indexed_spectrum_probe_factory",
    "make_spectrum_probe_factory",
]
