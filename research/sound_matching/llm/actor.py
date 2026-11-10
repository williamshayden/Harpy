"""Bounded cooperative text actor with explicit controller guards and attempt logs.

The HTTP/command transport owns connection details. No endpoint, credentials,
target manifest, or evaluator labels enter this actor's model messages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from harpy.tasks.sound_match.models import ActorSpec, Observation, Propose, Submit
from harpy.tasks.sound_match.objective import score
from harpy.tasks.sound_match.patch import Patch
from research.sound_matching.llm import features

PROTOCOL_ID = "harpy.llm-patch-json/1"
JOURNAL_SCHEMA_ID = "harpy.llm-attempt-journal/1"
MAX_RESPONSE_BYTES = 32_768
HISTORY_TURNS = 3
SYSTEM_PROMPT = """You control a deterministic synthesizer using public audio-derived features.
Recreate the target while changing only editable fields. Spectral peaks may be harmonics;
they are measurements, not ground-truth note labels. Lower spectral distance is better,
but it is not a perceptual success test. All notes use absolute cents: MIDI 60 is 6000.
Return one JSON object and nothing else. No Markdown, explanation, tools, or file access.
Allowed responses:
{"action":"propose","patch":{"notes_cents":[6000],"waveform":"sine",
"brightness_hz":6000,"attack_ms":10,"decay_ms":100,"sustain_db":-6,
"release_ms":200,"gain_db":-6}}
or {"action":"submit"}.
The patch example shows field names, not a recommendation. Supply all fields with legal
integer values; preserve locked fields and note count. Proposals are rendered and charged
by the environment. Submit selects the currently heard candidate, not an earlier patch.
Each reply costs one model call, including malformed replies. There are no hidden retries.
The controller may stop at its call/budget guard and restore the best heard patch before
submitting; such guard actions are controller decisions, not model decisions.
The provided conversation contains at most three previous turns from this episode.
"""


class ResponseError(ValueError):
    """A safe fixed error code, without response or credential text."""


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResponseError("duplicate_json_key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ResponseError("nonfinite_json")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ResponseError("nonfinite_json")
    return number


def parse_action(content: object) -> Propose | Submit:
    """Decode only the exact protocol; never repair, clamp, or infer an action."""
    if not isinstance(content, str):
        raise ResponseError("non_text_content")
    if len(content.encode("utf-8", errors="surrogatepass")) > MAX_RESPONSE_BYTES:
        raise ResponseError("response_too_large")
    try:
        value = json.loads(
            content,
            object_pairs_hook=_object_pairs,
            parse_constant=_nonfinite,
            parse_float=_finite_float,
        )
    except ResponseError:
        raise
    except (ValueError, TypeError, RecursionError) as exc:
        raise ResponseError("invalid_json") from exc
    if type(value) is not dict:
        raise ResponseError("unexpected_action_shape")
    if value == {"action": "submit"}:
        return Submit(model_calls=1)
    if set(value) != {"action", "patch"} or value["action"] != "propose":
        raise ResponseError("unexpected_action_shape")
    try:
        patch = Patch.from_dict(value["patch"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ResponseError("invalid_patch") from exc
    return Propose(patch, model_calls=1)


def _text_metadata(value: object) -> str | None:
    if isinstance(value, str) and len(value) <= 512 and not any(ord(c) < 32 for c in value):
        return value
    return None


def _usage_metadata(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
    return {key: value[key] for key in keys if type(value.get(key)) is int and value[key] >= 0}


def _validate_configuration(model: str, max_calls: int) -> None:
    if not isinstance(model, str) or not model.strip() or any(c in model for c in "\r\n\x00"):
        raise ValueError("model must be a nonempty single-line string")
    if type(max_calls) is not int or not 1 <= max_calls <= 64:
        raise ValueError("max_calls must be an integer in 1..64")


class LLMActor:
    """Fresh per-episode controller; max_calls bounds attempts, not render costs.

    A provider failure is raised after journaling. The unmodified core then reports
    an error and unknown native model-call cost; the journal retains attempted calls.
    Malformed text returns a counted invalid proposal. Guard submissions/restores
    are explicit zero-call controller actions and are marked separately in the log.
    """

    def __init__(
        self,
        transport,
        *,
        model: str,
        max_calls: int = 4,
        journal_path: Path | None = None,
    ) -> None:
        _validate_configuration(model, max_calls)
        if not callable(getattr(transport, "complete", None)):
            raise ValueError("transport must expose complete(model=..., messages=...)")
        if journal_path is not None and not isinstance(journal_path, Path):
            raise ValueError("journal_path must be a Path")
        self.transport = transport
        self.model = model
        self.max_calls = max_calls
        self.calls_made = 0
        self.controller_id = uuid.uuid4().hex
        self.closed = False
        self._history: list[tuple[dict, dict]] = []
        self._best_patch: Patch | None = None
        self._best_distance = float("inf")
        self._guard_reason: str | None = None
        self._episode_identity: str | None = None
        self._journal = None
        if journal_path is not None:
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal = journal_path.open("x", encoding="utf-8")
        self._event(
            "controller_start",
            protocol_id=PROTOCOL_ID,
            feature_id=features.FEATURE_ID,
            model=self.model,
            max_calls=self.max_calls,
            prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        )

    def _event(self, event: str, **payload) -> None:
        if self._journal is None:
            return
        document = {
            "schema_id": JOURNAL_SCHEMA_ID,
            "controller_id": self.controller_id,
            "event": event,
            "at_utc": datetime.now(UTC).isoformat(),
            **payload,
        }
        self._journal.write(json.dumps(document, allow_nan=False, ensure_ascii=True) + "\n")
        self._journal.flush()
        os.fsync(self._journal.fileno())

    def _guard(self, observation: Observation, reason: str) -> Propose | Submit:
        self._guard_reason = reason
        if (
            observation.patch != self._best_patch
            and observation.render_budget_remaining >= 1
            and observation.decision_budget_remaining >= 2
        ):
            action = Propose(self._best_patch, model_calls=0)
            self._event(
                "controller_guard",
                origin="controller",
                reason=reason,
                action="restore_best_heard",
                patch=self._best_patch.to_dict(),
                model_calls=0,
            )
            return action
        self._event(
            "controller_guard", origin="controller", reason=reason, action="submit", model_calls=0
        )
        return Submit(model_calls=0)

    def decide(self, observation: Observation) -> Propose | Submit:
        if self.closed:
            raise RuntimeError("controller is closed")
        identity = observation.episode_id
        if (
            not isinstance(identity, str)
            or not identity.strip()
            or any(character in identity for character in "\r\n\x00")
        ):
            raise ValueError("episode identity must be a nonempty single-line string")
        if self._episode_identity is None:
            self._episode_identity = identity
            # The local binding joins transcripts to result rows; it is never
            # included in model messages or feature extraction.
            self._event("episode_binding", episode_id=identity, sent_to_model=False)
        elif identity != self._episode_identity:
            raise RuntimeError("controller cannot be reused for a different episode")
        distance = score(observation.target_audio, observation.candidate_audio)["distance"]
        if distance < self._best_distance:
            self._best_distance = distance
            self._best_patch = Patch.from_dict(observation.patch.to_dict())
        if self._guard_reason is not None:
            return self._guard(observation, self._guard_reason)
        if self.calls_made >= self.max_calls:
            return self._guard(observation, "model_call_cap")
        if not observation.editable_fields:
            return self._guard(observation, "no_editable_fields")
        if observation.render_budget_remaining <= 1 or observation.decision_budget_remaining <= 2:
            return self._guard(observation, "reserve_restore_and_submit")

        payload = features.public_observation(observation)
        payload["budgets"]["model_calls_remaining"] = self.max_calls - self.calls_made
        payload["best_heard"] = {
            "patch": self._best_patch.to_dict(),
            "distance": self._best_distance,
        }
        current = {"role": "user", "content": json.dumps(payload, allow_nan=False, sort_keys=True)}
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user, assistant in self._history:
            messages.extend((dict(user), dict(assistant)))
        messages.append(dict(current))
        attempt = self.calls_made + 1
        self._event(
            "attempt_start",
            attempt=attempt,
            request={"model": self.model, "messages": messages, "stream": False},
        )
        self.calls_made = attempt
        started = time.perf_counter()
        try:
            response = self.transport.complete(model=self.model, messages=messages)
        except Exception as exc:
            self._event(
                "transport_failure",
                attempt=attempt,
                error_code="transport_error",
                error_type=type(exc).__name__,
                latency_seconds=time.perf_counter() - started,
                provider_execution="unknown",
            )
            raise
        response = response if isinstance(response, dict) else {}
        content = response.get("content")
        raw = content.encode("utf-8", errors="replace") if isinstance(content, str) else b""
        bounded_content = raw[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
        self._event(
            "response",
            attempt=attempt,
            raw_content=bounded_content if isinstance(content, str) else None,
            content_truncated=len(raw) > MAX_RESPONSE_BYTES,
            usage=_usage_metadata(response.get("usage")),
            response_id=_text_metadata(response.get("response_id")),
            response_model=_text_metadata(response.get("response_model")),
            latency_seconds=time.perf_counter() - started,
        )
        self._history.append((current, {"role": "assistant", "content": bounded_content}))
        self._history = self._history[-HISTORY_TURNS:]
        try:
            action = parse_action(content)
        except ResponseError as exc:
            self._event("parse_rejection", attempt=attempt, error_code=str(exc), model_calls=1)
            return Propose({}, model_calls=1)
        self._event(
            "model_decision",
            attempt=attempt,
            action="submit" if isinstance(action, Submit) else "propose",
            origin="model",
            model_calls=1,
        )
        return action

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self._event("controller_close", attempts=self.calls_made)
        finally:
            if self._journal is not None:
                self._journal.close()


def llm_actor_spec(
    *,
    transport_factory: Callable,
    model: str,
    max_calls: int = 4,
    name: str = "harnessbridge-text",
    journal_directory: Path | None = None,
    artifact_paths: tuple[Path, ...] = (),
) -> ActorSpec:
    """Build fresh transport/history per episode; bind only immutable artifacts."""
    _validate_configuration(model, max_calls)
    if not callable(transport_factory):
        raise ValueError("transport_factory must be callable")
    if journal_directory is not None and not isinstance(journal_directory, Path):
        raise ValueError("journal_directory must be a Path")
    ordinal = 0

    def factory():
        nonlocal ordinal
        ordinal += 1
        journal = (
            None
            if journal_directory is None
            else journal_directory / f"episode-{ordinal:06d}-{uuid.uuid4().hex}.jsonl"
        )
        return LLMActor(transport_factory(), model=model, max_calls=max_calls, journal_path=journal)

    sources = (Path(__file__), Path(features.__file__), *artifact_paths)
    return ActorSpec(
        name,
        factory,
        metadata={
            "kind": "exploratory-text-controller-over-dsp-features",
            "model": model,
            "protocol_id": PROTOCOL_ID,
            "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "features": features.feature_configuration(),
            "max_model_request_attempts": max_calls,
            "history_turns": HISTORY_TURNS,
            "response_byte_limit": MAX_RESPONSE_BYTES,
            "retry_policy": "none; each new decision may make one separately counted request",
            "controller_guard": "restore-best-heard-if-budget-allows-then-explicit-submit",
            "guard_is_model_decision": False,
            "transport_errors": "raise; native call count unknown, attempted requests in journal",
            "backend_tool_isolation": "must be established and recorded by the pilot runner",
            "quality_status": "unqualified-exploratory-pilot",
        },
        artifact_paths=tuple(dict.fromkeys(sources)),
    )


__all__ = [
    "HISTORY_TURNS",
    "MAX_RESPONSE_BYTES",
    "PROTOCOL_ID",
    "SYSTEM_PROMPT",
    "LLMActor",
    "ResponseError",
    "llm_actor_spec",
    "parse_action",
]
