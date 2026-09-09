"""Frozen, portable episode manifests independent of local experiment outputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import cache
from importlib.resources import files

from harpy.experiments.models import EpisodeSpec
from harpy.learning.pitch_data import (
    PitchEvaluationSuiteId,
    fixed_pitch_evaluation_suite,
    pitch_coordinate_split,
)

DEVELOPMENT_PROTOCOL_ID = "harpy-clean-development-v1"
CONFIRMATION_PROTOCOL_ID = "harpy-clean-confirmation-v1"
SMOKE_PROTOCOL_ID = "harpy-clean-engineering-smoke-v1"
CONFIRMATION_DIGEST_SHA256 = "74d5791439eabc71c9d9b91c10780fd7f435a4e7fb3691ffb70fdcf4d0a809e6"
EXCLUSIONS_DIGEST_SHA256 = "c75615d811c08c4aa490437cbbb95382ee14d4770d48daa11400c9095c9a1892"


def _canonical(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(document: object) -> str:
    return hashlib.sha256(_canonical(document)).hexdigest()


def _read(name: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate protocol key: {key}")
            result[key] = value
        return result

    return json.loads(
        files("harpy.experiments").joinpath("data", name).read_text(), object_pairs_hook=pairs
    )


def _nuisance_seed(source: int, target: int) -> int:
    token = f"harpy-static-nuisance-v1:{source}:{target}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


@cache
def benchmark_episodes() -> tuple[EpisodeSpec, ...]:
    """The exact 650 established source/target pairs, with explicit nuisance IDs."""
    episodes = []
    for partition, suite_id in (
        ("iid", PitchEvaluationSuiteId.IID),
        ("lower", PitchEvaluationSuiteId.OOD_LOWER),
        ("upper", PitchEvaluationSuiteId.OOD_UPPER),
    ):
        for index, episode in enumerate(fixed_pitch_evaluation_suite(suite_id).episodes):
            source, target = episode.source_pitch_cents, episode.target_note_index
            episodes.append(
                EpisodeSpec(
                    f"benchmark-{partition}-{index:04d}",
                    source,
                    target,
                    _nuisance_seed(source, target),
                    partition,
                )
            )
    return tuple(episodes)


development_episodes = benchmark_episodes


@cache
def excluded_historical_pairs() -> frozenset[tuple[int, int]]:
    """Previously evaluated pairs in (target, source) order, frozen before qualification."""
    document = _read("historical-exclusions-v1.json")
    if _digest(document) != EXCLUSIONS_DIGEST_SHA256:
        raise ValueError("historical episode exclusions do not match their frozen digest")
    return frozenset(tuple(pair) for pair in document["pairs_target_source"])


def _confirmation_document() -> dict:
    document = _read("confirmation-v1.json")
    payload = {key: value for key, value in document.items() if key != "digest_sha256"}
    if (
        document.get("digest_sha256") != CONFIRMATION_DIGEST_SHA256
        or _digest(payload) != CONFIRMATION_DIGEST_SHA256
    ):
        raise ValueError("confirmation manifest does not match its frozen digest")
    if document["exclusions_digest_sha256"] != EXCLUSIONS_DIGEST_SHA256:
        raise ValueError("confirmation exclusions do not match their frozen digest")
    return document


@cache
def confirmation_episodes() -> tuple[EpisodeSpec, ...]:
    """Return the 1000 frozen confirmation pairs without sampling or evaluating."""
    document = _confirmation_document()
    episodes = tuple(EpisodeSpec.from_document(item) for item in document["episodes"])
    pairs = {(episode.target_note_index, episode.source_pitch_cents) for episode in episodes}
    if len(episodes) != 1000 or len(pairs) != 1000 or pairs & excluded_historical_pairs():
        raise ValueError("confirmation pairs must be unique and historically unevaluated")
    split = pitch_coordinate_split()
    pools = {
        "iid": split.iid_holdout_coordinates,
        "lower": split.ood_lower_coordinates,
        "upper": split.ood_upper_coordinates,
    }
    for partition, count in (("iid", 400), ("lower", 300), ("upper", 300)):
        selected = [episode for episode in episodes if episode.partition == partition]
        if len(selected) != count or any(
            episode.source_pitch_cents not in pools[partition] for episode in selected
        ):
            raise ValueError("confirmation partition does not match the declared coordinate pool")
        if [episode.id for episode in selected] != document["partitions"][partition]:
            raise ValueError("confirmation partition IDs do not match the manifest")
    return episodes


@cache
def smoke_episodes() -> tuple[EpisodeSpec, ...]:
    """Two established pairs per register; an engineering check, never a release gate."""
    return tuple(
        episode
        for partition in ("iid", "lower", "upper")
        for episode in [item for item in benchmark_episodes() if item.partition == partition][:2]
    )


def _protocol_id(name: str) -> str:
    aliases = {
        "clean": DEVELOPMENT_PROTOCOL_ID,
        "robustness": DEVELOPMENT_PROTOCOL_ID,
        "benchmark": DEVELOPMENT_PROTOCOL_ID,
        "development": DEVELOPMENT_PROTOCOL_ID,
        "confirmation": CONFIRMATION_PROTOCOL_ID,
        "smoke": SMOKE_PROTOCOL_ID,
    }
    identifier = aliases.get(name, name)
    if identifier not in (DEVELOPMENT_PROTOCOL_ID, CONFIRMATION_PROTOCOL_ID, SMOKE_PROTOCOL_ID):
        raise ValueError("protocol must be clean, robustness, confirmation, or smoke")
    return identifier


def protocol_episodes(name: str) -> tuple[EpisodeSpec, ...]:
    identifier = _protocol_id(name)
    if identifier == CONFIRMATION_PROTOCOL_ID:
        return confirmation_episodes()
    return smoke_episodes() if identifier == SMOKE_PROTOCOL_ID else benchmark_episodes()


def protocol_document(name: str) -> dict:
    """An owned canonical manifest with a digest over all fields except that digest."""
    identifier = _protocol_id(name)
    if identifier == CONFIRMATION_PROTOCOL_ID:
        confirmation_episodes()
        return _confirmation_document()
    episodes = protocol_episodes(identifier)
    document = {
        "id": identifier,
        "scope": (
            "Engineering smoke only; ineligible for release qualification."
            if identifier == SMOKE_PROTOCOL_ID
            else "650 previously evaluated clean single-sine source/target pairs; "
            "development benchmark, not fresh confirmation."
        ),
        "source_split_sha256": pitch_coordinate_split().digest_sha256,
        "episodes": [episode.to_document() for episode in episodes],
        "partitions": {
            partition: [episode.id for episode in episodes if episode.partition == partition]
            for partition in ("iid", "lower", "upper")
        },
    }
    document["digest_sha256"] = _digest(document)
    return document


def validate_protocol_document(document: Mapping[str, object]) -> dict:
    """Bind declared protocol claims to exact shipped episodes and provenance."""
    if not isinstance(document, Mapping) or not isinstance(document.get("id"), str):
        raise ValueError("protocol document must identify a declared protocol")
    expected = protocol_document(document["id"])
    if _canonical(dict(document)) != _canonical(expected):
        raise ValueError("protocol document differs from its declared frozen manifest")
    return expected
