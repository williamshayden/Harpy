"""Task-neutral, strict JSON and create-only local file primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path

type JSONValue = bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None


def _owned_json(value: object, field: str) -> JSONValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_owned_json(item, field) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must contain only string-keyed JSON objects")
        return {key: _owned_json(item, field) for key, item in value.items()}
    raise ValueError(f"{field} must contain only JSON values")


def canonical_json_bytes(document: Mapping[str, JSONValue]) -> bytes:
    """Encode one finite JSON object with sorted keys, ASCII escapes, and a newline."""
    normalized = _owned_json(document, "document")
    if not isinstance(normalized, dict):
        raise ValueError("document must be a JSON object")
    return (
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON number: {value}; JSON numbers must be finite")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _finite_decoded_numbers(value: JSONValue) -> None:
    # json.loads already owns the tree and creates only JSON types. Validate
    # overflowed floats without allocating a second copy of a large document.
    if isinstance(value, dict):
        for item in value.values():
            _finite_decoded_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _finite_decoded_numbers(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON document must contain only finite JSON numbers")


def decode_json_bytes(content: bytes) -> dict[str, JSONValue]:
    """Decode one duplicate-free finite UTF-8 JSON object exactly once."""
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("content must be valid UTF-8") from error
    try:
        parsed = json.loads(text, object_pairs_hook=_pairs_object, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ValueError("content must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("JSON document must be an object")
    _finite_decoded_numbers(parsed)
    return parsed


def read_json_document(path: Path) -> dict[str, JSONValue]:
    """Read one JSON document with duplicate and finiteness checks."""
    return decode_json_bytes(path.read_bytes())


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without keeping its payload in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_bytes(path: Path, content: bytes) -> None:
    """Atomically publish a sibling file without a check-then-overwrite race."""
    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")
    parent = path.parent.resolve(strict=True)
    final = parent / path.name
    temporary = final.with_name(f".{final.stem}.{uuid.uuid4().hex}.tmp{final.suffix}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, final)
        temporary.unlink()
        descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
