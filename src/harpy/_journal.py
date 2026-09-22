"""Task-neutral, single-writer evaluation journals and bounded streaming readout.

The data log is durable before its commit line is appended and fsynced. Only a
complete commit line makes a member scientific evidence. An interrupted attempt
is retained separately and never assigned invented zero cost. Linux advisory
locks are released by the operating system, including after SIGKILL.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import re
import sqlite3
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    read_json_document,
    sha256_file,
    write_new_bytes,
)

BUNDLE_SCHEMA_ID = "harpy.evaluation-bundle/2"
COMPLETION_SCHEMA_ID = "harpy.evaluation-complete/2"
MAX_LINE_BYTES = 16 * 1024 * 1024
_LOGS = ("records.jsonl", "commits.jsonl", "attempts.jsonl")
_ROOT_FILES = {"manifest.json", "locators.json", "writer.lock", "complete.json", *_LOGS}
_ARTIFACT = re.compile(r"artifacts/[0-9a-f]{64}\.[a-z0-9]{1,12}\Z")
_TEMPORARY = re.compile(
    r"\.(?P<stem>[0-9a-f]{64}|manifest|locators|complete)\.[0-9a-f]{32}"
    r"\.tmp\.(?P<suffix>[a-z0-9]{1,12})\Z"
)


def exact(left: Mapping, right: Mapping, name: str) -> None:
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(f"{name} differs from frozen evidence")


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def finite(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def object_fields(value: object, fields: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("document has missing or unexpected fields")
    canonical_json_bytes(value)
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a lowercase SHA-256 digest")
    return value


def _key(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or not item or any(c in item for c in "\r\n\x00")
            for item in value
        )
    ):
        raise ValueError("member key must be a nonempty array of identities")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def lock_bundle(path: Path, *, exclusive: bool = True):
    """Use the kernel lock, never a PID or elapsed-time stale-lock heuristic."""
    try:
        import fcntl
    except ImportError as exc:
        raise RuntimeError("Durable bundle locking is currently qualified on Linux only") from exc
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path / "writer.lock", flags, 0o600)
    try:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Bundle is in use by another writer or reader") from exc
        yield
    finally:
        os.close(descriptor)


def read_configuration(path: Path) -> dict:
    """Read inert metadata only; never load an actor or follow its local locator."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bundle must be a real directory")
    for name in ("manifest.json", "locators.json"):
        if (path / name).is_symlink():
            raise ValueError("bundle metadata must not be symbolic links")
    manifest = read_json_document(path / "manifest.json")
    object_fields(manifest, {"schema_id", "task_id", "identity", "configuration", "expected"})
    if manifest["schema_id"] != BUNDLE_SCHEMA_ID:
        raise ValueError("unsupported durable bundle schema")
    if not isinstance(manifest["task_id"], str) or not manifest["task_id"]:
        raise ValueError("task identity is required")
    for name in ("identity", "configuration"):
        if not isinstance(manifest[name], dict):
            raise ValueError(f"{name} must be an object")
    integer(manifest["expected"], "expected", 1)
    locators = (
        read_json_document(path / "locators.json") if (path / "locators.json").exists() else {}
    )
    return {
        "task_id": manifest["task_id"],
        "identity": manifest["identity"],
        "configuration": manifest["configuration"],
        "locators": locators,
        "expected": manifest["expected"],
    }


@dataclass
class ScanState:
    completed: int = 0
    attempts: int = 0
    records_bytes: int = 0
    commits_bytes: int = 0
    attempts_bytes: int = 0
    known_elapsed_seconds: float = 0.0
    chain_sha256: str = ""
    is_complete: bool = False

    @property
    def interrupted_attempts(self) -> int:
        return self.attempts - self.completed


@dataclass(frozen=True)
class BundleReport:
    path: Path
    task_id: str
    complete: bool
    completed: int
    expected: int
    attempts: int
    interrupted_attempts: int
    known_elapsed_seconds: float
    metrics: Mapping = dataclass_field(default_factory=dict)

    @property
    def missing(self) -> int:
        return self.expected - self.completed

    def to_document(self) -> dict:
        return {
            "task_id": self.task_id,
            "complete": self.complete,
            "completed": self.completed,
            "expected": self.expected,
            "missing": self.missing,
            "attempts": self.attempts,
            "interrupted_attempts": self.interrupted_attempts,
            "known_elapsed_seconds": self.known_elapsed_seconds,
            "interrupted_costs": "unknown" if self.interrupted_attempts else "none",
            "metrics": dict(self.metrics),
        }


class MetricStore:
    """Disk-backed exact distributions and artifact membership with a 1 MiB cache.

    One short scalar row per measurement is stored on disk. No waveform, episode
    record list, or suite-sized quantile buffer is retained in Python memory.
    """

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="harpy-readout-")
        self.connection = sqlite3.connect(Path(self._temporary.name) / "metrics.sqlite3")
        self.connection.execute("PRAGMA cache_size=-1024")
        self.connection.execute("PRAGMA temp_store=FILE")
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute(
            "CREATE TABLE measurements (group_id TEXT, metric TEXT, value REAL)"
        )
        self.connection.execute("CREATE TABLE artifacts (path TEXT PRIMARY KEY)")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()
        self._temporary.cleanup()

    def add(self, group: str, metric: str, value: float | int | None) -> None:
        if value is not None:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("metric must be finite")
            self.connection.execute(
                "INSERT INTO measurements VALUES (?, ?, ?)", (group, metric, numeric)
            )

    def distribution(self, group: str, metric: str) -> dict:
        count, mean, minimum, maximum = self.connection.execute(
            "SELECT count(*), avg(value), min(value), max(value) FROM measurements "
            "WHERE group_id=? AND metric=?",
            (group, metric),
        ).fetchone()

        def quantile(fraction: float):
            if not count:
                return None
            position = fraction * (count - 1)
            lower = int(position)
            rows = self.connection.execute(
                "SELECT value FROM measurements WHERE group_id=? AND metric=? "
                "ORDER BY value LIMIT 2 OFFSET ?",
                (group, metric, lower),
            ).fetchall()
            a = rows[0][0]
            return a if len(rows) == 1 else a + (rows[1][0] - a) * (position - lower)

        return {
            "count": count,
            "mean": mean,
            "median": quantile(0.5),
            "p90": quantile(0.9),
            "p99": quantile(0.99),
            "min": minimum,
            "max": maximum,
        }

    def reference(self, path: str) -> None:
        self.connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?)", (path,))

    def referenced(self, path: str) -> bool:
        return (
            self.connection.execute("SELECT 1 FROM artifacts WHERE path=?", (path,)).fetchone()
            is not None
        )


def _lines(path: Path, state: ScanState, field: str) -> Iterator[tuple[dict, bytes]]:
    with path.open("rb") as stream:
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("journal line exceeds the bounded record limit")
            if not raw.endswith(b"\n"):
                # A final unterminated write never constitutes a commit.
                break
            document = decode_json_bytes(raw)
            if canonical_json_bytes(document) != raw:
                raise ValueError("journal lines must retain canonical bytes")
            setattr(state, field, stream.tell())
            yield document, raw


def _entries(path: Path) -> Iterator[Path]:
    # Path.iterdir on the qualified Python 3.12 runtime materializes os.listdir.
    # Large suites may export many files; stream their names instead.
    with os.scandir(path) as entries:
        for entry in entries:
            yield Path(entry.path)


def _safe_tree(
    path: Path,
    references: MetricStore | None = None,
    *,
    prune: bool = False,
    discard_temporaries: bool = False,
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bundle must be a real directory")
    for entry in _entries(path):
        if entry.is_symlink():
            raise ValueError("bundle must not contain symbolic links")
        temporary = _TEMPORARY.fullmatch(entry.name)
        if (
            temporary
            and temporary["stem"] in {"manifest", "locators", "complete"}
            and temporary["suffix"] == "json"
            and entry.is_file()
        ):
            if references is not None and not prune and not discard_temporaries:
                raise ValueError("completed bundle contains an unpublished metadata file")
            if prune or discard_temporaries:
                entry.unlink()
            continue
        if entry.name == "artifacts" and entry.is_dir():
            for artifact in _entries(entry):
                if artifact.is_symlink() or not artifact.is_file():
                    raise ValueError("artifact directory must contain only regular files")
                relative = f"artifacts/{artifact.name}"
                temporary = _TEMPORARY.fullmatch(artifact.name)
                owned_temporary = temporary and re.fullmatch(r"[0-9a-f]{64}", temporary["stem"])
                if discard_temporaries and owned_temporary:
                    artifact.unlink()
                    continue
                if references is not None and not references.referenced(relative):
                    if not prune:
                        raise ValueError("completed bundle contains an uncommitted artifact")
                    # This namespace is exclusively created by this journal writer.
                    # Reject unrelated user files rather than deleting them.
                    if not _ARTIFACT.fullmatch(relative) and not owned_temporary:
                        raise ValueError("unrecognized file in artifact namespace")
                    artifact.unlink()
            if prune or discard_temporaries:
                _fsync_directory(entry)
        elif entry.name not in _ROOT_FILES or not entry.is_file():
            raise ValueError("bundle contains an unexpected file or directory")
    if prune or discard_temporaries:
        _fsync_directory(path)


def _artifacts(path: Path, payload: dict, references: MetricStore) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) > 32:
        raise ValueError("each member requires a bounded artifact inventory")
    roles = set()
    for artifact in artifacts:
        object_fields(artifact, {"role", "path", "sha256", "size_bytes"})
        role, relative = artifact["role"], artifact["path"]
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("artifact roles must be unique")
        roles.add(role)
        if not isinstance(relative, str) or not _ARTIFACT.fullmatch(relative):
            raise ValueError("artifact path must be a content-addressed portable path")
        digest = _digest(artifact["sha256"])
        if relative.split("/")[1].split(".")[0] != digest:
            raise ValueError("artifact path differs from its content identity")
        size = integer(artifact["size_bytes"], "artifact size")
        file = path / relative
        if file.is_symlink() or not file.is_file():
            raise ValueError("committed artifact is missing or a symbolic link")
        if file.stat().st_size != size or sha256_file(file) != digest:
            raise ValueError("committed artifact content was changed")
        references.reference(relative)


def scan(
    path: Path,
    manifest: dict,
    state: ScanState,
    *,
    key_at: Callable[[int], list[str]],
    validate: Callable[[dict, int], None],
    references: MetricStore,
) -> Iterator[dict]:
    """Stream the committed prefix and validate it before yielding each member."""
    _safe_tree(path)
    state.chain_sha256 = sha256_file(path / "manifest.json")
    attempts = _lines(path / "attempts.jsonl", state, "attempts_bytes")
    current_attempt = None
    current_attempt_raw = b""

    def next_attempt():
        nonlocal current_attempt, current_attempt_raw
        try:
            current_attempt, current_attempt_raw = next(attempts)
        except StopIteration:
            return False
        object_fields(current_attempt, {"attempt", "index", "key", "started_utc"})
        state.attempts += 1
        if integer(current_attempt["attempt"], "attempt", 1) != state.attempts:
            raise ValueError("attempt identities must be contiguous")
        member = integer(current_attempt["index"], "attempt member", 1)
        if member > manifest["expected"] or member != state.completed + 1:
            raise ValueError("attempt refers to an already committed or unknown member")
        if _key(current_attempt["key"]) != key_at(member - 1):
            raise ValueError("attempt membership differs from frozen membership")
        if (
            not isinstance(current_attempt["started_utc"], str)
            or not current_attempt["started_utc"]
        ):
            raise ValueError("attempt requires its start timestamp")
        return True

    with (path / "records.jsonl").open("rb") as records:
        for commit, raw in _lines(path / "commits.jsonl", state, "commits_bytes"):
            object_fields(
                commit,
                {
                    "index",
                    "key",
                    "attempt",
                    "attempt_sha256",
                    "offset",
                    "length",
                    "sha256",
                    "previous_sha256",
                    "elapsed_seconds",
                },
            )
            index = integer(commit["index"], "commit index", 1)
            if index != state.completed + 1 or index > manifest["expected"]:
                raise ValueError("commits must cover an ordered prefix of membership")
            if _key(commit["key"]) != key_at(index - 1):
                raise ValueError("committed membership differs from frozen membership")
            wanted_attempt = integer(commit["attempt"], "commit attempt", 1)
            if wanted_attempt <= state.attempts:
                raise ValueError("commits must use distinct increasing attempts")
            while state.attempts < wanted_attempt:
                if not next_attempt():
                    raise ValueError("commit has no durable attempt start")
            if hashlib.sha256(current_attempt_raw).hexdigest() != _digest(commit["attempt_sha256"]):
                raise ValueError("committed attempt start was changed")
            if _digest(commit["previous_sha256"]) != state.chain_sha256:
                raise ValueError("commit chain was changed")
            if integer(commit["offset"], "record offset") != state.records_bytes:
                raise ValueError("record offsets must be contiguous")
            length = integer(commit["length"], "record length", 1)
            if length > MAX_LINE_BYTES:
                raise ValueError("record exceeds the bounded record limit")
            data = records.read(length)
            if (
                len(data) != length
                or not data.endswith(b"\n")
                or hashlib.sha256(data).hexdigest() != _digest(commit["sha256"])
            ):
                raise ValueError("committed record is missing, torn, or changed")
            payload = decode_json_bytes(data)
            if canonical_json_bytes(payload) != data or payload.get("key") != commit["key"]:
                raise ValueError("record bytes or membership differ from their commit")
            _artifacts(path, payload, references)
            try:
                validate(payload, index - 1)
            except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
                raise ValueError("committed task record contains malformed fields") from exc
            state.records_bytes += length
            state.chain_sha256 = hashlib.sha256(raw).hexdigest()
            state.known_elapsed_seconds += finite(commit["elapsed_seconds"], "attempt elapsed time")
            state.completed += 1
            yield payload
    while next_attempt():
        pass
    completion_path = path / "complete.json"
    if completion_path.exists():
        completion = read_json_document(completion_path)
        expected = completion_document(path, manifest, state)
        exact(completion, expected, "completion marker")
        if state.completed != manifest["expected"]:
            raise ValueError("completed bundle is missing expected members")
        if any(
            (path / name).stat().st_size != getattr(state, field)
            for name, field in (
                ("records.jsonl", "records_bytes"),
                ("commits.jsonl", "commits_bytes"),
                ("attempts.jsonl", "attempts_bytes"),
            )
        ):
            raise ValueError("completed bundle contains uncommitted log bytes")
        _safe_tree(path, references)
        state.is_complete = True


def completion_document(path: Path, manifest: dict, state: ScanState) -> dict:
    return {
        "schema_id": COMPLETION_SCHEMA_ID,
        "manifest_sha256": sha256_file(path / "manifest.json"),
        **{name.replace(".jsonl", "_sha256"): sha256_file(path / name) for name in _LOGS},
        "completed": state.completed,
        "expected": manifest["expected"],
        "attempts": state.attempts,
    }


class Journal:
    """A locked writer. Call recover with the task's validator before appending."""

    def __init__(
        self,
        path: Path,
        manifest: dict,
        *,
        resume: bool = False,
        locators: Mapping | None = None,
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        self.resume = resume
        self.locators = {} if locators is None else dict(locators)
        self.state = ScanState()
        self._lock = None
        self._active = None

    def __enter__(self):
        if not self.resume:
            self.path.mkdir(parents=True, exist_ok=False)
        elif self.path.is_symlink() or not self.path.is_dir():
            raise ValueError("resume requires an existing real bundle directory")
        self._lock = lock_bundle(self.path)
        self._lock.__enter__()
        try:
            if self.resume:
                read_configuration(self.path)
                frozen = read_json_document(self.path / "manifest.json")
                # Configuration may bind dataset/catalog provenance beyond the
                # expanded episodes. Only separate local locators may change.
                exact(frozen, self.manifest, "resume identity")
                self.manifest = frozen
            else:
                write_new_bytes(self.path / "manifest.json", canonical_json_bytes(self.manifest))
                write_new_bytes(self.path / "locators.json", canonical_json_bytes(self.locators))
                (self.path / "artifacts").mkdir()
                _fsync_directory(self.path)
            for name in _LOGS:
                if not (self.path / name).exists():
                    write_new_bytes(self.path / name, b"")
            if not (self.path / "artifacts").exists():
                (self.path / "artifacts").mkdir()
            _safe_tree(self.path)
        except BaseException:
            self._lock.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        if self._lock is not None:
            self._lock.__exit__(*args)

    def recover(self, *, key_at, validate) -> None:
        # A kill between atomic publication and temporary-name removal can even
        # leave complete.json already published. Remove only exact writer-owned
        # temporary names before scanning; never remove a committed final path.
        _safe_tree(self.path, discard_temporaries=True)
        with MetricStore() as refs:
            for _ in scan(
                self.path,
                self.manifest,
                self.state,
                key_at=key_at,
                validate=validate,
                references=refs,
            ):
                pass
        if self.state.is_complete:
            return
        for name, field in (
            ("records.jsonl", "records_bytes"),
            ("commits.jsonl", "commits_bytes"),
            ("attempts.jsonl", "attempts_bytes"),
        ):
            with (self.path / name).open("r+b") as stream:
                stream.truncate(getattr(self.state, field))
                stream.flush()
                os.fsync(stream.fileno())

    def _append(self, name: str, document: dict) -> bytes:
        data = canonical_json_bytes(document)
        if len(data) > MAX_LINE_BYTES:
            raise ValueError("journal line exceeds the bounded record limit")
        with (self.path / name).open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return data

    def begin(self, key: list[str]) -> None:
        if self.state.is_complete or self._active is not None:
            raise ValueError("journal cannot begin another active or completed member")
        document = {
            "attempt": self.state.attempts + 1,
            "index": self.state.completed + 1,
            "key": _key(key),
            "started_utc": datetime.now(UTC).isoformat(),
        }
        raw = self._append("attempts.jsonl", document)
        self.state.attempts += 1
        self.state.attempts_bytes += len(raw)
        self._active = (document, raw, time.perf_counter())

    def artifact(self, role: str, content: bytes, suffix: str) -> dict:
        if self._active is None:
            raise ValueError("artifacts require an active attempt")
        digest = hashlib.sha256(content).hexdigest()
        relative = f"artifacts/{digest}.{suffix}"
        if not _ARTIFACT.fullmatch(relative):
            raise ValueError("unsupported artifact suffix")
        target = self.path / relative
        if target.exists():
            if (
                target.is_symlink()
                or target.stat().st_size != len(content)
                or sha256_file(target) != digest
            ):
                raise ValueError("existing content-addressed artifact was changed")
        else:
            write_new_bytes(target, content)
        return {"role": role, "path": relative, "sha256": digest, "size_bytes": len(content)}

    def commit(self, payload: dict) -> None:
        if self._active is None:
            raise ValueError("commit requires an active attempt")
        attempt, attempt_raw, started = self._active
        if payload.get("key") != attempt["key"]:
            raise ValueError("record differs from active membership")
        record = self._append("records.jsonl", payload)
        elapsed = time.perf_counter() - started
        commit = {
            "index": self.state.completed + 1,
            "key": attempt["key"],
            "attempt": attempt["attempt"],
            "attempt_sha256": hashlib.sha256(attempt_raw).hexdigest(),
            "offset": self.state.records_bytes,
            "length": len(record),
            "sha256": hashlib.sha256(record).hexdigest(),
            "previous_sha256": self.state.chain_sha256,
            "elapsed_seconds": elapsed,
        }
        raw = self._append("commits.jsonl", commit)
        self.state.records_bytes += len(record)
        self.state.commits_bytes += len(raw)
        self.state.chain_sha256 = hashlib.sha256(raw).hexdigest()
        self.state.completed += 1
        self.state.known_elapsed_seconds += elapsed
        self._active = None

    def finish(self, *, key_at, validate) -> None:
        if self.state.is_complete:
            return
        if self._active is not None or self.state.completed != self.manifest["expected"]:
            raise ValueError("cannot complete a bundle with unfinished members")
        checked = ScanState()
        with MetricStore() as refs:
            for _ in scan(
                self.path, self.manifest, checked, key_at=key_at, validate=validate, references=refs
            ):
                pass
            _safe_tree(self.path, refs, prune=True)
        write_new_bytes(
            self.path / "complete.json",
            canonical_json_bytes(completion_document(self.path, self.manifest, checked)),
        )
        self.state.is_complete = True


def report(path: Path, manifest: dict, state: ScanState, metrics: Mapping) -> BundleReport:
    return BundleReport(
        path=path,
        task_id=manifest["task_id"],
        complete=state.is_complete,
        completed=state.completed,
        expected=manifest["expected"],
        attempts=state.attempts,
        interrupted_attempts=state.interrupted_attempts,
        known_elapsed_seconds=state.known_elapsed_seconds,
        metrics=metrics,
    )


def manifest_for(
    task_id: str, identity: dict, expected: int, configuration: Mapping | None
) -> dict:
    return {
        "schema_id": BUNDLE_SCHEMA_ID,
        "task_id": task_id,
        "identity": identity,
        "configuration": {} if configuration is None else dict(configuration),
        "expected": integer(expected, "expected", 1),
    }


def source_identity() -> dict:
    """Portable installed-package identity, independent of checkout location."""
    import harpy

    root = Path(harpy.__file__).parent
    files = []
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink():
            raise ValueError("package source must not contain symbolic links")
        size = path.stat().st_size
        digest = sha256_file(path)
        if path.stat().st_size != size:
            raise ValueError("package source changed while hashing")
        files.append(
            {"path": path.relative_to(root).as_posix(), "sha256": digest, "size_bytes": size}
        )
    return {
        "kind": "package-python-files",
        "package_version": harpy.__version__,
        "files": files,
        "sha256": hashlib.sha256(canonical_json_bytes({"files": files})).hexdigest(),
    }


def artifact_inventory(paths: tuple[Path, ...]) -> list[dict]:
    """Use supplied-root ordinals and relative names, never local absolute paths."""
    inventory = []
    for index, original in enumerate(paths):
        if original.is_symlink():
            raise ValueError("actor artifacts must not be symbolic links")
        root = original.resolve(strict=True)
        entries = (root,) if root.is_file() else sorted(root.rglob("*"))
        for path in entries:
            if path.is_symlink():
                raise ValueError("actor artifacts must not contain symbolic links")
            if path.is_file():
                name = path.name if root.is_file() else path.relative_to(root).as_posix()
                size = path.stat().st_size
                digest = sha256_file(path)
                if path.stat().st_size != size:
                    raise ValueError("actor artifact changed while hashing")
                inventory.append(
                    {"path": f"artifact-{index:04d}/{name}", "sha256": digest, "size_bytes": size}
                )
    return inventory


def callable_identity(factory: Callable) -> dict:
    """Bind inspectable Python factory code without persisting its local path.

    A factory's configuration belongs in actor metadata, including random seeds
    and artifact identities. Live closure state is intentionally not a resumable
    checkpoint; factories must create fresh episode controllers.
    """
    target = factory if inspect.isfunction(factory) or inspect.isclass(factory) else type(factory)
    source = inspect.getsourcefile(target)
    if source is None or not Path(source).is_file():
        raise ValueError("durable actors require an inspectable Python factory source file")
    path = Path(source)
    if path.is_symlink():
        raise ValueError("factory source must not be a symbolic link")
    return {
        "module": target.__module__,
        "qualname": target.__qualname__,
        "source_sha256": sha256_file(path),
    }


def validate_callable_identity(document: object) -> None:
    object_fields(document, {"module", "qualname", "source_sha256"})
    for name in ("module", "qualname"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError("factory identity requires module and qualified name")
    _digest(document["source_sha256"])


def validate_inventory(rows: object) -> None:
    if not isinstance(rows, list):
        raise ValueError("portable file inventory must be an array")
    names = set()
    for row in rows:
        object_fields(row, {"path", "sha256", "size_bytes"})
        name = row["path"]
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or "\\" in name
            or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or name in names
        ):
            raise ValueError("inventory paths must be unique portable relative paths")
        names.add(name)
        _digest(row["sha256"])
        integer(row["size_bytes"], "inventory size")


def validate_source_identity(document: object) -> None:
    object_fields(document, {"kind", "package_version", "files", "sha256"})
    if document["kind"] != "package-python-files":
        raise ValueError("unknown package source identity")
    if not isinstance(document["package_version"], str) or not document["package_version"]:
        raise ValueError("package version must be declared")
    validate_inventory(document["files"])
    if not document["files"]:
        raise ValueError("package source inventory cannot be empty")
    if hashlib.sha256(canonical_json_bytes({"files": document["files"]})).hexdigest() != _digest(
        document["sha256"]
    ):
        raise ValueError("package source digest differs from its inventory")
