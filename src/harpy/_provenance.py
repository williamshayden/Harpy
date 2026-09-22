"""Task-neutral file inventories and source identities, with stable JSON contracts.

Historical learning imports re-export these types. Source capture takes explicit
required-input paths so a caller declares the code needed for its own protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import operator
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from harpy._serialization import JSONValue
from harpy._serialization import sha256_file as _sha256_file

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {suffix}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(mapping: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _relative_path(value: object, field: str = "relative_path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return value


@dataclass(frozen=True, slots=True)
class FileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "size_bytes", _integer(self.size_bytes, "size_bytes", minimum=1))
        object.__setattr__(self, "sha256", _digest(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class SourceStatus:
    commit: str
    dirty_tree: bool
    tracked_diff_sha256: str
    dependency_lock_sha256: str
    required_inputs_committed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or _COMMIT_PATTERN.fullmatch(self.commit) is None:
            raise ValueError("commit must be a full lowercase Git object ID")
        object.__setattr__(self, "dirty_tree", _boolean(self.dirty_tree, "dirty_tree"))
        object.__setattr__(
            self,
            "tracked_diff_sha256",
            _digest(self.tracked_diff_sha256, "tracked_diff_sha256"),
        )
        object.__setattr__(
            self,
            "dependency_lock_sha256",
            _digest(self.dependency_lock_sha256, "dependency_lock_sha256"),
        )
        object.__setattr__(
            self,
            "required_inputs_committed",
            _boolean(self.required_inputs_committed, "required_inputs_committed"),
        )
        if not self.dirty_tree and self.tracked_diff_sha256 != _EMPTY_SHA256:
            raise ValueError("a clean source must have the empty tracked diff SHA-256")


@dataclass(frozen=True, slots=True)
class PackageSourceStatus:
    """Content identity for installed or archived code without checkout provenance.

    This source variant never qualifies for a frozen scientific criterion.  It
    deliberately makes no claim about a Git commit, clean tree, or lockfile.
    """

    distribution_name: str
    distribution_version: str | None
    package_sha256: str

    def __post_init__(self) -> None:
        if self.distribution_name != "harpy-audio":
            raise ValueError("distribution_name must be 'harpy-audio'")
        if self.distribution_version is not None:
            _string(self.distribution_version, "distribution_version")
        _digest(self.package_sha256, "package_sha256")


type SourceProvenance = SourceStatus | PackageSourceStatus


def _source_to_document(source: SourceProvenance) -> dict[str, JSONValue]:
    if isinstance(source, PackageSourceStatus):
        return {
            "source_kind": "package_snapshot",
            "distribution_name": source.distribution_name,
            "distribution_version": source.distribution_version,
            "package_sha256": source.package_sha256,
        }
    return {
        "commit": source.commit,
        "dirty_tree": source.dirty_tree,
        "tracked_diff_sha256": source.tracked_diff_sha256,
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "required_inputs_committed": source.required_inputs_committed,
    }


def _source_from_document(value: object) -> SourceProvenance:
    mapping = _mapping(value, "source")
    if mapping.get("source_kind") == "package_snapshot":
        _exact_fields(
            mapping,
            {"source_kind", "distribution_name", "distribution_version", "package_sha256"},
            "package source",
        )
        return PackageSourceStatus(
            distribution_name=_string(mapping["distribution_name"], "distribution_name"),
            distribution_version=(
                None
                if mapping["distribution_version"] is None
                else _string(mapping["distribution_version"], "distribution_version")
            ),
            package_sha256=_digest(mapping["package_sha256"], "package_sha256"),
        )
    _exact_fields(
        mapping,
        {
            "commit",
            "dirty_tree",
            "tracked_diff_sha256",
            "dependency_lock_sha256",
            "required_inputs_committed",
        },
        "source",
    )
    return SourceStatus(
        commit=_string(mapping["commit"], "commit"),
        dirty_tree=_boolean(mapping["dirty_tree"], "dirty_tree"),
        tracked_diff_sha256=_digest(mapping["tracked_diff_sha256"], "tracked_diff_sha256"),
        dependency_lock_sha256=_digest(mapping["dependency_lock_sha256"], "dependency_lock_sha256"),
        required_inputs_committed=_boolean(
            mapping["required_inputs_committed"], "required_inputs_committed"
        ),
    )


def _record_file(path: Path, relative_path: str, *, hash_file=None) -> FileRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact payload {relative_path!r} must be a regular file")
    size = path.stat().st_size
    return FileRecord(
        relative_path=relative_path,
        size_bytes=size,
        sha256=(_sha256_file if hash_file is None else hash_file)(path),
    )


def _verify_file_record(root: Path, record: FileRecord, *, hash_file=None) -> Path:
    path = _contained_file(root, record.relative_path)
    size = path.stat().st_size
    if size != record.size_bytes:
        raise ValueError(
            f"artifact payload {record.relative_path!r} size does not match its manifest"
        )
    digest = (_sha256_file if hash_file is None else hash_file)(path)
    if digest != record.sha256:
        raise ValueError(
            f"artifact payload {record.relative_path!r} hash does not match its manifest"
        )
    return path


def _contained_file(root: Path, relative_path: str) -> Path:
    normalized = _relative_path(relative_path)
    unresolved = root / normalized
    if unresolved.is_symlink():
        raise ValueError(f"artifact payload {relative_path!r} must not be a symbolic link")
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"artifact payload {relative_path!r} is missing") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact payload {relative_path!r} escapes the artifact root")
    if not resolved.is_file():
        raise ValueError(f"artifact payload {relative_path!r} must be a regular file")
    return resolved


def _file_from_records(root: Path, records: tuple[FileRecord, ...], relative_path: str) -> Path:
    normalized = _relative_path(relative_path)
    record = next((item for item in records if item.relative_path == normalized), None)
    if record is None:
        raise ValueError(f"artifact payload {normalized!r} is not available in this view")
    return _contained_file(root, record.relative_path)


def _git(anchor: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(anchor), *arguments],
        check=check,
        capture_output=True,
    )


def _capture_source_status(
    start: Path, *, required_inputs: tuple[str, ...], git_runner=None
) -> SourceProvenance:
    git = _git if git_runner is None else git_runner
    resolved = start.resolve(strict=True)
    anchor = resolved if resolved.is_dir() else resolved.parent
    try:
        discovery = git(anchor, "rev-parse", "--show-toplevel", check=False)
    except FileNotFoundError:
        discovery = None
    root = (
        Path(discovery.stdout.decode("utf-8").strip()).resolve(strict=True)
        if discovery is not None and discovery.returncode == 0
        else None
    )
    # A wheel installed in a checkout's .venv must not inherit that checkout's
    # identity.  Only the actual src/harpy tree (or explicit root anchor) owns it.
    if root is None or not (anchor == root or resolved.is_relative_to(root / "src" / "harpy")):
        return _capture_package_source(resolved)
    commit = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    status = git(root, "status", "--porcelain=v1", "--untracked-files=normal").stdout
    tracked_diff = git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--").stdout
    lock_content = (root / "uv.lock").read_bytes()
    required_inputs_committed = all(
        git(root, "cat-file", "-e", f"HEAD:{relative_path}", check=False).returncode == 0
        for relative_path in required_inputs
    )
    return SourceStatus(
        commit=commit,
        dirty_tree=bool(status),
        tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
        dependency_lock_sha256=hashlib.sha256(lock_content).hexdigest(),
        required_inputs_committed=required_inputs_committed,
    )


def _capture_package_source(start: Path) -> PackageSourceStatus:
    candidates = (start, *start.parents)
    package = next(
        (path for path in candidates if path.name == "harpy" and (path / "__init__.py").is_file()),
        None,
    )
    if package is None:
        raise ValueError("source anchor must belong to a Harpy checkout or Python package")
    inventory = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("package source must not contain symbolic links")
        if path.is_file():
            inventory.append((relative.as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    version = None
    try:
        distribution = importlib.metadata.distribution("harpy-audio")
        installed_init = Path(distribution.locate_file("harpy/__init__.py")).resolve()
        if installed_init == package / "__init__.py":
            version = distribution.version
    except importlib.metadata.PackageNotFoundError:
        pass
    return PackageSourceStatus(
        distribution_name="harpy-audio",
        distribution_version=version,
        package_sha256=hashlib.sha256(
            json.dumps(inventory, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )
