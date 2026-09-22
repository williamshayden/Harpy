"""Archive path isolation and split uniqueness fail before reading payload files."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

from harpy._serialization import canonical_json_bytes, sha256_file

_SPEC = importlib.util.spec_from_file_location(
    "sound_matching_archive", Path(__file__).parents[2] / "research/sound_matching/archive_study.py"
)
archive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(archive)


@pytest.mark.parametrize(
    "path", ("", ".", "../private", "/private", "a/../private", "a\\b", "C:secret", "a//b")
)
def test_portable_archive_paths_cannot_leave_the_snapshot(path):
    with pytest.raises(ValueError, match="portable relative path"):
        archive.portable_path(path)
    assert archive.portable_path("source/harpy/__init__.py") == "source/harpy/__init__.py"


def test_duplicate_report_split_is_rejected_before_model_or_archive_access(tmp_path):
    (tmp_path / "plan.json").write_bytes(canonical_json_bytes({}))
    (tmp_path / "report.json").write_bytes(
        canonical_json_bytes(
            {
                "study_plan_sha256": sha256_file(tmp_path / "plan.json"),
                "final_tests_complete": True,
                "splits": [
                    {"split": name}
                    for name in ("validation", "iid-test", "compositional-test", "validation")
                ],
            }
        )
    )
    output = tmp_path / "archive.tar.gz"
    with pytest.raises(ValueError, match="all three splits exactly once"):
        archive.prepare(
            tmp_path, tmp_path / "missing-models", tmp_path / "missing-source", [], output
        )
    assert not output.exists()


def _distribution(path, extras=()):
    source = b"# frozen source\n"
    payloads = {
        "pyproject.toml": b"[project]\n",
        "PKG-INFO": b"Name: harpy\n",
        "docs/v2-acceptance.md": b"Final qualification\r\n",
        "docs/v2-calibration.md": b"[source](../src/harpy/a.py)\n",
        "research/sound_matching/README.md": b"[docs](../../docs/v2-calibration.md)\n",
        "src/harpy/a.py": source,
    }
    with tarfile.open(path, "w:gz") as target:
        for name, payload in payloads.items():
            member = tarfile.TarInfo("harpy-fixture/" + name)
            member.size = len(payload)
            target.addfile(member, io.BytesIO(payload))
        for member in extras:
            target.addfile(member, io.BytesIO(b""))
    return {
        "files": [
            {
                "path": "a.py",
                "size_bytes": len(source),
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        ]
    }, payloads


def test_sdist_expansion_preserves_docs_and_frozen_source_bytes(tmp_path):
    distribution = tmp_path / "harpy.tar.gz"
    source, payloads = _distribution(distribution)
    destination = tmp_path / "expanded"
    result = archive.expand_distribution(distribution, destination, source)
    assert result["distribution_sha256"] == sha256_file(distribution)
    assert result["archive_prefix"] == "qualification/"
    assert len(result["members"]) == len(payloads)
    for name, payload in payloads.items():
        assert (destination / name).read_bytes() == payload


@pytest.mark.parametrize("kind", ("traversal", "symlink", "hardlink", "duplicate", "second_root"))
def test_sdist_expansion_rejects_unsafe_or_ambiguous_members(tmp_path, kind):
    names = {
        "traversal": "harpy-fixture/../outside",
        "symlink": "harpy-fixture/link",
        "hardlink": "harpy-fixture/link",
        "duplicate": "harpy-fixture/PKG-INFO",
        "second_root": "other/file",
    }
    member = tarfile.TarInfo(names[kind])
    if kind in ("symlink", "hardlink"):
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "../outside"
    distribution = tmp_path / "harpy.tar.gz"
    source, _ = _distribution(distribution, (member,))
    with pytest.raises(ValueError, match=r"portable relative|regular files|unique root"):
        archive.expand_distribution(distribution, tmp_path / "expanded", source)
    assert not (tmp_path / "outside").exists()
