"""Archive path isolation and split uniqueness fail before reading payload files."""

import importlib.util
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
