"""Listening examples cannot acquire a plan identity from unrelated inputs."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from harpy._serialization import canonical_json_bytes, sha256_file
from harpy.tasks.sound_match.datasets import load_manifest

ROOT = Path(__file__).parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "validation_listening", ROOT / "research/sound_matching/validation_listening.py"
)
listening = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(listening)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    study, evaluator = tmp_path / "study", tmp_path / "evaluator"
    (study / "validation").mkdir(parents=True)
    (evaluator / "research/sound_matching").mkdir(parents=True)
    rows = []
    for name in ("study.py", "listening.py"):
        relative = "research/sound_matching/" + name
        path = evaluator / relative
        path.write_bytes((ROOT / relative).read_bytes())
        rows.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    snapshot = {"schema_id": "harpy.source-snapshot/1", "files": rows}
    (evaluator / "snapshot-manifest.json").write_bytes(canonical_json_bytes(snapshot))
    real_reader = listening._load_module("_validation_test_driver", evaluator / rows[0]["path"])
    membership = load_manifest("validation")
    plan = {
        "schema_id": real_reader.STUDY_ID,
        "driver_sha256": rows[0]["sha256"],
        "actors": list(real_reader.ACTORS),
        "splits": {"validation": {"targets": 192, "sha256": membership.manifest_sha256}},
    }
    (study / "plan.json").write_bytes(canonical_json_bytes(plan))
    for name in ("manifest.json", "complete.json"):
        (study / "validation" / name).write_bytes(canonical_json_bytes({"fixture": name}))
    result = {
        "split": "validation",
        "bundle_manifest_sha256": sha256_file(study / "validation/manifest.json"),
        "completion_sha256": sha256_file(study / "validation/complete.json"),
        "report": {"complete": True, "completed": 2112},
        "actors": [{"actor": name, "episodes": 192} for name in real_reader.ACTORS],
    }
    quality = {"fixture": "strict reader result"}
    calls = []

    def split_report(bundle, actual_plan, plan_hash, split):
        calls.append((bundle, copy.deepcopy(actual_plan), plan_hash, split))
        return copy.deepcopy(result)

    reader = SimpleNamespace(
        STUDY_ID=real_reader.STUDY_ID,
        ACTORS=real_reader.ACTORS,
        split_report=split_report,
        reference_quality=lambda actors: quality,
    )
    report = {
        "schema_id": "harpy.sound-match.matched-study-report/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "splits": [copy.deepcopy(result)],
        "reference_quality": quality,
        "final_tests_complete": False,
        "perceptual_success_cutoff": None,
        "listening_review": "not-performed",
    }
    (study / "validation-report.json").write_bytes(canonical_json_bytes(report))
    original_load = listening._load_module
    monkeypatch.setattr(
        listening,
        "_load_module",
        lambda name, path: (
            reader if name == "_frozen_validation_reader" else original_load(name, path)
        ),
    )
    return SimpleNamespace(
        study=study,
        evaluator=evaluator,
        selector=evaluator / "research/sound_matching/listening.py",
        plan=plan,
        snapshot=snapshot,
        report=report,
        result=result,
        calls=calls,
    )


def test_selector_is_rejected_before_executing_its_code(frozen_inputs, tmp_path):
    fixture = frozen_inputs
    altered = tmp_path / "altered-selector.py"
    altered.write_text("raise AssertionError('unverified selector executed')\n")
    with pytest.raises(ValueError, match="selector differs"):
        listening._preflight(fixture.study, altered, fixture.evaluator)
    assert fixture.calls == []


@pytest.mark.parametrize("mutation", ("driver", "actors", "membership"))
def test_unrelated_plan_cannot_label_validation_audio(frozen_inputs, mutation):
    fixture = frozen_inputs
    if mutation == "driver":
        fixture.plan["driver_sha256"] = "0" * 64
    elif mutation == "actors":
        fixture.plan["actors"] = ["fixed"]
    else:
        fixture.plan["splits"]["validation"]["targets"] = 8
    (fixture.study / "plan.json").write_bytes(canonical_json_bytes(fixture.plan))
    with pytest.raises(ValueError, match=r"reader differs|full frozen"):
        listening._preflight(fixture.study, fixture.selector, fixture.evaluator)
    assert fixture.calls == []


def test_snapshot_cannot_have_ambiguous_selector_identity(frozen_inputs):
    fixture = frozen_inputs
    fixture.snapshot["files"].append(fixture.snapshot["files"][-1])
    (fixture.evaluator / "snapshot-manifest.json").write_bytes(
        canonical_json_bytes(fixture.snapshot)
    )
    with pytest.raises(ValueError, match="identify the listening inputs once"):
        listening._preflight(fixture.study, fixture.selector, fixture.evaluator)


def test_retained_report_requires_fresh_strict_recomputation(frozen_inputs):
    fixture = frozen_inputs
    fixture.report["reference_quality"] = {"all_three_passed": True}
    (fixture.study / "validation-report.json").write_bytes(canonical_json_bytes(fixture.report))
    with pytest.raises(ValueError, match="strict frozen study recomputation"):
        listening._preflight(fixture.study, fixture.selector, fixture.evaluator)
    assert fixture.calls == [
        (
            fixture.study / "validation",
            fixture.plan,
            sha256_file(fixture.study / "plan.json"),
            "validation",
        )
    ]


def test_partial_bundle_cannot_supply_listening_index(frozen_inputs):
    fixture = frozen_inputs
    fixture.result["report"]["completed"] -= 1
    with pytest.raises(ValueError, match="complete frozen validation split"):
        listening._preflight(fixture.study, fixture.selector, fixture.evaluator)


def test_inputs_must_stay_bound_through_selection(frozen_inputs, monkeypatch, tmp_path):
    fixture = frozen_inputs

    def changed_records(bundle):
        (bundle / "manifest.json").write_bytes(canonical_json_bytes({"changed": True}))
        yield from ()

    monkeypatch.setattr(listening, "iter_records", changed_records)
    output = tmp_path / "index.json"
    with pytest.raises(ValueError, match="changed during listening selection"):
        listening.prepare(fixture.study, fixture.selector, output, evaluator=fixture.evaluator)
    assert not output.exists()
