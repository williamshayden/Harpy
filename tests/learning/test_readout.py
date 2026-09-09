"""Saved evidence remains strict and usable without training dependencies."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from harpy import cli
from harpy.envs.baselines import BaselineKind
from harpy.learning.artifacts import SourceStatus, canonical_json_bytes
from harpy.learning.diagnostic_codecs import diagnostic_report_bytes
from harpy.learning.diagnostics import build_diagnostic_report
from harpy.learning.experiment_results import (
    ArtifactProvenance,
    ArtifactRunResult,
    ExploratoryPitchDiagnostics,
    ExploratoryPitchEvaluation,
    readout_bytes,
)
from harpy.learning.models import DeviceName, ProfileName
from harpy.learning.pitch_data import PitchEvaluationSuiteId, fixed_pitch_evaluation_suite
from harpy.learning.readout import summarize_report_bytes
from harpy.learning.trace import trace_from_document
from harpy.learning.workflows import evaluation_report_bytes

from .test_diagnostic_codecs import _pitch_episodes, _report
from .test_pitch_comparison import _report as _comparison_report

_V1_REPORT = Path(__file__).parents[1] / "fixtures/learning/schema-v1/evaluation-report.json"


def _pitch_fixture(name: str):
    # Existing synthetic report builders also import training test helpers. Keep
    # that dependency local so base-install collection still tests the reader.
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from . import test_pitch_workflows

    return getattr(test_pitch_workflows, name)()


def _smoke_report():
    return _pitch_fixture("_smoke_report")


def _checkpoint_report():
    return _pitch_fixture("_checkpoint_report")


def _e1_report():
    return _pitch_fixture("_e1_report")


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(
        artifact_schema_version=2,
        manifest_sha256="a" * 64,
        model_sha256="b" * 64,
        trainer="pitch",
        training_seed=0,
        profile=ProfileName.SMOKE,
        training_device=DeviceName.CPU,
        evaluation_device=DeviceName.CPU,
        source=SourceStatus(
            commit="1" * 40,
            dirty_tree=False,
            tracked_diff_sha256=hashlib.sha256(b"").hexdigest(),
            dependency_lock_sha256="d" * 64,
            required_inputs_committed=True,
        ),
    )


def _exploration() -> ExploratoryPitchEvaluation:
    rows = _smoke_report().terminal_rows
    return ExploratoryPitchEvaluation(
        provenance=_provenance(),
        terminal_rows=(
            rows[0],
            *(next(row for row in rows if row.actor_id == kind.value) for kind in BaselineKind),
        ),
    )


def _run_result() -> ArtifactRunResult:
    trace_document = json.loads((_V1_REPORT.parent / "episode-trace.json").read_bytes())
    return ArtifactRunResult(provenance=_provenance(), trace=trace_from_document(trace_document))


def _exploratory_diagnostics() -> ExploratoryPitchDiagnostics:
    suite = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.SMOKE)
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256="a" * 64,
        actor_semantics="harpy-sine-pitch-estimator-planner-v1",
        bound_mask=False,
        suite_id=suite.suite_id.value,
        suite_digest_sha256=suite.digest_sha256,
        episodes=_pitch_episodes(PitchEvaluationSuiteId.SMOKE),
    )
    return ExploratoryPitchDiagnostics(provenance=_provenance(), diagnostics=report)


@pytest.mark.parametrize("format", ["text", "markdown"])
def test_summary_preserves_source_and_explains_ineligible_smoke_results(
    format: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    report = tmp_path / "report.json"
    content = evaluation_report_bytes(_smoke_report())
    report.write_bytes(content)
    from harpy.learning import dependencies

    def unexpected_model_stack():
        pytest.fail("summary requested a model stack")

    monkeypatch.setattr(dependencies, "require_pitch_dependencies", unexpected_model_stack)
    monkeypatch.setattr(dependencies, "require_training_dependencies", unexpected_model_stack)

    assert cli.main(["summarize", str(report), "--format", format]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Pitch criterion: ineligible" in captured.out
    assert "Scientifically eligible: no" in captured.out
    assert "pitch-0" in captured.out and "spectrum_peak" in captured.out
    assert "Success" in captured.out and "MAE (cents)" in captured.out
    assert "do not pool" in captured.out
    assert report.read_bytes() == content
    assert list(tmp_path.iterdir()) == [report]
    if format == "markdown":
        assert captured.out.startswith("# Harpy research readout\n\n")
        assert "| Actor | Seed | Track | Suite |" in captured.out


def test_historical_bc_ppo_report_has_separate_status_and_baselines() -> None:
    summary = summarize_report_bytes(_V1_REPORT.read_bytes())
    assert "evaluation schema v1" in summary
    assert "BC criterion: ineligible" in summary
    assert "PPO criterion: ineligible" in summary
    assert "bc-0" in summary and "ppo-2" in summary and "oracle" in summary


@pytest.mark.parametrize("format", ["text", "markdown"])
def test_saved_decoder_comparison_cli_and_readout_preserve_identity(
    format: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from harpy.learning.pitch_comparison import comparison_bytes

    content = comparison_bytes(_comparison_report())
    output = tmp_path / "comparison.json"
    output.write_bytes(content)
    assert cli.main(["summarize", str(output), "--format", format]) == 0
    captured = capsys.readouterr()
    summary = summarize_report_bytes(content, format=format)
    assert captured.err == ""
    assert captured.out == summary
    assert output.read_bytes() == content
    assert "Scientifically eligible: no" in summary
    assert "Paired decoder changes" in summary
    assert "pitch-global-argmax-7" in summary and "pitch-feasible-argmax-7" in summary
    assert "Rescued" in summary and "Regressed" in summary
    assert "commits its initial plan" in summary


@pytest.mark.parametrize("factory", [_exploration, _run_result, _exploratory_diagnostics])
def test_portable_readout_keeps_provenance_and_explicit_ineligibility(factory) -> None:
    summary = summarize_report_bytes(readout_bytes(factory()))
    assert "Scientifically eligible: no" in summary
    assert "manifest sha256: " + "a" * 64 in summary
    assert "model sha256: " + "b" * 64 in summary
    if factory is _run_result:
        assert "Demonstration outcome" in summary
        assert "Episode seed:" in summary
        assert "single_episode_demonstration" in summary
    elif factory is _exploration:
        assert "pitch-0" in summary and "spectrum_peak" in summary
        assert "exploratory_single_artifact" in summary
    else:
        assert "Diagnostic hotspots" in summary
        assert "exploratory_single_artifact" in summary


@pytest.mark.parametrize("factory", [_checkpoint_report, _e1_report])
def test_checkpoint_and_cohort_readouts_preserve_verdict_and_all_seeds(factory) -> None:
    report = factory()
    summary = summarize_report_bytes(evaluation_report_bytes(report))
    assert f"Pitch criterion: {report.criterion.status}" in summary
    for gate in report.criterion.failed_gates:
        assert gate in summary
    for seed in (0, 1, 2):
        assert f"pitch-{seed}" in summary
    if report.schema_version == 3:
        assert report.cohort_digest_sha256 in summary
        assert "Training device: cuda" in summary
        assert "Evaluation device: cpu" in summary


def test_diagnostic_readout_identifies_failed_episodes_without_claiming_eligibility() -> None:
    summary = summarize_report_bytes(diagnostic_report_bytes(_report(source_pitch_cents=4_900)))
    assert "Diagnostic hotspots" in summary
    assert "submitted_failure" in summary
    assert "100" in summary
    assert "Loop episodes" in summary and "Bound blocks" in summary
    assert "do not grant scientific eligibility" in summary


def test_summary_rejects_tampered_evidence_before_printing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    document = _smoke_report().to_document()
    document["terminal_rows"][0]["metrics"]["submitted_success_rate"] = 0.987
    path = tmp_path / "tampered.json"
    path.write_bytes(canonical_json_bytes(document))

    assert cli.main(["summarize", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("harpy: ")


@pytest.mark.parametrize("content", [b"{", b'{"schema_version":1,"schema_version":1}', b"{}"])
def test_summary_rejects_malformed_or_unknown_documents(content: bytes) -> None:
    with pytest.raises(ValueError):
        summarize_report_bytes(content)


def test_all_report_codecs_can_be_read_without_torch_or_sb3(tmp_path: Path) -> None:
    from harpy.learning.pitch_comparison import comparison_bytes

    contents = [
        comparison_bytes(_comparison_report()),
        _V1_REPORT.read_bytes(),
        evaluation_report_bytes(_smoke_report()),
        evaluation_report_bytes(_e1_report()),
        diagnostic_report_bytes(_report()),
        readout_bytes(_exploration()),
        readout_bytes(_run_result()),
        readout_bytes(_exploratory_diagnostics()),
    ]
    paths = []
    for index, content in enumerate(contents):
        path = tmp_path / f"report-{index}.json"
        path.write_bytes(content)
        paths.append(str(path))
    code = """
import importlib.abc
import json
import pathlib
import sys
class BlockTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'stable_baselines3'}:
            raise AssertionError(f'readout imported {fullname}')
sys.meta_path.insert(0, BlockTraining())
from harpy.cli import main
from harpy.learning.readout import summarize_report_bytes
for path in json.loads(sys.argv[1]):
    assert 'Harpy research readout' in summarize_report_bytes(pathlib.Path(path).read_bytes())
    assert main(['summarize', path]) == 0
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(paths)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
