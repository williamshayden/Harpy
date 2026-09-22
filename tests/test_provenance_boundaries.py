"""Active infrastructure and read-only codecs do not import historical execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_active_experiment_io_does_not_import_historical_artifacts(tmp_path: Path) -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

class BlockHistorical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'harpy.learning.artifacts', 'harpy.learning.workflows'}:
            raise AssertionError('active experiment imported historical infrastructure: '
                                 + fullname)

sys.meta_path.insert(0, BlockHistorical())
from harpy.experiments import EpisodeSpec, run, save_result, load_result, spectrum_peak_actor_spec
result = run(EpisodeSpec('boundary', 6000, 12), spectrum_peak_actor_spec())
save_result(result, Path(sys.argv[1]))
assert load_result(Path(sys.argv[1])) == result
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "result.json")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_historical_readout_is_independent_of_execution_module() -> None:
    import runpy

    from harpy._serialization import canonical_json_bytes

    fixture = runpy.run_path(str(Path(__file__).parent / "learning" / "test_workflows.py"))
    payload = canonical_json_bytes(fixture["_report"]().to_document())
    script = """
import importlib.abc
import sys

class BlockExecution(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'harpy.learning.workflows', 'torch', 'stable_baselines3'}:
            raise AssertionError('read-only codec imported execution: ' + fullname)

sys.meta_path.insert(0, BlockExecution())
from harpy.learning.readout import summarize_report_bytes
summary = summarize_report_bytes(sys.stdin.buffer.read())
assert 'Harpy research readout' in summary
assert 'harpy.learning.workflows' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_historical_provenance_type_imports_retain_identity() -> None:
    from harpy import _provenance
    from harpy.learning import artifacts

    for name in ("FileRecord", "SourceStatus", "PackageSourceStatus"):
        assert getattr(artifacts, name) is getattr(_provenance, name)
