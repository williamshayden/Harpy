"""Real SIGKILL evidence for both adapters, including kernel lock recovery."""

from __future__ import annotations

import copy
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


def scientific(payload):
    result = copy.deepcopy(payload)
    result.pop("operational", None)
    if result["record"] is not None:
        for field in ("elapsed_seconds", "dsp_renders", "cache_hits"):
            result["record"].pop(field, None)
    return result


@pytest.mark.parametrize("task", ["sound", "pitch"])
def test_sigkill_retains_prefix_resumes_missing_only_and_releases_lock(tmp_path, task):
    module = importlib.import_module(
        "harpy.tasks.sound_match.durable" if task == "sound" else "harpy.experiments.durable"
    )
    worker = Path(__file__).parent / "fixtures" / "durable_worker.py"
    bundle, baseline, marker = tmp_path / "killed", tmp_path / "uninterrupted", tmp_path / "ready"

    def command(output, operation="new"):
        return [sys.executable, str(worker), task, str(output), operation]

    environment = {**os.environ, "HARPY_TEST_PAUSE": str(marker)}
    process = subprocess.Popen(
        command(bundle), env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), (
            process.communicate(timeout=2) if process.poll() is not None else "worker did not park"
        )
        # This is an actual second process contending with a live writer.
        competing = subprocess.run(
            command(bundle, "resume"), capture_output=True, text=True, timeout=20
        )
        assert competing.returncode != 0 and "in use" in competing.stderr
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    partial = module.read_bundle(bundle, allow_incomplete=True)
    assert (partial.completed, partial.expected, partial.attempts) == (2, 4, 3)
    assert partial.interrupted_attempts == 1 and partial.missing == 2
    prefix = (bundle / "records.jsonl").read_bytes()
    committed = (bundle / "commits.jsonl").read_bytes()
    subprocess.run(
        command(bundle, "resume"), check=True, capture_output=True, text=True, timeout=30
    )
    subprocess.run(command(baseline), check=True, capture_output=True, text=True, timeout=30)
    assert (bundle / "records.jsonl").read_bytes().startswith(prefix)
    assert (bundle / "commits.jsonl").read_bytes().startswith(committed)
    result = module.read_bundle(bundle)
    assert result.complete and result.completed == 4 and result.attempts == 5
    assert result.to_document()["interrupted_costs"] == "unknown"
    assert [scientific(row) for row in module.iter_records(bundle)] == [
        scientific(row) for row in module.iter_records(baseline)
    ]
    module.verify_replay(bundle)


@pytest.mark.parametrize("task", ["sound", "pitch"])
def test_relocated_readout_and_resume_need_no_original_artifacts_or_model_imports(tmp_path, task):
    worker = Path(__file__).parent / "fixtures" / "durable_worker.py"
    output = tmp_path / "original"
    subprocess.run(
        [sys.executable, str(worker), task, str(output), "new"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    relocated = tmp_path / "new-location"
    output.rename(relocated)
    module = "harpy.tasks.sound_match.durable" if task == "sound" else "harpy.experiments.durable"
    script = (
        "import sys\nfrom pathlib import Path\n"
        f"from {module} import read_bundle, summarize_bundle, verify_replay\n"
        "path = Path(sys.argv[1])\nassert read_bundle(path).complete\n"
        "for fmt in ('text', 'markdown', 'csv'): assert summarize_bundle(path, fmt)\n"
        "verify_replay(path)\n"
        "assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules\n"
    )
    subprocess.run(
        [sys.executable, "-c", script, str(relocated)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [sys.executable, str(worker), task, str(relocated), "resume"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
