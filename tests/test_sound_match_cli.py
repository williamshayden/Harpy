"""Exercise the new CLI through real rendering and saved evidence."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from harpy.cli import main


def test_sound_matching_cli_round_trip(tmp_path: Path, capsys) -> None:
    from harpy.tasks.sound_match.durable import read_bundle

    output = tmp_path / "sound"
    assert (
        main(
            [
                "run",
                "--task",
                "sound-match-v1",
                "--actor",
                "fixed",
                "--suite",
                "tone-smoke",
                "--episode",
                "tone-001",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_dir()
    assert read_bundle(output).complete
    for format in ("text", "markdown", "csv"):
        assert main(["summarize", str(output), "--format", format]) == 0
        assert "fixed" in capsys.readouterr().out
    assert main(["summarize", str(output), "--verify-replay"]) == 0
    assert "Verified recorded synthesis" in capsys.readouterr().err
    assert (
        main(
            [
                "run",
                "--task",
                "sound-match-v1",
                "--actor",
                "fixed",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "Output already exists" in capsys.readouterr().err


@pytest.mark.parametrize("suite", ["chord-smoke", "shaping-smoke"])
def test_sound_matching_evaluation_suites(tmp_path: Path, suite: str) -> None:
    output = tmp_path / suite
    assert (
        main(
            [
                "evaluate",
                "--task",
                "sound-match-v1",
                "--actor",
                "fixed",
                "--suite",
                suite,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert main(["summarize", str(output)]) == 0


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--episode", "unknown"], "Unknown episode"),
        (["--suite", "clean"], "Unknown sound-matching suite"),
        (["--source-cents", "6000"], "do not apply"),
        (["--seed", "-1"], "nonnegative"),
        (["--device", "cuda"], "CPU"),
    ],
)
def test_invalid_sound_options_fail_before_output(
    tmp_path: Path,
    capsys,
    arguments: list[str],
    message: str,
) -> None:
    output = tmp_path / "invalid"
    assert (
        main(
            [
                "run",
                "--task",
                "sound-match-v1",
                *arguments,
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_base_sound_workflow_has_no_optional_model_imports(tmp_path: Path) -> None:
    output = tmp_path / "base"
    script = (
        "import sys; from harpy.cli import main; "
        "assert main(['run','--task','sound-match-v1','--actor','fixed',"
        f"'--output',{str(output)!r}]) == 0; "
        "assert main(['summarize'," + repr(str(output)) + "]) == 0; "
        "assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
