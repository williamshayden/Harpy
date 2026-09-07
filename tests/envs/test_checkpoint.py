"""Machine-readable Milestone C checkpoint command."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from importlib.metadata import entry_points
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import pytest

import harpy.envs
import harpy.envs.checkpoint as checkpoint
from harpy.envs.baselines import BaselineSummary


def test_main_has_the_narrow_argv_api() -> None:
    assert tuple(signature(checkpoint.main).parameters) == ("argv",)
    assert get_type_hints(checkpoint.main) == {
        "argv": Sequence[str] | None,
        "return": int,
    }


def test_checkpoint_defaults_are_forwarded_once_and_serialized(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[int, int]] = []

    def evaluate(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]:
        calls.append((episodes, seed))
        return _SUMMARIES

    monkeypatch.setattr(checkpoint, "evaluate_all_baselines", evaluate)

    assert checkpoint.main([]) == 0
    captured = capsys.readouterr()

    assert calls == [(10, 0)]
    assert captured.out == _expected_document(episodes=10, seed=0)
    assert captured.err == ""


def test_checkpoint_retains_result_order_is_repeatable_and_writes_no_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, int]] = []

    def evaluate(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]:
        calls.append((episodes, seed))
        return _SUMMARIES

    monkeypatch.setattr(checkpoint, "evaluate_all_baselines", evaluate)
    monkeypatch.chdir(tmp_path)

    assert checkpoint.main(["--episodes", "3", "--seed", "7"]) == 0
    first = capsys.readouterr()
    assert checkpoint.main(["--episodes", "3", "--seed", "7"]) == 0
    second = capsys.readouterr()

    expected = _expected_document(episodes=3, seed=7)
    assert calls == [(3, 7), (3, 7)]
    assert first.out == expected
    assert second.out == expected
    assert first.out.encode() == second.out.encode()
    assert first.err == second.err == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["evaluation", "serialization"])
def test_checkpoint_builds_the_complete_document_before_writing_stdout(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if failure == "evaluation":

        def evaluate(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]:
            raise RuntimeError("evaluation failed")

        expected_error = RuntimeError
    else:

        class NonFiniteSummary:
            @staticmethod
            def to_dict() -> dict[str, float]:
                return {"mean_return": float("nan")}

        def evaluate(*, episodes: int, seed: int) -> tuple[NonFiniteSummary, ...]:
            return (NonFiniteSummary(),)

        expected_error = ValueError

    monkeypatch.setattr(checkpoint, "evaluate_all_baselines", evaluate)

    with pytest.raises(expected_error):
        checkpoint.main(["--episodes", "1", "--seed", "0"])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--episodes", "0"],
        ["--episodes", "-1"],
        ["--episodes", "1.5"],
        ["--episodes", "true"],
        ["--seed", "-1"],
        ["--seed", "1.5"],
        ["--seed", "false"],
        ["--unknown"],
        ["--episodes"],
        ["--seed"],
    ],
)
def test_invalid_arguments_use_standard_argparse_errors_without_evaluation(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_evaluation(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]:
        raise AssertionError("invalid arguments must not start evaluation")

    monkeypatch.setattr(checkpoint, "evaluate_all_baselines", forbidden_evaluation)

    with pytest.raises(SystemExit) as raised:
        checkpoint.main(argv)

    captured = capsys.readouterr()
    assert raised.value.code == 2
    assert captured.out == ""
    assert captured.err.startswith("usage: harpy-sine-gym")


def test_console_script_metadata_targets_checkpoint_main() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["harpy-sine-gym"] == "harpy.envs.checkpoint:main"


def test_checkpoint_is_not_a_harpy_envs_compatibility_export() -> None:
    assert "checkpoint" not in harpy.envs.__all__


@pytest.mark.parametrize("entry_point", ["module", "console"])
def test_entry_points_run_without_stderr_or_files(entry_point: str, tmp_path: Path) -> None:
    run_directory = tmp_path / "empty-cwd"
    run_directory.mkdir()
    if entry_point == "module":
        command = [sys.executable, "-m", "harpy.envs.checkpoint"]
    else:
        console_script = shutil.which("harpy-sine-gym")
        assert console_script is not None
        command = [console_script]

    completed = subprocess.run(
        [*command, "--episodes", "1", "--seed", "0"],
        cwd=run_directory,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == 1
    assert payload["checkpoint_id"] == "harpy-milestone-c-sine-pitch-v0"
    assert payload["config_id"] == "fixed-default-sine-v0"
    assert payload["episodes"] == 1
    assert payload["seed"] == 0
    assert [row["baseline"] for row in payload["results"]] == [
        "random",
        "spectrum_peak",
        "oracle",
        "reward_search",
    ]
    assert completed.stdout.endswith("\n")
    assert not completed.stdout.endswith("\n\n")
    assert list(run_directory.iterdir()) == []


def _expected_document(*, episodes: int, seed: int) -> str:
    payload = {
        "schema_version": 1,
        "checkpoint_id": "harpy-milestone-c-sine-pitch-v0",
        "config_id": "fixed-default-sine-v0",
        "episodes": episodes,
        "seed": seed,
        "results": [summary.to_dict() for summary in _SUMMARIES],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"


_SUMMARY_FIELDS = {
    "episodes": 99,
    "seed": 123,
    "submitted_success_rate": 0.25,
    "submitted_within_1_cent_rate": 0.125,
    "final_within_5_cents_rate": 0.5,
    "final_within_1_cent_rate": 0.25,
    "mean_absolute_final_error_cents": 17.0,
    "mean_actions": 8.5,
    "mean_excess_actions": None,
    "mean_return": -0.75,
    "truncation_rate": 0.125,
    "invalid_action_rate": 0.0625,
}
_SUMMARIES = (
    BaselineSummary(
        baseline="random",
        environment_id="Harpy/SinePitch-v0",
        observation_mode="spectrum",
        **_SUMMARY_FIELDS,
    ),
    BaselineSummary(
        baseline="spectrum_peak",
        environment_id="Harpy/SinePitch-v0",
        observation_mode="spectrum",
        **_SUMMARY_FIELDS,
    ),
    BaselineSummary(
        baseline="oracle",
        environment_id="Harpy/SinePitchOracle-v0",
        observation_mode="oracle",
        **_SUMMARY_FIELDS,
    ),
    BaselineSummary(
        baseline="reward_search",
        environment_id="Harpy/SinePitchRewardOnly-v0",
        observation_mode="reward_only",
        **_SUMMARY_FIELDS,
    ),
)
