import os
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path


def test_public_import_smoke_is_silent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import harpy, harpy.analysis, harpy.tuning; "
                "import harpy.envs, harpy.envs.sine_pitch; "
                "import harpy.cli, harpy.experiments; "
                "import harpy.synth, harpy.synth.curves, harpy.synth.engine, "
                "harpy.synth.patch_json"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


def test_ordinary_public_imports_do_not_load_training_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import harpy, harpy.analysis, harpy.tuning; "
                "import harpy.envs, harpy.envs.sine_pitch; "
                "import harpy.synth, harpy.synth.engine, harpy.synth.models; "
                "import harpy.cli, harpy.experiments; "
                "assert 'torch' not in sys.modules; "
                "assert 'stable_baselines3' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_top_level_package_import_does_not_load_gymnasium() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import harpy; assert 'gymnasium' not in sys.modules",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_environment_submodules_remain_importable_after_package_initialization() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import harpy.envs; "
                "import harpy.envs.models; "
                "import harpy.envs.sine_pitch; "
                "assert harpy.envs.models.ControlState; "
                "assert harpy.envs.sine_pitch.SinePitchEnv"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_environment_import_and_episode_do_not_load_training_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import gymnasium; "
                "import harpy.envs; "
                "env = gymnasium.make('Harpy/SinePitch-v0'); "
                "env.reset(seed=0); "
                "env.step(3); "
                "env.close(); "
                "assert 'torch' not in sys.modules; "
                "assert 'stable_baselines3' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_console_script_metadata_exposes_headless_commands_only() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["harpy"] == "harpy.cli:main"
    assert "harpy-sine-gym" not in scripts
    assert "harpy-sine-learn" not in scripts


def test_help_is_torch_and_sb3_free(
    tmp_path: Path,
) -> None:
    environment = _heavy_import_poisoned_environment(tmp_path)
    command = [sys.executable, "-m", "harpy", "--help"]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "{train,evaluate,run,summarize}" in completed.stdout


def _heavy_import_poisoned_environment(tmp_path: Path) -> dict[str, str]:
    poison = tmp_path / "import-poison"
    poison.mkdir()
    (poison / "sitecustomize.py").write_text(
        """
import builtins

_original_import = builtins.__import__

def _guarded_import(name, *args, **kwargs):
    blocked = ("torch", "stable_baselines3")
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise RuntimeError("help imported optional heavy dependency: " + name)
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _guarded_import
""".lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(poison), *(value for value in [python_path] if value)]
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment
