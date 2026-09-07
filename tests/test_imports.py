import subprocess
import sys


def test_public_import_smoke_is_silent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import harpy, harpy.analysis, harpy.tuning; "
                "import harpy.envs, harpy.envs.sine_pitch; "
                "import harpy.learning, harpy.learning.cli; "
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
                "import harpy.learning, harpy.learning.cli; "
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
