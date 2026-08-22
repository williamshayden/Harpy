import subprocess
import sys


def test_public_import_smoke_is_silent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; "
                "import harpy.gui.app, harpy.gui.envelope_editor, harpy.gui.window; "
                "import harpy.synth.curves, harpy.synth.engine, harpy.synth.patch_json"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout == b""
    assert result.stderr == b""


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


def test_environment_import_and_episode_do_not_load_qt_or_audio_devices() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['PySide6'] = None; "
                "import gymnasium; "
                "import harpy.envs; "
                "env = gymnasium.make('Harpy/SinePitch-v0'); "
                "env.reset(seed=0); "
                "env.step(3); "
                "env.close(); "
                "assert not any(name == 'harpy.gui' or name.startswith('harpy.gui.') "
                "or name == 'PySide6' or name.startswith('PySide6.') "
                "for name in sys.modules if sys.modules[name] is not None)"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()
