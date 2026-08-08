import subprocess
import sys


def test_public_import_smoke_is_silent() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; "
                "import harpy.gui.app; "
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
