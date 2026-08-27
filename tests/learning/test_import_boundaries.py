"""Import and optional-training-stack contracts for learning code."""

from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

import pytest

from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import DependencyUnavailableError


def test_learning_import_loads_no_training_or_qt_modules() -> None:
    """The public learning package stays importable without optional dependencies."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, harpy.learning; "
                "assert 'torch' not in sys.modules; "
                "assert 'stable_baselines3' not in sys.modules; "
                "assert not any(n == 'PySide6' or n.startswith('PySide6.') "
                "for n in sys.modules)"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_pitch_training_import_loads_no_qt_modules() -> None:
    """The explicit optional trainer import remains independent of the GUI stack."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, harpy.learning.pitch; "
                "assert not any(n == 'PySide6' or n.startswith('PySide6.') "
                "for n in sys.modules)"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("missing_name", ["torch", "stable_baselines3"])
def test_training_dependency_failure_has_install_instruction(missing_name: str) -> None:
    """A missing optional package reports the exact actionable installation command."""

    def unavailable_import(name: str) -> ModuleType:
        if name == missing_name:
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    with pytest.raises(DependencyUnavailableError, match="uv sync --group train"):
        require_training_dependencies(import_module=unavailable_import)


def test_training_stack_reports_installed_module_versions() -> None:
    """The lazy loader returns the exact versions of the imported training modules."""
    stack = require_training_dependencies()

    assert stack.torch_version == stack.torch.__version__
    assert stack.stable_baselines3_version == stack.stable_baselines3.__version__
