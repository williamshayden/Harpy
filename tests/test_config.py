import ast
from dataclasses import replace
from pathlib import Path

import pytest

from harpy.analysis import AnalysisConfig
from harpy.config import DEFAULT_CONFIG, AppConfig
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch


def test_default_config_is_the_approved_core_composition() -> None:
    assert AppConfig() == DEFAULT_CONFIG
    assert DEFAULT_CONFIG.tuning.reference_hz == 440.0
    assert DEFAULT_CONFIG.render == RenderConfig()
    assert DEFAULT_CONFIG.analysis == AnalysisConfig()
    assert DEFAULT_CONFIG.patch == SynthPatch()


def test_app_config_rejects_a_patch_with_a_sub_frame_segment() -> None:
    patch = SynthPatch(envelope=EnvelopeConfig(attack_seconds=1e-12))

    with pytest.raises(ValueError, match="attack segment must contain at least one frame"):
        AppConfig(patch=patch)


def test_app_config_rejects_analysis_that_exceeds_render_nyquist() -> None:
    analysis = replace(AnalysisConfig(), spectrum_max_hz=20_000.0)

    with pytest.raises(ValueError, match="within Nyquist"):
        AppConfig(render=RenderConfig(sample_rate_hz=32_000), analysis=analysis)


def test_config_has_no_direct_gui_imports() -> None:
    source = Path(__file__).parents[1] / "src" / "harpy" / "config.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        imported
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for imported in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
        )
    }

    assert not {name for name in imports if name.startswith("harpy.gui")}
