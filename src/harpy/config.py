from dataclasses import dataclass, field

from harpy.analysis import AnalysisConfig, validate_analysis_config
from harpy.synth.models import RenderConfig, SynthPatch, validate_renderable_patch
from harpy.tuning import Tuning


@dataclass(frozen=True, slots=True)
class AppConfig:
    tuning: Tuning = field(default_factory=Tuning)
    render: RenderConfig = field(default_factory=RenderConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    patch: SynthPatch = field(default_factory=SynthPatch)

    def __post_init__(self) -> None:
        validate_renderable_patch(self.patch, self.render)
        validate_analysis_config(self.analysis, self.render.sample_rate_hz)


DEFAULT_CONFIG = AppConfig()
