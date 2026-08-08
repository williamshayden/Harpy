import pytest

from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.tuning import Tuning


def test_workbench_range_is_runtime_tuned_c2_c3_c4() -> None:
    spec = WorkbenchSpec.from_tuning(Tuning(reference_hz=442.0))
    assert spec.center_frequency_hz**2 == pytest.approx(
        spec.minimum_frequency_hz * spec.maximum_frequency_hz,
        rel=1e-12,
    )
    assert Tuning(reference_hz=442.0).describe_frequency(spec.minimum_frequency_hz).name == "C2"
    assert Tuning(reference_hz=442.0).describe_frequency(spec.center_frequency_hz).name == "C3"
    assert Tuning(reference_hz=442.0).describe_frequency(spec.maximum_frequency_hz).name == "C4"
