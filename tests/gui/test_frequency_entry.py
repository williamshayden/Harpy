import math

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from harpy.gui.frequency_entry import FrequencyEntry
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.tuning import Tuning


def runtime_bounds() -> tuple[float, float]:
    spec = WorkbenchSpec.from_tuning(Tuning(reference_hz=442.0))
    return spec.minimum_frequency_hz, spec.maximum_frequency_hz


def make_entry(qtbot) -> FrequencyEntry:
    minimum_hz, maximum_hz = runtime_bounds()
    entry = FrequencyEntry(minimum_hz, maximum_hz)
    qtbot.addWidget(entry)
    entry.show()
    entry.setFocus()
    return entry


def commit(entry: FrequencyEntry, text: str) -> None:
    entry.setText(text)
    QTest.keyClick(entry, Qt.Key.Key_Return)


def test_manual_entry_respects_six_decimal_runtime_bounds(qtbot) -> None:
    # Hard-coded A440 endpoints would reject the valid 442-tuned values.
    entry = make_entry(qtbot)
    minimum_hz, maximum_hz = runtime_bounds()
    lowest_typeable = math.ceil(minimum_hz * 1_000_000.0) / 1_000_000.0
    highest_typeable = math.floor(maximum_hz * 1_000_000.0) / 1_000_000.0
    emitted: list[float] = []
    failures: list[str] = []
    entry.frequency_committed.connect(emitted.append)
    entry.validation_failed.connect(failures.append)

    commit(entry, f"{lowest_typeable:.6f}")
    commit(entry, f"{highest_typeable:.6f}")
    assert emitted == [lowest_typeable, highest_typeable]
    assert entry.text() == f"{highest_typeable:.3f}"

    commit(entry, f"{lowest_typeable - 0.000001:.6f}")
    assert entry.property("validationState") == "error"
    commit(entry, f"{highest_typeable + 0.000001:.6f}")
    assert len(failures) == 2
    assert emitted == [lowest_typeable, highest_typeable]


@pytest.mark.parametrize("text", ["1e2", "+", "-", "NaN", "Inf", "1,25", "1 25"])
def test_invalid_syntax_sets_field_error_without_frequency_signal(qtbot, text: str) -> None:
    # Permitting locale or non-finite spellings would let an ambiguous frequency through.
    entry = make_entry(qtbot)
    emitted: list[float] = []
    failures: list[str] = []
    entry.frequency_committed.connect(emitted.append)
    entry.validation_failed.connect(failures.append)

    commit(entry, text)

    assert entry.property("validationState") == "error"
    assert len(failures) == 1
    assert emitted == []


def test_successful_manual_commit_emits_parsed_float_and_renders_three_decimals(qtbot) -> None:
    # Reformatting before emission would lose the entered six-decimal value.
    entry = make_entry(qtbot)
    emitted: list[float] = []
    entry.frequency_committed.connect(emitted.append)

    commit(entry, "220.123456")

    assert emitted == [220.123456]
    assert entry.text() == "220.123"
    assert entry.property("validationState") != "error"


def test_escape_restores_last_valid_text_and_clears_error(qtbot) -> None:
    # Keeping an error after Escape would leave a restored value incorrectly invalid.
    entry = make_entry(qtbot)
    entry.set_frequency_hz(220.0)
    commit(entry, "not a number")
    assert entry.property("validationState") == "error"

    QTest.keyClick(entry, Qt.Key.Key_Escape)

    assert entry.text() == "220.000"
    assert entry.property("validationState") != "error"


def test_exact_programmatic_endpoints_survive_unchanged_enter(qtbot) -> None:
    # Reparsing a rounded irrational endpoint can place it outside the runtime bounds.
    entry = make_entry(qtbot)
    minimum_hz, maximum_hz = runtime_bounds()
    emitted: list[float] = []
    entry.frequency_committed.connect(emitted.append)

    entry.set_frequency_hz(minimum_hz)
    QTest.keyClick(entry, Qt.Key.Key_Return)
    entry.set_frequency_hz(maximum_hz)
    QTest.keyClick(entry, Qt.Key.Key_Return)

    assert emitted == [minimum_hz, maximum_hz]


def test_six_decimal_manual_value_survives_rounded_unchanged_enter(qtbot) -> None:
    # Forgetting the rendered-text cache would drift the accepted six-decimal frequency.
    entry = make_entry(qtbot)
    minimum_hz, _ = runtime_bounds()
    value = math.ceil(minimum_hz * 1_000_000.0) / 1_000_000.0
    emitted: list[float] = []
    entry.frequency_committed.connect(emitted.append)

    commit(entry, f"{value:.6f}")
    QTest.keyClick(entry, Qt.Key.Key_Return)

    assert emitted == [value, value]
    assert entry.text() == f"{value:.3f}"


def test_programmatic_set_validates_bounds_without_user_signal(qtbot) -> None:
    # Emitting from a model-to-view update would feed back as a user edit.
    entry = make_entry(qtbot)
    minimum_hz, _ = runtime_bounds()
    emitted: list[float] = []
    entry.frequency_committed.connect(emitted.append)

    entry.set_frequency_hz(220.0)
    with pytest.raises(ValueError):
        entry.set_frequency_hz(minimum_hz - 1.0)

    assert entry.text() == "220.000"
    assert emitted == []
