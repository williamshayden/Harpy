import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from harpy.gui.envelope_entry import (
    EnvelopeFieldKind,
    EnvelopeValueEntry,
    format_envelope_curvature,
    format_envelope_decibels,
    format_envelope_duration,
)


def test_focus_loss_requests_silent_transient_cancel(qtbot) -> None:
    # Omitting the signal would leave the transient owner unable to close a lost editor.
    host = QWidget()
    layout = QVBoxLayout(host)
    entry = EnvelopeValueEntry("Attack", EnvelopeFieldKind.DURATION, host)
    other = QLineEdit(host)
    layout.addWidget(entry)
    layout.addWidget(other)
    qtbot.addWidget(host)
    cancelled: list[None] = []
    entry.focus_cancel_requested.connect(lambda: cancelled.append(None))
    entry.set_exact_value(0.001)
    host.show()
    host.activateWindow()
    entry.setFocus()
    QApplication.processEvents()
    other.setFocus()
    QApplication.processEvents()
    assert cancelled == [None]
    assert other.hasFocus()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-6.0, "-6 dB"), (-0.0, "0 dB"), (-12.3456789, "-12.345679 dB")],
)
def test_decibel_formatter_is_canonical(value: float, expected: str) -> None:
    # A divergent dB formatter would make graph controls disagree with exact entry.
    assert format_envelope_decibels(value) == expected


def test_curvature_formatter_preserves_six_decimal_precision() -> None:
    # Curve handles and their contextual readout still share this canonical formatter.
    assert format_envelope_curvature(0.123456789) == "0.123457"
    assert format_envelope_curvature(-0.0) == "0"


def test_field_kinds_are_exactly_the_two_transient_text_modes() -> None:
    # Retaining curvature would keep the removed permanent curve-entry grammar alive.
    assert set(EnvelopeFieldKind) == {
        EnvelopeFieldKind.DURATION,
        EnvelopeFieldKind.DECIBELS,
    }


def test_parser_rejection_announces_and_restores_clean_accessibility(qtbot, monkeypatch) -> None:
    # Failing to announce a parser error leaves screen-reader users without feedback.
    entry = EnvelopeValueEntry("Attack", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    events: list[QAccessibleAnnouncementEvent] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)
    base_description = "Milliseconds or seconds. Drag vertically; Enter or F2 edits exactly."
    entry.set_editor_accessibility("Attack duration", base_description)
    entry.set_exact_value(0.001)
    entry.selectAll()
    qtbot.keyClicks(entry, "nope")
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert entry.accessibleDescription() == (
        f"{base_description} Attack: enter a plain duration with optional ms or s."
    )
    assert len(events) == 1
    assert events[0].object() is entry
    qtbot.keyPress(entry, Qt.Key.Key_Escape)
    assert entry.accessibleDescription() == base_description
    assert len(events) == 1


def test_public_error_clear_restores_the_accessible_description(qtbot) -> None:
    # A private-only clear boundary forces owner widgets to reach into entry internals.
    entry = EnvelopeValueEntry("Attack", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    base_description = "Milliseconds or seconds."
    entry.set_editor_accessibility("Attack duration", base_description)
    entry.set_exact_value(0.001)
    entry.mark_commit_rejected("Attack duration is invalid.")

    entry.clear_error()

    assert entry.property("validationState") is None
    assert entry.accessibleDescription() == base_description


def assert_invalid_commit(qtbot, entry: EnvelopeValueEntry, text: str, field: str) -> None:
    # Removing any grammar or bounds check would allow an invalid proposal through.
    initial = {
        EnvelopeFieldKind.DURATION: 0.6,
        EnvelopeFieldKind.DECIBELS: -6.0,
    }[entry.kind]
    qtbot.addWidget(entry)
    entry.set_exact_value(initial)
    committed: list[float] = []
    errors: list[str] = []
    entry.value_commit_requested.connect(committed.append)
    entry.validation_failed.connect(errors.append)
    entry.selectAll()
    if text:
        qtbot.keyClicks(entry, text)
    else:
        qtbot.keyPress(entry, Qt.Key.Key_Backspace)
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    assert committed == []
    assert len(errors) == 1 and field in errors[0]
    assert entry.text() == text
    assert entry.property("validationState") == "error"


@pytest.mark.parametrize(
    ("text", "expected_seconds"),
    [
        ("1 ms", 0.001),
        ("600ms", 0.600),
        ("0.6 s", 0.600),
        ("1.25s", 1.25),
    ],
)
def test_duration_entry_accepts_explicit_ms_and_s(
    qtbot,
    text: str,
    expected_seconds: float,
) -> None:
    # Breaking either unit branch would emit the wrong authored duration.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(
        lambda value: (committed.append(value), entry.accept_proposed_value(value))
    )
    entry.set_exact_value(0.001)
    entry.selectAll()
    qtbot.keyClicks(entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    assert committed == [expected_seconds]


@pytest.mark.parametrize(
    ("initial_seconds", "text", "expected_seconds"),
    [(0.6, "750", 0.750), (1.2, "2", 2.0)],
)
def test_bare_duration_reuses_the_current_rendered_unit(
    qtbot,
    initial_seconds: float,
    text: str,
    expected_seconds: float,
) -> None:
    # Defaulting bare text to one fixed unit would reinterpret an authored value.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(committed.append)
    entry.set_exact_value(initial_seconds)

    entry.selectAll()
    qtbot.keyClicks(entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == [expected_seconds]


def test_duration_entry_accepts_surrounding_whitespace(qtbot) -> None:
    # Rejecting harmless outer whitespace would make pasted durations fail.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(committed.append)
    entry.set_exact_value(0.6)

    entry.selectAll()
    qtbot.keyClicks(entry, "  1.25 s  ")
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == [1.25]


@pytest.mark.parametrize("text", ["1 MS", "1 S", "1 db", "1 DB"])
def test_duration_units_are_case_sensitive(qtbot, text: str) -> None:
    # Case-folding units would silently accept spellings outside the closed grammar.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    committed: list[float] = []
    errors: list[str] = []
    entry.value_commit_requested.connect(committed.append)
    entry.validation_failed.connect(errors.append)
    entry.set_exact_value(0.6)

    entry.selectAll()
    qtbot.keyClicks(entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == []
    assert len(errors) == 1


def test_focus_out_preserves_duration_draft_without_emitting(qtbot) -> None:
    # Treating focus loss as commit or cancel would destroy an in-progress draft.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    other = QLineEdit()
    qtbot.addWidget(entry)
    qtbot.addWidget(other)
    committed: list[float] = []
    reverted: list[None] = []
    entry.value_commit_requested.connect(committed.append)
    entry.draft_reverted.connect(lambda: reverted.append(None))
    entry.set_exact_value(0.6)
    entry.show()
    other.show()
    entry.setFocus()
    entry.selectAll()
    qtbot.keyClicks(entry, "750")

    other.setFocus()
    qtbot.waitUntil(other.hasFocus)

    assert entry.text() == "750"
    assert committed == []
    assert reverted == []


@pytest.mark.parametrize(
    ("text", "expected_db"),
    [
        ("0", 0.0),
        ("+0.0 dB", 0.0),
        ("-6", -6.0),
        ("-6 dB", -6.0),
        ("-.125dB", -0.125),
    ],
)
def test_decibel_entry_accepts_signed_plain_decimals(
    qtbot,
    text: str,
    expected_db: float,
) -> None:
    # Breaking optional dB parsing or signed decimals would lose valid authored levels.
    entry = EnvelopeValueEntry("sustain_db", EnvelopeFieldKind.DECIBELS)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(committed.append)
    entry.set_exact_value(-6.0)

    entry.selectAll()
    qtbot.keyClicks(entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == [expected_db]


@pytest.mark.parametrize(
    "text",
    ["0", "-1 ms", "NaN", "Infinity", "1e-3 s", "1,5 s", "1 MS", ".1234567890 s"],
)
def test_duration_entry_rejects_closed_grammar(qtbot, text: str) -> None:
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    assert_invalid_commit(qtbot, entry, text, "attack_seconds")


@pytest.mark.parametrize("text", ["", "ms", "1 minute", "+ s"])
def test_duration_entry_rejects_missing_numbers_and_unsupported_units(qtbot, text: str) -> None:
    # Letting absent numbers or arbitrary suffixes through would violate the closed grammar.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    assert_invalid_commit(qtbot, entry, text, "attack_seconds")


@pytest.mark.parametrize(
    "text",
    ["0.1", "NaN", "1e2", "-6 db", "-6 DB", "-1,5", "Infinity", "-Infinity"],
)
def test_decibel_entry_rejects_invalid_values(qtbot, text: str) -> None:
    entry = EnvelopeValueEntry("sustain_db", EnvelopeFieldKind.DECIBELS)
    assert_invalid_commit(qtbot, entry, text, "sustain_db")


@pytest.mark.parametrize(
    ("kind", "value", "expected_text"),
    [
        (EnvelopeFieldKind.DURATION, 0.000000001, "0.000001 ms"),
        (EnvelopeFieldKind.DURATION, 0.999999999, "999.999999 ms"),
        (EnvelopeFieldKind.DURATION, 1.0, "1 s"),
        (EnvelopeFieldKind.DECIBELS, -6.125, "-6.125 dB"),
    ],
)
def test_programmatic_set_renders_six_places_without_emitting(
    qtbot,
    kind: EnvelopeFieldKind,
    value: float,
    expected_text: str,
) -> None:
    # Emitting on model render would create feedback, while excess digits would leak precision.
    entry = EnvelopeValueEntry("field", kind)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(committed.append)

    entry.set_exact_value(value)

    assert entry.text() == expected_text
    assert committed == []


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (EnvelopeFieldKind.DURATION, 0.600000000123),
        (EnvelopeFieldKind.DECIBELS, -6.123456789),
    ],
)
def test_unchanged_return_emits_exact_cached_float(
    qtbot,
    kind: EnvelopeFieldKind,
    value: float,
) -> None:
    # Reparsing the six-place display would drift exact model values.
    entry = EnvelopeValueEntry("field", kind)
    qtbot.addWidget(entry)
    committed: list[float] = []
    entry.value_commit_requested.connect(committed.append)
    entry.set_exact_value(value)

    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == [value]


def test_accepted_manual_proposal_refreshes_exact_cache(qtbot) -> None:
    # Caching rendered text instead of the accepted proposal would lose its ninth-place precision.
    entry = EnvelopeValueEntry("sustain_db", EnvelopeFieldKind.DECIBELS)
    qtbot.addWidget(entry)
    committed: list[float] = []

    def accept(value: float) -> None:
        committed.append(value)
        entry.accept_proposed_value(value)

    entry.value_commit_requested.connect(accept)
    entry.set_exact_value(-6.0)
    entry.selectAll()
    qtbot.keyClicks(entry, "-0.123456789")
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    assert entry.text() == "-0.123457 dB"

    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert committed == [-0.123456789, -0.123456789]


def test_rejected_manual_proposal_preserves_draft_and_prior_exact_cache(qtbot) -> None:
    # Rejection must not make a locally valid but parent-invalid value restorable.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    proposed: list[float] = []
    errors: list[str] = []

    def reject(value: float) -> None:
        proposed.append(value)
        entry.mark_commit_rejected("Envelope duration is outside the renderable frame.")

    entry.value_commit_requested.connect(reject)
    entry.validation_failed.connect(errors.append)
    exact_prior = 0.600000000123
    entry.set_exact_value(exact_prior)
    entry.selectAll()
    qtbot.keyClicks(entry, "0.000001 ms")
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert proposed == pytest.approx([0.000000001])
    assert errors == ["Envelope duration is outside the renderable frame."]
    assert entry.text() == "0.000001 ms"
    assert entry.property("validationState") == "error"

    qtbot.keyPress(entry, Qt.Key.Key_Escape)
    entry.value_commit_requested.disconnect(reject)
    entry.value_commit_requested.connect(proposed.append)
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert entry.text() == "600 ms"
    assert proposed == pytest.approx([0.000000001, exact_prior])


def test_escape_restores_cached_value_and_clears_only_validation_state(qtbot) -> None:
    # Escape must recover authored state without proposing a value or erasing unrelated properties.
    entry = EnvelopeValueEntry("sustain_db", EnvelopeFieldKind.DECIBELS)
    qtbot.addWidget(entry)
    committed: list[float] = []
    reverted: list[None] = []
    entry.value_commit_requested.connect(committed.append)
    entry.draft_reverted.connect(lambda: reverted.append(None))
    entry.setProperty("editorState", "connected")
    entry.set_exact_value(-6.0)
    entry.selectAll()
    qtbot.keyClicks(entry, "not a level")
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    assert entry.property("validationState") == "error"
    committed.clear()

    qtbot.keyPress(entry, Qt.Key.Key_Escape)

    assert entry.text() == "-6 dB"
    assert entry.property("validationState") is None
    assert entry.property("editorState") == "connected"
    assert reverted == [None]
    assert committed == []


@pytest.mark.parametrize(
    ("kind", "invalid"),
    [
        (EnvelopeFieldKind.DURATION, 0.0),
        (EnvelopeFieldKind.DURATION, -0.001),
        (EnvelopeFieldKind.DURATION, float("nan")),
        (EnvelopeFieldKind.DECIBELS, 0.1),
        (EnvelopeFieldKind.DECIBELS, float("inf")),
        (EnvelopeFieldKind.DECIBELS, True),
    ],
)
def test_programmatic_values_are_validated_before_caching(
    qtbot,
    kind: EnvelopeFieldKind,
    invalid: float,
) -> None:
    # Accepting invalid parent data would poison the value Escape restores.
    entry = EnvelopeValueEntry("field", kind)
    qtbot.addWidget(entry)
    valid = {
        EnvelopeFieldKind.DURATION: 0.6,
        EnvelopeFieldKind.DECIBELS: -6.0,
    }[kind]
    entry.set_exact_value(valid)
    expected_text = entry.text()

    with pytest.raises(ValueError):
        entry.set_exact_value(invalid)
    with pytest.raises(ValueError):
        entry.accept_proposed_value(invalid)

    entry.restore_last_valid()
    assert entry.text() == expected_text


def test_kind_property_is_read_only(qtbot) -> None:
    # Mutating parsing mode after caching would reinterpret the current rendered text.
    entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
    qtbot.addWidget(entry)
    assert entry.kind is EnvelopeFieldKind.DURATION
    with pytest.raises(AttributeError):
        entry.kind = EnvelopeFieldKind.DECIBELS  # type: ignore[misc]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.000000001, "0.000001 ms"),
        (0.125, "125 ms"),
        (0.999999999, "999.999999 ms"),
        (1.0, "1 s"),
        (1.23456789, "1.234568 s"),
    ],
)
def test_format_envelope_duration_uses_canonical_unit_and_precision(
    seconds: float,
    expected: str,
) -> None:
    # Changing graph-label formatting independently would disagree with entry rendering.
    assert format_envelope_duration(seconds) == expected
