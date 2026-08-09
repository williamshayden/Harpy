# Harpy Milestone B Graph-Native Envelope Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the permanent ADSR/curvature entry grid with direct, accessible envelope-graph controls and a contextual envelope-only Reset, then complete Milestone B acceptance.

**Architecture:** `EnvelopeEditor` remains the sole immutable-patch draft and validation owner. `EnvelopeGraph` composes four proposal-only inline stage controls plus the existing three proposal-only curve handles; a stage control creates one exact editor only for the duration of an explicit edit. The controller, audio boundary, synth renderer, codec, capture, frequency control, and deferred-apply semantics remain unchanged.

**Tech Stack:** Python 3.12, PySide6, NumPy, pytest/pytest-qt, Ruff, uv, native WSLg/PulseAudio acceptance.

## Global Constraints

- The approved specification is `docs/superpowers/specs/2026-08-08-harpy-milestone-b-envelope-authoring-design.md` at status `approved for implementation`.
- Harpy remains a native PySide6 application; add no browser surface, JUCE/C++, database, MIDI, Gym, model adapter, or new dependency.
- A held or releasing voice finishes with its starting patch. Valid authoring applies only when the existing controller admits the complete replacement, and the next Play auditions it.
- `EnvelopeEditor` owns authored/draft patch truth and emits only complete immutable `SynthPatch` candidates. Graph controls and handles emit proposals and never cache a second model value.
- A/D/R duration display and exact parsing use milliseconds below one second and seconds at or above one second. Sustain always displays its actual dB value and never the word `hold`.
- A/D/R scrub is logarithmic: 100 px up doubles and 100 px down halves. S scrub is linear at 0.1 dB/px. Shift scales only subsequent motion by 0.1.
- Stage controls use the platform drag threshold, at least 24 x 24 device-independent hit/focus targets, pointer capture after threshold, and one commit on release. Escape, capture loss, and window deactivation revert without a commit.
- Stage arrows commit immediately per accepted event, including auto-repeat: A/D/R use 1% or 0.1% with Shift; S uses 0.1 dB or 0.01 dB with Shift.
- Curve handles remain vertically constrained, use 0.01 or 0.001 Shift keyboard steps, accept auto-repeat, and expose a contextual readout only on hover/focus/drag.
- Exact editing is transient: double-click, Enter, or F2 opens one `QLineEdit`; Enter commits, Escape cancels and returns focus, focus loss cancels without reclaiming focus, and rejected Enter retains exact text and focus.
- `Reset` replaces only `EnvelopeConfig` with `EnvelopeConfig()`. It preserves selected frequency and every non-envelope patch field, while normal capture invalidation and deferred application still apply.
- Preserve strict v1-read/v2-write patch behavior, 48 kHz float32 mono render truth, generation-safe capture, release-tail watermark behavior, and all Milestone A/B numerical characterizations.
- Use TDD for every behavior change: record the focused RED command and expected failure before production edits, then run focused GREEN, full pytest, Ruff check, Ruff format check, and diff check before each task commit.
- Do not weaken or delete a test merely because its old widget name disappeared. Replace it with a graph-native assertion of the same product invariant.

## Baseline and execution topology

- Baseline code commit: `356ef4a feat: integrate envelope authoring workbench`.
- Approved design commits: `309c374` and `af8e25b`.
- This is the authoritative amendment for the envelope UI and paused acceptance work.
  The original Milestone B plan remains historical evidence for completed DSP, codec,
  controller, audio, knob, and initial workbench tasks; do not re-run its permanent-field
  editor or old Task 11 instructions.
- Tasks 1–4 change Python behavior. Task 5 changes documentation/evidence only after the behavior is accepted.
- Tasks are dependency ordered. Task 1 establishes the stage-control/transient-entry contract; Task 2 composes it additively into the graph; Task 3 atomically cuts over graph, editor, window, and tests; Task 4 hardens lifecycle/layout and is the clean implementation gate; Task 5 resumes honest native acceptance.
- For subagent-driven execution, use a fresh implementation agent per task and run specification-compliance and code-quality reviews after every task. The two review passes may run in parallel; resolve all Critical/Important findings before advancing.

## Final file structure

- Create `src/harpy/gui/envelope_stage_control.py`: direct stage-value painting, pointer/keyboard behavior, transient exact editor lifecycle, and stage-control accessibility.
- Modify `src/harpy/gui/envelope_entry.py`: strict transient parser/exact-value cache and focus lifecycle; remove dead curvature-entry behavior atomically with the editor cutover.
- Modify `src/harpy/gui/envelope_graph.py`: compose four stage controls, three curve handles, one contextual curve readout, unified field proposals, geometry, focus order, and interaction cancellation.
- Modify `src/harpy/gui/envelope_editor.py`: immutable patch/draft ownership, unified field validation, status/error state, contextual Reset, facts, and file-action signals; remove permanent field-grid state.
- Modify `src/harpy/gui/window.py`: Reset naming/shortcut and deactivation cancellation before force-stop.
- Modify `tests/gui/test_envelope_entry.py`, `tests/gui/test_envelope_graph.py`, `tests/gui/test_envelope_editor.py`, `tests/gui/test_window.py`, and `tests/gui/test_app.py`; create `tests/gui/test_envelope_stage_control.py`.
- Modify `README.md` and `docs/project-notebook.md`; create `docs/verification/2026-08-08-milestone-b-acceptance.md` only after clean acceptance evidence exists.

---

### Task 1: Build the direct stage-value control and prepare transient entry lifecycle

**Files:**

- Create: `src/harpy/gui/envelope_stage_control.py`
- Create: `tests/gui/test_envelope_stage_control.py`
- Modify: `src/harpy/gui/envelope_entry.py`
- Modify: `tests/gui/test_envelope_entry.py`

**Interfaces:**

- Consumes: `EnvelopeFieldKind.DURATION`, `EnvelopeFieldKind.DECIBELS`, `EnvelopeValueEntry`, `format_envelope_duration()`, `RenderConfig`, and an owner-supplied `Callable[[], float]`.
- Produces:

  ```python
  class EnvelopeValueStage(StrEnum):
      ATTACK = "attack"
      DECAY = "decay"
      SUSTAIN = "sustain"
      RELEASE = "release"

  class EnvelopeStageControl(QWidget):
      value_previewed = Signal(str, float)
      value_commit_requested = Signal(str, float)
      value_reverted = Signal(str)
      validation_failed = Signal(str, str)
      editing_changed = Signal(str, bool)
  ```

- `EnvelopeValueStage.field_name: str` and `.short_label: str` are closed mappings, not
  mutable attributes.
- `EnvelopeStageControl(stage, render, current_value, parent=None)` takes
  `EnvelopeValueStage`, `RenderConfig`, and `Callable[[], float]`.
- Its read-only properties are `current_value: float`, `display_text: str`, and
  `is_interacting: bool`.
- Its owner-response methods are `open_exact_editor()`, `accept_exact_value(value)`,
  `reject_exact_value(message)`, `mark_error(message)`, `clear_error()`, and
  `cancel_interaction(emit_revert=True, return_focus=False)`. The model owner calls
  `owner_value_changed(previous, current)` after accepting a changed immutable value so
  the control repaints and emits one accessibility value-change event without caching it.

- `EnvelopeValueEntry` retains `value_commit_requested(float)`, `validation_failed(str)`, and `draft_reverted()`, and adds `focus_cancel_requested()`. Its constructor remains `EnvelopeValueEntry(field_name, kind, parent=None)` so the stage control can create one lazily. Task 1 temporarily retains the current curvature mode because the still-live legacy editor consumes it; Task 3 removes that mode in the same commit that removes `curveEntry`.
- Add `set_editor_accessibility(name: str, description: str)`: it records the clean base
  description. Parser and owner rejection append the exact error to that description and
  emit one `QAccessibleAnnouncementEvent`; successful acceptance/Escape restores the base
  description.

- Add `format_envelope_decibels(value: float) -> str`, which normalizes signed zero and returns canonical text such as `-6 dB` or `0 dB` using the same six-decimal trimming as duration display.
- Add `format_envelope_curvature(value: float) -> str` as a display-only six-decimal
  trimmed formatter (`0.123456789 -> "0.123457"`, signed zero -> `"0"`). Removing curve
  text entry in Task 3 does not remove this contextual-readout formatter.

- [ ] **Step 1: Write the failing transient-entry tests**

  Add these exact contracts while temporarily retaining the curvature cases needed by the
  pre-cutover editor:

  ```python
  def test_focus_loss_requests_silent_transient_cancel(qtbot) -> None:
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
      other.setFocus()
      QApplication.processEvents()
      assert cancelled == [None]
      assert other.hasFocus()


  @pytest.mark.parametrize(
      ("value", "expected"),
      [(-6.0, "-6 dB"), (-0.0, "0 dB"), (-12.3456789, "-12.345679 dB")],
  )
  def test_decibel_formatter_is_canonical(value: float, expected: str) -> None:
      assert format_envelope_decibels(value) == expected


  def test_curvature_formatter_preserves_six_decimal_precision() -> None:
      assert format_envelope_curvature(0.123456789) == "0.123457"
      assert format_envelope_curvature(-0.0) == "0"
  ```

  Monkeypatch `QAccessible.updateAccessibility` and add parser-rejection tests that call
  `set_editor_accessibility("Attack duration", base_description)`, submit invalid text, and assert
  exactly one `QAccessibleAnnouncementEvent` for the entry with the exact validation
  message appended to its accessible description. Assert successful acceptance and
  Escape restore the base description without another announcement. Add the equivalent
  owner-rejection assertion to the stage-control RED before implementing that response.

  Retain all duration/dB/temporary-curvature parser, exact-cache, unchanged-commit,
  invalid-text, boolean, non-finite, sign, unit, and Escape tests in this task.

- [ ] **Step 2: Run the entry RED**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py -q
  ```

  Expected: FAIL because `focus_cancel_requested`, `format_envelope_decibels`, and
  `format_envelope_curvature` do not exist. Record the exact failure count in the task
  report.

- [ ] **Step 3: Implement the minimal transient-entry changes**

  In `envelope_entry.py`:

  ```python
  class EnvelopeFieldKind(StrEnum):
      DURATION = "duration"
      DECIBELS = "decibels"
      CURVATURE = "curvature"  # Removed atomically with curveEntry in Task 3.


  def format_envelope_decibels(value: float) -> str:
      numeric = 0.0 if value == 0.0 else value
      return f"{_plain_decimal(numeric)} dB"


  def format_envelope_curvature(value: float) -> str:
      numeric = 0.0 if value == 0.0 else value
      return _plain_decimal(numeric)


  class EnvelopeValueEntry(QLineEdit):
      focus_cancel_requested = Signal()

      def focusOutEvent(self, event: QFocusEvent) -> None:
          super().focusOutEvent(event)
          self.focus_cancel_requested.emit()
  ```

  Use `format_envelope_decibels()` for the dB render path. Retain the curvature branch
  until Task 3 so the full suite remains runnable after this task. Do not add show/hide,
  graph, patch, or controller ownership to this class. In `_set_error()`, update the
  accessible description and send one `QAccessibleAnnouncementEvent(self, message)`
  before emitting `validation_failed`; `_clear_error()` restores the recorded base
  description without an announcement.

- [ ] **Step 4: Write the failing stage-control construction and display tests**

  Create `tests/gui/test_envelope_stage_control.py` with a mutable owner loopback:

  ```python
  def make_control(qtbot, stage, value, *, sample_rate=48_000):
      owner = {stage.field_name: value}
      control = EnvelopeStageControl(
          stage,
          RenderConfig(sample_rate_hz=sample_rate),
          lambda: owner[stage.field_name],
      )

      def accept(field: str, proposed: float) -> None:
          previous = owner[field]
          owner[field] = proposed
          control.owner_value_changed(previous, proposed)

      control.value_previewed.connect(accept)
      control.value_commit_requested.connect(accept)
      control.resize(84, 28)
      qtbot.addWidget(control)
      control.show()
      QApplication.processEvents()
      return control, owner


  @pytest.mark.parametrize(
      ("stage", "value", "object_name", "text"),
      [
          (EnvelopeValueStage.ATTACK, 0.001, "attackValueControl", "A 1 ms"),
          (EnvelopeValueStage.DECAY, 0.600, "decayValueControl", "D 600 ms"),
          (EnvelopeValueStage.SUSTAIN, -6.0, "sustainValueControl", "S -6 dB"),
          (EnvelopeValueStage.RELEASE, 0.600, "releaseValueControl", "R 600 ms"),
      ],
  )
  def test_control_is_graph_native_and_queries_owner_truth(
      qtbot, stage, value, object_name, text
  ) -> None:
      control, owner = make_control(qtbot, stage, value)
      assert control.objectName() == object_name
      assert control.display_text == text
      owner[stage.field_name] = value * 2 if stage is not EnvelopeValueStage.SUSTAIN else -9.0
      control.update()
      assert control.current_value == owner[stage.field_name]
  ```

  Assert there is no `QLineEdit` child before exact editing begins and every control's
  `minimumSizeHint()` is at least `24 x 24`.

- [ ] **Step 5: Run the construction RED**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_stage_control.py -q
  ```

  Expected: collection FAIL with `ModuleNotFoundError` for
  `harpy.gui.envelope_stage_control`.

- [ ] **Step 6: Implement the stage enum, painting, and owner-query boundary**

  Create `envelope_stage_control.py`. Map fields exactly:

  ```python
  _FIELD_NAMES = {
      EnvelopeValueStage.ATTACK: "attack_seconds",
      EnvelopeValueStage.DECAY: "decay_seconds",
      EnvelopeValueStage.SUSTAIN: "sustain_db",
      EnvelopeValueStage.RELEASE: "release_seconds",
  }
  _SHORT_LABELS = {
      EnvelopeValueStage.ATTACK: "A",
      EnvelopeValueStage.DECAY: "D",
      EnvelopeValueStage.SUSTAIN: "S",
      EnvelopeValueStage.RELEASE: "R",
  }
  ```

  Query `current_value()` every time text, paint, keyboard, drag, or accessibility needs
  model truth. Paint restrained hover/focus/drag backgrounds and canonical text; do not
  store a durable value outside the active pointer origin. Set StrongFocus,
  `SizeVerCursor`, the exact object names above, and a minimum size of `24 x 24`.

- [ ] **Step 7: Write pointer-gesture RED tests**

  Use real `QMouseEvent` moves with global coordinates and the platform
  `QApplication.startDragDistance()`:

  - sub-threshold press/move/release emits no preview, commit, or revert;
  - 100 px upward A/D/R motion produces exactly `origin * 2`, and 100 px downward
    produces `origin / 2`;
  - 20 px upward S motion produces `origin + 2.0 dB`, clamped at `0 dB`;
  - a coarse move, zero-motion Shift transition, fine move, zero-motion Shift release,
    and second coarse move accumulate piecewise with no jump on either modifier event;
  - graph-style owner loopback and widget relayout between moves do not alter the value
    derived from global motion;
  - horizontal-only motion and arbitrary horizontal offsets never change a value;
  - extreme synthetic vertical coordinates cannot raise or emit a non-finite proposal;
    overflow saturates to the largest finite float so owner validation can reject it;
  - release outside the widget commits exactly the last preview once;
  - Escape, `QEvent.UngrabMouse`, and explicit `cancel_interaction()` after a preview emit
    exactly one revert and no commit;
  - a native double-click sequence never emits a scrub preview/commit; its exact-editor
    result is pinned in Step 10.

  Compare duration values with `pytest.approx(expected, rel=0, abs=1e-15)` and dB values with
  `abs=1e-12`.

- [ ] **Step 8: Run the gesture RED**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_stage_control.py -q
  ```

  Expected: display tests PASS and gesture tests FAIL because pointer state is not yet
  implemented.

- [ ] **Step 9: Implement thresholded piecewise pointer scrubbing**

  Keep `_press_global`, `_last_global_y`, `_origin_value`, `_accumulated_units`,
  `_dragging`, and `_had_preview` only while armed/dragging. For each move:

  ```python
  upward_pixels = self._last_global_y - event.globalPosition().y()
  scale = 0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
  self._last_global_y = event.globalPosition().y()
  if self._stage is EnvelopeValueStage.SUSTAIN:
      self._accumulated_units += upward_pixels * 0.1 * scale
      value = min(self._origin_value + self._accumulated_units, 0.0)
  else:
      self._accumulated_units += upward_pixels / 100.0 * scale
      value = max(
          _scaled_duration(self._origin_value, self._accumulated_units),
          1.0 / self._render.sample_rate_hz,
      )
  ```

  Implement the helper exactly as a finite boundary before the one-frame clamp:

  ```python
  def _scaled_duration(origin: float, units: float) -> float:
      try:
          value = origin * math.exp2(units)
      except OverflowError:
          return sys.float_info.max
      if math.isinf(value):
          return sys.float_info.max
      return 0.0 if value == 0.0 else value
  ```

  It never returns NaN because both interaction origin and accumulated pointer units are
  validated finite values; negative underflow becomes zero and is handled by the
  one-frame clamp.

  Accumulate motion before crossing threshold but emit the first preview only when the
  total press-to-current distance reaches the platform threshold. Call `grabMouse()` at
  that transition, clear interaction state before `releaseMouse()` on accepted release,
  and treat an unexpected `UngrabMouse` while still dragging as a revert. An unchanged or
  clamped proposal is a no-op.

- [ ] **Step 10: Write keyboard, exact-editor, and accessibility RED tests**

  For exact-editor cases, parent the control to a visible `240 x 120` host widget so the
  overlay can expand inside a graph-like parent. Assert with real events:

  - Up/Right and Down/Left commit the declared normal/fine duration or dB step, including
    accepted auto-repeat; clamped/no-change emits nothing;
  - Enter, F2, and double-click each open one `EnvelopeValueEntry` named
    `envelopeInlineEditor`;
  - duration entries have the stage-specific accessible name `Attack duration`, `Decay
    duration`, or `Release duration` and a clean description naming milliseconds/seconds;
    Sustain uses `Sustain level` and a clean description naming decibels; every clean
    description includes the vertical-drag and Enter/F2 exact-edit instructions;
  - invalid Enter retains text, focus, `validationState="error"`, and emits one
    field-named validation signal; owner rejection behaves identically;
  - parser and owner rejection append the exact error to that clean accessible
    description and emit one announcement, while acceptance/Escape restores the clean
    description without another announcement;
  - accepted Enter destroys/hides the transient entry and returns focus; Escape does the
    same without a commit; focus loss cancels without stealing the destination focus;
  - Space typed into the entry is text and does not reach the control;
  - each control exposes `QAccessible.Role.SpinBox`, exact name/value/unit, one-frame
    duration minimum or `0 dB` sustain maximum, fine minimum step, increase/decrease/focus
  actions, and value-change events after owner loopback.

  Pin duration arrows as `current * factor` for increase and `current / factor` for
  decrease, where `factor` is `1.01` normally and `1.001` with Shift. This makes opposite
  steps reversible instead of subtracting a percentage from the already changed value.

- [ ] **Step 11: Implement keyboard, transient editing, and accessibility**

  Lazy-create `EnvelopeValueEntry` with the stage label and duration/dB kind. Parent the
  overlay to the stage control's parent graph so it may expand to at least `96 px` while
  clamping its geometry inside that parent. Immediately call
  `entry.set_editor_accessibility(stage_specific_name, stage_specific_description)`
  before showing it: duration descriptions name milliseconds/seconds; Sustain names
  decibels; all descriptions include the vertical-drag and Enter/F2 exact-edit
  instructions. Connect its signals before showing it. The stage control remains the
  lifecycle owner and exposes only proposal/accept/reject methods to the graph/editor.

  Guard intentional exact-editor closure with `_closing_exact_editor`: set it before
  hide/delete/focus transfer, ignore the resulting `focus_cancel_requested`, then clear
  it after the queued deletion boundary. Emit `editing_changed(field, True)` once when an
  editor opens and `editing_changed(field, False)` once when it closes. Explicit
  Enter/Escape returns focus; focus-loss closure never does.

  Install a type-specific `QAccessibleWidget`/`QAccessibleValueInterface` factory for
  `EnvelopeStageControl`, using role `SpinBox`. Accessibility set/increase/decrease calls
  `value_commit_requested`; it never mutates local truth. Use one-frame minimum and no
  finite maximum for durations, no lower bound and `0.0` maximum for Sustain, and the
  declared fine steps. `owner_value_changed()` verifies the callback now returns the
  supplied current value, repaints, and emits `QAccessibleValueChangeEvent` only when
  previous and current differ.

- [ ] **Step 12: Verify Task 1 GREEN and commit**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py tests/gui/test_envelope_stage_control.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  ```

  Then commit only Task 1 files:

  ```bash
  git add src/harpy/gui/envelope_entry.py src/harpy/gui/envelope_stage_control.py tests/gui/test_envelope_entry.py tests/gui/test_envelope_stage_control.py
  git commit -m "feat: add graph-native envelope value controls"
  ```

---
### Task 2: Compose graph-native values, contextual curve readout, and unified field intents

**Files:**

- Modify: `src/harpy/gui/envelope_graph.py`
- Modify: `tests/gui/test_envelope_graph.py`

**Interfaces:**

- Consumes: Task 1 `EnvelopeStageControl`, `EnvelopeValueStage`, transient-entry
  owner-response methods, existing `CurveStage`, pure preview sampler, and
  `EnvelopeConfig`/`RenderConfig`.
- Produces this graph/editor boundary:

  ```python
  class EnvelopeGraph(QWidget):
      field_previewed = Signal(str, float)
      field_commit_requested = Signal(str, float)
      field_reverted = Signal(str)
      field_validation_failed = Signal(str, str)
      field_editing_changed = Signal(str, bool)
  ```

- Required methods are `set_envelope(envelope)`,
  `accept_exact_edit(field_name, value)`, `reject_exact_edit(field_name, message)`,
  `mark_field_error(field_name, message)`, `clear_field_error(field_name)`, and
  `cancel_interactions(emit_revert=True)`.

- In Task 2 the new `field_*` signals carry only A/D/S/R value-field intents. Temporarily
  retain `curve_previewed`, `curve_commit_requested`, `curve_reverted`, and
  `stage_selected` unchanged so the pre-cutover `EnvelopeEditor` and full suite remain
  green. Task 3 atomically maps curve intents to `attack_curve`, `decay_curve`, and
  `release_curve`, migrates all consumers, and removes the old signals.
- Rename `EnvelopeGraphGeometry.label_rects` to
  `stage_control_rects: Sequence[tuple[str, QRectF]]`. It is the single geometry truth
  for child placement and test inspection; painted x remains presentation-only.
- Keep `set_envelope()` as the only model-to-view synchronization method. Every child
  queries the graph's current immutable envelope; no child proposal mutates it.

- [ ] **Step 1: Replace the graph surface test with a failing graph-native contract**

  In `tests/gui/test_envelope_graph.py`, keep the pure fraction/preview/geometry tests and
  add:

  ```python
  @pytest.mark.parametrize(
      ("name", "text"),
      [
          ("attackValueControl", "A 125 ms"),
          ("decayValueControl", "D 400 ms"),
          ("sustainValueControl", "S -9 dB"),
          ("releaseValueControl", "R 850 ms"),
      ],
  )
  def test_graph_uses_real_inline_controls_and_actual_sustain(
      qtbot, name: str, text: str
  ) -> None:
      graph = make_graph(
          qtbot,
          EnvelopeConfig(
              attack_seconds=0.125,
              decay_seconds=0.400,
              sustain_db=-9.0,
              release_seconds=0.850,
          ),
      )
      control = graph.findChild(EnvelopeStageControl, name)
      assert control is not None
      assert control.display_text == text
      assert "hold" not in control.display_text.lower()
  ```

  Assert all four controls are StrongFocus, use `SizeVerCursor`, have non-empty tooltips,
  and occupy the corresponding `stage_control_rects` with at least 24 px height. Assert
  the graph contains zero idle `EnvelopeValueEntry` instances and contains one hidden,
  mouse-transparent `QLabel` named `curveValueReadout`.

- [ ] **Step 2: Run the graph-surface RED**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_graph.py -q
  ```

  Expected: FAIL because `EnvelopeGraph` still paints duration labels, reports `S hold`,
  and has no stage-control children or contextual readout.

- [ ] **Step 3: Compose stage controls and rename geometry**

  Construct one `EnvelopeStageControl` for each `EnvelopeValueStage` with a callback that
  reads the matching field from `self._envelope`. Forward all stage-control signals to the
  unified graph signals, but do not yet coordinate one control against another; that
  behavior receives its RED in Step 8 and its implementation in Step 9. Replace label
  painting with child placement from `stage_control_rects`; expand the label band from
  18 px to at least 24 px while keeping every rect inside `contents` at 240, 288, 360,
  and 512 px graph widths.

  In `set_envelope()`, compare previous/new A/D/S/R values and call the matching
  control's `owner_value_changed(previous, current)` after installing the new immutable
  envelope. Keep the existing equivalent curve-handle accessibility notification.

  Create `curveValueReadout` once, set
  `WA_TransparentForMouseEvents`, keep it hidden when no curve handle is contextual, and
  derive text only from `self._envelope` through Task 1
  `format_envelope_curvature()`, for example `Curve 0.123457`, `Curve -0.375`, and
  `Curve 0`. Never reduce the model value to three-decimal display precision.

- [ ] **Step 4: Write the failing value-field proposal and focus-order tests**

  Keep existing curve tests on their temporary stage-based curve signals. Retain their
  existing owner loopback and add a separate value-field owner loopback:

  ```python
  def accept_curve_proposals(graph: EnvelopeGraph) -> None:
      def accept(stage: str, value: float) -> None:
          graph.set_envelope(replace(graph.envelope, **{f"{stage}_curve": value}))

      graph.curve_previewed.connect(accept)
      graph.curve_commit_requested.connect(accept)


  def accept_value_field_proposals(graph: EnvelopeGraph) -> None:
      def accept(field_name: str, value: float) -> None:
          graph.set_envelope(replace(graph.envelope, **{field_name: value}))

      graph.field_previewed.connect(accept)
      graph.field_commit_requested.connect(accept)
  ```

  Assert Attack curve still emits `attack` through `curve_*`, Sustain value emits
  `sustain_db` through `field_*`, and Release value emits `release_seconds` through
  `field_*`. The unified `attack_curve` field name is introduced and asserted only in
  Task 3. With real Tab key events, assert focus order exactly:

  ```text
  attackValueControl
  attackCurveHandle
  decayValueControl
  decayCurveHandle
  sustainValueControl
  releaseValueControl
  releaseCurveHandle
  ```

  An ownerless proposal must leave `graph.envelope` and every accessible current value
  unchanged.

  Run `uv run pytest tests/gui/test_envelope_graph.py -q`. Expected: the surface slice is
  GREEN and value-field signal/focus-order assertions FAIL against the old curve-only
  graph.

- [ ] **Step 5: Implement value-field signals, owner responses, and focus order**

  `accept_exact_edit()` and `reject_exact_edit()` must resolve only A/D/S/R value fields
  to the owning stage control; curve fields never open a text editor. `mark_field_error()`
  and `clear_field_error()` target either the stage control or curve handle without
  storing a value. `cancel_interactions()` closes one transient entry and cancels every
  active value/curve pointer gesture.

  After child layout, call `QWidget.setTabOrder()` for each adjacent pair in the declared
  seven-control sequence. Task 2 retains and emits `stage_selected` for the still-live
  permanent curve entry; Task 3 removes both consumer and signal atomically.

- [ ] **Step 6: Write the failing curve-interaction and readout tests**

  Retain all curve mathematics, x-invariance, zero-span, and accessibility assertions,
  then harden interaction tests:

  - press/release below platform threshold is a no-op;
  - drag threshold begins preview, out-of-bounds release commits once, and intentional
    `releaseMouse()` cannot trigger a second revert;
  - Escape, unexpected `UngrabMouse`, and `cancel_interactions()` after preview revert
    once with no commit;
  - native double-click resets only that curve to zero with one commit and no first-click
    commit;
  - every accepted arrow/Home event, including auto-repeat, emits one immediate field
    commit through the temporary curve signal; clamped/unchanged values emit nothing;
  - `curveValueReadout` is visible with model-derived text only while the corresponding
    handle is hovered, focused, or dragged, and is hidden again afterward;
  - each curve handle is at least 24 x 24 and retains Slider role/value/actions.

  Run `uv run pytest tests/gui/test_envelope_graph.py -q`. Expected: unified graph tests
  remain GREEN and the new curve threshold/capture/auto-repeat/readout assertions FAIL.

- [ ] **Step 7: Implement curve threshold, cancellation, auto-repeat, and readout**

  Give `_CurveHandle` the same armed-versus-dragging threshold distinction and
  capture-loss guard as the value control. Clear its drag state before intentional
  `releaseMouse()`. Remove the current auto-repeat suppression branch; route every
  accepted repeat through `_request_curve()`, whose clamp/unchanged guard prevents
  redundant commits.

  Add one context signal emitted on hover enter/leave, focus in/out, drag start/end, and
  model update. `EnvelopeGraph` shows the shared readout only when the source handle's
  `underMouse()`, `hasFocus()`, or active-drag state is true. Do not create three readout
  labels or store a selected numeric copy.

- [ ] **Step 8: Write and run the transient-edit integration RED**

  Drive the real stage controls inside `EnvelopeGraph` and assert:

  - Enter, F2, and native double-click produce exactly one active
    `envelopeInlineEditor` overlay initialized from graph truth;
  - opening a second stage silently cancels the first before showing the second;
  - text editing emits `field_editing_changed(field, True)`; Escape/focus loss emits
    `field_editing_changed(field, False)` plus one `field_reverted(field)`;
  - parser rejection emits one `field_validation_failed(field, message)` and keeps the
    overlay text/focus;
  - `field_commit_requested(field, value)` leaves the overlay open until the fake owner
    calls `accept_exact_edit()` or `reject_exact_edit()`;
  - accept closes and explicitly returns focus; reject retains focus/text/error;
  - focus-loss cancellation preserves the new focus target;
  - a same-envelope `set_envelope()` does not overwrite active invalid text;
  - `cancel_interactions()` closes without a field commit and Space remains text while
    the overlay is open.

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_graph.py -q
  ```

  Expected: the earlier graph slices remain GREEN and the new cross-control coordination,
  unified forwarding, owner accept/reject, and same-envelope refresh assertions FAIL.

- [ ] **Step 9: Implement transient coordination at the graph boundary**

  Add only graph forwarding/target resolution needed for the Step 8 contracts; the stage
  control remains the transient editor's lifecycle owner. When a new control reports
  editing active, cancel the old control with `return_focus=False`; its internal close
  guard suppresses duplicate focus-out cancellation and leaves the new editor focused.
  Never accept or reject an exact proposal before the owning editor explicitly calls the
  corresponding graph method.

- [ ] **Step 10: Verify Task 2 GREEN and commit**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_stage_control.py tests/gui/test_envelope_graph.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  ```

  Require zero source/test matches for the deleted graph-only contract:

  ```bash
  ! rg -n 'label_rects|S  hold' src/harpy/gui/envelope_graph.py tests/gui/test_envelope_graph.py
  ```

  Commit:

  ```bash
  git add src/harpy/gui/envelope_graph.py tests/gui/test_envelope_graph.py
  git commit -m "feat: compose graph-native envelope controls"
  ```

---

### Task 3: Simplify EnvelopeEditor around unified graph fields and contextual Reset

**Files:**

- Modify: `src/harpy/gui/envelope_entry.py`
- Modify: `src/harpy/gui/envelope_graph.py`
- Modify: `src/harpy/gui/envelope_editor.py`
- Modify: `src/harpy/gui/window.py`
- Modify: `tests/gui/test_envelope_entry.py`
- Modify: `tests/gui/test_envelope_graph.py`
- Modify: `tests/gui/test_envelope_editor.py`
- Modify: `tests/gui/test_window.py`
- Modify: `tests/gui/test_app.py`

**Interfaces:**

- Consumes: Task 2 value-field `EnvelopeGraph` signals/owner responses plus its temporary
  curve/stage-selection signals,
  `EnvelopeConfig()`, immutable `SynthPatch`, `validate_renderable_patch()`, and
  `PatchApplyState`.
- Produces the retained public editor boundary:

  ```python
  class EnvelopeEditor(QFrame):
      patch_commit_requested = Signal(object)
      validation_failed = Signal(str)
      validation_cleared = Signal()
      load_requested = Signal()
      save_requested = Signal()
  ```

- Retained methods are
  `set_patch_state(patch, apply_state, discard_draft=False)`, `discard_draft()`, and
  `cancel_interactions()`.

- Internal graph handlers are field-name based:

  `_preview_field(field_name, value)`, `_commit_field(field_name, value)`,
  `_revert_field(field_name)`, `_set_field_editing(field_name, dirty)`, and
  `_reset_envelope()`.

- Delete `_NUMERIC_FIELDS`, `_entries`, `_curve_entry`, `_selected_curve_stage`, selected
  curve rendering/selection methods, `_reset_all_curves`, and `_format_curvature`.
- Atomically map curve proposals to the unified `field_*` signals; remove
  `curve_previewed`, `curve_commit_requested`, `curve_reverted`, and `stage_selected` only
  after `EnvelopeEditor` consumes the unified boundary.
- Remove `EnvelopeFieldKind.CURVATURE` and its parser/tests only after `curveEntry` is gone.
- Rename the window reset consumer/action to `resetEnvelopeButton` and
  `resetEnvelopeAction` in this same task; retain no compatibility object or action.

- [ ] **Step 1: Write the complete atomic cutover RED and migrate affected tests**

  Rewrite the first editor surface test to assert these exact children:

  ```text
  envelopeGraph
  attackValueControl
  decayValueControl
  sustainValueControl
  releaseValueControl
  attackCurveHandle
  decayCurveHandle
  releaseCurveHandle
  curveValueReadout
  patchStatusLabel
  resetEnvelopeButton
  envelopeFieldError
  oscillatorFact
  outputFact
  loadPatchButton
  savePatchButton
  ```

  Assert `resetEnvelopeButton.text() == "Reset"` and require no idle
  `EnvelopeValueEntry`. Explicitly assert these names are absent:

  ```text
  attackEntry
  decayEntry
  sustainEntry
  releaseEntry
  curveEntry
  resetCurvesButton
  ```

  In the same RED, assert `set(EnvelopeFieldKind)` is exactly DURATION/DECIBELS, graph
  curve interactions emit `attack_curve`/`decay_curve`/`release_curve` through
  `field_*`, old curve/stage-selection signals are absent, and `HarpyWindow` contains
  `resetEnvelopeAction` with `Ctrl+R` but no `resetCurvesAction`.

  Before any production edit, also write every unified preview/commit/revert assertion
  enumerated in Step 3, every status/error/accessibility assertion enumerated in Step 5,
  and every Reset assertion enumerated in Step 7. Complete the legacy-test migration
  specified in Step 9 now, including `test_window.py` and `test_app.py`. Those later
  sections are implementation requirements and audit checklists, not later opportunities
  to add the first failing test for behavior already implemented.

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py tests/gui/test_envelope_stage_control.py tests/gui/test_envelope_graph.py tests/gui/test_envelope_editor.py tests/gui/test_window.py tests/gui/test_app.py -q
  ```

  Expected: FAIL because the permanent grid, curvature entry/parser, old graph signals,
  old reset wiring, and old field/status/Reset behaviors still exist. Record the exact
  failure count before touching production.

- [ ] **Step 2: Remove the permanent grid and compose the contextual Reset**

  Treat Steps 2–8 as one atomic production edit after the complete Step 1 RED. Do not
  instantiate the editor or run an intermediate GREEN between these steps: temporary
  missing attributes are permitted only inside the untested edit, and every handler and
  connection must exist before Step 9 audits the result and Step 10 runs tests.

  Remove `QGridLayout`/entry construction and curve-selection state. Keep heading, patch
  status, graph, one compact Reset button, error label, oscillator/output facts, and
  Load/Save actions. Name the button `resetEnvelopeButton`; do not retain a compatibility
  alias. Let `HarpyWindow` own the inspector's fixed 288 px width rather than setting a
  conflicting fixed width inside the editor.

  Remove curvature parsing from `envelope_entry.py` and its now-superseded tests. In
  `envelope_graph.py`, map every curve handle signal through `_curve_field(stage_name)` to
  the unified `field_*` signals, then remove the temporary curve/stage-selection outward
  signals. In `window.py`, resolve `resetEnvelopeButton`, rename the action to
  `resetEnvelopeAction`, preserve `Ctrl+R`, and set tooltip
  `Reset envelope to defaults (Ctrl+R)`.

  Do not connect the graph to methods that have not been implemented. Complete all
  handlers in Steps 3–8, then make the six connections exactly once at the end of Step 8.

- [ ] **Step 3: Implement the unified preview/commit/revert contract**

  Satisfy the pre-written RED by driving graph signals and real controls for each of the
  seven fields through generalized handlers. The required assertions are:

  - previews modify only `_draft_envelope`/graph projection and emit no patch;
  - release or keyboard commit validates and emits exactly one complete `SynthPatch`;
  - exact and pointer commits for the same value produce equal immutable patches;
  - Attack/Decay/Sustain/Release and their three curves alter only their named field;
  - Escape/capture-loss revert restores the latest authored value for only that field;
  - owner acknowledgement updates canonical labels/handles without a feedback commit;
  - `set_patch_state()` with the same patch does not overwrite active invalid transient
    text, while `discard_draft=True` closes it and replaces every graph field.

  Use the real transient editor for one duration and Sustain case; do not recreate a
  permanent-entry helper in the tests.

- [ ] **Step 4: Complete one generalized immutable-patch path**

  `_preview_field()` first builds a complete candidate from
  `replace(self._draft_envelope, **{field_name: value})` and validates
  `replace(self._authored_patch, envelope=candidate_envelope)` with
  `validate_renderable_patch()`. Only a valid preview updates draft/graph; an invalid or
  overflow proposal marks that field, leaves the previous draft projection intact, and
  never emits. `_commit_field()` uses the same construction and validation:

  ```python
  candidate_envelope = replace(self._draft_envelope, **{field_name: value})
  candidate_patch = replace(self._authored_patch, envelope=candidate_envelope)
  validate_renderable_patch(candidate_patch, self._render)
  ```

  On rejection, call `graph.reject_exact_edit()` when that field owns the active exact
  editor; its existing entry validation signal is the one path back to the editor error
  surface. Otherwise call `graph.mark_field_error()` and publish the error once directly.
  Authored patch, awaiting acknowledgement, capture, and command stream remain unchanged.

  On success, update draft/graph, call `graph.accept_exact_edit()` **before** emitting the
  patch signal, clear only that field's error/dirty marker, establish
  `_awaiting_ack_patch`, and then emit `patch_commit_requested`. Preserve the existing
  synchronous acknowledgement guard so a controller callback cannot be overwritten by
  the emitter's `finally` block.

- [ ] **Step 5: Implement the pre-written status/error/accessibility contract**

  Satisfy the Step 1 RED for these exact states:

  - clean applied patch => `Active`;
  - local preview, typed text, or local validation error => `Editing`;
  - controller pending plus an invalid transient edit => status remains `Pending` while
    entry/error styling and exact invalid text remain visible;
  - clearing/reverting an editor error never hides a file/device error owned by the
    window;
  - rejected exact commit keeps entry focus, augments its accessible description with the
    field error, and produces one `QAccessibleAnnouncementEvent`;
  - status text/property changes emit one accessible state-change event only when the
    state actually changes.

  Use a narrow monkeypatch collector around `QAccessible.updateAccessibility` and assert
  event type/widget/message rather than testing only a local Python signal.

- [ ] **Step 6: Complete status/error ownership without duplicate announcements**

  Keep precedence:

  ```python
  if self._apply_state is PatchApplyState.PENDING:
      text, state = "Pending", "pending"
  elif self._has_local_draft():
      text, state = "Editing", "editing"
  else:
      text, state = "Active", "active"
  ```

  Parser rejection and active-entry composed rejection are already marked/announced by
  the transient entry; the editor mirrors their forwarded message into
  `envelopeFieldError` and outward `validation_failed` without announcing twice. A
  non-entry composed rejection marks the owning graph control, announces once there, and
  publishes the same message once. Track the last status state and emit a Qt accessible
  state-change event only on a real transition.

- [ ] **Step 7: Implement the pre-written Reset contract**

  Satisfy the Step 1 RED: with a non-default patch containing non-default ADSR/curves,
  oscillator, and output gain, click the real Reset button and require one emitted
  candidate equal to:

  ```python
  expected = replace(starting_patch, envelope=EnvelopeConfig())
  ```

  Add cases for:

  - selected frequency is absent from editor state and therefore cannot change;
  - clean authored defaults emit no patch;
  - invalid transient text at defaults is discarded and its editor error clears without
    a redundant patch;
  - active curve/value preview is reverted before Reset constructs the candidate, so no
    preview value leaks into defaults;
  - Pending Reset updates the latest authored candidate once and relies on controller
    coalescing; the editor does not send audio commands;
  - Reset clears only editor-originated errors and leaves external file/device error
    ownership to `HarpyWindow`.

- [ ] **Step 8: Complete atomic envelope-only Reset, cancellation, and signal wiring**

  `cancel_interactions()` delegates to `graph.cancel_interactions()` and leaves authored
  truth unchanged. `_reset_envelope()` performs this order:

  ```python
  self.cancel_interactions()
  default_envelope = EnvelopeConfig()
  candidate_patch = replace(self._authored_patch, envelope=default_envelope)
  try:
      validate_renderable_patch(candidate_patch, self._render)
  except ValueError as error:
      self._show_reset_error(f"Envelope: {str(error).rstrip('.')}.")
      return
  had_error = self._error_field is not None
  self._clear_all_error_state()
  if had_error:
      self.validation_cleared.emit()
  self._text_dirty_fields.clear()
  self._draft_envelope = default_envelope
  self._graph.set_envelope(default_envelope)
  if default_envelope == self._authored_patch.envelope:
      self._update_status()
      return
  self._emit_validated_candidate(candidate_patch)
  ```

  The complete-envelope helper validates `replace(self._authored_patch,
  envelope=default_envelope)`, preserving every other patch field. Emit
  `validation_cleared` only if an editor error actually existed. If composed validation
  rejects the default under a nonstandard `RenderConfig`, mark `resetEnvelopeButton`,
  prefix the validator's exact detail with `Envelope:` on the editor error surface, leave
  authored/draft truth unchanged, and emit no patch.

  Only after every referenced handler exists, connect the graph and Reset button exactly
  once:

  ```python
  self._graph.field_previewed.connect(self._preview_field)
  self._graph.field_commit_requested.connect(self._commit_field)
  self._graph.field_reverted.connect(self._revert_field)
  self._graph.field_editing_changed.connect(self._set_field_editing)
  self._graph.field_validation_failed.connect(self._field_parse_failed)
  self._reset_envelope_button.clicked.connect(self._reset_envelope)
  ```

- [ ] **Step 9: Audit the test migration completed before production edits**

  Confirm the Step 1 migration deleted only helpers/assertions tied to permanent entries
  or selected curve text and retained/adapted tests for:

  - same-patch refresh and acknowledgement re-entrancy;
  - programmatic v1/v2 patch synchronization;
  - pending/active status;
  - parser versus composed-render rejection;
  - graph preview/commit/revert ownership;
  - Load/Save signals and oscillator/output facts;
  - invalid draft discard and validation-cleared signaling.

  The already-migrated `tests/gui/test_window.py` and `tests/gui/test_app.py` must use the
  real transient path:

  ```python
  def open_exact_stage_editor(qtbot, window, control_name: str) -> EnvelopeValueEntry:
      control = editor_child(window, EnvelopeStageControl, control_name)
      control.setFocus(Qt.FocusReason.OtherFocusReason)
      qtbot.keyPress(control, Qt.Key.Key_F2)
      entry = window.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
      assert entry is not None and entry.isVisible() and entry.hasFocus()
      return entry


  def commit_exact_stage(qtbot, window, control_name: str, text: str) -> None:
      entry = open_exact_stage_editor(qtbot, window, control_name)
      entry.selectAll()
      qtbot.keyClicks(entry, text)
      qtbot.keyPress(entry, Qt.Key.Key_Return)
  ```

  Map Attack/Decay/Sustain/Release cases to their four control names. Replace selected
  `curveEntry` cases with real curve-handle keyboard/drag/readout assertions. Preserve
  every existing controller, capture, error-priority, dialog, shutdown, and layout
  assertion after changing only its UI driver. If this audit finds a missing invariant,
  restore its graph-native test and confirm it would fail against the Task 2 baseline
  before accepting the Task 3 GREEN gate.

- [ ] **Step 10: Verify Task 3 GREEN and commit**

  Run:

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py tests/gui/test_envelope_stage_control.py tests/gui/test_envelope_graph.py tests/gui/test_envelope_editor.py tests/gui/test_window.py tests/gui/test_app.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  ```

  Require zero source/test matches:

  ```bash
  ! rg -n 'attackEntry|decayEntry|sustainEntry|releaseEntry|curveEntry|resetCurvesButton|resetCurvesAction|Reset curves to linear|EnvelopeFieldKind\.CURVATURE|curve_previewed|curve_commit_requested|curve_reverted|stage_selected' src/harpy/gui tests/gui
  ```

  Commit:

  ```bash
  git add src/harpy/gui/envelope_entry.py src/harpy/gui/envelope_graph.py src/harpy/gui/envelope_editor.py src/harpy/gui/window.py tests/gui/test_envelope_entry.py tests/gui/test_envelope_graph.py tests/gui/test_envelope_editor.py tests/gui/test_window.py tests/gui/test_app.py
  git commit -m "feat: cut over graph-native envelope editor"
  ```

---

### Task 4: Harden end-to-end cancellation, deferred application, and layout

**Files:**

- Modify: `src/harpy/gui/window.py`
- Modify: `tests/gui/test_window.py`
- Modify: `tests/gui/test_app.py`

**Interfaces:**

- Consumes: the Task 3 final surface and `EnvelopeEditor.cancel_interactions()`, existing
  controller deferred-apply behavior, patch dialogs, capture views, and backend signals.
- Produces: cancellation-before-force-stop on real window deactivation plus complete
  end-to-end regression evidence for Space, refresh, Reset, Load, failure/recovery,
  shutdown, and layout.
- Does not redesign `WorkbenchController`, `QtAudioBackend`, `SynthEngine`, codec, or
  runtime composition. A genuine failure in those layers returns to its owning task with
  a focused regression rather than receiving a GUI workaround.

- [ ] **Step 1: Write the focused deactivation RED**

  Start and hold a voice through the real window, then start one accepted value or curve
  pointer preview. Record graph `field_reverted` and controller command events in one
  ordered list, then send `QEvent.WindowDeactivate`. Assert:

  ```text
  field_reverted
  submit:RESET
  ```

  The preview emits no patch commit, active transient editing closes, the Play button is
  raised, and a repeated deactivation emits neither a second revert nor RESET. In a fresh
  held-voice case, add the same assertion for an active exact editor with invalid text:
  it cancels before force-stop and does not commit. The held voice is the explicit state
  that makes `WorkbenchController.force_stop()` enqueue the one expected RESET; preview
  state alone is not treated as audio activity.

- [ ] **Step 2: Run the deactivation RED**

  Run:

  ```bash
  uv run pytest tests/gui/test_window.py -k 'deactivation and envelope' -q
  ```

  Expected: FAIL because Task 3's window force-stops without first cancelling the graph
  interaction.

- [ ] **Step 3: Implement cancellation before window force-stop**

  In the non-dialog `WindowDeactivate` path only:

  ```python
  self.envelope_editor.cancel_interactions()
  self.handle_force_stop()
  ```

  Do not put cancellation inside generic backend `handle_force_stop()`: editing must
  remain available through audio failure. Dialog chooser deactivation remains exempt, and
  shutdown continues to discard the complete editor draft before force-stop.

- [ ] **Step 4: Add end-to-end graph-native characterization coverage**

  Through the real window/editor/controller graph, retain or add assertions that:

  - Space on focused stage controls/curve handles starts and releases Play, while Space in
    the transient `QLineEdit` remains text and sends no note command;
  - the 34 ms refresh path preserves active invalid text, focus, error styling, and
    Pending status;
  - explicit Enter/Escape returns focus, while clicking Clear or Load cancels without
    reclaiming the destination focus;
  - idle exact edit, value scrub, curve drag, and Reset each send one `REPLACE_PATCH`,
    clear retained capture, preserve selected frequency, and leave gate/voice idle;
  - held/releasing edit or Reset sends no immediate replacement, retains the old release
    capture, becomes Pending, and applies once after matching drained idle;
  - multiple value/curve/Reset intents coalesce to the final complete patch, stale idle is
    rejected, and next Play uses only that patch;
  - v1/v2 Load closes invalid transient state, refreshes labels/handles, and preserves
    frequency; invalid exact/JSON input leaves all semantic state unchanged;
  - editing while unavailable updates the existing backend cache and recovery uses the
    latest complete patch;
  - close/quit preserves RESET-or-pending-replacement before backend shutdown.

  These are characterization tests for already-approved core/controller behavior and may
  begin GREEN. If one fails, isolate the failure to its owning layer before changing
  production. Never deliver source idle synchronously while holding the source I/O lock;
  queue and deliver through the existing signal boundary.

- [ ] **Step 5: Re-run native-size/layout containment**

  At 1280 x 720 and 1024 x 640, assert:

  - `EnvelopeEditor.width() == 288`;
  - waveform and spectrum data canvases remain at least 320 px high;
  - four stage controls, seven focus targets, contextual readout, error label, Reset,
    facts, Load, and Save remain within inspector contents;
  - no horizontal scrollbar appears;
  - each stage's transient editor overlay remains fully inside graph bounds at minimum
    size;
  - healthy and error-banner layouts satisfy the same constraints.

- [ ] **Step 6: Verify Task 4 GREEN and commit**

  Run:

  ```bash
  uv run pytest tests/gui/test_window.py tests/gui/test_app.py tests/gui/test_audio_backend.py tests/gui/test_workbench_controller.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  ```

  Require zero source/test matches:

  ```bash
  ! rg -n 'attackEntry|decayEntry|sustainEntry|releaseEntry|curveEntry|resetCurvesButton|resetCurvesAction|Reset curves to linear|S  hold' src/harpy tests
  ```

  Commit:

  ```bash
  git add src/harpy/gui/window.py tests/gui/test_window.py tests/gui/test_app.py
  git commit -m "test: harden graph-native envelope lifecycle"
  ```

---

### Task 5: Verify and document the completed graph-native Milestone B

**Files:**

- Modify: `README.md`
- Modify: `docs/project-notebook.md`
- Create: `docs/verification/2026-08-08-milestone-b-acceptance.md`
- Create only when genuinely captured: `docs/verification/assets/milestone-b-idle.png`
- Create only when genuinely captured: `docs/verification/assets/milestone-b-editing.png`
- Create only when genuinely captured: `docs/verification/assets/milestone-b-pending.png`
- Create only when genuinely captured: `docs/verification/assets/milestone-b-live.png`

**Interfaces:**

- Consumes: the complete integrated Milestone B branch and directly observed automated,
  WSLg, PulseAudio, visual, and human evidence.
- Produces: accurate current-capability documentation and a reproducible acceptance
  record. It changes no Python behavior.

- [ ] **Step 1: Run the clean automated gates from the feature worktree**

  Record date, command, exit code, and exact result for:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run python -c "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.gui.envelope_stage_control; import harpy.synth.curves, harpy.synth.engine, harpy.synth.patch_json"
  git diff --check origin/main...HEAD
  find src/harpy -type f -name '*.py' -print0 | xargs -0 wc -l
  ```

  The import smoke must exit zero with zero stdout and zero stderr. If a gate fails, stop
  documentation, add a focused regression to the owning task, fix it under TDD, re-review
  that task, and restart this step from a clean worktree.

- [ ] **Step 2: Run structural contract searches**

  Require zero output:

  ```bash
  ! rg -n 'attackEntry|decayEntry|sustainEntry|releaseEntry|curveEntry|resetCurvesButton|resetCurvesAction|Reset curves to linear|S  hold|EnvelopeFieldKind\.CURVATURE' src/harpy tests
  ! rg -n 'LinearEnvelope|patchFacts|_patch_value_labels|_set_patch_facts' src/harpy
  ! rg -n 'Gymnasium|gymnasium|reward|actor adapter|model adapter' src/harpy
  ! rg -n 'from PySide6|import PySide6|pyqtgraph' src/harpy/synth src/harpy/analysis.py src/harpy/capture.py src/harpy/playback.py src/harpy/tuning.py
  ```

  Inspect every production module's line count and responsibility. The new stage-control
  module owns interaction/accessibility only; graph owns composition/projection; editor
  owns patch draft/validation. Flag duplicate patch truth, dead compatibility code, or a
  second curve formula. Do not split `qt_audio.py` merely because it is the largest
  platform adapter.

- [ ] **Step 3: Launch the exact native WSLg application and verify its audio route**

  Launch:

  ```bash
  XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
  ```

  Preserve inherited `PULSE_SERVER`. Once Play creates the stream, resolve the sole
  Pulse input with `application.name == "harpy"` and prove its process belongs to this
  worktree:

  ```bash
  harpy_audio_pid="$(
    pactl --format=json list sink-inputs |
      uv run python -c 'import json, sys; items = json.load(sys.stdin); matches = [item for item in items if item.get("properties", {}).get("application.name") == "harpy"]; assert len(matches) == 1; print(matches[0]["properties"]["application.process.id"])'
  )"
  test "$(readlink -f "/proc/$harpy_audio_pid/cwd")" = "$(pwd -P)"
  tr '\0' ' ' <"/proc/$harpy_audio_pid/cmdline"
  ```

  Verify 48 kHz float32 stereo device output belongs to the visible process. Close only
  the PID/window proven above; use no broad process kill.

- [ ] **Step 4: Exercise graph-native envelope and deferred-apply acceptance**

  Observe and record:

  - idle shows `A 1 ms`, `D 600 ms`, `S -6 dB`, `R 600 ms`, no `hold`, three linear
    handles, no permanent ADSR/curve text boxes, and a contextual Reset;
  - vertical stage scrubbing follows the declared duration/dB scale, dynamic Shift does
    not jump, arrows/auto-repeat work, and exact editing exists only transiently;
  - curve readout appears only on handle hover/focus/drag;
  - v1 default Load refreshes inline values/handles and Save As emits canonical v2;
  - a held note and its complete release tail remain objectively unchanged while edits
    become Pending; multiple edits coalesce; only the final patch is heard/rendered on
    the next Play;
  - Reset restores exactly `EnvelopeConfig()` while preserving selected frequency,
    oscillator, and output gain;
  - invalid exact/JSON edits are field-specific and atomic;
  - retained capture survives the old release and clears only when pending replacement is
    admitted;
  - Clear, Load, simulated device failure/recovery, deactivation, and close leave no stuck
    voice or transient editor.

  Use generated/captured samples, command order, generation, and PCM watermark evidence
  for objective claims. Label subjective hearing only as a human observation.

- [ ] **Step 5: Exercise knob, accessibility, and native layout acceptance**

  Retain the completed knob checks: continuous vertical drag, dynamic Shift fine mode,
  wheel/arrows, C3 reset, clamping, live/release retune without retrigger, pointer,
  landmarks, cursor, focus, and accessible Dial semantics.

  For the envelope, verify seven-control Tab order, 24 px targets, adjustable roles and
  values, contextual readout, exact editor name/unit/error announcement, explicit versus
  focus-loss focus behavior, Space routing, and Reset shortcut.

  Visually inspect 1280 x 720 and 1024 x 640 at nominal Windows 125% and 100% scaling.
  Record host scale before each check:

  ```bash
  powershell.exe -NoProfile -Command "Get-ItemProperty 'HKCU:\Control Panel\Desktop\WindowMetrics' -Name AppliedDPI"
  ```

  If authority/tooling cannot change scale or capture a complete frame, record the exact
  check as pending rather than inferring a pass.

- [ ] **Step 6: Capture screenshots only when the evidence qualifies**

  Capture complete native outer frames with no private desktop data for idle, editing,
  pending, and live states at the four declared asset paths. Editing must show an inline
  value interaction, not the deleted field grid. Pending must show the old voice/capture
  retained while the newest authored patch waits. If complete-frame capture is
  unavailable, create no placeholder files and mark screenshots pending.

- [ ] **Step 7: Update README and project notebook from observed facts**

  README current capabilities must say:

  - Milestone B now provides graph-native A/D/S/R scrubbing, transient exact editing,
    constrained Attack/Decay/Release curve handles, contextual Reset, and the refined
    frequency knob;
  - Harpy reads strict schema-v1 linear patches and strict schema-v2 curve-enabled
    patches, and writes canonical schema v2;
  - patch JSON contains oscillator, envelope/curves, and output gain, while selected or
    played frequency remains performance state outside JSON;
  - only the sine-only Gym remains Milestone C roadmap work;
  - native run and exact pytest/Ruff commands remain reproducible.

  Link the approved design, this implementation plan, and the acceptance record. In
  `docs/project-notebook.md`, add a dated Milestone B record explaining that patch
  authoring creates reproducible sources but the first retuning actor still manipulates
  rendered audio rather than controlling the synth.

- [ ] **Step 8: Write the evidence record with honest status**

  Use exactly:

  ```markdown
  # Milestone B acceptance evidence
  ## Acceptance status
  ## Automated evidence
  ## Structural evidence
  ## Native WSLg observations
  ## Screenshot evidence
  ## Pending manual and visual checks
  ```

  Record exact commit range and distinguish automated proof, native observation, human
  judgment, and pending checks. Never claim audibility, 100% scaling, complete-frame
  screenshots, or real physical device loss unless directly observed.

- [ ] **Step 9: Verify documentation, commit, and stop at the Milestone C boundary**

  Run:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git status --short
  ```

  Stage only files that exist:

  ```bash
  git add README.md docs/project-notebook.md docs/verification/2026-08-08-milestone-b-acceptance.md
  test ! -d docs/verification/assets || git add docs/verification/assets
  git commit -m "docs: record graph-native milestone B acceptance"
  ```

  Require a clean worktree and `git diff --check origin/main...HEAD`. Stop after Milestone
  B. Do not begin Gym code; Milestone C starts with a new brainstorming/specification pass
  for actions, observations, rewards, leakage controls, model adapters, and benchmark
  protocol.

---

## Completion conditions

Milestone B is complete only when:

1. every task has focused RED/GREEN evidence, full pytest, Ruff lint, Ruff format, diff
   checks, a scoped commit, and clean specification/code-quality reviews;
2. no permanent ADSR or curvature text-entry grid remains;
3. A/D/S/R values are directly scrubbed and transiently edited on the graph, and Sustain
   always reports its actual dB level;
4. curve handles and stage controls remain proposal-only projections of one immutable
   editor-owned patch draft;
5. pointer, keyboard, exact-entry, focus, accessibility, cancellation, error, and Reset
   contracts match the approved specification;
6. Reset restores only the complete default envelope and follows existing capture and
   deferred-apply semantics;
7. current held/releasing audio is unchanged and untruncated, pending intent coalesces,
   and the next Play auditions only the admitted final patch;
8. strict v1/v2 codec, deterministic DSP, controller, audio recovery, shutdown, knob,
   measurement, and native layout behavior remains green;
9. documentation records only evidence actually observed and leaves the sine-only Gym in
   Milestone C.
