# Harpy Milestone B Envelope Authoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic editable ADSR curves, strict patch-schema v2, deferred patch application after release, a compact native envelope inspector, and a professional frequency-knob presentation without beginning the reinforcement-learning Gym.

**Architecture:** Extend the immutable patch and NumPy renderer with one shared constrained-quadratic curve implementation, then preserve the controller/audio-thread boundary by committing only complete patches and deferring active-voice edits until a drained release watermark reports true idle. Build numeric fields, the accessible envelope graph, and the inspector as focused native components; `HarpyWindow` remains a client of the controller, while the DSP and codec remain Qt-free.

**Tech Stack:** Python 3.12, NumPy 2.x, PySide6/Qt Multimedia 6.x, pyqtgraph, uv, pytest, pytest-qt, Ruff.

## Global Constraints

- Use `/home/haydenw/Projects/Harpy/.worktrees/native-sine-lab` on
  `william/milestone-b-envelope-authoring` as the integration worktree. Parallel workers
  may write only in the isolated task worktrees named in the execution topology below;
  never let two agents edit one worktree concurrently.
- Treat `docs/superpowers/specs/2026-08-08-harpy-milestone-b-envelope-authoring-design.md` as the approved contract. Do not reopen its product decisions during implementation.
- Milestone B ends at patch/envelope authoring and knob refinement. Do not add Gymnasium, actor observations, rewards, model adapters, training, benchmark generation, other oscillators, polyphony, MIDI input, imported-audio editing, browser code, JUCE, or C++.
- Preserve `SynthPatch` as immutable sound configuration with no played frequency, gate, capture, file path, device, window, or schema-version state.
- Preserve the public `SynthEngine(render, patch)` API and its `note_on`, `retune`, `note_off`, `replace_patch`, `render`, and `reset` operations.
- Keep the renderer mono `float32`, phase-continuous across block partitions, and broad in frequency: every finite positive hertz value strictly below Nyquist. C2-C4 remains a GUI-only range.
- Curvature is one finite number per Attack, Decay, and Release in `[-1.0, +1.0]`; `0.0` must reproduce the Milestone A linear sample sequence exactly. Sustain remains flat.
- Use the approved constrained quadratic with midpoint control x-coordinate `0.5`; preview and DSP must call one shared implementation.
- Read strict patch-schema v1 and strict patch-schema v2. Save canonical schema v2 only. Do not keep the v1 curve string in the runtime model or add a compatibility alias.
- An editor commit always carries one complete immutable patch. Never send per-parameter audio commands.
- A held or releasing voice finishes with its starting patch. Coalesce active-voice edits in memory, apply only after the release-bearing audio block has left replaceable staging, and audition on the next Play.
- Keep Qt and pyqtgraph out of `harpy.synth`, `harpy.analysis`, `harpy.capture`, `harpy.playback`, and `harpy.tuning`.
- Keep one Qt audio adapter in `src/harpy/gui/qt_audio.py`; do not introduce a second backend or cosmetically split the platform adapter.
- Keep the minimum client size `1024 x 640`, default size `1280 x 720`, envelope inspector width `288 px`, dial size `88 x 88 px`, and both plot ViewBoxes at least `320 px` high.
- Tests remain part of the public research artifact. Do not delete a test unless the same commit replaces an explicitly superseded assertion.
- Use `apply_patch` for every source, test, and documentation edit. Preserve unrelated user changes.
- Every task follows RED -> GREEN -> focused verification -> full verification -> commit. Before every task commit run:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  ```

- A task is complete only when every command above exits zero and `git status --short` contains exactly that task's intended files before commit.

### Safe parallel execution topology

Use `superpowers:using-git-worktrees` before creating these lanes. Before Task 1 begins,
require the integration worktree to be clean, then create the two independent branches
from its exact plan-commit baseline:

```bash
git worktree add \
  -b william/milestone-b-envelope-entry \
  /home/haydenw/Projects/Harpy/.worktrees/milestone-b-entry \
  william/milestone-b-envelope-authoring
git worktree add \
  -b william/milestone-b-frequency-knob \
  /home/haydenw/Projects/Harpy/.worktrees/milestone-b-knob \
  william/milestone-b-envelope-authoring
```

Run these lanes concurrently, with one worker per worktree:

```text
integration lane: Task 1 -> Task 2 -> Task 3
entry lane:                         Task 6
knob lane:                          Task 9
```

After Tasks 3, 6, and 9 have independently passed their full gates and committed, the
integration owner—not a lane worker—cherry-picks the two disjoint component commits:

```bash
git cherry-pick william/milestone-b-envelope-entry
git cherry-pick william/milestone-b-frequency-knob
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Then create the graph lane from that tested integration tip:

```bash
git worktree add \
  -b william/milestone-b-envelope-graph \
  /home/haydenw/Projects/Harpy/.worktrees/milestone-b-graph \
  william/milestone-b-envelope-authoring
```

Run Tasks 4 -> 5 in the integration lane while the graph lane runs Task 7. When both
lanes are green and committed, cherry-pick the graph branch into the integration lane and
rerun the four gates. Task 8 then runs on the integrated Tasks 3, 5, 6, and 7; Tasks 10
and 11 remain serial integration tasks:

```bash
git cherry-pick william/milestone-b-envelope-graph
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Do not stage, commit, rebase, or cherry-pick from a worktree while another agent owns
that same worktree. Keep lane worktrees until their commits have been integrated and
verified; cleanup belongs to the final branch-finishing workflow.

### Implementation decisions fixed by this plan

- A same-value authored commit is a no-op. If a pending draft is edited back to the
  privately tracked applied patch, Pending is cancelled and no replacement is sent.
- Clear remains available while Pending. It intentionally clears measurement, advances
  the active generation through the existing alias path, and leaves pending patch intent
  intact.
- Returning encoded bytes from `SynthAudioSource.readData()` is the observable source
  handoff boundary. The source short-returns at the release watermark.
- Preview stages use `129` endpoint-inclusive samples each. Attack, Decay, and Release
  are separate arrays; the graph draws the Sustain shelf and joins stage endpoints once.
- Duration fields accept surrounding whitespace and case-sensitive `ms` or `s`; Sustain
  accepts case-sensitive `dB`; curvature has no suffix. Return commits, Escape reverts,
  and focus-out never commits.
- The selected curvature value is editable. Each non-autorepeat arrow/Home key press is
  one complete patch commit; Shift changes the arrow increment from `0.01` to `0.001`.
- Status priority is `Pending`, then `Editing`, then `Active`, so an unauditioned authored
  patch is never visually masked by a newer incomplete local draft.
- A zero-amplitude-span handle is painted at its stage midpoint x and common endpoint y.
  Mouse drag is a no-op for that handle; numeric, keyboard, and accessibility editing
  remain available.
- Every accepted numeric zero is normalized to positive `0.0` in the immutable model, so
  canonical v2 cannot emit a second `-0.0` representation of a linear curve, zero-decibel
  Sustain, or zero-decibel output gain.
- The in-process command sink must accept every validated non-`NOTE_ON` command, including
  `REPLACE_PATCH` while unavailable. Unexpected command-sink exceptions are fatal
  programming/platform errors; the controller does not invent a partial rollback over an
  already-advanced capture generation.

---

## File and responsibility map

### Core files retained and modified

- `src/harpy/analysis.py` — composed analysis validation; only Task 1 changes its overflowing-duration error boundary.
- `src/harpy/synth/models.py` — immutable render/oscillator/envelope/patch models and validation.
- `src/harpy/synth/curves.py` — new Qt-free constrained-quadratic evaluation and nominal preview sampling.
- `src/harpy/synth/envelope.py` — private performance-state ADSR renderer; renamed from linear-only semantics.
- `src/harpy/synth/engine.py` — authoritative oscillator/envelope composition; public methods remain unchanged.
- `src/harpy/synth/patch_json.py` — strict per-version decoding and canonical atomic v2 serialization.
- `src/harpy/gui/workbench_controller.py` — latest-authored versus applied/pending patch state and semantic commands.
- `src/harpy/gui/qt_audio.py` — release-block staging watermark plus existing device/cache lifecycle.

### New native authoring files

- `src/harpy/gui/envelope_entry.py` — exact cached duration, dB, and curvature text entry.
- `src/harpy/gui/envelope_graph.py` — transformed-time preview, three accessible constrained handles, and graph interaction signals.
- `src/harpy/gui/envelope_editor.py` — inspector composition, local draft ownership, complete-patch commit signals, status, and patch actions.

### Native composition files retained and modified

- `src/harpy/gui/frequency_knob.py` — existing tuning mechanics plus the approved pro-audio visual shell and discoverability states.
- `src/harpy/gui/window.py` — transport/measurement/inspector composition and controller routing; no draft mathematics.
- `src/harpy/gui/app.py` — passes stable render configuration into the native window; runtime ownership stays unchanged.

### Test and documentation files

- Add focused counterparts `tests/synth/test_curves.py`, `tests/gui/test_envelope_entry.py`, `tests/gui/test_envelope_graph.py`, and `tests/gui/test_envelope_editor.py`.
- Extend existing model, envelope, engine, codec, controller, audio-backend, knob, window, app, config, and import tests in the same task as their production change.
- Update `README.md` only after the complete feature is integrated.
- Create `docs/verification/2026-08-08-milestone-b-acceptance.md` from observed results; never pre-fill unperformed native checks as passing.

---

### Task 1: Close the overflowing analysis-window validation edge

**Files:**

- Modify: `src/harpy/analysis.py:57-65`
- Modify: `tests/test_analysis.py`

**Interfaces:**

- Consumes: `validate_analysis_config(config: AnalysisConfig, sample_rate_hz: int) -> None`.
- Produces: the same API, with every invalid waveform-duration product reported as `ValueError` rather than leaking `OverflowError`.

- [ ] **Step 1: Add the focused failing regression**

  Add this test beside the existing waveform-frame validation cases:

  ```python
  def test_validation_rejects_waveform_duration_whose_frame_product_overflows() -> None:
      config = AnalysisConfig(waveform_window_seconds=1e308)

      with pytest.raises(ValueError, match="waveform_window_seconds"):
          validate_analysis_config(config, 48_000)
  ```

- [ ] **Step 2: Run the regression and verify RED**

  ```bash
  uv run pytest tests/test_analysis.py::test_validation_rejects_waveform_duration_whose_frame_product_overflows -q
  ```

  Expected: FAIL because `_waveform_frames()` raises `OverflowError` from `math.floor(inf)`.

- [ ] **Step 3: Validate the product before rounding**

  In `_waveform_frames`, calculate and check the product explicitly:

  ```python
  frame_count = duration * sample_rate_hz
  if not math.isfinite(frame_count):
      raise ValueError("waveform_window_seconds must produce 1..fft_frames")
  return math.floor(frame_count + 0.5)
  ```

  Preserve all ordinary half-up rounding and existing error copy.

- [ ] **Step 4: Run focused analysis tests and verify GREEN**

  ```bash
  uv run pytest tests/test_analysis.py -q
  ```

  Expected: every analysis test passes, including the 9,009-case peak sweep.

- [ ] **Step 5: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/analysis.py tests/test_analysis.py
  git commit -m "fix: reject overflowing analysis windows"
  ```

---

### Task 2: Introduce the shared constrained-quadratic evaluator

**Files:**

- Create: `src/harpy/synth/curves.py`
- Create: `tests/synth/test_curves.py`
- Modify: `tests/test_imports.py`

**Interfaces:**

- Consumes: plain finite core numerical values and NumPy `float64` arrays.
- Produces:

  ```python
  def quadratic_control_level(
      start_level: float,
      end_level: float,
      curvature: float,
  ) -> float:


  def curvature_from_control_level(
      start_level: float,
      end_level: float,
      control_level: float,
  ) -> float:


  def evaluate_quadratic_segment(
      start_level: float,
      end_level: float,
      curvature: float,
      position: float | np.ndarray,
  ) -> float | np.ndarray:
  ```

  Scalar input returns `float`; a one-dimensional input returns an owned contiguous read-only `float64` array with the same shape.

- [ ] **Step 1: Write RED tests for exact scalar semantics**

  Create `tests/synth/test_curves.py` with exact endpoint, midpoint, and sign assertions:

  ```python
  import numpy as np
  import pytest

  from harpy.synth.curves import (
      curvature_from_control_level,
      evaluate_quadratic_segment,
      quadratic_control_level,
  )


  @pytest.mark.parametrize("curvature", [-1.0, -0.25, 0.0, 0.25, 1.0])
  @pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
  def test_quadratic_segment_has_exact_endpoints(
      curvature: float,
      start: float,
      end: float,
  ) -> None:
      assert evaluate_quadratic_segment(start, end, curvature, 0.0) == start
      assert evaluate_quadratic_segment(start, end, curvature, 1.0) == end


  def test_zero_curvature_is_linear_and_sign_reaches_target_early() -> None:
      assert evaluate_quadratic_segment(0.0, 1.0, 0.0, 0.5) == 0.5
      assert evaluate_quadratic_segment(0.0, 1.0, 1.0, 0.5) == 0.75
      assert evaluate_quadratic_segment(0.0, 1.0, -1.0, 0.5) == 0.25
      assert evaluate_quadratic_segment(1.0, 0.0, 1.0, 0.5) == 0.25
      assert evaluate_quadratic_segment(1.0, 0.0, -1.0, 0.5) == 0.75
  ```

  Add a round-trip test over rising and falling nonzero spans and all 201 dense curve
  values:

  ```python
  @pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
  @pytest.mark.parametrize("curvature", np.linspace(-1.0, 1.0, 201))
  def test_control_level_and_inverse_round_trip(
      start: float,
      end: float,
      curvature: float,
  ) -> None:
      control = quadratic_control_level(start, end, float(curvature))
      recovered = curvature_from_control_level(start, end, control)
      assert recovered == pytest.approx(float(curvature), rel=0.0, abs=2e-15)
  ```

  Assert `curvature_from_control_level()` rejects a zero endpoint span with a named
  `ValueError`; the zero-span GUI path never calls the inverse. Directly test both helper
  functions against boolean and non-finite endpoints, out-of-range curvature, and a
  control level outside the closed endpoint interval. Field names must appear in each
  `ValueError`.

- [ ] **Step 2: Run the scalar tests and verify RED**

  ```bash
  uv run pytest tests/synth/test_curves.py -q
  ```

  Expected: collection fails because `harpy.synth.curves` does not exist.

- [ ] **Step 3: Add dense array, ownership, validation, and import-boundary RED tests**

  Add these exact categories:

  ```python
  @pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
  @pytest.mark.parametrize("curvature", np.linspace(-1.0, 1.0, 201))
  def test_dense_curve_is_monotone_and_bounded(
      start: float,
      end: float,
      curvature: float,
  ) -> None:
      positions = np.linspace(0.0, 1.0, 1_001)
      values = evaluate_quadratic_segment(start, end, float(curvature), positions)
      assert isinstance(values, np.ndarray)
      assert values.dtype == np.float64
      assert values.flags.c_contiguous
      assert not values.flags.writeable
      assert values[0] == start
      assert values[-1] == end
      low, high = sorted((start, end))
      assert np.all(values >= low)
      assert np.all(values <= high)
      differences = np.diff(values)
      if end >= start:
          assert np.all(differences >= 0.0)
      else:
          assert np.all(differences <= 0.0)


  @pytest.mark.parametrize(
      ("argument", "value"),
      [
          ("start_level", True),
          ("end_level", float("nan")),
          ("curvature", -1.001),
          ("curvature", 1.001),
          ("curvature", False),
          ("position", -0.001),
          ("position", 1.001),
          ("position", np.array([True, False])),
          ("position", np.array([0.0, float("inf")])),
          ("position", np.array([[0.0, 1.0]])),
      ],
  )
  def test_quadratic_segment_rejects_invalid_inputs(argument: str, value: object) -> None:
      arguments: dict[str, object] = {
          "start_level": 0.0,
          "end_level": 1.0,
          "curvature": 0.0,
          "position": 0.5,
      }
      arguments[argument] = value
      with pytest.raises(ValueError, match=argument):
          evaluate_quadratic_segment(**arguments)
  ```

  Add one test that mutating the supplied position array after the call cannot change the returned array.

  Extend `tests/test_imports.py` so importing `harpy.synth.curves` in a clean subprocess
  neither constructs Qt Multimedia nor writes stdout/stderr. Use the same subprocess
  helper and environment as the existing public-import smoke. The subprocess program
  becomes exactly:

  ```python
  "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.synth.curves, harpy.synth.engine, harpy.synth.patch_json"
  ```

  Retain `returncode == 0`, `stdout == b""`, and `stderr == b""`. Then run the complete
  new test surface before implementation:

  ```bash
  uv run pytest tests/synth/test_curves.py tests/test_imports.py -q
  ```

  Expected: RED because the new pure module is still absent.

- [ ] **Step 4: Implement the one authoritative polynomial**

  In `src/harpy/synth/curves.py`, validate scalar values without accepting booleans,
  accept only scalar or one-dimensional positions, and calculate the control level once
  through `quadratic_control_level()` before evaluating exactly. The inverse rejects a
  zero endpoint span and a control level outside the closed endpoint interval rather
  than returning an out-of-contract curvature:

  ```python
  control_level = quadratic_control_level(start, end, curve)
  values = (
      (1.0 - positions) ** 2 * start
      + 2.0 * (1.0 - positions) * positions * control_level
      + positions**2 * end
  )
  ```

  Explicitly assign exact endpoints when a supplied position equals `0.0` or `1.0`. Return `float(values)` for scalar input. For array input, copy to a contiguous `float64` array and mark it read-only before returning. Do not add Qt imports, GUI projection math, envelope state, or a second curve formula.

- [ ] **Step 5: Run focused curve/import tests and verify GREEN**

  ```bash
  uv run pytest tests/synth/test_curves.py tests/test_imports.py -q
  ```

- [ ] **Step 6: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/synth/curves.py tests/synth/test_curves.py tests/test_imports.py
  git commit -m "feat: add constrained envelope curves"
  ```

---

### Task 3: Migrate the patch, renderer, preview, and codec as one coherent v2 contract

**Files:**

- Modify: `src/harpy/synth/models.py`
- Modify: `src/harpy/synth/curves.py`
- Modify: `src/harpy/synth/envelope.py`
- Modify: `src/harpy/synth/engine.py`
- Modify: `src/harpy/synth/patch_json.py`
- Modify: `src/harpy/gui/window.py:402-414` only for the temporary facts-row summary retained until Task 10
- Modify: `tests/synth/test_models.py`
- Modify: `tests/synth/test_curves.py`
- Modify: `tests/synth/test_envelope.py`
- Modify: `tests/synth/test_engine.py`
- Modify: `tests/synth/test_patch_json.py`
- Modify: `tests/gui/test_window.py:443-452` only for the temporary facts-row assertion
- Modify: `tests/test_config.py`

**Interfaces:**

- Consumes: `evaluate_quadratic_segment()` from Task 2 and every Milestone A engine/patch public method.
- Produces:

  ```python
  @dataclass(frozen=True, slots=True)
  class EnvelopeConfig:
      attack_seconds: float = 0.001
      decay_seconds: float = 0.600
      sustain_db: float = -6.0
      release_seconds: float = 0.600
      attack_curve: float = 0.0
      decay_curve: float = 0.0
      release_curve: float = 0.0


  @dataclass(frozen=True, slots=True)
  class EnvelopePreview:
      position: np.ndarray
      attack_level: np.ndarray
      decay_level: np.ndarray
      release_level: np.ndarray
      sustain_level: float
      attack_frames: int
      decay_frames: int
      release_frames: int


  def sample_envelope_preview(
      envelope: EnvelopeConfig,
      render: RenderConfig,
      samples_per_stage: int = 129,
  ) -> EnvelopePreview:
  ```

  `SynthEngine` keeps its existing signature and methods. `loads_patch()` accepts strict v1 and v2; `dumps_patch()` and `save_patch()` emit canonical v2.

- [ ] **Step 1: Pin the awkward-frame Milestone A output before production changes**

  Add passing characterization tests against current `LinearEnvelope` and
  `SynthEngine`. Use the existing 7/9/8-frame event and assert the literal bit patterns,
  not recomputed expected formulas:

  ```python
  EXPECTED_LINEAR_ENVELOPE_BITS = np.array(
      [
          4594314991293244562,
          4598818590920615058,
          4601392076421969627,
          4603322190547985554,
          4604608933298662838,
          4605895676049340122,
          4607182418800017408,
          4606613483139180583,
          4606044547478343758,
          4605475611817506933,
          4604906676156670108,
          4604065244080245896,
          4603223812003821684,
          4602085940682148033,
          4600403076529299610,
          4598720212376451187,
          4595899476901929112,
          4591395877274558612,
          0,
      ],
      dtype=np.uint64,
  )

  EXPECTED_LINEAR_ENGINE_BITS = np.array(
      [
          0,
          1025724698,
          1036759167,
          1041246260,
          1040312847,
          1026296886,
          3183795601,
          3192413184,
          3194009889,
          3190642546,
          3180535862,
          1023623637,
          1036772298,
          1038855006,
          1035049145,
          1025520120,
          584321663,
          3159726069,
          2147483648,
      ],
      dtype=np.uint32,
  )
  ```

  Configure envelope durations `0.7`, `0.9`, and `0.8` seconds at 10 Hz with Sustain
  `-7.3 dB`; render 11 frames after note-on, then eight after note-off. For engine bits,
  use a 32 Hz renderer, 3 Hz oscillator, durations `7/32`, `9/32`, `8/32`, render the
  same 11/eight event, concatenate, and compare `.view(np.uint64)`/`.view(np.uint32)`
  with `np.testing.assert_array_equal`.

  Run before changing production:

  ```bash
  uv run pytest tests/synth/test_envelope.py tests/synth/test_engine.py -q
  ```

  Expected: PASS against Milestone A. These literals remain unchanged after migration.

- [ ] **Step 2: Rewrite model tests for the curve-enabled runtime model**

  Replace the default-envelope expectation with:

  ```python
  assert SynthPatch().envelope == EnvelopeConfig(
      attack_seconds=0.001,
      decay_seconds=0.600,
      sustain_db=-6.0,
      release_seconds=0.600,
      attack_curve=0.0,
      decay_curve=0.0,
      release_curve=0.0,
  )
  assert not hasattr(SynthPatch().envelope, "curve")
  ```

  Parametrize each curve field over `[-1.001, 1.001, True, False, nan, inf, -inf, "0"]` and assert the named field in `ValueError`. Add boolean rejection for durations, sustain, and output gain so Python booleans are never silently converted to floats. Assert every accepted `-0.0` curve, Sustain, and output-gain input is stored with positive-zero sign using `math.copysign(1.0, value) == 1.0`.

- [ ] **Step 3: Add preview and curved-renderer RED tests**

  Extend `tests/synth/test_curves.py` with:

  ```python
  def test_nominal_preview_uses_the_shared_curve_at_renderer_positions() -> None:
      render = RenderConfig(sample_rate_hz=4, block_frames=2)
      envelope = EnvelopeConfig(
          attack_seconds=1.0,
          decay_seconds=1.0,
          sustain_db=20.0 * math.log10(0.5),
          release_seconds=1.0,
          attack_curve=-0.5,
          decay_curve=0.25,
          release_curve=1.0,
      )
      preview = sample_envelope_preview(envelope, render, samples_per_stage=5)
      np.testing.assert_array_equal(preview.position, np.linspace(0.0, 1.0, 5))
      np.testing.assert_array_equal(
          preview.attack_level,
          evaluate_quadratic_segment(0.0, 1.0, -0.5, preview.position),
      )
      np.testing.assert_array_equal(
          preview.decay_level,
          evaluate_quadratic_segment(1.0, 0.5, 0.25, preview.position),
      )
      np.testing.assert_array_equal(
          preview.release_level,
          evaluate_quadratic_segment(0.5, 0.0, 1.0, preview.position),
      )
      assert (preview.attack_frames, preview.decay_frames, preview.release_frames) == (4, 4, 4)
  ```

  Assert all four preview arrays are one-dimensional, contiguous, owned/read-only
  `float64` data.

  Add composed preview validation cases for a duration that rounds below one frame and
  `1e308` duration-to-frame overflow. Each must raise a named `ValueError` for the exact
  Attack/Decay/Release field before allocating preview arrays:

  ```python
  @pytest.mark.parametrize("attack_seconds", [1e-12, 1e308])
  def test_preview_rejects_unrenderable_envelope(
      attack_seconds: float,
  ) -> None:
      envelope = replace(EnvelopeConfig(), attack_seconds=attack_seconds)

      with pytest.raises(ValueError, match="attack"):
          sample_envelope_preview(envelope, RenderConfig())
  ```

  In `tests/synth/test_envelope.py`, rename imports to `AdsrEnvelope` and add exact four-frame expectations:

  ```python
  def test_curved_attack_decay_and_release_use_stage_specific_curves() -> None:
      config = replace(
          half_sustain_config(),
          attack_curve=-1.0,
          decay_curve=1.0,
          release_curve=-1.0,
      )
      envelope = AdsrEnvelope(config, sample_rate_hz=4)
      envelope.note_on()
      np.testing.assert_array_equal(envelope.render(4), [0.0625, 0.25, 0.5625, 1.0])
      np.testing.assert_array_equal(envelope.render(4), [0.78125, 0.625, 0.53125, 0.5])
      envelope.note_off()
      np.testing.assert_array_equal(envelope.render(4), [0.46875, 0.375, 0.21875, 0.0])
  ```

  Retain the existing zero-curve vectors byte-for-byte as the legacy characterization.

- [ ] **Step 4: Add block-partition and release-source RED tests**

  Parameterize Attack, Decay, and Release curvature over `-1.0`, `-0.35`, `0.0`,
  `0.65`, and `1.0`. For each source stage (Attack, Decay, Sustain), render the same
  note-on/note-off event at identical frame offsets under partitions `[1]`, `[3, 5, 7]`,
  `[256]`, and one full block. Concatenate `float32` engine output and use
  `np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))`, not
  tolerance or direct floating-point equality. Comparing the encoded words is required
  because NumPy considers `+0.0` and `-0.0` equal even though their bits differ. Assert
  Release snapshots the exact last-emitted level, its first sample agrees with
  `evaluate_quadratic_segment(last_level, 0.0, release_curve, 1 / release_frames)` at
  `rtol=0`, `atol=1e-12`, and its final sample is exact zero. The separate zero-curve
  literal vectors, not this tolerance assertion, remain the bit-for-bit compatibility
  proof for the additive linear fast path.

  Use one event-aware helper that clamps every requested chunk at the fixed note-off frame
  before calling `render`; partition choice must never move the semantic event:

  ```python
  def render_partitioned_event(
      engine: SynthEngine,
      *,
      note_off_frame: int,
      total_frames: int,
      partition: list[int],
  ) -> np.ndarray:
      engine.note_on(3.0)
      rendered: list[np.ndarray] = []
      position = 0
      note_off_sent = False
      chunks = itertools.cycle(partition)
      while position < total_frames:
          if not note_off_sent and position == note_off_frame:
              engine.note_off()
              note_off_sent = True
          requested = next(chunks)
          stop = min(position + requested, total_frames)
          if not note_off_sent:
              stop = min(stop, note_off_frame)
          assert stop > position
          rendered.append(engine.render(stop - position))
          position = stop
      assert note_off_sent
      return np.concatenate(rendered).astype(np.float32, copy=False)
  ```

  Derive `note_off_frame` from the configured frame counts: an interior Attack frame,
  `attack_frames +` an interior Decay frame, and
  `attack_frames + decay_frames + 2` for Sustain. Set `total_frames` to the event offset
  plus the full Release plus five idle frames. For the “one full block” case pass
  `[total_frames]`; the helper still splits once at the event and therefore compares only
  block partitioning, not event timing. Explicitly assert that the Release-final sample
  retains the already-pinned oscillator-derived signed-zero word, while every one of the
  five subsequent Idle samples is canonical positive zero
  (`np.float32(0.0).view(np.uint32)`) under every partition.

  Upgrade the existing long held-note partition test from tolerance to encoded-word
  `assert_array_equal`. Add a fixed-frame Retune case where one engine reaches the Retune
  in one render and another reaches it through several partitions; require identical
  `uint32` output words before and after Retune. Keep the invalid-Retune atomicity and
  repeated-note-off tests exact so rejected or idempotent semantic events cannot silently
  reanchor phase.

  Add note-off before the first Attack sample and Sustain underflow-to-zero cases. Both
  must remain in Release for the full configured frame count, emit exact zero throughout,
  and transition to Idle only after the final Release frame.

  For frame counts `1`, `2`, `7`, and `257` and curves `-1.0`, `-0.73`, `0.0`,
  `0.61`, and `1.0`, construct preview with `samples_per_stage=frames + 1`. Compare
  endpoint-excluded preview positions `preview.*_level[1:]` against the corresponding
  float64 `AdsrEnvelope` stage output using:

  ```python
  np.testing.assert_allclose(
      rendered_stage,
      preview_stage[1:],
      rtol=0.0,
      atol=1e-12,
  )
  ```

  Add literal one-frame tests proving Attack emits `1.0`, Decay emits Sustain, Release
  emits `0.0`, and each transition occurs only after that emitted sample.

- [ ] **Step 5: Run the model/preview/envelope tests and verify RED**

  ```bash
  uv run pytest tests/synth/test_models.py tests/synth/test_curves.py tests/synth/test_envelope.py tests/synth/test_engine.py -q
  ```

  Expected: failures identify the missing curve fields, preview type, generic renderer,
  full zero-valued Release, and current block-relative oscillator phase bits.

- [ ] **Step 6: Replace the runtime envelope model without a legacy alias**

  Remove `EnvelopeConfig.curve`. Add the three curvature fields and a helper that rejects booleans, non-numbers, non-finite values, and values outside `[-1.0, 1.0]` while naming the exact field. Tighten the existing finite-float helper so booleans are rejected consistently for duration, sustain, and gain fields. Normalize every accepted numerical zero through `0.0 if number == 0.0 else number` before storing it.

  The stored curvature loop is:

  ```python
  for field_name in ("attack_curve", "decay_curve", "release_curve"):
      value = _finite_float(getattr(self, field_name), field_name)
      if not -1.0 <= value <= 1.0:
          raise ValueError(f"{field_name} must be finite and within -1..1")
      object.__setattr__(self, field_name, 0.0 if value == 0.0 else value)
  ```

  Add `EnvelopePreview` and `sample_envelope_preview()` to `curves.py`. Validate
  `EnvelopeConfig`, `RenderConfig`, and a non-boolean integer `samples_per_stage >= 2`.
  Before allocation, reuse `validate_renderable_patch(SynthPatch(envelope=envelope),
  render)` so sub-frame and overflowing stage durations share the authoritative composed
  field errors; do not invent a GUI-only duration policy. Use one read-only
  `np.linspace(0.0, 1.0, samples_per_stage, dtype=np.float64)` and call
  `evaluate_quadratic_segment()` for all three shaped stages. Release preview always
  starts from configured sustain amplitude.

- [ ] **Step 7: Replace `LinearEnvelope` with the generic `AdsrEnvelope` state machine**

  Keep the existing stage enum and public performance methods, but store segment start, target, frame count, emitted count, curvature, and the zero-curve linear step. The per-sample branch is:

  ```python
  self._emitted += 1
  if self._curvature == 0.0:
      self._level += self._linear_step
  else:
      self._level = evaluate_quadratic_segment(
          self._segment_start,
          self._target,
          self._curvature,
          self._emitted / self._segment_frames,
      )
  if self._emitted == self._segment_frames:
      self._level = self._target
  ```

  Attack selects `attack_curve`, Decay selects `decay_curve`, and Release snapshots the current level then selects `release_curve`. Preserve the zero-curve additive recurrence exactly. Note-off from any non-Idle/non-Release stage always begins the configured full-duration Release, including when the current level is exact zero. Delete the `LinearEnvelope` name; do not add an alias. Update `SynthEngine` to construct `AdsrEnvelope` without changing any public engine method.

  Make oscillator phase bit-for-bit partition invariant at the same time. The current
  engine applies `fmod` at every caller block, so mathematically equivalent partitions
  can differ by float32 bits. Replace block-relative `_phase` updates with one semantic
  interval anchor and an absolute integer frame offset:

  ```python
  self._phase_anchor = 0.0
  self._phase_increment = 0.0
  self._phase_frame_offset = 0

  frame_positions = np.arange(
      self._phase_frame_offset,
      self._phase_frame_offset + frame_count,
      dtype=np.float64,
  )
  phases = self._phase_anchor + self._phase_increment * frame_positions
  self._phase_frame_offset += frame_count
  ```

  `NOTE_ON` starts a new interval at anchor/offset zero. Before an accepted Retune, and
  before the first accepted `NOTE_OFF` that enters Release, call one helper:

  ```python
  def _reanchor_phase(self) -> None:
      self._phase_anchor = math.fmod(
          self._phase_anchor + self._phase_increment * self._phase_frame_offset,
          math.tau,
      )
      self._phase_frame_offset = 0
  ```

  Validate Retune before reanchoring so an invalid request is atomic. Repeated note-off
  during Release, note-off while Idle, and zero-length render do not reanchor. Patch
  replacement and Reset restore anchor/increment/offset to zero. This semantic-event
  anchoring preserves the existing 11-frame NoteOn / eight-frame Release literals (the
  accepted NoteOff creates the same one-time wrap they already characterize), while all
  caller partitions inside each interval evaluate the identical absolute float64 frame
  positions.

  Preserve that exactness when a render call crosses from Release into Idle without
  forcing an idle engine to keep evaluating its oscillator. Add a read-only private-DSP
  seam such as `AdsrEnvelope.release_frames_remaining: int | None`: when the call begins
  in Release it returns the number of samples through and including the Release-final
  sample, and otherwise returns `None`. `SynthEngine.render()` snapshots that value before
  rendering the envelope. After the oscillator/envelope product has been converted to
  `float32`, if the snapshot is not `None` and is smaller than `frame_count`, overwrite
  only `samples[release_frames_remaining:]` with canonical `np.float32(0.0)`. Never
  overwrite the Release-final sample at index `release_frames_remaining - 1`; its legacy
  oscillator-derived sign bit remains pinned. Calls that begin Idle keep the existing
  positive-zero fast return. This makes trailing Idle words identical whether they were
  requested in the Release-bearing block or a later block, without advancing idle phase.

- [ ] **Step 8: Run the numerical/engine slice and verify GREEN**

  ```bash
  uv run pytest tests/synth/test_models.py tests/synth/test_curves.py tests/synth/test_envelope.py tests/synth/test_engine.py tests/test_config.py -q
  ```

- [ ] **Step 9: Replace codec fixtures with strict v1-read/v2-write tests**

  Keep an exact `V1_DEFAULT` fixture containing `"curve": "linear_amplitude"`. Define exact `V2_DEFAULT` with schema version `2` and the three curve numbers in Attack/Decay/Release order. Add assertions:

  ```python
  def test_v1_migrates_to_zero_curve_runtime_patch() -> None:
      assert loads_patch(V1_DEFAULT) == SynthPatch()


  def test_default_patch_writes_canonical_v2() -> None:
      assert dumps_patch(SynthPatch()) == V2_DEFAULT


  def test_v1_loaded_then_saved_becomes_canonical_v2() -> None:
      assert dumps_patch(loads_patch(V1_DEFAULT)) == V2_DEFAULT
  ```

  Add a v2 fixture containing `-0.0` for all three curves, Sustain, and output gain;
  decoding then dumping must contain no `-0.0` token and must equal the corresponding
  positive-zero canonical document.

  Add closed-key tests proving v1 rejects v2 curve-number keys and v2 rejects the v1 `curve` string. Test unknown integer versions `0`, `3`, and `999`; non-integer versions; missing/extra/duplicate keys; booleans; `NaN`/infinities for all three curve fields; `1e400`; 400-digit integers; `-1.001`/`1.001`; trailing content; UTF-8; and round-trip. Inject both a partial stream-write failure and an `os.replace` failure; each must preserve the prior destination and remove the temporary sibling.

  Run the codec suite before changing the codec:

  ```bash
  uv run pytest tests/synth/test_patch_json.py -q
  ```

  Expected: RED because the strict v2 fixtures and migration behavior are not yet
  implemented.

- [ ] **Step 10: Implement strict per-version parsing and canonical v2 output**

  Use `_V1_ENVELOPE_KEYS` and `_V2_ENVELOPE_KEYS`. Make schema validation return integer `1` or `2`; reject wrong types separately from unsupported integers. For v1, require the exact linear string and construct three zero curves. For v2, validate all seven envelope numbers before constructing `EnvelopeConfig`. Extend non-finite field diagnosis to all curve fields. Keep runtime models schema-free.

  Build the output mapping in this exact order:

  ```python
  document = {
      "schema_version": 2,
      "oscillator": {"type": patch.oscillator.type.value},
      "envelope": {
          "attack_seconds": patch.envelope.attack_seconds,
          "decay_seconds": patch.envelope.decay_seconds,
          "sustain_db": patch.envelope.sustain_db,
          "release_seconds": patch.envelope.release_seconds,
          "attack_curve": patch.envelope.attack_curve,
          "decay_curve": patch.envelope.decay_curve,
          "release_curve": patch.envelope.release_curve,
      },
      "output_gain_dbfs": patch.output_gain_dbfs,
  }
  ```

  Preserve the existing temporary-sibling write, flush/close, `os.replace`, and cleanup behavior.

- [ ] **Step 11: Keep the temporary facts row truthful until integration**

  First update only
  `test_patch_facts_include_curve_and_keep_output_as_seventh_value` to require `Linear`
  for a default patch and `Curved` for a window constructed from a patch whose Attack
  curve is nonzero. Extend that test module's `make_window()` with an optional patch
  argument rather than exercising file I/O in this temporary presentation test.
  Run that focused test before changing the window:

  ```bash
  uv run pytest \
    tests/gui/test_window.py::test_patch_facts_include_curve_and_keep_output_as_seventh_value -q
  ```

  Expected: RED because the old facts row still accesses the removed `envelope.curve`
  field and uses the obsolete copy.

  Then replace the obsolete access with one summary: `Linear` when all three curves equal
  zero, otherwise `Curved`. Update no other window behavior. Task 10 removes this row
  entirely; do not add curve editors to `HarpyWindow` here.

  ```python
  curves = (envelope.attack_curve, envelope.decay_curve, envelope.release_curve)
  values["Curve"] = "Linear" if curves == (0.0, 0.0, 0.0) else "Curved"
  ```

- [ ] **Step 12: Run the migration slice and verify GREEN**

  ```bash
  uv run pytest tests/synth tests/test_config.py tests/gui/test_window.py -q
  ```

  Require zero output from the runtime-field/alias search:

  ```bash
  ! rg -n 'LinearEnvelope|envelope\.curve|curve: str = "linear_amplitude"' \
    src/harpy --glob '*.py'
  ```

  Then require `linear_amplitude` to occur only in the strict v1 branch of
  `src/harpy/synth/patch_json.py`:

  ```bash
  rg -n 'linear_amplitude' src/harpy --glob '*.py'
  ```

  No runtime-model field, renderer alias, engine path, or window access may contain it.

- [ ] **Step 13: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/synth src/harpy/gui/window.py tests/synth tests/gui/test_window.py tests/test_config.py
  git commit -m "feat: add curve-enabled patch schema"
  ```

---

### Task 4: Move natural idle to the drained release-block boundary

**Files:**

- Modify: `src/harpy/gui/qt_audio.py:83-197`
- Modify: `tests/gui/test_audio_backend.py`

**Interfaces:**

- Consumes: existing `SynthAudioSource.submit()`, `readData()`, and `voice_idle(int)`.
- Produces: the same signal signature, but it fires only after the encoded render block that first observed envelope-idle has been removed from replaceable staging into a `readData()` return.

- [ ] **Step 1: Add a RED test that proves a partial read cannot report idle**

  Build two sources with a four-frame release and the existing float-mono test format.
  Start the same note on both, consume Attack/Decay, submit `NOTE_OFF`, then request the
  release from one source in two partial reads and from the control source in one read:

  ```python
  def test_natural_idle_waits_until_release_bearing_pcm_leaves_staging() -> None:
      patch = short_patch(release_frames=4)
      source, _, render, audio_format = source_setup(patch=patch, generation=7)
      control, _, _, _ = source_setup(patch=patch, generation=7)
      idle_generations: list[int] = []
      source.voice_idle.connect(idle_generations.append)
      note_on = AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=7)
      source.submit(note_on)
      control.submit(note_on)
      block_bytes = render.block_frames * audio_format.bytesPerFrame()
      assert source.readData(block_bytes) == control.readData(block_bytes)
      source.submit(AudioCommand(AudioCommandKind.NOTE_OFF))
      control.submit(AudioCommand(AudioCommandKind.NOTE_OFF))

      half_bytes = 2 * audio_format.bytesPerFrame()
      first_half = source.readData(half_bytes)

      assert len(first_half) == half_bytes
      assert idle_generations == []
      second_half = source.readData(half_bytes)
      assert len(second_half) == half_bytes
      assert idle_generations == [7]
      assert first_half + second_half == control.readData(block_bytes)
      assert np.frombuffer(second_half, dtype=np.float32)[-1] == 0.0
  ```

  Use the test module's existing real format/source helpers instead of creating a
  parallel fake source API. The exact control-source comparison pins every release byte,
  while the last decoded mono frame pins exact silence.

- [ ] **Step 2: Run the focused test and verify RED**

  ```bash
  uv run pytest tests/gui/test_audio_backend.py::test_natural_idle_waits_until_release_bearing_pcm_leaves_staging -q
  ```

  Expected: FAIL because the current source emits `voice_idle` as soon as rendering reaches idle, before the first partial return drains staging.

- [ ] **Step 3: Add generation, cancellation, and conservative-block tests**

  Add tests proving:

  - the byte boundary works for mono/stereo float32 and mono/stereo int16 device formats;
    it is never tracked as mono sample frames;
  - held PCM already staged before the release block is included before the watermark;
  - the idle signal carries the generation captured when the transition block was rendered, even if `CLEAR_CAPTURE` later changes the append generation;
  - `RESET`, `REPLACE_PATCH`, and a new `NOTE_ON` cancel an undelivered old idle watermark and cannot emit a stale idle;
  - rendering additional trailing-silence blocks does not move the first watermark;
  - a `maxlen` spanning many render blocks short-returns at the release watermark rather
    than filling the request with arbitrary silence;
  - one `readData()` return that crosses the watermark emits exactly once;
  - `NOTE_ON` followed by `NOTE_OFF` before the first Attack sample still renders the
    complete zero-valued Release block and reports idle only after its encoded watermark
    drains;
  - the existing backend generation-alias chain forwards a later Clear generation correctly.

  Use signal lists and exact command sequences; do not add sleeps or inspect wall-clock time.

  Use this parameter table and exact cancellation assertion:

  ```python
  def _audio_format_for_test(
      channels: int,
      sample_format: QAudioFormat.SampleFormat,
  ) -> QAudioFormat:
      value = QAudioFormat()
      value.setSampleRate(SAMPLE_RATE_HZ)
      value.setChannelCount(channels)
      value.setSampleFormat(sample_format)
      return value


  def drive_to_release(source: SynthAudioSource, audio_format: QAudioFormat) -> None:
      source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=3))
      source.readData(4 * audio_format.bytesPerFrame())
      source.submit(AudioCommand(AudioCommandKind.NOTE_OFF))


  @pytest.mark.parametrize(
      ("channels", "sample_format"),
      [
          (1, QAudioFormat.SampleFormat.Float),
          (2, QAudioFormat.SampleFormat.Float),
          (1, QAudioFormat.SampleFormat.Int16),
          (2, QAudioFormat.SampleFormat.Int16),
      ],
  )
  def test_idle_watermark_counts_encoded_bytes(
      channels: int,
      sample_format: QAudioFormat.SampleFormat,
  ) -> None:
      audio_format = _audio_format_for_test(channels, sample_format)
      source, _, _, _ = source_setup(
          patch=short_patch(release_frames=4),
          generation=3,
          audio_format=audio_format,
      )
      idle: list[int] = []
      source.voice_idle.connect(idle.append)
      drive_to_release(source, audio_format)
      bytes_per_half = 2 * audio_format.bytesPerFrame()
      assert len(source.readData(bytes_per_half)) == bytes_per_half
      assert idle == []
      assert len(source.readData(bytes_per_half)) == bytes_per_half
      assert idle == [3]
  ```

  Extend existing `source_setup()` with an optional `audio_format` parameter that defaults
  to its current float-mono candidate. For each cancel command, assert `idle == []` after
  the next read.

  Run the expanded source tests before implementation:

  ```bash
  uv run pytest tests/gui/test_audio_backend.py -q
  ```

  Expected: RED in the new partial-drain, encoded-format, short-return, and cancellation
  cases while the pre-existing backend cases remain green.

- [ ] **Step 4: Replace `_natural_idle_pending` with an encoded-byte watermark**

  Store:

  ```python
  self._idle_watermark_bytes: int | None = None
  self._idle_generation: int | None = None
  ```

  When a render call changes `was_idle=False` to `engine.is_idle=True`, set the watermark to `len(self._staging)` immediately after appending that encoded block and store the current capture generation. Do not overwrite an existing watermark with later silence.

  Once a watermark exists, stop the fill loop even when `_staging` is shorter than
  `maxlen`. Return only bytes through that watermark. This short return is required: the
  source may not hand the sink an arbitrary number of trailing-silence blocks before the
  controller can enqueue the pending replacement.

  After selecting and deleting the bytes for the current `readData()` result, subtract the returned byte count from the watermark. Emit the stored generation once the count reaches zero, then clear both fields. Emission occurs after the bytes are removed into the current return, never while they remain replaceable. The conservative boundary can contain at most the trailing silence inside that single release-bearing render block.

  `RESET`, `REPLACE_PATCH`, and `NOTE_ON` call one private
  `_cancel_idle_watermark()` helper. `CLEAR_CAPTURE` changes only current append
  generation. `NOTE_OFF` never creates an immediate natural-idle notification: Task 3
  requires the full configured Release even from exact zero, so the watermark is recorded
  only when a later rendered release-bearing block performs the real transition to Idle.

- [ ] **Step 5: Run the complete source/backend suite and verify GREEN**

  ```bash
  uv run pytest tests/gui/test_audio_backend.py -q
  ```

  Confirm existing reset/replace preemption, lock linearization, queued-signal, device failure, hotplug, cache, alias, and shutdown tests remain unchanged and green.

- [ ] **Step 6: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/qt_audio.py tests/gui/test_audio_backend.py
  git commit -m "fix: drain release pcm before idle"
  ```

---

### Task 5: Add authored, pending, and applied patch state to the controller

**Files:**

- Modify: `src/harpy/gui/workbench_controller.py`
- Modify: `tests/gui/test_workbench_controller.py`

**Interfaces:**

- Consumes: the generation-tagged drained `voice_idle(int)` boundary from Task 4 and existing `REPLACE_PATCH`/`RESET` commands.
- Produces:

  ```python
  class PatchApplyState(StrEnum):
      APPLIED = "applied"
      PENDING = "pending"


  @dataclass(frozen=True, slots=True)
  class WorkbenchState:
      selected_frequency_hz: float
      gate_held: bool
      voice_may_be_active: bool
      patch: SynthPatch
      patch_apply_state: PatchApplyState
      capture: CaptureView
      audio_available: bool
      audio_error: str | None


  def commit_authored_patch(self, patch: SynthPatch) -> WorkbenchState:
  ```

  `replace_patch()` remains the immediate force-stop path used by file Load. `WorkbenchState.patch` is always the latest valid authored patch; `_applied_patch` is private controller state used only to resolve pending/cancelled intent.

- [ ] **Step 1: Add RED tests for idle authoring and active deferral**

  Add these controller assertions:

  ```python
  def test_idle_authoring_applies_one_complete_patch_and_clears_capture() -> None:
      controller, commands, _ = make_controller()
      patch = SynthPatch(envelope=EnvelopeConfig(attack_curve=0.5))

      state = controller.commit_authored_patch(patch)

      assert state.patch == patch
      assert state.patch_apply_state is PatchApplyState.APPLIED
      assert state.capture.state is CaptureState.EMPTY
      assert [command.kind for command in commands] == [AudioCommandKind.REPLACE_PATCH]
      assert commands[0].patch == patch
      assert commands[0].generation == state.capture.generation


  def test_held_authoring_changes_intent_without_touching_current_voice_or_capture() -> None:
      controller, commands, _ = make_controller()
      active = controller.press_play()
      patch = SynthPatch(envelope=EnvelopeConfig(decay_curve=-0.5))

      state = controller.commit_authored_patch(patch)

      assert state.patch == patch
      assert state.patch_apply_state is PatchApplyState.PENDING
      assert state.gate_held
      assert state.voice_may_be_active
      assert state.capture is active.capture
      assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
  ```

- [ ] **Step 2: Run the two tests and verify RED**

  ```bash
  uv run pytest \
    tests/gui/test_workbench_controller.py::test_idle_authoring_applies_one_complete_patch_and_clears_capture \
    tests/gui/test_workbench_controller.py::test_held_authoring_changes_intent_without_touching_current_voice_or_capture -q
  ```

  Expected: collection or attribute failures because `PatchApplyState` and `commit_authored_patch()` do not exist.

- [ ] **Step 3: Add coalescing, idle-generation, and Play-blocking tests**

  Add exact tests for this sequence:

  1. `NOTE_ON(original_generation)`;
  2. author patch A while held;
  3. author patch B while held;
  4. `NOTE_OFF`;
  5. `press_play()` while pending returns unchanged and adds no command;
  6. stale idle for `original_generation - 1` does nothing;
  7. matching idle allocates exactly one new capture generation and sends only `REPLACE_PATCH(B)`;
  8. the next `press_play()` queues `NOTE_ON` after replacement.

  Assert the command kinds exactly:

  ```python
  [
      AudioCommandKind.NOTE_ON,
      AudioCommandKind.NOTE_OFF,
      AudioCommandKind.REPLACE_PATCH,
      AudioCommandKind.NOTE_ON,
  ]
  ```

  Add a release-phase edit case, an active Clear generation-alias case, and a reversion case where authoring the privately applied patch cancels pending state without sending replacement.

  Clear remains enabled while Pending: it advances the active voice token through
  `CaptureCoordinator.clear()`, intentionally clears the retained observation, leaves
  `patch_apply_state` Pending, and causes only the aliased current idle generation to
  apply the patch. Assert the older pre-Clear token is ignored.

- [ ] **Step 4: Add force-stop, file-load, unavailable, and invalid-patch tests**

  Prove all of these atomically:

  - `force_stop()` with no pending patch retains the existing single `RESET` behavior;
  - `force_stop()` with a pending patch emits one `REPLACE_PATCH(latest)` instead of `RESET`, clears capture, and becomes applied;
  - both device callback orders (availability then force-stop, force-stop then availability) emit at most one replacement;
  - immediate `replace_patch(loaded)` discards pending intent, stops the voice, clears capture, and emits only the loaded replacement;
  - idle authoring while unavailable still emits one replacement for the backend cache;
  - sub-frame, overflowing-duration, wrong-type, and non-patch inputs leave authored/applied state, capture generation, gate, and command list unchanged;
  - a same-patch idle authoring commit is an exact no-op.

  Pin the pending force-stop command list exactly:

  ```python
  controller.press_play()
  latest = SynthPatch(envelope=EnvelopeConfig(release_curve=0.75))
  controller.commit_authored_patch(latest)

  stopped = controller.force_stop()
  stopped_again = controller.force_stop()

  assert [command.kind for command in commands] == [
      AudioCommandKind.NOTE_ON,
      AudioCommandKind.REPLACE_PATCH,
  ]
  assert commands[-1].patch == latest
  assert stopped == stopped_again
  assert stopped.patch_apply_state is PatchApplyState.APPLIED
  ```

  Run the complete expanded controller suite before implementation:

  ```bash
  uv run pytest tests/gui/test_workbench_controller.py -q
  ```

  Expected: RED across the new apply-state, coalescing, stale-generation, pending
  force-stop, and unavailable-cache command contracts.

- [ ] **Step 5: Implement the state machine with one replacement helper**

  Add private `_authored_patch`, `_applied_patch`, and `_patch_apply_state`. Return `_authored_patch` through `WorkbenchState.patch`.

  Use one helper with this exact responsibility:

  ```python
  def _admit_authored_patch(self) -> None:
      generation = self._capture.reset()
      self._capture_view = self._capture.refresh()
      self._send_command(
          AudioCommand(
              AudioCommandKind.REPLACE_PATCH,
              patch=self._authored_patch,
              generation=generation,
          )
      )
      self._applied_patch = self._authored_patch
      self._patch_apply_state = PatchApplyState.APPLIED
  ```

  Implement an internal helper that resets capture, sends one `REPLACE_PATCH`, records `_applied_patch = _authored_patch`, and sets `APPLIED`. Use it only when the patch differs from `_applied_patch` or an immediate file Load explicitly requires force-stop replacement.

  `commit_authored_patch()` validates before mutation. While `_voice_may_be_active`, it stores the latest patch and sets `PENDING` unless the candidate equals `_applied_patch`, which cancels pending. While idle, it immediately applies a changed candidate.

  `mark_voice_idle()` first rejects stale tokens. For the matching token, clear voice state and apply the latest pending patch exactly once. `press_play()` returns unchanged when state is pending. `force_stop()` uses the pending replacement as its reset operation; otherwise it preserves existing RESET idempotence. `replace_patch()` remains immediate and always sets both authored and applied patch to the loaded patch.

  Leave `clear_measurement()` callable in both apply states; no pending-specific command
  kind or disabled controller action is added.

- [ ] **Step 6: Run the controller suite and verify GREEN**

  ```bash
  uv run pytest tests/gui/test_workbench_controller.py -q
  ```

- [ ] **Step 7: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/workbench_controller.py tests/gui/test_workbench_controller.py
  git commit -m "feat: defer active patch authoring"
  ```

---

### Task 6: Build exact cached envelope value entry

**Files:**

- Create: `src/harpy/gui/envelope_entry.py`
- Create: `tests/gui/test_envelope_entry.py`

**Interfaces:**

- Consumes: human-entered duration, decibel, and curvature text; no controller or audio dependency.
- Produces:

  ```python
  class EnvelopeFieldKind(StrEnum):
      DURATION = "duration"
      DECIBELS = "decibels"
      CURVATURE = "curvature"


  class EnvelopeValueEntry(QLineEdit):
      value_commit_requested = Signal(float)
      validation_failed = Signal(str)
      draft_reverted = Signal()

      def __init__(
          self,
          field_name: str,
          kind: EnvelopeFieldKind,
          parent: QWidget | None = None,
      ) -> None:

      def set_exact_value(self, value: float) -> None:

      def accept_proposed_value(self, value: float) -> None:

      def mark_commit_rejected(self, message: str) -> None:

      def restore_last_valid(self) -> None:

      @property
      def kind(self) -> EnvelopeFieldKind:


  def format_envelope_duration(seconds: float) -> str:
  ```

- [ ] **Step 1: Add RED duration parsing and unit tests**

  Create `tests/gui/test_envelope_entry.py` and test real Return/Escape key events:

  ```python
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
  ```

  Add tests that bare input uses the currently rendered suffix: after `set_exact_value(0.6)`, bare `750` commits `0.750` seconds because the field displays `ms`; after `set_exact_value(1.2)`, bare `2` commits `2.0` seconds because the field displays `s`.

  Surrounding whitespace is accepted. Units are case-sensitive: `MS`, `S`, `db`, and
  `DB` are rejected. Return is the only commit event; focus-out preserves the draft
  without emitting.

- [ ] **Step 2: Run the duration tests and verify RED**

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py -q
  ```

  Expected: collection fails because `harpy.gui.envelope_entry` does not exist.

- [ ] **Step 3: Add dB, curvature, cache, and error tests**

  Cover these exact contracts:

  - dB accepts signed plain decimals with optional case-sensitive `dB`, rejects values above `0.0`, booleans supplied programmatically, commas, exponents, `NaN`, and infinities;
  - curvature accepts signed plain decimals in `[-1.0, 1.0]`, rejects just-outside values and suffixes;
  - duration rejects zero, negative, missing numeric text, unsupported units, commas, exponents, and more than nine fractional places;
  - programmatic `set_exact_value()` never emits;
  - display is `ms` below one second and `s` at or above one second;
  - display rounds to at most six fractional places, but unchanged Return emits the exact cached float rather than reparsing rounded text;
  - a valid manual proposal refreshes that exact cache only after
    `accept_proposed_value()`, so a second unchanged Return emits the first accepted exact
    float;
  - a locally valid proposal followed by `mark_commit_rejected()` preserves its typed
    text and prior exact cache; Escape then restores that prior value;
  - invalid Return preserves the typed text, sets `validationState="error"`, and emits one field-named error;
  - Escape restores the last valid rendered value, clears only the entry's validation
    property, and emits one `draft_reverted` without emitting a value proposal.

  Use these exact invalid-text tables:

  ```python
  @pytest.mark.parametrize(
      "text",
      ["0", "-1 ms", "NaN", "Infinity", "1e-3 s", "1,5 s", "1 MS", ".1234567890 s"],
  )
  def test_duration_entry_rejects_closed_grammar(qtbot, text: str) -> None:
      entry = EnvelopeValueEntry("attack_seconds", EnvelopeFieldKind.DURATION)
      assert_invalid_commit(qtbot, entry, text, "attack_seconds")


  @pytest.mark.parametrize("text", ["0.1", "NaN", "1e2", "-6 db", "-6 DB"])
  def test_decibel_entry_rejects_invalid_values(qtbot, text: str) -> None:
      entry = EnvelopeValueEntry("sustain_db", EnvelopeFieldKind.DECIBELS)
      assert_invalid_commit(qtbot, entry, text, "sustain_db")


  @pytest.mark.parametrize("text", ["-1.001", "1.001", "NaN", "0.5 dB", "1e-3"])
  def test_curvature_entry_rejects_invalid_values(qtbot, text: str) -> None:
      entry = EnvelopeValueEntry("attack_curve", EnvelopeFieldKind.CURVATURE)
      assert_invalid_commit(qtbot, entry, text, "attack_curve")
  ```

  Define the helper once:

  ```python
  def assert_invalid_commit(qtbot, entry: EnvelopeValueEntry, text: str, field: str) -> None:
      initial = {
          EnvelopeFieldKind.DURATION: 0.6,
          EnvelopeFieldKind.DECIBELS: -6.0,
          EnvelopeFieldKind.CURVATURE: 0.5,
      }[entry.kind]
      entry.set_exact_value(initial)
      committed: list[float] = []
      errors: list[str] = []
      entry.value_commit_requested.connect(committed.append)
      entry.validation_failed.connect(errors.append)
      entry.selectAll()
      qtbot.keyClicks(entry, text)
      qtbot.keyPress(entry, Qt.Key.Key_Return)
      assert committed == []
      assert len(errors) == 1 and field in errors[0]
      assert entry.text() == text
      assert entry.property("validationState") == "error"
  ```

  Expose a read-only `kind` property for this test and for editor composition; it returns
  the constructor's closed `EnvelopeFieldKind`.

  Run the complete entry suite again before implementation:

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py -q
  ```

  Expected: RED with all duration, dB, curvature, exact-cache, and error contracts now
  collected against the missing component.

- [ ] **Step 4: Implement one mode-driven exact-value entry**

  Use a closed regex for plain signed decimals with at most nine fractional places. Do not install `QValidator`; Return must reach the commit handler for invalid text. For duration, parse optional `ms`/`s` and divide milliseconds by `1_000.0`. For bare duration text, reuse the last canonical display unit.

  Render finite values with fixed six-decimal formatting followed by trailing-zero and trailing-point removal:

  ```python
  def _plain_decimal(value: float) -> str:
      text = f"{value:.6f}".rstrip("0").rstrip(".")
      return text if text not in ("", "-0") else "0"
  ```

  Cache both rendered text and exact model value on every successful programmatic set or
  parent-accepted manual proposal. A locally valid Return emits
  `value_commit_requested(value)` without changing the cache or canonicalizing the typed
  text. The owning editor then either calls `accept_proposed_value(value)` after complete
  patch/render validation or `mark_commit_rejected(message)` to preserve the draft and
  old cache while applying the error style. This is a real two-phase boundary; a
  sub-frame duration must never become the entry's last valid value.

  Validate kind-specific bounds before proposing or accepting. Re-polish the widget when
  changing `validationState` so tests exercise the real error style. Escape restores the
  cached authored value and emits exactly one `draft_reverted`; focus-out emits neither a
  value proposal nor a revert.

  Export `format_envelope_duration()` as the one GUI formatting helper used by entries
  and graph duration labels; it uses the same below-one-second unit switch and fixed-six-
  decimal trimming. It does not parse, validate renderability, or import Qt.

- [ ] **Step 5: Run the complete entry suite and verify GREEN**

  ```bash
  uv run pytest tests/gui/test_envelope_entry.py -q
  ```

- [ ] **Step 6: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/envelope_entry.py tests/gui/test_envelope_entry.py
  git commit -m "feat: add exact envelope value entry"
  ```

---

### Task 7: Build the transformed-time envelope graph and accessible handles

**Files:**

- Create: `src/harpy/gui/envelope_graph.py`
- Create: `tests/gui/test_envelope_graph.py`

**Interfaces:**

- Consumes: `EnvelopeConfig`, `RenderConfig`, and `sample_envelope_preview()` from Task 3.
- Produces:

  ```python
  class CurveStage(StrEnum):
      ATTACK = "attack"
      DECAY = "decay"
      RELEASE = "release"


  @dataclass(frozen=True, slots=True)
  class EnvelopeGraphGeometry:
      contents: QRectF
      stage_rects: tuple[tuple[str, QRectF], ...]
      label_rects: tuple[QRectF, ...]
      handle_centers: tuple[tuple[CurveStage, QPointF], ...]


  def envelope_stage_fractions(
      envelope: EnvelopeConfig,
      render: RenderConfig,
  ) -> dict[str, float]:


  class EnvelopeGraph(QWidget):
      curve_previewed = Signal(str, float)
      curve_commit_requested = Signal(str, float)
      curve_reverted = Signal(str)
      stage_selected = Signal(str)

      def __init__(self, render: RenderConfig, parent: QWidget | None = None) -> None:

      def set_envelope(self, envelope: EnvelopeConfig) -> None:

      def select_stage(self, stage: CurveStage) -> None:

      @property
      def selected_stage(self) -> CurveStage:

      @property
      def envelope(self) -> EnvelopeConfig:

      def preview_levels(self, stage: CurveStage) -> np.ndarray:

      def display_geometry(self) -> EnvelopeGraphGeometry:
  ```

- [ ] **Step 1: Add RED tests for the exact display projection**

  Test `envelope_stage_fractions()` independently:

  ```python
  def test_stage_fractions_keep_fixed_sustain_and_log_distribute_durations() -> None:
      render = RenderConfig(sample_rate_hz=48_000)
      envelope = EnvelopeConfig(
          attack_seconds=0.001,
          decay_seconds=0.600,
          release_seconds=1.200,
      )

      fractions = envelope_stage_fractions(envelope, render)

      assert fractions["sustain"] == 0.16
      assert sum(fractions.values()) == pytest.approx(1.0, rel=0.0, abs=1e-15)
      assert fractions["attack"] >= 0.14
      assert fractions["decay"] >= 0.14
      assert fractions["release"] >= 0.14
      assert fractions["attack"] < fractions["decay"] < fractions["release"]
  ```

  Calculate shaped fractions as `0.14 + 0.42 * weight / total_weight`, where each weight is `log1p(stage_frames)`. Validate that the exact three shaped fractions plus `0.16` sum to one; assign floating residual to Release so layout has no gap.

  Add a 1/3/7-frame case, whose `log1p` weights are in the ratio `1:2:3`, and
  assert exact expected fractions with tight tolerance:

  ```python
  assert fractions == pytest.approx(
      {"attack": 0.21, "decay": 0.28, "sustain": 0.16, "release": 0.35},
      rel=0.0,
      abs=1e-15,
  )
  ```

- [ ] **Step 2: Add RED signal tests for mouse preview versus one commit**

  Construct a visible `EnvelopeGraph`, find `attackCurveHandle`, press it, send three vertical mouse moves, and release. Assert at least one `curve_previewed("attack", value)` per changed move and exactly one `curve_commit_requested("attack", final_value)` on release. Assert every value is clamped to `[-1.0, 1.0]`, x does not change the model value, and no controller/audio object is required.

  ```python
  previews: list[tuple[str, float]] = []
  commits: list[tuple[str, float]] = []
  graph.curve_previewed.connect(lambda stage, value: previews.append((stage, value)))
  graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))
  handle = graph.findChild(QWidget, "attackCurveHandle")
  assert handle is not None
  send_handle_drag(handle, y_offsets=[-6, -12, -18], x_offsets=[-14, 0, 19])
  assert len(previews) == 3
  assert all(stage == "attack" and -1.0 <= value <= 1.0 for stage, value in previews)
  assert commits == [previews[-1]]
  ```

  Define the real-event helper:

  ```python
  def send_handle_drag(
      handle: QWidget,
      y_offsets: list[int],
      x_offsets: list[int] | None = None,
  ) -> None:
      origin = handle.rect().center()
      horizontal = x_offsets if x_offsets is not None else [0] * len(y_offsets)
      assert len(horizontal) == len(y_offsets)
      QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=origin)
      for x_offset, y_offset in zip(horizontal, y_offsets, strict=True):
          position = QPointF(origin.x() + x_offset, origin.y() + y_offset)
          event = QMouseEvent(
              QEvent.Type.MouseMove,
              position,
              position,
              handle.mapToGlobal(position.toPoint()),
              Qt.MouseButton.NoButton,
              Qt.MouseButton.LeftButton,
              Qt.KeyboardModifier.NoModifier,
          )
          QApplication.sendEvent(handle, event)
      QTest.mouseRelease(
          handle,
          Qt.MouseButton.LeftButton,
          pos=QPoint(
              origin.x() + horizontal[-1],
              origin.y() + y_offsets[-1],
          ),
      )
  ```

  Run a second drag from the same starting envelope with identical y offsets and
  radically different x offsets; assert the preview and final commit values are exactly
  equal. This is the executable proof that x motion cannot change curvature.

- [ ] **Step 3: Add keyboard, accessibility, identity, and paint-boundary RED tests**

  For each object name `attackCurveHandle`, `decayCurveHandle`, and `releaseCurveHandle`, assert:

  - Tab focus reaches the handle;
  - accessible role is `QAccessible.Role.Slider`, name contains the stage, and the value
    interface reports current float, minimum `-1.0`, maximum `1.0`, and minimum step
    `0.001`;
  - Right/Up commits `+0.01`, Left/Down commits `-0.01`, and Shift changes the step to `0.001`;
  - Home commits exactly `0.0`;
  - Escape emits only `curve_reverted(stage)`;
  - double-click commits only that stage at `0.0`;
  - focusing or pressing a handle emits `stage_selected(stage)`.

  Use actual Qt key/mouse events and `QAccessible.queryAccessibleInterface`; do not call event handlers directly.

  Each non-autorepeat arrow or Home press emits one commit immediately. Autorepeat is
  consumed without extra patch commits. The accessibility minimum step reports `0.001`;
  increase/decrease accessibility actions use the ordinary `0.01` increment.

  In standalone graph tests, connect proposal signals to a tiny owner harness that uses
  `dataclasses.replace()` and `graph.set_envelope()` before the next event. This mirrors
  the editor's real synchronous ownership without allowing `_CurveHandle` or
  `EnvelopeGraph` to invent a second curvature truth.

  ```python
  def accept_graph_curve(stage: str, value: float) -> None:
      graph.set_envelope(replace(graph.envelope, **{f"{stage}_curve": value}))


  graph.curve_previewed.connect(accept_graph_curve)
  graph.curve_commit_requested.connect(accept_graph_curve)
  ```

  Set Sustain to `0 dB` to create a zero-span Decay and to an underflowing finite level
  to create a visually zero-span Release. Assert no divide-by-zero, stable common-endpoint
  handle placement, no mouse commit, and successful keyboard/accessibility commits.

  Pin an ordinary keyboard sequence exactly:

  ```python
  handle.setFocus()
  qtbot.keyPress(handle, Qt.Key.Key_Right)
  qtbot.keyPress(handle, Qt.Key.Key_Up, modifier=Qt.KeyboardModifier.ShiftModifier)
  qtbot.keyPress(handle, Qt.Key.Key_Home)
  assert commits[-3:] == [
      ("attack", 0.01),
      ("attack", 0.011),
      ("attack", 0.0),
  ]
  ```

  For all three stages, compare graph-supplied preview levels with
  `evaluate_quadratic_segment()` using `rtol=0` and `atol=1e-12`. Resize to widths `288`,
  `360`, and `512`; call `display_geometry()` and assert every returned stage rectangle,
  label rectangle, and handle center stays within its returned contents rectangle.
  Render focused, selected, and linear states into
  `QImage` and assert they differ while model values remain unchanged.

  ```python
  def render_widget(widget: QWidget) -> QImage:
      image = QImage(widget.size(), QImage.Format.Format_ARGB32)
      image.fill(Qt.GlobalColor.transparent)
      painter = QPainter(image)
      widget.render(painter)
      painter.end()
      return image


  def image_bytes(image: QImage) -> bytes:
      return bytes(image.constBits()[: image.sizeInBytes()])


  np.testing.assert_allclose(
      graph.preview_levels(CurveStage.ATTACK),
      sample_envelope_preview(envelope, render).attack_level,
      rtol=0.0,
      atol=1e-12,
  )
  before = graph.envelope
  idle_image = render_widget(graph)
  graph.findChild(QWidget, "attackCurveHandle").setFocus()
  focused_image = render_widget(graph)
  assert image_bytes(idle_image) != image_bytes(focused_image)
  assert graph.envelope == before
  ```

  The read-only `envelope` and `preview_levels(stage)` properties exist for deterministic
  component tests; the latter returns owned read-only data and neither permits mutation
  of patch truth.

  Run the full graph test file before implementation:

  ```bash
  uv run pytest tests/gui/test_envelope_graph.py -q
  ```

  Expected: RED for the missing projection, real drag, keyboard, zero-span, and
  accessibility component.

- [ ] **Step 4: Implement the graph projection and handle model mapping**

  Paint a nominal complete envelope: Attack `0 -> 1`, Decay `1 -> sustain`, the fixed
  Sustain shelf, and Release `sustain -> 0`. Compute one fresh
  `EnvelopeGraphGeometry` per layout pass and use it for paint, child-handle placement,
  hit-independent labels, and the read-only test seam; never recalculate stage/label
  rectangles in separate paths. Return fresh `QRectF`/`QPointF` values so callers cannot
  mutate widget truth.

  Use `sample_envelope_preview()` for every curve polyline. Import and use
  `quadratic_control_level()` and `curvature_from_control_level()` for handle projection;
  do not duplicate either equation in paint or event code. Their shared contract is:

  ```python
  control_level = (start + end) / 2.0 + curvature * (end - start) / 2.0
  curvature = 2.0 * (control_level - (start + end) / 2.0) / (end - start)
  ```

  Keep each handle's x at the stage midpoint. Clamp vertical drag and model curvature; never derive physical duration from display x. When a stage amplitude span is visually unresolved, keep the handle keyboard/accessibility value authoritative.

  `set_envelope()` is the graph's only persistent value update. A handle gesture emits a
  proposed stage/value; the owning editor replaces its one `_draft_envelope` and calls
  `set_envelope()` back synchronously. The graph and its children never mutate or retain
  a parallel curvature field merely because a signal was emitted.

  For an exact zero span, place the handle at stage midpoint x and the shared endpoint
  y; vertical mouse movement emits no preview or commit. The handle remains focusable and
  adjustable through keyboard, the selected curvature entry, and accessibility actions.

  Paint stage boundary lines, `A`, `D`, `S`, `R` labels, canonical duration labels, the three guide-segment pairs `P0-P1-P2`, and distinct focused/selected handle states. Do not draw axes that imply linear physical time.

- [ ] **Step 5: Implement three focusable private handle widgets**

  Give each `_CurveHandle` a stable object name, strong focus, vertical-drag cursor,
  mouse grab through release, per-interaction origin value, and one commit signal on
  release. Install a `QAccessibleValueInterface` bridge following the existing
  frequency-knob pattern, but expose curvature `[-1.0, 1.0]` rather than integer pixels.
  The handle receives a callable that queries the parent graph's current immutable
  envelope; it keeps no durable curvature value of its own. Only a mouse interaction's
  origin value may be stored transiently until release/cancel.

  ```python
  class _CurveHandle(QWidget):
      previewed = Signal(str, float)
      commit_requested = Signal(str, float)
      reverted = Signal(str)
      selected = Signal(str)

      def __init__(
          self,
          stage: CurveStage,
          current_curve: Callable[[], float],
          parent: QWidget,
      ) -> None:

      @property
      def current_curve(self) -> float:
          return self._current_curve()
  ```

  Keyboard and accessibility actions calculate their next value from `current_curve` and
  emit a request. Programmatic graph synchronization replaces the graph's immutable
  envelope and repaints; it never pushes copied values into handle fields.

- [ ] **Step 6: Run the graph suite and verify GREEN**

  ```bash
  uv run pytest tests/gui/test_envelope_graph.py tests/synth/test_curves.py -q
  ```

- [ ] **Step 7: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/envelope_graph.py tests/gui/test_envelope_graph.py
  git commit -m "feat: add accessible envelope graph"
  ```

---

### Task 8: Compose the local-draft envelope inspector

**Files:**

- Create: `src/harpy/gui/envelope_editor.py`
- Create: `tests/gui/test_envelope_editor.py`

**Interfaces:**

- Consumes: `EnvelopeValueEntry`, `EnvelopeGraph`, `RenderConfig`, immutable `SynthPatch`, and `PatchApplyState`.
- Produces:

  ```python
  class EnvelopeEditor(QFrame):
      patch_commit_requested = Signal(object)
      validation_failed = Signal(str)
      validation_cleared = Signal()
      load_requested = Signal()
      save_requested = Signal()

      def __init__(
          self,
          render: RenderConfig,
          patch: SynthPatch,
          parent: QWidget | None = None,
      ) -> None:

      def set_patch_state(
          self,
          patch: SynthPatch,
          apply_state: PatchApplyState,
          *,
          discard_draft: bool = False,
      ) -> None:

      def discard_draft(self) -> None:
  ```

- [ ] **Step 1: Add RED construction and object-contract tests**

  Assert one `EnvelopeEditor` contains these exact object names:

  ```text
  envelopeEditor
  envelopeGraph
  attackEntry
  decayEntry
  sustainEntry
  releaseEntry
  curveEntry
  patchStatusLabel
  envelopeFieldError
  resetCurvesButton
  oscillatorFact
  outputFact
  loadPatchButton
  savePatchButton
  ```

  Verify Attack/Decay/Release entries are duration kind, Sustain is dB kind, the selected-handle entry is curvature kind, and all five entries have explicit accessible name and unit/range description.

- [ ] **Step 2: Add RED tests for complete-patch numeric commits**

  Start from a non-default patch so preservation is observable. Commit Attack through its field and assert one emitted `SynthPatch` has only `envelope.attack_seconds` changed; oscillator, other envelope values, curves, and output gain must compare equal to the starting patch. Repeat for Decay, Sustain, and Release.

  Test a sub-frame duration: the field retains invalid text, `envelopeFieldError` names
  Attack and “at least one frame,” no patch signal fires, and the latest valid patch
  remains unchanged. Then press Escape and assert the entry restores the original
  authored Attack text—not the locally parseable sub-frame proposal—and clears Editing.

  ```python
  emitted: list[SynthPatch] = []
  editor.patch_commit_requested.connect(emitted.append)
  attack = editor.findChild(EnvelopeValueEntry, "attackEntry")
  assert attack is not None
  attack.selectAll()
  qtbot.keyClicks(attack, "250 ms")
  qtbot.keyPress(attack, Qt.Key.Key_Return)
  assert len(emitted) == 1
  assert emitted[0].envelope.attack_seconds == 0.250
  assert emitted[0].oscillator == starting_patch.oscillator
  assert emitted[0].output_gain_dbfs == starting_patch.output_gain_dbfs
  assert replace(emitted[0].envelope, attack_seconds=starting_patch.envelope.attack_seconds) == (
      starting_patch.envelope
  )
  ```

- [ ] **Step 3: Add RED tests for graph draft and commit behavior**

  Feed three `curve_previewed` values for Attack and assert the graph plus curve entry show the latest local draft while `patch_commit_requested` remains empty. Feed one `curve_commit_requested`; assert exactly one full patch is emitted. Verify Decay and Release map only to their respective fields. Double-click stage reset emits one patch; `Reset curves to linear` sets all three to zero in one patch.

  Select Decay through the real graph selection signal, type `0.375` into `curveEntry`,
  press Return, and assert one complete patch changes only `decay_curve` while the Decay
  handle synchronizes without feedback. Repeat selection for Attack and Release, assert
  programmatic selection emits no patch, and assert invalid curvature text stays local
  until Escape restores the selected stage's latest authored value.

  ```python
  graph.curve_previewed.emit("attack", 0.10)
  graph.curve_previewed.emit("attack", 0.20)
  graph.curve_previewed.emit("attack", 0.30)
  assert emitted == []
  assert graph.envelope.attack_curve == 0.30
  assert curve_entry.text() == "0.3"
  graph.curve_commit_requested.emit("attack", 0.30)
  assert len(emitted) == 1
  assert emitted[0].envelope.attack_curve == 0.30
  ```

- [ ] **Step 4: Add RED state-synchronization tests**

  Prove:

  - `set_patch_state(same_patch, APPLIED)` during a 34 ms-style refresh preserves modified or invalid field text;
  - a successful emitted candidate returned through `set_patch_state(candidate, APPLIED)`
    clears Editing and canonicalizes every field/handle, including a semantically
    same-value commit whose controller action is an intentional no-op;
  - `set_patch_state(candidate, PENDING)` displays `Pending`, including while a newer
    local draft or invalid field exists;
  - a local draft displays `Editing` only when the underlying apply state is Applied;
    reverting it returns to `Active`;
  - `discard_draft=True` replaces invalid text and graph draft with the supplied file-loaded patch;
  - programmatic synchronization emits no commit, load, save, or validation signal;
  - Save/Load buttons emit their one intent signal and do not access the filesystem
    themselves; the editor creates no application/window shortcut.

  ```python
  attack.selectAll()
  qtbot.keyClicks(attack, "not complete")
  editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)
  assert attack.text() == "not complete"
  assert status.text() == "Editing"
  editor.set_patch_state(starting_patch, PatchApplyState.PENDING)
  assert attack.text() == "not complete"
  assert status.text() == "Pending"
  loaded_patch = replace(
      starting_patch,
      envelope=replace(starting_patch.envelope, attack_seconds=0.250),
  )
  editor.set_patch_state(loaded_patch, PatchApplyState.APPLIED, discard_draft=True)
  assert attack.text() == "250 ms"
  assert status.text() == "Active"
  ```

  Run the complete editor test file before implementation:

  ```bash
  uv run pytest tests/gui/test_envelope_editor.py -q
  ```

  Expected: RED for the missing component, complete-patch commits, draft lifecycle,
  state priority, and intent-only file actions.

- [ ] **Step 5: Implement inspector draft ownership and atomic candidate construction**

  Store `_authored_patch`, `_draft_envelope`, `_apply_state`, a local dirty/invalid field
  marker, and `_awaiting_ack_patch: SynthPatch | None`. Use `dataclasses.replace()` twice
  to construct candidates:

  ```python
  candidate_envelope = replace(self._draft_envelope, **{field_name: value})
  candidate_patch = replace(self._authored_patch, envelope=candidate_envelope)
  validate_renderable_patch(candidate_patch, self._render)
  ```

  Only after complete validation should `_awaiting_ack_patch` be set and
  `patch_commit_requested.emit(candidate_patch)` occur. For a numeric proposal, call
  `source_entry.accept_proposed_value(value)` only after that validation passes; on
  failure call `source_entry.mark_commit_rejected(message)` so its authored cache is not
  advanced. On success, assign `_draft_envelope = candidate_envelope`, synchronize graph
  presentation, accept the source entry, set the acknowledgement marker, then emit the
  complete patch in that order. Graph previews replace only `_draft_envelope` and update
  presentation. A graph release, numeric Return, selected-curve Return, per-stage
  double-click, or reset-all action emits at most one candidate.

  `set_patch_state()` treats equality with `_awaiting_ack_patch` as the synchronous
  controller acknowledgement: clear the marker, accept/canonicalize the supplied patch,
  and clear Editing even when the candidate equals the prior authored patch. A routine
  timer call has no acknowledgement marker and therefore preserves same-patch local text.
  A different supplied patch or `discard_draft=True` clears the marker and replaces the
  draft deterministically.

  Connect each field's user-only `textEdited` signal to mark local Editing without
  parsing, validating, emitting global errors, or building a patch. Programmatic text
  updates and the 34 ms capture refresh never set that marker.

  Status priority is exact: `Pending` for `PatchApplyState.PENDING`; otherwise `Editing`
  for any unresolved local draft or invalid text; otherwise `Active`. Sine and output
  gain remain read-only compact facts.

  Compose the fixed-width inspector vertically in this order: Envelope heading plus
  status, graph, a compact two-column A/D/S/R field grid, selected Curve field, reset
  action, local error line, Sine/output facts, and one Load/Save As row. Use tight
  contents margins and stretch only the graph so actions stay visible at minimum height;
  do not introduce a scroll area or collapse editor controls behind disclosure widgets.

- [ ] **Step 6: Keep validation ownership local and typed**

  Mark only the failing field/handle, show one concise `envelopeFieldError`, and emit the
  full field-specific message through `validation_failed`. A valid local commit emits
  `validation_cleared` and clears only editor-originated state. Connect every entry's
  `draft_reverted` signal so Escape clears that field's dirty/error marker and restores
  the latest authored value without emitting a candidate; handle reverts use the same
  editor path. Do not read controller, engine, device, or capture state from the editor.

  ```python
  try:
      validate_renderable_patch(candidate_patch, self._render)
  except ValueError as error:
      self._show_field_error(field_name, str(error))
      self.validation_failed.emit(str(error))
      return
  self._clear_field_error(field_name)
  self.validation_cleared.emit()
  self.patch_commit_requested.emit(candidate_patch)
  ```

  Style the editor as one restrained instrument panel using the existing workbench
  palette. Numeric fields use the existing monospace/error vocabulary; `Pending` uses
  violet, `Editing` uses cyan, `Active` uses neutral text, and the local error uses the
  existing red. Do not add shadows, gradients outside the graph/knob paint code,
  animation, bitmap assets, or a second global stylesheet.

- [ ] **Step 7: Run the editor component suite and verify GREEN**

  ```bash
  uv run pytest \
    tests/gui/test_envelope_entry.py \
    tests/gui/test_envelope_graph.py \
    tests/gui/test_envelope_editor.py -q
  ```

- [ ] **Step 8: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/envelope_editor.py tests/gui/test_envelope_editor.py
  git commit -m "feat: add envelope inspector component"
  ```

---

### Task 9: Redesign the frequency knob without changing its tuning mechanics

**Files:**

- Modify: `src/harpy/gui/frequency_knob.py`
- Modify: `tests/gui/test_frequency_knob.py`

**Interfaces:**

- Consumes: the existing logarithmic frequency domain and every Milestone A mouse, keyboard, wheel, center-reset, clamping, signal, and accessibility behavior.
- Produces:

  ```python
  def unit_to_angle_degrees(unit: float) -> float:
  ```

  Unit `0.0` maps to `225.0` degrees (lower-left), `0.5` to `90.0` degrees (top), and `1.0` to `-45.0` degrees (lower-right). `FrequencyKnob.frequency_changed` and all existing methods remain unchanged.

- [ ] **Step 1: Add RED orientation and geometry tests**

  Add exact pure assertions:

  ```python
  def test_conventional_dial_orientation_places_center_at_twelve_oclock() -> None:
      assert unit_to_angle_degrees(0.0) == 225.0
      assert unit_to_angle_degrees(0.5) == 90.0
      assert unit_to_angle_degrees(1.0) == -45.0
  ```

  Retain the existing `frequency_to_unit()` midpoint test and add one combined assertion proving tuned C3 is both logarithmic unit `0.5` and top angle `90.0` under a non-A440 tuning-derived range.

- [ ] **Step 2: Add RED hover, dragging, cursor, tooltip, and focus tests**

  Send real Enter, mouse press/move/release, Leave, and focus events. Assert:

  - Enter sets `_hovered` presentation and `Qt.SizeVerCursor`;
  - press sets `_dragging`, release clears it, and neither transition changes frequency without movement;
  - Leave clears hover only when not dragging;
  - the tooltip names vertical drag, Shift fine mode, wheel/arrows, and double-click center reset;
  - focus renders a circular/arc-integrated high-contrast state rather than the old dominant square border;
  - model values and emitted signals still follow every existing mechanics test exactly.

  ```python
  before = knob.frequency_hz
  QTest.mouseMove(knob, knob.rect().center())
  assert knob.cursor().shape() is Qt.CursorShape.SizeVerCursor
  assert knob._hovered
  QTest.mousePress(knob, Qt.MouseButton.LeftButton, pos=knob.rect().center())
  assert knob._dragging
  assert knob.frequency_hz == before
  QTest.mouseRelease(knob, Qt.MouseButton.LeftButton, pos=knob.rect().center())
  assert knob._hovered and not knob._dragging
  assert knob.frequency_hz == before
  assert all(
      phrase in knob.toolTip()
      for phrase in ("Vertical drag", "Shift", "Wheel", "arrow", "Double-click")
  )
  ```

- [ ] **Step 3: Add RED offscreen paint probes for the pro-audio shell**

  At `88 x 88`, render minimum, center, maximum, hover, dragging, and keyboard-focus states to owned `QImage`s. Assert minimum/center/maximum images differ, hover differs from idle, dragging differs from hover, and focus differs without changing the value. Probe the center pointer's top-sector pixels and require the old center-right indicator sector to remain background. Keep these probes component-local; they are not substitutes for native acceptance screenshots.

  ```python
  knob.set_frequency_hz(knob_center_hz)
  center_pixels = render_widget(knob)
  assert region_has_accent(center_pixels, QRect(40, 10, 8, 22))
  assert not region_has_accent(center_pixels, QRect(66, 38, 12, 12))
  knob.set_frequency_hz(knob_minimum_hz)
  minimum_pixels = render_widget(knob)
  knob.set_frequency_hz(knob_maximum_hz)
  maximum_pixels = render_widget(knob)
  assert image_bytes(minimum_pixels) != image_bytes(center_pixels)
  assert image_bytes(center_pixels) != image_bytes(maximum_pixels)
  ```

  Define local `render_widget()` and `image_bytes()` helpers in
  `tests/gui/test_frequency_knob.py` using the same owned-`QImage` implementation shown in
  Task 7; do not import a later task's test module or production graph code.
  `region_has_accent(image, rectangle)` must iterate explicit `(x, y)` coordinates and
  inspect `image.pixelColor(x, y)`, so width, row stride, and channel ordering are never
  guessed from raw bytes. Treat a pixel as accent when blue exceeds `200` and red is
  below `160`. Obtain frequency bounds from the test's constructor arguments rather than
  private widget state.

  Run the complete knob suite before implementation:

  ```bash
  uv run pytest tests/gui/test_frequency_knob.py -q
  ```

  Expected: RED only in the new sweep, presentation-state, tooltip, and paint probes;
  the frozen Milestone A tuning-mechanics tests remain green.

- [ ] **Step 4: Implement the conventional sweep and interaction states**

  Use:

  ```python
  def unit_to_angle_degrees(unit: float) -> float:
      return 225.0 - 270.0 * min(1.0, max(0.0, unit))
  ```

  Replace the floating dot with a solid circular cap, restrained radial face shading, an outer endpoint-gap track, an emphasized center notch, and a pointer from cap center toward the current angle. Draw `C2`, `C3`, and `C4` landmarks within the widget bounds; C3 is centered above the cap. Use existing palette colors `#11151b`, `#181e27`, `#2a3441`, `#65d8ff`, `#bd8cff`, and readable neutral text rather than adding assets.

  Track `_hovered` and `_dragging`; repaint on transitions. Hover brightens the outer track, dragging uses the violet accent, and focus adds a narrow high-contrast circular halo. Set the vertical-resize cursor and the exact interaction tooltip. Keep relative-drag origin, dynamic Shift sampling, cents-per-pixel, wheel, arrows, double-click, no-wrap clamping, signals, and `QAccessibleValueInterface` calculations unchanged.

- [ ] **Step 5: Run the complete knob suite and verify GREEN**

  ```bash
  uv run pytest tests/gui/test_frequency_knob.py -q
  ```

  The pre-existing mechanics tests must pass without relaxed assertions.

- [ ] **Step 6: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/frequency_knob.py tests/gui/test_frequency_knob.py
  git commit -m "feat: refine the frequency control"
  ```

---

### Task 10: Integrate the envelope inspector into the native workbench

**Files:**

- Modify: `src/harpy/gui/window.py`
- Modify: `src/harpy/gui/app.py`
- Modify: `tests/gui/test_window.py`
- Modify: `tests/gui/test_app.py`
- Modify: `tests/gui/test_audio_backend.py`
- Modify: `tests/test_imports.py`

**Interfaces:**

- Consumes: `EnvelopeEditor`, `WorkbenchController.commit_authored_patch()`, `PatchApplyState`, drained idle notifications, redesigned `FrequencyKnob`, existing patch dialogs, and existing capture views.
- Produces this constructor and no backend ownership in the window:

  ```python
  def __init__(
      self,
      controller: WorkbenchController,
      tuning: Tuning,
      spec: WorkbenchSpec,
      render: RenderConfig,
      patch_dialogs: PatchDialogPort,
  ) -> None:
  ```

  `build_runtime()` passes `config.render`; signal ownership and shutdown order stay unchanged.

- [ ] **Step 1: Replace window contract tests and verify RED**

  Update the widget contract to require:

  ```text
  frequencyControlGroup
  frequencyKnob
  frequencyEntry
  derivedPitchLabel
  playButton
  measurementStateLabel
  clearButton
  waveformPlot
  spectrumPlot
  envelopeEditor
  envelopeGraph
  attackEntry
  decayEntry
  sustainEntry
  releaseEntry
  curveEntry
  patchStatusLabel
  resetCurvesButton
  oscillatorFact
  outputFact
  loadPatchButton
  savePatchButton
  audioErrorBanner
  ```

  Assert `patchFacts` and all seven legacy fact labels are absent. Update window factories to pass `RenderConfig`; run the focused construction test and expect RED because the old constructor/layout still exists.

- [ ] **Step 2: Add RED layout tests at default and minimum sizes**

  At `1280 x 720` and `1024 x 640`, process layout events and assert:

  - `EnvelopeEditor.width() == 288`;
  - it spans the same main-row height as the plot container;
  - Waveform and Spectrum retain `4:6` horizontal stretch inside the remaining width;
  - both actual pyqtgraph ViewBoxes have data-canvas height at least `320 px` in idle, pending, and error-banner states;
  - no horizontal scrollbar appears;
  - every editor control is contained by the inspector rectangle;
  - `frequencyKnob.size() == QSize(88, 88)` and the Frequency label, dial, Hz entry, and derived note all lie inside `frequencyControlGroup`.

  ```python
  @pytest.mark.parametrize("size", [QSize(1_280, 720), QSize(1_024, 640)])
  def test_envelope_workbench_layout_contract(qtbot, size: QSize) -> None:
      window, _, _, _, _ = make_window(qtbot)
      window.resize(size)
      window.show()
      QApplication.processEvents()
      editor = window.findChild(EnvelopeEditor, "envelopeEditor")
      assert editor is not None and editor.width() == 288
      assert window.frequency_knob.size() == QSize(88, 88)
      assert window.waveform_view.plot_item.vb.height() >= 320
      assert window.spectrum_view.plot_item.vb.height() >= 320
      for child in editor.findChildren(QWidget):
          assert editor.rect().contains(child.mapTo(editor, child.rect().topLeft()))
  ```

  Extend the containment assertion to each child's bottom-right point and assert the plot
  container widths with `abs(waveform_width / spectrum_width - 4 / 6) <= 0.03`.

- [ ] **Step 3: Add RED end-to-end authoring lifecycle tests**

  Through real editor widgets and a fake command sink, test:

  1. idle Attack commit sends one `REPLACE_PATCH`, displays `Active`, clears captured plots, and preserves selected hertz;
  2. held commit displays `Pending`, sends no replacement, keeps Play enabled/down so mouse or Space release can dispatch `NOTE_OFF`, and leaves the current observation visible;
  3. after release, Play is disabled while the release remains possible;
  4. stale idle leaves Pending;
  5. matching drained idle sends one latest replacement, clears plots, displays `Active`, and enables next Play when audio is available;
  6. three graph previews plus one release produce one replacement, not a command storm;
  7. a new Play cannot pass controller guards while Pending even if invoked directly.

  Assert exact command kinds and generation values rather than only button copy.

  ```python
  window._press_play()
  held_generation = controller.state.capture.generation
  editor.patch_commit_requested.emit(patch_a)
  editor.patch_commit_requested.emit(patch_b)
  assert controller.state.patch_apply_state is PatchApplyState.PENDING
  assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
  assert window.play_button.isEnabled()
  window._release_play()
  assert not window.play_button.isEnabled()
  window.handle_voice_idle(held_generation - 1)
  assert controller.state.patch_apply_state is PatchApplyState.PENDING
  window.handle_voice_idle(held_generation)
  assert commands[-1].kind is AudioCommandKind.REPLACE_PATCH
  assert commands[-1].patch == patch_b
  assert window.play_button.isEnabled()
  ```

- [ ] **Step 4: Add RED editor/file/error synchronization tests**

  Prove:

  - the 34 ms capture refresh does not overwrite invalid numeric text or a graph draft;
  - an editor error marks the field and uses the single banner without hiding an existing device error;
  - a later valid editor commit clears only editor error; file/device errors remain;
  - device recovery clears only the audio error and reveals any still-current frequency,
    editor, or file error under the declared priority; successful file/frequency actions
    likewise clear only their own category;
  - Save As writes `state.patch`, including a pending authored patch, as canonical v2;
  - valid v1 Load force-stops immediately, discards pending/invalid editor draft, refreshes three linear handles, preserves frequency, and later saves v2;
  - valid v2 Load refreshes all values/handles without feedback signals;
  - either successful Load clears the file error and the discarded editor-local error,
    while preserving any unrelated frequency error; Save success clears only file error;
  - invalid JSON, render-invalid patch, read failure, save failure, and cancelled dialogs leave patch, apply state, capture, gate, selected frequency, editor draft, and commands unchanged as specified;
  - Load during a held Space gesture cannot strand the gate;
  - Space in each of `attackEntry`, `decayEntry`, `sustainEntry`, `releaseEntry`, and
    `curveEntry` remains text input and emits no `NOTE_ON`/`NOTE_OFF`, while a Space
    release from any owned modal or changed focus still releases an already-held gate;
  - Clear remains enabled while Pending, clears the measurement by explicit user intent,
    preserves Pending, and applies only on the aliased current idle token;
  - close/about-to-quit with Pending sends the one replacement before backend shutdown,
    is idempotent, discards invalid local draft, and never emits Save As;
  - Reset curves, Clear, Load, and Save As keep non-Space keyboard routes.
  - Ctrl+O and the platform Save As sequence each invoke exactly one dialog; there is one
    window-owned `QShortcut` per action and no editor-owned duplicate.

  Pin error ownership with one sequence:

  ```python
  window.handle_audio_failure("device failed")
  editor.validation_failed.emit("attack_seconds must contain at least one frame")
  assert window._error_label.text() == "device failed"
  editor.validation_cleared.emit()
  assert window._error_label.text() == "device failed"
  window.handle_audio_availability(True)
  assert window._error_label.text() == ""
  ```

  In a second sequence, create a real Save failure, then emit editor and frequency
  validation errors and refresh capture; the banner must continue to show the file
  failure. A later successful Save clears only that file error and reveals the still-
  current editor error.

  For pending Save, call the real dialog port and assert:

  ```python
  text = saved_path.read_text(encoding="utf-8")
  assert json.loads(text)["schema_version"] == 2
  assert text == dumps_patch(controller.state.patch)
  assert load_patch(saved_path) == controller.state.patch
  ```

  Extend the real backend fake integration to author while unavailable, then recover and
  assert the rebuilt source is idle with the latest patch. Add the pending-device-failure
  sequence and prove the backend cache receives one replacement generation with no
  duplicate reset.

  Add one source/controller integration test without sleeps: pause after a partial return
  has rendered envelope-idle but before the release watermark drains; author a pending
  patch and assert no replacement command has reached the source. Drain the remaining
  bytes, deliver queued idle, and assert exactly one replacement follows. Decode the
  complete release bytes and compare them bit-for-bit with an unedited control source.
  Repeat the boundary with device failure before watermark handoff and prove the latest
  authored patch enters the unavailable backend cache once while the stale idle is
  harmless.

  Run the complete replacement/integration surface before changing production:

  ```bash
  uv run pytest \
    tests/gui/test_window.py \
    tests/gui/test_app.py \
    tests/gui/test_audio_backend.py \
    tests/test_imports.py -q
  ```

  Expected: RED in the constructor, layout, authoring lifecycle, error ownership,
  patch-file synchronization, pending shutdown, and cross-boundary release cases.

- [ ] **Step 5: Replace the workbench layout without moving domain logic into the window**

  Build a visible `Frequency` group in the existing compact transport and set the knob to
  exactly `88 x 88`. Use a titled `QFrame`/layout with object name
  `frequencyControlGroup`; place the Frequency label above or beside one coherent row
  containing dial, exact-Hz field/suffix, and derived note/cents. Preserve the compact
  transport height and make the group's border/fill use the existing panel palette so it
  reads as one control rather than four unrelated widgets.

  Replace the plot-only row plus bottom facts row with one main `QHBoxLayout`: a plot
  container holding the existing `4:6` layout at stretch `1`, then `EnvelopeEditor` at
  fixed width `288`. Remove `_patch_value_labels`, `_set_patch_facts()`, the bottom row,
  and the obsolete Curve fact summary.

  Pass `RenderConfig` only to `EnvelopeEditor`. The window must not evaluate Bézier mathematics, derive stage widths, or hold a second patch draft.

  The main-row composition is:

  ```python
  plot_container = QWidget()
  plot_layout = QHBoxLayout(plot_container)
  plot_layout.setContentsMargins(0, 0, 0, 0)
  plot_layout.setSpacing(12)
  plot_layout.addWidget(self.waveform_view, 4)
  plot_layout.addWidget(self.spectrum_view, 6)
  self.envelope_editor = EnvelopeEditor(render, controller.state.patch)
  self.envelope_editor.setObjectName("envelopeEditor")
  self.envelope_editor.setFixedWidth(288)
  main_row = QHBoxLayout()
  main_row.addWidget(plot_container, 1)
  main_row.addWidget(self.envelope_editor)
  root.addLayout(main_row, 1)
  ```

- [ ] **Step 6: Wire complete-patch intent and state synchronization**

  Connect:

  ```python
  self.envelope_editor.patch_commit_requested.connect(self._commit_authored_patch)
  self.envelope_editor.validation_failed.connect(self._show_editor_error)
  self.envelope_editor.validation_cleared.connect(self._clear_editor_error)
  self.envelope_editor.load_requested.connect(self._load_patch)
  self.envelope_editor.save_requested.connect(self._save_patch)
  ```

  Retain the existing window-owned Clear, Open, and Save As `QShortcut`s and connect each
  to exactly one action path. Remove the old window-owned Load/Save button objects with
  the facts row, and do not create shortcuts inside `EnvelopeEditor`; its new buttons
  are the only additional intent sources.

  `_commit_authored_patch(patch)` calls only `controller.commit_authored_patch(patch)`, then applies returned state. `_apply_state()` passes `state.patch` and `state.patch_apply_state` to the editor. Routine timer updates preserve editor draft; successful commits synchronize the new patch; file Load calls the editor synchronization with `discard_draft=True`.

  Keep separate `_frequency_error`, `_editor_error`, and `_file_error`. Render exactly
  one existing banner with priority `state.audio_error`, file error, editor error, then
  frequency error. A valid operation clears only its own category, except successful
  Load also clears `_editor_error` because `discard_draft=True` intentionally destroys
  that invalid local draft. This priority ensures a later editor/frequency refresh cannot
  hide a still-current file or device failure.

  Generalize the global Space-filter exclusion from the one frequency entry to every
  `QLineEdit` owned by the window. Do not enumerate editor object names in event routing;
  text editors own Space, and all other focused workbench controls keep press-and-hold
  Play semantics.

- [ ] **Step 7: Pin Play enabling and force-stop behavior**

  Use this semantic rule:

  ```python
  can_release_held_gate = state.gate_held
  can_start_new_voice = state.audio_available and state.patch_apply_state is PatchApplyState.APPLIED
  self.play_button.setEnabled(can_release_held_gate or can_start_new_voice)
  ```

  Keep the button down for a held mouse/Space gesture. After `NOTE_OFF`, Pending disables new Play until matching idle applies replacement. Deactivation, device failure, close, and quit continue through one `controller.force_stop()` path; a pending patch makes that one command `REPLACE_PATCH`, not RESET plus replacement.

- [ ] **Step 8: Update runtime composition and backend recovery integration**

  In `build_runtime`, pass arguments in this exact order:

  ```python
  window = HarpyWindow(
      controller,
      config.tuning,
      spec,
      config.render,
      patch_dialogs if patch_dialogs is not None else NativePatchDialogs(),
  )
  ```

  Preserve signal connections before `audio.start()`. Satisfy the unavailable-recovery,
  pending-failure, and release-watermark integration regressions already written in Step
  4; do not add a second runtime graph or synchronous source callback path.

- [ ] **Step 9: Run focused integration tests and verify GREEN**

  ```bash
  uv run pytest \
    tests/gui/test_window.py \
    tests/gui/test_app.py \
    tests/gui/test_audio_backend.py \
    tests/test_imports.py -q
  ```

  Run these structural searches and require zero output:

  ```bash
  ! rg -n 'patchFacts|_patch_value_labels|_set_patch_facts|LinearEnvelope' src/harpy
  ! rg -n 'Gymnasium|gymnasium|reward|actor adapter|model adapter' src/harpy
  ```

- [ ] **Step 10: Run task-wide gates and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git add src/harpy/gui/window.py src/harpy/gui/app.py tests/gui/test_window.py tests/gui/test_app.py tests/gui/test_audio_backend.py tests/test_imports.py
  git commit -m "feat: integrate envelope authoring workbench"
  ```

---

### Task 11: Verify and document Milestone B as an honest research artifact

**Files:**

- Modify: `README.md`
- Modify: `docs/project-notebook.md`
- Create: `docs/verification/2026-08-08-milestone-b-acceptance.md`
- Create only when captured: `docs/verification/assets/milestone-b-idle.png`
- Create only when captured: `docs/verification/assets/milestone-b-editing.png`
- Create only when captured: `docs/verification/assets/milestone-b-pending.png`
- Create only when captured: `docs/verification/assets/milestone-b-live.png`

**Interfaces:**

- Consumes: the complete integrated Milestone B workbench and observed automated/native evidence.
- Produces: accurate current-capability documentation and a reproducible acceptance record; it does not change Python behavior.

- [ ] **Step 1: Run the exact clean automated gates**

  From the feature worktree, record command, date, exit code, and exact result for:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run python -c "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.synth.curves, harpy.synth.engine, harpy.synth.patch_json"
  git diff --check origin/main...HEAD
  find src/harpy -type f -name '*.py' -print0 | xargs -0 wc -l
  ```

  The import smoke must exit zero with zero stdout and zero stderr. If any gate fails, stop documentation and return to the owning task with a new failing regression.

- [ ] **Step 2: Run structural contract searches**

  Require zero production matches for removed or deferred concepts:

  ```bash
  ! rg -n 'LinearEnvelope|patchFacts|_patch_value_labels|_set_patch_facts' src/harpy
  ! rg -n 'Gymnasium|gymnasium|reward|actor adapter|model adapter' src/harpy
  ! rg -n 'from PySide6|import PySide6|pyqtgraph' src/harpy/synth src/harpy/analysis.py src/harpy/capture.py src/harpy/playback.py src/harpy/tuning.py
  ```

  Inspect every production module's line count and responsibility. Do not split `qt_audio.py` merely because it remains the largest platform file; flag only mixed responsibilities or dead paths.

- [ ] **Step 3: Exercise native WSLg launch and audio route**

  Launch exactly:

  ```bash
  XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
  ```

  Preserve the inherited `PULSE_SERVER`. Once Play creates the stream, resolve the sole
  Pulse input whose `application.name` is `harpy`, then prove its process working
  directory is the current worktree:

  ```bash
  harpy_audio_pid="$(
    pactl --format=json list sink-inputs |
      uv run python -c 'import json, sys; items = json.load(sys.stdin); matches = [item for item in items if item.get("properties", {}).get("application.name") == "harpy"]; assert len(matches) == 1; print(matches[0]["properties"]["application.process.id"])'
  )"
  test "$(readlink -f "/proc/$harpy_audio_pid/cwd")" = "$(pwd -P)"
  tr '\0' ' ' <"/proc/$harpy_audio_pid/cmdline"
  ```

  Also verify that stream is 48 kHz and belongs to the visible process. Do not use broad
  process-kill commands; close only the verified Harpy process through its window.

- [ ] **Step 4: Exercise native envelope and deferred-apply acceptance**

  Observe and record:

  - v1 default Load displays 1 ms/600 ms/-6 dB/600 ms and three zero handles, then Save As emits canonical v2;
  - numeric edits and each graphical handle produce the configured curve on the next Play;
  - a held note remains audibly/observably unchanged while editing;
  - after release, the full old release tail reaches zero before Pending becomes Active;
  - multiple held/release edits apply only the final patch;
  - the next Play uses that final patch;
  - invalid numeric and JSON edits are atomic and field-specific;
  - Clear, Load, device-failure simulation/recovery, deactivation, and close leave no stuck sound;
  - waveform/spectrum capture stays retained through the old release and clears when pending replacement applies.

  Use real generated/captured samples and command/generation evidence for objective claims. Label subjective hearing as human observation, not automated proof.

  During each held/release phase, record the verified Harpy stream only:

  ```bash
  pactl list short sink-inputs
  pactl list sink-inputs | rg -n -C 3 'State:|application.name|application.process.id|media.name'
  ps -eo pid,ppid,etime,args | rg '[h]arpy'
  ```

  If using `RDPSink.monitor`, bound the capture, ensure Harpy is the sole sink input, and
  record the selected patch, generation, sample count, measured tail peak, and comparison
  hash. Do not claim subjective audibility from Pulse evidence.

- [ ] **Step 5: Exercise native knob and layout acceptance**

  Verify continuous vertical drag, dynamic Shift fine mode, one-cent wheel/arrows, Shift `0.1` cent, C3 double-click reset, clamping, live/release retune without retrigger, pointer orientation, landmarks, hover cursor, and focus/accessibility. Visually inspect default and minimum client sizes at nominal Windows `125%` and `100%` scaling. If authority or tooling cannot change scale or capture the complete outer frame, record that exact check as pending rather than inferring a pass.

  Record the current scale evidence before each visual check:

  ```bash
  powershell.exe -NoProfile -Command \
    "Get-ItemProperty 'HKCU:\Control Panel\Desktop\WindowMetrics' -Name AppliedDPI"
  ```

  Pair that host result with the native client geometry reported by Qt; neither one alone
  proves complete-frame fit.

- [ ] **Step 6: Capture qualifying screenshots only when the evidence is real**

  Capture idle, editing, pending, and live states with the complete native outer frame and no private desktop data. Store them at the four listed asset paths and link them relatively from the acceptance document. If complete-frame capture is unavailable, create no placeholder image and record screenshots as pending.

  ```text
  docs/verification/assets/milestone-b-idle.png
  docs/verification/assets/milestone-b-editing.png
  docs/verification/assets/milestone-b-pending.png
  docs/verification/assets/milestone-b-live.png
  ```

- [ ] **Step 7: Update current-capability documentation**

  In `README.md`:

  - describe Milestone B numeric/graphical ADSR authoring and the refined knob as current only after acceptance gates pass;
  - document strict v1 read/canonical v2 write and list all v2 envelope keys;
  - explain that selected/played frequency remains outside patch JSON;
  - link the Milestone B design and acceptance record;
  - move only the sine-only Gym to Milestone C roadmap language;
  - retain native run and exact test/lint commands.

  In `docs/project-notebook.md`, update the date and add a concise milestone record: Milestone B is patch authoring infrastructure for reproducible sources, not an RL synth-control task; the first retuning actor still manipulates immutable audio in Milestone C.

  The README schema statement must be semantically equivalent to:

  ```text
  Harpy reads strict schema-v1 linear patches and strict schema-v2 curve-enabled patches.
  It writes canonical schema v2. A patch contains oscillator, ADSR/curve configuration,
  and output gain; selected or played frequency remains performance state outside JSON.
  ```

- [ ] **Step 8: Write the evidence document from observed facts**

  Mirror the Milestone A evidence structure: environment, automated table, structural table, native observations, screenshot evidence, pending manual checks, and exact commit range. Never claim human audibility, 100% scale, complete-frame visuals, or real device loss/recovery unless actually observed.

  Use these exact headings:

  ```markdown
  # Milestone B acceptance evidence
  ## Acceptance status
  ## Automated evidence
  ## Structural evidence
  ## Native WSLg observations
  ## Screenshot evidence
  ## Pending manual and visual checks
  ```

- [ ] **Step 9: Verify documentation and commit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check
  git status --short
  git add README.md docs/project-notebook.md docs/verification/2026-08-08-milestone-b-acceptance.md docs/verification/assets
  git commit -m "docs: record milestone B envelope acceptance"
  ```

  If no screenshot assets were produced, omit `docs/verification/assets` from `git add`. After commit, require a clean worktree and rerun `git diff --check origin/main...HEAD`.

---

## Final completion conditions

Milestone B is complete only when:

1. every task commit passed focused tests, the full suite, Ruff lint, Ruff format, and diff checks;
2. strict v1 documents migrate to a schema-free runtime patch and canonical v2 saves;
3. zero curves preserve the exact Milestone A envelope output and nonzero curves are deterministic and monotone;
4. current held/releasing audio is never changed or truncated by editor commits;
5. pending intent coalesces and the next Play auditions only the final applied patch;
6. numeric and graphical editors share immutable patch truth and the DSP curve implementation;
7. the knob preserves all tuning mechanics while meeting the approved orientation, grouping, and discoverability contract;
8. native layout, error, patch-file, failure/recovery, and shutdown behavior remains truthful and atomic;
9. production contains no runtime v1 alias, duplicate curve formula, dead facts row, Qt-bound core code, or Milestone C implementation;
10. documentation distinguishes automated proof, native observation, human judgment, and pending checks.

After these conditions pass, stop. Do not start Milestone C implementation. Return to brainstorming for the sine-only Gym's action, observation, reward, leakage, model-adapter, and benchmark design.
