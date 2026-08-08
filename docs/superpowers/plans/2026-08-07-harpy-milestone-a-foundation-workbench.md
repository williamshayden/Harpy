# Harpy Milestone A Foundation Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current native sine lab with a direct-hertz synth/analysis API and a clean native workbench featuring a C2–C4 rotary pitch control, exact Hz entry, accurate waveform/spectrum measurements, and stateless strict patch JSON load/save.

**Architecture:** Build the non-Qt contracts first: immutable patch/render models, tuning conversion, a deterministic monophonic synth engine, pure analysis, and generation-safe capture. Add a semantic workbench controller and one Qt audio adapter at the platform edge. The native window consumes those contracts; it never becomes the API. Keep the working legacy surface available while isolated replacement modules are developed, then switch composition once and delete every superseded model, module, test, and compatibility path.

**Tech Stack:** Python 3.12, NumPy, PySide6/Qt Multimedia, pyqtgraph, uv, pytest, pytest-qt, Ruff.

## Global Constraints

- Work only in `/home/haydenw/Projects/Harpy/.worktrees/native-sine-lab` on `william/native-sine-lab`.
- Treat `docs/superpowers/specs/2026-08-07-harpy-milestone-a-foundation-workbench-design.md` as the approved product contract. This plan chooses implementation boundaries; it does not reopen product scope.
- Keep Milestone A sine-only and monophonic. Do not add ADSR editing, Bézier curves, waveform morphing, polyphony, randomized patches, Gymnasium, model adapters, storage, MIDI input, JUCE, or C++.
- Keep the engine frequency domain broad: finite, positive hertz strictly below Nyquist. Only the native knob/editor are bounded to tuned C2–C4.
- Keep `SynthPatch` independent from played frequency and gate state. JSON serializes only the patch.
- Keep the authoritative renderer mono `float32` at 48 kHz/256-frame defaults. Device channel/sample conversion stays in the Qt adapter.
- Preserve the proven pull-device behavior: sequential `QIODevice`, positive `bytesAvailable()`, existing format-probe order, PCM conversion, hotplug teardown, and no tight retry loop.
- A successful audio path has no status copy. Surface only a deduplicated actionable failure and disable Play until recovery.
- Tests are part of the public research artifact. Delete tests only when they assert explicitly superseded behavior, and replace them in the same integration commit.
- Do not leave re-exports, aliases, `V2` classes, dead files, or parallel legacy/new composition after Task 9.
- Every implementation task follows red → green → full verification → commit. Before each commit run:

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  ```

- Expected full verification result at every green boundary: all tests pass, Ruff reports no errors, and formatting check exits 0.

---

## Task 1: Introduce immutable synth models and the strict v1 patch codec

**Files:**

- Create: `src/harpy/synth/models.py`
- Create: `src/harpy/synth/patch_json.py`
- Create: `tests/synth/test_models.py`
- Create: `tests/synth/test_patch_json.py`
- Keep temporarily: `src/harpy/synth/specs.py` and its consumers; Task 9 removes them.

- [ ] **Step 1: Write failing model tests**

  Add tests that establish this exact public surface:

  ```python
  from dataclasses import replace

  import pytest

  from harpy.synth.models import (
      EnvelopeConfig,
      OscillatorConfig,
      OscillatorType,
      RenderConfig,
      SynthPatch,
      seconds_to_frames,
      validate_renderable_patch,
  )


  def test_default_patch_is_the_approved_linear_sine_patch() -> None:
      patch = SynthPatch()
      assert patch.oscillator == OscillatorConfig(type=OscillatorType.SINE)
      assert patch.envelope == EnvelopeConfig(
          attack_seconds=0.001,
          decay_seconds=0.600,
          sustain_db=-6.0,
          release_seconds=0.600,
          curve="linear_amplitude",
      )
      assert patch.output_gain_dbfs == -12.0


  def test_default_render_config_is_authoritative_mono_float32() -> None:
      assert RenderConfig() == RenderConfig(
          sample_rate_hz=48_000,
          block_frames=256,
          channels=1,
          internal_dtype="float32",
      )


  @pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
  def test_envelope_durations_must_be_positive_and_finite(value: float) -> None:
      with pytest.raises(ValueError):
          EnvelopeConfig(attack_seconds=value)


  def test_composed_patch_requires_at_least_one_frame_per_segment() -> None:
      patch = replace(SynthPatch(), envelope=EnvelopeConfig(attack_seconds=0.000001))
      with pytest.raises(ValueError, match="attack"):
          validate_renderable_patch(patch, RenderConfig())


  def test_seconds_to_frames_uses_half_up_rounding() -> None:
      assert seconds_to_frames(2.5 / 48_000, 48_000) == 3
  ```

  Model rules:

  - `OscillatorType(StrEnum)` contains only `SINE = "sine"`.
  - `OscillatorConfig.type` defaults to `OscillatorType.SINE` and rejects strings/enums not in that closed set.
  - `EnvelopeConfig` durations are finite and strictly positive; sustain is finite and `<= 0 dB`; curve is exactly `"linear_amplitude"`.
  - `SynthPatch.output_gain_dbfs` is finite and `<= 0 dBFS`; `output_gain` converts dBFS to linear amplitude.
  - `RenderConfig` validates positive integer sample rate/block size, mono channel count, and `float32` internal dtype. Booleans are not integers for validation.
  - `validate_renderable_patch()` uses half-up frame conversion and names the invalid segment.

- [ ] **Step 2: Run the focused model tests and prove they fail**

  ```bash
  uv run pytest tests/synth/test_models.py -q
  ```

  Expected: collection fails because `harpy.synth.models` does not exist.

- [ ] **Step 3: Implement the minimal immutable models**

  Use frozen, slotted dataclasses and these signatures:

  ```python
  class OscillatorType(StrEnum):
      SINE = "sine"


  @dataclass(frozen=True, slots=True)
  class OscillatorConfig:
      type: OscillatorType = OscillatorType.SINE


  @dataclass(frozen=True, slots=True)
  class EnvelopeConfig:
      attack_seconds: float = 0.001
      decay_seconds: float = 0.600
      sustain_db: float = -6.0
      release_seconds: float = 0.600
      curve: str = "linear_amplitude"

      @property
      def sustain_amplitude(self) -> float: ...


  @dataclass(frozen=True, slots=True)
  class SynthPatch:
      oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
      envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
      output_gain_dbfs: float = -12.0

      @property
      def output_gain(self) -> float: ...


  @dataclass(frozen=True, slots=True)
  class RenderConfig:
      sample_rate_hz: int = 48_000
      block_frames: int = 256
      channels: int = 1
      internal_dtype: str = "float32"


  def seconds_to_frames(seconds: float, sample_rate_hz: int) -> int: ...
  def validate_renderable_patch(patch: SynthPatch, render: RenderConfig) -> None: ...
  ```

  In the implementation, replace each `...` above with the validated calculation described by the tests; no placeholder remains in source.

- [ ] **Step 4: Write failing strict-codec tests**

  Establish these functions:

  ```python
  def loads_patch(text: str) -> SynthPatch: ...
  def dumps_patch(patch: SynthPatch) -> str: ...
  def load_patch(path: Path) -> SynthPatch: ...
  def save_patch(path: Path, patch: SynthPatch) -> None: ...
  ```

  Add an exact canonical-output assertion:

  ```python
  EXPECTED_DEFAULT = """{
    "schema_version": 1,
    "oscillator": {
      "type": "sine"
    },
    "envelope": {
      "attack_seconds": 0.001,
      "decay_seconds": 0.6,
      "sustain_db": -6.0,
      "release_seconds": 0.6,
      "curve": "linear_amplitude"
    },
    "output_gain_dbfs": -12.0
  }
  """


  def test_default_patch_has_canonical_json() -> None:
      assert dumps_patch(SynthPatch()) == EXPECTED_DEFAULT


  def test_patch_json_round_trip_preserves_values() -> None:
      patch = SynthPatch(
          envelope=EnvelopeConfig(attack_seconds=0.125, sustain_db=-9.5),
          output_gain_dbfs=-18.25,
      )
      assert loads_patch(dumps_patch(patch)) == patch
  ```

  Parametrize rejection for unknown keys, missing keys, schema versions other than the
  integer `1`, duplicate keys, booleans used as numbers, `NaN`, `Infinity`,
  `-Infinity`, unsupported oscillator/curve values, and trailing non-whitespace
  content. Assert the exception text names the failing field when a field exists. Add
  a positive test that leading and trailing JSON whitespace are accepted.

- [ ] **Step 5: Run the codec tests and prove they fail**

  ```bash
  uv run pytest tests/synth/test_patch_json.py -q
  ```

  Expected: collection fails because `harpy.synth.patch_json` does not exist.

- [ ] **Step 6: Implement the strict codec and file wrappers**

  Parse with `json.JSONDecoder(object_pairs_hook=...)`, where the hook raises on a
  duplicate before converting ordered pairs to a dict. Pass `parse_constant` that
  always raises. Find the first non-whitespace index before calling `raw_decode`
  (`raw_decode` does not skip leading whitespace), then reject any non-whitespace
  suffix. Validate exact key sets at every object level before constructing the
  complete immutable patch. Use `json.dumps(..., indent=2, allow_nan=False)` over an
  insertion-ordered mapping plus one final newline. Read and write UTF-8 only.

- [ ] **Step 7: Verify Task 1 and commit**

  ```bash
  uv run pytest tests/synth/test_models.py tests/synth/test_patch_json.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/synth/models.py src/harpy/synth/patch_json.py tests/synth/test_models.py tests/synth/test_patch_json.py
  git commit -m "feat: define synth patch contracts"
  ```

---

## Task 2: Add hertz-first tuning and the runtime C2–C4 workbench range

**Files:**

- Create: `src/harpy/tuning.py`
- Create: `src/harpy/gui/workbench_spec.py`
- Create: `tests/test_tuning.py`
- Create: `tests/gui/test_workbench_spec.py`
- Keep temporarily: `src/harpy/pitch.py`, `src/harpy/note_names.py`, and `src/harpy/gui/specs.py`; Task 9 removes them.

- [ ] **Step 1: Write failing hertz-first conversion tests**

  ```python
  import pytest

  from harpy.tuning import NoteReading, Tuning


  def test_default_tuning_round_trips_reference_frequency() -> None:
      tuning = Tuning()
      assert tuning.frequency_hz_for_midi_coordinate(69.0) == 440.0
      assert tuning.midi_coordinate_for_frequency_hz(440.0) == 69.0


  @pytest.mark.parametrize(
      ("frequency_hz", "expected"),
      [
          (261.6255653005986, NoteReading("C3", 0.0)),
          (440.0, NoteReading("A3", 0.0)),
          (440.0 * 2.0 ** (25.0 / 1200.0), NoteReading("A3", 25.0)),
      ],
  )
  def test_frequency_reading_uses_ableton_octaves(
      frequency_hz: float,
      expected: NoteReading,
  ) -> None:
      reading = Tuning().describe_frequency(frequency_hz)
      assert reading.name == expected.name
      assert reading.cents == pytest.approx(expected.cents, abs=1e-9)


  @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
  def test_frequency_conversion_rejects_nonphysical_hertz(value: float) -> None:
      with pytest.raises(ValueError):
          Tuning().midi_coordinate_for_frequency_hz(value)
  ```

- [ ] **Step 2: Run the focused tests and prove they fail**

  ```bash
  uv run pytest tests/test_tuning.py -q
  ```

  Expected: collection fails because `harpy.tuning` is absent.

- [ ] **Step 3: Implement the pure music boundary**

  Use this API; the continuous MIDI coordinate is an internal conversion coordinate, not the engine's pitch representation and not primary UI state:

  ```python
  @dataclass(frozen=True, slots=True)
  class NoteReading:
      name: str
      cents: float


  @dataclass(frozen=True, slots=True)
  class Tuning:
      reference_hz: float = 440.0
      reference_midi_coordinate: float = 69.0
      middle_c_octave: int = 3

      def frequency_hz_for_midi_coordinate(self, coordinate: float) -> float: ...
      def midi_coordinate_for_frequency_hz(self, frequency_hz: float) -> float: ...
      def describe_frequency(self, frequency_hz: float) -> NoteReading: ...
  ```

  Validate a positive finite reference hertz value, a finite reference coordinate, and
  a non-boolean integer octave convention. Nearest-note rounding is exactly
  `math.floor(coordinate + 0.5)`, including half-semitone ties and negative
  coordinates; cents are `100 * (coordinate - nearest_integer)`. Note names use
  Unicode sharps and MIDI coordinate 60 displays as C3.

- [ ] **Step 4: Run the tuning tests green**

  ```bash
  uv run pytest tests/test_tuning.py -q
  ```

- [ ] **Step 5: Write the failing GUI-only range test**

  ```python
  @dataclass(frozen=True, slots=True)
  class WorkbenchSpec:
      minimum_frequency_hz: float
      center_frequency_hz: float
      maximum_frequency_hz: float

      @classmethod
      def from_tuning(cls, tuning: Tuning) -> WorkbenchSpec:
          return cls(
              minimum_frequency_hz=tuning.frequency_hz_for_midi_coordinate(48.0),
              center_frequency_hz=tuning.frequency_hz_for_midi_coordinate(60.0),
              maximum_frequency_hz=tuning.frequency_hz_for_midi_coordinate(72.0),
          )
  ```

  Assert C3 is the logarithmic midpoint:

  ```python
  def test_workbench_range_is_runtime_tuned_c2_c3_c4() -> None:
      spec = WorkbenchSpec.from_tuning(Tuning(reference_hz=442.0))
      assert spec.center_frequency_hz**2 == pytest.approx(
          spec.minimum_frequency_hz * spec.maximum_frequency_hz,
          rel=1e-12,
      )
      assert Tuning(reference_hz=442.0).describe_frequency(spec.minimum_frequency_hz).name == "C2"
      assert Tuning(reference_hz=442.0).describe_frequency(spec.center_frequency_hz).name == "C3"
      assert Tuning(reference_hz=442.0).describe_frequency(spec.maximum_frequency_hz).name == "C4"
  ```

- [ ] **Step 6: Run the workbench-range test and prove it fails**

  ```bash
  uv run pytest tests/gui/test_workbench_spec.py -q
  ```

  Expected: collection fails because `harpy.gui.workbench_spec` is absent.

- [ ] **Step 7: Implement `WorkbenchSpec` with the exact runtime conversion above**

  Keep this GUI preference out of `Tuning`, `SynthEngine`, `SynthPatch`, and the
  eventual core `AppConfig`.

- [ ] **Step 8: Verify Task 2 and commit**

  ```bash
  uv run pytest tests/test_tuning.py tests/gui/test_workbench_spec.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/tuning.py src/harpy/gui/workbench_spec.py tests/test_tuning.py tests/gui/test_workbench_spec.py
  git commit -m "feat: add hertz-first tuning model"
  ```

---

## Task 3: Implement the deterministic direct-hertz synth engine

**Files:**

- Create: `src/harpy/synth/engine.py`
- Modify: `src/harpy/synth/envelope.py`
- Modify: `tests/synth/test_envelope.py`
- Create: `tests/synth/test_engine.py`
- Keep temporarily: `src/harpy/synth/reference.py`; Task 9 removes it.

- [ ] **Step 1: Migrate envelope tests to `EnvelopeConfig` and pin sample boundaries**

  Preserve the existing update-before-emission convention with exact endpoints:

  ```python
  def test_attack_decay_and_sustain_emit_exact_endpoints() -> None:
      envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
      envelope.note_on()
      np.testing.assert_allclose(envelope.render(4), [0.25, 0.5, 0.75, 1.0])
      assert envelope.stage is EnvelopeStage.DECAY
      np.testing.assert_allclose(envelope.render(4), [0.875, 0.75, 0.625, 0.5])
      assert envelope.stage is EnvelopeStage.SUSTAIN
  ```

  Replace only the imported immutable type. Do not change the proven envelope state machine beyond the type migration.

- [ ] **Step 2: Write failing direct-hertz engine tests**

  Required API:

  ```python
  class SynthEngine:
      def __init__(self, render: RenderConfig, patch: SynthPatch) -> None: ...
      @property
      def is_idle(self) -> bool: ...
      @property
      def patch(self) -> SynthPatch: ...
      def note_on(self, frequency_hz: float) -> None: ...
      def retune(self, frequency_hz: float) -> None: ...
      def note_off(self) -> None: ...
      def replace_patch(self, patch: SynthPatch) -> None: ...
      def render(self, frame_count: int) -> np.ndarray: ...
      def reset(self) -> None: ...
  ```

  Add these behavioral tests:

  ```python
  def test_idle_engine_emits_exact_mono_float32_silence() -> None:
      samples = SynthEngine(RenderConfig(), SynthPatch()).render(256)
      assert samples.shape == (256,)
      assert samples.dtype == np.float32
      np.testing.assert_array_equal(samples, np.zeros(256, dtype=np.float32))


  def test_live_retune_preserves_phase_and_envelope() -> None:
      retuned = SynthEngine(RenderConfig(), SynthPatch())
      control = SynthEngine(RenderConfig(), SynthPatch())
      retuned.note_on(220.0)
      control.note_on(220.0)
      np.testing.assert_array_equal(retuned.render(1_000), control.render(1_000))
      retuned.retune(330.0)
      np.testing.assert_allclose(retuned.render(1), control.render(1), rtol=0.0, atol=1e-7)
      assert not np.array_equal(retuned.render(64), control.render(64))


  def test_render_is_block_partition_invariant() -> None:
      whole = SynthEngine(RenderConfig(), SynthPatch())
      chunked = SynthEngine(RenderConfig(), SynthPatch())
      whole.note_on(261.6255653005986)
      chunked.note_on(261.6255653005986)
      expected = whole.render(10_000)
      actual = np.concatenate(
          [chunked.render(1), chunked.render(255), chunked.render(4_096), chunked.render(5_648)]
      )
      np.testing.assert_allclose(expected, actual, rtol=0.0, atol=1e-6)
  ```

  Also test:

  - note-on and retune reject `NaN`, infinities, zero, negatives, Nyquist, and above-Nyquist without mutating prior state;
  - retune raises a clear state error while idle and works during release;
  - note-on resets phase and retriggers from silence;
  - note-off releases from the last emitted envelope level and reaches exact silence after the declared release frames;
  - repeated note-off is idempotent;
  - replace-patch validates renderability first, then atomically swaps patch and resets to exact silence;
  - reset clears phase, increment, and envelope immediately;
  - render rejects booleans, non-integers, and negative frame counts.

- [ ] **Step 3: Run the engine tests and prove they fail**

  ```bash
  uv run pytest tests/synth/test_envelope.py tests/synth/test_engine.py -q
  ```

  Expected: `test_engine.py` fails at import; migrated envelope tests fail until the implementation accepts `EnvelopeConfig`.

- [ ] **Step 4: Implement the engine with one authoritative render path**

  Port the proven oscillator math from `SineVoice`, but remove all tuning/Pitch imports. Validate hertz before mutation. Use a float64 phase accumulator and intermediate oscillator/envelope math, wrap with `math.fmod`, multiply by `patch.output_gain`, and return a new contiguous `float32` mono array. The engine name stays generic while its only supported `OscillatorType` is sine.

- [ ] **Step 5: Verify Task 3 and commit**

  ```bash
  uv run pytest tests/synth/test_envelope.py tests/synth/test_engine.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/synth/engine.py src/harpy/synth/envelope.py tests/synth/test_envelope.py tests/synth/test_engine.py
  git commit -m "feat: add direct hertz synth engine"
  ```

---

## Task 4: Build the pure waveform and spectrum observation API

**Files:**

- Create: `src/harpy/analysis.py`
- Create: `tests/test_analysis.py`
- Keep temporarily: `src/harpy/gui/visualizer.py`; Task 9 removes it.

- [ ] **Step 1: Write failing validation and empty-observation tests**

  Define this immutable contract:

  ```python
  @dataclass(frozen=True, slots=True)
  class AnalysisConfig:
      waveform_window_seconds: float = 0.050
      fft_frames: int = 16_384
      spectrum_min_hz: float = 20.0
      spectrum_max_hz: float = 20_000.0
      spectrum_floor_dbfs: float = -120.0
      window: str = "hann"


  @dataclass(frozen=True, slots=True)
  class AudioObservation:
      has_signal: bool
      waveform_samples: np.ndarray
      waveform_time_ms: np.ndarray
      spectrum_frequency_hz: np.ndarray
      spectrum_level_dbfs: np.ndarray
      peak_amplitude_fs: float | None
      peak_frequency_hz: float | None
      peak_level_dbfs: float | None


  def validate_analysis_config(config: AnalysisConfig, sample_rate_hz: int) -> None: ...


  def analyze(
      samples: np.ndarray,
      sample_rate_hz: int,
      config: AnalysisConfig = AnalysisConfig(),
  ) -> AudioObservation: ...
  ```

  Tests must prove:

  ```python
  def test_analysis_rejects_short_fft_input() -> None:
      with pytest.raises(ValueError, match="16384"):
          analyze(np.zeros(16_383, dtype=np.float32), 48_000)


  @pytest.mark.parametrize("level", [0.0, 1e-6])
  def test_silent_tail_returns_owned_empty_observation(level: float) -> None:
      samples = np.zeros(16_384, dtype=np.float32)
      samples[-2_400:] = level
      result = analyze(samples, 48_000)
      assert not result.has_signal
      assert result.peak_amplitude_fs is None
      assert result.peak_frequency_hz is None
      assert result.peak_level_dbfs is None
      assert all(
          array.ndim == 1 and array.size == 0 and not array.flags.writeable and array.flags.owndata
          for array in (
              result.waveform_samples,
              result.waveform_time_ms,
              result.spectrum_frequency_hz,
              result.spectrum_level_dbfs,
          )
      )
  ```

  Include config composition failures through `validate_analysis_config()`: odd/nonpositive
  FFT size, waveform frames outside `1..fft_frames`, invalid bounds or Nyquist,
  non-finite/non-positive sample rate, non-finite positive dB floor, and window other
  than `hann`. Add one test proving `analyze()` calls the same validator rather than
  duplicating or silently clamping these rules.

- [ ] **Step 2: Run focused tests and prove they fail**

  ```bash
  uv run pytest tests/test_analysis.py -q
  ```

  Expected: collection fails because `harpy.analysis` does not exist.

- [ ] **Step 3: Add deterministic waveform tests**

  Verify half-up duration conversion gives exactly 2,400 frames at 48 kHz; timestamps equal `arange(2400) * 1000 / 48000`; samples retain full-scale amplitude; and the view starts after the most recent nonpositive-to-positive crossing with a complete 2,400-frame suffix. If no eligible crossing exists, it uses the trailing 2,400 frames.

- [ ] **Step 4: Add calibrated spectrum tests**

  For a bin-centered sustained sine, assert the dBFS peak using Hann coherent-gain correction within `0.02 dB`. Assert output frequencies are strictly increasing, positive, and within 20–20,000 Hz. Characterize the approved acceptance surface exactly:

  ```python
  @pytest.mark.parametrize("frequency_hz", np.linspace(130.8127826502993, 523.2511306011972, 1_001))
  @pytest.mark.parametrize("phase", np.linspace(0.0, math.tau, 9, endpoint=False))
  def test_clean_sine_peak_is_within_tenth_hz(frequency_hz: float, phase: float) -> None:
      frames = np.arange(16_384, dtype=np.float64)
      samples = (0.25 * np.sin(math.tau * frequency_hz * frames / 48_000 + phase)).astype(np.float32)
      result = analyze(samples, 48_000)
      assert result.peak_frequency_hz is not None
      assert abs(result.peak_frequency_hz - frequency_hz) <= 0.1
  ```

  The 1,001 frequencies × 9 phases are the approved 9,009-case characterization.
  Also assert the measured peak is derived from samples, not any configured synth
  frequency. Define “no in-range spectral peak” precisely: when every raw in-range
  one-sided amplitude is at or below the configured dBFS-floor amplitude, keep
  `has_signal=True` but return `None` for both spectral peak fields.

- [ ] **Step 5: Implement pure analysis**

  Required numerical rules:

  - Consume the most recent exact `fft_frames`; never zero-pad.
  - Determine `has_signal` from the untriggered trailing waveform slice using `peak > 1e-6`.
  - Use `np.hanning(fft_frames)` and one-sided amplitude `2 * abs(rfft) / sum(window)`; zero Hz is excluded. Do not double the Nyquist bin.
  - Call `validate_analysis_config(config, sample_rate_hz)` before inspecting samples.
  - Clamp only for logarithm display using the configured floor; do not normalize captures.
  - Find the in-range winning bin. If both neighbors exist, fit a quadratic to its three dB values and use the vertex for peak frequency and level; otherwise use the bin center.
  - `AudioObservation.__post_init__` copies every supplied public array, forces
    C-contiguous one-dimensional shape, validates paired lengths, and sets
    `write=False`; callers cannot bypass immutability by constructing the dataclass
    directly.

- [ ] **Step 6: Verify Task 4 and commit**

  ```bash
  uv run pytest tests/test_analysis.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/analysis.py tests/test_analysis.py
  git commit -m "feat: add calibrated audio observations"
  ```

---

## Task 5: Add generation-safe bounded history and live capture state

**Files:**

- Create: `src/harpy/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write failing history tests**

  Establish these types:

  ```python
  @dataclass(frozen=True, slots=True)
  class HistorySnapshot:
      generation: int
      samples: np.ndarray


  class SampleHistory:
      def __init__(self, capacity_frames: int) -> None: ...
      @property
      def generation(self) -> int: ...
      @property
      def available_frames(self) -> int: ...
      def begin_generation(self) -> int: ...
      def append(self, samples: np.ndarray, generation: int) -> bool: ...
      def snapshot_recent(self, frame_count: int) -> HistorySnapshot: ...
  ```

  Assert that snapshots return only actually available samples with no left padding, capacity keeps the newest frames, public snapshots own read-only arrays, and an append tagged with a superseded generation returns `False` without changing history.

- [ ] **Step 2: Write the state-machine tests**

  ```python
  class CaptureState(StrEnum):
      EMPTY = "empty"
      MEASURING = "measuring"
      LIVE = "live"
      CAPTURED = "captured"


  @dataclass(frozen=True, slots=True)
  class CaptureView:
      state: CaptureState
      observation: AudioObservation | None
      generation: int


  class CaptureCoordinator:
      def __init__(
          self,
          history: SampleHistory,
          sample_rate_hz: int,
          analysis_config: AnalysisConfig,
          analyzer: Callable[..., AudioObservation] = analyze,
      ) -> None: ...
      def begin(self) -> int: ...
      def clear(self, voice_active: bool) -> int: ...
      def reset(self) -> int: ...
      def refresh(self) -> CaptureView: ...
  ```

  Cover every transition table row from the approved spec:

  - launch/reset → Empty;
  - begin/note-on → Measuring;
  - insufficient frames remain Measuring;
  - first valid observation → Live;
  - later valid observation refreshes Live;
  - silence before any Live → Empty;
  - silence after Live → Captured while retaining the last valid object;
  - clear idle → Empty/new generation;
  - clear active → Measuring/new generation;
  - reset/patch/device failure → Empty/new generation.

  For every begin, clear, and reset event, assert the history generation increments
  exactly once. Queue two semantic events before any simulated audio drain and assert
  an append tagged with the first generation is rejected while an append tagged with
  the second succeeds.

  For stale analysis, inject an analyzer that calls `coordinator.clear(False)` before returning. `refresh()` must reject that result because its private captured generation no longer matches.

- [ ] **Step 3: Run focused tests and prove they fail**

  ```bash
  uv run pytest tests/test_capture.py -q
  ```

  Expected: collection fails because `harpy.capture` does not exist.

- [ ] **Step 4: Implement capture without Qt or DSP dependencies**

  Protect all history mutation/snapshot operations with one lock, but copy samples and
  run FFT analysis after releasing it. `SampleHistory.begin_generation()` is the sole
  generation allocator and clearer: each `CaptureCoordinator.begin()`, `clear()`, or
  `reset()` calls it exactly once and returns that identifier. No audio adapter method
  may increment or install history generations. `CaptureCoordinator` keeps generation
  metadata in a private `CapturedObservation`, never inside `AudioObservation`. It
  calls the pure analyzer only when at least `fft_frames` are available.

- [ ] **Step 5: Verify Task 5 and commit**

  ```bash
  uv run pytest tests/test_capture.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/capture.py tests/test_capture.py
  git commit -m "feat: add generation safe live capture"
  ```

---

## Task 6: Define playback commands and the semantic workbench controller

**Files:**

- Create: `src/harpy/playback.py`
- Create: `src/harpy/gui/workbench_controller.py`
- Create: `tests/test_playback.py`
- Create: `tests/gui/test_workbench_controller.py`
- Keep temporarily: `src/harpy/gui/controller.py`; Task 9 removes it.

- [ ] **Step 1: Write failing command-contract tests**

  ```python
  class AudioCommandKind(StrEnum):
      NOTE_ON = "note_on"
      RETUNE = "retune"
      NOTE_OFF = "note_off"
      CLEAR_CAPTURE = "clear_capture"
      REPLACE_PATCH = "replace_patch"
      RESET = "reset"


  @dataclass(frozen=True, slots=True)
  class AudioCommand:
      kind: AudioCommandKind
      frequency_hz: float | None = None
      patch: SynthPatch | None = None
      generation: int | None = None
  ```

  Validate exact payload shapes:

  - NOTE_ON requires frequency and generation;
  - RETUNE requires only frequency;
  - NOTE_OFF requires no payload;
  - CLEAR_CAPTURE requires only generation;
  - REPLACE_PATCH requires patch and generation;
  - RESET requires generation;
  - booleans, negative generations, and extra payloads fail construction.

- [ ] **Step 2: Write failing controller tests**

  Required state/API:

  ```python
  @dataclass(frozen=True, slots=True)
  class WorkbenchState:
      selected_frequency_hz: float
      gate_held: bool
      voice_may_be_active: bool
      patch: SynthPatch
      capture: CaptureView
      audio_available: bool
      audio_error: str | None


  class WorkbenchController:
      @property
      def state(self) -> WorkbenchState: ...
      def set_frequency(self, frequency_hz: float) -> WorkbenchState: ...
      def press_play(self) -> WorkbenchState: ...
      def release_play(self) -> WorkbenchState: ...
      def clear_measurement(self) -> WorkbenchState: ...
      def replace_patch(self, patch: SynthPatch) -> WorkbenchState: ...
      def refresh_capture(self) -> WorkbenchState: ...
      def mark_voice_idle(self, generation: int) -> WorkbenchState: ...
      def force_stop(self) -> WorkbenchState: ...
      def set_audio_availability(
          self,
          available: bool,
          error: str | None = None,
      ) -> WorkbenchState: ...
  ```

  Construct it with `WorkbenchSpec`, `RenderConfig`, initial `SynthPatch`,
  `CaptureCoordinator`, and `Callable[[AudioCommand], None]`. The render config is
  required so `replace_patch()` can call `validate_renderable_patch()` before any
  controller, capture, or queued-command mutation.

  Pin these sequences in tests:

  ```python
  def test_frequency_change_retunes_without_retrigger_while_held() -> None:
      controller, commands = make_controller()
      controller.press_play()
      controller.set_frequency(330.0)
      assert [command.kind for command in commands] == [
          AudioCommandKind.NOTE_ON,
          AudioCommandKind.RETUNE,
      ]
      assert commands[-1].frequency_hz == 330.0


  def test_replace_patch_is_one_atomic_audio_command() -> None:
      controller, commands = make_controller()
      controller.press_play()
      replacement = SynthPatch(output_gain_dbfs=-18.0)
      state = controller.replace_patch(replacement)
      assert commands[-1].kind is AudioCommandKind.REPLACE_PATCH
      assert commands[-1].patch == replacement
      assert state.patch == replacement
      assert not state.gate_held
      assert not state.voice_may_be_active
      assert state.capture.state is CaptureState.EMPTY
  ```

  Also test initial tuned C3, finite/range validation, idle edits sending no command,
  press/release idempotence, retune during release, and the backend idle callback:
  after `mark_voice_idle(current_generation)`, an edit sends no RETUNE and Clear enters
  Empty rather than Measuring. Retrigger to a newer generation, deliver a delayed idle
  callback for the prior generation, and assert it is ignored: the new voice remains
  active and frequency edits still RETUNE. Test Clear active/idle generations,
  including active Clear updating the controller's private voice generation: reject an
  idle callback carrying the pre-Clear generation, accept the callback carrying the
  Clear generation after release, and then classify Clear as idle/Empty. Test
  force-stop idempotence, unavailable
  audio blocking NOTE_ON while patch inspection/editing remains usable, and recovery
  clearing the error without healthy status text. Exercise availability-false and
  force-stop in both callback orders and assert exactly one reset command/generation.
  A patch whose positive envelope
  duration rounds to zero frames must leave patch, capture, and command list byte-for-
  byte unchanged.

- [ ] **Step 3: Run focused tests and prove they fail**

  ```bash
  uv run pytest tests/test_playback.py tests/gui/test_workbench_controller.py -q
  ```

- [ ] **Step 4: Implement commands and controller state transitions**

  The controller stores semantic state only—no note/MIDI/frequency strings and no
  widgets. A selected-frequency edit queues RETUNE whenever `voice_may_be_active` is
  true. The audio backend's natural-release completion signal calls
  `mark_voice_idle(generation)`, which clears that flag without clearing a retained
  capture only when the generation matches the controller's current voice generation.
  The controller stores that generation privately when it issues NOTE_ON; it is not
  patch state or primary UI state. Clear while attack/decay/sustain/release is active
  also replaces this private token with the new CLEAR_CAPTURE generation, matching the
  source's adopted append/idle generation; idle Clear leaves the token empty.
  `replace_patch()` validates the complete patch against its `RenderConfig` before
  mutating controller state. `set_audio_availability()` changes availability/error
  state only; the backend's separate `force_stop_requested` notification calls the
  one idempotent `force_stop()` path that resets voice/capture. This prevents one
  device failure from allocating two generations regardless of signal order. The
  coordinator remains the sole allocator of command generations; the controller only
  places the returned identifier in the command.

- [ ] **Step 5: Verify Task 6 and commit**

  ```bash
  uv run pytest tests/test_playback.py tests/gui/test_workbench_controller.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/playback.py src/harpy/gui/workbench_controller.py tests/test_playback.py tests/gui/test_workbench_controller.py
  git commit -m "feat: add semantic workbench controller"
  ```

---

## Task 7: Build the replacement Qt audio adapter around the direct-hertz engine

**Files:**

- Modify: `src/harpy/gui/qt_audio.py`
- Create: `tests/gui/test_audio_backend.py`
- Keep the current legacy classes temporarily in the same platform-adapter module so
  the old app remains green; Task 9 removes only those legacy classes/tests. The final
  repository still has exactly one Qt adapter file: `gui/qt_audio.py`.

- [ ] **Step 1: Port the proven format/encoding tests to the replacement adapter**

  Preserve these public pure helpers and exact priority:

  ```python
  def audio_format_candidates(sample_rate_hz: int) -> tuple[QAudioFormat, ...]: ...
  def choose_audio_format(device: QAudioDevice, sample_rate_hz: int) -> QAudioFormat | None: ...
  def encode_mono_samples(samples: np.ndarray, audio_format: QAudioFormat) -> bytes: ...
  ```

  Candidate order remains stereo float, mono float, stereo int16, mono int16. Float stereo duplicates mono; int16 clips to `[-1, 1]`, rounds against 32767, and duplicates only when the negotiated format is stereo.

- [ ] **Step 2: Write failing source lifecycle tests**

  Use final names:

  ```python
  class SynthAudioSource(QIODevice):
      voice_idle = Signal(int)

      def __init__(
          self,
          render: RenderConfig,
          patch: SynthPatch,
          audio_format: QAudioFormat,
          history: SampleHistory,
          capture_generation: int,
      ) -> None: ...
      def submit(self, command: AudioCommand) -> None: ...


  class QtAudioBackend(QObject):
      availability_changed = Signal(bool)
      audio_failure = Signal(str)
      force_stop_requested = Signal()
      voice_idle = Signal(int)

      def __init__(
          self,
          render: RenderConfig,
          patch: SynthPatch,
          history: SampleHistory,
          *,
          media_devices: QObject | None = None,
          sink_factory: Callable[..., QAudioSink] = QAudioSink,
          parent: QObject | None = None,
      ) -> None: ...

      def start(self) -> None: ...
      def submit(self, command: AudioCommand) -> None: ...
      def shutdown(self) -> None: ...
  ```

  Pin these source behaviors:

  - `isSequential()` is true and `bytesAvailable()` is positive;
  - NOTE_ON adopts its already-reserved command generation as the source's local append
    tag, renders nonzero audio, and appends the same mono samples to history;
  - RETUNE changes the engine frequency without phase/envelope reset;
  - RETUNE after engine idle is ignored and does not become a backend failure;
  - NOTE_OFF releases normally;
  - CLEAR_CAPTURE changes only the source's local append generation;
  - RESET and REPLACE_PATCH adopt the supplied generation, discard any partially
    consumed staging bytes before returning further data, reset/replace the engine,
    and emit exact silence until another NOTE_ON;
  - the first block after REPLACE_PATCH is entirely silence or entirely the new patch—never a mixture.
  - two generation-bearing commands queued before a read leave the source on the newer
    generation; any delayed append tagged with the older one is rejected by history;
  - natural release completion emits `voice_idle(capture_generation)` exactly once;
    reset/replace and an initially idle renderer do not create a false
    natural-completion event;
  - a queued idle signal from an older generation arriving after retrigger is forwarded
    with its old identifier and cannot mark the newer controller voice idle.

- [ ] **Step 3: Write backend failure/hotplug tests**

  Reuse the current fake device/sink pattern. Assert:

  ```python
  def test_healthy_start_emits_availability_without_device_copy(qapp) -> None:
      backend, availability, failures = start_fake_backend()
      assert availability == [True]
      assert failures == []


  def test_repeated_sink_failure_is_reported_once_and_force_stops(qapp) -> None:
      backend, sink, forced_stops, failures = start_failing_backend()
      sink.stateChanged.emit(QAudio.State.StoppedState)
      sink.stateChanged.emit(QAudio.State.StoppedState)
      assert forced_stops == [True]
      assert len(failures) == 1
      assert "OpenError" in failures[0]
  ```

  Keep coverage for cross-binding enum values, default
  `QMediaDevices.defaultAudioOutput()`, null device, no supported format, output
  hotplug rebuilding once, DeferredDelete cleanup/disconnection, repeated shutdown,
  and successful recovery emitting `availability_changed(True)` without a status
  message. Every startup/runtime failure must emit `availability_changed(False)`;
  repeated notifications for one active failure remain deduplicated.

  Add an unavailable-device sequence: start with no source, submit REPLACE_PATCH and
  RESET, then recover/hotplug. Neither control command raises; the rebuilt source uses
  the replacement patch and newest generation and remains idle. NOTE_OFF and RETUNE are
  harmless no-ops without a source; NOTE_ON raises a clear unavailable error as a
  defensive invariant because the controller normally blocks it.

- [ ] **Step 4: Run focused tests and prove they fail**

  ```bash
  uv run pytest tests/gui/test_audio_backend.py -q
  ```

- [ ] **Step 5: Implement the adapter by extracting, not duplicating, proven mechanics**

  Refactor the already-tested device negotiation/PCM/lifecycle logic in
  `gui/qt_audio.py`, then add the replacement source around `SynthEngine`,
  `AudioCommand`, and `SampleHistory`. The coordinator/history is the sole generation
  owner: the source only stores the command's identifier as a local append tag and
  never calls `begin_generation()` or clears history itself. Commands are drained
  before serving staged bytes whenever RESET or REPLACE_PATCH is pending. Keep all
  engine mutation on the audio callback/source side.

  `QtAudioBackend` caches the latest validated patch and generation independently of a
  live source. REPLACE_PATCH/RESET/CLEAR_CAPTURE update that cache even while output is
  unavailable, and every rebuild constructs a new idle source from the cache. Forward
  the source's one-shot natural `voice_idle` signal. Deduplicate identical active
  failures until a successful rebuild resets the failure key.
  Connect `SynthAudioSource.voice_idle(int)` to the backend/controller with an explicit
  Qt queued connection so the audio callback never mutates controller or widget state
  and stale completion can be rejected by generation.

- [ ] **Step 6: Verify Task 7 and commit**

  ```bash
  uv run pytest tests/gui/test_audio_backend.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/gui/qt_audio.py tests/gui/test_audio_backend.py
  git commit -m "feat: adapt qt audio to synth engine"
  ```

---

## Task 8: Implement the native frequency controls, scientific plots, and patch-dialog port

**Files:**

- Create: `src/harpy/gui/frequency_knob.py`
- Create: `src/harpy/gui/frequency_entry.py`
- Create: `src/harpy/gui/signal_views.py`
- Create: `src/harpy/gui/patch_dialogs.py`
- Create: `tests/gui/test_frequency_knob.py`
- Create: `tests/gui/test_frequency_entry.py`
- Create: `tests/gui/test_signal_views.py`
- Create: `tests/gui/test_patch_dialogs.py`

- [ ] **Step 1: Write failing rotary-knob tests**

  Required API:

  ```python
  def frequency_to_unit(
      frequency_hz: float,
      minimum_hz: float,
      maximum_hz: float,
  ) -> float: ...


  class FrequencyKnob(QWidget):
      frequency_changed = Signal(float)

      def __init__(
          self,
          minimum_hz: float,
          center_hz: float,
          maximum_hz: float,
          *,
          cents_per_pixel: float = 12.0,
          parent: QWidget | None = None,
      ) -> None: ...

      @property
      def frequency_hz(self) -> float: ...
      def set_frequency_hz(self, frequency_hz: float, *, emit: bool = False) -> None: ...
  ```

  Test min/center/max map to normalized `0.0/0.5/1.0` in log2 space, and
  cover vertical relative drag without jump, Shift drag at one tenth sensitivity,
  Up/Right and wheel at +1 cent, Down/Left at -1 cent, Shift arrows at 0.1 cent,
  clamping without wrap, double-click restoring runtime C3, focus/accessible name,
  and programmatic no-emit updates. The default 12 cents/pixel must traverse the full
  two-octave range in exactly 200 upward pixels; Shift preserves fine control.

- [ ] **Step 2: Run the knob tests and prove they fail**

  ```bash
  uv run pytest tests/gui/test_frequency_knob.py -q
  ```

- [ ] **Step 3: Implement the purpose-built knob**

  Store hertz as the sole value. Convert interaction deltas with
  `new_hz = old_hz * 2 ** (cents / 1200)`. Paint a restrained endpoint-gap arc,
  ticks at C2/C3/C4, and a non-wrapping indicator. Do not subclass `QDial`: its
  integer range and absolute-angle behavior conflict with the approved interaction.

  ```bash
  uv run pytest tests/gui/test_frequency_knob.py -q
  ```

- [ ] **Step 4: Write failing exact-Hz entry tests**

  ```python
  class FrequencyEntry(QLineEdit):
      frequency_committed = Signal(float)
      validation_failed = Signal(str)

      def __init__(
          self,
          minimum_hz: float,
          maximum_hz: float,
          parent: QWidget | None = None,
      ) -> None: ...
      def set_frequency_hz(self, frequency_hz: float) -> None: ...
      def restore_last_valid(self) -> None: ...
  ```

  Enter accepts a plain finite decimal with zero through six fractional places and
  inclusive constructor bounds. Equal-tempered endpoints are generally irrational:
  using `WorkbenchSpec.from_tuning(Tuning(reference_hz=442.0))`, test manual entry at
  `ceil(minimum_hz * 1e6) / 1e6` and `floor(maximum_hz * 1e6) / 1e6`, plus the adjacent
  one-microhertz values outside the range. This prevents hard-coded A440 bounds without
  pretending the exact endpoint is typeable in six decimals. It emits the parsed
  float and re-renders exactly three decimals. Invalid syntax/range sets dynamic
  property `validationState="error"`,
  emits one field-level message, and does not emit frequency. Escape restores the last
  valid three-decimal string and clears the error. Reject exponent notation, signs
  without digits, NaN/Inf, and locale separators.

- [ ] **Step 5: Run the entry tests and prove they fail**

  ```bash
  uv run pytest tests/gui/test_frequency_entry.py -q
  ```

- [ ] **Step 6: Implement `FrequencyEntry` against its runtime bounds**

  Programmatic `set_frequency_hz()` and every successful manual commit validate the
  same inclusive bounds, then cache the exact float keyed to its rendered three-decimal
  text. Programmatic sets never emit a user-commit signal. If Enter is pressed while
  the text still equals the last rendered text, emit/retain the cached exact float
  instead of reparsing the rounded text. Test that exact runtime C2/C4 programmatic
  values survive this round trip, and that a valid six-decimal manual value near C2
  survives commit → three-decimal display → unchanged Enter without rejection or
  value drift.

  ```bash
  uv run pytest tests/gui/test_frequency_entry.py -q
  ```

- [ ] **Step 7: Write failing fixed scientific-view tests**

  Define `WaveformView.set_observation(AudioObservation | None)` and
  `SpectrumView.set_observation(AudioObservation | None)`. Assert:

  - Waveform fixed axes `0..50 ms`, `-1..1 FS`, labels `Time (ms)` and
    `Amplitude (FS)`, plus `Peak 0.251 FS` from the observation;
  - Spectrum logarithmic X `20..20_000 Hz`, fixed Y `-120..0 dBFS`, labels
    `Frequency (Hz)` and `Level (dBFS)`, with emphasized ticks exactly at
    `20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k`;
  - navigation, wheel zoom, and context menus are disabled;
  - `None` clears curves/marker/readouts and shows
    `Hold Play to inspect the signal.`;
  - a valid observation uses only observation arrays/peak fields;
  - Spectrum excludes nonpositive X data and places its marker/readout at measured
    peak, never at a selected/configured hertz value.

- [ ] **Step 8: Run the view tests and prove they fail**

  ```bash
  uv run pytest tests/gui/test_signal_views.py -q
  ```

- [ ] **Step 9: Implement the two observation-only views**

  Disable pyqtgraph mouse interaction and menus at both PlotItem and ViewBox levels.
  Empty views have no curve data item containing fabricated zeros/floor values.

  ```bash
  uv run pytest tests/gui/test_signal_views.py -q
  ```

- [ ] **Step 10: Write failing native patch-dialog tests**

  ```python
  class PatchDialogPort(Protocol):
      def choose_open_path(self, parent: QWidget) -> Path | None: ...
      def choose_save_path(self, parent: QWidget) -> Path | None: ...


  class NativePatchDialogs:
      def choose_open_path(self, parent: QWidget) -> Path | None: ...
      def choose_save_path(self, parent: QWidget) -> Path | None: ...
  ```

  Monkeypatch the two static QFileDialog functions and assert JSON filter/path
  conversion plus `None` on cancel; never open an interactive dialog in automation.

- [ ] **Step 11: Run the dialog tests and prove they fail**

  ```bash
  uv run pytest tests/gui/test_patch_dialogs.py -q
  ```

- [ ] **Step 12: Implement the stateless native dialog port**

  Default implementation calls `QFileDialog.getOpenFileName`/`getSaveFileName` and
  holds no remembered path, recent-file list, or storage state.

  ```bash
  uv run pytest tests/gui/test_patch_dialogs.py -q
  ```

- [ ] **Step 13: Run focused and full verification, then commit**

  ```bash
  uv run pytest tests/gui/test_frequency_knob.py tests/gui/test_frequency_entry.py tests/gui/test_signal_views.py tests/gui/test_patch_dialogs.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git add src/harpy/gui/frequency_knob.py src/harpy/gui/frequency_entry.py src/harpy/gui/signal_views.py src/harpy/gui/patch_dialogs.py tests/gui/test_frequency_knob.py tests/gui/test_frequency_entry.py tests/gui/test_signal_views.py tests/gui/test_patch_dialogs.py
  git commit -m "feat: add native workbench components"
  ```

---

## Task 9: Replace application composition and delete the legacy sine-lab surface

**Files:**

- Rewrite: `src/harpy/config.py`
- Rewrite: `src/harpy/gui/window.py`
- Rewrite: `src/harpy/gui/app.py`
- Rewrite: `src/harpy/gui/qt_audio.py` to retain only the replacement adapter classes
- Rewrite: `tests/test_config.py`
- Rewrite: `tests/gui/test_window.py`
- Rewrite: `tests/gui/test_app.py`
- Modify: `src/harpy/synth/__init__.py`
- Modify: `src/harpy/gui/__init__.py`
- Delete: `src/harpy/pitch.py`
- Delete: `src/harpy/note_names.py`
- Delete: `src/harpy/synth/specs.py`
- Delete: `src/harpy/synth/reference.py`
- Delete: `src/harpy/gui/specs.py`
- Delete: `src/harpy/gui/visualizer.py`
- Delete: `src/harpy/gui/controller.py`
- Delete/replace obsolete tests: `tests/test_pitch.py`, `tests/test_note_names.py`, `tests/synth/test_reference.py`, `tests/gui/test_controller.py`, `tests/gui/test_visualizer.py`, `tests/gui/test_qt_audio.py`

- [ ] **Step 1: Write the final core composition test**

  `AppConfig` may import core modules only:

  ```python
  @dataclass(frozen=True, slots=True)
  class AppConfig:
      tuning: Tuning = field(default_factory=Tuning)
      render: RenderConfig = field(default_factory=RenderConfig)
      analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
      patch: SynthPatch = field(default_factory=SynthPatch)


  DEFAULT_CONFIG = AppConfig()
  ```

  Its `__post_init__` calls `validate_renderable_patch()` and the Task 4
  `validate_analysis_config(self.analysis, self.render.sample_rate_hz)` function. Add
  tests for both invalid compositions and for `DEFAULT_CONFIG == AppConfig()`. Add an
  import-graph test that walks `harpy.config` direct imports and asserts none begin
  with `harpy.gui`. `WorkbenchSpec.from_tuning(config.tuning)` belongs in GUI
  composition, not `AppConfig`.

- [ ] **Step 2: Write the replacement window tests before rewriting it**

  The final widget contract is:

  ```text
  QMainWindow title: Harpy
  FrequencyKnob: frequencyKnob
  FrequencyEntry: frequencyEntry
  QLabel: derivedPitchLabel
  QPushButton: playButton
  QLabel: measurementStateLabel
  QPushButton: clearButton
  WaveformView: waveformPlot
  SpectrumView: spectrumPlot
  QFrame: patchFacts
  QPushButton: loadPatchButton
  QPushButton: savePatchButton
  QFrame: audioErrorBanner
  ```

  Add tests for:

  - title is only `Harpy` and content contains no `Harpy Sine Lab`, `Native Sine Lab`, `Synth Engine`, MIDI number, backend, device description, channel count, sample format, `Output ready`, or healthy status;
  - knob and entry update one controller frequency without feedback loops;
  - derived label formats `C3 +0.0¢` from `Tuning.describe_frequency()`;
  - mouse hold/release and Space hold/release issue one NOTE_ON/OFF, while Space does nothing when the frequency editor has focus;
  - a live frequency edit sends RETUNE without another NOTE_ON;
  - natural release completion marks the controller idle, so a later edit sends no
    RETUNE and Clear enters Empty;
  - measurement label is blank/`Measuring…`/`Live`/`Captured` and the views retain captured observations until Clear;
  - empty views show prompts with no fabricated zero or -120 dB traces;
  - patch facts show six separately labelled values and update after load;
  - load cancel is a no-op; invalid load leaves patch/frequency/capture unchanged and reports the named field; valid load preserves frequency, force-stops, replaces patch, and clears capture;
  - valid load while audio is unavailable still updates patch facts, and the recovered
    backend uses that replacement patch rather than its startup patch;
  - Save As writes only canonical `SynthPatch` JSON; write failure does not mutate state;
  - audio failure shows one concise banner, disables Play, and repeated identical notifications do not multiply UI; recovery hides the banner and restores Play without healthy copy;
  - deactivation, close, and shutdown converge on idempotent force-stop.

- [ ] **Step 3: Write the layout/visual regression tests**

  Assert `minimumSize() == QSize(1024, 640)` and initial client `size() == QSize(1280, 720)`. At both sizes after `show()`/event processing, transport height is `<=144`, patch facts height is `<=96`, both plot canvases are at least 320 logical pixels high, Spectrum is wider than Waveform, and every named control/readout has a nonempty visible geometry inside the client rect. Grab a 1280×720 pixmap and assert its size and non-null image; retain the image only on failure via pytest's temporary directory.

- [ ] **Step 4: Run replacement tests and prove the legacy window fails them**

  ```bash
  uv run pytest tests/test_config.py tests/gui/test_window.py tests/gui/test_app.py -q
  ```

  Expected: failures identify the old MIDI slider, old title/status copy, missing knob/views/actions, and GUI-bound config.

- [ ] **Step 5: Rewrite app composition**

  `app.py` constructs one graph in this order:

  ```text
  AppConfig
    ├─ WorkbenchSpec.from_tuning(tuning)
    ├─ SampleHistory(capacity >= fft_frames)
    ├─ CaptureCoordinator(history, render.sample_rate_hz, analysis)
    ├─ QtAudioBackend(render, patch, history)
    ├─ WorkbenchController(spec, render, patch, capture, backend.submit)
    └─ HarpyWindow(controller, tuning, workbench spec, patch dialogs)
  ```

  Connect backend `availability_changed`, `audio_failure`, `force_stop_requested`, and
  `voice_idle` to the single controller/window paths; `voice_idle(generation)` calls
  `controller.mark_voice_idle(generation)`. Connect the window refresh timer at no more than
  30 Hz to `controller.refresh_capture()`. Start backend only after every failure and
  lifecycle handler is connected. Close/shutdown order is force-stop, backend
  shutdown, then widget teardown.

- [ ] **Step 6: Rewrite the window as one professional workbench**

  Layout, top to bottom:

  1. compact transport strip: pitch knob, primary Hz editor/suffix, derived note/cents, normal-sized press-and-hold Play;
  2. shared measurement header: state text and Clear;
  3. Waveform/Spectrum row with stretch 4:6;
  4. compact read-only patch facts and adjacent Load/Save As actions.

  Apply one restrained charcoal stylesheet with tabular numerals, one signal accent, one spectral measurement accent, primary numeric text at least 20 px, ordinary labels at least 12 px, and explicit focus/error styles. Do not add a decorative app heading or device-health area.

- [ ] **Step 7: Delete every superseded path in the same green change**

  Remove the listed old files, the legacy classes inside `gui/qt_audio.py`, and obsolete
  tests. Update imports and package exports directly to final names. Run these
  searches; each must produce no output:

  ```bash
  rg 'KeyboardViewSpec|SineVoice|SinePatch|RenderSpec|EnvelopeSpec|LabController|SampleRingBuffer|format_pitch_readout|MIDI [0-9]|status_changed' src tests
  rg 'harpy\.pitch|harpy\.note_names|gui\.visualizer|synth\.reference|synth\.specs' src tests
  rg 'Harpy · Sine Lab|Native Sine Lab|Output ready|48 kHz|2 ch|Float' src/harpy/gui
  ```

- [ ] **Step 8: Verify Task 9 and commit**

  ```bash
  uv run pytest tests/test_config.py tests/gui/test_window.py tests/gui/test_app.py -q
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run python -c "from harpy.config import DEFAULT_CONFIG; from harpy.synth.engine import SynthEngine; print(SynthEngine(DEFAULT_CONFIG.render, DEFAULT_CONFIG.patch).render(1).dtype)"
  git add -A src tests
  git commit -m "feat: replace sine lab with native workbench"
  ```

  Expected import smoke output: `float32`.

---

## Task 10: Document, audit, and perform native acceptance

**Files:**

- Modify: `README.md`
- Modify: `docs/project-notebook.md` if it contains obsolete browser/page, MIDI-slider, or output-status language
- Modify: `docs/superpowers/specs/2026-08-07-harpy-milestone-a-foundation-workbench-design.md` only for implementation-status wording; do not alter approved behavior
- Create: `docs/verification/2026-08-07-milestone-a-acceptance.md`

- [ ] **Step 1: Update researcher-facing documentation**

  README must explain:

  - Harpy is currently a native deterministic sine workbench and future audio-RL foundation;
  - `uv sync`, `uv run harpy`, and test/lint commands;
  - the public non-Qt API (`SynthPatch`, strict patch JSON, `SynthEngine`, `Tuning`, `analyze`/`AudioObservation`);
  - GUI range C2–C4 versus engine range below Nyquist;
  - patch JSON contains no played note, selected frequency, render setting, or storage metadata;
  - Milestone B and sine-only Gym are deferred roadmap items, not implemented claims.

- [ ] **Step 2: Run the final automated and structural audit**

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  uv run python -c "import harpy.analysis, harpy.capture, harpy.playback, harpy.tuning; import harpy.gui.app; import harpy.synth.engine, harpy.synth.patch_json"
  find src/harpy -type f -name '*.py' -print0 | xargs -0 wc -l
  git diff --check origin/william/native-sine-lab...HEAD
  git status --short
  ```

  Expected: all tests pass; imports are silent; Ruff and diff checks exit 0; the source listing has one responsibility per module; status shows only the documentation/acceptance changes before their commit.

- [ ] **Step 3: Launch through WSLg and exercise the native workbench**

  Use the actual runtime directory if the inherited shell lacks the WSLg socket:

  ```bash
  XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
  ```

  Manually verify and record pass/fail/evidence for each item in the acceptance document:

  - full outer frame remains on-screen at current 125% Windows scaling and at 100%; default client is 1280×720 and minimum remains usable;
  - initial C3 is audible while Play is held, Space behaves identically outside the editor, and release reaches silence without a stuck voice;
  - knob drag is continuous/logarithmic, Shift is fine, wheel/arrows step musically, double-click restores C3, and live/release retuning does not retrigger;
  - after 16,384 stable frames, measured peak follows the selected clean sine within 0.1 Hz;
  - Waveform uses 0–50 ms and ±1 FS; Spectrum uses log 20 Hz–20 kHz and -120–0 dBFS; empty views show no fabricated traces;
  - release retains Captured data and Clear empties it;
  - valid JSON saves/loads/round-trips; invalid JSON names the field and leaves all state intact;
  - no healthy device/status copy is present; simulated/actual device loss produces one error and disables Play; recovery is silent and re-enables it;
  - closing during a held note leaves no sound or live audio stream.

- [ ] **Step 4: Capture visual evidence**

  Save one idle and one live default-size screenshot under `docs/verification/assets/` only if they show the complete outer window frame and contain no machine-private information. Reference them from the acceptance document with relative Markdown links. If native scale switching cannot be automated, state exactly which scale was manually verified and leave the other item explicitly pending rather than claiming it passed.

- [ ] **Step 5: Request whole-branch review, resolve findings, and commit docs**

  The independent reviewer must compare the complete branch against both the approved design and this plan, inspect for legacy/dead paths, and rerun the full gates. Address every Critical or Important finding before completion.

  ```bash
  uv run pytest
  uv run ruff check .
  uv run ruff format --check .
  git diff --check origin/william/native-sine-lab...HEAD
  git add README.md docs
  git commit -m "docs: record milestone A workbench acceptance"
  ```

  Do not mark Milestone A complete or make the draft PR ready until automated gates, native audio, and native visual acceptance are all evidenced.
