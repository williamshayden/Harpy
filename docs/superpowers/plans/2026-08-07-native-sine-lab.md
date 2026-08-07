# Native Sine Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a native WSLg desktop lab that deterministically plays and visualizes one configurable sine note with the approved A440 pitch model and linear ADSR envelope.

**Architecture:** A Qt-free numerical core owns typed musical pitch, validated configuration, the sample-indexed envelope, and the monophonic NumPy sine voice. A PySide6 adapter converts its mono float32 blocks for the default Qt audio device, while a small controller and native Qt Widgets window handle press/hold/release interaction and observational plots. The same block renderer remains authoritative for future offline work; device timing and visualization never affect synth truth.

**Tech Stack:** Python 3.12, uv, NumPy, PySide6 Qt Widgets/Qt Multimedia, pyqtgraph, pytest, pytest-qt, Ruff, Git.

## Global Constraints

- Follow the approved specification at `docs/superpowers/specs/2026-08-07-native-sine-lab-design.md`.
- Use `uv` for dependency management and command execution; commit both `pyproject.toml` and `uv.lock`.
- Target `requires-python = ">=3.12,<3.13"` and put `3.12` in `.python-version`.
- Keep the application native and local. Add no browser, web server, HTML, or JavaScript dependency.
- Keep AMY, SignalFlow, pyo, JUCE, C++, MIDI, databases, audio importing/editing, and RL dependencies out of this milestone.
- Keep PySide6 imports out of `harpy.pitch` and `harpy.synth`.
- Use MIDI note 60 as the initial note and display it as `C3` under the Ableton octave convention.
- Use MIDI note 69 at 440.0 Hz as the default equal-temperament reference.
- Use 48,000 Hz, 256-frame staging blocks, mono float32 authoritative output, and a fixed sine patch at -12 dBFS peak.
- Use a linear-amplitude envelope with 0.001 s attack, 0.600 s decay, -6.0 dB sustain, and 0.600 s release.
- Store sound-producing values in immutable typed configuration objects; do not read them from process environment variables.
- Implement each task test-first, run the focused tests before the full suite, and commit only after its checks pass.
- Preserve unrelated user changes if the worktree becomes dirty during execution.

---

## Final File Responsibility Map

| File | Responsibility |
| --- | --- |
| `.python-version` | Select Python 3.12 for uv. |
| `pyproject.toml` | Package metadata, dependencies, console entry point, pytest, and Ruff settings. |
| `uv.lock` | Reproducible resolved dependency graph. |
| `src/harpy/pitch.py` | MIDI note validation, continuous cents coordinate, and equal-temperament frequency conversion. |
| `src/harpy/note_names.py` | Ableton-style note labels and display readouts only. |
| `src/harpy/config.py` | Compose validated defaults into `AppConfig`. |
| `src/harpy/synth/specs.py` | Envelope, render, and sine-patch immutable specifications. |
| `src/harpy/synth/envelope.py` | Exact sample-domain linear ADSR state machine. |
| `src/harpy/synth/reference.py` | Deterministic monophonic sine voice and block renderer. |
| `src/harpy/gui/specs.py` | Immutable note-selector/display specification. |
| `src/harpy/gui/controller.py` | GUI-neutral note/gate state and typed audio commands. |
| `src/harpy/gui/visualizer.py` | Bounded sample history plus waveform and normalized dBFS spectrum data. |
| `src/harpy/gui/qt_audio.py` | Qt format probing, PCM conversion, QIODevice source, and QAudioSink lifecycle. |
| `src/harpy/gui/window.py` | Native widgets, style, interaction wiring, and plot updates. |
| `src/harpy/gui/app.py` | Application composition, console entry point, and lifetime ownership. |
| `tests/` | Device-free unit and Qt widget tests mirroring the source boundaries. |

---

### Task 1: Bootstrap the uv Package and Musical Pitch Model

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `src/harpy/__init__.py`
- Create: `src/harpy/pitch.py`
- Create: `src/harpy/note_names.py`
- Create: `tests/test_pitch.py`
- Create: `tests/test_note_names.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: no application code.
- Produces: `MidiNote(number: int)`, `Pitch(cents_from_midi_zero: float)`, `Pitch.from_midi(note: MidiNote | int, detune_cents: float = 0.0) -> Pitch`, `EqualTemperament.frequency_hz(pitch: Pitch) -> float`, `format_note_name(note: MidiNote, middle_c_octave: int = 3) -> str`, `format_pitch_readout(note: MidiNote, tuning: EqualTemperament, middle_c_octave: int = 3) -> str`, and `format_tuning_readout(tuning: EqualTemperament) -> str`.

- [ ] **Step 1: Create the package scaffold, metadata, and uv environment**

Create `.python-version`:

```text
3.12
```

Create `src/harpy/__init__.py` so the editable package exists before `uv sync` builds it:

```python
__version__ = "0.1.0"
```

Create `pyproject.toml`:

```toml
[project]
name = "harpy-audio"
version = "0.1.0"
description = "A constrained audio-control benchmark and native demonstration lab"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
  "numpy>=2.0,<3",
  "pyqtgraph>=0.13.7,<1",
  "pyside6>=6.8,<7",
]

[project.scripts]
harpy = "harpy.gui.app:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/harpy"]

[dependency-groups]
dev = [
  "pytest>=8,<9",
  "pytest-qt>=4.4,<5",
  "ruff>=0.9,<1",
]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
qt_api = "pyside6"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["B", "E", "F", "I", "RUF", "SIM", "UP"]
```

Append these generated-package entries to `.gitignore`:

```gitignore
*.egg-info/
dist/
build/
.coverage
htmlcov/
```

Run:

```bash
uv lock
uv sync --dev
```

Expected: `uv.lock` is created and the environment resolves under Python 3.12.

- [ ] **Step 2: Write failing pitch and naming tests**

Create `tests/test_pitch.py`:

```python
import math

import pytest

from harpy.pitch import EqualTemperament, MidiNote, Pitch


@pytest.mark.parametrize("number", [-1, 128, True, 60.0])
def test_midi_note_rejects_invalid_numbers(number: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MidiNote(number)  # type: ignore[arg-type]


def test_default_tuning_maps_reference_note_exactly() -> None:
    tuning = EqualTemperament()
    assert tuning.frequency_hz(Pitch.from_midi(MidiNote(69))) == 440.0


def test_default_tuning_maps_middle_c() -> None:
    frequency = EqualTemperament().frequency_hz(Pitch.from_midi(60))
    assert frequency == pytest.approx(261.6255653005986)


def test_reference_frequency_is_configurable() -> None:
    tuning = EqualTemperament(reference_hz=442.0)
    assert tuning.frequency_hz(Pitch.from_midi(69)) == 442.0


def test_continuous_cents_preserve_interval_ratios() -> None:
    tuning = EqualTemperament()
    base = tuning.frequency_hz(Pitch.from_midi(60))
    semitone = tuning.frequency_hz(Pitch.from_midi(60, detune_cents=100.0))
    octave = tuning.frequency_hz(Pitch.from_midi(60, detune_cents=1200.0))
    assert semitone / base == pytest.approx(2 ** (1 / 12))
    assert octave / base == pytest.approx(2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_pitch_rejects_non_finite_cents(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Pitch(value)


@pytest.mark.parametrize("reference_hz", [0.0, -440.0, math.nan, math.inf])
def test_tuning_rejects_invalid_reference(reference_hz: float) -> None:
    with pytest.raises(ValueError, match="reference_hz"):
        EqualTemperament(reference_hz=reference_hz)


def test_tuning_requires_a_midi_note_reference() -> None:
    with pytest.raises(TypeError, match="reference_note"):
        EqualTemperament(reference_note=69)  # type: ignore[arg-type]
```

Create `tests/test_note_names.py`:

```python
import pytest

from harpy.note_names import format_note_name, format_pitch_readout, format_tuning_readout
from harpy.pitch import EqualTemperament, MidiNote


@pytest.mark.parametrize(
    ("number", "expected"),
    [(0, "C-2"), (60, "C3"), (69, "A3"), (127, "G8")],
)
def test_ableton_note_names(number: int, expected: str) -> None:
    assert format_note_name(MidiNote(number)) == expected


def test_pitch_readout_contains_note_midi_and_frequency() -> None:
    assert format_pitch_readout(MidiNote(60), EqualTemperament()) == ("C3 · MIDI 60 · 261.626 Hz")


def test_tuning_readout_avoids_octave_naming_ambiguity() -> None:
    assert format_tuning_readout(EqualTemperament()) == ("Concert A reference (MIDI 69) · 440.0 Hz")
```

- [ ] **Step 3: Run the focused tests and verify the expected failure**

Run:

```bash
uv run pytest tests/test_pitch.py tests/test_note_names.py -v
```

Expected: collection fails because `harpy.pitch` and `harpy.note_names` do not exist.

- [ ] **Step 4: Implement the numerical pitch and presentation modules**

Implement `src/harpy/pitch.py` with these exact public definitions and validation:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MidiNote:
    number: int

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int):
            raise TypeError("MIDI note number must be an integer")
        if not 0 <= self.number <= 127:
            raise ValueError("MIDI note number must be between 0 and 127")


@dataclass(frozen=True, slots=True)
class Pitch:
    cents_from_midi_zero: float

    def __post_init__(self) -> None:
        value = float(self.cents_from_midi_zero)
        if not math.isfinite(value):
            raise ValueError("pitch cents must be finite")
        object.__setattr__(self, "cents_from_midi_zero", value)

    @classmethod
    def from_midi(
        cls,
        note: MidiNote | int,
        detune_cents: float = 0.0,
    ) -> Pitch:
        midi_note = note if isinstance(note, MidiNote) else MidiNote(note)
        detune = float(detune_cents)
        if not math.isfinite(detune):
            raise ValueError("detune_cents must be finite")
        return cls(100.0 * midi_note.number + detune)


@dataclass(frozen=True, slots=True)
class EqualTemperament:
    reference_note: MidiNote = field(default_factory=lambda: MidiNote(69))
    reference_hz: float = 440.0

    def __post_init__(self) -> None:
        if not isinstance(self.reference_note, MidiNote):
            raise TypeError("reference_note must be a MidiNote")
        value = float(self.reference_hz)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("reference_hz must be positive and finite")
        object.__setattr__(self, "reference_hz", value)

    def frequency_hz(self, pitch: Pitch) -> float:
        exponent = (pitch.cents_from_midi_zero - 100.0 * self.reference_note.number) / 1200.0
        try:
            frequency = self.reference_hz * (2.0**exponent)
        except OverflowError as error:
            raise ValueError("pitch maps outside the finite positive frequency range") from error
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("pitch maps outside the finite positive frequency range")
        return frequency
```

Implement `src/harpy/note_names.py`:

```python
from harpy.pitch import EqualTemperament, MidiNote, Pitch

_PITCH_CLASSES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")


def format_note_name(note: MidiNote, middle_c_octave: int = 3) -> str:
    octave = note.number // 12 + middle_c_octave - 5
    return f"{_PITCH_CLASSES[note.number % 12]}{octave}"


def format_pitch_readout(
    note: MidiNote,
    tuning: EqualTemperament,
    middle_c_octave: int = 3,
) -> str:
    frequency = tuning.frequency_hz(Pitch.from_midi(note))
    return f"{format_note_name(note, middle_c_octave)} · MIDI {note.number} · {frequency:.3f} Hz"


def format_tuning_readout(tuning: EqualTemperament) -> str:
    return (
        f"Concert A reference (MIDI {tuning.reference_note.number}) · {tuning.reference_hz:.1f} Hz"
    )
```

- [ ] **Step 5: Run focused and style checks**

Run:

```bash
uv run pytest tests/test_pitch.py tests/test_note_names.py -v
uv run ruff check src/harpy/pitch.py src/harpy/note_names.py tests/test_pitch.py tests/test_note_names.py
uv run ruff format --check src/harpy/pitch.py src/harpy/note_names.py tests/test_pitch.py tests/test_note_names.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 6: Commit the package and pitch foundation**

```bash
git add .python-version pyproject.toml uv.lock .gitignore src/harpy tests/test_pitch.py tests/test_note_names.py
git commit -m "feat: add uv package and pitch model"
```

---

### Task 2: Add Validated Synth, Render, Keyboard, and Application Configuration

**Files:**
- Create: `src/harpy/synth/__init__.py`
- Create: `src/harpy/synth/specs.py`
- Create: `src/harpy/gui/__init__.py`
- Create: `src/harpy/gui/specs.py`
- Create: `src/harpy/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `MidiNote` and `EqualTemperament` from Task 1.
- Produces: `seconds_to_frames(seconds: float, sample_rate_hz: int) -> int`, `EnvelopeSpec`, `RenderSpec`, `SinePatch`, `KeyboardViewSpec`, `AppConfig`, and `DEFAULT_CONFIG`.

- [ ] **Step 1: Write failing configuration tests**

Create `tests/test_config.py`:

```python
import math

import pytest

from harpy.config import AppConfig, DEFAULT_CONFIG
from harpy.gui.specs import KeyboardViewSpec
from harpy.pitch import MidiNote
from harpy.synth.specs import EnvelopeSpec, RenderSpec, SinePatch, seconds_to_frames


def test_default_configuration_matches_approved_values() -> None:
    config = DEFAULT_CONFIG
    assert config.tuning.reference_note == MidiNote(69)
    assert config.tuning.reference_hz == 440.0
    assert config.keyboard == KeyboardViewSpec()
    assert config.render == RenderSpec()
    assert config.patch.envelope == EnvelopeSpec()
    assert config.patch.peak_gain_dbfs == -12.0
    assert config.patch.peak_gain == pytest.approx(10 ** (-12.0 / 20.0))


def test_half_up_frame_rounding() -> None:
    assert seconds_to_frames(0.001, 48_000) == 48
    assert seconds_to_frames(0.0005, 1_000) == 1


def test_default_envelope_frame_counts() -> None:
    envelope = EnvelopeSpec()
    assert envelope.attack_frames(48_000) == 48
    assert envelope.decay_frames(48_000) == 28_800
    assert envelope.release_frames(48_000) == 28_800
    assert envelope.sustain_amplitude == pytest.approx(0.5011872336272722)


@pytest.mark.parametrize("field", ["attack_seconds", "decay_seconds", "release_seconds"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_envelope_rejects_invalid_durations(field: str, value: float) -> None:
    values = {
        "attack_seconds": 0.001,
        "decay_seconds": 0.600,
        "sustain_db": -6.0,
        "release_seconds": 0.600,
    }
    values[field] = value
    with pytest.raises(ValueError):
        EnvelopeSpec(**values)


@pytest.mark.parametrize("sustain_db", [0.1, math.nan, math.inf])
def test_envelope_rejects_invalid_sustain(sustain_db: float) -> None:
    with pytest.raises(ValueError, match="sustain_db"):
        EnvelopeSpec(sustain_db=sustain_db)


def test_keyboard_requires_ordered_range() -> None:
    with pytest.raises(ValueError, match="minimum.*initial.*maximum"):
        KeyboardViewSpec(
            minimum_note=MidiNote(61),
            initial_note=MidiNote(60),
            maximum_note=MidiNote(72),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"sample_rate_hz": 0},
        {"block_frames": 0},
        {"channels": 2},
        {"internal_dtype": "float64"},
    ],
)
def test_render_spec_rejects_incompatible_settings(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RenderSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("peak_gain_dbfs", [0.1, math.nan, math.inf])
def test_patch_rejects_invalid_peak_gain(peak_gain_dbfs: float) -> None:
    with pytest.raises(ValueError, match="peak_gain_dbfs"):
        SinePatch(peak_gain_dbfs=peak_gain_dbfs)


def test_application_rejects_sub_frame_envelope_stage() -> None:
    patch = SinePatch(envelope=EnvelopeSpec(attack_seconds=0.00001))
    with pytest.raises(ValueError, match="attack.*one frame"):
        AppConfig(patch=patch)
```

- [ ] **Step 2: Run the configuration tests and verify failure**

Run:

```bash
uv run pytest tests/test_config.py -v
```

Expected: collection fails because the configuration modules do not exist.

- [ ] **Step 3: Implement immutable subsystem specifications**

Create empty `src/harpy/synth/__init__.py` and `src/harpy/gui/__init__.py` files.

Implement `src/harpy/synth/specs.py` with:

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field


def seconds_to_frames(seconds: float, sample_rate_hz: int) -> int:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be a positive integer")
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("seconds must be finite and non-negative")
    return math.floor(seconds * sample_rate_hz + 0.5)


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    attack_seconds: float = 0.001
    decay_seconds: float = 0.600
    sustain_db: float = -6.0
    release_seconds: float = 0.600
    curve: str = "linear_amplitude"

    def __post_init__(self) -> None:
        for name in ("attack_seconds", "decay_seconds", "release_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        sustain = float(self.sustain_db)
        if not math.isfinite(sustain) or sustain > 0.0:
            raise ValueError("sustain_db must be finite and no greater than 0 dB")
        object.__setattr__(self, "sustain_db", sustain)
        if self.curve != "linear_amplitude":
            raise ValueError("v1 supports only the linear_amplitude envelope curve")

    @property
    def sustain_amplitude(self) -> float:
        return 10.0 ** (self.sustain_db / 20.0)

    def attack_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.attack_seconds, sample_rate_hz)

    def decay_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.decay_seconds, sample_rate_hz)

    def release_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.release_seconds, sample_rate_hz)


@dataclass(frozen=True, slots=True)
class RenderSpec:
    sample_rate_hz: int = 48_000
    block_frames: int = 256
    channels: int = 1
    internal_dtype: str = "float32"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive integer")
        if (
            isinstance(self.block_frames, bool)
            or not isinstance(self.block_frames, int)
            or self.block_frames <= 0
        ):
            raise ValueError("block_frames must be a positive integer")
        if self.channels != 1:
            raise ValueError("the authoritative v1 renderer must be mono")
        if self.internal_dtype != "float32":
            raise ValueError("the authoritative v1 renderer must use float32")


@dataclass(frozen=True, slots=True)
class SinePatch:
    peak_gain_dbfs: float = -12.0
    envelope: EnvelopeSpec = field(default_factory=EnvelopeSpec)

    def __post_init__(self) -> None:
        gain = float(self.peak_gain_dbfs)
        if not math.isfinite(gain) or gain > 0.0:
            raise ValueError("peak_gain_dbfs must be finite and no greater than 0 dBFS")
        object.__setattr__(self, "peak_gain_dbfs", gain)

    @property
    def peak_gain(self) -> float:
        return 10.0 ** (self.peak_gain_dbfs / 20.0)
```

Implement `src/harpy/gui/specs.py`:

```python
from dataclasses import dataclass, field

from harpy.pitch import MidiNote


@dataclass(frozen=True, slots=True)
class KeyboardViewSpec:
    minimum_note: MidiNote = field(default_factory=lambda: MidiNote(48))
    initial_note: MidiNote = field(default_factory=lambda: MidiNote(60))
    maximum_note: MidiNote = field(default_factory=lambda: MidiNote(72))
    middle_c_octave: int = 3

    def __post_init__(self) -> None:
        if not (self.minimum_note.number <= self.initial_note.number <= self.maximum_note.number):
            raise ValueError("minimum_note <= initial_note <= maximum_note is required")
        if isinstance(self.middle_c_octave, bool) or not isinstance(self.middle_c_octave, int):
            raise TypeError("middle_c_octave must be an integer")
```

- [ ] **Step 4: Compose and validate application defaults**

Implement `src/harpy/config.py`:

```python
from dataclasses import dataclass, field

from harpy.gui.specs import KeyboardViewSpec
from harpy.pitch import EqualTemperament
from harpy.synth.specs import RenderSpec, SinePatch


@dataclass(frozen=True, slots=True)
class AppConfig:
    tuning: EqualTemperament = field(default_factory=EqualTemperament)
    keyboard: KeyboardViewSpec = field(default_factory=KeyboardViewSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    patch: SinePatch = field(default_factory=SinePatch)

    def __post_init__(self) -> None:
        sample_rate = self.render.sample_rate_hz
        frame_counts = {
            "attack": self.patch.envelope.attack_frames(sample_rate),
            "decay": self.patch.envelope.decay_frames(sample_rate),
            "release": self.patch.envelope.release_frames(sample_rate),
        }
        for name, frames in frame_counts.items():
            if frames < 1:
                raise ValueError(f"{name} must produce at least one frame")


DEFAULT_CONFIG = AppConfig()
```

- [ ] **Step 5: Run configuration and regression checks**

Run:

```bash
uv run pytest tests/test_config.py tests/test_pitch.py tests/test_note_names.py -v
uv run ruff check src tests/test_config.py
uv run ruff format --check src tests/test_config.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 6: Commit the configuration model**

```bash
git add src/harpy/config.py src/harpy/synth src/harpy/gui tests/test_config.py
git commit -m "feat: add validated sine lab configuration"
```

---

### Task 3: Implement the Exact Linear ADSR State Machine

**Files:**
- Create: `src/harpy/synth/envelope.py`
- Create: `tests/synth/__init__.py`
- Create: `tests/synth/test_envelope.py`

**Interfaces:**
- Consumes: `EnvelopeSpec` and a positive sample rate.
- Produces: `EnvelopeStage` and `LinearEnvelope` with `note_on()`, `note_off()`, `reset()`, `render(frame_count)`, `level`, `stage`, and `is_idle`.

- [ ] **Step 1: Write exact discrete-time boundary tests**

Create empty `tests/synth/__init__.py` and create `tests/synth/test_envelope.py`:

```python
import math

import numpy as np
import pytest

from harpy.synth.envelope import EnvelopeStage, LinearEnvelope
from harpy.synth.specs import EnvelopeSpec


def half_sustain_spec() -> EnvelopeSpec:
    return EnvelopeSpec(
        attack_seconds=1.0,
        decay_seconds=1.0,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=1.0,
    )


def test_attack_decay_and_sustain_emit_exact_endpoints() -> None:
    envelope = LinearEnvelope(half_sustain_spec(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(4), [0.25, 0.5, 0.75, 1.0])
    assert envelope.stage is EnvelopeStage.DECAY
    np.testing.assert_allclose(envelope.render(4), [0.875, 0.75, 0.625, 0.5])
    assert envelope.stage is EnvelopeStage.SUSTAIN
    np.testing.assert_allclose(envelope.render(3), [0.5, 0.5, 0.5])


def test_early_note_off_releases_from_last_emitted_level() -> None:
    envelope = LinearEnvelope(half_sustain_spec(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(1), [0.25])
    envelope.note_off()
    np.testing.assert_allclose(envelope.render(4), [0.1875, 0.125, 0.0625, 0.0])
    assert envelope.stage is EnvelopeStage.IDLE
    assert envelope.level == 0.0
    np.testing.assert_array_equal(envelope.render(3), np.zeros(3))


def test_note_off_is_idempotent_during_release() -> None:
    envelope = LinearEnvelope(half_sustain_spec(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(4)
    envelope.note_off()
    first = envelope.render(1)
    envelope.note_off()
    rest = envelope.render(3)
    np.testing.assert_allclose(np.concatenate((first, rest)), [0.75, 0.5, 0.25, 0.0])


def test_retrigger_restarts_attack_from_zero() -> None:
    envelope = LinearEnvelope(half_sustain_spec(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(3)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(2), [0.25, 0.5])


@pytest.mark.parametrize("frame_count", [-1, 1.5, True])
def test_render_rejects_invalid_frame_counts(frame_count: object) -> None:
    envelope = LinearEnvelope(half_sustain_spec(), sample_rate_hz=4)
    with pytest.raises((TypeError, ValueError)):
        envelope.render(frame_count)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the envelope tests and verify failure**

Run:

```bash
uv run pytest tests/synth/test_envelope.py -v
```

Expected: collection fails because `harpy.synth.envelope` does not exist.

- [ ] **Step 3: Implement the sample-domain envelope**

Implement `src/harpy/synth/envelope.py` using this state transition structure:

```python
from __future__ import annotations

from enum import StrEnum

import numpy as np

from harpy.synth.specs import EnvelopeSpec


class EnvelopeStage(StrEnum):
    IDLE = "idle"
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"


class LinearEnvelope:
    def __init__(self, spec: EnvelopeSpec, sample_rate_hz: int) -> None:
        self.spec = spec
        self.sample_rate_hz = sample_rate_hz
        self._attack_frames = spec.attack_frames(sample_rate_hz)
        self._decay_frames = spec.decay_frames(sample_rate_hz)
        self._release_frames = spec.release_frames(sample_rate_hz)
        if min(self._attack_frames, self._decay_frames, self._release_frames) < 1:
            raise ValueError("every envelope stage must contain at least one frame")
        self.reset()

    @property
    def level(self) -> float:
        return self._level

    @property
    def stage(self) -> EnvelopeStage:
        return self._stage

    @property
    def is_idle(self) -> bool:
        return self._stage is EnvelopeStage.IDLE

    def reset(self) -> None:
        self._stage = EnvelopeStage.IDLE
        self._level = 0.0
        self._target = 0.0
        self._step = 0.0
        self._remaining = 0

    def note_on(self) -> None:
        self._level = 0.0
        self._begin_segment(EnvelopeStage.ATTACK, 1.0, self._attack_frames)

    def note_off(self) -> None:
        if self._stage in (EnvelopeStage.IDLE, EnvelopeStage.RELEASE):
            return
        if self._level <= 0.0:
            self.reset()
            return
        self._begin_segment(EnvelopeStage.RELEASE, 0.0, self._release_frames)

    def render(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        output = np.empty(frame_count, dtype=np.float64)
        for index in range(frame_count):
            if self._stage is EnvelopeStage.IDLE:
                output[index] = 0.0
                continue
            if self._stage is EnvelopeStage.SUSTAIN:
                output[index] = self._level
                continue
            self._level += self._step
            self._remaining -= 1
            if self._remaining == 0:
                self._level = self._target
            output[index] = self._level
            if self._remaining == 0:
                self._finish_segment()
        return output

    def _begin_segment(
        self,
        stage: EnvelopeStage,
        target: float,
        frames: int,
    ) -> None:
        self._stage = stage
        self._target = target
        self._remaining = frames
        self._step = (target - self._level) / frames

    def _finish_segment(self) -> None:
        if self._stage is EnvelopeStage.ATTACK:
            self._begin_segment(
                EnvelopeStage.DECAY,
                self.spec.sustain_amplitude,
                self._decay_frames,
            )
        elif self._stage is EnvelopeStage.DECAY:
            self._stage = EnvelopeStage.SUSTAIN
            self._level = self.spec.sustain_amplitude
            self._remaining = 0
            self._step = 0.0
        elif self._stage is EnvelopeStage.RELEASE:
            self.reset()
```

- [ ] **Step 4: Run exact envelope and full numerical tests**

Run:

```bash
uv run pytest tests/synth/test_envelope.py tests/test_config.py -v
uv run ruff check src/harpy/synth/envelope.py tests/synth/test_envelope.py
uv run ruff format --check src/harpy/synth/envelope.py tests/synth/test_envelope.py
```

Expected: all checks pass, including exact attack/decay/release arrays.

- [ ] **Step 5: Commit the envelope state machine**

```bash
git add src/harpy/synth/envelope.py tests/synth
git commit -m "feat: add deterministic linear envelope"
```

---

### Task 4: Implement the Deterministic Monophonic Sine Voice

**Files:**
- Create: `src/harpy/synth/reference.py`
- Create: `tests/synth/test_reference.py`

**Interfaces:**
- Consumes: `EqualTemperament`, `Pitch`, `RenderSpec`, `SinePatch`, and `LinearEnvelope`.
- Produces: `SineVoice.note_on(pitch)`, `note_off()`, `reset()`, and `render_block(frame_count) -> np.ndarray` with shape `(frame_count,)` and dtype float32.

- [ ] **Step 1: Write failing voice, invariance, and FFT tests**

Create `tests/synth/test_reference.py`:

```python
import math

import numpy as np
import pytest

from harpy.config import DEFAULT_CONFIG
from harpy.pitch import Pitch
from harpy.synth.reference import SineVoice


def make_voice() -> SineVoice:
    return SineVoice(
        tuning=DEFAULT_CONFIG.tuning,
        render_spec=DEFAULT_CONFIG.render,
        patch=DEFAULT_CONFIG.patch,
    )


def test_idle_voice_emits_exact_mono_float32_silence() -> None:
    samples = make_voice().render_block(256)
    assert samples.shape == (256,)
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.zeros(256, dtype=np.float32))


def test_note_on_resets_phase_and_respects_peak_gain() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    first = voice.render_block(2_048)
    voice.note_on(Pitch.from_midi(60))
    second = voice.render_block(2_048)
    assert first[0] == 0.0
    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first)) <= DEFAULT_CONFIG.patch.peak_gain + 1e-7


def test_rendering_is_invariant_to_block_partitioning() -> None:
    whole_voice = make_voice()
    chunked_voice = make_voice()
    pitch = Pitch.from_midi(60)
    whole_voice.note_on(pitch)
    chunked_voice.note_on(pitch)
    whole = whole_voice.render_block(10_000)
    chunked = np.concatenate(
        [
            chunked_voice.render_block(1),
            chunked_voice.render_block(255),
            chunked_voice.render_block(4_096),
            chunked_voice.render_block(5_648),
        ]
    )
    np.testing.assert_allclose(whole, chunked, rtol=0.0, atol=1e-6)


def test_release_reaches_exact_silence() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    voice.render_block(1_000)
    voice.note_off()
    voice.render_block(DEFAULT_CONFIG.patch.envelope.release_frames(48_000))
    np.testing.assert_array_equal(
        voice.render_block(256),
        np.zeros(256, dtype=np.float32),
    )


def test_fft_peak_matches_middle_c_within_one_bin() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    settle_frames = DEFAULT_CONFIG.patch.envelope.attack_frames(
        48_000
    ) + DEFAULT_CONFIG.patch.envelope.decay_frames(48_000)
    voice.render_block(settle_frames)
    samples = voice.render_block(96_000).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / 48_000)
    peak_hz = frequencies[int(np.argmax(spectrum[1:]) + 1)]
    expected_hz = DEFAULT_CONFIG.tuning.frequency_hz(Pitch.from_midi(60))
    assert abs(peak_hz - expected_hz) <= 0.5


def test_nyquist_frequency_is_rejected() -> None:
    voice = make_voice()
    above_nyquist_cents = 6_900.0 + 1_200.0 * math.log2(25_000.0 / 440.0)
    with pytest.raises(ValueError, match="Nyquist"):
        voice.note_on(Pitch(above_nyquist_cents))
```

- [ ] **Step 2: Run the reference voice tests and verify failure**

Run:

```bash
uv run pytest tests/synth/test_reference.py -v
```

Expected: collection fails because `harpy.synth.reference` does not exist.

- [ ] **Step 3: Implement the authoritative block renderer**

Implement `src/harpy/synth/reference.py`:

```python
from __future__ import annotations

import math

import numpy as np

from harpy.pitch import EqualTemperament, Pitch
from harpy.synth.envelope import LinearEnvelope
from harpy.synth.specs import RenderSpec, SinePatch


class SineVoice:
    def __init__(
        self,
        tuning: EqualTemperament,
        render_spec: RenderSpec,
        patch: SinePatch,
    ) -> None:
        self.tuning = tuning
        self.render_spec = render_spec
        self.patch = patch
        self._envelope = LinearEnvelope(patch.envelope, render_spec.sample_rate_hz)
        self._phase = 0.0
        self._phase_increment = 0.0

    @property
    def is_idle(self) -> bool:
        return self._envelope.is_idle

    def note_on(self, pitch: Pitch) -> None:
        frequency_hz = self.tuning.frequency_hz(pitch)
        nyquist_hz = self.render_spec.sample_rate_hz / 2.0
        if frequency_hz >= nyquist_hz:
            raise ValueError("frequency must be below Nyquist")
        self._phase = 0.0
        self._phase_increment = math.tau * frequency_hz / self.render_spec.sample_rate_hz
        self._envelope.note_on()

    def note_off(self) -> None:
        self._envelope.note_off()

    def reset(self) -> None:
        self._phase = 0.0
        self._phase_increment = 0.0
        self._envelope.reset()

    def render_block(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)
        if self._envelope.is_idle:
            return np.zeros(frame_count, dtype=np.float32)
        offsets = np.arange(frame_count, dtype=np.float64)
        phases = self._phase + self._phase_increment * offsets
        oscillator = np.sin(phases)
        envelope = self._envelope.render(frame_count)
        self._phase = math.fmod(
            self._phase + self._phase_increment * frame_count,
            math.tau,
        )
        samples = oscillator * envelope * self.patch.peak_gain
        return samples.astype(np.float32)
```

- [ ] **Step 4: Run focused, full numerical, and style checks**

Run:

```bash
uv run pytest tests/synth tests/test_pitch.py tests/test_config.py -v
uv run ruff check src/harpy/synth tests/synth
uv run ruff format --check src/harpy/synth tests/synth
```

Expected: all tests pass, the FFT error is at most 0.5 Hz, and chunking error stays below `1e-6` absolute.

- [ ] **Step 5: Commit the reference synth**

```bash
git add src/harpy/synth/reference.py tests/synth/test_reference.py
git commit -m "feat: add deterministic sine voice"
```

---

### Task 5: Add the Non-Blocking Visualization Sample Buffer and dBFS Spectrum

**Files:**
- Create: `src/harpy/gui/visualizer.py`
- Create: `tests/gui/__init__.py`
- Create: `tests/gui/test_visualizer.py`

**Interfaces:**
- Consumes: mono NumPy sample blocks and sample rate.
- Produces: `SampleRingBuffer.append(samples: np.ndarray) -> bool`, `clear() -> None`, `snapshot(frame_count: int) -> np.ndarray`, `waveform_time_ms(frame_count: int, sample_rate_hz: int) -> np.ndarray`, and `spectrum_dbfs(samples: np.ndarray, sample_rate_hz: int, fft_frames: int = 4096, floor_dbfs: float = -120.0) -> tuple[np.ndarray, np.ndarray]`.

- [ ] **Step 1: Write failing bounded-buffer and normalized-spectrum tests**

Create empty `tests/gui/__init__.py` and create `tests/gui/test_visualizer.py`:

```python
import numpy as np
import pytest

from harpy.gui.visualizer import SampleRingBuffer, spectrum_dbfs, waveform_time_ms


def test_ring_buffer_keeps_only_newest_samples() -> None:
    buffer = SampleRingBuffer(capacity_frames=5)
    buffer.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    buffer.append(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(buffer.snapshot(5), [3.0, 4.0, 5.0, 6.0, 7.0])


def test_snapshot_left_pads_missing_history_with_zero() -> None:
    buffer = SampleRingBuffer(capacity_frames=8)
    buffer.append(np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(buffer.snapshot(4), [0.0, 0.0, 1.0, 2.0])


def test_snapshot_is_a_copy_and_clear_restores_silence() -> None:
    buffer = SampleRingBuffer(capacity_frames=4)
    buffer.append(np.ones(4, dtype=np.float32))
    snapshot = buffer.snapshot(4)
    snapshot[:] = 9.0
    np.testing.assert_array_equal(buffer.snapshot(4), np.ones(4))
    buffer.clear()
    np.testing.assert_array_equal(buffer.snapshot(4), np.zeros(4))


def test_bin_centered_sine_reports_correct_dbfs() -> None:
    sample_rate = 48_000
    frame_count = 4_096
    frequency_hz = 375.0
    amplitude = 0.25
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    samples = amplitude * np.sin(2.0 * np.pi * frequency_hz * time)
    frequencies, levels = spectrum_dbfs(samples, sample_rate, fft_frames=frame_count)
    peak_index = int(np.argmax(levels))
    assert frequencies[peak_index] == pytest.approx(frequency_hz)
    assert levels[peak_index] == pytest.approx(20.0 * np.log10(amplitude), abs=0.1)
    assert np.min(levels) >= -120.0


def test_waveform_axis_is_in_milliseconds() -> None:
    np.testing.assert_allclose(
        waveform_time_ms(3, sample_rate_hz=1_000),
        [-2.0, -1.0, 0.0],
    )
```

- [ ] **Step 2: Run visualizer tests and verify failure**

Run:

```bash
uv run pytest tests/gui/test_visualizer.py -v
```

Expected: collection fails because `harpy.gui.visualizer` does not exist.

- [ ] **Step 3: Implement bounded history with a non-blocking audio-thread append**

Implement the ring-buffer portion of `src/harpy/gui/visualizer.py`:

```python
from __future__ import annotations

import threading

import numpy as np


class SampleRingBuffer:
    def __init__(self, capacity_frames: int) -> None:
        if not isinstance(capacity_frames, int) or capacity_frames <= 0:
            raise ValueError("capacity_frames must be a positive integer")
        self._capacity = capacity_frames
        self._data = np.zeros(capacity_frames, dtype=np.float32)
        self._write_index = 0
        self._size = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> bool:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return True
        if not self._lock.acquire(blocking=False):
            return False
        try:
            values = values[-self._capacity :]
            first_count = min(values.size, self._capacity - self._write_index)
            self._data[self._write_index : self._write_index + first_count] = values[:first_count]
            second_count = values.size - first_count
            if second_count:
                self._data[:second_count] = values[first_count:]
            self._write_index = (self._write_index + values.size) % self._capacity
            self._size = min(self._capacity, self._size + values.size)
            return True
        finally:
            self._lock.release()

    def clear(self) -> None:
        with self._lock:
            self._data.fill(0.0)
            self._write_index = 0
            self._size = 0

    def snapshot(self, frame_count: int) -> np.ndarray:
        if not isinstance(frame_count, int) or not 0 <= frame_count <= self._capacity:
            raise ValueError("frame_count must be between zero and capacity_frames")
        with self._lock:
            available = min(frame_count, self._size)
            output = np.zeros(frame_count, dtype=np.float32)
            if available == 0:
                return output
            start = (self._write_index - available) % self._capacity
            first_count = min(available, self._capacity - start)
            destination = frame_count - available
            output[destination : destination + first_count] = self._data[
                start : start + first_count
            ]
            if first_count < available:
                output[destination + first_count :] = self._data[: available - first_count]
            return output
```

- [ ] **Step 4: Implement waveform and coherent-gain-normalized spectrum data**

Add these functions to `src/harpy/gui/visualizer.py`:

```python
def waveform_time_ms(frame_count: int, sample_rate_hz: int) -> np.ndarray:
    if frame_count < 0 or sample_rate_hz <= 0:
        raise ValueError("frame_count must be non-negative and sample_rate_hz positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.float64)
    return (np.arange(frame_count, dtype=np.float64) - (frame_count - 1)) * (
        1_000.0 / sample_rate_hz
    )


def spectrum_dbfs(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    fft_frames: int = 4_096,
    floor_dbfs: float = -120.0,
) -> tuple[np.ndarray, np.ndarray]:
    if fft_frames <= 0 or fft_frames % 2:
        raise ValueError("fft_frames must be a positive even integer")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    frame = np.zeros(fft_frames, dtype=np.float64)
    copied = min(values.size, fft_frames)
    if copied:
        frame[-copied:] = values[-copied:]
    window = np.hanning(fft_frames)
    magnitudes = np.abs(np.fft.rfft(frame * window)) / np.sum(window)
    if magnitudes.size > 2:
        magnitudes[1:-1] *= 2.0
    floor_amplitude = 10.0 ** (floor_dbfs / 20.0)
    levels = 20.0 * np.log10(np.maximum(magnitudes, floor_amplitude))
    frequencies = np.fft.rfftfreq(fft_frames, d=1.0 / sample_rate_hz)
    return frequencies, levels
```

- [ ] **Step 5: Run visualizer and style checks**

Run:

```bash
uv run pytest tests/gui/test_visualizer.py -v
uv run ruff check src/harpy/gui/visualizer.py tests/gui/test_visualizer.py
uv run ruff format --check src/harpy/gui/visualizer.py tests/gui/test_visualizer.py
```

Expected: the bounded buffer and -12.04 dBFS sine assertions pass.

- [ ] **Step 6: Commit the visualization data path**

```bash
git add src/harpy/gui/visualizer.py tests/gui
git commit -m "feat: add waveform and spectrum data path"
```

---

### Task 6: Add the GUI-Neutral Gate Controller and Typed Audio Commands

**Files:**
- Create: `src/harpy/gui/controller.py`
- Create: `tests/gui/test_controller.py`

**Interfaces:**
- Consumes: `AppConfig`, pitch/readout helpers, and a callable `send_command(AudioCommand) -> None`.
- Produces: `AudioCommandKind`, `AudioCommand`, `ControllerState`, and `LabController` with `set_note(number: int) -> ControllerState`, `press_play() -> ControllerState`, `release_play() -> ControllerState`, and idempotent `force_stop() -> ControllerState`.

- [ ] **Step 1: Write failing controller trajectory tests**

Create `tests/gui/test_controller.py`:

```python
import pytest

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind, LabController
from harpy.pitch import MidiNote, Pitch


def make_controller() -> tuple[LabController, list[AudioCommand]]:
    commands: list[AudioCommand] = []
    return LabController(DEFAULT_CONFIG, commands.append), commands


def test_initial_state_is_middle_c_and_idle() -> None:
    controller, commands = make_controller()
    assert controller.state.note == MidiNote(60)
    assert controller.state.readout == "C3 · MIDI 60 · 261.626 Hz"
    assert not controller.state.gate_active
    assert controller.state.selector_enabled
    assert commands == []


def test_press_and_release_send_one_command_each() -> None:
    controller, commands = make_controller()
    controller.press_play()
    controller.press_play()
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert controller.state.gate_active
    assert not controller.state.selector_enabled
    controller.release_play()
    controller.release_play()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert not controller.state.gate_active
    assert controller.state.selector_enabled


def test_note_selection_is_blocked_while_gated() -> None:
    controller, _ = make_controller()
    controller.press_play()
    with pytest.raises(RuntimeError, match="while Play is held"):
        controller.set_note(61)


@pytest.mark.parametrize("number", [47, 73])
def test_note_selection_rejects_values_outside_the_visible_range(number: int) -> None:
    controller, _ = make_controller()
    with pytest.raises(ValueError, match="configured selector range"):
        controller.set_note(number)


def test_audio_commands_validate_pitch_payloads() -> None:
    with pytest.raises(ValueError, match="requires a pitch"):
        AudioCommand(AudioCommandKind.NOTE_ON)
    with pytest.raises(ValueError, match="does not accept"):
        AudioCommand(AudioCommandKind.NOTE_OFF, Pitch.from_midi(60))


def test_force_stop_clears_state_and_is_idempotent() -> None:
    controller, commands = make_controller()
    controller.press_play()
    controller.force_stop()
    controller.force_stop()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not controller.state.gate_active
    assert controller.state.selector_enabled


def test_note_can_change_after_normal_release() -> None:
    controller, _ = make_controller()
    controller.press_play()
    controller.release_play()
    state = controller.set_note(61)
    assert state.note == MidiNote(61)
    assert state.readout.startswith("C♯3 · MIDI 61")
```

- [ ] **Step 2: Run controller tests and verify failure**

Run:

```bash
uv run pytest tests/gui/test_controller.py -v
```

Expected: collection fails because `harpy.gui.controller` does not exist.

- [ ] **Step 3: Implement command validation and controller state**

Implement `src/harpy/gui/controller.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from harpy.config import AppConfig
from harpy.note_names import format_pitch_readout
from harpy.pitch import MidiNote, Pitch


class AudioCommandKind(StrEnum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class AudioCommand:
    kind: AudioCommandKind
    pitch: Pitch | None = None

    def __post_init__(self) -> None:
        if self.kind is AudioCommandKind.NOTE_ON and self.pitch is None:
            raise ValueError("NOTE_ON requires a pitch")
        if self.kind is not AudioCommandKind.NOTE_ON and self.pitch is not None:
            raise ValueError(f"{self.kind} does not accept a pitch")


@dataclass(frozen=True, slots=True)
class ControllerState:
    note: MidiNote
    readout: str
    gate_active: bool
    selector_enabled: bool


class LabController:
    def __init__(
        self,
        config: AppConfig,
        send_command: Callable[[AudioCommand], None],
    ) -> None:
        self._config = config
        self._send_command = send_command
        self._note = config.keyboard.initial_note
        self._gate_active = False
        self._voice_started = False

    @property
    def state(self) -> ControllerState:
        return ControllerState(
            note=self._note,
            readout=format_pitch_readout(
                self._note,
                self._config.tuning,
                self._config.keyboard.middle_c_octave,
            ),
            gate_active=self._gate_active,
            selector_enabled=not self._gate_active,
        )

    def set_note(self, number: int) -> ControllerState:
        if self._gate_active:
            raise RuntimeError("pitch cannot change while Play is held")
        note = MidiNote(number)
        if not (
            self._config.keyboard.minimum_note.number
            <= number
            <= self._config.keyboard.maximum_note.number
        ):
            raise ValueError("note is outside the configured selector range")
        self._note = note
        return self.state

    def press_play(self) -> ControllerState:
        if self._gate_active:
            return self.state
        self._send_command(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(self._note)))
        self._gate_active = True
        self._voice_started = True
        return self.state

    def release_play(self) -> ControllerState:
        if not self._gate_active:
            return self.state
        self._send_command(AudioCommand(AudioCommandKind.NOTE_OFF))
        self._gate_active = False
        return self.state

    def force_stop(self) -> ControllerState:
        self._gate_active = False
        if self._voice_started:
            self._send_command(AudioCommand(AudioCommandKind.RESET))
            self._voice_started = False
        return self.state
```

- [ ] **Step 4: Run controller, pitch, and style checks**

Run:

```bash
uv run pytest tests/gui/test_controller.py tests/test_pitch.py tests/test_note_names.py -v
uv run ruff check src/harpy/gui/controller.py tests/gui/test_controller.py
uv run ruff format --check src/harpy/gui/controller.py tests/gui/test_controller.py
```

Expected: all tests and checks pass.

- [ ] **Step 5: Commit the control boundary**

```bash
git add src/harpy/gui/controller.py tests/gui/test_controller.py
git commit -m "feat: add sine lab gate controller"
```

---

### Task 7: Implement Qt Audio Format Negotiation, Streaming, and Lifecycle

**Files:**
- Create: `src/harpy/gui/qt_audio.py`
- Create: `tests/gui/test_qt_audio.py`

**Interfaces:**
- Consumes: `AppConfig`, `AudioCommand`, `SineVoice`, and `SampleRingBuffer`.
- Produces: `audio_format_candidates(sample_rate_hz: int) -> tuple[QAudioFormat, ...]`, `choose_audio_format(device: QAudioDevice, sample_rate_hz: int) -> QAudioFormat | None`, `encode_mono_samples(samples: np.ndarray, audio_format: QAudioFormat) -> bytes`, `SynthAudioDevice`, and `QtAudioEngine` signals/methods: `status_changed(str, bool)`, `force_stop_requested()`, `start() -> None`, `submit(command: AudioCommand) -> None`, and `shutdown() -> None`.

- [ ] **Step 1: Write failing format-priority and PCM conversion tests**

Create the first part of `tests/gui/test_qt_audio.py`:

```python
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind
from harpy.gui.qt_audio import (
    QtAudioEngine,
    SynthAudioDevice,
    audio_format_candidates,
    choose_audio_format,
    encode_mono_samples,
)
from harpy.gui.visualizer import SampleRingBuffer
from harpy.pitch import Pitch


class FormatDevice:
    def __init__(self, accepted: set[tuple[int, QAudioFormat.SampleFormat]]) -> None:
        self.accepted = accepted
        self.probed: list[tuple[int, QAudioFormat.SampleFormat]] = []

    def isFormatSupported(self, audio_format: QAudioFormat) -> bool:
        value = (audio_format.channelCount(), audio_format.sampleFormat())
        self.probed.append(value)
        return value in self.accepted


def test_format_candidates_have_approved_priority() -> None:
    values = [
        (item.channelCount(), item.sampleFormat()) for item in audio_format_candidates(48_000)
    ]
    assert values == [
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
        (1, QAudioFormat.SampleFormat.Int16),
    ]


def test_choose_audio_format_stops_at_first_supported_candidate() -> None:
    device = FormatDevice({(2, QAudioFormat.SampleFormat.Int16)})
    selected = choose_audio_format(device, 48_000)
    assert selected is not None
    assert selected.channelCount() == 2
    assert selected.sampleFormat() is QAudioFormat.SampleFormat.Int16
    assert device.probed == [
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
    ]


def test_float_stereo_encoding_duplicates_mono() -> None:
    audio_format = audio_format_candidates(48_000)[0]
    encoded = encode_mono_samples(np.array([0.25, -0.5], dtype=np.float32), audio_format)
    decoded = np.frombuffer(encoded, dtype=np.float32)
    np.testing.assert_array_equal(decoded, [0.25, 0.25, -0.5, -0.5])


def test_int16_encoding_clips_rounds_and_duplicates() -> None:
    audio_format = audio_format_candidates(48_000)[2]
    encoded = encode_mono_samples(
        np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
        audio_format,
    )
    decoded = np.frombuffer(encoded, dtype=np.int16).reshape(-1, 2)
    np.testing.assert_array_equal(
        decoded,
        [[-32767, -32767], [-16384, -16384], [16384, 16384], [32767, 32767]],
    )
```

- [ ] **Step 2: Run the Qt audio helper tests and verify failure**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_qt_audio.py -v
```

Expected: collection fails because `harpy.gui.qt_audio` does not exist.

- [ ] **Step 3: Implement deterministic format probing and boundary conversion**

Start `src/harpy/gui/qt_audio.py` with:

```python
from __future__ import annotations

import queue
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QIODevice, QObject, Signal
from PySide6.QtMultimedia import (
    QAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from harpy.config import AppConfig
from harpy.gui.controller import AudioCommand, AudioCommandKind
from harpy.gui.visualizer import SampleRingBuffer
from harpy.synth.reference import SineVoice


def audio_format_candidates(sample_rate_hz: int) -> tuple[QAudioFormat, ...]:
    formats: list[QAudioFormat] = []
    for channels, sample_format in (
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
        (1, QAudioFormat.SampleFormat.Int16),
    ):
        audio_format = QAudioFormat()
        audio_format.setSampleRate(sample_rate_hz)
        audio_format.setChannelCount(channels)
        audio_format.setSampleFormat(sample_format)
        formats.append(audio_format)
    return tuple(formats)


def choose_audio_format(
    device: QAudioDevice,
    sample_rate_hz: int,
) -> QAudioFormat | None:
    return next(
        (
            audio_format
            for audio_format in audio_format_candidates(sample_rate_hz)
            if device.isFormatSupported(audio_format)
        ),
        None,
    )


def encode_mono_samples(samples: np.ndarray, audio_format: QAudioFormat) -> bytes:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    channels = audio_format.channelCount()
    interleaved = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Float:
        return interleaved.astype(np.float32, copy=False).tobytes()
    if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Int16:
        pcm = np.rint(np.clip(interleaved, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()
    raise ValueError("unsupported device sample format")
```

- [ ] **Step 4: Add failing QIODevice streaming and reset tests**

Append to `tests/gui/test_qt_audio.py`:

```python
def test_audio_device_applies_note_command_and_streams_one_stereo_block() -> None:
    history = SampleRingBuffer(capacity_frames=48_000)
    audio_format = audio_format_candidates(48_000)[0]
    source = SynthAudioDevice(DEFAULT_CONFIG, audio_format, history)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(60)))
    byte_count = DEFAULT_CONFIG.render.block_frames * audio_format.bytesPerFrame()
    encoded = source.readData(byte_count)
    decoded = np.frombuffer(encoded, dtype=np.float32).reshape(-1, 2)
    assert encoded and len(encoded) == byte_count
    np.testing.assert_array_equal(decoded[:, 0], decoded[:, 1])
    assert np.any(decoded != 0.0)
    assert np.any(history.snapshot(256) != 0.0)


def test_reset_command_makes_next_staging_block_exact_silence() -> None:
    history = SampleRingBuffer(capacity_frames=48_000)
    audio_format = audio_format_candidates(48_000)[0]
    source = SynthAudioDevice(DEFAULT_CONFIG, audio_format, history)
    byte_count = DEFAULT_CONFIG.render.block_frames * audio_format.bytesPerFrame()
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(60)))
    source.readData(byte_count)
    source.submit(AudioCommand(AudioCommandKind.RESET))
    decoded = np.frombuffer(source.readData(byte_count), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.zeros(decoded.size, dtype=np.float32))
```

- [ ] **Step 5: Implement the pull-mode QIODevice source**

Add `SynthAudioDevice` to `src/harpy/gui/qt_audio.py`:

```python
class SynthAudioDevice(QIODevice):
    def __init__(
        self,
        config: AppConfig,
        audio_format: QAudioFormat,
        sample_history: SampleRingBuffer,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._audio_format = audio_format
        self._sample_history = sample_history
        self._voice = SineVoice(config.tuning, config.render, config.patch)
        self._commands: queue.SimpleQueue[AudioCommand] = queue.SimpleQueue()
        self._staging = bytearray()
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def submit(self, command: AudioCommand) -> None:
        self._commands.put(command)

    def readData(self, maxlen: int) -> bytes:
        if maxlen <= 0:
            return b""
        while len(self._staging) < maxlen:
            self._drain_commands()
            mono = self._voice.render_block(self._config.render.block_frames)
            self._sample_history.append(mono)
            self._staging.extend(encode_mono_samples(mono, self._audio_format))
        output = bytes(self._staging[:maxlen])
        del self._staging[:maxlen]
        return output

    def writeData(self, data: bytes) -> int:
        return -1

    def isSequential(self) -> bool:
        return True

    def reset_after_sink_stop(self) -> None:
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        self._voice.reset()
        self._staging.clear()
        self._sample_history.clear()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command.kind is AudioCommandKind.NOTE_ON:
                if command.pitch is None:
                    raise RuntimeError("validated NOTE_ON command lost its pitch")
                self._voice.note_on(command.pitch)
            elif command.kind is AudioCommandKind.NOTE_OFF:
                self._voice.note_off()
            elif command.kind is AudioCommandKind.RESET:
                self._voice.reset()
                self._staging.clear()
```

- [ ] **Step 6: Add fake-sink lifecycle and hotplug tests**

Append these fakes and tests to `tests/gui/test_qt_audio.py`:

```python
class FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def connect(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def emit(self) -> None:
        for callback in self.callbacks:
            callback()


class DefaultDevice(FormatDevice):
    def isNull(self) -> bool:
        return False

    def description(self) -> str:
        return "Fake speakers"


class NullDevice(DefaultDevice):
    def __init__(self) -> None:
        super().__init__(set())

    def isNull(self) -> bool:
        return True


class FakeMediaDevices:
    def __init__(self, device: DefaultDevice) -> None:
        self.device = device
        self.audioOutputsChanged = FakeSignal()

    def defaultAudioOutput(self) -> DefaultDevice:
        return self.device


class FakeSink(QObject):
    stateChanged = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.source: object | None = None
        self.reset_count = 0
        self.current_error = QAudio.Error.NoError

    def start(self, source: object) -> None:
        self.source = source

    def reset(self) -> None:
        self.reset_count += 1

    def error(self) -> QAudio.Error:
        return self.current_error


def test_engine_starts_default_device_and_rebuilds_once_on_hotplug(qapp) -> None:
    device = DefaultDevice({(2, QAudioFormat.SampleFormat.Float)})
    media_devices = FakeMediaDevices(device)
    sinks: list[FakeSink] = []

    def sink_factory(*_args: object) -> FakeSink:
        sink = FakeSink()
        sinks.append(sink)
        return sink

    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(
        DEFAULT_CONFIG,
        history,
        media_devices=media_devices,
        sink_factory=sink_factory,
    )
    forced_stops: list[bool] = []
    engine.force_stop_requested.connect(lambda: forced_stops.append(True))
    engine.start()
    assert len(sinks) == 1
    assert sinks[0].source is not None
    media_devices.audioOutputsChanged.emit()
    assert forced_stops == [True]
    assert sinks[0].reset_count == 1
    assert len(sinks) == 2


def test_engine_disables_play_when_default_format_is_incompatible(qapp) -> None:
    media_devices = FakeMediaDevices(DefaultDevice(set()))
    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(DEFAULT_CONFIG, history, media_devices=media_devices)
    statuses: list[tuple[str, bool]] = []
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    assert statuses == [("Default output has no compatible 48 kHz format", False)]


def test_engine_disables_play_when_no_default_output_exists(qapp) -> None:
    media_devices = FakeMediaDevices(NullDevice())
    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(DEFAULT_CONFIG, history, media_devices=media_devices)
    statuses: list[tuple[str, bool]] = []
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    assert statuses == [("No default audio output", False)]


def test_sink_error_forces_stop_and_reports_failure(qapp) -> None:
    device = DefaultDevice({(2, QAudioFormat.SampleFormat.Float)})
    media_devices = FakeMediaDevices(device)
    sink = FakeSink()
    engine = QtAudioEngine(
        DEFAULT_CONFIG,
        SampleRingBuffer(capacity_frames=48_000),
        media_devices=media_devices,
        sink_factory=lambda *_args: sink,
    )
    forced_stops: list[bool] = []
    statuses: list[tuple[str, bool]] = []
    engine.force_stop_requested.connect(lambda: forced_stops.append(True))
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    sink.current_error = QAudio.Error.OpenError
    sink.stateChanged.emit(QAudio.State.StoppedState)
    assert forced_stops == [True]
    assert statuses[-1] == ("Audio output failed: OpenError", False)
```

- [ ] **Step 7: Implement the default-device QAudioSink lifecycle**

Add `QtAudioEngine` to `src/harpy/gui/qt_audio.py`. Keep the injected factory signature so tests never open a physical device:

```python
SinkFactory = Callable[[QAudioDevice, QAudioFormat, QObject], QAudioSink]


class QtAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(
        self,
        config: AppConfig,
        sample_history: SampleRingBuffer,
        *,
        media_devices: QMediaDevices | None = None,
        sink_factory: SinkFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._sample_history = sample_history
        self._media_devices = media_devices or QMediaDevices(self)
        self._sink_factory = sink_factory or (
            lambda device, audio_format, owner: QAudioSink(device, audio_format, owner)
        )
        self._sink: QAudioSink | None = None
        self._source: SynthAudioDevice | None = None
        self._disposing = False
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)

    def start(self) -> None:
        self._initialize_default_output()

    def submit(self, command: AudioCommand) -> None:
        if self._source is None:
            raise RuntimeError("audio output is unavailable")
        self._source.submit(command)

    def shutdown(self) -> None:
        self._dispose_sink()

    def _initialize_default_output(self) -> None:
        device = self._media_devices.defaultAudioOutput()
        if device.isNull():
            self.status_changed.emit("No default audio output", False)
            return
        audio_format = choose_audio_format(device, self._config.render.sample_rate_hz)
        if audio_format is None:
            self.status_changed.emit("Default output has no compatible 48 kHz format", False)
            return
        self._source = SynthAudioDevice(
            self._config,
            audio_format,
            self._sample_history,
            self,
        )
        self._sink = self._sink_factory(device, audio_format, self)
        self._sink.stateChanged.connect(self._on_sink_state_changed)
        sink = self._sink
        sink.start(self._source)
        if sink.error() is not QAudio.Error.NoError:
            self._on_sink_state_changed(QAudio.State.StoppedState)
            return
        if self._sink is not sink:
            return
        format_name = (
            "Float" if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Float else "Int16"
        )
        self.status_changed.emit(
            f"{device.description()} · 48 kHz · {audio_format.channelCount()} ch · {format_name}",
            True,
        )

    def _on_audio_outputs_changed(self) -> None:
        self.force_stop_requested.emit()
        self._dispose_sink()
        self._initialize_default_output()

    def _on_sink_state_changed(self, state: QAudio.State) -> None:
        if self._disposing or self._sink is None:
            return
        if state is QAudio.State.StoppedState and self._sink.error() is not QAudio.Error.NoError:
            error_name = self._sink.error().name
            self.force_stop_requested.emit()
            self._dispose_sink()
            self.status_changed.emit(f"Audio output failed: {error_name}", False)

    def _dispose_sink(self) -> None:
        self._disposing = True
        try:
            if self._sink is not None:
                self._sink.reset()
            if self._source is not None:
                self._source.reset_after_sink_stop()
                self._source.close()
            self._sink = None
            self._source = None
        finally:
            self._disposing = False
```

- [ ] **Step 8: Run all audio-adapter tests and style checks**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_qt_audio.py tests/gui/test_controller.py tests/synth -v
uv run ruff check src/harpy/gui/qt_audio.py tests/gui/test_qt_audio.py
uv run ruff format --check src/harpy/gui/qt_audio.py tests/gui/test_qt_audio.py
```

Expected: fake-device tests pass without opening WSLg audio, format probing stops at the first supported candidate, and reset produces exact zeros.

- [ ] **Step 9: Commit the Qt audio adapter**

```bash
git add src/harpy/gui/qt_audio.py tests/gui/test_qt_audio.py
git commit -m "feat: add native Qt audio adapter"
```

---

### Task 8: Build the Native Sine Lab Window and Interaction Tests

**Files:**
- Create: `src/harpy/gui/window.py`
- Create: `tests/gui/test_window.py`

**Interfaces:**
- Consumes: `AppConfig`, `LabController`, `QtAudioEngine`, `SampleRingBuffer`, `waveform_time_ms(frame_count: int, sample_rate_hz: int) -> np.ndarray`, and `spectrum_dbfs(samples: np.ndarray, sample_rate_hz: int, fft_frames: int = 4096, floor_dbfs: float = -120.0) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `HarpyWindow` with public test handles `pitch_slider`, `pitch_readout`, `tuning_readout`, `play_button`, `status_label`, `waveform_plot`, and `spectrum_plot`.

- [ ] **Step 1: Write failing native-widget behavior tests**

Create `tests/gui/test_window.py`:

```python
from PySide6.QtCore import QEvent, QObject, Qt, Signal

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind, LabController
from harpy.gui.visualizer import SampleRingBuffer
from harpy.gui.window import HarpyWindow


class FakeAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.commands: list[AudioCommand] = []
        self.shutdown_count = 0

    def submit(self, command: AudioCommand) -> None:
        self.commands.append(command)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def make_window(qtbot) -> tuple[HarpyWindow, FakeAudioEngine]:
    audio = FakeAudioEngine()
    controller = LabController(DEFAULT_CONFIG, audio.submit)
    history = SampleRingBuffer(capacity_frames=48_000)
    window = HarpyWindow(DEFAULT_CONFIG, controller, audio, history)
    qtbot.addWidget(window)
    audio.status_changed.emit("Fake speakers · 48 kHz", True)
    return window, audio


def test_window_starts_at_middle_c_with_fixed_patch_copy(qtbot) -> None:
    window, _ = make_window(qtbot)
    assert window.pitch_slider.value() == 60
    assert window.pitch_readout.text() == "C3 · MIDI 60 · 261.626 Hz"
    assert window.tuning_readout.text() == ("Concert A reference (MIDI 69) · 440.0 Hz")
    assert "Sine" in window.patch_label.text()
    assert "−12 dBFS" in window.patch_label.text()
    assert "1 ms" in window.patch_label.text()
    assert "600 ms" in window.patch_label.text()
    assert "−6 dB" in window.patch_label.text()


def test_hold_button_locks_pitch_and_sends_note_on_off(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    assert not window.pitch_slider.isEnabled()
    assert [command.kind for command in audio.commands] == [AudioCommandKind.NOTE_ON]
    qtbot.mouseRelease(window.play_button, Qt.MouseButton.LeftButton)
    assert window.pitch_slider.isEnabled()
    assert [command.kind for command in audio.commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]


def test_slider_updates_readout_when_idle(qtbot) -> None:
    window, _ = make_window(qtbot)
    window.pitch_slider.setValue(61)
    assert window.pitch_readout.text().startswith("C♯3 · MIDI 61")


def test_deactivation_forces_stop_and_raises_button(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    event = QEvent(QEvent.Type.WindowDeactivate)
    window.event(event)
    assert [command.kind for command in audio.commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not window.play_button.isDown()
    assert window.pitch_slider.isEnabled()


def test_audio_failure_disables_play_and_forces_stop(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    audio.force_stop_requested.emit()
    audio.status_changed.emit("Audio output failed: OpenError", False)
    assert not window.play_button.isEnabled()
    assert window.status_label.text() == "Audio output failed: OpenError"
    assert audio.commands[-1].kind is AudioCommandKind.RESET


def test_close_forces_stop_and_shuts_down_audio(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    window.close()
    assert audio.commands[-1].kind is AudioCommandKind.RESET
    assert audio.shutdown_count == 1


def test_plot_failure_stops_visual_timer_without_touching_audio(qtbot, monkeypatch) -> None:
    window, audio = make_window(qtbot)

    def fail_spectrum(*_args: object, **_kwargs: object) -> tuple[object, object]:
        raise RuntimeError("plot calculation failed")

    monkeypatch.setattr("harpy.gui.window.spectrum_dbfs", fail_spectrum)
    window._update_plots()
    assert not window._plot_timer.isActive()
    assert audio.commands == []
```

- [ ] **Step 2: Run window tests and verify failure**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_window.py -v
```

Expected: collection fails because `harpy.gui.window` does not exist.

- [ ] **Step 3: Implement the native widget hierarchy and styling**

Create `src/harpy/gui/window.py` with `HarpyWindow(QMainWindow)`. Build the public widgets with these fixed properties:

```python
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from harpy.config import AppConfig
from harpy.gui.controller import ControllerState, LabController
from harpy.gui.qt_audio import QtAudioEngine
from harpy.gui.visualizer import SampleRingBuffer, spectrum_dbfs, waveform_time_ms
from harpy.note_names import format_tuning_readout


class HarpyWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        controller: LabController,
        audio_engine: QtAudioEngine,
        sample_history: SampleRingBuffer,
    ) -> None:
        super().__init__()
        self._config = config
        self._controller = controller
        self._audio_engine = audio_engine
        self._sample_history = sample_history
        self.setWindowTitle("Harpy · Sine Lab")
        self.resize(1_280, 720)

        self.status_label = QLabel("Audio output not initialized")
        self.pitch_readout = QLabel()
        self.tuning_readout = QLabel(format_tuning_readout(config.tuning))
        self.patch_label = QLabel(
            "Sine · Peak −12 dBFS · A 1 ms · D 600 ms · S −6 dB · R 600 ms · Linear amplitude"
        )
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(
            config.keyboard.minimum_note.number,
            config.keyboard.maximum_note.number,
        )
        self.pitch_slider.setSingleStep(1)
        self.pitch_slider.setPageStep(1)
        self.play_button = QPushButton("Hold to Play")
        self.play_button.setObjectName("playButton")
        self.play_button.setMinimumHeight(76)
        self.waveform_plot = pg.PlotWidget(title="Recent waveform")
        self.spectrum_plot = pg.PlotWidget(title="Spectrum")
        self._waveform_curve = self.waveform_plot.plot(pen=pg.mkPen("#68d6ff", width=2))
        self._spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#b388ff", width=2))

        header = QHBoxLayout()
        title = QLabel("Harpy · Sine Lab")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)

        pitch_panel = QFrame()
        pitch_layout = QVBoxLayout(pitch_panel)
        pitch_layout.addWidget(self.pitch_readout)
        pitch_layout.addWidget(self.pitch_slider)
        pitch_layout.addWidget(self.tuning_readout)
        pitch_layout.addWidget(self.patch_label)
        pitch_layout.addWidget(self.play_button)

        plots = QHBoxLayout()
        plots.addWidget(self.waveform_plot, 1)
        plots.addWidget(self.spectrum_plot, 1)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(header)
        layout.addWidget(pitch_panel)
        layout.addLayout(plots, 1)
        self.setCentralWidget(root)

        self.pitch_slider.valueChanged.connect(self._on_note_changed)
        self.play_button.pressed.connect(self._on_play_pressed)
        self.play_button.released.connect(self._on_play_released)
        audio_engine.status_changed.connect(self._on_audio_status)
        audio_engine.force_stop_requested.connect(self._force_stop)

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(33)
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start()
        self._audio_playable = False
        self._apply_state(controller.state)
        self._apply_style()
```

Implement `_apply_style()` with this concrete QSS so the first screen is capture-ready:

```python
    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111318; color: #eef2f7; }
            QFrame { background: #181c23; border: 1px solid #2a313d; border-radius: 12px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel { font-size: 15px; }
            QPushButton#playButton {
                background: #68d6ff; color: #071017; border: 0; border-radius: 12px;
                font-size: 20px; font-weight: 700; padding: 16px;
            }
            QPushButton#playButton:pressed { background: #b388ff; }
            QPushButton#playButton:disabled { background: #343b46; color: #7f8998; }
            QSlider::groove:horizontal { height: 8px; background: #2a313d; border-radius: 4px; }
            QSlider::handle:horizontal {
                width: 22px; margin: -8px 0; background: #68d6ff; border-radius: 11px;
            }
            """
        )
```

- [ ] **Step 4: Implement interaction, forced-stop, and plot update methods**

Add these methods to `HarpyWindow`:

```python
def _apply_state(self, state: ControllerState) -> None:
    self.pitch_slider.blockSignals(True)
    self.pitch_slider.setValue(state.note.number)
    self.pitch_slider.blockSignals(False)
    self.pitch_readout.setText(state.readout)
    self.pitch_slider.setEnabled(state.selector_enabled)
    self.play_button.setEnabled(self._audio_playable)


def _on_note_changed(self, number: int) -> None:
    self._apply_state(self._controller.set_note(number))


def _on_play_pressed(self) -> None:
    self._apply_state(self._controller.press_play())


def _on_play_released(self) -> None:
    self._apply_state(self._controller.release_play())


def _force_stop(self) -> None:
    self.play_button.setDown(False)
    self._sample_history.clear()
    self._apply_state(self._controller.force_stop())


def _on_audio_status(self, message: str, playable: bool) -> None:
    self._audio_playable = playable
    self.status_label.setText(message)
    self._apply_state(self._controller.state)


def _update_plots(self) -> None:
    try:
        waveform = self._sample_history.snapshot(2_048)
        time_ms = waveform_time_ms(waveform.size, self._config.render.sample_rate_hz)
        self._waveform_curve.setData(time_ms, waveform)
        frequencies, levels = spectrum_dbfs(
            self._sample_history.snapshot(4_096),
            self._config.render.sample_rate_hz,
        )
        self._spectrum_curve.setData(frequencies, levels)
        self.waveform_plot.setLabel("bottom", "Time", units="ms")
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Level", units="dBFS")
        self.spectrum_plot.setYRange(-120.0, 0.0)
        self.spectrum_plot.setXRange(0.0, 2_000.0)
    except (FloatingPointError, RuntimeError, ValueError) as error:
        self._plot_timer.stop()
        message = f"Visualization disabled: {error}"
        self.waveform_plot.setTitle(message)
        self.spectrum_plot.setTitle(message)


def event(self, event: QEvent) -> bool:
    if event.type() is QEvent.Type.WindowDeactivate:
        self._force_stop()
    return super().event(event)


def closeEvent(self, event: QCloseEvent) -> None:
    self._force_stop()
    self._audio_engine.shutdown()
    event.accept()
```

- [ ] **Step 5: Run offscreen widget and regression checks**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_window.py tests/gui/test_controller.py tests/gui/test_visualizer.py -v
QT_QPA_PLATFORM=offscreen uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: all widget tests pass without a browser or physical audio device.

- [ ] **Step 6: Commit the native window**

```bash
git add src/harpy/gui/window.py tests/gui/test_window.py
git commit -m "feat: add native sine lab window"
```

---

### Task 9: Compose the Application, Document uv Usage, and Verify the Complete Slice

**Files:**
- Create: `src/harpy/gui/app.py`
- Create: `tests/gui/test_app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: every component from Tasks 1 through 8.
- Produces: `LabRuntime`, `build_runtime(app: QApplication, config: AppConfig = DEFAULT_CONFIG, audio_factory: AudioFactory = QtAudioEngine) -> LabRuntime`, and console entry point `main(argv: list[str] | None = None) -> int` used by `uv run harpy`.

- [ ] **Step 1: Write a failing composition test with an injected audio engine**

Create `tests/gui/test_app.py`:

```python
from PySide6.QtCore import QObject, Signal

from harpy.config import AppConfig, DEFAULT_CONFIG
from harpy.gui.app import build_runtime
from harpy.gui.controller import AudioCommand
from harpy.gui.visualizer import SampleRingBuffer


class ComposedAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(self, config: AppConfig, history: SampleRingBuffer) -> None:
        super().__init__()
        self.config = config
        self.history = history
        self.commands: list[AudioCommand] = []
        self.started = False

    def submit(self, command: AudioCommand) -> None:
        self.commands.append(command)

    def start(self) -> None:
        self.started = True
        self.status_changed.emit("Fake speakers · 48 kHz", True)

    def shutdown(self) -> None:
        self.started = False


def test_build_runtime_owns_connected_components(qapp) -> None:
    created: list[ComposedAudioEngine] = []

    def factory(config: AppConfig, history: SampleRingBuffer) -> ComposedAudioEngine:
        engine = ComposedAudioEngine(config, history)
        created.append(engine)
        return engine

    runtime = build_runtime(qapp, DEFAULT_CONFIG, audio_factory=factory)
    assert runtime.config is DEFAULT_CONFIG
    assert runtime.audio is created[0]
    assert runtime.window.pitch_slider.value() == 60
    runtime.audio.start()
    assert runtime.audio.started
    assert runtime.window.play_button.isEnabled()
    runtime.window.close()
```

- [ ] **Step 2: Run the app composition test and verify failure**

Run:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/gui/test_app.py -v
```

Expected: collection fails because `harpy.gui.app` does not exist.

- [ ] **Step 3: Implement runtime composition and the console entry point**

Create `src/harpy/gui/app.py`:

```python
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QMessageBox

from harpy.config import AppConfig, DEFAULT_CONFIG
from harpy.gui.controller import LabController
from harpy.gui.qt_audio import QtAudioEngine
from harpy.gui.visualizer import SampleRingBuffer
from harpy.gui.window import HarpyWindow

AudioFactory = Callable[[AppConfig, SampleRingBuffer], QtAudioEngine]


@dataclass(slots=True)
class LabRuntime:
    config: AppConfig
    sample_history: SampleRingBuffer
    audio: QtAudioEngine
    controller: LabController
    window: HarpyWindow


def build_runtime(
    app: QApplication,
    config: AppConfig = DEFAULT_CONFIG,
    *,
    audio_factory: AudioFactory = QtAudioEngine,
) -> LabRuntime:
    sample_history = SampleRingBuffer(config.render.sample_rate_hz)
    audio = audio_factory(config, sample_history)
    controller = LabController(config, audio.submit)
    window = HarpyWindow(config, controller, audio, sample_history)
    app.aboutToQuit.connect(controller.force_stop)
    app.aboutToQuit.connect(audio.shutdown)
    return LabRuntime(config, sample_history, audio, controller, window)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    app = QApplication.instance() or QApplication(arguments)
    try:
        runtime = build_runtime(app)
    except (TypeError, ValueError) as error:
        QMessageBox.critical(None, "Harpy configuration error", str(error))
        return 2
    runtime.window.show()
    runtime.audio.start()
    return app.exec()
```

- [ ] **Step 4: Update README with the exact native-lab workflow and scope**

Append this section after the existing native sine lab implementation-plan link:

````markdown
## Native sine lab

The first implementation slice is a native PySide6 application: one deterministic NumPy sine voice, a MIDI-note selector centered on C3/MIDI 60, the fixed 1 ms / 600 ms / -6 dB / 600 ms linear envelope, and waveform/spectrum plots.

```bash
uv sync --dev
uv run harpy
```

Run the automated checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

This slice intentionally contains no browser UI, MIDI, imported-audio editing, RL environment, database, or third-party synth engine.
````

- [ ] **Step 5: Run the complete automated verification suite**

Run:

```bash
uv sync --dev --locked
QT_QPA_PLATFORM=offscreen uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run python -c "from harpy.gui.app import main; assert callable(main)"
git diff --check
```

Expected: dependency sync honors `uv.lock`; all tests, Ruff checks, import smoke test, and whitespace checks pass.

- [ ] **Step 6: Run the native WSLg audio and visual smoke test**

Confirm WSLg variables and launch:

```bash
test -n "$DISPLAY"
test -n "$WAYLAND_DISPLAY"
uv run harpy
```

Verify all seven observations before closing the window:

1. A native window opens and no browser launches.
2. The initial readout is `C3 · MIDI 60 · 261.626 Hz` and concert A is 440.0 Hz.
3. Holding Play produces a stable sine and locks the note slider.
4. Releasing Play gives an audible 600 ms release and unlocks the slider.
5. Moving the idle slider advances in semitones and updates label/frequency.
6. Waveform and spectrum update smoothly without audible underruns.
7. Deactivating or closing the window leaves no stuck tone.

If the default output is unavailable or rejects every approved 48 kHz format, capture the exact status text and treat the milestone as incomplete; do not silently change the core sample rate.

- [ ] **Step 7: Commit the completed native slice**

```bash
git add src/harpy/gui/app.py tests/gui/test_app.py README.md uv.lock
git commit -m "feat: complete native sine lab"
```

- [ ] **Step 8: Record final evidence**

Run:

```bash
git status --short
git log --oneline -10
uv run pytest -q
```

Expected: the worktree is clean, the nine task commits are visible, and the final test summary is passing.
