# Harpy Milestone A: Foundation and Workbench Design

Date: 2026-08-07

Status: implemented; completion acceptance remains pending native visual scaling and human hearing checks

## Purpose

Milestone A replaces the current native sine lab with a small, credible scientific
instrument built on stable engine and analysis contracts. It keeps the now-working
audio path, removes GUI concepts from the synthesis core, and makes the native UI one
client of the same API that a future experiment harness will use to generate procedural
source material.

This is a deliberate replacement of the current product surface, not a sequence of
cosmetic additions to it. The milestone is complete only when a researcher can load a
patch, choose a continuous frequency, audition it, and inspect an accurately labelled
waveform and spectrum without understanding Qt, MIDI numbers, or audio-device formats.

## Product principles

1. **The UI is not the API.** The synth engine, patch schema, and analysis results are
   usable without constructing a Qt application. A future Gym harness may use the
   engine to generate deterministic audio fixtures, but the RL actor manipulates the
   resulting audio asset rather than controlling the synth.
2. **Frequency is physical.** The engine receives finite positive hertz values. MIDI,
   note names, semitones, and cents are conversions outside the DSP engine.
3. **A patch is not a performance.** Oscillator and envelope settings describe the
   sound. The frequency being played and whether a note is held describe a performance.
4. **Measurements have declared units.** Waveform amplitude is full scale, spectrum
   level is dBFS, and configured frequency is distinct from measured peak frequency.
5. **No healthy-state chrome.** When audio works, the interface says nothing about the
   device. When audio fails, it reports a specific error rather than failing silently.
6. **No application-managed storage.** Patch JSON files are explicit transient inputs
   and outputs. Harpy owns no database, patch library, recent-file list, autosave, or
   experiment history.
7. **Future capability must not distort the first proof.** The core must not depend on
   sine-specific UI vocabulary, but Milestone A implements exactly one oscillator and
   one simultaneous voice.

## Scope

### Included

- A direct-hertz, deterministic, monophonic `SynthEngine`.
- A `SynthPatch` independent of the selected or played frequency.
- A sine-only oscillator configuration whose serialized form names `"sine"`.
- The existing linear ADSR semantics and safe `-12 dBFS` output gain.
- Continuous live frequency selection in the GUI over C2 through C4 under Harpy's
  Ableton-style note labels: approximately `130.812783` through `523.251131 Hz` at
  A = 440 Hz.
- Broader engine/API frequency support for any finite positive value strictly below
  Nyquist. The GUI range is not an engine bound.
- A pure analysis contract shared by the GUI and a future experiment harness.
- A triggered oscilloscope-style Waveform view with explicit full-scale amplitude.
- A logarithmic-frequency Spectrum view with a measured peak marker and readout.
- Explicit Empty, Measuring, Live, and Captured measurement states plus a Clear action.
- Stateless JSON patch load and save through native file dialogs.
- A complete replacement of the current main-window layout and copy.
- Automated deterministic tests and a native WSLg playback/visual smoke test.

### Deferred to Milestone B

- Editing attack, decay, sustain, and release from the GUI.
- Bézier or other envelope curvature.
- A graphical envelope authoring surface.
- Patch-browser, preset-management, or application-owned persistence features.

### Deferred until after the sine-only Gym proof

- Gymnasium or another RL environment.
- Model adapters, tool definitions, observations for an agent, rewards, training, or
  benchmark reporting.
- Oscillator morphing, saw, square, triangle, or wavetable synthesis.
- Polyphony, chords, voice allocation, or per-note velocity.
- Random patch generation, dataset manifests, experiment storage, or gauntlet levels.
- MIDI input, JUCE, C++, plugins, DAW integration, or native Windows packaging.

## Conceptual model

```text
SynthPatch ───────┐
                  v
Note event ─> SynthEngine ─> mono float32 samples ─> analyze ─> AudioObservation
                      │                                  │
                      v                                  ├─> native workbench
                Qt audio adapter                         └─> future experiment harness
```

Qt playback and a future experiment harness are peers. Neither reaches into the other,
and neither extracts numerical state from rendered pixels. In the first retuning Gym,
the harness may call `SynthEngine` to create immutable candidate/reference audio and may
call the analyzer to score or inspect it. The actor receives only the observations and
audio-editing actions defined by that later Gym; it does not receive synth patch or
oscillator controls.

### `SynthPatch`

`SynthPatch` is immutable, validated, and JSON-serializable. It describes sound identity
without containing a note, root frequency, gate state, audio device, or GUI setting.

Milestone A fields are conceptually:

```text
SynthPatch
  oscillator
    type                 = "sine"
  envelope
    attack_seconds       = 0.001
    decay_seconds        = 0.600
    sustain_db           = -6.0
    release_seconds      = 0.600
    curve                = "linear_amplitude"
  output_gain_dbfs       = -12.0
```

The gain remains explicit even though it is not a primary GUI control. Without it,
waveform amplitude and spectrum level would depend on an unexplained hard-coded value.

The oscillator is named rather than represented as `shape = 0.0`. A normalized shape
coordinate is not added until its waveform stops, interpolation, band-limiting, and
serialization semantics are specified. This avoids freezing an arbitrary morph path
into the public patch format.

### `RenderConfig`

`RenderConfig` contains machine/render facts rather than patch identity:

```text
RenderConfig
  sample_rate_hz         = 48000
  block_frames           = 256
  channels               = 1
  internal_dtype         = float32
```

The engine remains mono internally. The Qt boundary duplicates or converts samples for
the negotiated output-device format. Milestone A keeps the defaults stable; it does not
add main-window controls for sample rate or block size.

### Note events and engine state

The frequency selected in the workbench is application/controller state. It is not
stored in `SynthPatch`, and an idle engine does not own a "next frequency." The engine
stores only the active voice's frequency, phase, envelope, and gate/stage state.

The public engine behavior is:

```python
engine.note_on(frequency_hz)
engine.retune(frequency_hz)
engine.note_off()
engine.replace_patch(patch)
engine.render(frame_count)
engine.reset()
```

- `note_on` validates the physical frequency, resets oscillator phase, and starts the
  configured envelope.
- `retune` changes an attacking, decaying, sustaining, or releasing oscillator's phase
  increment without resetting phase or retriggering the envelope. Calling it while the
  voice is idle raises a state error; the workbench controller simply retains its own
  selected frequency until the next `note_on`. It applies the same finite, positive,
  strictly-below-Nyquist frequency validation as `note_on` before mutating engine state.
- `note_off` starts release from the last emitted envelope level.
- `replace_patch` atomically resets the voice and replaces its immutable patch. The next
  `render` is exact silence until a later `note_on`.
- `render` returns exactly the requested number of mono `float32` samples and is the
  authoritative path for both live and future offline rendering.
- `reset` immediately clears oscillator, envelope, gate, and voice state and causes
  subsequent rendering to emit exact silence.

Repeated `note_on` remains a monophonic retrigger in Milestone A. A future polyphonic
engine may return voice identifiers, but no voice allocator or speculative abstraction
is added now.

At the Qt boundary, patch replacement is one queued `REPLACE_PATCH` command, not a GUI
mutation of the audio-thread engine. Before the next block is rendered, the audio source
drains that command, calls `replace_patch`, discards staged encoded bytes, and clears the
live sample history. No output block may combine samples from the old and new patches.

### Tuning and musical conversion

The DSP engine does not import the tuning or note-name module. A separate pure musical
conversion boundary owns:

- tuning reference, defaulting to A = 440 Hz;
- hertz to continuous MIDI coordinate;
- MIDI coordinate to hertz;
- nearest note name under Harpy's Ableton-style octave convention;
- cents offset from the nearest equal-tempered note.

Hertz is the authoritative engine input. MIDI remains useful for future dataset import
and musical actions, but its integer number is not displayed in the primary UI.

## Patch JSON

Milestone A defines a versioned, human-readable JSON document:

```json
{
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
```

Requirements:

- Loading constructs and validates a complete new `SynthPatch` before changing live
  state.
- An invalid file leaves the current patch, frequency, and captured observation intact
  and reports the failing field or unsupported schema version.
- A successful load force-stops audio, replaces the in-memory patch, clears the current
  observation, and preserves the selected performance frequency.
- Save writes the active patch only. It does not include render settings, selected
  frequency, window geometry, audio device, or analysis state. A save completes through
  a fully written temporary sibling and atomic replacement, so a failed overwrite keeps
  the prior destination intact.
- Load and Save As use native file dialogs. Harpy does not remember the path after the
  operation and does not create a recent-file list.
- JSON serialization is deterministic: equivalent patches produce equivalent parsed
  data, field names and units are explicit, and load/save round-trips preserve values.
- Milestone B extends the envelope portion deliberately; it does not overload the
  linear curve string with undefined control points.

The v1 grammar is closed and strict:

- The root object requires exactly `schema_version`, `oscillator`, `envelope`, and
  `output_gain_dbfs`; unknown or missing keys are rejected.
- `schema_version` is the JSON integer `1`; booleans, floats, and strings are rejected.
- `oscillator` requires exactly `type`, whose only accepted value is `"sine"`.
- `envelope` requires exactly `attack_seconds`, `decay_seconds`, `sustain_db`,
  `release_seconds`, and `curve`; unknown or missing keys are rejected.
- Attack, decay, and release are finite JSON numbers greater than zero. Composing a
  patch with a render config also requires each duration to round to at least one frame.
- Sustain is a finite JSON number no greater than `0 dB`. It is relative to the
  envelope's attack peak, not dBFS: the default `-6 dB` sustain combined with the
  `-12 dBFS` patch gain yields approximately `-18 dBFS` steady-state peak level.
- Output gain is a finite JSON number no greater than `0 dBFS`.
- `curve` must be `"linear_amplitude"` in schema version 1.
- JSON booleans are never accepted as numbers. Non-standard `NaN`, positive/negative
  infinity, duplicate object keys, and trailing non-whitespace content are rejected.

Save output is reproducible UTF-8 text: keys use the order shown in the example, nested
keys use their shown order, indentation is two spaces, finite floats use Python's
shortest round-trippable decimal representation, and the file ends with one newline.
Loading is order-independent even though saving is canonical.

## Shared analysis contract

Pure analysis moves out of the GUI package. Its public result is an immutable
`AudioObservation` containing numerical data, not presentation objects:

```text
AudioObservation
  has_signal: bool
  waveform_samples: float32 array
  waveform_time_ms: float64 array
  spectrum_frequency_hz: float64 array
  spectrum_level_dbfs: float64 array
  peak_amplitude_fs: float | None
  peak_frequency_hz: float | None
  peak_level_dbfs: float | None
```

The pure API is `analyze(samples, sample_rate_hz, config) -> AudioObservation`; it does
not read widgets, the Qt audio sink, or screen pixels. Milestone A defaults are:

```text
AnalysisConfig
  waveform_window_seconds = 0.050
  fft_frames              = 16384
  spectrum_min_hz         = 20.0
  spectrum_max_hz         = 20000.0
  spectrum_floor_dbfs     = -120.0
  window                  = "hann"
```

`AnalysisConfig` requires `0 < spectrum_min_hz < spectrum_max_hz <= Nyquist`, an even
FFT length of at least four, a positive waveform duration no longer than the FFT
window, and a finite non-positive floor. The composed default therefore requires a
sample rate above 40 kHz and is valid at 48 kHz. A non-default render config must
provide compatible analysis bounds rather than silently clamping them.

Waveform duration converts to frames using nearest-integer, half-up rounding:
`floor(seconds * sample_rate_hz + 0.5)`, matching envelope duration conversion. The
result must be at least one frame and no greater than `fft_frames`. For `F` returned
waveform samples, timestamps are exactly `arange(F) * 1000 / sample_rate_hz`, so the
first displayed sample is at `0 ms`.

Analysis semantics:

- Signal validity is determined from the untriggered most-recent waveform-duration
  slice of the FFT input. Exact silence and slices whose peak absolute amplitude does
  not exceed `1e-6` produce `has_signal = false`, empty plot arrays, and `None` peak
  values. Older transients elsewhere in the FFT interval cannot make a silent current
  waveform appear live.
- Waveform samples retain full-scale amplitude. Analysis does not normalize each
  capture to its own peak.
- `peak_amplitude_fs` is the greatest absolute waveform sample in the observation and
  is `None` when `has_signal` is false.
- Spectrum magnitudes use Hann-window coherent-gain correction and one-sided amplitude
  normalization, so a bin-centered sine's peak is reported in dBFS.
- The zero-hertz FFT bin is excluded from the log-view spectrum.
- Peak detection considers positive-frequency bins within the declared spectrum range.
  Quadratic interpolation fits the winning bin and its two neighbors in log-magnitude
  dB space. The parabola vertex supplies both refined frequency and peak level; edge
  bins use their bin-center frequency and level.
- Peak frequency is a measurement. It is never populated from the configured engine
  frequency.
- If the time-domain input exceeds the signal threshold but has no in-range spectral
  peak, `has_signal` remains true while spectral arrays contain the configured in-range
  bins and both spectral peak values are `None`.

Analysis consumes at least `fft_frames` samples and uses the most recent exact
`fft_frames`; shorter input is a validation error rather than an implicitly zero-padded
measurement. At 48 kHz the default spectrum therefore represents the trailing
`341.333 ms`. The Waveform is selected from that same interval and always contains
exactly 2,400 samples at the default 50 ms duration.

All observation arrays are one-dimensional, copied out of working buffers, set
non-writeable before publication, and owned by the observation. Waveform sample/time
arrays have equal length; spectrum frequency/level arrays have equal length and strictly
positive increasing frequencies.

The future Gym harness may use this object for environment construction or scoring, but
the actor-facing observation is a separate later decision. Milestone A does not shape
the data around a model.

## Live capture state

The native application owns a bounded, thread-safe history of at least `fft_frames`
mono samples. The analyzer remains pure. Clear is an action, not a fifth kind of data;
a small capture coordinator publishes four presentation states:

1. **Empty** — no current capture exists and no voice is producing a new one.
2. **Measuring** — samples are arriving, but fewer than `fft_frames` have accumulated
   since the last note/reset/clear generation.
3. **Live** — a complete valid trailing window exists and observations refresh.
4. **Captured** — a previous Live observation is retained after the most recent
   waveform-duration slice becomes sub-threshold.

Transitions are deterministic:

| Event | Prior state | Result |
| --- | --- | --- |
| application launch, patch replacement, engine reset, or device failure | any | clear history/capture and enter Empty |
| `note_on` or monophonic retrigger | any | clear history/capture, start a new generation, and enter Measuring |
| enough samples for one FFT window with signal | Measuring or Live | publish observation and enter/remain Live |
| enough samples for one FFT window, but the trailing waveform slice is sub-threshold before any Live observation | Measuring | discard the unobservable attempt and enter Empty |
| the trailing waveform-duration slice is sub-threshold after Live | Live | retain the last valid observation and enter Captured |
| Clear while the engine is idle | any | clear history/capture, increment generation, and enter Empty |
| Clear while attack/decay/sustain/release is active | any | clear history/capture, increment generation, and enter Measuring |
| `note_off` | Measuring or Live | keep the state while release samples continue; Captured is entered only by the complete sub-threshold-window rule |

Each history clear increments a generation identifier. Generation is capture metadata,
not part of the pure `AudioObservation`: the coordinator creates a private
`CapturedObservation(generation, observation)` wrapper. It publishes the wrapped result
only if the generation still matches, preventing an in-flight pre-reset result from
restoring stale data after patch load, Clear, reset, or device failure.

The spectrum is a trailing 341.333 ms measurement, so immediately after live retuning
it may legitimately contain both old and new frequencies. After 16,384 newly rendered
frames at a stable frequency, none of the prior frequency remains in the analysis
window and the measured peak must meet the declared clean-sine tolerance.

This state machine is application logic, not part of the DSP engine or
`AudioObservation`.

## Native workbench

### Identity and hierarchy

The product is Harpy. The operating-system window title is `Harpy`; the content does not
repeat `Harpy`, `Sine Lab`, `Native Sine Lab`, or `Synth Engine` as a decorative heading.

The visual hierarchy is:

1. continuous frequency selection and compact Play;
2. Waveform and Spectrum measurements;
3. read-only active-patch facts and adjacent JSON actions.

There is no healthy audio-output status. Device description, channel count, negotiated
sample format, and backend name do not appear on the main surface.

### Frequency knob and transport strip

The top strip contains:

- a purpose-built rotary pitch knob;
- a large directly editable frequency value with `Hz` suffix;
- a secondary derived nearest-note label and cents offset, without a MIDI number;
- a normal-sized press-and-hold `Play` button.

Patch file actions and Clear do not compete with transport in this strip. `Load Patch`
and `Save Patch As` sit with Patch facts; Clear sits in the shared measurement header.

The knob is continuous and logarithmic in frequency so equal angular distances
represent equal musical intervals. Its endpoints and center are computed from the
runtime tuning reference rather than hard-coded hertz literals. At the default A =
440 Hz reference it maps C2 to
the minimum (`130.812783 Hz`), C3 to the exact center (`261.625565 Hz`), and C4 to the
maximum (`523.251131 Hz`) under Harpy's Ableton-style note naming. It does not snap to
semitones and does not expose a MIDI value.

The knob uses vertical drag interaction: upward increases pitch and downward decreases
it without jumping to the pointer's absolute angle. Shift-drag applies one tenth of the
normal sensitivity, arrow keys adjust by one cent, Shift+arrow adjusts by one tenth of
a cent, and double-click restores C3. The mouse wheel uses the same one-cent increment.
The visual indicator never wraps across the endpoint gap.

The frequency field displays three decimal places and accepts finite decimal input with
up to six fractional places. Enter commits only values within the GUI's C2-C4 range;
invalid or out-of-range input receives an inline error treatment and leaves controller
and engine state unchanged. Escape restores the last valid value. The note label is
formatted as, for example, `C3 +0.0¢` from the configured concert reference.

The knob and direct entry share the GUI range. The engine and JSON/API are not
restricted to it, allowing later bass benchmarks without changing DSP primitives. The
range will become an explicit workbench preference when a concrete benchmark requires
it; Milestone A does not add a preferences system or keyboard mode.

Frequency edits while Play or release is active call `retune` and are audible without
an envelope retrigger or oscillator-phase reset. Spacebar performs press-and-hold gating
when focus is not inside the frequency editor.

### Waveform

The left measurement panel is titled `Waveform`.

- Y axis: `Amplitude (FS)`, fixed at `-1.0` through `+1.0`.
- X axis: `Time (ms)`, fixed at `0` through `50` for the default analysis config.
- The view begins at sample `i + 1` for the most recent pair satisfying
  `samples[i] <= 0 < samples[i + 1]` that still has a complete 50 ms window after it,
  yielding a deterministic stable oscilloscope display for a sustained periodic signal.
- If no crossing is available, the most recent complete 50 ms window is shown without
  fabricating data.
- Mouse panning, wheel zoom, and pyqtgraph context menus are disabled.
- Empty state shows `Hold Play to inspect the signal.` with no zero trace.
- Captured state retains the last valid waveform until Clear, patch load, or reset.

The fixed full-scale axis makes the configured `-12 dBFS` peak visible as approximately
`±0.251`. A small readout reports `Peak 0.251 FS` from `peak_amplitude_fs`. The plot does
not autoscale in a way that hides absolute level.

### Spectrum

The right measurement panel is titled `Spectrum` and receives slightly more horizontal
space than Waveform.

- X axis: `Frequency (Hz)` in logarithmic mode, fixed from `20 Hz` through `20 kHz`.
- Y axis: `Level (dBFS)`, fixed from `-120` through `0`.
- Useful ticks and grid lines emphasize `20`, `50`, `100`, `200`, `500`, `1k`, `2k`,
  `5k`, `10k`, and `20k`.
- Mouse panning, wheel zoom, and pyqtgraph context menus are disabled.
- A visible marker identifies the measured spectral maximum.
- A readout reports measured peak frequency and level separately from configured
  frequency, for example `Peak 261.6 Hz · -12.0 dBFS`.
- Empty state shows `Hold Play to inspect the signal.` with no synthetic
  `-120 dBFS` line.
- Captured state retains the last valid spectrum until Clear, patch load, or reset.

One shared measurement header above the panels contains the compact state text and
Clear action. It shows `Measuring…`, `Live`, or `Captured` when those distinctions are
meaningful and shows no device/backend health. Empty state relies on the plot prompt
rather than an additional `Empty` badge.

### Patch facts

Milestone A does not create the Milestone B editor. It shows the active patch as compact,
separately labelled read-only values rather than one compressed sentence:

- Oscillator: Sine
- Attack: 1 ms
- Decay: 600 ms
- Sustain: -6 dB
- Release: 600 ms
- Curve: Linear amplitude

These values update after loading a patch. Milestone B turns this area into the envelope
authoring surface rather than introducing a second competing representation.

`Load Patch` and `Save Patch As` are compact secondary actions inside this Patch area,
not primary transport controls.

### Visual language and sizing

- Restrained charcoal neutral surfaces with low-contrast separators.
- One signal accent for waveform/active transport and one measurement accent for the
  spectrum marker.
- Tabular numerals for frequency, time, amplitude, and spectral values.
- Compact controls, consistent spacing, and no oversized decorative headings or CTA.
- No decorative animation. Motion exists only when it communicates live measurements
  or envelope stage.
- Primary text and editable values have at least `4.5:1` contrast against their
  background; secondary text, axes, focus indicators, and meaningful graphics have at
  least `3:1`. Primary numeric text is at least 20 logical pixels and ordinary labels
  are at least 12 logical pixels.
- The minimum logical window is `1024 x 640`; the default is `1280 x 720`. The transport
  region consumes at most 144 logical pixels of height, Patch facts at most 96, and the
  measurement region retains at least 320 logical pixels of plot-canvas height.
- The default window is fully usable within a 1280 x 720 client area. On the current
  machine's 125% Windows scale (`AppliedDPI = 120`) and 1536 x 864 effective desktop,
  the entire outer OS window frame—including title bar and borders—must remain on-screen.
  The same full-frame check is repeated at 100% scaling; neither mode may rely on OS
  clipping or manual maximization.
- The layout uses explicit minimum sizes and stretch behavior so neither plot nor its
  axes are clipped at the default window size.

## Audio failure behavior

Successful audio initialization produces no status label, indicator, or toast. If the
default output cannot be opened or loses compatibility:

- any active voice is force-stopped;
- Play becomes unavailable;
- Harpy presents one concise error naming the failed operation and actionable device
  fact when available;
- repeated backend state notifications do not create repeated dialogs;
- successful reinitialization after an output-device change restores Play without
  adding permanent healthy-state chrome.

The existing Qt device negotiation, PCM conversion, hotplug handling, and pull-device
availability fix remain at the platform boundary.

## Code boundaries

The Milestone A implementation must end with these responsibilities and no dead
compatibility layers:

- **synth configuration** — immutable patch and render values plus validation;
- **synth engine** — oscillator, envelope, performance state, and deterministic render;
- **music conversion** — tuning, hertz, MIDI coordinates, note names, and cents;
- **analysis** — pure waveform/spectrum observations and peak measurement;
- **live capture** — bounded concurrent history and empty/live/captured state;
- **Qt audio adapter** — device formats, PCM encoding, `QIODevice`, sink lifecycle;
- **workbench controller** — semantic frequency, patch, transport, and capture state;
- **workbench window** — Qt widgets, formatting, layout, and user actions.

Targeted current-code corrections include:

- Rename the authoritative `synth/reference.py` concept to engine/voice vocabulary.
- Remove the top-level configuration dependency on `gui.specs.KeyboardViewSpec`.
- Move pure waveform and FFT analysis out of `gui/visualizer.py`.
- Keep the bounded live ring buffer outside the pure analyzer.
- Replace the controller's MIDI-only state and preformatted readout with semantic hertz
  and gate state; the UI formats note labels.
- Keep `qt_audio.py` as one platform adapter unless a concrete second backend makes a
  split useful. Its device lifecycle complexity is real, not automatically poor design.
- Remove superseded classes, files, tests, copy, and compatibility aliases once their
  replacements are verified.

The current production package contains 1,106 physical Python lines excluding tests.
Milestone A has no arbitrary line-count target. Review instead enforces one clear
responsibility per module, absence of dead compatibility code, and a documented public
surface small enough to explain without reading Qt internals.

`from __future__ import annotations` may remain consistently where useful; it postpones
type-annotation evaluation and has no application behavior. Empty package `__init__.py`
files may remain as conventional package markers.

## Testing and verification

Tests are part of the public research artifact and remain in the repository. Milestone
A requires:

### Engine and model tests

- `SynthPatch` validation and deterministic JSON round-trip.
- Invalid and unsupported JSON fails atomically without changing active state.
- Direct-hertz validation rejects non-finite, non-positive, and at/above-Nyquist values.
- `note_on`, live/release `retune`, `note_off`, `replace_patch`, and `reset` semantics.
- Live retuning preserves phase and envelope continuity.
- Queued patch replacement clears old staged PCM/history and never emits a block mixing
  old and new patches.
- Block-invariant deterministic rendering remains within the existing tolerance.
- Patch loading preserves selected frequency but resets active audio and capture.

### Analysis tests

- Exact silence and sub-threshold input produce an empty observation.
- Analysis validates sample rate/config composition and waveform-duration half-up frame
  conversion; timestamps begin at zero with exact sample spacing.
- A hand-generated known sine produces full-scale waveform samples without per-capture
  normalization.
- Log-view spectrum data excludes zero and stays within declared positive bounds.
- Bin-centered sine level is correct under Hann coherent-gain normalization.
- Interpolated peak frequency error is at most `0.1 Hz` for clean sine inputs across
  the C2-C4 GUI range. A pre-implementation characterization of 9,009 frequency/phase
  cases with the declared 16,384-frame Hann analysis produced a maximum absolute error
  below `0.047 Hz`; the acceptance margin is deliberately wider than that result.
- Observation arrays are one-dimensional, shape-matched, independently owned, and
  non-writeable.
- Capture-state transitions cover note-on, retrigger, release, Clear while active and
  idle, patch replacement, reset, device failure, and stale-generation rejection.
- A short note that becomes silent before producing a Live observation transitions from
  Measuring back to Empty rather than remaining indefinitely in Measuring.
- A transient outside the trailing waveform slice cannot mark a currently silent
  waveform as live.
- Retained capture survives later silence and clears only on the declared events.

### Workbench tests

- Pitch-knob C2/C3/C4 mapping, continuous drag, fine modifiers, key/wheel increments,
  reset gesture, direct-entry validation/precision, and derived note/cents label.
- Live frequency edits reach the engine without retriggering.
- Hold-to-play mouse and Spacebar behavior.
- Healthy audio details are absent from the main surface.
- Audio failure is visible once and disables Play.
- Empty plots contain no fabricated trace; live and captured plots use observation data.
- Measuring, Live, and Captured are visibly distinguishable without an output-health
  indicator.
- Axes, units, fixed ranges, log mode, peak readout, and disabled plot navigation.
- Patch load/save actions use the patch codec and preserve atomic failure behavior.
- Logical 1024 x 640 minimum and 1280 x 720 default layouts retain visible controls,
  plot titles, axes, readouts, and the declared minimum plot height. Native screenshots
  verify that the complete outer window frame fits the current 1536 x 864 effective
  desktop at 125% Windows scaling and an equivalent desktop at 100% scaling.

### Native acceptance

- Launch through WSLg against the default Windows audio route.
- Hear a stable C3 sine, continuous knob retuning through hold/release, and the
  configured release.
- Observe nonzero Waveform and Spectrum data while playing.
- Verify the measured peak follows live retuning after one complete FFT-window settling
  interval and remains distinct from the configured-frequency readout.
- Release, capture retention, Clear, JSON load, JSON save/reload, audio failure, and
  clean shutdown are manually exercised.
- Capture and inspect screenshots at the default window size and Windows scale used for
  the YouTube walkthrough.

## Milestone A completion criteria

Milestone A is complete when all of the following are true:

1. Engine, patch, tuning, analysis, and Qt presentation boundaries match this design.
2. The current working native playback mechanics remain verified.
3. The old sine-lab layout and its obsolete GUI-bound model are removed rather than
   hidden behind the new surface.
4. The workbench is fully usable at its default size and presents no device-format or
   MIDI-number clutter.
5. A user can continuously select C2-C4 with the pitch knob or exact Hz entry, hold
   Play, retune live, and understand both plots from their titles, units, and
   measurement readouts.
6. Empty plots are genuinely empty, captured plots are intentionally retained, and
   Clear behaves predictably.
7. A patch JSON file can be loaded, saved, and round-tripped without application-owned
   storage or hidden state.
8. Automated tests, lint, import smoke checks, and the native acceptance pass.
9. Production-code review finds no orphaned old model, dead compatibility layer, or
   GUI dependency inside the DSP and analysis core.

## Roadmap after Milestone A

1. **Milestone B:** editable ADSR values, deterministic constrained curve parameters,
   graphical envelope authoring, and the corresponding patch-schema extension.
2. **Sine-only Gym proof:** constrained pitch actions, target/input cases, objective
   observations and rewards, model/tool adapters, and benchmark reporting.
3. **Oscillator expansion:** define and add waveform semantics, then repeat Gym
   validation.
4. **Polyphony expansion:** add voice identity, chords, and later velocity, then repeat
   validation across oscillator types.
5. **Harness expansion:** randomized patches, dataset/experiment manifests, result
   storage, and the staged gauntlet.
