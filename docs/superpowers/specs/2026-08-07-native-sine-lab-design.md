# Harpy native sine lab design

Date: 2026-08-07

Status: design approved; written specification awaiting user review

## Purpose

Build Harpy's first executable slice: a small native desktop lab that plays one deterministic sine voice at a selected musical note. This slice validates the pitch model, envelope semantics, real-time audio path, native GUI stack, and testing approach before rendered-candidate editing or an RL environment is introduced.

The lab is also the first video-friendly explanation surface. It must make the relationship between musical note, tuning reference, frequency, waveform, and spectrum visible without becoming a general synthesizer.

## Decisions

- Use Python 3.12 and `uv` for environment management, dependency resolution, command execution, and the committed `uv.lock` file.
- Use a `src/` Python package layout.
- Use PySide6 Qt Widgets for a native desktop UI. No browser or local web server is part of this milestone.
- Use Qt Multimedia's `QAudioSink` for device playback and pyqtgraph for the waveform and spectrum.
- Make a small NumPy reference renderer authoritative for v1 rather than adopting a general synth engine.
- Keep AMY as the leading future optional backend. Do not add it as a v1 dependency.
- Represent musical identity numerically. MIDI note 60 is the initial note and is displayed as `C3` using Ableton's octave-label convention.
- Keep the tuning reference independent from the selected note. The default is MIDI note 69 at 440 Hz.
- Use a fixed, linear-amplitude envelope: 1 ms attack, 600 ms decay, -6 dB sustain, and 600 ms release.
- Store tuning, note-selection, render, envelope, and patch values in validated immutable configuration objects. Defaults are versioned in source. They are not implicit OS environment variables.
- Do not add a database or persistent settings system.

## Scope

### Included

- A native window launched with `uv run harpy`.
- An integer musical-note selector centered on MIDI note 60.
- A note readout containing the Ableton-style label, MIDI number, and calculated frequency.
- A visible tuning readout for the concert-A reference.
- A hold-to-play button: press starts the voice and release starts its release stage.
- A locked sine waveform and fixed envelope, both described in the UI but not editable there.
- A scrolling recent-waveform view and a frequency-spectrum view.
- Deterministic, block-based synthesis shared by live playback and future offline rendering.
- Automated tests that do not require a physical audio device.
- A WSLg manual smoke test of the actual audio device and window.

### Deferred

- Rendering or editing a bounded ten-second candidate asset.
- Uploading or importing audio.
- Octave, semitone, coarse, or fine pitch-transformation actions.
- An RL/Gymnasium environment, rewards, actors, training, telemetry, or model adapters.
- MIDI input.
- Chords, polyphony, saw, square, filters, modulation, effects, or editable patches.
- Editable ADSR controls, alternate envelope curves, and persistent user settings.
- AMY, SignalFlow, pyo, JUCE, C++, plugins, and DAW integration.
- Native Windows packaging; WSLg is the development target for this slice.

## Package and development workflow

`pyproject.toml` is the human-edited package definition and `uv.lock` is committed. The initial runtime dependencies are:

- NumPy for block synthesis and analysis;
- PySide6 for Qt Widgets and Qt Multimedia;
- pyqtgraph for native plots.

The development dependency group contains pytest, pytest-qt, and Ruff. A type checker can be added when it provides concrete value; it is not required merely to create the project.

The standard commands will be:

```text
uv sync --dev
uv run harpy
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The project targets Python 3.12 through `requires-python = ">=3.12,<3.13"`, and `.python-version` asks `uv` for Python 3.12. The console entry point is `harpy = harpy.gui.app:main`.

## Configuration and musical model

Configuration is explicit application data, not ambient process state. Small immutable specifications live beside the subsystem that owns them, and `AppConfig` composes the default application configuration.

The defaults are conceptually:

```text
EqualTemperament
  reference_note       = 69
  reference_hz         = 440.0

KeyboardViewSpec
  minimum_note         = 48
  initial_note         = 60
  maximum_note         = 72
  middle_c_octave      = 3

EnvelopeSpec
  attack_seconds       = 0.001
  decay_seconds        = 0.600
  sustain_db           = -6.0
  release_seconds      = 0.600
  curve                = linear_amplitude

RenderSpec
  sample_rate_hz       = 48000
  block_frames         = 256
  channels             = 1
  internal_dtype       = float32

SinePatch
  peak_gain_dbfs       = -12.0
  envelope             = EnvelopeSpec
```

The -6 dB sustain value is relative to the envelope peak. Its linear amplitude is `10 ** (-6 / 20)`, approximately `0.501187`. The fixed -12 dBFS patch gain provides device headroom and is independent from the envelope shape.

Values may later be overridden through a CLI, settings surface, or experiment manifest. Any override that affects sound must be included in render or experiment metadata. Reading these values directly from ad hoc OS environment variables is deliberately deferred because it would create hidden, difficult-to-reproduce state. In this design, “configurable variables” means fields on these typed specifications, not process environment variables.

### Pitch representation

Core pitch values never depend on note-name strings:

```text
MidiNote
  number: integer in [0, 127]

Pitch
  cents_from_midi_zero: finite float

EqualTemperament
  reference_note: MidiNote
  reference_hz: positive finite float
```

`Pitch.from_midi(note, detune_cents=0)` creates the continuous pitch coordinate. Keeping cents in the foundational pitch model prevents a later API break, but this milestone exposes no detune or pitch-transformation action. Frequency under twelve-tone equal temperament is:

```text
reference_hz * 2 ** (
  (pitch.cents_from_midi_zero - 100 * reference_note.number) / 1200
)
```

MIDI note 60 therefore produces approximately 261.625565 Hz under the default tuning. Note 60 is shown as `C3`, and note 69 is shown as `A3`, because note names follow Ableton's display convention. The UI labels the tuning field `Concert A reference (MIDI 69)` so the physical reference is clear despite octave-label differences among music applications.

The initial note control is an integer slider over MIDI notes 48 through 72. It advances by one semitone and starts at 60. The slider is disabled while a note gate is active; pitch is selected before Play is pressed.

## Reference synth semantics

The reference synth is a monophonic state machine, not a general synth framework. It owns:

- oscillator phase;
- the current continuous pitch and phase increment;
- envelope stage and frame position;
- instantaneous envelope level;
- gate state.

Its public behavior is `note_on(pitch)`, `note_off()`, `reset()`, and `render_block(frame_count) -> mono float32`. Qt types and audio-device concerns never enter this package.

### Oscillator

The sine oscillator uses a float64 phase accumulator and emits float32 samples. Phase advances continuously across render calls and wraps without losing the fractional remainder. A note-on resets phase to zero, making equivalent renders deterministic. Frequencies must be finite, positive, and below Nyquist.

### Envelope

All envelope segments interpolate amplitude linearly per sample:

1. Attack rises from zero to 1.0 over 1 ms.
2. Decay falls from 1.0 to approximately 0.501187 over 600 ms.
3. Sustain remains at that level while Play is held.
4. Release falls from the instantaneous current level to zero over 600 ms.

Starting release during attack or decay must not introduce an envelope discontinuity. Segment durations are converted to frames with nearest-integer, half-up rounding. At 48 kHz the defaults are exactly 48 attack frames, 28,800 decay frames, and 28,800 release frames.

Discrete-time endpoints are defined exactly. For a segment lasting `F` frames, the state starts at the previously emitted level, advances by `(target - start) / F` before each emission, emits the target on frame `F`, and enters the next stage on the following frame. `note_off()` snapshots the last emitted level and applies the same rule toward zero. Attack, decay, and release must each contain at least one frame in v1; zero-duration segments are rejected. For every audio frame, the oscillator sample is read at the current phase and the phase is then advanced.

A new note-on retriggers the sole voice, resets its phase and envelope, and replaces any active or releasing note. Duplicate GUI press events while the gate is already active are ignored. When release reaches zero the voice becomes idle and subsequent blocks contain exact zeros.

### Block invariance

Rendering `N` frames in one call must match concatenating arbitrary smaller calls totaling `N` with `rtol=0` and `atol=1e-6` on emitted float32 samples. Block size is an I/O scheduling choice and cannot change pitch or envelope behavior. A future offline renderer will schedule events at exact frame offsets and drive this same voice.

## Native GUI

The window uses Qt Widgets and is laid out for clear capture in a 16:9 video frame. The first version uses a restrained dark neutral palette, high-contrast text, and no decorative animation.

The canvas contains:

1. A title/status row showing `Harpy · Sine Lab` and audio-device state.
2. A pitch panel with the note slider and a large `C3 · MIDI 60 · 261.626 Hz` readout.
3. A patch panel showing `Sine` and the fixed envelope values.
4. A large hold-to-play button.
5. A waveform plot showing a short recent time window.
6. A spectrum plot computed from recent samples and labeled in Hz and dBFS.

Pressing Play enqueues one note-on command. Releasing it enqueues one note-off command and preserves the normal 600 ms release. Window deactivation, application shutdown, or audio failure instead invokes one `force_stop()` path: it clears the controller's gate, raises the Play button, unlocks the pitch selector, stops device callbacks when necessary, and forces silence without the normal release so a tone cannot remain stuck.

The visualization path is observational only. The audio adapter copies recent rendered samples into a bounded ring buffer. A GUI timer reads snapshots at no more than 30 updates per second; FFT and plotting work never run in the audio read path. The spectrum uses the latest 4,096 samples and a Hann window. One-sided amplitude is normalized by the window's coherent gain: interior bins use `2 * abs(rfft(x * window)) / sum(window)`, while DC and Nyquist are not doubled. The dBFS display is `20 * log10(max(amplitude, 1e-6))`, producing a -120 dBFS floor. Spectrum output is not fed back into synth truth.

## Audio adapter and data flow

```text
Qt widgets
  -> GUI controller
  -> thread-safe note command queue
  -> QIODevice audio source
  -> reference SineVoice.render_block()
  -> mono float32 blocks
  -> device channel/sample conversion
  -> QAudioSink / WSLg audio device

                         mono block copy
                                |
                                v
                   bounded visualization buffer
                                |
                                v
                 GUI timer -> waveform and spectrum
```

While the sink is running, the audio source owns synth mutation. The GUI thread only enqueues typed commands, avoiding simultaneous access to oscillator and envelope state. The adapter fills a byte staging buffer in 256-frame render chunks and consumes pending commands before each chunk. The scheduling portion of live gate latency is therefore at most 5.33 ms at 48 kHz, in addition to Qt and device buffering.

`force_stop()` is idempotent. With a healthy running sink it clears GUI gate state immediately and enqueues one reset for the next render chunk. On shutdown, hotplug, or sink failure it first stops further device callbacks and then resets the now-uncontended audio source synchronously. Any later button-release signal is ignored because the controller gate is already clear.

The authoritative render format remains 48 kHz mono float32. V1 has no device selector: it uses `QMediaDevices.defaultAudioOutput()` and reports that device's description. The adapter calls `isFormatSupported()` in this order: 48 kHz stereo Float, 48 kHz mono Float, 48 kHz stereo Int16, then 48 kHz mono Int16. It duplicates mono samples for a stereo format and performs clipping, rounding, and sample conversion only at the device boundary. It never resamples or silently changes tuning, core sample rate, or reference rendering. If none of those formats is supported, Play is disabled with an actionable error.

The adapter listens for `QMediaDevices.audioOutputsChanged()`. A change invokes `force_stop()`, discards the old sink, and makes one attempt to initialize the new default output with the same probe order. Success re-enables Play; failure leaves it disabled. There is no polling or tight retry loop.

Audio-device latency, PulseAudio behavior, channel duplication, and device-format conversion are demonstration concerns. They are never used as future benchmark observations, rewards, or evaluator truth.

## Error handling

- Invalid or non-finite configuration fails at startup with a precise field-level message.
- Attack, decay, and release durations must be finite and produce at least one frame; sustain and patch gain must be finite and no greater than 0 dB.
- MIDI notes outside 0 through 127 and frequencies at or above Nyquist are rejected before rendering.
- GUI controls constrain normal input, while the controller validates again at its boundary.
- An unavailable or unsupported audio device leaves the window usable for inspection but disables Play and shows the reason.
- `QAudioSink` errors invoke `force_stop()`, update status, and never retry in a tight loop.
- Window close and application deactivation invoke the same idempotent `force_stop()` path.
- Plotting failures may disable visualization but must not destabilize audio playback.

No error path silently substitutes a different tuning reference, sample rate, envelope, or note.

## File structure

```text
.python-version
pyproject.toml
uv.lock
src/harpy/
  __init__.py
  config.py                 # AppConfig composition and defaults
  pitch.py                  # MidiNote, Pitch, and tuning math
  note_names.py             # Ableton-style presentation formatter
  synth/
    __init__.py
    specs.py                # EnvelopeSpec, RenderSpec, SinePatch
    envelope.py             # sample-domain envelope state machine
    reference.py            # deterministic monophonic SineVoice
  gui/
    __init__.py
    app.py                   # console entry point and application lifetime
    specs.py                 # KeyboardViewSpec
    controller.py            # widget-to-audio command boundary
    window.py                # native canvas and widgets
    qt_audio.py              # QIODevice, QAudioSink, device conversion
    visualizer.py            # bounded sample buffer and plot updates
tests/
  test_config.py
  test_pitch.py
  test_note_names.py
  synth/
    test_envelope.py
    test_reference.py
  gui/
    test_controller.py
    test_visualizer.py
```

`offline.py`, pitch-processing tools, asset manifests, and `gym_env.py` are added only in later approved milestones. An AMY adapter will be introduced only when a second backend is actually exercised; AMY objects will not enter Harpy's public pitch or configuration types.

## Testing strategy

### Pitch and configuration

- MIDI note 69 maps exactly to 440 Hz under the default tuning.
- MIDI note 60 maps to approximately 261.625565 Hz.
- A 442 Hz reference changes note 69 to exactly 442 Hz.
- Adding 100 cents multiplies frequency by `2 ** (1 / 12)`; adding 1,200 cents doubles it.
- Note labels include note 60 as `C3`, note 69 as `A3`, and the 0/127 boundaries.
- Configuration rejects invalid note ranges, non-finite numbers, non-positive reference frequency, invalid sustain, and incompatible sample settings.

### Envelope and synthesis

- Exact values are asserted at every attack, decay, sustain, and release boundary.
- Early note-off begins release from the instantaneous amplitude without a jump.
- Release ends at exact zero and remains silent.
- One large render matches arbitrarily chunked renders with `rtol=0` and `atol=1e-6`.
- Repeated identical note events produce deterministic samples.
- Samples are finite, bounded by the configured peak gain, mono, and float32.
- FFT verification over a two-second, 96,000-frame steady-state capture with a Hann window finds the C3 peak within one 0.5 Hz FFT bin of its expected frequency.
- Nyquist and non-finite pitch inputs are rejected.

### GUI and adapter

- The initial controller state is MIDI 60 and the formatter produces the complete C3 readout.
- One button press/release produces exactly one note-on/note-off command.
- The pitch selector locks while gated and unlocks on release.
- Window deactivation and close invoke `force_stop()`, clear the gate, raise Play, unlock pitch selection, and request exactly one reset.
- A fake audio source verifies channel duplication and device-format conversion without a speaker.
- The visualization buffer is bounded and thread-safe; plotting reads never mutate synth state.
- pytest-qt exercises widget behavior without asserting that CI can hear audio.

### Manual WSLg smoke test

- `uv sync --dev` completes from a clean checkout.
- `uv run harpy` opens a native window rather than a browser.
- The default readout is `C3 · MIDI 60 · 261.626 Hz` with a 440 Hz concert-A reference.
- Holding Play produces a stable sine; releasing it audibly follows the 600 ms release.
- Moving the idle note slider changes note label and frequency by semitone steps.
- Waveform and spectrum remain responsive without audible underruns.
- Closing or unfocusing the application cannot leave a stuck tone.

## Acceptance criteria

The milestone is complete when:

1. A clean checkout can be installed from `pyproject.toml` and the committed `uv.lock` using `uv sync --dev`.
2. All automated tests and Ruff checks pass through `uv run`.
3. `uv run harpy` launches the native WSLg GUI with no browser dependency.
4. The default configuration and readouts match the approved MIDI 60, A440, sine, and ADSR semantics.
5. Press/hold/release behavior, audio playback, and both plots pass the manual WSLg smoke test.
6. Reference rendering is deterministic and independent of caller block partitioning.
7. No deferred RL, audio-editing, MIDI, third-party synth-engine, database, or native executable-packaging work has entered the slice.

## Future engine boundary

AMY is the first engine to reconsider when Harpy reaches band-limited harmonic oscillators, chords, richer patches, or a desire to compare renderer implementations. It is MIT-licensed, supports real-time and offline buffers, and successfully built from source under this project's WSL/Python 3.12 environment during design research.

It is not the v1 reference because its Python package currently requires a source checkout and local C build, the `amy` name on PyPI belongs to an unrelated project, native Windows Python installation is not turnkey, and desktop render format is compile-time constrained. A later adapter must pin an exact AMY revision and prove that its output and packaging are reproducible before it can be used in benchmark generation.

## References

- [AMY repository](https://github.com/shorepine/amy)
- [AMY tutorial](https://shorepine.github.io/amy/tutorial.html)
- [Qt for Python](https://doc.qt.io/qtforpython-6/)
- [QAudioSink](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QAudioSink.html)
- [QMediaDevices](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QMediaDevices.html)
- [QAudioDevice format support](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QAudioDevice.html)
- [pyqtgraph](https://pyqtgraph.readthedocs.io/en/latest/)
- [MIDI note-number and octave-label discussion](https://midi.org/community/midi-specifications/midi-octave-and-note-numbering-standard)
- [Python Packaging Authority `src` layout guidance](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- [`uv` project documentation](https://docs.astral.sh/uv/)
