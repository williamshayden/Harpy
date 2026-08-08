# Harpy

Harpy is currently a native deterministic sine workbench and the foundation for a
future audio-reinforcement-learning research environment. Milestone A provides one
monophonic NumPy sine engine, a PySide6/Qt Multimedia desktop workbench, strict patch
files, and pure waveform/spectrum analysis. It does not yet provide an RL environment.

The project notebook records the broader research hypotheses, references, decisions,
and open questions:

- [Project notebook](docs/project-notebook.md)
- [Milestone A design](docs/superpowers/specs/2026-08-07-harpy-milestone-a-foundation-workbench-design.md)
- [Milestone A acceptance evidence](docs/verification/2026-08-07-milestone-a-acceptance.md)

## Install, run, and verify

Harpy requires Python 3.12 and uses `uv` for its environment and lockfile.

```bash
uv sync
uv run harpy
```

Under WSLg, use the compositor's runtime directory if the inherited directory does not
contain its Wayland socket. Preserve the existing `PULSE_SERVER` value:

```bash
XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir uv run harpy
```

Run the complete automated checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Native workbench

The GUI selects a continuous frequency from C2 through C4 (approximately
130.812783–523.251131 Hz at A4 = 440 Hz), starts at C3, and exposes hold-to-play by
mouse or Space. It shows a fixed-range 0–50 ms, ±1 FS waveform and a logarithmic
20 Hz–20 kHz, -120–0 dBFS spectrum based on measured audio. The current patch is a
sine oscillator with a 1 ms attack, 600 ms decay, -6 dB sustain, 600 ms release, and
-12 dBFS output gain.

The C2–C4 limit belongs only to the GUI. `SynthEngine` accepts every finite positive
frequency strictly below half the configured sample rate (Nyquist).

## Public non-Qt API

The reusable research surface does not require a Qt application:

- `harpy.synth.SynthPatch` is an immutable validated sound description; its nested
  oscillator and envelope values live in `harpy.synth.models`.
- `harpy.synth.patch_json` supplies `dumps_patch`, `loads_patch`, `save_patch`, and
  `load_patch`. The version-1 codec rejects missing, extra, duplicate, non-finite, and
  incorrectly typed fields instead of accepting a partial document.
- `harpy.synth.SynthEngine` renders deterministic mono `float32` blocks and exposes
  `note_on`, `retune`, `note_off`, `replace_patch`, `render`, and `reset`.
- `harpy.tuning.Tuning` converts between hertz and MIDI coordinates and derives
  note-name/cents readings. MIDI coordinates are conversion values, not engine state
  or GUI controls.
- `harpy.analysis.analyze` maps a one-dimensional sample array and sample rate to an
  immutable `AudioObservation`, using an optional validated `AnalysisConfig`.

A patch JSON document contains only `schema_version`, oscillator configuration,
envelope configuration, and `output_gain_dbfs`. It contains no played note, selected
frequency, render/sample-rate setting, file history, path, identifier, or other
application-owned storage metadata. A patch describes a sound, not a performance or a
saved workbench session.

## Roadmap, not current capability

Milestone B is deferred. It is intended to add editable ADSR values, constrained curve
parameters, graphical envelope authoring, and an explicitly versioned patch-schema
extension.

The sine-only Gym proof is also deferred. Constrained pitch actions, targets,
actor-facing observations, rewards, model/tool adapters, training, and benchmark
reporting are roadmap items, not implemented claims. Browser UI, MIDI input,
imported-audio editing, a database, additional oscillators, polyphony, and third-party
synth engines are likewise outside the current implementation.
