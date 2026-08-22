# Harpy

Harpy is a deterministic audio-control research workbench. Milestones A and B provide
one monophonic NumPy sine engine, a PySide6/Qt Multimedia desktop workbench, strict
versioned patch files, pure waveform/spectrum analysis, and graph-native envelope
authoring. Milestone C adds Harpy's first headless Gymnasium checkpoint: a frozen,
sine-only pitch-control environment and deterministic evaluation matrix. It proves the
environment and evaluation contract; it does not provide or claim a trained RL policy.

The project notebook records the broader research hypotheses, references, decisions,
and open questions:

- [Project notebook](docs/project-notebook.md)
- [Milestone A design](docs/superpowers/specs/2026-08-07-harpy-milestone-a-foundation-workbench-design.md)
- [Milestone A acceptance evidence](docs/verification/2026-08-07-milestone-a-acceptance.md)
- [Milestone B approved design](docs/superpowers/specs/2026-08-08-harpy-milestone-b-envelope-authoring-design.md)
- [Milestone B graph-native implementation plan](docs/superpowers/plans/2026-08-08-harpy-milestone-b-graph-native-envelope-controls.md)
- [Milestone B acceptance evidence](docs/verification/2026-08-08-milestone-b-acceptance.md)
- [Milestone C approved design](docs/superpowers/specs/2026-08-09-harpy-milestone-c-sine-pitch-gym-design.md)
- [Milestone C implementation plan](docs/superpowers/plans/2026-08-09-harpy-milestone-c-sine-pitch-gym.md)
- [Milestone C acceptance evidence](docs/verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md)

## Install, run, and verify

Harpy requires Python 3.12 and uses `uv` for its environment and lockfile.

```bash
uv sync
uv run harpy
```

Run the headless Milestone C checkpoint and write one JSON document to standard output:

```bash
uv run harpy-sine-gym --episodes 10 --seed 0
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

## Sine-pitch Gymnasium checkpoint

Importing `harpy.envs` registers three versioned environments:

| Environment ID | Observation track |
| --- | --- |
| `Harpy/SinePitch-v0` | Headline normalized log-frequency spectrum |
| `Harpy/SinePitchOracle-v0` | Exact current-pitch coordinate control |
| `Harpy/SinePitchRewardOnly-v0` | Controls, target, budget, and scalar feedback only |

The default spectrum observation contains a normalized `float32` log-frequency magnitude
array with shape `(1961,)`, the symbolic target note, the three control values, and the
remaining action budget. Oracle replaces the spectrum with one exact current-pitch
coordinate. Reward-only contains neither. These tracks answer different questions and
their results must never be pooled.

The stable discrete action IDs are:

| ID | Action |
| ---: | --- |
| 0 | Octave Down |
| 1 | Semitone Down |
| 2 | Cent Down |
| 3 | Submit |
| 4 | Cent Up |
| 5 | Semitone Up |
| 6 | Octave Up |

Octave, Semitone, and Cent are independent bounded controls with ranges `-2..2`,
`-12..12`, and `-100..100`; they never carry into one another. An episode succeeds only
when the actor explicitly submits at an inclusive absolute error of at most 5 cents.
Accuracy within 1 cent is reported separately, and merely passing through either region
does not terminate the episode.

The public environment surface is:

```python
from harpy.envs import (
    ControlState,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    SinePitchEnv,
    register_envs,
)
```

Actor-facing observations and `info` omit source pitch, exact error, optimal actions,
and other evaluator truth. A harness may read immutable `EpisodeResult` only after the
episode is done. This is an honest capability boundary for supported actors, not a
security sandbox against Python code deliberately reaching into private state or
`env.unwrapped`.

The checkpoint command evaluates Random and Spectrum Peak on the spectrum track, Oracle
on the oracle track, and Reward Search on the reward-only track. It emits separate rows
under schema version 1 with checkpoint ID `harpy-milestone-c-sine-pitch-v0` and config ID
`fixed-default-sine-v0`. Rows report submitted and positional accuracy separately,
absolute final error, action and excess-action means, return, truncation, and invalid
actions. The four baselines are deterministic, frozen, untrained reference policies;
their output is not evidence of learned listening or model quality.

For this procedural checkpoint, every applied pitch action directly re-synthesizes a
fresh sine from immutable source truth plus cumulative controls. That is an ideal
sine-only transformation backend, not recorded-audio pitch shifting.

## Native workbench

The GUI selects a continuous frequency from C2 through C4 (approximately
130.812783–523.251131 Hz at A4 = 440 Hz), starts at C3, and exposes hold-to-play by
mouse or Space. It shows a fixed-range 0–50 ms, ±1 FS waveform and a logarithmic
20 Hz–20 kHz, -120–0 dBFS spectrum based on measured audio. The current patch is a
sine oscillator with a 1 ms attack, 600 ms decay, -6 dB sustain, 600 ms release, and
-12 dBFS output gain.

The C2–C4 limit belongs only to the GUI. `SynthEngine` accepts every finite positive
frequency strictly below half the configured sample rate (Nyquist).

Milestone B adds graph-native A/D/S/R vertical scrubbing, transient exact editing,
constrained Attack/Decay/Release curve handles, and a contextual envelope-only Reset.
The frequency selector is a continuous logarithmic pro-audio dial with vertical drag,
dynamic Shift fine mode, wheel and keyboard steps, a C3 reset, landmarks, and native
accessible Dial semantics. Envelope edits made during a held note or its release are
deferred and coalesced; the current voice finishes unchanged and the final admitted
patch is rendered on the next Play.

## Public non-Qt API

The reusable research surface does not require a Qt application:

- `harpy.synth.SynthPatch` is an immutable validated sound description; its nested
  oscillator and envelope values live in `harpy.synth.models`.
- `harpy.synth.patch_json` supplies `dumps_patch`, `loads_patch`, `save_patch`, and
  `load_patch`. Harpy strictly reads schema-v1 linear patches and schema-v2
  curve-enabled patches, rejecting missing, extra, duplicate, non-finite, and
  incorrectly typed fields instead of accepting a partial document. It writes only
  canonical schema v2.
- `harpy.synth.SynthEngine` renders deterministic mono `float32` blocks and exposes
  `note_on`, `retune`, `note_off`, `replace_patch`, `render`, and `reset`.
- `harpy.tuning.Tuning` converts between hertz and MIDI coordinates and derives
  note-name/cents readings. MIDI coordinates are conversion values, not engine state
  or GUI controls.
- `harpy.analysis.analyze` maps a one-dimensional sample array and sample rate to an
  immutable `AudioObservation`, using an optional validated `AnalysisConfig`.

A patch JSON document contains only `schema_version`, oscillator configuration,
envelope values and curves, and `output_gain_dbfs`. Selected or played frequency is
performance state outside patch JSON. The document also contains no render/sample-rate
setting, file history, path, identifier, or other application-owned storage metadata.
A patch describes a sound, not a performance or a saved workbench session.

## Roadmap, not current capability

Model training and adapters, train/test splits and held-out generalization, recorded
assets and real audio pitch shifting, raw-waveform observations, additional oscillators,
chords and polyphony, hosted actors, telemetry, durable experiment storage, and Gym
episode replay in the GUI remain later work. Browser UI, MIDI input, imported-audio
editing, a database, and third-party synth engines are likewise outside the current
implementation.
