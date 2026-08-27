# Harpy

Harpy is a deterministic audio-control research workbench. Milestones A and B provide
one monophonic NumPy sine engine, a PySide6/Qt Multimedia desktop workbench, strict
versioned patch files, pure waveform/spectrum analysis, and graph-native envelope
authoring. Milestone C adds a frozen, sine-only Gymnasium pitch-control environment and
deterministic evaluation matrix. Milestone D adds Harpy's first learned policies on
that unchanged environment: a behavior-cloning diagnostic followed by freshly
initialized PPO. Milestone E adds a spectrum-only neural pitch estimator that replans
on the same task through Harpy's existing symbolic planner and bounded controls.

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
- [Milestone D approved design](docs/superpowers/specs/2026-08-10-harpy-milestone-d-learned-sine-policy-design.md)
- [Milestone D implementation plan](docs/superpowers/plans/2026-08-10-harpy-milestone-d-learned-sine-policy.md)
- [Milestone D acceptance evidence](docs/verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md)
- [Milestone E approved design](docs/superpowers/specs/2026-08-22-harpy-milestone-e-reliable-learned-tuning-design.md)
- [Milestone E implementation plan](docs/superpowers/plans/2026-08-27-harpy-milestone-e-reliable-learned-tuning.md)

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

## Learned tuning and policy workflow

Install the optional learned-policy stack with this single step:

```bash
uv sync --group train
```

Run the complete Milestone E smoke workflow with four commands:

```bash
uv run harpy-sine-learn train-pitch \
  --profile smoke --seed 0 --output runs/milestone-e-pitch-smoke --device cpu

uv run harpy-sine-learn diagnose runs/milestone-e-pitch-smoke \
  --suite smoke --output runs/milestone-e-pitch-smoke-diagnostics.json --device cpu

uv run harpy-sine-learn evaluate runs/milestone-e-pitch-smoke \
  --output runs/milestone-e-pitch-smoke-report.json --device cpu

uv run harpy-sine-learn run runs/milestone-e-pitch-smoke \
  --seed 123 --device cpu
```

Artifact directories and diagnostic/evaluation files are create-only. Every path
above must therefore be new, and diagnostic or report files must remain outside all
input artifact directories. `diagnose` and `evaluate` write the same canonical JSON
bytes to the requested file and standard output. `run` prints one complete
human-readable action trace. Add `--json` to capture the same run as deterministic
canonical JSON on standard output.

The eligible checkpoint uses three fresh, clean CPU artifacts with the exact seeds
0, 1, and 2:

```bash
uv run harpy-sine-learn train-pitch \
  --profile checkpoint --seed 0 --output runs/milestone-e-pitch-0 --device cpu
uv run harpy-sine-learn train-pitch \
  --profile checkpoint --seed 1 --output runs/milestone-e-pitch-1 --device cpu
uv run harpy-sine-learn train-pitch \
  --profile checkpoint --seed 2 --output runs/milestone-e-pitch-2 --device cpu

uv run harpy-sine-learn evaluate \
  runs/milestone-e-pitch-0 runs/milestone-e-pitch-1 runs/milestone-e-pitch-2 \
  --output runs/milestone-e-pitch-report.json --device cpu

uv run harpy-sine-learn diagnose \
  runs/milestone-e-pitch-0 runs/milestone-e-pitch-1 runs/milestone-e-pitch-2 \
  --suite iid --output runs/milestone-e-pitch-iid-diagnostics.json --device cpu
```

CPU is the authoritative Milestone E device. `--device cuda` is always explicit,
fails when CUDA is unavailable, and cannot produce scientific-criterion evidence.
Only the compatible, committed, CPU checkpoint triple for seeds 0, 1, and 2 may
access the final Milestone E suites.

The learned estimator sees only the actor-visible candidate spectrum and predicts a
location on Harpy's fixed five-cent grid. A stateless symbolic planner then uses the
public target and Octave, Semitone, and Cent controls, replans after every observation,
and executes the existing seven actions. There is no hidden pitch, target/reference
spectrum, Spectrum Peak fallback, or evaluator rescue in this actor.

This lane tests a narrow learned-perception claim on the clean procedural single-sine
task, not end-to-end learned-policy or reinforcement-learning mastery. It cannot
establish performance on recorded, noisy, polyphonic, chordal, or pitch-shifted audio.
The smoke profile proves only that training, persistence, reload, diagnostics,
evaluation, and tracing work; it is never scientific evidence. Checkpoint reports
remain valid when they honestly conclude `criterion_not_met`.

The Milestone D learned-policy controls remain available with their original syntax
and semantics:

```bash
uv run harpy-sine-learn train-bc \
  --profile smoke --seed 0 --output runs/bc-smoke

uv run harpy-sine-learn train-ppo \
  --profile smoke --seed 0 --output runs/ppo-smoke

uv run harpy-sine-learn evaluate runs/ppo-smoke

uv run harpy-sine-learn run runs/ppo-smoke --seed 123
```

CPU remains Milestone D's default and authoritative checkpoint device. Adding
`--device cuda` is an explicit exploratory choice: it fails if CUDA is unavailable,
records CUDA provenance, and produces criterion-ineligible artifacts. Evaluation and
hands-on runs also default to CPU.

Behavior cloning is a supervised representation-and-control diagnostic trained from
oracle action labels. PPO starts from a fresh random initialization and never reuses
BC weights. Spectrum Peak is a separately labeled classical control for the clean
procedural sine, not a learned-policy result. The candidate spectrum is already
continuously visible in `Harpy/SinePitch-v0`; Milestone D does not add analysis tools
or model-selected tool calls.

The smoke profile establishes an engineering result: real training, persistence,
reload, evaluation, and trace paths execute on the optional stack. It is deliberately
ineligible for scientific criteria. Scientific outcomes come only from the declared
clean CPU checkpoint runs and are reported honestly as `criterion_met` or
`criterion_not_met`; PPO is not required to beat Spectrum Peak.

Pitch and BC `.pt` files and PPO `.zip` files are trusted-local model artifacts. The
CLI warns before loading them; do not load model files from untrusted sources.

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

Explicit analysis tools, recorded assets and real audio pitch shifting, raw-waveform
observations and waveform generalization, additional oscillators, chords and
polyphony, hosted actors, telemetry, durable experiment storage, and Gym episode replay
in the GUI remain later work. Browser UI, MIDI input, imported-audio editing, a
database, and third-party synth engines are likewise outside the current
implementation. Milestone D's learned result applies only to the current clean,
procedural single-sine spectrum task; it is not evidence for recorded audio, chords,
or other waveforms. Milestone E adds learned spectral localization on that same narrow
task; it does not broaden the audio domain.
