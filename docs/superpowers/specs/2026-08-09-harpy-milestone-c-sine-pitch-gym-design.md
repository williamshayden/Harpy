# Harpy Milestone C: Sine Pitch Gym Design

Date: 2026-08-09

Status: Approved. The project notebook and the decisions recorded in this document were
approved by the user for implementation on 2026-08-09.

## Outcome

Milestone C adds Harpy's first headless Gymnasium checkpoint: a deterministic,
sine-only environment in which an actor receives a symbolic target note plus declared
audio-derived evidence and uses distinct Octave, Semitone, and Cent controls to tune a
candidate before explicitly submitting it.

This checkpoint proves the environment contract, procedural episode generation,
immutable-source rendering, action semantics, reward and terminal behavior, leakage
boundaries, reproducibility, and baseline evaluation. It does not claim that tuning a
single sine is an unsolved audio problem, and it does not yet claim a learned RL result.

## Scientific boundary

The headline track asks whether an actor can combine a symbolic musical target with a
non-oracle log-frequency spectrum and constrained controls. The actor is not given the
current frequency, source frequency, exact cents error, hidden target render, or an
explicit F0 estimate.

Two separately labeled controls establish what the result means:

- `oracle`: adds the evaluator's exact current pitch coordinate. It validates planning
  and the complete environment pipeline; it is not evidence of learned listening.
- `reward_only`: removes audio evidence. It measures how far a search policy can get
  from action state and scalar feedback alone; it is not evidence of listening.

The three tracks must never be pooled into one headline score. The default registered
environment is the non-oracle spectrum track.

## Scope

Milestone C includes:

- Gymnasium as a small, direct runtime dependency;
- a Qt-free `SinePitchEnv` implementing the current Gymnasium reset/step contract;
- deterministic procedural source/target sampling;
- a fixed-length normalized log-frequency magnitude spectrum;
- actor-visible symbolic target, independent pitch-control state, and remaining budget;
- named unit actions for Octave, Semitone, Cent, and Submit;
- progress-based shaping, explicit submission, and evaluator-owned terminal metrics;
- default Gymnasium registration and environment-checker coverage;
- deterministic random, oracle, spectrum-peak, and reward-only search baselines;
- a small checkpoint runner that emits machine-readable JSON summaries;
- documentation of the exact claim, public API, and reproduction commands.

Milestone C does not include:

- synth-parameter control, envelope editing, oscillator choice, or polyphony;
- a real recorded-audio pitch shifter or repeated transformation of sample buffers;
- Stable-Baselines3, PyTorch, a trained policy, checkpoints, or hyperparameter search;
- supervised training, hosted models, OpenRouter, Codex, or generic model adapters;
- raw-waveform, audible-reference, chord, percussion, or uploaded-asset tracks;
- GUI integration, live audio devices, MIDI, browser UI, a database, or telemetry;
- claims of research novelty or claims that the single-sine task itself is difficult.

Those are later checkpoints. In particular, the first learned policy will consume this
frozen environment rather than being bundled into its definition.

## Architecture and dependency direction

The environment is an offline client of the existing numerical core:

```text
Gymnasium caller
      |
      v
harpy.envs.SinePitchEnv
      |-- episode truth and musical controls
      |-- fresh deterministic render per state
      |-- actor-safe observation/reward/info
      |
      +--> harpy.synth.SynthEngine / SynthPatch / RenderConfig
      +--> harpy.tuning.Tuning
      +--> harpy.analysis.analyze
```

Nothing under `harpy.envs` imports `harpy.gui`, Qt, live playback commands,
`SampleHistory`, `CaptureCoordinator`, or `WorkbenchController`. The GUI remains a
separate consumer of the synth and analysis core.

The public package is split by responsibility:

- `harpy.envs.models`: immutable configuration, actions, observation modes, control
  state, terminal result, and validation;
- `harpy.envs.spectrum`: the fixed log-frequency encoder and its grid;
- `harpy.envs.sine_pitch`: episode generation, rendering, Gymnasium spaces, reset,
  step, reward, and evaluator result;
- `harpy.envs.baselines`: deterministic baseline actor policies and evaluation loop;
- `harpy.envs.checkpoint`: the JSON-emitting checkpoint command;
- `harpy.envs.__init__`: deliberate public exports and idempotent Gym registration.

The constructor accepts only the selected `ObservationMode` and `render_mode=None`.
Version `v0` freezes `Tuning()`, `RenderConfig()`, `SynthPatch()`, the action bounds,
source/target range, 262,144-frame analysis size, 5-cent log grid, reward constants,
tolerances, and 64-step budget. This prevents a valid but very quiet patch, unusual
tuning, or low sample rate from silently breaking the fixed observation guarantee.
Configurable patch distributions belong to a later environment version.

The package does not grow a framework-wide actor abstraction, renderer registry,
storage layer, or model-serving interface in this checkpoint.

## Musical coordinates and episode distribution

Pitch arithmetic uses integer cents on the existing tuning coordinate. MIDI remains a
conversion coordinate, not a GUI control and not an actor-visible raw pitch value.
With the default A4 = 440 Hz tuning:

- target notes are integer coordinates 48 through 72 inclusive, displayed as C2
  through C4 using Harpy's existing octave convention;
- the hidden source is an integer-cent coordinate from 4,800 through 7,200 inclusive;
- a target/source pair is sampled uniformly from the allowed pair set by rejection
  sampling independent uniform target and source values until the initial absolute
  error is greater than 5 cents;
- sampling uses only Gymnasium's seeded `self.np_random`; module-global Python and
  NumPy random state are never used;
- `reset` always calls `super().reset(seed=seed)` before sampling;
- `reset(seed=N)` reproduces the same latent pair and byte-identical observation;
- `reset(seed=None)` continues the environment's seeded random stream;
- reset options use one closed injection schema: `None` and Gymnasium's empty-dict
  checker sentinel `{}` both request an ordinary sampled reset; every nonempty
  `options` dict must contain exactly `target_note_index` and `source_pitch_cents`, and
  partial or unknown keys fail;
- both injected values use `operator.index`, reject booleans, and must respectively be
  inside `0..24` and `4800..7200`;
- injected pairs may start inside tolerance so Submit boundaries can be evaluated;
  fully injected resets initialize Gym's RNG but consume no random draws, and injected
  truth is not echoed into actor-facing `info`.

The patch is exactly `SynthPatch()`, tuning is exactly `Tuning()`, and the render format
is exactly `RenderConfig()`. Patch variation is not part of the episode distribution.

## Immutable-source render semantics

An episode stores the hidden source coordinate and the three actor-controlled offsets.
The effective candidate coordinate is always:

```text
source_cents
+ 1,200 * octaves
+   100 * semitones
+         cents
```

Reset clears every prior trajectory/result, restores controls to `(0, 0, 0)`, sets the
step count to zero and remaining budget to 64, and immediately creates the initial
source observation through the same fresh-render path used after accepted actions.

Every accepted pitch action computes that total and renders with a fresh
`SynthEngine`. It never retunes a carried live engine and never transforms the previous
candidate buffer. Equal cumulative control states therefore produce byte-identical
audio regardless of action order.

For this procedural sine checkpoint, the transformation backend synthesizes the
effective frequency directly. This is an artifact-free ideal pitch transformation for
the sine curriculum rung, not a claim that Harpy already pitch-shifts recorded audio.
Later asset-based environments must retain the same immutable-source rule while
supplying a separately specified pitch-shift renderer.

Each fresh render:

1. creates `SynthEngine(render, patch)`;
2. calls `note_on(effective_frequency_hz)`;
3. renders through the complete Attack and Decay plus one analysis window;
4. analyzes the final `fft_frames`, which lie at the fixed Sustain level.

The render length, envelope, gain, polarity, phase reset, and array shape are identical
for every episode and action. No filenames, lengths, amplitude choices, phase offsets,
IDs, or metadata encode evaluator truth.

## Action and control contract

`PitchAction` is a stable seven-value integer enum presented in this order:

| Index | Public action | State effect |
| ---: | --- | --- |
| 0 | `OCTAVE_DOWN` | octaves `- 1` |
| 1 | `SEMITONE_DOWN` | semitones `- 1` |
| 2 | `CENT_DOWN` | cents `- 1` |
| 3 | `SUBMIT` | terminate and score the current candidate |
| 4 | `CENT_UP` | cents `+ 1` |
| 5 | `SEMITONE_UP` | semitones `+ 1` |
| 6 | `OCTAVE_UP` | octaves `+ 1` |

The actor-facing vocabulary is musical. Raw `+1200`, `+100`, coarse, or fine pitch
actions do not appear in the public action surface.

The three controls are independent and never carry into one another:

- Octaves: `-2..+2`, one octave per accepted action;
- Semitones: `-12..+12`, one semitone per accepted action;
- Cents: `-100..+100`, one cent per accepted action.

The intentional overlap means, for example, that one Octave action and twelve
Semitone actions can reach the same effective pitch. That redundancy measures whether
an actor learns the efficient hierarchy. The evaluator's shortest-path metric chooses
the least number of legal unit actions.

An action that would move its named control outside its bound is an invalid no-op. It
does not alter or re-render the candidate, but it consumes one environment step,
receives the invalid-action penalty, and is recorded in actor-safe `info` and terminal
metrics. Non-integral actions, booleans, values outside the seven action indices, and
steps after episode completion raise immediately without mutating state.

Action coercion uses `operator.index`, so Python integers and NumPy integer scalars are
accepted. Booleans, floats, arrays, and out-of-range integers are rejected before any
step counter, RNG, control, trajectory, reward, or render state changes.

## Episode budget and reachability

The budget is 64 total actions, including `Submit` and invalid attempts. Every allowed
initial source/target difference has an exact-zero correction within 57 pitch actions.
When any endpoint inside the approved ±5-cent tolerance is allowed, the worst
tolerance-optimal solution takes 52 pitch actions plus Submit: at most 53 total actions,
leaving eleven actions of slack.

The budget is part of the observation. A non-Submit action that consumes the final
step sets `truncated=True`. `Submit` always sets `terminated=True`; it never sets
`truncated=True`, even when it is the sixty-fourth action.

## Success, reward, and terminal semantics

Submitting with absolute evaluator error at or below 5 cents is successful. Harpy also
records strict accuracy at or below 1 cent. Merely passing through the success region
does not end an episode; the actor must choose `Submit`.

For every valid non-Submit pitch action, let `e_before` and `e_after` be the hidden
absolute integer-cent errors. The maximum possible error under the source and control
bounds is 6,100 cents. The progress reward is:

```text
(e_before - e_after) / 6,100 - 0.00001
```

This is a bounded undiscounted telescoping progress difference plus a fixed action
cost. It is not a policy-invariance claim for arbitrary trainer discount factors. Its
cumulative progress component telescopes when `gamma = 1.0`, so decomposing one Octave
move into many smaller moves cannot manufacture extra progress reward; the step cost
makes shorter legal paths strictly better.

Other rewards are exact:

- invalid bound no-op: `-0.01`;
- successful Submit: `+1.0`;
- unsuccessful Submit: `-1.0`;
- budget truncation: add `-1.0` to the final pitch-action reward.

The truncation surcharge applies to every sixty-fourth non-Submit attempt. A valid
pitch action receives its progress reward minus `1.0`; a bound no-op receives `-1.01`.
Submit on step 64 receives only its Submit reward and terminates rather than truncates.

Reward exposes controlled progress feedback and can support black-box search. That is
why the reward-only baseline is mandatory and why a sequential spectrum result alone
cannot establish learned perception. Reported return for this checkpoint is
undiscounted (`gamma = 1.0`); applying another discount changes the shaping incentives
and must be reported as a separate training decision. Later sparse-reward studies may
be configured as separate experiments; they do not silently change this environment
version.

## Headline observation

The default `spectrum` observation is a Gymnasium `spaces.Dict` containing:

- `spectrum`: normalized log-frequency magnitude, `float32`, shape `(1961,)`, bounds
  `[0, 1]`;
- `target_note`: integer `0..24`, where zero is C2 and 24 is C4;
- `controls`: `int16` array `(octaves, semitones, cents)`, shape `(3,)`, with bounds
  `(-2, -12, -100)` through `(2, 12, 100)`;
- `steps_remaining`: integer `0..64`.

The spectrum grid covers coordinate 11.00 through 109.00 inclusive at 5-cent spacing.
That exact range covers every candidate reachable from a C2-C4 source under all three
control bounds. Grid frequencies are derived through the configured `Tuning`; they are
configuration, not per-episode observation fields.

The encoder uses a 262,144-frame instance of Harpy's calibrated Hann-window spectrum.
Its dedicated `AnalysisConfig` includes every positive RFFT bin: the minimum is one FFT
bin (`48,000 / 262,144 Hz`) and the maximum is Nyquist (`24,000 Hz`). The log grid is
strictly inside those source bins, including coordinates 11.00 and 109.00, so
interpolation never extrapolates a reachable candidate. It linearly interpolates dBFS
levels onto the log-frequency grid, clamps at `-120 dBFS`, and maps
`[-120, 0] dBFS` to `[0, 1]`. It does not include the analyzer's interpolated peak,
frequency axis, waveform, or an F0/cents estimate. All observations are owned,
contiguous arrays matching the declared dtypes and spaces.

The `oracle` mode replaces `spectrum` with:

- `current_pitch_coordinate`: exact evaluator pitch coordinate as a one-element
  `float32` array bounded by the reachable coordinate range.

The oracle observation does not also carry the spectrum; it is a planning upper bound,
not a combined lane. The `reward_only` mode omits both `spectrum` and exact pitch. It
retains the symbolic target, controls, and remaining budget. Observation modes are
chosen at construction; their spaces never change during an episode.

## Actor-safe info and evaluator truth

The tuple returned by `reset` and `step` exposes only actor-safe `info`:

- step count and remaining budget;
- whether the requested action was applied;
- whether Submit was requested;
- `submitted_success` only after Submit; truncation always reports it as false;
- a stable public action label for the completed step.

It never contains source frequency, target frequency, current frequency, signed or
absolute cents error, shortest path, hidden ideal audio, procedural seed-derived ID,
or analyzer peak.

After termination or truncation, the harness may read immutable `EpisodeResult` from
the environment. That evaluator-only record contains latent source/target/final pitch,
signed and absolute initial/final errors, separate `submitted_success`,
`within_5_cents`, and `within_1_cent` fields, action trajectory, invalid-action count,
the initial state's tolerance-optimal successful path including Submit, return, and
terminal reason. Headline success is always `submitted_success`; positional diagnostics
never turn an unsubmitted truncation into success. Excess actions are reported only for
successful episodes as `actual_steps - optimal_total_actions`; failed and truncated
episodes report that field as null. Calling for a result before completion raises.
Future actor adapters receive only the Gym observation/reward/done/info tuple, never
the environment object or `EpisodeResult`.

This is an API boundary for honest experiments, not a security sandbox against Python
code intentionally reaching through `env.unwrapped`. The evaluation harness enforces
actor capability separation.

## Gymnasium contract and registration

`SinePitchEnv` subclasses `gymnasium.Env` and implements:

```python
reset(*, seed: int | None = None, options: dict | None = None)
    -> tuple[Observation, Info]

step(action: int)
    -> tuple[Observation, float, bool, bool, Info]
```

It supports `render_mode=None` only and declares no visual render modes. All configured
instances pass `gymnasium.utils.env_checker.check_env` without warnings attributable
to Harpy.

Idempotent registration exposes:

- `Harpy/SinePitch-v0` for the default spectrum track;
- `Harpy/SinePitchOracle-v0` for the explicitly labeled oracle track;
- `Harpy/SinePitchRewardOnly-v0` for the explicitly labeled reward-only track.

Importing `harpy.envs` initializes no Qt type or audio device. Importing top-level
`harpy` remains lightweight and does not implicitly import Gymnasium.

## Baselines and checkpoint runner

The checkpoint contains four small, frozen evaluation baselines:

- random legal action sampling;
- oracle shortest-path planning from evaluator truth;
- spectrum-peak planning from the headline log spectrum and target symbol;
- reward-only hierarchical hill climbing, which receives no spectrum or oracle pitch.

These baselines are deterministic under a declared seed. They implement only the
episode act/feedback loop needed by this checkpoint; they do not create the broader
hosted/local actor protocol yet.

The oracle planner enumerates bounded control triples, selects the reachable endpoint
inside ±5 cents with minimum L1 distance from current controls, breaks ties by
smaller final absolute error and then lexicographic control triple, emits Octave,
Semitone, and Cent unit actions in that order, then submits. Its exhaustive
action-lattice tests are the reachability oracle for the environment.

The spectrum-peak baseline takes the maximum log-spectrum bin as a declared classical
estimate, converts that grid coordinate into a correction, chooses the nearest legal
control target, and submits. It is a simple preprocessing/controller baseline, not a
learned model.

The reward-only search baseline visits Octave, Semitone, then Cent. At each scale it
probes Down first when legal. A strictly positive transition reward means continue in
that direction; a nonpositive applied probe is undone once, after which Up is probed
and continued by the same rule. A blocked probe changes direction without an undo.
After the first nonpositive continuation it undoes that applied step and advances to
the next scale. It always reserves its last remaining action for Submit, even when
refinement is incomplete. Truncation is therefore not expected from the reference
implementation, but inaccurate submissions are an allowed and measured result. Its
purpose is to expose how much the shaped reward solves by search, not to guarantee the
headline task.

The random baseline samples only actions currently legal under the visible control
bounds, with Submit included; it never samples a known blocked bound action.

`python -m harpy.envs.checkpoint --episodes N --seed S` runs exactly this labeled
matrix: random and spectrum-peak on `spectrum`, oracle on `oracle`, and reward search
on `reward_only`. Results are never aggregated across observation modes. Episode
`i` uses environment seed `S + i`; each stochastic policy uses an independent NumPy
generator seeded through `SeedSequence([S, stable_policy_code, i])`, where
`stable_policy_code` is a checked-in integer rather than a Python hash. The command
writes one JSON document to standard output. The document includes
the environment/version/config identity, baseline identity, episode count, seed,
submitted-success rate, submitted-within-1-cent rate, final positional rates within 5
and 1 cents, mean absolute final error, mean actions, mean excess actions over
successful episodes, mean return, truncation rate, and invalid-action rate. It contains
no audio arrays and does not write files.

The runner is an executable acceptance and documentation surface, not a training
framework or durable experiment database.

## Determinism and leakage acceptance

Automated acceptance must establish:

- same config plus `reset(seed=N)` produces byte-identical observations and identical
  evaluator truth;
- equal control totals reached in different action orders render byte-identical audio
  and observations;
- no observation shape, dtype, array length, `info` key, amplitude envelope, render
  length, or phase policy changes with hidden source/target truth;
- source and target sampling never produces an initially successful episode;
- all 4,801 possible integer-cent corrections in `-2400..+2400` have an exact-zero
  correction within 57 pitch actions and a tolerance-optimal successful plan within
  52 pitch actions plus Submit;
- bound no-ops, invalid action types, post-terminal steps, Submit success/failure, and
  final-step truncation obey the exact mutation and reward rules;
- strict nonempty reset options reject partial, unknown, boolean, nonintegral, and
  out-of-range values without consuming an episode draw or mutating prior state, while
  `options={}` remains the Gymnasium-compatible sampled-reset sentinel;
- every observation belongs to its declared Gymnasium space;
- across all 2,401 one-cent C2-C4 source pitches, the maximum-bin estimate from the
  frozen log-spectrum representation is never more than 5 cents from source truth;
- the three registered environments pass Gymnasium's checker;
- actor-safe observations and info contain none of the evaluator-only fields;
- the oracle baseline succeeds on exhaustive correction values without truncation;
- checkpoint summaries are deterministic, valid JSON, and clearly label each track;
- `import harpy.envs` succeeds with GUI/Qt modules absent from `sys.modules`.

The complete existing suite, Ruff lint, Ruff format, import smoke, and diff checks remain
mandatory. Tests ship with the public repository because determinism and leakage
claims require executable evidence.

## Documentation and claim discipline

README and project-notebook updates may claim that Harpy now provides its first
deterministic sine-pitch Gymnasium checkpoint. They must also state:

- the default track uses a log-frequency spectrum and hides explicit pitch;
- oracle and reward-only modes are separate controls;
- the actor controls pitch transformation tools, not the synth patch;
- procedural direct-frequency synthesis is an ideal sine-only transformation backend,
  not a recorded-audio pitch shifter;
- the shipped baselines are not a trained RL policy;
- model training, model adapters, richer waveforms, polyphony, and real asset
  transformation remain later work.

The notebook's earlier hypothetical auto-termination language must be replaced with
Submit-gated success. Its previously open coarse/fine action vocabulary must be
replaced with the approved bounded Octave/Semitone/Cent controls, and its one-shot,
supervised, and learned-RL comparisons must be labeled as later checkpoints rather
than Milestone C acceptance requirements.

No novelty claim is made. Prior synth-control and inverse-synthesis RL work establishes
the broader field; Harpy's present contribution is a small, auditable benchmark
contract and curriculum checkpoint.

## Deferred decisions

The following are intentionally deferred until this environment is accepted:

- which learned RL algorithm and encoder form the first trained baseline;
- IID versus held-out-frequency training/evaluation protocol;
- sparse-reward comparison;
- raw waveform or callable analyzer observations;
- model/checkpoint and hosted-agent adapters;
- real immutable-buffer pitch shifting and its artifact budget;
- patch randomization, new oscillator shapes, chords, percussion, and polyphony;
- GUI episode replay and video walkthrough integration;
- compute, GPU, energy, provider-usage, and cost telemetry.

Those additions require their own approved design because each changes what a result
demonstrates.
