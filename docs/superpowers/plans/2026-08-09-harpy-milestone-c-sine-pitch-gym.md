# Harpy Milestone C Sine Pitch Gym Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Ship Harpy's first deterministic, headless Gymnasium checkpoint: a
single-sine symbolic-note tuning environment with non-oracle log-spectrum evidence,
distinct bounded Octave/Semitone/Cent controls, explicit Submit, honest control lanes,
and reproducible baseline summaries.

**Architecture:** A new Qt-free `harpy.envs` package consumes only the existing
`SynthEngine`, immutable synth models, `Tuning`, and pure analysis core. Integer-cent
episode truth drives a fresh deterministic render after each applied pitch action;
Gym-facing observations expose only the selected evidence lane. Small baseline and CLI
modules consume the public Gym contract rather than introducing a trainer, actor
framework, or storage layer.

**Tech Stack:** Python 3.12, NumPy 2.x, Gymnasium 1.3.x, existing Harpy numerical core,
pytest, Ruff, uv.

## Global Constraints

- The approved design is
  `docs/superpowers/specs/2026-08-09-harpy-milestone-c-sine-pitch-gym-design.md`;
  exact values and claim boundaries in that document govern every task.
- The default headline lane exposes a `(1961,)` normalized 5-cent log-frequency
  spectrum and no F0, peak frequency, source pitch, candidate pitch, or cents error.
- Oracle and reward-only lanes are separate registered environments and separate
  result rows; results from different observation modes are never pooled.
- Actor actions are exactly Octave Down, Semitone Down, Cent Down, Submit, Cent Up,
  Semitone Up, and Octave Up with stable integer IDs `0..6` in that order.
- Controls remain independent and never carry: Octaves `-2..2`, Semitones `-12..12`,
  Cents `-100..100`, each changed by one named unit per applied action.
- Source truth is an integer-cent coordinate `4800..7200`; target note index is
  `0..24` mapping to coordinates `48..72`; random episodes exclude initial error
  `<=5` cents.
- Success requires Submit with final absolute evaluator error `<=5` cents; strict
  positional accuracy is `<=1` cent. Merely entering tolerance never terminates.
- The action budget is 64 total calls including Submit and blocked attempts.
- Valid pitch reward is `(before_abs_error - after_abs_error) / 6100 - 0.00001`;
  blocked no-op is `-0.01`; Submit is `+1` or `-1`; the final non-Submit attempt adds
  a `-1` truncation surcharge.
- Version `v0` freezes `Tuning()`, `RenderConfig()`, `SynthPatch()`, a 262,144-frame
  full-positive-bin analysis, the log grid, bounds, reward, tolerance, and budget.
- Every applied pitch action renders from a fresh `SynthEngine` using immutable source
  truth plus cumulative integer controls. Never call live `retune`, reuse the previous
  candidate buffer, or import GUI/playback/capture/controller code.
- No Stable-Baselines3, PyTorch, SciPy, librosa, SoundFile, hosted SDK, telemetry,
  database, GUI change, real asset pitch shifter, or model-training code enters this
  milestone.
- Production behavior changes follow strict RED → observed expected failure → minimal
  GREEN → refactor. Tests exercise real code and hand-derived expectations.
- Every task commits only its scoped files after focused tests, the full suite, Ruff
  lint, Ruff format, and `git diff --check` are green.

---

## Final File Map

| File | Responsibility |
| --- | --- |
| `src/harpy/envs/models.py` | Stable action/mode enums, bounded control state, terminal result, constants. |
| `src/harpy/envs/planning.py` | Pure tolerance-aware shortest musical action planner. |
| `src/harpy/envs/spectrum.py` | Frozen full-RFFT analysis config, log grid, normalized encoder. |
| `src/harpy/envs/sine_pitch.py` | Procedural truth, fresh rendering, Gym spaces, reset/step/result. |
| `src/harpy/envs/baselines.py` | Four narrow baseline rollouts and aggregate summaries. |
| `src/harpy/envs/checkpoint.py` | Deterministic JSON checkpoint command. |
| `src/harpy/envs/__init__.py` | Public exports and idempotent Gym registration. |
| `tests/envs/test_models.py` | Action/control/result validation. |
| `tests/envs/test_planning.py` | Exact and tolerance-optimal planner behavior/reachability. |
| `tests/envs/test_spectrum.py` | Spectrum shape/calibration/grid/accuracy characterization. |
| `tests/envs/test_sine_pitch.py` | Gym dynamics, reward, leakage, and determinism. |
| `tests/envs/test_registration.py` | Gym registration, checker, and Qt-free import boundary. |
| `tests/envs/test_baseline_planners.py` | Oracle and spectrum planner capability boundaries. |
| `tests/envs/test_baseline_evaluation.py` | Random/reward policies, rollouts, metrics, deterministic seeds. |
| `tests/envs/test_checkpoint.py` | CLI JSON, error behavior, no-write/headless boundary. |
| `README.md` | Implemented Gym API, commands, and precise nonclaims. |
| `docs/project-notebook.md` | Locked Milestone C decisions and corrected earlier hypotheses. |
| `docs/verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md` | Fresh automated and checkpoint evidence. |
| `pyproject.toml`, `uv.lock` | Gymnasium dependency and `harpy-sine-gym` command. |

---

### Task 1: Musical Action State and Tolerance-Aware Planner

**Files:**
- Create: `src/harpy/envs/models.py`
- Create: `src/harpy/envs/planning.py`
- Create: `src/harpy/envs/__init__.py`
- Create: `tests/envs/__init__.py`
- Create: `tests/envs/test_models.py`
- Create: `tests/envs/test_planning.py`

**Interfaces:**
- Produces `PitchAction(IntEnum)`, `ObservationMode(StrEnum)`,
  `TerminalReason(StrEnum)`, frozen `ControlState`, and frozen `EpisodeResult`.
- Produces immutable `ZERO_CONTROLS = ControlState()` and
  `minimum_action_plan(base_error_cents: int, controls: ControlState = ZERO_CONTROLS,
  *, tolerance_cents: int = 5) -> tuple[PitchAction, ...]`.
- The returned plan always includes final `SUBMIT`; `base_error_cents` means hidden
  `source_pitch_cents - target_pitch_cents` before tool controls.
- Later tasks consume constants and types directly; this task imports no Gymnasium or
  Qt module.

- [ ] **Step 1: Write the action/control RED tests**

Create tests with literal expectations:

```python
def test_pitch_actions_have_stable_symmetric_ids():
    assert [(action.name, action.value) for action in PitchAction] == [
        ("OCTAVE_DOWN", 0),
        ("SEMITONE_DOWN", 1),
        ("CENT_DOWN", 2),
        ("SUBMIT", 3),
        ("CENT_UP", 4),
        ("SEMITONE_UP", 5),
        ("OCTAVE_UP", 6),
    ]


def test_controls_apply_one_named_unit_without_carry():
    state = ControlState(octaves=1, semitones=12, cents=100)
    next_state, applied = state.apply(PitchAction.OCTAVE_UP)
    assert applied is True
    assert next_state == ControlState(octaves=2, semitones=12, cents=100)
    assert next_state.offset_cents == 3_700


def test_bound_attempt_is_the_same_immutable_state():
    state = ControlState(octaves=2, semitones=12, cents=100)
    next_state, applied = state.apply(PitchAction.CENT_UP)
    assert applied is False
    assert next_state is state
```

Also cover every lower/upper bound, invalid booleans/non-enum values, `SUBMIT` not being
a control mutation, exact integer validation, and the exact actor-facing labels.

- [ ] **Step 2: Run the action/control tests and observe RED**

Run:

```bash
uv run pytest tests/envs/test_models.py -q
```

Expected: collection fails because `harpy.envs.models` does not exist.

- [ ] **Step 3: Implement the immutable model surface**

Use these exact constants and shapes:

```python
SOURCE_MIN_CENTS = 4_800
SOURCE_MAX_CENTS = 7_200
TARGET_MIN_COORDINATE = 48
TARGET_NOTE_COUNT = 25
OCTAVE_MIN, OCTAVE_MAX = -2, 2
SEMITONE_MIN, SEMITONE_MAX = -12, 12
CENT_MIN, CENT_MAX = -100, 100
SUCCESS_TOLERANCE_CENTS = 5
STRICT_TOLERANCE_CENTS = 1
MAX_STEPS = 64
MAX_ABSOLUTE_ERROR_CENTS = 6_100
```

`ControlState.apply` returns `(self, False)` at a bound, returns a new frozen state for
one legal musical unit, and rejects `SUBMIT`. `EpisodeResult` owns only immutable
scalars/tuples and includes:

```python
source_pitch_cents: int
target_note_index: int
target_pitch_cents: int
final_pitch_cents: int
initial_signed_error_cents: int
initial_absolute_error_cents: int
final_signed_error_cents: int
final_absolute_error_cents: int
submitted_success: bool
within_5_cents: bool
within_1_cent: bool
actions: tuple[PitchAction, ...]
invalid_action_count: int
optimal_actions: tuple[PitchAction, ...]
excess_actions: int | None
total_return: float
terminal_reason: TerminalReason
```

`invalid_action_count` counts accepted action IDs that were blocked by a named control
bound. Malformed/out-of-range API inputs are exceptions before the episode trajectory
and are not counted. `optimal_actions` owns the initial state's tolerance-optimal path,
including Submit, and read-only `optimal_total_actions` derives its length rather than
storing a second truth. Validate every `EpisodeResult` invariant at construction, own
both action tuples, and reject booleans where an integer scalar is required.

`TerminalReason` values are exactly `SUBMITTED_SUCCESS`, `SUBMITTED_FAILURE`, and
`BUDGET_EXHAUSTED`. `PitchAction.label` returns the human strings `Octave Down`,
`Semitone Down`, `Cent Down`, `Submit`, `Cent Up`, `Semitone Up`, and `Octave Up`.
Create an otherwise side-effect-free `harpy.envs.__init__` that imports nothing yet;
Task 4 owns final exports and registration.

- [ ] **Step 4: Run model tests GREEN**

Run `uv run pytest tests/envs/test_models.py -q` and require all tests to pass.

- [ ] **Step 5: Write planner RED tests**

Use hand-derived plans and exhaustive outcome checks:

```python
def test_planner_uses_the_named_hierarchy_and_submit():
    assert minimum_action_plan(-1_305, tolerance_cents=0) == (
        PitchAction.OCTAVE_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )


def test_tolerance_can_stop_at_five_cents():
    plan = minimum_action_plan(-6, tolerance_cents=5)
    assert plan == (PitchAction.CENT_UP, PitchAction.SUBMIT)
```

For every `base_error_cents` in `-2400..2400`, apply the returned pitch actions to a
real `ControlState`, derive final integer error independently, assert exact zero when
`tolerance_cents=0`, assert `<=5` for the default, assert exact plans use at most 57
pitch actions, and assert tolerance plans use at most 52 plus one Submit. Add arbitrary
nonzero starting-control cases and reject bool/noninteger/out-of-range tolerances.

- [ ] **Step 6: Run planner tests and observe RED**

Run `uv run pytest tests/envs/test_planning.py -q`; expected failure is the absent
planner module/function, not a fixture error.

- [ ] **Step 7: Implement the pure enumerating planner**

Enumerate target Octave and Semitone positions. For each pair, derive the inclusive
Cent interval that puts final error within tolerance, clamp the current Cent position
to that interval to minimize Cent presses, then rank candidates by:

```python
(
    l1_action_count,
    absolute_final_error,
    target_octaves,
    target_semitones,
    target_cents,
)
```

Emit Octave presses, then Semitone presses, then Cent presses, and append one Submit.
Never use frequency floats or the production function under test to derive test
expectations.

- [ ] **Step 8: Run Task 1 gates and commit**

```bash
uv run pytest tests/envs/test_models.py tests/envs/test_planning.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/models.py src/harpy/envs/planning.py src/harpy/envs/__init__.py tests/envs
git commit -m "feat: define sine gym musical controls"
```

---

### Task 2: Fixed Log-Frequency Spectrum Evidence

**Files:**
- Create: `src/harpy/envs/spectrum.py`
- Create: `tests/envs/test_spectrum.py`

**Interfaces:**
- Consumes fixed core `Tuning()`, `RenderConfig()`, `AnalysisConfig`, and `analyze`.
- Produces `LOG_SPECTRUM_SIZE = 1961`, read-only
  `LOG_FREQUENCY_GRID_HZ: np.ndarray`, and
  `encode_log_spectrum(samples: np.ndarray) -> np.ndarray`.
- The encoder returns a fresh C-contiguous `float32` array of shape `(1961,)` in
  `[0, 1]`; it never returns peak/F0/waveform metadata.

- [ ] **Step 1: Write grid/normalization RED tests**

In `tests/envs/test_spectrum.py`, define a local `render_candidate_audio(cents: int,
fft_frames: int = 262_144) -> np.ndarray` helper using only the existing `SynthEngine`,
`RenderConfig`, `SynthPatch`, `Tuning`, and Attack+Decay settle-frame arithmetic. This
is test setup, not a fixture or production encoder, and it returns the final Sustain
window. Then write:

```python
def test_log_grid_is_five_cent_coordinate_grid_covering_every_candidate():
    assert LOG_SPECTRUM_SIZE == 1_961
    assert LOG_FREQUENCY_GRID_HZ.shape == (1_961,)
    tuning = Tuning()
    assert LOG_FREQUENCY_GRID_HZ[0] == pytest.approx(tuning.frequency_hz_for_midi_coordinate(11.0))
    assert LOG_FREQUENCY_GRID_HZ[-1] == pytest.approx(
        tuning.frequency_hz_for_midi_coordinate(109.0)
    )
    coordinates = [
        tuning.midi_coordinate_for_frequency_hz(float(value)) for value in LOG_FREQUENCY_GRID_HZ
    ]
    assert np.diff(coordinates) == pytest.approx(np.full(1_960, 0.05))


def test_encoder_returns_owned_bounded_float32_evidence():
    encoded = encode_log_spectrum(render_candidate_audio(6_000))
    assert encoded.shape == (1_961,)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert np.all((0.0 <= encoded) & (encoded <= 1.0))
```

Also assert the dedicated analysis source begins at `48_000 / 262_144` Hz and ends at
24,000 Hz, strictly bracketing the log grid; silence returns a fixed zero vector; bad
rank/short inputs preserve the core analyzer's actionable validation. Render the legal
candidate-coordinate extremes 11.00 and 109.00 and require finite fixed-shape output
without interpolation extrapolation.

- [ ] **Step 2: Run spectrum tests and observe RED**

Run `uv run pytest tests/envs/test_spectrum.py -q`; expected failure is the missing
module/API.

- [ ] **Step 3: Implement the fixed encoder**

Define:

```python
GYM_ANALYSIS_CONFIG = AnalysisConfig(
    fft_frames=262_144,
    spectrum_min_hz=48_000 / 262_144,
    spectrum_max_hz=24_000.0,
    spectrum_floor_dbfs=-120.0,
)
```

Build coordinate cents with `np.arange(1100, 10901, 5, dtype=np.float64)`, convert each
coordinate through `Tuning()`, and freeze the owned grid. Call `analyze` with the fixed
sample rate/config, interpolate its calibrated dBFS levels onto the log grid, clip to
`[-120, 0]`, normalize using `(dbfs + 120) / 120`, and return an owned `float32` array.
When `has_signal` is false, return 1,961 zeros without fabricating F0 metadata.

- [ ] **Step 4: Write and observe the exhaustive representation RED**

Add one characterization test that renders a real fixed-patch `SynthEngine` source for
every integer cent coordinate `4800..7200`, encodes it, converts `argmax` back through
the known 5-cent grid, and asserts maximum absolute estimate error `<=5` cents. First
mutation-run this test with `fft_frames=16_384` and record an expected failure above
five cents; restore the specified 262,144 frames and run GREEN. Keep this as one loop,
not 2,401 separately reported pytest cases.

- [ ] **Step 5: Run Task 2 gates and commit**

```bash
uv run pytest tests/envs/test_spectrum.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/spectrum.py tests/envs/test_spectrum.py
git commit -m "feat: encode sine gym log spectrum"
```

---

### Task 3: Deterministic Gymnasium Episode Semantics

**Files:**
- Create: `src/harpy/envs/sine_pitch.py`
- Create: `tests/envs/test_sine_pitch.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes all Task 1/2 types/functions.
- Produces `SinePitchEnv(gymnasium.Env)` with constructor
  `SinePitchEnv(observation_mode: ObservationMode = ObservationMode.SPECTRUM,
  render_mode: None = None)`.
- Declares exact `action_space = gymnasium.spaces.Discrete(7)` and one fixed Dict
  observation space per mode.
- Produces `episode_result: EpisodeResult` property, available only when done.

- [ ] **Step 1: Write constructor/reset/space RED tests**

Tests must pin all three exact Dict schemas:

```python
def test_spectrum_reset_is_seeded_fixed_and_actor_safe():
    env = SinePitchEnv()
    assert env.action_space == gymnasium.spaces.Discrete(7)
    observation, info = env.reset(seed=17)
    assert set(observation) == {"spectrum", "target_note", "controls", "steps_remaining"}
    assert env.observation_space.contains(observation)
    assert observation["spectrum"].shape == (1_961,)
    assert observation["spectrum"].dtype == np.float32
    assert observation["controls"].tolist() == [0, 0, 0]
    assert int(observation["steps_remaining"]) == 64
    assert info == {"step_count": 0, "steps_remaining": 64}


def test_oracle_replaces_spectrum_and_reward_only_has_neither():
    oracle, _ = SinePitchEnv(ObservationMode.ORACLE).reset(seed=4)
    reward_only, _ = SinePitchEnv(ObservationMode.REWARD_ONLY).reset(seed=4)
    assert "spectrum" not in oracle
    assert oracle["current_pitch_coordinate"].shape == (1,)
    assert "spectrum" not in reward_only
    assert "current_pitch_coordinate" not in reward_only
```

Assert actual spaces/dtypes: target is `Discrete(25)`/`np.int64`; controls are int16
Box with literal bounds; remaining is `Discrete(65)`/`np.int64`; spectrum is float32
Box `[0,1]`; oracle pitch is float32 `(1,)` bounded coordinates 11..109. Reject any
non-null render mode.
Also reject strings or arbitrary objects in place of `ObservationMode`; registration
passes the enum itself. Seed validation accepts Python/NumPy integer scalars, rejects
booleans/nonintegers/negative values atomically, and permits `None` stream continuation.

- [ ] **Step 2: Run environment tests and observe RED**

Run `uv run pytest tests/envs/test_sine_pitch.py -q`; expected collection failure first
names missing `gymnasium`/`SinePitchEnv`.

- [ ] **Step 3: Add Gymnasium and implement reset/render observation GREEN**

Add the exact dependency with:

```bash
uv add 'gymnasium>=1.3,<2'
```

In `reset`, first validate the complete `options` structure and values into local
variables without touching environment or RNG state. Treat both `None` and Gymnasium's
checker sentinel `{}` as an ordinary sampled reset; every nonempty dict is an injection
request and must contain both exact keys. Only after validation succeeds,
call `super().reset(seed=seed)`, clear prior state, controls, result, trajectory,
counters, and return, then either:

- sample independent uniform target index/source cents and reject the entire pair while
  absolute initial error is `<=5`; or
- accept `options` only when its exact keys are `target_note_index` and
  `source_pitch_cents`, validating both with `operator.index`, rejecting booleans,
  partial/extra keys, and range violations, while allowing injected in-tolerance pairs.

For fully injected options, initialize but do not draw from `self.np_random`.
An invalid options call leaves the previous episode and RNG stream unchanged.
`options` must be either `None` or a real `dict`; mapping-like/string inputs are not
silently accepted. Add a RED/GREEN test that `reset(options={})` samples normally and
preserves the declared observation/info contract, alongside the partial/unknown-key
rejections. A sibling environment with the same seed must prove a rejected call did
not advance the deterministic stream.

Fresh rendering must be one call-equivalent path:

```python
engine = SynthEngine(RenderConfig(), SynthPatch())
engine.note_on(Tuning().frequency_hz_for_midi_coordinate(candidate_cents / 100.0))
settle = seconds_to_frames(0.001, 48_000) + seconds_to_frames(0.600, 48_000)
rendered = engine.render(settle + GYM_ANALYSIS_CONFIG.fft_frames)
candidate_audio = rendered[-GYM_ANALYSIS_CONFIG.fft_frames :]
```

Store an owned read-only candidate buffer, encode its spectrum for every mode, and
return only mode-approved keys. Repeated seed/options must reproduce byte-identical
evidence; injected resets with the same source and different targets must keep candidate
audio and spectrum byte-identical while only the public target symbol changes.

- [ ] **Step 4: Write action/reward/terminal RED tests**

Use injected literal episodes to prove:

```python
def test_submit_is_required_and_five_cents_is_success():
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_995})
    _, reward, terminated, truncated, info = env.step(PitchAction.SUBMIT)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["submitted_success"] is True
    assert env.episode_result.final_absolute_error_cents == 5


def test_six_cents_fails_only_when_submitted():
    env = SinePitchEnv()
    env.reset(options={"target_note_index": 12, "source_pitch_cents": 5_994})
    _, reward, terminated, truncated, _ = env.step(PitchAction.CENT_UP)
    assert reward == pytest.approx(1 / 6_100 - 0.00001)
    assert (terminated, truncated) == (False, False)
```

Add exact tests for movement toward/away, blocked `-0.01`, no rerender on blocked/Submit,
all independent bounds/no carry, ±5/±1 boundaries, passing through tolerance, failed
Submit, sixty-fourth valid action truncation surcharge, sixty-fourth blocked reward
`-1.01`, Submit on step 64 precedence, and post-done rejection.

Action validation accepts Python/NumPy integer scalars but rejects bool, float, arrays,
and out-of-range values without mutating any state. Compare control-equivalent action
orders and inverse sequences for bit-identical candidate audio/spectrum. Assert target
changes alone never change audio evidence and recursively scan observation/info keys for
forbidden evaluator truth.

- [ ] **Step 5: Implement step/result behavior GREEN**

Coerce using `operator.index`; compute evaluator errors entirely in integer cents. Each
valid pitch action creates a new `ControlState`, fresh-renders from hidden source plus
total controls, and receives the exact formula. A blocked action consumes a step but
retains the identical state/audio. Submit terminates regardless of correctness. A final
non-Submit action truncates and adds `-1`.

Every step info is exactly:

```python
{
    "step_count": int,
    "steps_remaining": int,
    "action": str,
    "action_applied": bool,
    "submitted": bool,
    "submitted_success": bool,
}
```

`action` is `PitchAction.label` (the human musical label), `action_applied` is false
for Submit and blocked moves, `submitted` is true only for Submit, and
`submitted_success` is false for every non-Submit transition and truncation.

Create `EpisodeResult` only at termination/truncation. `optimal_actions` is
`minimum_action_plan(initial_error, tolerance_cents=5)` from the initial zero controls;
its count is exposed through the derived `optimal_total_actions` property.
`excess_actions` is set only for submitted success.

- [ ] **Step 6: Run Task 3 gates and commit**

```bash
uv run pytest tests/envs/test_sine_pitch.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add pyproject.toml uv.lock src/harpy/envs/sine_pitch.py tests/envs/test_sine_pitch.py
git commit -m "feat: add deterministic sine pitch gym"
```

---

### Task 4: Gym Registration, Checker, and Qt-Free Import Boundary

**Files:**
- Modify: `src/harpy/envs/__init__.py`
- Create: `tests/envs/test_registration.py`
- Modify: `tests/test_imports.py`

**Interfaces:**
- Produces idempotent `register_envs() -> None`.
- Registers the three exact public IDs and mode kwargs without a time-limit wrapper.
- Imports and environment use remain free of Qt, GUI, playback, capture, and devices.

- [ ] **Step 1: Write registration/checker/import RED tests**

Assert importing `harpy.envs` registers all IDs once, repeated `register_envs()` and
module reload are safe, each `gym.make` instance has the correct observation mode, and
all unwrapped instances pass `gymnasium.utils.env_checker.check_env` without Harpy
warnings. The exact registrations are:

```python
{
    "Harpy/SinePitch-v0": ObservationMode.SPECTRUM,
    "Harpy/SinePitchOracle-v0": ObservationMode.ORACLE,
    "Harpy/SinePitchRewardOnly-v0": ObservationMode.REWARD_ONLY,
}
```

Extend the subprocess import smoke to remove/poison PySide6 and assert `harpy.envs`,
environment creation, reset, and one step import no `harpy.gui`/`PySide6` module and
initialize no audio device. Also prove importing top-level `harpy` alone does not load
Gymnasium. Pre-register one of Harpy's IDs with a foreign entry point and prove
`register_envs()` raises an actionable collision error instead of silently accepting or
overwriting it.

- [ ] **Step 2: Run the registration tests and observe RED**

```bash
uv run pytest tests/envs/test_registration.py tests/test_imports.py -q
```

Expected failures name absent exports/registrations, not a test fixture or Gym checker
contract error.

- [ ] **Step 3: Implement registration and final public exports GREEN**

Use exact entry point `harpy.envs.sine_pitch:SinePitchEnv`, mode kwargs, no
`max_episode_steps` wrapper, and inspect Gymnasium's registry before registering. An
existing entry is idempotent only when its entry point and observation-mode kwargs
match exactly; otherwise raise without changing the registry.
Export only `EpisodeResult`, `ObservationMode`, `PitchAction`, `ControlState`,
`SinePitchEnv`, and `register_envs`; invoke registration once from
`harpy.envs.__init__`. Do not import `harpy.envs` from top-level `harpy`.

- [ ] **Step 4: Run Task 4 gates and commit**

```bash
uv run pytest tests/envs/test_registration.py tests/test_imports.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/__init__.py tests/envs/test_registration.py tests/test_imports.py
git commit -m "feat: register sine pitch gym environments"
```

---

### Task 5: Oracle and Spectrum Planner Baselines

**Files:**
- Create: `src/harpy/envs/baselines.py`
- Create: `tests/envs/test_baseline_planners.py`

**Interfaces:**
- Consumes only public observations/reward/info plus the pure planner/grid constants.
- Produces `BaselineKind(StrEnum)` with fixed lane mapping plus pure
  `oracle_plan(observation) -> tuple[PitchAction, ...]` and
  `spectrum_peak_plan(observation) -> tuple[PitchAction, ...]`.
- Neither planner receives an environment, reward, evaluator result, or private truth.
- `BaselineKind` values are exactly `random`, `spectrum_peak`, `oracle`, and
  `reward_search`; its `observation_mode` and `environment_id` properties define the
  fixed lane matrix without a separate mutable lookup table.

- [ ] **Step 1: Write planner-backed baseline RED tests**

Pin lane assignments and actor capability boundaries:

```python
@pytest.mark.parametrize(
    (kind, mode),
    [
        (BaselineKind.RANDOM, ObservationMode.SPECTRUM),
        (BaselineKind.SPECTRUM_PEAK, ObservationMode.SPECTRUM),
        (BaselineKind.ORACLE, ObservationMode.ORACLE),
        (BaselineKind.REWARD_SEARCH, ObservationMode.REWARD_ONLY),
    ],
)
def test_each_baseline_has_one_declared_lane(kind, mode):
    assert kind.observation_mode is mode
```

For Oracle, derive exact current cents only from its float32 oracle observation, derive
the integer coordinate with `round(100 * float(coordinate))`, derive base error by
subtracting visible control offset and target coordinate, call the tolerance-aware planner, and
assert submitted success for every initial error `-2400..2400` using a pure simulated
control loop before running selected real environments.

For Spectrum Peak, use only `argmax(spectrum)`, map the index to `1100 + 5 * index`,
derive base error from visible controls/target, and request an exact (`tolerance=0`)
planner correction so representation quantization remains inside the real ±5 success
boundary. Poison any oracle/error field in fixtures to prove it is never accessed.

- [ ] **Step 2: Run planner baseline tests and observe RED**

Run `uv run pytest tests/envs/test_baseline_planners.py -q`; expected failure is missing
`harpy.envs.baselines`.

- [ ] **Step 3: Implement Oracle and Spectrum Peak GREEN**

Oracle and Spectrum Peak derive one immutable queued plan and never inspect step reward.
The Oracle planner uses the visible exact coordinate; Spectrum Peak uses only the
visible target/control fields and `argmax` mapped through the public checked-in grid.
Both end in Submit and call `minimum_action_plan` with tolerance `5` and `0`
respectively.

- [ ] **Step 4: Run Task 5 gates and commit**

```bash
uv run pytest tests/envs/test_baseline_planners.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/baselines.py tests/envs/test_baseline_planners.py
git commit -m "feat: add sine gym evidence planners"
```

---

### Task 6: Random and Reward Policies with Evaluation Summaries

**Files:**
- Modify: `src/harpy/envs/baselines.py`
- Create: `tests/envs/test_baseline_evaluation.py`

**Interfaces:**
- Adds random and reward-search episode policies to the two planner policies.
- Produces immutable `BaselineSummary`,
  `evaluate_baseline(kind: BaselineKind, *, episodes: int, seed: int) -> BaselineSummary`,
  and
  `evaluate_all_baselines(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]`
  in fixed baseline order.
- No policy receives the environment during action selection. The evaluator reads
  `episode_result` only after done and never exposes it back to the actor.
- Internal stateful policy objects have one narrow
  `act(observation, previous_reward, previous_info) -> PitchAction` interface. Planner
  policies ignore feedback; reward search consumes only the immediately prior scalar
  reward and actor-safe info. The evaluator alone owns reset/step/result access.
- Produces `RandomPolicy(rng: np.random.Generator)` and `RewardSearchPolicy()` inside
  `harpy.envs.baselines` (they are not re-exported from `harpy.envs`). Their exact
  method is:

```python
def act(
    self,
    observation: Mapping[str, object],
    previous_reward: float | None,
    previous_info: Mapping[str, object] | None,
) -> PitchAction: ...
```

The first call of every episode receives `(observation, None, None)`; each later call
receives the immediately preceding transition's scalar reward and actor-safe info.
Construct a fresh policy per episode. `RandomPolicy` requires an actual
`np.random.Generator`; `RewardSearchPolicy` has no truth/RNG constructor input.

- [ ] **Step 1: Write random-policy and reward-search RED tests**

Random builds the currently legal action list from visible controls, always includes
Submit, and samples with its injected policy RNG. Assert it never chooses a masked
bound action and is deterministic for a fixed policy seed.

Build real reward-only episodes that require each direction and scale. Assert this exact
state machine:

1. visit Octave, then Semitone, then Cent;
2. probe Down first if legal;
3. if the first applied Down probe is nonpositive, undo it once and begin Up;
4. if the first Down probe is positive, continue Down while reward is strictly positive;
5. the first nonpositive applied Down continuation is undone once and advances directly
   to the next scale, without an Up search at that already-established direction;
6. any blocked probe or continuation reverses direction at the same scale without an
   undo—including blocked Down/Up continuations reached at either control bound;
7. after a reversal, continue under the same rule: strictly positive applied moves
   continue; the first nonpositive applied move is undone once and advances the scale;
8. an Up probe following the special nonpositive-first-Down undo likewise continues
   only while positive, then is undone once and advances;
9. whenever `steps_remaining == 1`, Submit immediately.

Track whether the current scale has ever produced a positive applied move. The only
case that explores the opposite direction after a nonpositive applied move is the
initial Down probe before any positive improvement; every later nonpositive applied
move is undone and completes that scale. A blocked transition never changes controls
and therefore never schedules an undo by itself.

Assert it never reads spectrum/oracle keys, never truncates, and always submits by step
64, while allowing measured inaccurate submissions. Include scripted actor-safe
transition traces for a blocked first Down probe, blocked Down continuation, blocked Up
probe, and blocked Up continuation so each no-undo reversal is independently pinned.

- [ ] **Step 2: Run policy tests and observe RED**

Run `uv run pytest tests/envs/test_baseline_evaluation.py -q`; expected failures name
the absent policies, while Task 5 planner tests remain GREEN.

- [ ] **Step 3: Implement Random and Reward Search GREEN**

Implement the two stateful policy objects and only enough rollout-free behavior to
make the policy tests green. Do not add aggregation or inspect `EpisodeResult` yet.

- [ ] **Step 4: Write evaluator/summary RED tests**

Assert the exact baseline/mode matrix, matched latent pairs from environment seed
`seed + i`, stable policy codes, byte-identical trajectories/summaries for repeated
runs, separate unpooled rows, correct metric denominators, no NaN, and no audio/private
truth in serialized summaries. Pin the full immutable `BaselineSummary` field list and
fresh `to_dict` ownership. Mutation-check each stable policy code and one metric
denominator so these tests fail for the intended reason before restoration.

- [ ] **Step 5: Run evaluator tests and observe RED**

Run `uv run pytest tests/envs/test_baseline_evaluation.py -q`; the policy subset stays
GREEN and the new failures name the absent evaluator/summary behavior.

- [ ] **Step 6: Implement evaluator and summary GREEN**

The rollout loop passes a policy only observation plus the previous transition's
reward/info. After done, the evaluator reads `episode_result`. For episode index `i`,
reset with environment seed `seed + i`; create the stochastic policy RNG through
`np.random.SeedSequence([seed, stable_policy_code, i])`, where checked-in codes are
Random `1`, Spectrum Peak `2`, Oracle `3`, Reward Search `4`.

`BaselineSummary` contains literals matching the spec:

```python
baseline: str
environment_id: str
observation_mode: str
episodes: int
seed: int
submitted_success_rate: float
submitted_within_1_cent_rate: float
final_within_5_cents_rate: float
final_within_1_cent_rate: float
mean_absolute_final_error_cents: float
mean_actions: float
mean_excess_actions: float | None
mean_return: float
truncation_rate: float
invalid_action_rate: float
```

Define `type JSONScalar = str | int | float | bool | None` in the module. The summary
provides `to_dict() -> dict[str, JSONScalar]` that returns a new plain mapping.
Rates divide by all requested episodes except invalid-action rate, which divides total
blocked attempts by total accepted action IDs. `mean_actions` uses trajectory length;
submitted-within-one requires both Submit and `<=1`, while final positional rates do
not imply protocol success. Mean excess is averaged over submitted successes only.

Reject booleans/nonpositive episodes and boolean/noninteger/negative seeds before
constructing an environment or `SeedSequence`. Empty successful sets produce
`mean_excess_actions=None`, not NaN. No result array is shared or mutable.

- [ ] **Step 7: Run Task 6 gates and commit**

```bash
uv run pytest tests/envs/test_baseline_planners.py tests/envs/test_baseline_evaluation.py -q
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add src/harpy/envs/baselines.py tests/envs/test_baseline_evaluation.py
git commit -m "feat: evaluate sine gym baselines"
```

---

### Task 7: Machine-Readable Checkpoint Command

**Files:**
- Create: `src/harpy/envs/checkpoint.py`
- Create: `tests/envs/test_checkpoint.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces `main(argv: Sequence[str] | None = None) -> int`.
- Exposes both `python -m harpy.envs.checkpoint` and console script
  `harpy-sine-gym = "harpy.envs.checkpoint:main"`.
- Writes one sorted JSON document to stdout and no files; diagnostics/errors use
  argparse stderr and nonzero exit.

- [ ] **Step 1: Write CLI RED tests**

```python
def test_checkpoint_writes_one_deterministic_labeled_json_document(capsys):
    assert main(["--episodes", "1", "--seed", "7"]) == 0
    first = capsys.readouterr()
    payload = json.loads(first.out)
    assert first.err == ""
    assert payload["schema_version"] == 1
    assert payload["checkpoint_id"] == "harpy-milestone-c-sine-pitch-v0"
    assert payload["config_id"] == "fixed-default-sine-v0"
    assert payload["episodes"] == 1
    assert payload["seed"] == 7
    assert [(row["baseline"], row["observation_mode"]) for row in payload["results"]] == [
        ("random", "spectrum"),
        ("spectrum_peak", "spectrum"),
        ("oracle", "oracle"),
        ("reward_search", "reward_only"),
    ]
```

Run twice for byte-identical stdout. Execute in an empty temporary directory and assert
the directory stays empty. Assert `--episodes 0`, booleans/nonintegers, unknown flags,
and missing values fail through argparse. Run a subprocess with Qt imports poisoned and
assert zero stdout noise outside the JSON document and zero stderr.

- [ ] **Step 2: Run CLI tests and observe RED**

Run `uv run pytest tests/envs/test_checkpoint.py -q`; expected failure is the absent
checkpoint module/entry point.

- [ ] **Step 3: Implement the minimal command GREEN**

Use argparse defaults `--episodes 10` and `--seed 0`; validate episodes as a strictly
positive decimal integer and seed as a nonnegative decimal integer (including an
explicit `--seed -1` failure test). Evaluate the four baseline
kinds in the fixed order and serialize:

```python
{
    "schema_version": 1,
    "checkpoint_id": "harpy-milestone-c-sine-pitch-v0",
    "config_id": "fixed-default-sine-v0",
    "episodes": episodes,
    "seed": seed,
    "results": [summary.to_dict() for summary in summaries],
}
```

Use `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)` followed
by exactly one newline. Do not inspect Git/hardware, write an experiment ledger, emit
progress, import Qt, or add a logging framework.
Each result row includes its exact `environment_id`, so oracle and reward-only results
are never mislabeled as headline spectrum results.

- [ ] **Step 4: Run Task 7 gates and commit**

```bash
uv run pytest tests/envs/test_checkpoint.py -q
uv run harpy-sine-gym --episodes 1 --seed 0 | uv run python -m json.tool >/dev/null
uv run pytest
uv run ruff check .
uv run ruff format --check .
git diff --check
git add pyproject.toml uv.lock src/harpy/envs/checkpoint.py tests/envs/test_checkpoint.py
git commit -m "feat: add sine gym checkpoint command"
```

---

### Task 8: Public Documentation and Acceptance Evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/project-notebook.md`
- Create: `docs/verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md`

**Interfaces:**
- Documents only behavior proven by Tasks 1–7 and fresh Task 8 commands.
- Corrects exploratory notebook statements superseded by the approved spec.
- Adds no production or test compatibility aliases.

- [ ] **Step 1: Update README implemented surface and commands**

Replace “Harpy does not yet provide an RL environment” and the Milestone C roadmap
paragraph with the exact implemented claim. Add:

```bash
uv run harpy-sine-gym --episodes 10 --seed 0
```

Document public imports/registered IDs, the spectrum/oracle/reward-only separation,
Submit-gated ±5 success, and the musical tool bounds. State explicitly that direct
sine resynthesis is not recorded-audio pitch shifting and the baselines are not a
trained RL policy. Keep model training/adapters, new waveforms, chords, real assets,
telemetry, and GUI replay in Roadmap.

- [ ] **Step 2: Correct and extend the project notebook**

Change the episode flow from automatic tolerance termination to Submit-gated success.
Replace the open coarse/fine action list with the exact seven named actions and three
independent bounds. Mark one-shot supervised and learned-RL comparisons as later
checkpoints rather than Milestone C acceptance. Add a 2026-08-09 decision entry naming
the three observation tracks, frozen distribution, reward, and nonclaim.

- [ ] **Step 3: Run fresh acceptance commands and write evidence**

Run and record exact command outputs/counts without claiming subjective audio or model
training:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run python -c "import sys; import harpy.envs; assert not any(name == 'PySide6' or name.startswith('harpy.gui') for name in sys.modules)"
uv run python - <<'PY'
import gymnasium as gym
import harpy.envs
from gymnasium.utils.env_checker import check_env

for environment_id in (
    "Harpy/SinePitch-v0",
    "Harpy/SinePitchOracle-v0",
    "Harpy/SinePitchRewardOnly-v0",
):
    environment = gym.make(environment_id)
    check_env(environment.unwrapped)
    environment.close()
PY
uv run harpy-sine-gym --episodes 2 --seed 0
git diff --check
git diff --check origin/main...HEAD
```

The acceptance document distinguishes automated proof from roadmap. Include the exact
baseline JSON instead of editorializing about model quality.

- [ ] **Step 4: Structural/scope review**

Run:

```bash
if rg -n "PySide6|harpy\.gui|QtAudio|Workbench|SampleHistory|CaptureCoordinator|StableBaselines|stable_baselines|torch|librosa|OpenRouter|telemetry" src/harpy/envs; then exit 1; fi
rg -n "PySide6|harpy\.gui|QtAudio|Workbench|SampleHistory|CaptureCoordinator|StableBaselines|stable_baselines|torch|librosa|OpenRouter|telemetry" tests/envs || true
find src/harpy/envs -type f -name '*.py' -print0 | xargs -0 wc -l
git status --short
```

The first search must return no production dependency hits; test strings used only to
assert forbidden imports must be reviewed rather than mechanically deleted. Confirm
each module matches the Final File Map and no generic trainer/actor/storage abstraction
or dead compatibility path entered the branch.

- [ ] **Step 5: Commit documentation/evidence**

```bash
git add README.md docs/project-notebook.md docs/verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md
git diff --cached --check
git commit -m "docs: record milestone C gym checkpoint"
```

---

## Whole-Branch Completion Gate

After every task has its scoped implementation review, generate a review package from
the branch merge base and dispatch one most-capable whole-branch reviewer. Resolve one
final fix wave if required, then freshly run:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run harpy-sine-gym --episodes 10 --seed 0 | uv run python -m json.tool >/dev/null
uv run python -c "import sys; import harpy.envs; assert not any(name == 'PySide6' or name.startswith('harpy.gui') for name in sys.modules)"
git diff --check origin/main...HEAD
git status --short
```

Completion requires a clean worktree, no open Critical/Important review finding, the
documented JSON checkpoint, and no claim that a learned policy or recorded-audio pitch
shifter exists.
