# Harpy Milestone E Reliable Learned Tuning Implementation Plan

**Goal:** Deliver a reliable learned spectral-perception checkpoint on the unchanged
`Harpy/SinePitch-v0` environment: a spectrum-only pitch estimator that replans through
Harpy's existing Octave, Semitone, Cent, and Submit actions.

**Approved contract:**
`docs/superpowers/specs/2026-08-22-harpy-milestone-e-reliable-learned-tuning-design.md`.
The design's frozen environment boundary, split identities, suite construction,
artifact compatibility, CLI grammar, scientific thresholds, and nonclaims govern this
plan.

**Implementation base:** `william/milestone-e-reliable-tuning`, created from the
normal-ancestry recovered Milestone D commit `9bc0fff`. Do not implement on the
historical broken-ancestry Milestone D checkout. Do not rewrite, discard, or force-push
recovered history.

**Method:** Strict RED → observed failure → minimal GREEN → refactor. Each task owns a
small commit and runs its focused tests, the accumulated Milestone E slice, Ruff, Ruff
format check, and `git diff --check`. Optional Torch/SB3 imports remain lazy and Qt-free.

## Frozen boundaries

- Do not change the registered Gym environment, seven actions, observation keys,
  rewards, rendering, reset semantics, 64-step budget, or registration kwargs.
- The estimator receives only the raw `(1961,)` spectrum. It never receives target,
  controls, reward, hidden pitch, source coordinate, suite identity, or evaluator data.
- The planner reconstructs target/control arithmetic only from public observation
  fields and replans after every action.
- Existing schema-v1 BC/PPO bytes and behavior remain strict-loadable and unchanged.
- Only `train-pitch` creates schema-v2 pitch artifacts. Existing `train-bc` and
  `train-ppo` keep schema v1.
- CPU is authoritative. CUDA is explicit and criterion-ineligible.
- Final suites and final coordinate records are inaccessible to training and model
  selection.
- A failed scientific criterion still produces valid artifacts and a
  `criterion_not_met` report; it is not an engineering failure.

## Execution topology

```text
Wave 1: Task 1
Wave 2: Task 2 | Task 3
Wave 3: Task 4 | Task 5
Wave 4: Task 6
Wave 5: Task 7
Wave 6: Task 8
Wave 7: Task 9
Wave 8: Task 10
```

Tasks 2 and 3 start only after the pinned split/suite contracts exist. Tasks 4 and 5
may then proceed in parallel: training/model persistence versus diagnostics. Task 6
integrates strict schema-v2 artifacts, Task 7 integrates final evaluation, and Task 8
owns CLI/workflow closure. Tasks 9 and 10 are real execution and acceptance rather
than additional product scope.

## Task 1: Pin data, suite, and action-mask contracts

**Production files:**

- Modify `src/harpy/learning/models.py`
- Modify `src/harpy/learning/suites.py`
- Create `src/harpy/learning/pitch_data.py`
- Create `src/harpy/learning/action_masks.py`

**Test files:**

- Create `tests/learning/test_pitch_data.py`
- Extend `tests/learning/test_models.py`
- Extend `tests/learning/test_suites.py`
- Create `tests/learning/test_action_masks.py`

**RED contracts:**

- Add `TrainerKind.PITCH` only through a schema-aware contract; do not make v1 codecs
  accept it.
- Pin `harpy-sine-pitch-grid-v1`, the ordered 1,400/200/401 central split, the two
  ordered 200-coordinate OOD pools, the canonical split preimage, and its literal
  SHA-256 digest.
- Independently reconstruct and pin all four E-suite digests before any trainer code:
  smoke 50, IID 250, OOD-lower 200, OOD-upper 200.
- Prove exact per-target counts, distance exclusion, source pools, pair uniqueness,
  declared order, initial-source separation, and seed independence.
- Pin the single v2 shuffled-spectrum permutation from the IID digest.
- Exhaustively test all seven legal-action-mask entries at and around every public
  control bound. Submit is always legal; inputs contain only public controls.

**GREEN:** Implement immutable dependency-light contracts and cached deterministic
constructors. Keep the historical v1 suite functions byte-for-byte compatible.

**Gate:** focused tests pass twice with identical literal digests; import tests prove
no Torch, SB3, or Qt import.

## Task 2: Implement the location-preserving estimator and actor

**Production files:**

- Create `src/harpy/learning/pitch_network.py`
- Create `src/harpy/learning/pitch_actor.py`
- Modify `src/harpy/learning/actors.py` only for typed, backward-compatible dispatch

**Test files:**

- Create `tests/learning/test_pitch_network.py`
- Create `tests/learning/test_pitch_actor.py`

**RED contracts:**

- Exact Conv1d topology `1→16→16→1`, kernels `9,9,1`, stride one, padding `4,4,0`,
  ReLU placement, 1,961 logits, parameter names/shapes/count, and finite state.
- No flattening, pooling, stride, recurrence, target/control input, positional input,
  or handcrafted peak feature.
- Pin nearest-five-cent label boundaries and first-maximum tie behavior.
- Reject wrong-shaped, wrong-dtype, nonfinite input/output and malformed model state.
- Validate the full raw observation allowlist on every decision.
- Test target arithmetic, visible-control offset, `[-2400,2400]` clamp,
  tolerance-zero planning, first-action execution, and action legality.
- Prove the estimator is called again after every environment observation and that no
  fallback or hidden truth reaches it.

**GREEN:** Implement the fully convolutional scorer, strict inference adapter,
immutable `PitchDecision`, and stateless `PitchPlannerActor`. Preserve the narrow
`Actor.act(observation)` protocol.

## Task 3: Implement trajectory diagnostics and masked-BC adapter

**Production files:**

- Create `src/harpy/learning/diagnostics.py`
- Create `src/harpy/learning/diagnostic_codecs.py`
- Extend `src/harpy/learning/actors.py` with an explicit `MaskedBCActor`

**Test files:**

- Create `tests/learning/test_diagnostics.py`
- Create `tests/learning/test_diagnostic_codecs.py`

**RED contracts:**

- Craft episodes for commuting shortest actions, canonical mismatch, true divergence,
  early/late Submit, bound blocks, repeated-state visits, loop transitions, and
  recoverable/unrecoverable budgets.
- Pin canonical 7×7 confusion matrices, per-action precision/recall/support,
  set-valued shortest-action accuracy, divergence/mismatch steps, survival curves,
  Submit error bands, and null rules.
- Estimator fields appear only through typed `PitchDecision`; legacy actors record
  null estimates.
- Diagnostic JSON rejects duplicate keys, nonfinite values, derived-field mismatch,
  duplicate/missing seeds, bad inventory, and output aliases.
- Masked BC validates logits then masks before first-maximum selection. It does not
  mutate or silently replace legacy `BCActor` semantics.

**GREEN:** Implement evaluator-owned diagnostics with strict immutable report/bundle
codecs and dependency-light action-path analysis.

## Task 4: Implement deterministic pitch training

**Production files:**

- Create `src/harpy/learning/pitch.py`
- Extend `src/harpy/learning/dependencies.py` only as needed for lazy Torch loading

**Test files:**

- Create `tests/learning/test_pitch_training.py`
- Extend `tests/learning/test_import_boundaries.py`

**RED contracts:**

- Build ordered immutable spectra only from training/validation coordinates using the
  existing real renderer and `SpectrumEvidenceCache`.
- Training spies fail if any final IID/OOD coordinate or E final suite is accessed.
- Pin smoke/checkpoint profiles, AdamW settings, batch size, deterministic shuffle,
  seed initialization, `num_workers=0`, and validation ordering.
- Recompute loss, MAE, median error, within-one, and within-five metrics from raw
  prediction records.
- Pin full lexicographic epoch rank, exact-tie retention, patience behavior, deep CPU
  state capture, selected-state restoration, and persisted-byte reload.
- Same-runtime same-seed CPU smoke training produces bit-identical selected state and
  metrics; wrong/nonfinite state fails closed.

**GREEN:** Implement the supervised trainer with no action labels, rewards, final
suite access, or evaluator records.

## Task 5: Add direct-coordinate evaluation models and criteria

**Production files:**

- Extend `src/harpy/learning/models.py`
- Extend `src/harpy/learning/evaluation.py`
- Create `src/harpy/learning/pitch_evaluation.py`

**Test files:**

- Create `tests/learning/test_pitch_evaluation.py`
- Extend `tests/learning/test_criteria.py`

**RED contracts:**

- Persist one ordered `PitchCoordinateRecord` per seed and final coordinate.
- Decode by re-deriving grid index, predicted cents, signed error, absolute error,
  every count/rate, MAE, nearest-rank percentile, residue row, and register row.
- Model lower and upper OOD as separate suites and a typed two-register aggregate;
  never synthesize a combined suite row.
- Pin the exact learned/baseline/probe row matrix and complete criterion recomputation.
- Prove zero/shuffled probes change only spectrum evidence and use the exact fixed
  permutation.

**GREEN:** Add direct perception evaluation and closed-loop pitch-planner evaluation
without weakening historical baseline/evaluation behavior.

## Task 6: Add strict schema-v2 pitch artifacts

**Production files:**

- Extend `src/harpy/learning/artifacts.py`
- Extend `src/harpy/learning/pitch.py`
- Add small schema-specific codec modules if needed to keep v1/v2 dispatch explicit

**Test files:**

- Extend `tests/learning/test_artifacts.py`
- Create `tests/learning/test_pitch_artifacts.py`
- Add canonical fixtures under `tests/fixtures/learning/`

**RED contracts:**

- Freeze complete canonical schema-v1 BC/PPO fixture families and prove byte-identical
  load/re-encode after v2 exists.
- Dispatch schema first and reject every cross-schema trainer/profile/suite pairing.
- Pin schema-v2 manifest/config/summary/evaluation fields, compatibility digest,
  exact six-file inventory, hashes, strict trusted-local `model.pt`, and eligibility.
- Test every bootstrap/write/reload/smoke/complete/interruption boundary, incomplete
  recovery behavior, create-only paths, unknown files, corrupted payloads, and no
  fallible work after final completion rename.
- Exact-three aggregate preflight rejects missing, duplicate, extra, wrong-seed,
  ineligible, dirty-source, CUDA, source/lock mismatch, and compatibility mismatch
  before model/cache/suite construction.

**GREEN:** Reuse the existing atomic lifecycle while keeping immutable v1 and v2
registries separate. Only schema-v2 pitch artifacts may access E final evaluation.

## Task 7: Integrate checkpoint evaluation and diagnostics

**Production files:**

- Extend `src/harpy/learning/workflows.py`
- Extend `src/harpy/learning/evaluation.py`
- Extend `src/harpy/learning/trace.py`
- Integrate `src/harpy/learning/diagnostics.py`

**Test files:**

- Extend `tests/learning/test_workflows.py`
- Extend `tests/learning/test_evaluation.py`
- Extend `tests/learning/test_trace.py`
- Add `tests/learning/test_pitch_workflows.py`

**RED contracts:**

- Preflight all three artifacts before constructing actors, suites, caches, or outputs.
- Pin exact smoke and checkpoint row order, per-seed records, baseline identities,
  two-register aggregate, direct-coordinate tables, and criterion state.
- Same artifact and episode seed render byte-identical canonical traces.
- Loaded pitch actions match the trainer's persisted-reload actor.
- Final diagnostics require the same exact-three preflight; legacy diagnostics remain
  limited to historical suites.

**GREEN:** Add exhaustive trainer/schema dispatch; remove no historical paths and use
no `if BC else PPO` fallthrough.

## Task 8: Close the CLI and hands-on workflow

**Production files:**

- Extend `src/harpy/learning/cli.py`
- Update `README.md`

**Test files:**

- Extend `tests/learning/test_cli.py`
- Extend `tests/learning/test_training_smoke.py`

**RED contracts:**

- Pin `train-pitch` and `diagnose` grammar plus pitch-aware `evaluate` and `run`.
- Validate aliases, exact-three inputs, create-only output, output/input separation,
  trusted-local stderr warning, canonical stdout, unavailable CUDA, missing dependency,
  and stable `0/1/2/130` exits.
- `--help` and ordinary Harpy imports remain Torch/SB3-lazy and Qt-free.
- Run one real CPU smoke train → persisted reload → diagnose → evaluate → run cycle;
  capture a trace twice and compare bytes.

**GREEN:** Wire only the approved commands. Document precise claims, commands,
artifact trust boundary, and exclusions.

## Task 9: Run checkpoint science

Train fresh checkpoint artifacts for seeds `0`, `1`, and `2` in explicit new output
directories. Do not overwrite or reuse Milestone D artifacts. After all three close,
run the aggregate evaluator and final diagnostics exactly once through the strict
preflight.

Record:

- artifact and manifest hashes;
- source commit and lock digest;
- direct IID/lower-OOD/upper-OOD perception metrics per seed;
- all terminal/probe/baseline rows;
- action efficiency and failure reasons;
- criterion state and any deterministic warnings.

If the criterion is not met, diagnose the persisted evidence before changing a
preregistered contract. Any material architecture/data/threshold change requires a
new reviewed version rather than silently moving this milestone's gate.

## Task 10: Acceptance and independent review

**Files:**

- Create
  `docs/verification/2026-08-27-milestone-e-reliable-learned-tuning-acceptance.md`

Run:

```bash
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Also run one dependency-free import/CLI gate, the real smoke workflow, strict reload of
all three checkpoint artifacts and final reports, duplicate hands-on trace comparison,
and mutation checks that break the grid label, spatial architecture, replanning,
legal mask, v1 dispatch, three-seed preflight, direct-record criterion, and completion
rename.

Obtain independent reviews for:

1. environment immutability and hidden-truth leakage;
2. scientific split/evaluation correctness;
3. artifact/CLI atomicity and v1 compatibility.

Milestone E is complete only when the tracked worktree is clean, no trainer remains
running, persisted evidence strict-loads, and the acceptance document reports the
actual criterion outcome without inflating the claim.

## Post-E release checkpoint

Release hardening is deliberately separate from Milestone E science. After E closes:

- merge/publish the normal-ancestry recovered history;
- choose and add the project license plus third-party notices;
- add locked Python 3.12 CI, build/install-from-artifact smoke, and headless GUI smoke;
- complete the declared native GUI/audio support checks;
- finalize metadata, compatibility statement, changelog, release notes, and hashes;
- cut an RC, verify it from a clean clone/machine, then tag `v1.0.0`.
