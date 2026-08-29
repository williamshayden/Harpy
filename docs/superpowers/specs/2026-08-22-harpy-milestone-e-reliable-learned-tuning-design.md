# Harpy Milestone E: Reliable Learned Tuning Design

Date: 2026-08-22

Status: Approved by the user on 2026-08-27. This document is the locked scientific
and product contract for implementation and acceptance.

Amendment notice (2026-08-28): this preregistration and its CPU outcome remain
unchanged. The approved additive CUDA-cohort protocol is specified separately in
`2026-08-28-harpy-milestone-e1-cuda-device-cohort-design.md`; it does not
retroactively alter any schema-v2 artifact, report, criterion, or conclusion here.

## Outcome

Milestone E turns the first learned-policy checkpoint into a reliable learned
perception checkpoint on the unchanged `Harpy/SinePitch-v0` task. A compact,
location-preserving neural estimator reads the actor-visible candidate spectrum,
predicts its pitch on Harpy's existing five-cent analysis grid, and feeds that estimate
to the existing bounded `minimum_action_plan` planner. The actor replans after every
observation and still executes the real Octave, Semitone, Cent, and Submit actions.

The intended claim is deliberately narrow:

> Learned spectral perception can reliably close the frozen single-sine tuning task
> through Harpy's frozen symbolic controls and planner.

This is a learned-audio result, but it is not an end-to-end reinforcement-learning
mastery claim. The classical Spectrum Peak and Oracle lanes remain controls, and PPO
remains reported as the Milestone D result until a later masked-PPO comparison is
explicitly authorized.

The hands-on workflow remains one command to train, one to evaluate, one to diagnose,
and one to watch a complete action trace.

## Why this milestone

Milestone D established that the environment and evidence are sufficient while also
isolating the learned-policy failure:

- Spectrum Peak submitted successfully on every IID and register-OOD episode, proving
  the spectrum, environment, reward, and planner can solve this clean-sine task.
- The checkpoint BC model reached `0.8834252450980392` held-out next-action accuracy,
  yet only 6 of 256 IID rollouts submitted successfully.
- BC submitted in 254 of 256 IID episodes, had only about 0.09% blocked actions, and
  truncated only about 0.78% of episodes. Its dominant failure was premature Submit
  after compounded classification errors, not illegal controls.
- The average oracle trajectory had about 25.5 actions. As a naive independence
  heuristic, multiplying marginal per-action BC accuracy across that horizon gives
  about 4.2% exact canonical-trajectory survival, close to the observed 2.34% rollout
  success. This is motivation, not an independence claim or a substitute for the new
  set-valued trajectory diagnostics.
- PPO truncated 33.6% to 87.1% of IID episodes across seeds. Bound masks are useful
  infrastructure for a later PPO comparison, but masking alone cannot explain or fix
  the BC result.
- The Milestone D encoder uses two stride-four convolutions and adaptive pooling from
  121 positions to 16. That is a plausible bottleneck for fine pitch localization.
- BC labels also encode one canonical Octave-then-Semitone-then-Cent ordering even
  though many shortest control actions commute. Canonical next-action accuracy alone
  therefore understates valid planning behavior while still failing to measure
  closed-loop recovery.

Milestone E addresses the representation and rollout boundary directly instead of
running a larger policy or another five-seed PPO sweep without a sharper hypothesis.

## Scope

Milestone E includes:

- deterministic trajectory diagnostics that distinguish canonical disagreement from
  leaving the set of shortest successful actions;
- a dependency-light legal-action mask derived only from public control state;
- a spectrum-only, location-preserving five-cent pitch estimator;
- a stateless estimator-plus-planner actor that replans on every observation;
- deterministic train/selection/initial-source-holdout pitch-coordinate splits and
  new final episode suites fixed before training;
- smoke and checkpoint training profiles;
- three fresh checkpoint training seeds;
- strict, versioned, hash-checked estimator artifacts with legacy Milestone D loading;
- `train-pitch` and `diagnose` CLI commands integrated with the existing `evaluate`
  and `run` workflow;
- real smoke training and a final checkpoint report with spectral ablations.

Milestone E does not include:

- any change to `Harpy/SinePitch-v0` actions, observations, rewards, bounds, rendering,
  registration, terminal semantics, or 64-action budget;
- a target or reference spectrum;
- Analyze Source, Analyze Candidate, or any other tool-use action;
- a Spectrum Peak fallback inside the learned actor;
- hidden pitch, signed error, source coordinate, suite membership, reward components,
  or evaluator records at inference;
- DAgger training, a new direct-action BC checkpoint, or a five-seed masked-PPO run;
- BC-to-PPO or BC-to-estimator warm starts;
- waveform, patch, oscillator, polyphonic, noisy-audio, recorded-audio, or pitch-shift
  expansion;
- GUI, live-audio, hosted-training, database, telemetry, or experiment-platform work.

DAgger and masked PPO are follow-on experiments after the estimator checkpoint. They
are specified at the end of this document so their boundaries are not lost, but they
are not Milestone E acceptance gates.

## Frozen environment and actor boundary

The registered environment remains byte-for-byte and behaviorally unchanged. The
estimator actor receives exactly the current raw spectrum observation:

- `spectrum`: float32 `(1961,)` on the existing fixed five-cent grid;
- `target_note`: public note index;
- `controls`: public Octave, Semitone, and Cent values;
- `steps_remaining`: public remaining budget.

The actor never reads environment internals, `EpisodeResult`, reset options, source
pitch, target pitch in cents, current pitch in cents, or reward. Training may use
candidate-pitch labels because this lane is explicitly supervised; those labels are
not serialized into observations and are not available to the loaded actor.

The existing `Actor` protocol remains observation-only:

```python
class Actor(Protocol):
    def act(self, observation: Mapping[str, object]) -> PitchAction: ...
```

No new observation key is added for action masks or estimates.

## Location-preserving pitch estimator

### Output contract

The estimator consumes only the spectrum tensor. It returns exactly 1,961 finite
logits, one for each public analysis-grid center from 1,100 through 10,900 cents in
five-cent increments. Inference selects the first maximum and maps it back to:

```text
estimated_candidate_cents = 1100 + 5 * argmax(logits)
```

The network does not consume the target note, controls, budget, reward, or a handcrafted
peak coordinate. This makes its scientific responsibility explicit: localize the
current sine from audio-derived evidence.

### Architecture

The checked-in v1 estimator is a small fully convolutional scorer:

```text
spectrum (batch, 1961)
  -> add one channel
  -> Conv1d(1, 16, kernel=9, stride=1, padding=4) + ReLU
  -> Conv1d(16, 16, kernel=9, stride=1, padding=4) + ReLU
  -> Conv1d(16, 1, kernel=1, stride=1)
  -> logits (batch, 1961)
```

It has no striding, flattening, adaptive/global pooling, recurrent state, or positional
embedding. Translation-equivariant shared weights preserve frequency location while
keeping the model intentionally small. Model type, parameter names, shapes, count,
finite values, and architecture identifier are strict artifact contracts.

### Labels and loss

Every training item is one immutable spectrum paired with the nearest public grid
center. For integer candidate cents, the class index is:

```text
(candidate_cents - 1100 + 2) // 5
```

This gives a deterministic nearest-center rule with no half-bin tie. Training uses
ordinary unweighted cross-entropy. It does not use inverse action-frequency weights,
teacher actions, rewards, rollout outcomes, or final evaluation suites.

The optimizer is AdamW with learning rate `0.001`, weight decay `0.0001`, batch size
64, and deterministic per-run shuffling. The checkpoint profile permits 50 epochs
with early-stopping patience 8. The selected epoch is ranked by highest validation
within-five-cent rate, then highest within-one-cent rate, then lowest mean absolute
cent error, then lowest validation loss, then earliest epoch. The smoke profile uses
the same optimizer with two epochs and no scientific eligibility.

At trainer entry, the nonnegative run seed initializes Python, NumPy, Torch CPU, and
requested Torch-device RNGs; deterministic Torch algorithms are enabled, cuDNN
benchmarking is disabled, and cuDNN deterministic mode is enabled when present. The
training DataLoader uses `num_workers=0`, the ordered coordinate dataset, and a CPU
`torch.Generator().manual_seed(seed % (2**63 - 1))`; validation never shuffles. No
global RNG is used for suite or coordinate construction.

The complete lexicographic rank above determines improvement. Exact rank ties retain
the earlier epoch; equivalently the minimized rank is
`(-within5_count, -within1_count, mean_absolute_error, validation_loss, epoch)` over
the fixed validation denominator. Patience resets only on a strictly better full rank; training stops
immediately after eight consecutive completed non-improving epochs. The selected
state is a deep, finite CPU state-dictionary copy captured at improvement time,
restored before persistence, and independently reloaded from persisted bytes. The
summary records framework/device determinism settings; bit-identical weights are
required only for repeated same-runtime CPU tests, not across PyTorch releases or
hardware.

## Deterministic coordinate data

The estimator data distribution is versioned separately from episode suites as
`harpy-sine-pitch-grid-v1`.

Central candidate coordinates are every integer cent from 5,000 through 7,000. For
each coordinate `p`, the split key is the raw SHA-256 digest of the ASCII bytes
`"harpy-sine-pitch-grid-v1:{p}"`. Coordinates are ordered by `(digest_bytes, p)` and
divided without replacement into:

- 1,400 checkpoint training coordinates;
- 200 checkpoint validation coordinates;
- 401 untouched IID source coordinates reserved for final episode construction.

The smoke profile uses the first 256 checkpoint-training coordinates and first 64
checkpoint-validation coordinates. Smoke is diagnostic and cannot satisfy the final
criterion.

The register-OOD source pools are all 200 integer coordinates from 4,800 through
4,999 and all 200 from 7,001 through 7,200. No OOD coordinate is used for estimator
training or selection.

Spectra come through the existing immutable `SpectrumEvidenceCache` and the real
Milestone C renderer. Cache identity, rendering, FFT configuration, normalization,
and the 1,100-through-10,900 reachable range remain unchanged. The model never sees a
materialized coordinate value as an input.

The split digest preimage is canonical UTF-8 JSON plus one trailing newline, encoded
with sorted keys, compact separators, and nonfinite values forbidden. It contains
exactly `schema_version`, `distribution_id`, `training_coordinates`,
`validation_coordinates`, `iid_holdout_coordinates`, `ood_lower_coordinates`, and
`ood_upper_coordinates`; the digest itself is excluded. Lists retain their declared
order. The partition is seed-independent, so the resulting single v1 split SHA-256
hex digest is checked in during the first suite/model TDD task, before trainer work.

Epoch selection uses only the 200-coordinate validation split. After all three models
are trained, closed, and proven to share identical source/lock/split provenance and
the same seed-neutral compatibility digest, the aggregate evaluator measures direct pitch-estimation performance on
all 401 untouched IID coordinates and all 400 OOD coordinates. Those final direct
coordinate metrics, not the selection metric, own the perception criterion.

The episode suites are initial-source-disjoint from training, not globally
observation-coordinate-disjoint. Octave, Semitone, and Cent trajectories traverse the
same finite musical grid and may legitimately revisit a training coordinate after
reset. The final direct-coordinate evaluation provides the disjoint perception test;
episode rollouts provide the closed-loop control test.

## New final episode suites

Milestone D's observed suites remain historical evidence and are not reused as
Milestone E final gates. New suite membership is generated, serialized, and digest
locked before training starts:

- `harpy-sine-pitch-e-iid-v1`: 250 episodes, exactly ten per target note, with source
  coordinates drawn from the 401-coordinate IID holdout pool;
- `harpy-sine-pitch-e-ood-lower-v1`: 200 episodes, exactly eight per target note,
  using only 4,800-through-4,999 sources;
- `harpy-sine-pitch-e-ood-upper-v1`: 200 episodes, exactly eight per target note,
  using only 7,001-through-7,200 sources;
- `harpy-sine-pitch-e-smoke-v1`: 50 ineligible episodes, exactly two per target note.

Every pair excludes initial distance within five cents. Suite codes are `401` for
smoke, `402` for IID, `403` for lower OOD, and `404` for upper OOD. For each target
index, selection uses
`default_rng(SeedSequence([1, suite_code, target_index])).choice(...)` over the sorted
eligible source list with `replace=False` and an exact size of two, ten, or eight as
declared above. Episodes are ordered first by ascending target index and then by the
returned draw order. Source reuse across different targets is allowed; duplicate
`(target, source)` pairs are forbidden. Smoke sources come from the 200-coordinate
validation split. The exact membership digests are checked in before trainer code is
written.

Each suite uses `suite_seed = suite_code`. Its digest preimage is canonical UTF-8 JSON
plus one trailing newline with exactly `schema_version: 2`, `suite_id`, `suite_seed`,
and `episodes`. Each ordered episode object contains exactly `episode_index`,
`target_note_index`, and `source_pitch_cents`; the digest itself is excluded. Encoding
uses sorted keys, compact separators, and forbids nonfinite numbers. The first suite
TDD task independently reconstructs and checks in all four golden SHA-256 hex digests
before any trainer imports or reads these suites.

Zero-spectrum and deterministically shuffled-spectrum probes use the exact IID
episode membership. For v2 shuffling, the 32-byte suite digest is decoded as eight
unsigned 32-bit big-endian integers. The single permutation is
`default_rng(SeedSequence([1, 405, *digest_words])).permutation(1961)` and is applied
unchanged to every episode and actor in that suite. Milestone D v1 retains its
existing global permutation exactly.

Lower and upper OOD remain separate suite rows with separate identities and digests.
A typed `RegisterOODAggregate` carries both suite IDs/digests and derives combined
counts and metrics from their concatenated terminal records. There is no synthetic
combined suite row, and the aggregate cannot be constructed from only one register.

The final report persists one `PitchCoordinateRecord` for every seed and every final
coordinate, ordered by seed and then ascending coordinate within IID, lower OOD, and
upper OOD. Each record contains split identity, true coordinate, predicted grid index,
predicted cents, signed error, and absolute error. Decoding re-derives every field and
all aggregate rates; aggregate-only evidence is invalid.

Direct-error percentiles are nearest-rank values at p50, p90, p95, and p99, using
`sorted_errors[ceil(q * n) - 1]`, plus maximum. The residue table has exact rows for
true coordinate modulo five `0..4`; the register table has `iid`, `ood_lower`, and
`ood_upper`. Every row stores count, within-one count/rate, within-five count/rate,
mean absolute error, and the four percentiles. This gives complete perception coverage
even when target-balanced episode suites reuse a source across different targets.

## Estimator-planner actor

`PitchPlannerActor` validates the complete raw observation on every call, obtains the
estimated current candidate pitch from the spectrum-only model, and reconstructs only
public musical arithmetic:

```text
target_cents = 4800 + 100 * target_note
visible_offset = 1200 * octaves + 100 * semitones + cents
estimated_base_error = estimated_candidate_cents - visible_offset - target_cents
estimated_base_error = clamp(estimated_base_error, -2400, 2400)
action = minimum_action_plan(
    estimated_base_error,
    controls,
    tolerance_cents=0,
)[0]
```

The actor executes only the first planned action. It retains no plan cursor and
repeats estimation and planning after every new observation. A correct nearest-grid
classification has at most two cents of quantization error; closed-loop replanning
keeps that residual from becoming a long open-loop drift.

For typed diagnostics, `PitchPlannerActor` also exposes
`decide(observation) -> PitchDecision`, where the immutable decision contains the
chosen action and estimated candidate cents. `act()` delegates to `decide()` and
returns only its action, preserving the narrow shared actor protocol. The diagnostic
runner records a null estimate for legacy actors that have no decision capability; it
never reads mutable `last_estimate` state.

There is no classical peak fallback, confidence shortcut, hidden-state correction,
or evaluator-side rescue. Any nonfinite/wrong-shaped estimator output is an execution
error, not Submit.

## Legal-action masks

`legal_action_mask(controls)` is a dependency-light, pure function returning seven
booleans in exact `PitchAction` order. It masks only a mutation that would cross its
current public bound. Submit is always legal. Remaining budget does not alter the mask
in this milestone.

The estimator-planner actor verifies that its planned action is legal. A separate
explicit `MaskedBCActor` diagnostic adapter may apply the mask to already validated
finite BC logits before first-maximum selection. `diagnose --bound-mask` is accepted
only for a BC artifact and uses a dedicated strict masked-BC loader; the runner does
not reach into an existing `BCActor`'s private model. Without that flag diagnostics
use original semantics. Existing Milestone D `BCActor` and `PPOActor` behavior remains
unchanged so loading an old artifact reproduces its original actions.

PPO masks must eventually affect sampling, log probabilities, entropy, and training
loss. Post-hoc replacement of an illegal PPO action is forbidden. That requires a new
versioned policy and PPO artifact identity and is deferred with the actual masked-PPO
run.

## Diagnostics

Diagnostics rerun an actor on a fixed suite because terminal Milestone D records do
not contain enough trajectory detail. Evaluator-owned truth may be used to score the
actor but is never passed to `act()`.

Each `DiagnosticReport` contains:

- a canonical 7-by-7 teacher/prediction confusion matrix;
- per-action precision, recall, and support;
- set-valued shortest-action accuracy;
- first canonical mismatch step and action;
- first step that leaves every shortest successful path;
- a survival curve for remaining on at least one shortest path;
- Submit decisions grouped by true absolute-error bands `0..1`, `2..5`, `6..25`,
  `26..99`, and `100+` cents;
- legal, bound-blocked, repeated-state, and loop counts;
- shortest remaining successful length versus public steps remaining;
- optional estimated candidate cents plus signed/absolute estimator error, present
  only when the actor exposes a typed `PitchDecision`;
- terminal reason, final error, action count, and excess actions.

Diagnostic decision steps are one-based and refer to the pre-action state. The
canonical teacher is the first action from `minimum_action_plan` using true base error,
current controls, and tolerance five. An action belongs to the shortest-action set
exactly when it is legal and its successor's tolerance-five shortest successful plan
is one action shorter. Membership is musical and independent of remaining budget;
recoverability is reported separately as `shortest_plan_length <= steps_remaining`,
with Submit included in plan length. Submit belongs to the set only when true current
error is within five cents.

Within an episode, state identity is the public `ControlState`; source is fixed, so
this also fixes true candidate pitch. Decreasing `steps_remaining` is deliberately
excluded. A repeated-state visit is every pre-action visit after the first to an
already seen state. A loop transition is a nonterminal action whose successor state
has already appeared; the two counts are stored separately. No mismatch, divergence,
loop, or unrecoverable step is JSON `null`, never a sentinel integer.

Precision or recall with a zero denominator is JSON `null`; raw numerator,
denominator, and support are always retained. Each survival entry contains one-based
step, the number of episodes that actually took that decision (`at_risk`), the number
with no shortest-set divergence through that decision (`survivors`), and
`survivors / at_risk`; entries stop when `at_risk` is zero. These rules separate
harmless canonical ordering differences from genuine divergence and keep canonical
JSON free of NaN.

The `diagnose` command supports legacy BC/PPO artifacts and new pitch artifacts. A
masked legacy BC lane is labeled as a diagnostic wrapper; it is never written back
into or confused with the original artifact.

`DiagnosticReport` has its own strict schema ID and duplicate/nonfinite-rejecting
canonical codec. It records input artifact manifest hash, actor semantics, bound-mask
mode, suite ID/digest, ordered episode records, derived matrices, and optional
estimator fields. Its output path is create-only and must be outside the input
artifact directory.

Every `diagnose` invocation emits a `DiagnosticBundle`. A single-artifact smoke or
legacy invocation contains exactly one report. An E final-suite invocation contains
exactly three reports ordered by seeds 0, 1, and 2; each report records its own seed
and manifest hash. The bundle records schema, suite ID/digest, device, mask mode,
ordered report identities, and bundle-level inventory. It rejects duplicate or missing
seeds and re-derives every report before canonical encoding.

## Artifacts and compatibility

The estimator is a distinct supervised trainer kind, `pitch`; it is not called BC or
RL. Artifact dispatch becomes exhaustive across `bc`, `ppo`, and `pitch` rather than
using an `if BC else PPO` fallback.

Pitch artifacts reuse the existing create-only lifecycle:

1. create and publish an incomplete manifest;
2. write strict training config, summary, and `model.pt` payloads;
3. reload the persisted model through the actor adapter;
4. run only the ineligible smoke evaluation and its two probes;
5. atomically complete the manifest last;
6. return the already validated completion object.

Checkpoint training never runs the final IID, OOD, or direct-coordinate gates. Each
seed closes independently with the exact six-file inventory
`manifest.json`, `training-config.json`, `training-summary.json`, `model.pt`,
`evaluation-smoke.json`, and `evaluation-smoke-probes.json`. After all three artifacts
are closed, the aggregate evaluator first proves identical source commit and lock plus
one seed-neutral compatibility digest covering profile, architecture, preprocessing,
and coordinate-split IDs/digests, then runs all final gates without changing any
artifact.

The pitch manifest records coordinate-split IDs/digests and counts, architecture and
preprocessing IDs, profile, seed, selected epoch, parameter count, runtime/source/lock
provenance, evaluation suite IDs/digests, device, payload inventory, sizes, and hashes.
The summary records epoch loss plus mean/median absolute error and within-one/within-
five rates for training and validation.

Each manifest also records a derived `compatibility_sha256`. Its canonical preimage
contains schema/trainer, checkpoint profile and every hyperparameter, environment and
spectrum schema, action identity/order, architecture, preprocessing and policy
semantics, coordinate distribution/split digest, smoke/final suite IDs/digests, and
required device policy. It explicitly excludes run seed, timestamps, output path,
runtime host, wall times, selected epoch/metrics, and payload/model hashes. The loader
recomputes it; exact-three aggregation requires the same compatibility digest while
separately requiring the distinct seed set `{0, 1, 2}`.

An eligible checkpoint pitch artifact records `eligible_for_aggregate` and
`criterion_met = null`; no single seed owns the three-seed verdict. Smoke, CUDA,
wrong-seed, dirty-source, or uncommitted-input artifacts are individually
`ineligible`. Cross-artifact incompatibility is not an individual property: the
aggregate preflight rejects the set without changing any manifest. Only the canonical
three-seed evaluation report records `criterion_met` or `criterion_not_met`.

Artifact schema v2 adds the pitch trainer and explicit policy-semantics identity.
Decoding is schema-first: duplicate-key/nonfinite-safe raw JSON is decoded just far
enough to read the schema ID, then dispatched through immutable v1 or v2 registries.
The registries separately own allowed trainer/profile/suite/architecture/
preprocessing identities, manifest fields, config/summary/evaluation codecs,
inventories, criterion states, and aggregate-report rules. Current Milestone D
constants and v1 bytes are not replaced with v2 globals.

Milestone D v1 BC/PPO manifests, configs, summaries, evaluation payloads, reports, and
traces remain strict-loadable and retain their original actor and shuffle semantics.
They are never silently upgraded, remasked, or made criterion-eligible for Milestone
E. Cross-version artifact aggregation is rejected before actor construction.

The schema/trainer matrix is closed:

| Codec family | Schema v1 | Schema v2 |
| --- | --- | --- |
| manifest/config/summary/evaluation | `bc`, `ppo` only | `pitch` only |
| aggregate report | exact Milestone D BC/PPO/baseline matrix | exact three-seed pitch/baseline matrix |
| run trace | one v1 BC or PPO artifact | one v2 pitch artifact |
| diagnostic report | accepted input schema recorded by separate diagnostic v1 codec | accepted input schema recorded by separate diagnostic v1 codec |

Existing `train-bc` and `train-ppo` always continue producing schema v1. Only
`train-pitch` produces schema v2. Schema v2 cannot contain BC/PPO trainer values, and
schema v1 cannot contain pitch. Fixed canonical byte fixtures for one complete v1 BC
artifact, one v1 PPO artifact, their config/summary/evaluation payloads, one aggregate
report, and one trace must load and re-encode byte-identically after v2 lands.

Every JSON payload remains duplicate-key rejecting, nonfinite rejecting, canonical,
and hash checked. Every Torch payload remains trusted-local only, weights-only where
supported, exact-type/exact-state validated, finite, and CPU-loaded for authoritative
evaluation.

Before final E-suite or direct-coordinate access, checkpoint `evaluate` requires
exactly three distinct complete schema-v2 pitch artifacts with seed set `{0, 1, 2}`,
checkpoint profile, individual `eligible_for_aggregate` status, CPU training/internal
evaluation, identical source commit/lock/required-input status, and one exact derived
compatibility digest. It strict-validates every
inventory, JSON payload, and model before constructing any actor, cache, environment,
or final-suite object. Duplicate, missing, extra, ineligible, or incompatible inputs
are closed-contract errors. Exploratory/ineligible checkpoint triples cannot access
the final suites; smoke evaluation remains single-artifact and smoke-suite only.

Final-suite pitch diagnostics use the same exact-three preflight before actors or
suites are constructed. Legacy v1 diagnostics use only their already-observed v1
suites and do not participate in E criterion computation.

## CLI and hands-on workflow

The existing entry point gains two commands without changing existing syntax:

```text
uv run harpy-sine-learn train-pitch --profile smoke --seed 0 \
  --output runs/milestone-e-pitch-smoke

uv run harpy-sine-learn diagnose runs/milestone-e-pitch-smoke \
  --suite smoke --output runs/milestone-e-pitch-smoke-diagnostics.json

uv run harpy-sine-learn evaluate runs/milestone-e-pitch-smoke \
  --output runs/milestone-e-pitch-smoke-report.json

uv run harpy-sine-learn run runs/milestone-e-pitch-smoke --seed 123
```

Checkpoint training uses seeds 0, 1, and 2 in three explicit create-only output
directories. `evaluate` accepts the compatible three-artifact set and emits canonical
per-seed, aggregate, baseline, and probe rows. `run` retains the existing actor-neutral
trace contract and chosen symbolic actions. Estimator values live in the typed
diagnostic report; the narrow `Actor` protocol is not weakened with mutable
`last_estimate` state merely for presentation.

The checkpoint report's terminal matrix is exact: five pitch rows per seed in seed
order (`iid`, `ood_lower`, `ood_upper`, `iid_zero`, `iid_shuffled`) followed by three
rows for each baseline in stable Random, Reward Search, Spectrum Peak, Oracle order
(`iid`, `ood_lower`, `ood_upper`). That is 15 learned plus 12 baseline terminal rows.
Direct-coordinate records and the typed two-register aggregate are separate sections,
not synthetic terminal rows. The smoke report contains one pitch base row, two pitch
probe rows, and one smoke row for each of the four baselines, in that order.

`--help` and all nontraining imports remain Torch/SB3-lazy and Qt-free. Existing exit
codes, path preflight, incomplete-artifact rejection, trusted-local warning, and
create-only report behavior remain authoritative.

The exact diagnostic grammar is:

```text
harpy-sine-learn diagnose ARTIFACT [ARTIFACT ...]
  --suite {smoke,iid,ood-lower,ood-upper}
  --output FILE
  [--device {cpu,cuda}]
  [--bound-mask]
```

Device defaults to CPU. Output is required, create-only, outside every input artifact,
and its canonical JSON plus newline is also written byte-identically to stdout;
trusted-local warnings use stderr. Suite aliases resolve through the input schema's
immutable registry. A single pitch artifact may diagnose only `smoke`; E final aliases
require the exact eligible three-artifact preflight. `--bound-mask` is accepted only
with exactly one schema-v1 BC artifact and is rejected before model construction
otherwise. Explicit unavailable CUDA is an execution failure and diagnostic CUDA
results are never criterion evidence.

CLI statuses remain `0` for successful completion including `criterion_not_met`, `2`
for argument/path/closed-contract errors (including wrong seed sets, unsupported
schema/trainer combinations, or invalid mask usage), `1` for missing dependency,
artifact I/O/integrity, model, training, evaluation, or requested-device execution
failure, and `130` for a handled user interrupt. Preflight errors occur before output
creation or expensive work.

## Evaluation and preregistered criteria

### Engineering gates

- The complete registered Gym action, observation, reward, rendering, and terminal
  contract is unchanged.
- Legacy Milestone D artifacts strict-load and reproduce their original actor actions.
- Every pitch artifact reloads from persisted bytes before internal evaluation.
- Same artifact and episode seed produce byte-identical canonical traces.
- The actor receives no hidden pitch/error/suite/cache identity.
- Every estimator-planner decision is a legal, non-bound-blocked action for its public
  control state, independent of model quality.
- Submit is never masked.
- Zero and shuffled probes alter only spectrum evidence.
- Optional training imports remain lazy and Qt-free.
- Artifact completion, interruption, corruption, inventory, and path boundaries retain
  the existing atomic guarantees.

### Scientific criterion

Only clean, committed, CPU-trained and CPU-evaluated checkpoint artifacts for exact
seeds 0, 1, and 2 are eligible. All three must share the exact source commit, lock, and
seed-neutral compatibility digest; their declared run seeds must be exactly distinct.
Smoke and CUDA artifacts remain useful but ineligible.

The Milestone E criterion is met only when all of the following hold:

- each seed estimates at least 99% of all 401 untouched IID coordinates within five
  cents;
- each seed estimates at least 95% of all 400 untouched OOD coordinates within five
  cents, with neither register below 90%;
- each seed achieves at least 95% submitted success on the new 250-episode IID suite;
- every seed has exactly zero IID bound-blocked actions and zero IID truncations;
- each seed's IID submitted success exceeds both its zero-spectrum and shuffled-
  spectrum success by at least 50 percentage points;
- median combined register-OOD submitted success is at least 90%, and no seed is below
  80%;
- mean successful excess actions on IID is at most 4.0 for every seed. A perfect
  nearest-five-cent estimator averages about 2.836 excess actions on this suite because
  its zero-tolerance plan is compared with the environment's tolerance-five optimum;
  the threshold intentionally permits that reliability tradeoff.

The report includes every ordered `PitchCoordinateRecord`, re-derived within-one/
within-five rates, mean error, nearest-rank p50/p90/p95/p99/max cent errors, residue and
register tables, terminal reasons, final cent error, action counts, successful excess,
probe results, and matched Random, Reward Search, Spectrum Peak, and Oracle rows.

Failure to meet the scientific criterion still completes valid artifacts and reports
`criterion_not_met`; it is not an engineering failure.

## Follow-on experiments

### DAgger direct-action recovery

If direct-action learning remains valuable after diagnostics, a later checkpoint may
roll a newly versioned masked BC actor on deterministic training-only episodes, label
visited states with the evaluator's set of shortest legal actions, deduplicate states,
cap each aggregation round, and retrain from a fresh initialization. Final suites and
their reachable-state coordinates are excluded. DAgger must compare teacher-only
masked BC against aggregated masked BC and must not reuse a selected final model.

### Masked PPO

A later PPO checkpoint may introduce a custom versioned SB3 policy whose mask affects
sampling, evaluation, log probability, and entropy. It starts from fresh weights,
retains the current reward and environment, uses the same fixed final suites, and is a
comparison rather than a blocker for the reliable estimator-planner checkpoint.

Neither follow-on is authorized by this specification.

## Required test and error matrix

Implementation planning must preserve test-first RED/GREEN boundaries for at least:

- independently reconstructed golden coordinate-split and four suite digests, exact
  counts/order/pools/exclusions, source-disjointness, and training spies that fail if
  any final coordinate or suite is accessed;
- exhaustive legal masks at every Octave, Semitone, and Cent bound, Submit visibility,
  exact action order, and no evaluator truth in mask inputs;
- exact model topology/parameter inventory, location preservation, label boundaries,
  first-maximum ties, finite/wrong-shape failures, deterministic training seed/order/
  patience/checkpoint restoration, and direct-record recomputation;
- raw observation key/dtype/shape/range allowlists; spectrum-only estimator inputs;
  re-estimation after every action; correct target/control arithmetic, clamping,
  tolerance-zero planning, and no fallback on malformed output;
- crafted diagnostic episodes covering commuting shortest actions, canonical mismatch,
  genuine divergence, early/late Submit, bound blocks, repeated states, loops,
  recoverable/unrecoverable budgets, no-event nulls, zero-denominator metrics, and
  canonical survival curves;
- frozen canonical v1 BC/PPO bytes for every codec family; schema-first v1/v2 dispatch;
  rejection of every cross-schema/trainer/profile/suite combination; exhaustive
  trainer dispatch with no `else PPO` fallthrough;
- pitch artifact bootstrap, every publication/interruption boundary, exact six-file
  inventory, corrupt/duplicate/nonfinite JSON, wrong or nonfinite model state, hashes,
  strict pending/completion behavior, persisted reload before smoke evaluation, and
  no fallible work after the completion rename;
- exact three-seed preflight before actor/model/cache/final-suite construction,
  duplicate/missing/extra/ineligible/incompatible sets, final direct and terminal row
  persistence, canonical report re-decode/re-encode, criterion recomputation, and OOD
  two-digest aggregation;
- exact CLI grammar, lazy help/imports, path preflight, stdout/stderr separation,
  create-only outputs, trusted-local warning, missing dependencies, unavailable CUDA,
  stable `0/1/2/130` status mapping, and interruption cleanup;
- one real CPU smoke train/save/reload/diagnose/evaluate/run workflow, repeated loaded
  actions and byte-identical traces, followed by the normal full suite, Ruff, format,
  diff, import, and no-skip gates.

Mutation checks must physically break the grid-label rule, spatial architecture,
planner re-estimation, action mask, v1 schema dispatch, three-seed preflight, direct-
record criterion, and atomic completion, then demonstrate the focused tests fail
before restoration.

## Completion boundary

Milestone E is complete when:

1. the approved implementation plan has been executed with test-first boundaries;
2. all engineering gates pass on a clean normal-ancestry branch;
3. three eligible checkpoint artifacts and the canonical report exist and strict-load;
4. the scientific criterion is recomputed from persisted direct-coordinate and
   terminal records;
5. one deterministic hands-on trace is captured twice and compares byte-identically;
6. an acceptance document records exact source, lock, runtime, artifact hashes,
   metrics, warnings, and criterion outcome;
7. independent environment/leakage, scientific, and artifact/CLI reviews are clear;
8. no hidden training process remains and the tracked worktree is clean.
