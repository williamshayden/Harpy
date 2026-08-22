# Harpy Milestone D: Learned Sine Policy Design

Date: 2026-08-10

Status: Approved. The design decisions and this consolidated written specification
were approved by the user for implementation planning on 2026-08-10.

## Outcome

Milestone D produces Harpy's first learned policy on the frozen
`Harpy/SinePitch-v0` task. It first trains a supervised behavior-cloning diagnostic
from deterministic oracle trajectories, then trains Proximal Policy Optimization
(PPO) from a fresh initialization. Both policies are evaluated through the real
environment rather than by action accuracy or training return alone.

This milestone answers a deliberately narrow question: can a compact learned policy
combine the existing candidate spectrum, symbolic target note, independent Octave,
Semitone, and Cent control state, and remaining budget well enough to complete the
single-sine tuning task? It does not change the task to make that result easier and it
does not yet add explicit analysis tools.

The intended user experience is equally narrow: install the optional training group,
run one command to train, one to evaluate, and one to watch a trained actor's complete
action trace.

## Scientific boundary

The default actor receives exactly the current `Harpy/SinePitch-v0` spectrum
observation. The spectrum is audio-derived evidence for the current candidate, not an
environment-supplied pitch estimate. The actor also receives the public target-note
index, independent control values, and remaining step count already defined by
Milestone C.

The actor never receives source pitch, candidate pitch, target pitch in cents or Hz,
signed or absolute error, an FFT peak coordinate, an oracle action, the target
spectrum, terminal `EpisodeResult`, suite metadata, or cache keys. Evaluator truth is
read only after termination or truncation.

Milestone D must preserve these distinctions:

- Behavior cloning is a supervised representation-and-control diagnostic. Its oracle
  labels are training truth, so a successful BC model is not an RL result.
- PPO is the first learned reinforcement-learning result. It starts from a new random
  initialization and does not reuse BC weights, optimizer state, or logits.
- Spectrum Peak is a strong classical baseline for a noiseless sine. PPO is not
  required to beat it in this milestone.
- Oracle validates reachability and planning. It is not a perception result.
- Random establishes a paired lower control on the same episodes.

Candidate spectra remain continuously visible because that is the frozen v0
observation contract. On-demand `Analyze Source` and `Analyze Candidate` actions are
deferred to a separately versioned `SinePitchTools-v0` environment. Target-spectrum
matching, reference-audio access, and model-defined tool calls are also deferred.

The milestone measures a spectrum-conditioned policy on the dense, clean
representation of a procedural sine. Spectrum perturbations measure policy sensitivity
only; without a separately trained scalar-only learned control they do not establish
that spectral evidence was causally necessary. The result must not be presented as
evidence that Harpy can yet tune immutable recorded audio, chords, percussion, noisy
sources, or arbitrary synth patches.

## Scope

Milestone D includes:

- an optional training dependency group containing PyTorch and Stable-Baselines3;
- a new Qt-free `harpy.learning` package;
- deterministic, versioned train, validation, IID evaluation, and register-OOD
  episode definitions;
- a learning-only, semantics-preserving lazy spectrum cache;
- one shared compact spectrum-and-state feature architecture;
- deterministic oracle-trajectory generation and weighted behavior cloning;
- an SB3 PPO adapter trained from a fresh initialization;
- a narrow actor protocol and terminal evaluator shared by learned policies;
- Random, Reward Search, Spectrum Peak, and Oracle comparison rows on matched
  episodes;
- smoke and checkpoint training profiles;
- versioned, hash-checked local artifacts with explicit incomplete/complete lifecycle;
- `train-bc`, `train-ppo`, `evaluate`, and `run` CLI commands;
- automated leakage, determinism, cache, model, artifact, CLI, and smoke-training
  tests;
- an actual five-seed PPO checkpoint report and one BC checkpoint report.

Milestone D does not include:

- any change to the seven actions, rewards, bounds, observation fields, step budget,
  episode result, or registration of `Harpy/SinePitch-v0`;
- BC-to-PPO warm starting, offline RL, DQN, recurrent policies, transformers, or a
  generic trainer abstraction;
- tool-use actions, source/target analysis calls, hidden scratch state, or an LLM
  adapter;
- waveform observations, target audio, target spectra, uploaded samples, or a real
  pitch-shift backend;
- oscillator variation, polyphony, chords, modes, percussion, MIDI, or synth control;
- GUI integration, live playback, browser UI, hosted training, model serving,
  OpenRouter, a database, experiment tracking service, or machine telemetry;
- hyperparameter search against the final IID or OOD suites;
- a promise of bit-identical learned weights across machines or PyTorch releases.

## Dependency and package boundary

The normal Harpy installation remains lightweight. The optional `train` group pins
`stable-baselines3==2.9.0` and a compatible `torch>=2.8,<3` release; the lockfile pins
the exact resolved versions. Training instructions use:

```text
uv sync --group train
```

Importing Harpy's synth, analysis, GUI, or environment packages must not import
PyTorch or Stable-Baselines3. Importing `harpy.learning` itself must also remain safe
without the optional group: dependency-heavy modules load lazily at the CLI or trainer
boundary and produce the exact installation instruction above when unavailable.

All learning code remains independent of Qt and live audio:

```text
harpy-sine-learn
      |
      v
harpy.learning
      |-- suites and learning-only evidence cache
      |-- feature contract and actor adapters
      |-- BC and PPO training
      |-- terminal evaluator and artifacts
      |
      +--> harpy.envs.SinePitchEnv / EpisodeResult / planning
      +--> harpy.envs spectrum encoder and fixed grid
      +--> PyTorch and Stable-Baselines3  [optional train group]
```

The package is divided by responsibility rather than by framework ceremony:

- `harpy.learning.models`: immutable profile, suite, artifact, and evaluation models;
- `harpy.learning.suites`: deterministic episode construction and split validation;
- `harpy.learning.cache`: the bounded spectrum evidence provider;
- `harpy.learning.features`: normalization and the shared compact network;
- `harpy.learning.actors`: the narrow inference protocol and BC/PPO adapters;
- `harpy.learning.bc`: oracle examples and behavior-cloning training;
- `harpy.learning.ppo`: SB3 environment adapter and PPO training;
- `harpy.learning.evaluation`: matched terminal rollout evaluation;
- `harpy.learning.artifacts`: atomic local persistence, hashes, and loading;
- `harpy.learning.cli`: the four user commands.

These modules may be combined where the implementation is clearer; their
responsibilities and dependency direction must remain distinct. Milestone D does not
create a registry of trainers, generic experiment platform, storage service, or model
server.

## Frozen environment contract

Learned-policy training and learned-actor evaluation consume only
`Harpy/SinePitch-v0`. Comparison baselines may retain their declared
capability-specific environment IDs at the evaluator boundary. The following
Milestone C semantics remain authoritative:

- action IDs are the existing seven `PitchAction` values;
- Octave, Semitone, and Cent controls remain independent and bounded;
- blocked bound actions consume a step and receive the existing penalty;
- Submit is explicit, and success requires submission within 5 cents;
- the total budget is 64 actions;
- reward values and `gamma=1.0` interpretation are unchanged;
- observations retain the exact keys, dtypes, shapes, and ranges;
- candidate rendering uses the current default patch, tuning, analysis grid, and fresh
  deterministic render semantics;
- evaluator metrics come from the existing validated `EpisodeResult`.

Learning code must not fork or reproduce the transition/reward implementation. It
drives the real environment. The registered default environment continues to use the
direct numerical renderer, so normal Gym behavior is independent of the learning
cache.

## Episode distributions and suites

### Training distribution

The learned-policy training distribution is versioned as
`harpy-sine-policy-train-v1`:

- target indices cover all 25 notes from C2 through C4;
- source pitch is sampled from integer cents `5000..7000` inclusive;
- source/target pairs with initial absolute error at or below 5 cents are excluded;
- target scheduling is balanced in deterministic 25-episode permutation blocks;
- eligible source cents are sampled uniformly conditional on the target;
- fixed final-evaluation pairs are excluded from training;
- each logical episode is derived from a non-negative run seed and absolute episode
  index with NumPy `SeedSequence`/`PCG64`, never Python hash or module-global RNG.

BC training and BC validation use distinct deterministic index namespaces and sample
their finite pair sets without replacement. Their episode pairs cannot overlap each
other or either final evaluation suite. PPO uses the same allowed source/target pool
with its own run-seed namespace and may revisit training pairs; it still excludes the
fixed final-evaluation pairs. PPO experience is not materialized as a persistent
dataset.

Profile-specific training and validation episode counts are checked-in configuration,
recorded in every artifact, and never inferred from final evaluation performance.

### Fixed IID evaluation

`harpy-sine-policy-eval-iid-v1` contains exactly 256 immutable episodes:

- every source lies in `5000..7000`;
- every target appears at least ten times and no target count differs by more than one;
- eligible source values are uniform conditional on their scheduled target;
- every initial error is greater than 5 cents;
- no pair appears in BC training, BC validation, or another fixed suite.

Its fixed suite seed is `202608101`. Every pair within the suite is unique.

### Fixed register-OOD evaluation

`harpy-sine-policy-eval-register-ood-v1` contains exactly 256 immutable episodes:

- 128 sources lie in the lower band `4800..4999`;
- 128 sources lie in the upper band `7001..7200`;
- all 25 targets are covered with counts differing by at most one;
- for each target, lower- and upper-band episode counts differ by at most one;
- every initial error is greater than 5 cents;
- no pair appears in another fixed suite.

Its fixed suite seed is `202608102`. Every pair within the suite is unique.

This suite evaluates held-out initial-source bands and reports lower, upper, and
combined results. It is not an unseen-spectrum test: central-source training
trajectories can visit some of the same effective candidate frequencies after control
actions. It is also not a causal register-isolation test because error and direction
distributions change with the initial band. The suite is informational and cannot be
used to choose epochs, early stopping, architecture, optimizer settings, PPO
hyperparameters, or the preferred training seed.

### Fixed smoke evaluation

`harpy-sine-policy-eval-smoke-v1` contains exactly 32 immutable, unique episodes. Its
fixed suite seed is `202608100`; sources lie in `5000..7000`, every initial error is
greater than 5 cents, all 25 targets appear at least once, and the remaining seven
targets follow the next deterministic permutation positions. It is disjoint from BC
smoke training/validation and both final suites. Its sole artifact filename is
`evaluation-smoke.json`. This suite verifies execution only and is never a scientific
result.

### Suite construction and identity

Targets are assigned by deterministic shuffled permutation blocks. Each source is
then drawn from the sorted target-eligible source set for that suite. OOD construction
uses a deterministic balanced lower/upper schedule before its within-band draw.

Suite schema version, suite ID, fixed suite seed, episode index, target index, and
source coordinate fully determine an episode. The implementation exposes immutable
`EpisodeSpec` values containing only the injection values needed by `env.reset`.
Golden suite digests pin order and membership. Runtime manifests record those digests,
not a mutable filename or implicit random-generator state.

The evaluator presents the same ordered episode list to every compatible actor and
baseline. Pairwise comparisons are therefore episode matched.

## Learning-only spectrum cache

The v0 spectrum is a pure function of effective integer pitch cents because patch,
tuning, phase origin, sample rate, render length, FFT configuration, and frequency
grid are frozen. Milestone D may exploit that fact without changing the registered
environment.

A learning-only evidence provider caches spectra by effective integer pitch cents:

- valid keys are `1100..10900` inclusive, the complete reachable v0 range;
- there are at most 9,801 entries;
- every value is a contiguous, immutable float32 array of shape `(1961,)`;
- a miss invokes the exact existing fresh synth-render and spectrum-encoding path;
- only the spectrum is retained; rendered audio is discarded;
- a hit never exposes the cache's owned array for actor mutation;
- observations remain freshly owned and mutable to the same degree as the direct env;
- cache state never enters observations, rewards, info, suite identity, or artifacts;
- the full cache requires about 77 MB (73.3 MiB) plus bounded mapping overhead;
- invalid keys, wrong shapes, non-finite values, or mutated entries fail immediately.

The learning factory constructs a non-registered cached environment through a narrow
internal evidence seam. `gym.make("Harpy/SinePitch-v0")` remains direct-rendered.
The registered base environment retains its current `_candidate_audio` ownership,
immutability, rollback, and fresh-render behavior so existing Milestone C
characterization tests remain unchanged. A protected internal evidence-result seam
allows only the non-registered learning subclass to omit retained audio after both
cache misses and hits and keep the identical spectrum. The subclass explicitly relaxes
that private audio storage invariant without changing any public observation or
transition invariant.
The environment constructor and registered kwargs do not gain an evidence-provider
parameter.

Training-distribution injection is likewise external to the registered contract. A
learning-only Gym wrapper selects the next deterministic `EpisodeSpec` and delegates
to the real environment's existing complete `reset(options={target_note_index,
source_pitch_cents})` path. It does not implement reset, transition, reward, or terminal
semantics itself.

Identity tests compare direct misses, first cached misses, repeated hits, inverse-action
returns, and action-order-equivalent states bit for bit. Cache use must not change
observation bytes, rewards, info, terminal flags, `EpisodeResult`, action count, or RNG
state. If this equivalence cannot be demonstrated, the cache is disabled rather than
relaxing environment assertions.

## Shared observation preprocessing

Both BC and PPO use the same checked-in preprocessing contract. The 1,961-bin spectrum
remains float32 in its existing `[0, 1]` range and is presented as one 1D channel.

The other observation fields become five float32 scalars in this exact order:

1. target note: `(target_note - 12) / 12`, producing `[-1, 1]`;
2. octave control: `octaves / 2`;
3. semitone control: `semitones / 12`;
4. cent control: `cents / 100`;
5. remaining budget: `steps_remaining / 64`, producing `[0, 1]`.

Preprocessing validates the original Gym keys, scalar/array types, dtypes, shapes,
bounds, and finiteness before conversion. It does not accept evaluator-only aliases or
silently clamp malformed observations.

Shuffled-spectrum and zero-spectrum probes use the same scalar features. They are
reported as sensitivity diagnostics only. They do not establish causal use of spectral
evidence and do not replace the separately labeled Reward Search control.

## Shared compact network

The BC classifier and PPO policy share one versioned feature architecture:

- spectrum branch: `Conv1d(1, 16, kernel_size=9, stride=4)`, ReLU,
  `Conv1d(16, 32, kernel_size=7, stride=4)`, ReLU, then adaptive average pooling to
  16 positions;
- scalar branch: Linear `5 -> 32`, ReLU, Linear `32 -> 32`, ReLU;
- combined trunk: concatenate both branches, Linear `544 -> 128`, ReLU;
- policy output: Linear `128 -> 7` unnormalized action logits.

PPO adds its required scalar value head from the same 128-feature representation. BC
has no value head. No recurrence, attention, batch normalization, observation running
statistics, action masking, or pretrained weights are used.

Weights use framework-standard seeded initialization. The manifest records the
architecture schema, exact layer configuration, activation, preprocessing schema,
parameter count, and framework versions. Loading rejects an architecture or parameter
shape mismatch rather than partially applying a state dictionary.

## Behavior-cloning diagnostic

BC examples come from deterministic `minimum_action_plan` trajectories over the BC
training and validation splits. Each example contains the actor-visible observation
at one real environment state and the next oracle action. Submit is included as the
last label. Hidden source/error values may generate labels but never enter model
features or serialized actor input.

The dataset stores compact episode/state coordinates and labels rather than duplicate
1,961-float spectra. Batches resolve spectra through the bounded evidence provider.
Training and validation ordering are deterministic under the run seed.

BC uses weighted cross-entropy. For each action class, its weight is
`N / (7 * class_count)` using training examples only; all seven classes must be
present. Validation labels do not affect weights. The optimizer, batch size, maximum
epochs, patience, and learning rate are part of the checked-in profile and artifact.
Early stopping selects the earliest epoch with the lowest internal validation loss,
with deterministic next-action accuracy reported alongside it. A strictly lower loss
resets patience; ties retain the earlier checkpoint.

Validation accuracy is an internal model-selection diagnostic, not the held-out BC
gate. After the epoch is selected, held-out next-action accuracy is computed exactly
once over oracle trajectories for the fixed IID suite. Those oracle labels do not
affect weights, class weights, epoch selection, architecture, configuration, or
hyperparameters. Closed-loop IID rollout uses the same fixed episodes but requires the
policy to generate its own complete trajectories.

The final BC claim is based on closed-loop rollouts through the real environment. A
high held-out action accuracy without successful rollouts is reported as a failed
diagnostic, not rounded up to success.

## PPO checkpoint

PPO uses Stable-Baselines3 2.9 with `MultiInputPolicy` and the shared Harpy feature
extractor. Its policy and optimizer are freshly initialized for every run. No BC model,
BC feature weights, BC logits, demonstration buffer, or oracle action enters PPO.

A Harpy policy-observation wrapper first validates the raw frozen Gym observation and
converts it into exactly two float32 `Box` entries: `spectrum` with shape `(1961,)` and
`state` with the five normalized scalars declared above. This occurs before SB3's
preprocessor, so SB3 never one-hot encodes the raw `Discrete` target or remaining-step
spaces. BC calls the same wrapper directly.

PPO `policy_kwargs` pin Harpy's custom 128-dimensional feature extractor,
`net_arch=[]`, and `share_features_extractor=True`. SB3 therefore attaches its direct
seven-action and scalar-value heads to the declared shared 128-vector rather than
inserting the default separate 64-by-64 policy/value MLPs.

Harpy budget exhaustion is a true finite terminal for learning even though the public
Gym API correctly reports it as `truncated=True`. SB3 normally treats a truncation as
a time limit and adds `gamma * V(terminal_observation)` to the collected reward. A
Harpy-specific VecEnv adapter suppresses that timeout bootstrap only for validated
`BUDGET_EXHAUSTED` transitions while preserving the underlying Gym observation,
reward, `terminated=False`, `truncated=True`, info, and `EpisodeResult` contract.
Submit termination remains unchanged. A forced 64-step regression must show that the
rollout-buffer reward exactly equals Harpy's returned reward with no terminal-value
addition.

The invariant settings are:

- `gamma=1.0`, matching the undiscounted Milestone C shaping contract;
- one synchronous CPU environment by default;
- `device="cpu"` unless the user explicitly requests an available CUDA device;
- deterministic run, environment, suite, NumPy, Python, Torch, and SB3 seeding;
- deterministic inference for all final evaluation;
- no `VecNormalize`, reward normalization, observation normalization, action mask, or
  callback that reads evaluator truth;
- no hyperparameter adjustment from IID/OOD final results.

The checkpoint profile initially pins 256,000 environment steps, PPO rollout length
1,024, batch size 256, ten optimization epochs, learning rate `3e-4`,
`gae_lambda=0.95`, clipping range `0.2`, entropy coefficient `0.01`, and value
coefficient `0.5`. These values may be changed only by revising the design/config
version before a result is run, never by silently editing an artifact or selecting on
final-suite performance.

The declared BC checkpoint seed is `0`. The declared PPO checkpoint training seeds are
`0, 1, 2, 3, 4`. Each seed produces a separate artifact and result row. Aggregate PPO
success is computed across all five; a single best seed is not the headline result.
The CLI permits other non-negative seeds for explicit exploration, but marks those
checkpoint-profile artifacts `criterion_eligible=false`; they cannot replace or join
the declared result set. A BC checkpoint seed other than 0 is treated the same way.

## Training profiles

Two checked-in profiles keep development fast without confusing smoke evidence with a
checkpoint:

| Setting | `smoke` | `checkpoint` |
| --- | ---: | ---: |
| BC training episodes | 128 | 4,096 |
| BC validation episodes | 64 | 512 |
| BC maximum epochs | 2 | 50 |
| BC early-stop patience | disabled | 5 |
| BC batch size | 128 | 256 |
| BC optimizer | AdamW | AdamW |
| BC learning rate | `3e-4` | `3e-4` |
| BC weight decay | `1e-4` | `1e-4` |
| PPO environment steps | 2,048 | 256,000 |
| PPO rollout length | 256 | 1,024 |
| PPO batch size | 64 | 256 |
| Final evaluation | tiny smoke suite | full IID + OOD suites |

Smoke artifacts are marked `profile="smoke"` and are never eligible for scientific
criteria. The checkpoint profile is the only profile reported as Milestone D evidence.
All PPO settings not varied in this table use the invariant values declared in the PPO
section for both profiles, including ten optimization epochs, `3e-4` learning rate,
`gae_lambda=0.95`, `0.2` clipping, `0.01` entropy coefficient, and `0.5` value
coefficient. “Tiny smoke suite” means exactly
`harpy-sine-policy-eval-smoke-v1`.

## Actor and rollout boundary

Learned inference adapters implement one narrow, stateless actor operation: accept one
validated actor-safe observation and return one `PitchAction`. BC selects the first
maximum logit. PPO calls SB3 prediction with `deterministic=True`. Ties therefore have
a stable first-index behavior.

The generic rollout loop owns environment reset/step and passes only the current
observation to learned actors. It may record actor-safe reward and info for the trace,
but learned inference does not receive a hidden prior-reward input. PPO's trainer sees
rewards only through the normal transition interface.

After termination or truncation, the evaluator reads the validated `EpisodeResult`.
No reference to that object, its fields, or an equivalent derived value is reachable
from the actor before its final action.

Existing nonlearned baselines keep their capability-specific interfaces. The learning
package adapts them at the evaluator boundary rather than broadening the learned actor
API to expose reward-only or oracle capabilities.

## Evaluation protocol

Every checkpoint actor is evaluated in deterministic inference mode on the ordered
256-episode IID suite and the ordered 256-episode register-OOD suite. The comparison
matrix contains separate rows for:

- the BC checkpoint;
- PPO seed 0;
- PPO seed 1;
- PPO seed 2;
- PPO seed 3;
- PPO seed 4;
- Random;
- Reward Search;
- Spectrum Peak;
- Oracle.

Random uses seed-scheme version `harpy-baseline-policy-seed-v1`: for each fixed episode
its RNG is `default_rng(SeedSequence([suite_seed, 1, episode_index]))`, where code `1`
is the existing checked-in Random policy code. Reward Search, Spectrum Peak, and Oracle
use their existing declared evidence lanes and planning logic. They share source/target
episode pairs with learned actors but retain distinct environment IDs where capability
isolation requires it. Rows are never pooled across observation modes, and Reward
Search is reported as a scalar-feedback control rather than a learned-policy gate.

For each suite and actor, evaluation records:

- submitted success rate within 5 cents;
- submitted success rate within 1 cent;
- final within-5 and within-1 rates whether or not submitted;
- mean and median absolute final error in cents;
- mean actions;
- mean successful excess actions;
- mean return;
- truncation rate;
- invalid-action rate;
- model parameter count, training environment steps, training examples, and training
  wall time where applicable.

Register-OOD aggregates are additionally partitioned into lower- and upper-source-band
rows before the combined row.

The evaluation JSON includes aggregate metrics plus the terminal metrics needed for
paired episode comparisons. Hidden truth is an evaluator artifact, never actor input.
Rates use exact episode-count denominators, absent successful-excess values serialize
as `null`, and JSON never emits NaN or Infinity. Invalid-action rate is the total
blocked-action count divided by total action count, not by episode count; its zero
denominator is impossible because every terminal episode has at least one action.

The final report includes shuffled-spectrum and zero-spectrum rollouts for BC and PPO.
Those probes are diagnostic and do not replace the main IID result.

## Milestone success criteria

Engineering acceptance and model performance are separate.

The BC diagnostic meets its scientific criterion only if both are true:

- held-out next-action accuracy is at least 90%; and
- deterministic IID closed-loop submitted success is at least 75%.

The PPO checkpoint meets its scientific criterion only if both are true:

- median IID submitted success across the five declared seeds is at least 50%; and
- at least four of five PPO seeds have strictly higher paired IID submitted success
  than Random.

Only clean-source, CPU-trained, checkpoint-profile artifacts with the declared seeds
are included in either scientific criterion.

OOD is reported but is not a gate. PPO is not required to beat Spectrum Peak. BC and
PPO criteria are reported independently; BC success cannot substitute for PPO failure.

If the environment, training, artifact, and evaluation pipeline is valid but a model
misses its criterion, the report states `criterion_not_met` with the actual numbers.
The team does not tune against the final suites, suppress weak seeds, or redefine the
threshold after seeing the run. A failed criterion guides a new explicitly versioned
milestone or experiment.

## Hands-on command surface

The single executable is `harpy-sine-learn`:

```text
uv run harpy-sine-learn train-bc \
  --profile smoke --seed 0 --output runs/bc-smoke

uv run harpy-sine-learn train-ppo \
  --profile smoke --seed 0 --output runs/ppo-smoke

uv run harpy-sine-learn evaluate runs/ppo-smoke

uv run harpy-sine-learn run runs/ppo-smoke --seed 123
```

Checkpoint PPO is intentionally one artifact per command and seed. The five-run report
uses seeds `0..4`; `evaluate` accepts one or more artifact directories so those rows
can be produced together without inventing a run database.

`evaluate` never mutates a completed training artifact. By default it writes one
strict JSON result to stdout. An optional `--output FILE` writes the same bytes through
an atomic create-only operation and refuses an existing file. That output must not
equal or be located inside any input artifact directory. Multiple artifacts must share
compatible environment, suite, grid, preprocessing, policy architecture, and profile
versions; the PPO-only value head is not a policy-architecture mismatch with BC.
Duplicate paths and duplicate `(trainer, seed)` rows are rejected. Output order is
canonical by trainer then seed, independent of argument order. Checkpoint evaluation
adds Random, Reward Search, Spectrum Peak, and Oracle once. It computes the aggregate
PPO criterion only for the exact CPU-trained seed set `{0, 1, 2, 3, 4}`, with no
additional PPO seed, and accepts at most the declared BC seed 0 in that report.

All train commands require an explicit output directory that does not exist. They
refuse an existing path, including an empty directory. `--device cpu` is the default;
`--device cuda` is accepted only when the installed Torch build reports CUDA
availability.

Missing parent directories such as the example `runs/` are created recursively after
validating the nearest existing ancestor. The final artifact directory itself is still
created exclusively. Evaluation output follows the same parent rule.

`run` executes one complete episode and prints the target note, each selected action,
each returned scalar reward, terminal reason, final cents error, submitted success,
action count, excess actions when defined, and total return. Evaluator-only pitch/error
truth appears only after the episode finishes. `--json` emits one strict JSON object
instead of the human trace. Given the same complete artifact, suite/environment
version, dependency lock, and seed, action and result JSON must repeat on the same
supported stack.

By default, `run --seed N` calls the frozen full-range
`Harpy/SinePitch-v0.reset(seed=N)` distribution (`4800..7200` source cents). This makes
changing the seed a simple hands-on way to see central and edge-register episodes. The
trace labels that distribution explicitly; it is a demonstration, not an IID-suite
score.

## Artifact contract

Each training command owns one newly created directory. A complete artifact contains:

```text
<output>/
  manifest.json
  training-config.json
  training-summary.json
  model.pt                 # BC, state_dict only
  model.zip                # PPO, SB3 native archive
  evaluation-iid.json      # checkpoint profile
  evaluation-ood.json      # checkpoint profile
```

Only the model file appropriate to the trainer is present. Smoke evaluation uses an
explicitly named smoke-evaluation file rather than impersonating the final suites.

`manifest.json` is schema version 1 and records at least:

- lifecycle status, trainer kind, profile, seed, and creation/completion timestamps;
- Harpy commit, dirty-tree flag, tracked-diff digest, Python/platform, dependency lock
  identity, SB3/Torch, and CPU/CUDA device details;
- environment ID and frozen environment contract version;
- suite IDs and digests;
- spectrum grid and preprocessing schema versions;
- network architecture and exact parameter count;
- checked-in training configuration and total completed steps/examples;
- relative artifact filenames, byte sizes, and SHA-256 hashes;
- per-artifact evaluation eligibility and results. The BC artifact may record its
  single-model criterion only for checkpoint profile, seed 0, trained and evaluated
  on CPU; the aggregate PPO criterion belongs only to a compatible five-artifact
  CPU evaluation report. CUDA artifacts remain diagnostic and exploratory.

Before creating output, checkpoint training records source status. Scientific
`criterion_eligible=true` additionally requires a clean worktree at the recorded
commit, with the trainer configuration, suite generator and golden digests, and
dependency lockfile all committed there. A dirty checkpoint run remains permitted for
hands-on exploration but is marked ineligible and cannot join the declared BC or
five-seed PPO result. Milestone D does not archive arbitrary dirty or untracked source
as a substitute for a clean commit.

The output directory is created exclusively. An atomic manifest with
`status="incomplete"` is written first. Every subsequent JSON/model file is written to
a sibling temporary file, flushed, and atomically renamed. File hashes are computed
after final rename. The manifest switches to `status="complete"` only after all
required files and hashes exist and validate.

The manifest hashes every required file except itself; self-hashing is deliberately
not part of schema v1. The trainer/profile determines a closed final file inventory.
A complete artifact contains no temporary or unknown regular file, and the loader
rejects both missing required files and unexpected files.

The implementation masks handled interrupts across the tiny bootstrap section that
creates the final directory and atomically publishes the first incomplete manifest. A
handled failure during bootstrap removes the newly created empty directory. An
uncatchable process or host failure may still leave an empty or temp-only directory;
the loader rejects it with a specific safe-to-remove diagnostic, and it is never
described as an inspectable training run. After the incomplete manifest exists, every
handled interrupt or training failure leaves that manifest in place.

There is no lifecycle loop between training and evaluation. A train command saves its
new model while the artifact is still incomplete, reloads that persisted model through
the same validated actor adapter, evaluates the reloaded actor, writes the
profile-required evaluation files, and marks the manifest complete last. The public
`evaluate` and `run` commands are separate consumers that reject incomplete artifacts.

Interrupted and failed runs remain inspectable but cannot be evaluated, loaded by
`run`, or presented as complete. Automatic resume, partial overwrite, checkpoint
merging, remote upload, and garbage collection are deferred.

Artifact loaders reject:

- incomplete or unsupported manifest schemas;
- unknown trainer, profile, environment, suite, grid, preprocessing, or architecture
  versions;
- missing required, unexpected, wrong-size, or hash-mismatched files;
- malformed or non-finite JSON values;
- parameter names, shapes, dtypes, or counts that do not match the manifest;
- an explicit `--device cuda` request when a compatible CUDA device is unavailable.

Training-device provenance does not make an artifact device locked. Evaluation and
`run` load on CPU by default, including artifacts originally trained on CUDA, when the
framework supports safe parameter mapping.

For PPO, each configured environment-step budget is rollout aligned. The manifest
records both the requested budget and SB3's observed completed timestep count and
requires them to match for the two approved profiles.

SB3 archives and Torch model files are trusted local artifacts. Harpy does not claim
that they are safe to load from an untrusted source. BC loads a state dictionary into
code-constructed architecture and uses the safest weights-only loader available in
the pinned Torch release.

## Reproducibility contract

CPU is the authoritative checkpoint profile. Training seeds Python, NumPy, Torch,
SB3, Gym, suite construction, minibatch ordering, and environment reset streams.
Deterministic Torch behavior is enabled where the pinned CPU operators support it.

CUDA is an explicit exploratory profile. Its device, CUDA, driver, and library details
are recorded, but its weights and metrics are not assumed to match CPU.

Harpy promises the following on the same supported platform and locked dependencies:

- identical suite membership and ordering;
- identical direct and cached environment observations;
- identical loaded-model deterministic action traces;
- identical terminal episode records and evaluation JSON apart from explicitly
  excluded timing/path fields.

Harpy does not promise bit-identical training weights across hardware, PyTorch
releases, BLAS implementations, or CUDA stacks. Training wall time is descriptive, not
deterministic. The artifact records enough context to distinguish those cases.

## Error behavior

All failures use concise stderr diagnostics and nonzero exit status. A failure never
overwrites an existing directory or marks a partial artifact complete.

Validation happens before expensive work where possible:

- seeds must be non-negative integers and booleans are rejected;
- profiles, devices, trainer kinds, and suite IDs are closed enums;
- input artifact paths must be nonempty existing directories; for output paths the
  nearest existing ancestor must be a writable directory, missing parents are created,
  and the final target must not exist;
- checkpoint profile evaluation suites must match their frozen digests;
- observations, logits, probabilities, actions, metrics, and JSON numbers must be
  finite and in contract;
- learned actions must be integer `PitchAction` values before `env.step`;
- explicit device requests fail rather than silently falling back;
- optional dependency failure prints `uv sync --group train`;
- after bootstrap, Ctrl-C or trainer failure leaves an incomplete manifest and
  propagates failure; the documented pre-manifest crash boundary is the sole exception.

CLI exit statuses are stable: `0` for a successfully completed command, including a
valid checkpoint whose scientific criterion is not met; `2` for argument or closed
contract validation errors; `1` for dependency, artifact I/O/integrity, training, or
evaluation execution failure; and `130` for an handled user interrupt.

Evaluation and `run` validate the entire artifact before constructing an actor. They
must not partially load compatible-looking tensors or accept a hash failure with a
warning.

## Test strategy

Normal automated tests cover:

### Suites and leakage

- deterministic construction, golden digest, fixed length, balanced targets, source
  bands, initial-error exclusion, and split disjointness;
- identical ordered episode injection across learned actors and baselines;
- recursive actor-input allowlist with forbidden truth names and values;
- evaluator truth unreachable before done;
- target/control-only, shuffled-spectrum, and zero-spectrum probes;
- no use of final suite outcomes in BC early stopping or PPO callbacks.

### Cache and frozen environment

- every reachable coordinate has valid finite fixed-shape evidence;
- direct render, first miss, repeated hit, inverse action, and equivalent action-order
  observations are bit-identical;
- cached values and returned observations cannot mutate one another;
- cache cardinality never exceeds 9,801 and only spectrum arrays are retained;
- rewards, info, flags, action counts, RNG state, and `EpisodeResult` remain unchanged;
- all existing Milestone C environment and baseline tests remain green;
- registered `Harpy/SinePitch-v0` still uses the direct path.

### Features and models

- exact scalar order/formulas and spectrum shape;
- malformed dtype, shape, bounds, or non-finite input rejection;
- deterministic forward shape and seven finite logits;
- architecture, parameter count, state-dict, CPU device, and first-argmax behavior;
- BC and PPO adapters receive only actor-safe observation keys;
- BC weights use training counts only and all seven labels are represented;
- PPO starts with weights independent of the BC artifact.
- the policy-observation wrapper prevents SB3 one-hot preprocessing and emits the
  exact two-Box model input;
- PPO uses the custom shared extractor with empty SB3 `net_arch` and direct heads;
- forced budget exhaustion stores Harpy's exact terminal reward without timeout-value
  bootstrapping while the public Gym truncation flag remains unchanged.

### Artifacts and CLI

- exclusive directory creation and refusal to overwrite;
- incomplete-first/complete-last ordering under injected failures and interrupts;
- atomic JSON/model publication and SHA-256 verification;
- every manifest compatibility and corruption rejection path;
- trusted-local warning and optional-dependency guidance;
- human and strict-JSON `run` output, argument errors, and stable exit codes;
- module and console-script execution remain Qt-free.

### Training and evaluation

- a tiny deterministic BC train/save/reload/rollout smoke;
- a tiny deterministic PPO train/save/reload/rollout smoke;
- deterministic action traces after reload on the same stack;
- matched episode denominators and paired Random comparison;
- fixed Random seed-scheme reproduction and a separate Reward Search control row;
- exact metric aggregation, `null` handling, and no NaN JSON;
- checkpoint criteria computation without best-seed suppression;
- clean-source/declared-seed/CPU eligibility and exploratory-artifact exclusion;
- CPU default and explicit unavailable-CUDA failure.

The expensive 4,096-episode BC and five-seed 256,000-step PPO checkpoint is not part of
ordinary `pytest`. It is a separately invoked acceptance run whose artifacts and
machine-readable report are retained.

Torch/SB3-dependent test modules are skipped only when the optional training group is
absent; default-install tests still exercise lazy imports and the exact dependency
guidance. The authoritative development gate runs `uv sync --group train` followed by
the full test suite and verifies that no training-marked smoke test was skipped.

## Acceptance and documentation

Engineering acceptance requires:

1. the optional training group resolves from a clean checkout;
2. normal Harpy imports remain Torch/SB3-free and existing tests pass;
3. suite, cache, feature, artifact, evaluator, and CLI tests pass;
4. BC and PPO smoke profiles train, save, reload, evaluate, and run on CPU;
5. cache identity proves no `SinePitch-v0` semantic change;
6. artifact corruption and incomplete-run cases fail closed;
7. README commands work as written;
8. one BC checkpoint and five declared PPO checkpoint seeds are actually run;
9. the result document reports every seed, fixed suite, baseline, diagnostic probe,
   criterion, dependency/device detail, and artifact hash.

The result document distinguishes `engineering_passed` from `criterion_met` for BC
and PPO. Weak or failed learned results are still documented in full. No scientific
claim is based only on a smoke profile, training reward curve, best seed, action
accuracy, or OOD subset.

## Deferred next checkpoints

After Milestone D, the next environment decision is intentionally reopened with real
learned-policy evidence in hand. Candidate follow-ups include:

- `SinePitchTools-v0` with explicit, costed Analyze Source and Analyze Candidate
  actions and hidden-until-requested observations;
- DQN as a discrete-action/replay-buffer ablation;
- richer single-oscillator waveforms with the same tuning task;
- polyphony and chords only after single-note waveform generalization is measured;
- an immutable recorded-audio pitch-shift backend and fidelity reward;
- optional model-serving adapters only after the local actor contract is stable.

Those additions require new designs and environment versions. They do not enter
Milestone D through configuration flags or unreported compatibility paths.

## Resolved decisions

- The first learned checkpoint is BC diagnostic followed by fresh PPO.
- Stable-Baselines3 2.9 and PyTorch live only in an optional `train` group.
- The existing candidate spectrum remains the learned actor's audio evidence.
- Explicit analysis actions are deferred rather than added mid-version.
- Training uses the central source register; IID and edge-register suites are fixed and
  disjoint.
- A bounded spectrum-only cache is permitted only after bit-identical equivalence.
- BC and PPO share a compact architecture, but never share learned weights.
- CPU is authoritative; CUDA is explicit and exploratory.
- Final performance is judged by closed-loop terminal results on matched episodes.
- Failure to meet the declared model threshold is reported, not patched by eval tuning.
- Artifacts are local, versioned, hash checked, incomplete-first, and never overwritten.
- The user-facing surface is one small CLI, not an experiment platform.

No unresolved design question remains. Implementation begins only after this written
specification is reviewed and approved, followed by a separate implementation plan.
