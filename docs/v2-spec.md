# harPY v2: bounded sound matching

Status: implementation specification, 2026-09-16. **The features and commands in
this document are proposed, not implemented.** The released package remains v1.
This document defines the intended v2 scope; [the delivery plan](v2-plan.md)
records the v1 cleanup, milestones, and evidence needed to release it.

## 1. Product and outcome

harPY remains a **headless Python/API/CLI toolkit for training and evaluating
agents that recreate and shape synthesized sounds and chords**. An episode gives
an actor a short target recording, a starting patch, editable controls, and an
evaluation budget. The actor proposes patches, hears their rendered audio, and
explicitly submits a candidate. The result retains the editable patch, sound,
decisions, quality measurements, and cost.

The first useful workflow is: generate a known target, compare classical search
and a trained estimator, inspect their submitted sounds, reload the saved result,
and explain where either method failed. Sound resemblance is the outcome; exact
recovery of a non-unique generating patch is not required.

### Release scope

| Capability | v2 requirement |
| --- | --- |
| Reconstruction | Match a two-second sound from a compact, deterministic built-in synth |
| Polyphony | One voice and three simultaneous notes, reported separately |
| Shaping | Match a rendered target while changing only declared fields of an existing patch |
| Baselines | Fixed patch, seeded random search, bounded coordinate/pattern search |
| Learning | Supervised patch estimator; the same estimator plus budgeted refinement |
| Extension | Fresh Python actor factories with shared immutable model weights |
| Evidence | Reproducible targets, metered auditions, traces, portable and resumable results |
| Compatibility | Existing v1 task, training, artifacts, reports, and frozen evidence remain usable |

No graphical interface, live audio, DAW/VST hosting, external model service,
natural-language sound editing, or arbitrary-instrument reconstruction is part of
this release. Effects, modulation, variable note timing, per-voice timbre, stereo,
and more than three voices are later tasks. Existing v1 PPO remains supported;
training a new iterative RL policy is a follow-up after the new baselines work.
v2 still ships a real new training workflow, not only a renderer or model adapter.

## 2. Deterministic synth contract

Task identity: `sound-match-v1`. Initial renderer identity:
`harpy-compact-synth-v1`. These identifiers are independent of the package version.

`render(patch, schedule, config)` starts with fresh oscillator, filter, and
envelope state and returns exactly 96,000 finite mono float32 samples: two seconds
at 48 kHz. All voices start at sample zero; note-off is sample 48,000. There is no
random phase, noise, automatic gain normalization, limiter, or implicit clipping.
Target and candidate use the same frozen renderer and public timing schedule.
The schedule contains no target note pitches.

Patch schema identity: `harpy.sound-match.patch/1`; do not reuse the historical
synth patch's numeric schema versions (which already include versions 1 and 2).

| Field | Allowed value | Default starting patch |
| --- | --- | --- |
| `notes_cents` | Canonical sorted tuple of 1 or 3 distinct integers, 4800..7200 inclusive; MIDI note 60 = 6000 | `(6000,)` or `(6000, 6400, 6700)` |
| `waveform` | `sine`, `triangle`, `saw`, `square` | `sine` |
| `brightness_hz` | Integer 200..12000; coefficient frequency for fixed low-pass stages | 6000 |
| `attack_ms` | Integer 5..250 | 10 |
| `decay_ms` | Integer 20..500 | 100 |
| `sustain_db` | Integer -24..0 | -6 |
| `release_ms` | Integer 20..750 | 200 |
| `gain_db` | Integer -18..0 | -6 |

This finite grid makes canonicalization and replay unambiguous. Models may
predict continuous values internally, but their declared decoder produces legal
patches. The environment never silently clamps or rounds an invalid proposal.
Reject booleans as integers, nonfinite values, unknown/missing fields, and invalid
note counts. Accept input notes in any order and sort before locked-field checks;
sorting is the sole input canonicalization, with no value conversion. Note count
is public and fixed for each episode.

Use band-limited harmonic sums for the four oscillators: fundamental sine;
odd `(-1)**((h-1)/2)/h^2` triangle harmonics with factor `8/pi^2`; odd `1/h` square
harmonics with factor `4/pi`; and `(-1)**(h+1)/h` saw harmonics with factor
`2/pi`. Use sine-phase terms, omit harmonics at or above Nyquist, and sum in
ascending harmonic order. Frequency is `440 * 2**((cents - 6900) / 1200)`.
Each voice starts at zero phase. Sum voices in ascending pitch order with equal
weight `1 / voice_count`, then apply two identical zero-state low-pass stages:
`a = exp(-2*pi*brightness_hz/48000)`, `y[n] = (1-a)*x[n] + a*y[n-1]`.

Apply one shared linear-amplitude ADSR envelope after the filter. Attack rises
from zero to one; decay falls to `10**(sustain_db/20)`; sustain lasts until note-off;
release falls from the level at note-off to zero. Define segment endpoints by
sample indices with no repeated endpoint sample. Multiply by
`0.5 * 10**(gain_db/20)` for fixed headroom. The longest attack plus decay ends
before note-off; the longest release ends by 1.75 seconds. The implementation
must freeze its accumulator precision, final float32 conversion, endpoint
convention, and golden fixtures before benchmark generation. No render-dependent
peak scaling is allowed. Bounds tests must show every generated fixture finite
and unclipped; an out-of-range output is a renderer failure, not repaired audio.

Canonicalize a chord **before rendering, summation, hashing, serialization, and
comparison**. All six orderings of a three-note patch must produce identical
canonical bytes and audio. Actors replace the entire note tuple atomically;
there are no persistent voice indices. Three-note targets use distinct notes.
Exact replay is required in the qualified pinned CPU environment. Cross-platform
bit identity is unqualified until measured; dependency/runtime identity travels
with the renderer rather than silently accepting a changed target hash.

## 3. Episodes and information access

Reconstruction starts from the fixed patch above with all fields editable.
Shaping starts from a supplied patch and an editable-field mask. Initial shaping
cohorts are pitch/chord-only, envelope-only, and brightness-only edits. For a
benchmark target, locked fields equal the starting patch, so the target is
reachable. The note tuple is one field; a mask cannot expose only an unstable
voice index. A proposal modifying any locked field is invalid.

The actor's observation contains:

- Target samples and currently rendered candidate samples, with sample rate.
- Current canonical patch, public timing, allowed values, and editable mask.
- Remaining decision and candidate-evaluation budgets, last action status, and
  opaque episode identity.

Do not include the generating patch, target parameter labels, generation seed,
source coordinates, target-derived names, or evaluator-only pitch diagnostics.
Training data may expose labels in its separate training interface. Features
must be derived from public audio by a declared transform; neither the existing
trailing-window analyzer nor its silent-tail behavior defines this new task.
Full-clip/time-resolved analysis must retain attack, sustain, and release.

Actors own their observation copies; modifications cannot affect subsequent
observations, the target, or another actor. Factories create fresh episode state
and may share immutable weights. Actor cleanup runs on submission, truncation,
error, or cancellation. An invalid proposal releases only call-local resources
and leaves the actor alive for another decision. General Python actors are cooperative research extensions,
not a security sandbox; undeclared evaluator access invalidates a comparison.

## 4. Actions, budgets, and terminal states

The public API has exactly two operations:

| Operation | Behavior | Charged cost |
| --- | --- | --- |
| `Propose(patch)` | Validate a complete patch atomically, render from fresh state, and return its observation | One decision and one candidate evaluation when valid |
| `Submit()` | Terminate with the currently rendered patch/audio; no new patch argument | One decision; no new evaluation |

Defaults are **32 candidate evaluations and 64 decisions**. The starting patch is
rendered and charged as evaluation 1 before the first decision; generating or
loading the fixed target is setup cost, recorded separately. There can therefore
be at most 31 additional valid proposals at the default budget. Each valid
proposal is charged even when unchanged or supplied from a cache. No batch or
hidden renderer call is free. Listening to existing samples and computing public
audio features does not consume a candidate evaluation.

An invalid proposal consumes a decision, consumes no render evaluation, returns
a structured reason, and leaves the current patch/audio unchanged. This includes
proposing after render budget exhaustion. Exhausting render evaluations alone
does not terminate: the next decision may still submit the current candidate.
Using the last decision on `Submit()` is valid; using it otherwise truncates the
episode. A minimum budget is one evaluation and one decision. Early submission
of the initial patch is valid. There is no unrendered final candidate and no
implicit submission on timeout, failure, or budget exhaustion.

Charge a validated proposal's decision and evaluation before attempting DSP. If
that render fails, keep the previous candidate and terminate with error evidence.
An initial-candidate render failure has no candidate: initial, best, last, and
submitted candidate-dependent fields are `null`. Failure to generate or validate
the target is a run-setup failure, not an actor result.

Terminal statuses are `submitted`, `truncated`, or `error`, with an explicit
reason. An actor exception or renderer failure produces error evidence and
cleanup; a whole-run interruption instead leaves an incomplete run. A truncated
episode retains its last candidate for diagnosis, but has no submitted score.
Report submission/error/truncation rates alongside distances so actors cannot
look good merely by declining difficult targets.

Meter candidate evaluations, actual DSP renders/cache hits, decisions, invalid
actions, declared model calls, and wall time separately. Unknown model-call or
internal forward/backward counts are `null`, never invented zeros. Training
examples, updates, time, device, and peak memory are separate offline costs.

### Comparable information tracks

The primary `audio-only-metered` track permits DSP analysis and the public sound
objective on target/candidate audio. Search actors and learned actors have exactly
that access. Every candidate synthesis used to choose a patch goes through the
metered proposal interface. Training-time synthesis does not count as test-time
search but its dataset and offline cost must be declared.

An actor using an unmetered local renderer replica is ineligible for that track.
White-box/differentiable optimization needs a separately labeled track with
forward/backward/render counts; hidden-patch oracle replay is an evaluator check,
not a competing method. Hosted models can later use Python adapters, but v2
includes no provider integration, credential handling, or hosted evaluation.

## 5. Sound objective and interpretation

Primary evaluation uses the **whole rendered clip**. Begin calibration with a
multi-resolution spectral objective, without time alignment or independent gain
normalization. For each FFT size `N` in `{512, 2048, 8192}`, use periodic Hann,
hop `N/4`, no centering, frames starting at `0, hop, ... < 96000`, and zero-pad the
final frames. Define one-sided magnitude as `abs(rfft(window*frame))/sum(window)`
(no doubled interior bins). Let `T` and `C` be target/candidate magnitudes and
`eps = 1e-7`:

```text
convergence_N = ||T - C||_F / max(||T||_F, eps)
log_distance_N = mean(abs(log(max(T, eps)) - log(max(C, eps))))
distance = mean_N(convergence_N + log_distance_N)
```

Use float64 analysis accumulation; fix the implementation and identity before
final evaluation. These weights and floors are a **calibration starting point**,
not validated perceptual acceptance. Report both components at each resolution
even when a scalar ranks candidates. Any calibrated replacement gets a new
objective identity; do not change historical scores in place.

Always report temporal envelope error (RMS in contiguous 480-sample bins), whole
clip RMS level difference in dB using a declared floor, and evaluator-only pitch
set error in cents. Equal-cardinality one-dimensional note sets are matched in
sorted order; voice permutations cannot incur pitch error. Envelope RMSE and
level difference are diagnostics, not hidden corrections to the audio. Parameter
error is explanatory: equivalent sounds need not have identical patches.

Retain initial distance, submitted distance, and best auditioned distance as
distinct fields. Record quality versus evaluations at 1, 4, 16, and 32; report
actual cost and carry forward a terminated actor's best audition for this curve.
Do not label that curve as submitted success. Report per-cohort distributions,
median/p90 distance, and paired differences on matched episode membership.
Never pool single-note/chord, reconstruction/shaping, or information tracks into
one unqualified score. Include all terminal statuses in outcome counts; report
submitted distances with their denominator and last-candidate distances separately,
including the count of episodes with no candidate due to an initial-render error.

Calibration must include identity/oracle replay, note permutations, small and
large detuning, a missing chord tone, brightness/envelope/level changes, silence,
clipping, and phase changes. Invalid-patch perturbations may be supplied as audio
fixtures outside the candidate space. Retain target/candidate WAV pairs for
listening. Check that substantial mistakes are detectable and equivalent sounds
are not rejected solely for differing labels. No perceptual success rate may be
claimed until an acceptance rule is calibrated, documented, and frozen before
final evaluation. Until then publish distances and limitations.

## 6. Datasets, splits, and baselines

Maintain separate calibration, training, validation, IID-test, and
compositional-test manifests. Use opaque public episode IDs. Evaluator manifests
bind canonical target/start patches, masks, render hashes, generator identity,
split/group IDs, and versions; public actor payloads exclude this hidden data.
Training examples expose only the training partition's labels.

Assign a generator source-family ID to each base target and its explicitly
derived shaping/nuisance variants, and keep that group within one split. Also
group exact duplicate rendered targets and structural aliases: for example,
decay duration is inactive at 0 dB sustain. Freeze the structural alias rules
and exact audio-hash deduplication procedure in the generator identity. Reusing
a public starting patch across splits is allowed; do not group all reachable
edits or take transitive closure over approximate audio similarity. Investigate
near-equivalent audio separately during calibration with a recorded threshold
and contamination report. Different RNG seeds alone do not establish independence.

Generate pitch across the declared range; use musical chord families plus
independently varied non-pitch controls. Compositional tests hold
out target parameter combinations and target chord interval families with their
transpositions and inversions. This is target-distribution holdout, not a claim
that an actor never encountered those families in a public starting patch or
training-time audition. Record those exposures; a stronger unseen-family claim
requires excluding them too. Record the grouping rule and occupied pitch ranges. Held-out
waveform families would be a separate extrapolation study.

Start with a CPU pilot of 96 target episodes: 24 each for single reconstruction,
chord reconstruction, single shaping, and chord shaping; balance the three masks
within shaping. Use this only to check range, calibration, runtime, memory, and
failure modes. Freeze final counts and distributions after that pilot, before
model selection or final test use. v1's consumed cohorts do not qualify v2.
The pilot is a starting workload, not a promised adequate statistical sample.

Required actors use identical starting patches and evaluation budgets:

1. `fixed`: submit the starting patch.
2. `random-search`: seeded legal proposals; submit the best heard patch by
   proposing it again if it is no longer current (that proposal also costs).
3. `pattern-search`: deterministic bounded coordinate/pattern search over
   continuous-grid controls and categorical waveform, with a declared pitch-set
   proposal policy. Restore the best candidate through the same charged API.
4. `inverse`: a supervised estimator predicts a patch, proposes it, then submits.
5. `inverse-refine`: the same checkpoint followed by the declared bounded search.

Search must reserve enough evaluations to restore its selected candidate, or
submit its current candidate. There is no privileged `Submit(best_patch)` shortcut.
Seeded stochastic actors use a recorded per-episode seed policy independent of
target generation; seeds 0, 1, and 2 are used for learned training and stochastic
baseline repeats. Preserve every seed result rather than selecting the best.

The inverse trainer uses a small Torch audio encoder and typed parameter heads,
with note count/mask/current patch as public conditioning. Begin with a fixed
log-magnitude spectrogram encoder; compare representation changes later under
the same task. Train on generated labels; evaluate decoded, rendered audio
through this runner. The initial supervised contract is:

- Waveform: categorical cross-entropy. Notes: multi-label binary cross-entropy
  averaged across the 2,401 cent bins, positive weight `(2401-k)/k` for known count `k`.
  Decode the top `k` distinct bins, break ties toward lower pitch, then sort.
- Remaining controls: mean-squared error in `[0, 1]`; log-normalize brightness
  and ADSR time controls, linearly normalize sustain/gain dB. The actor decoder
  clamps its own predictions, converts to physical units, rounds to nearest
  legal integer with ties lower, and returns a valid patch.
- Average eligible field losses with equal field weights (the note-set loss is
  one field). Exclude locked fields and structurally inactive fields such as
  decay at 0 dB sustain. Copy locked values from the initial patch; inactive
  unlocked fields use a declared canonical value. Training masks may use training
  labels. Inference canonicalization uses only the decoded patch and public
  locked values, never the hidden target parameters.

Record architecture, feature transform, decoder, normalization, optimization,
data membership, seed, and checkpoint selection criterion. Select on a frozen
macro-average of validation audio distance across voice-count/task-mask cells,
with equal cell weights; report each cell as well. Never select on final-test
labels. A changed loss, decoder, or selection weighting is a recorded experiment,
not an undisclosed correction to a reference artifact.

Ship real CPU smoke updates, changed weights, save/reload equivalence, and normal
evaluation of the reloaded artifact. Report estimator alone versus estimator
plus refinement at matched total render budgets. A model need not beat the best
classical search for the toolkit to be useful; any trained-reference quality
claim needs a predeclared criterion and all-seed evidence. If a reference model
fails its criterion, retain the result and label it experimental. Designate
seed 0 before qualification if a checkpoint is bundled.

## 7. API, CLI, artifacts, and results

Keep `train`, `run`, `evaluate`, and `summarize`. Existing invocations retain v1
semantics; new workflows explicitly use `--task sound-match-v1`. Proposed usage:

```bash
# Proposed v2 commands; unavailable in the released v1 package.
harpy run --task sound-match-v1 --suite tone-smoke --episode tone-001 \
  --actor pattern-search --output runs/tone-example
harpy evaluate --task sound-match-v1 --suite chord-smoke \
  --actor random-search --actor pattern-search --output runs/chord-comparison
harpy train inverse --task sound-match-v1 --profile quick --seed 0 \
  --output runs/inverse-seed0
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor inverse=runs/inverse-seed0 --output runs/inverse-comparison
harpy summarize runs/inverse-comparison --format markdown
harpy evaluate --resume runs/interrupted-comparison
```

Suite manifests fix budgets, membership, masks, and objective; no unrecorded
command-line override changes a reserved protocol. A resume reads the existing
run configuration; changed task, members, actors, weights, or objective fail
closed. Summarization detects v1 files versus v2 bundles and reads either without
loading model packages. Empty, malformed, partial, and completed runs have
distinct readout behavior.

Expose typed `Patch`, episode, observation, `Propose`, `Submit`, actor-factory,
`render`, `run`, and `evaluate` interfaces under `harpy.tasks.sound_match`.
Share neutral lifecycle, provenance, and IO helpers with the current experiment
API, not a generic plugin framework. Preserve existing v1 public imports; its
pitch-specific episode/action/result types do not acquire nullable synth fields.

Base installation provides synthesis, analysis, classical/search actors, result
reading, and export. Add a Torch-only `learn` extra for inverse training; keep
`pitch` as a compatible Torch extra and `train` for existing SB3/PPO. Torch/SB3
must remain lazy and absent from base import requirements. Never load a
sound-matching artifact through a v1 pitch/PPO architecture inference shortcut.

Each new model artifact binds schema, task, renderer, objective used for
selection, observation transform, action decoder, architecture, training
manifest, seed, source identity, dependencies, and weight hashes. Unknown or
mismatched identities are explicit errors. Loading verifies declared artifacts;
hashes establish identity, not trust in arbitrary serialized Python objects.

A v2 run bundle contains a frozen manifest, append-only completed-episode records,
traces, canonical submitted patches, and target/submitted audio or sufficient
hash-bound renderer inputs to regenerate them. Every episode retains a compact
decision ledger: actions, canonical valid proposals, invalid reasons, evaluation
indices, candidate audio hashes, and objective components. Readers recompute
budgets, curves, and aggregates from it. Detailed observation dumps and intermediate
WAVs are optional; their omission must not remove the decision evidence.
Qualification bundles retain an explicit listening sample and every
submitted patch. Use portable relative artifact IDs plus hashes/sizes; local
resolution paths stay outside scientific identity. Do not embed machine roots
in distributable results.

Persist each completed actor-episode under an incomplete run. Validate the whole
membership and recompute aggregates before atomically creating a completion
marker; partial output cannot masquerade as final qualification. On resume,
verify configuration/artifact/renderer/runtime identity and prior records,
discard only a torn uncommitted final record, then execute members without a
committed terminal record using fresh actors and stable per-episode seeds. Error
and truncated records are completed members. A retry requires a new output bundle
referencing the original run/member; it never replaces the original denominator
or failure record. Patches, decisions, logical evaluation counts, and scores must
match uninterrupted execution in the qualified environment. Execution-attempt
records retain actual DSP renders, cache hits, timing, and interrupted work;
these can differ after a cold-cache restart. Unrecoverable interrupted costs are
unknown, not zero. Final output remains create-only.
Refuse concurrent writers. Memory and audio caches must be bounded independently
of total episode count, with cache policy recorded.

## 8. Release gates and deferred decisions

The [delivery plan](v2-plan.md) owns executable acceptance work. v2 requires:

- Deterministic legal rendering, chord permutation equivalence, locked-field
  enforcement, observation ownership, charged candidate use, and explicit submit.
- A complete one-voice, chord, and shaping workflow through run/save/reload/readout.
- Calibrated, frozen metric definitions and split membership before final study.
- Real inverse training and checkpoint reload, all-seed comparison against the
  stated baselines, and truthful qualification or experimental labeling.
- Safe interruption/resume, portable inventories, strict single-load readout,
  preserved v1 readers, and installed-wheel workflows without ML dependencies.
- Measured CPU render/episode time, throughput, peak memory, and disk use. CPU
  is the default; CUDA and additional platforms need their own qualification.

Open empirical choices are the calibrated sound-acceptance rule, final study
size, and whether the default 32 evaluations provide a useful runtime/quality
tradeoff. Resolve these in the CPU pilot and freeze them before final training
and qualification. They are not reasons to leave the implementation contract
undefined. New-task PPO, differentiable search, external WAV input, and larger
synths are explicitly subsequent milestones.

## References

- [DDSP: Differentiable Digital Signal Processing](https://arxiv.org/abs/2001.04643)
  motivates using audio reconstruction with interpretable synthesis components;
  it is not evidence that this renderer or chord benchmark is qualified.
- [Evaluating Sound Similarity Metrics for Differentiable, Iterative Sound-Matching](https://arxiv.org/abs/2506.22628)
  reports that loss performance depends on the synthesizer, supporting a local
  calibration step rather than importing a universal perceptual cutoff.

These are research references, not required package dependencies.
