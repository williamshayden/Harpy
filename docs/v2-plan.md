# harPY v2 delivery and v1 cleanup plan

Status: proposed implementation plan, 2026-09-16. Scope is defined in
[v2-spec.md](v2-spec.md). This planning change does not implement the new task.

Implementation has since started on the development branch. The
[first milestone record](v2-m1-verification.md) distinguishes completed M1/M0
work from the remaining calibration, training, persistence, and release gates.

The audit baseline is released v1 source
`2ea00c692dfdee216ea9036b2d3f7c45e81bc9c6`, matching `main` and tag `v1` when checked
on 2026-09-16. The older `v1.0.0` tag is a separate historical checkpoint.
Implementation should proceed in small reviewable milestones from this baseline;
do not develop against the older milestone branch or rewrite frozen evidence.

## 1. What v1 needs alongside the new task

These findings are source-confirmed unless explicitly labeled otherwise. Priority
is relative implementation order, not a claim that v1's scientific results fail.

| Priority | Finding and evidence | Required improvement | Verification |
| --- | --- | --- | --- |
| P1 | No tracked CI configuration; checks live in [the contribution guide](v1-contributing.md#check-what-the-change-affects) and dated release evidence | Add Linux/Python 3.12 CI for contracts, base imports, builds, and installed examples; keep optional training checks separately visible | Locked development checks plus normal wheel/sdist installation outside the checkout; base jobs have no Torch/SB3; report skipped optional cases |
| P1 | [The runner](../src/harpy/experiments/runner.py) accumulates all episode records; [the CLI](../src/harpy/cli.py) saves only after evaluation completes | Add bounded progress and resumable persistence through shared infrastructure, for v2 and an opt-in v1 bundle mode | Interrupt after completed episodes, resume missing work only, reject changed inputs and concurrent writers, preserve final create-only publication |
| P1 | Custom artifact inventories serialize resolved absolute filenames in [the runner](../src/harpy/experiments/runner.py) | Separate portable IDs from local lookup paths while retaining source identity, hashes, sizes, and mutation detection | Same content in two checkout roots has equivalent portable inventory; no machine-root prefixes in publication exports; original v1 bytes unchanged |
| P2 | CLI summary dispatch parses the entire result, then [the result reader](../src/harpy/experiments/results.py) reads/parses it again | Dispatch and strictly validate a single decoded document; preserve metric recomputation | One file load; duplicate keys/nonfinite values still rejected; identical scientific records and readout across current/historical fixtures; measure retained large study |
| P2 | [Active artifact code](../src/harpy/experiments/artifacts.py) imports neutral helpers from historical `learning.artifacts`; historical readout reaches training workflows | Extract shared JSON, hashing, source identity, and atomic publication helpers; isolate historical readers/execution with compatibility shims | Current and historical fixtures, installed imports, checkpoint reload, and source-provenance checks pass without broad deletion |
| P2 | Full three-seed reference qualification and seed-1/2 artifacts remain local, as disclosed in [README](../README.md#read-the-evidence) | Prepare a checksum-bound portable v1 evidence bundle or keep its access limitation explicit; require distributable v2 evidence before claiming independent reproducibility | Recompute qualification from archived records and reload all cited artifacts without retraining or choosing a new seed |
| P2 | [The FFT guide](v1-waveform-fft.md) previously described only the local study archive | Point current guidance to the published portable result; distinguish original JSON, portable JSON, and compressed-download hashes | Verify manifest, hashes, and download/decompress/read instructions; preserve dated reviews as historical records |

The final documentation item is corrected in this planning change. The remaining
items are planned work, not completed fixes. Evidence publication is a separate
release action after the local bundle has been prepared and checked.

A read-only audit probe of the existing 17,995,025-byte study observed two full
reads, 6.34 seconds elapsed, and 215.3 MiB peak RSS for an ordinary CLI summary.
That is one machine/run, not a performance guarantee or statistical benchmark.
The previous summary-aggregation optimization is already present; avoid claiming
to fix it again. Repeat the same retained input before/after the single-load fix.

### Preserve what is already sound

- Fresh actors per episode, shared immutable weights, owned observations, cleanup,
  artifact mutation checks, strict membership, and recomputed result summaries.
- Atomic, create-only final output; the new journal complements this protection.
- Existing v1 source/target semantics, seven actions, 64-step limit, inclusive
  five-cent explicit-submission criterion, and frozen renderer/observation bytes.
- Current v1 training, optional dependency isolation, historical readers, and
  serialized class paths required to reload existing checkpoints.

Do not reopen already-fixed finite-input validation, retired CLI duplication,
reserved smoke membership, or the bounded waveform cache without new evidence.
An ADSR Python loop is a profiling candidate, not a proven bottleneck. Do not
delete old BC/training modules solely because they look historical: inventory
reader and checkpoint dependencies first.

## 2. Architecture and migration decisions

The existing `EpisodeSpec`, `Decision`, runner, and strict result validators are
pitch-task contracts, not task-neutral containers. Add typed sound-matching
models and a new versioned result bundle. Reuse internal lifecycle and provenance
primitives; keep public v1 wrappers and exact readers for old schemas. Task
selection is two explicit built-ins, not a plugin registry.

Proposed source organization:

```text
harpy/tasks/sound_match/   patch, renderer, episode, observation, actions, actors
harpy/experiments/         existing v1 API plus shared lifecycle/result dispatch
harpy/<shared helpers>/    canonical JSON, hashing, provenance, atomic IO
harpy/learning/            existing trainers/readers plus inverse model adapter
```

Choose the shared helper module name during extraction; avoid a broad directory
reorganization. The sound-matching renderer does not alter the stateful v1
`SynthEngine`, its patch codecs, or the trailing-window spectrum API.

New artifacts identify task, renderer, observation, action decoder, architecture,
and training manifest explicitly. Unknown identities fail before running a model.
Old pitch/PPO artifact readers continue to apply their original validation.
New result writers use portable artifact references; old result readers preserve
their recorded absolute labels. A portable export of existing evidence records
the exact transformation and both hashes, as the published FFT study does.

Keep default v1 CLI commands and output-file semantics. Add an explicit bundle
output option to v1 evaluation for journaling/resume; ordinary in-memory Python
`evaluate` and legacy JSON output remain supported. Both tasks use the same
strict resume machinery, while final records retain task-specific validation.
Current aggregate metrics remain compatible; portable/resumable v1 bundles have
an explicit new envelope identity and do not pretend to be old frozen files.

Keep Python 3.12/Linux/WSL as the initially qualified environment. Add the
Torch-only `learn` extra without changing `pitch` and `train` compatibility.
New source/spec documents must be included in the source distribution when v2
packaging is implemented; this planning change leaves the v1 package manifest
and version unchanged. No desktop, web, provider, or database layer is needed.

## 3. Milestones and completion evidence

### M0 — Protect v1 and untangle shared infrastructure

Add base CI and installed-package smoke coverage. Remove duplicate result loading
without relaxing strict parsing. Extract neutral serialization/provenance helpers
with compatibility imports. Inventory checkpoint/report dependencies before any
retirement. Define portable artifact IDs and read old inventories unchanged.

Done when focused contracts and appropriate historical artifact/readout fixtures
pass; built wheel and sdist work outside the checkout without Torch/SB3; the
retained study gives identical readout through one load; artifact mutation and
malformed evidence are still rejected. Publish before/after summary time and
memory with the measurement environment. Do not require speculative speedups.

### M1 — One complete single-voice experiment

Implement the versioned patch, renderer, observation, `Propose`/`Submit` protocol,
meters, fixed/random/pattern actors, full-clip objective, and traced run bundle.
Build the simple end-to-end path before generalizing shared abstractions further.

Done when a one-voice target runs through propose, audition, explicit submission,
save, reload, summary, and WAV/patch inspection. Verify legal-boundary rendering,
nonfinite rejection, sample count, sample endpoints, fresh-state replay, cache
identity, immutable observations, last-decision submit, post-budget submit,
invalid proposals, and exception cleanup. Render oracle identity and meaningful
pitch/envelope/brightness perturbations. Inspect actual saved audio and listen
to representative target/candidate pairs; numeric tests alone do not qualify
sound quality. Freeze renderer precision and golden fixtures here.

### M2 — Chords, shaping, and sound calibration

Extend the same task to three voices and editable-field masks. Add the four pilot
cohorts from the spec, their membership checks, and score/diagnostic views.
Canonicalize chord order before synthesis and identity. Run the bounded CPU pilot
with fixed/random/pattern actors and retain failure/listening examples.

Done when all chord permutations are identical, pitch-set diagnostics ignore
voice labels, locked fields cannot change, and every generated shaping target is
reachable. Record CPU render/episode timing, peak memory, cache behavior, and disk
use. Establish whether 32 evaluations is useful within available resources;
adjust defaults only with recorded evidence before freezing the protocol.

Use calibration fixtures to finalize objective definitions. If a defensible
sound-acceptance cutoff is unavailable, freeze continuous distance reporting and
explicitly leave perceptual success unclaimed. Freeze split/group rules, final
counts, manifests, baseline settings, and the planned reference-model criterion
before final model selection and evaluation. Do not expose final tests to tuning.

### M3 — Train a useful initial-patch model

Implement the inverse estimator, feature/decoder identity, generated training
loader, CPU `quick` profile, save/reload, and `inverse-refine` actor. Keep data
generation/cache and minibatches bounded; no full waveform corpus in RAM.
Use canonical target labels and reconstruct the legal patch at inference.

Done when a CPU smoke run performs real optimization, changes weights, reloads
equivalently, and evaluates through the ordinary task runner. Run training seeds
0/1/2 and compare estimator, estimator+refinement, and all baselines on matched
validation memberships and candidate costs. Final-test comparisons wait until
M5 after actors and evaluation infrastructure are frozen. Report single-note/chord and
shaping results separately. Training hardware/time is explicit. No silent CUDA
fallback, outside downloads, best-seed selection, or inherited v1 quality claim.

A weak model is a reported research result, not grounds to change the final
test set or advertise success. The toolkit can ship with an experimental learned
actor if its workflow is correct; qualifying a bundled reference additionally
requires the predeclared quality criterion. Do not make superiority over the
best search baseline an unsupported release promise.

### M4 — Durable matched evaluation for both tasks

Complete append-only episode journaling, bounded progress reporting, single-writer
ownership, manifest validation, resume, and atomic final completion. Apply it to
v2 and opt-in v1 bundles. Complete portable exports, task-aware artifact validation,
and strict text/Markdown/CSV summaries without model imports.

Done when a forced interruption after several episodes leaves readable partial
evidence; resume preserves completed records and produces the same scientific
outcomes as uninterrupted execution. Reject changed membership, model bytes,
renderer/objective, malformed committed records, unsafe bundle-relative paths,
and concurrent writes. Recover a torn last uncommitted record without discarding
valid completed records. Bound cache/working memory independently of suite size.
Verify clean relocation of a bundle and exact trace replay in the qualified runtime.

### M5 — Qualify, package, and document

Run the frozen study after M0-M4 pass. Prepare portable manifests, all-seed
metrics, submitted patches, listening examples, checkpoints for every claimed
reference, source/dependency identities, checksums, and reproduction instructions.
Prepare the outstanding v1 archive bundle where available; otherwise retain its
explicit access limitation instead of claiming independent reproduction.

Done when a clean installed wheel can run base examples, read both generations
of reports, render saved patches, and recompute released metrics from the prepared
evidence. Verify optional inverse and v1 pitch/PPO reload/training separately and
report skips or platform limitations. Review saved sounds and failure cases.
Update README, getting started, migration, acceptance, package metadata, and the
source-distribution contents to describe actual shipped behavior. Publication
and tagging follow review of these concrete artifacts; this spec creates no
release or remote deployment.

## 4. Subsequent experiments

New-task PPO requires a real observation/action encoding for complete patch
proposals, task-aware termination, new architecture metadata, and real update/
reload tests. The existing seven-action pitch PPO cannot simply be pointed at
this environment. Compare any iterative policy against the frozen inverse and
search baselines before making a quality claim.

Differentiable optimization gets its own access/cost track. External WAV targets
need explicit import, level/timing, achievable-target, and scoring policies.
More oscillators, per-voice controls, effects, and natural-language requests each
expand the task contract and dataset; they do not enter through a renderer change
under the existing identity.

## 5. Verification of this planning change

The baseline focused v1 checks passed: 75 tests across imports, CLI, experiment
runner, and result validation using the existing released environment on
2026-09-16. This is a baseline check, not a full suite, fresh training run,
cross-platform qualification, or evidence that the proposed v2 works.

The planning change is documentation only. Its five changed documents passed
local-link/anchor and fenced-block checks (23 local links), plus whitespace
validation. Independent architecture and scientific reviews checked the action,
budget, persistence, dataset, and training contracts; their corrections are
incorporated. The FFT guide's three hashes match the retained release artifacts.
Freshly decompressing the retained published gzip and running the documented
`harpy summarize` command succeeded with no stderr. No large asset download,
training, or new-task execution was performed. Future commands above remain
visibly labeled as proposals until implementation lands.
