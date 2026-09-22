# V2 acceptance and evidence

Implementation and the frozen numerical study completed on 2026-09-22. Actual
listening review remains an open acceptance gate. This record distinguishes
implemented behavior, checked evidence, and that remaining review. Package version
is `2.0.0`; no v2 public tag or remote publication is implied by the local checkout.
The original [M1 verification](v2-m1-verification.md) remains a dated record.

## Scope and protocol

V2 implements deterministic known-synth sound matching with one or three notes,
metered patch proposals, locked-field shaping, fixed/random/pattern baselines,
supervised inverse estimation and inverse-plus-pattern refinement. It preserves
v1 pitch semantics, models, readers, and default JSON CLI behavior.

M2 froze the renderer/objective, 32-evaluation/64-decision budget, group separation,
source manifests, seed policy, validation weighting, and reference criterion
before model selection and final evaluation. Counts are 96 calibration, 2,016
training, 192 validation, 192 IID test, and 192 compositional test targets. The
objective is a continuous research distance; perceptual acceptance is unqualified.
See [calibration](v2-calibration.md) and [inverse method](v2-inverse.md).

## Engineering checks completed

- The base release selection passed 10,855 checks in 159.53 seconds, with five
  optional dependency skips and two explicitly excluded model-loading cases.
- The optional CPU selection passed all 335 checks in 81.60 seconds, including
  actual v1 pitch/PPO updates, frozen historical checkpoint loading, report
  compatibility, and all inverse training/inference tests.
- The current CLI/provenance selection passed 35 checks, including actual sound
  and pitch runs, old-result readout, resume, incomplete readout, and replay.
- Shared journal/task tests passed forced SIGKILL, an actual competing writer,
  preserved-prefix resume, relocated readout/resume/replay, torn-tail recovery,
  strict corruption rejection, cleanup, and v1 scientific-record equality.
- Eighteen inverse checks passed, including real optimization, changed weights,
  repeated-seed tensor identity, strict tensor reload, ordinary inverse/refinement
  execution across all eight cells, masks, decoder behavior, and mutation checks.
- The 96-target pilot retained 288 submitted records with no errors/truncations.
  Reverification checked all 74 evaluator source files, 2,688 frozen target audio
  identities, split disjointness, 14 calibration fixtures, and 28 fixture WAVs.

After the exact objective cache and target-hash guards, the focused sound-task,
CLI, journal and process selection passed 198 checks in 55.57 seconds, with one
Torch-only module skipped in the base environment. The final inverse metadata
regression also performed actual quick training, reload and eight-cell actor
evaluation. [Cache verification](v2-objective-cache.md) records exact equality
against the original scorer, recorded pilot scores and full actor traces.
A further 73 focused checks passed for paired differences, listening selection,
resource and scheduling receipts, strict study binding and safe archive expansion. The custom
Python actor example also ran from the installed base wheel outside the checkout:
both targets submitted with two evaluations each and zero invalid actions.

The frozen evaluator source is
`958bdcf36a297fc67bfa7a0a24d18ce1c52afe9e443d0064169ff2637e0dfa5e`,
committed locally in `410241e`. Both wheel and sdist installed into independent
environments outside the checkout and passed base workflows, both historical
whole-report schemas, v1 archive qualification, relocation, replay and exact
saved-patch rendering. Torch and Stable-Baselines3 were absent in those base
environments. Installed inverse, pitch and PPO smoke lanes each proved real
parameter updates, exact reload and ordinary durable evaluation/replay.

| Installed lane | Seconds | Peak RSS, MiB |
| --- | ---: | ---: |
| Wheel, base | 30.19 | 274.44 |
| Source distribution, base | 29.66 | 274.68 |
| Wheel, inverse + pitch + PPO | 46.61 | 445.32 |

All 99 wheel payload files and 95 sdist package files were compared against the
installed files. Base dependencies were installed offline from the existing
cache. The optional environment reused exact verified CPU dependency files from
the validated environment after an offline registry lookup failed; the project
itself was installed from its wheel. It was not an independent dependency download.
The [portable engineering receipt](../research/sound_matching/evidence/engineering-qualification.json)
binds source, distributions, versions and original local proof-report hashes.
Hosted CI execution, native Windows/macOS and fresh CUDA execution are unqualified.
Waveform caches, objective-analysis caches, journal record processing and the
durable reader's score-quantile working storage are bounded. Membership manifests,
file inventories and the paired postprocessor's compact per-episode values still
scale with membership; the implementation does not claim constant total memory
for arbitrarily large manifests.

## V1 infrastructure and retained evidence

Current experiment infrastructure uses neutral provenance helpers. Historical
type imports remain compatible aliases; historical report reading no longer
imports execution workflows or the optional ML stack. V1's renderer and control
contracts remain unchanged. Durable bundles are opt-in for pitch evaluation.

The retained representation study is exactly 17,995,025 bytes, SHA-256
`69db356439daf33f320d7a596d236584c8053f55b1954927c452dbda8679be8f`.
One ordinary summary process on 2026-09-22 used 135.26 MiB peak RSS and took
7.75 seconds. The recorded earlier probe used 215.3 MiB and 6.34 seconds. A focused
test asserts one file load/decode instead of two. These are single-run measurements
on different days; inverse training ran concurrently with the newer measurement.
The evidence supports reduced retained memory, not a speed claim.

The portable v1 qualification archive is prepared locally as
`outputs/v1-portable-qualification-20260922.tar.gz` (1,266,213 bytes), SHA-256
`b5ad97fd6a0a5e1b045dcd28c26ddf9f5ab51e3a738b80d67f8190365c81f03c`.
Its 23,800 records were strictly reloaded and all 18 qualification gates recomputed
true. All three model payloads were hash-verified. The manifest records original
result hashes and the portable actor-path relabeling. It has not been published;
a source clone alone does not provide this archive.

## Frozen matched study

The complete study retains all 6,336 actor-episodes: 192 validation, 192 IID-test
and 192 compositional-test targets, each evaluated by eleven actors. Every
actor-episode submitted, with zero errors, truncations or invalid actions.
The [results tables](v2-results.md) report all eleven actors across all three
splits and separate the four reconstruction/shaping cohorts. The
[full report](../research/sound_matching/evidence/final-study.json) and
[paired differences](../research/sound_matching/evidence/paired-differences.json)
retain exact cell metrics, distributions, costs and episode-matched comparisons.

All 2,112 matched validation actor-episodes submitted, with zero errors and
truncations. The three inverse training seeds passed the predeclared criterion:
each selected model has lower equal-eight-cell mean submitted distance than the
fixed starting-patch actor. Their results reproduce the selected training-time
validation scores. No model or protocol was changed after these comparisons.

| Actor | Validation mean distance |
| --- | ---: |
| fixed | 1.855378 |
| pattern-search | 0.976593 |
| random-search-0 | 0.713268 |
| random-search-1 | 0.744264 |
| random-search-2 | 0.739019 |
| inverse-0 | 1.151327 |
| inverse-refine-0 | 0.522320 |
| inverse-1 | 1.204497 |
| inverse-refine-1 | 0.524790 |
| inverse-2 | 1.185955 |
| inverse-refine-2 | 0.528213 |

Lower distance is better under the frozen objective. Each of the eight cells
has equal weight, so the two reconstruction cells contribute 25% of this mean
and the six shaping cells contribute 75%. The complete
[validation report](../research/sound_matching/evidence/validation-study.json)
retains every cell, diagnostic, cost and denominator. Actors share ceilings of
32 evaluations and 64 decisions; fixed uses one evaluation, inverse uses two,
and the search actors can spend the full budget.

The one-proposal inverse models exceed the full-search baseline distances in
this comparison. Inverse plus refinement has lower validation means than the
declared search baselines for every seed. This pattern also holds for the
equal-eight-cell means on both final tests: refinement spans 0.439459–0.480492
on IID and 0.589246–0.631621 on the compositional split. It does not hold on every
individual target or every cell. The retained
median/worst examples include large chord pitch-set errors and a single-note
shaping error of 702 cents. Submission alone is not a sound-quality result.

The validation process took 1,800.99 seconds and peaked at 312.09 MiB RSS. Its
last episode checkpoint was at 1,597.69 seconds; strict completion/readout and
target-hash verification added 199.15 seconds inside the driver. An independent
120-record comparison against the earlier evaluator prefix preserved every
scientific score, patch and cost count exactly; only elapsed times were ignored.

Whole-process measurements include startup, target-hash preflight, evaluation,
persistence and strict completion checks:

| Split | Wall seconds | Peak RSS, MiB |
| --- | ---: | ---: |
| Validation | 1,800.99 | 312.09 |
| IID test | 2,326.08 | 314.71 |
| Compositional test | 2,650.34 | 314.90 |

Each worker used one CPU thread. The two final workers overlapped on the shared
host after memory became available; these times are operational measurements.
Each invocation also performed 192 target-hash preflight renders outside the
bundle's DSP counters. The [resource receipt](../research/sound_matching/evidence/resource-receipts.json)
and [scheduling receipt](../research/sound_matching/evidence/scheduling-receipt.json)
retain exact runtimes, raw-log hashes and the observed concurrency bounds.

## Local distributions and evidence

The final local distributions are prepared under `outputs/v2-release-20260922/`:
`harpy_audio-2.0.0-py3-none-any.whl` and `harpy_audio-2.0.0.tar.gz`. The source
distribution includes the current guides, research drivers, qualification tests,
their frozen FFT reference fixture, and `uv.lock`.

The complete evidence archive is prepared as
`outputs/v2-complete-study-20260922.tar.gz`, with a neighboring
`.tar.gz.sha256` checksum. It retains all three selected models, the three study
bundles, paired differences, listening WAVs, frozen evaluation source, final
distributions and documentation, and the separate M2/training/v1 evidence.
Compressed archive hashes live outside the archives to avoid a self-reference.
See the [archive guide](../research/sound_matching/README.md) for verification,
model paths and reproduction commands. These are local artifacts; archive
preparation does not publish them remotely.

The final installed-package verification travels as the separate companion
`outputs/v2-installed-release-proof-20260922.tar.gz`. It retains exact package
identities, the wheel's full-study and paired-report comparisons, and the sdist
installation/fixture checks. An external release index binds this companion,
the scientific archive and both distributions by SHA-256. Keeping the final
proof separate lets installation and archive verification run independently.

## Listening review remains open

The frozen listening index retains all 264 declared median/worst pairs. Its
[receipt](../research/sound_matching/evidence/final-listening-receipt.json) binds
the index to the full report and completed bundles. There are no terminal errors
or truncations to add. Four 9.75-second validation review clips are also retained
in the archive: seed-0 inverse-refine median and worst examples in each cohort,
with original samples and gain and only silent gaps added.

Numerical inspection found all sixteen source segments finite and nonzero, with
zero endpoints and no sample magnitude at or above one. Those checks and the
listening index do not establish audible quality. The original M1/M5 plan requires
actual listening feedback; it remains recorded as unperformed. Full v2 acceptance
is therefore open at that review gate, and perceptual success remains unclaimed.
