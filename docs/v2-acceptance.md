# V2 acceptance and evidence

Qualification in progress on 2026-09-22. This record distinguishes implemented
behavior, checked evidence, and outstanding release gates. Package version is
`2.0.0`; no v2 public tag or remote publication is implied by the local checkout.
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

Later source changes require relevant reruns; final qualification below will
identify the frozen evaluator and packaged artifacts rather than treating these
development checks as an immutable release run.

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

## Outstanding qualification

The three-seed inverse artifacts, complete matched validation/final studies,
optional legacy training checks, installed wheel/sdist verification, final
checksums/source archive, and listening/failure examples are being assembled.
Do not infer learned superiority, perceptual success, a public release, or full
v2 completion from this work-in-progress record.
