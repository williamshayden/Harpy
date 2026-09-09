# Pitch decoding improvement, 2026-09-08

The release implementation is open for improvement. The pushed `v1.0.0` tag remains
at `1f2ba3fe1829537c3f8e274dd45abeeafaa744d2`; this work is isolated on
`william/pitch-decoding-v1`. No release reference is moved or published.

## What changed and why

`PitchPlannerActor(..., decoding=PitchDecoding.FEASIBLE_ARGMAX)` chooses the highest
model logit in the physically feasible interval. Candidate pitch must equal an
unknown source in 4800..7200 cents plus the observable control offset. Five-cent
classes may round at the interval endpoints by up to two cents. Selecting within
that interval is different from clamping an impossible winning estimate to a bound:
it retains the model's relative scores for feasible candidates.

The intervention adds no weights, training examples, dependencies, hidden-pitch
access, baseline fallback, or tuning parameters. It preserves the exact planner.
The original global-argmax actor remains the default for old artifacts and frozen
scientific workflows, with a separate identity for the new decoder.

`compare-pitch` makes the intervention usable: one artifact, the same episode
membership for both decoders, Spectrum Peak on the spectrum observation track,
complete terminal records, derived rescues/regressions, and separate model and
evaluator provenance. Actual package files, including untracked new source, are
hashed. Reports are explicitly exploratory and can be read without Torch.

## Evidence

Three CUDA-trained checkpoint models from the release replication were reused
without selection or retraining. CPU evaluation used one Torch thread. The first
ablation used an isolated prototype of constrained argmax on all existing IID and
register episodes; the original decoder reproduced the recorded release outcomes.

| Training seed | Original successes / 650 | Feasible successes / 650 | Rescued | Regressed |
| --- | ---: | ---: | ---: | ---: |
| 0 | 610 | 650 | 40 | 0 |
| 1 | 631 | 650 | 19 | 0 |
| 2 | 650 | 650 | 0 | 0 |

These 650 cases are the already inspected 250 IID, 200 lower-register and 200
upper-register episodes. They are development regression evidence, not a fresh
test set or a replacement result for the frozen criterion.

The implemented package actor was then tested on 100 fresh target/source pairs
with every model and Spectrum Peak. Generation used
`numpy.default_rng(SeedSequence([20260908, 1]))`; all existing pitch-suite pairs,
duplicate pairs and initially-within-five-cent cases were excluded. The exact
membership, protocol and model hashes were saved before evaluation. No tuning or
retries were used.

| Training seed | Original successes / 100 | Feasible successes / 100 | Original loop episodes | Feasible loop episodes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 98 | 100 | 2 | 0 |
| 1 | 98 | 100 | 2 | 0 |
| 2 | 100 | 100 | 0 | 0 |

Spectrum Peak succeeded on 100/100. Feasible decoding retained final MAE 1.94 cents
and mean action count 29.25 for every seed; Spectrum Peak had MAE 1.25 and mean
action count 31.14. Fresh pairs still share the same deterministic renderer and
source-coordinate family; they do not test new timbres or real audio. The three
model rows reuse the same 100 pairs and must not be treated as 300 independent
audio samples.

Local raw evidence is under `outputs/decoder-probe/`: `probe.py`, `result.json`,
`fresh.py`, `fresh-protocol.json`, and `fresh-result.json`. The prototype's summary
initially used the wrong truncation label; it was corrected by re-deriving counts
from the preserved `budget_exhausted` terminal records. No episode outcomes changed.

## Research findings that affect the design

This is an application of established constrained inference, not a new ML
algorithm. [Wang et al., ICML 2023](https://proceedings.mlr.press/v202/wang23h.html)
analyze how label constraints can affect prediction. Here the constraint follows
directly from Harpy's public environment contract, rather than an assumed musical
prior. It belongs at the model/controller boundary.

High isolated pitch accuracy does not establish closed-loop reliability. In our
development cohort, rare impossible estimates led to repeated states; fixing
decoding rescued 59 failed model/episode runs without improving the spectrum-only
network. This is consistent with the general sequential-distribution concern
studied by [Ross et al., AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html).
We did not implement or evaluate DAgger, and its learning guarantees are not claims
about this decoder.

One-cent performance also depends on the controller. Spectrum Peak computes and
commits a plan once per episode; the learned actor re-estimates and replans each
step. Even a perfect nearest-five-cent predictor can submit at the first point
inside the target class, usually two cents away. A pure-planner counterexample,
target 6000 and sources 6063..6067, produces final errors -2,-1,0,1,2 for a committed
plan and -2,-2,-2,-2,-2 for repeated replanning. The 2% IID one-cent score therefore
does not by itself establish poor model class accuracy. A matched-controller
comparison should precede more training or a larger network.

[CREPE](https://arxiv.org/html/1802.06182) provides precedent for smooth pitch labels
and weighted pitch decoding. Its [authors' implementation](https://github.com/marl/crepe/blob/master/crepe/core.py)
uses a local weighted average. Harpy's categorical logits require an adaptation,
not direct reuse of CREPE's sigmoid salience weights. A predeclared validation-only
screen tested local softmax at temperature one with radii 1, 2 and 4 five-cent bins.
Across 200 validation coordinates, seed 0 improved from 117 to 120 within one cent;
seeds 1 and 2 stayed at 117. Within-five counts stayed 199,199,200 for every choice.
The distant aliases remained. This gain does not justify another shipped decoder.
All predictions and independently checked logits are retained under
`outputs/local-decoding-validation/`.

## Verification and limits

The actor, comparison codec/API, CLI, readout, legacy pitch workflows and diagnostic
codecs passed 256 focused tests. Checks include endpoint rounding, extreme controls,
first-maximum ties, no hidden observation fields, unchanged global behavior, report
tampering, paired outcomes, create-only paths and readers without training imports.

One initial test run lacked the sibling virtual environment's console scripts on
PATH; using that environment's bin directory resolved it without a repository
workaround. A new readout test also initially assumed seed 0 although its shared
fixture deliberately uses seed 7; the assertion was corrected to the fixture's
identity. After final artifact-mutation and CPU-only guards, all 26 comparison
tests passed, including four added checks. Repository-wide Ruff, changed-file
format checks and `git diff --check` passed.

The real `compare-pitch` CLI completed on the unchanged CUDA-trained seed-0
checkpoint. Every global and feasible terminal record matched the development
prototype, including all 40 rescues and zero regressions. Canonical round-trip
bytes matched stdout and the saved file. The report SHA-256 is
`ac0d6c8593e13d1f5020215a8a9240aa5cadf2bbb6c43b16e5b5ee051cd78170`;
its exact evaluator package snapshot is
`b9de1e45f79eb205c280cff5f77d65273dbd334a32f86ef4a0cdd3dbe608fe92`.

A candidate wheel built successfully and was installed outside Git into a fresh
base-only environment. With neither Torch nor Stable-Baselines3 installed, that
installation read both the historical release cohort and the new comparison,
produced Markdown, and exposed `compare-pitch --help`. Its base NumPy version was
2.5.3; the evaluation used the original training environment's NumPy 2.5.1. This
checks installed report consumption, not installed-wheel model execution or new
training. The candidate retains the development tree's 1.0.0 metadata and is not a
replacement for the published tag or an artifact to distribute as the original
release. No package publication was performed.

The create-only comparison, rendered readout, CLI streams and machine-readable
verification are in `outputs/decoder-probe/cli-comparison.json`,
`cli-comparison.md`, `cli-comparison.stdout.json`, `cli-comparison.stderr.log` and
`verification.json`. The pure-planner counterexample is also retained there.

This does not establish dependable tuning for arbitrary audio, a new formal
scientific pass, or superiority to Spectrum Peak. The most valuable next experiments
are matched planning semantics, smooth supervision if precision still warrants it,
and a separately versioned noise/level robustness task with nuisance-aware evidence
keys. Preserve the clean task as a controlled baseline.
