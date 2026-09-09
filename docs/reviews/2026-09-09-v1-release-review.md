# Revised v1 repository and release review

Reviewed 9 September 2026 in `william/pitch-decoding-v1`, against the dirty revised
candidate based on `1f2ba3fe1829537c3f8e274dd45abeeafaa744d2`. This records an
incremental cleanup of that candidate, not a review of only the original tag.
The original `v1.0.0` tag is preserved. Nothing was committed, pushed, published,
or deployed by this review.

## Purpose and assessment

Harpy serves researchers and developers comparing audio representations,
estimators, and bounded control on a reproducible single-sine task. Excellence
means a clear first experiment, fair comparisons, inspectable failures, reliable
saved results, and a small supported extension surface. It does not require a
package UI, hosted service, plugin registry, or additional model architecture.

The core has substance: deterministic rendering and static corruption, fresh
actor state per episode, separated estimator/decoder/controller identities,
create-only outputs, artifact mutation checks, and a checkpoint with recorded
clean qualification. The training tests establish real parameter updates and
reloads. Saved failures and separate information tracks are retained rather than
hidden behind a pooled score. This review found no new confirmed defect in the
audio task, tuning rules, or qualified reference model.

The main weaknesses were duplicated command ownership, expensive report readout,
incomplete validation of provenance claims, and documentation whose current and
historical roles were unclear. Those are addressed below.

## Changes and priorities

| Finding | Priority and decision | Result |
| --- | --- | --- |
| The ordinary result reader accepted null source/runtime objects, invalid identities, negative elapsed time, and device claims inconsistent with actors. | High confidence; result integrity; small validation change. | Validate known fields, require a package identity or an explicit unknown-source reason, and preserve finite extension metadata. Existing package-hash-only producers remain supported. |
| The reserved clean smoke protocol ID could be attached to substituted membership with a recomputed digest. | High confidence; protocol integrity; low regression risk. | Apply the same frozen-membership validation used for the other reserved clean protocols. |
| Summary generation repeatedly reconstructed action histories and recomputed aggregate metrics. | Measured user-facing delay on real results; small implementation change. | Replay only prefixes needed for estimates, group records once, and compute each summary view once. No persistent cache or new result format. |
| `harpy.learning.cli` still exposed retired workflows alongside the new four-operation CLI. | Duplicated supported surface; clear migration boundary. | Remove the 649-line command module. Historical execution uses the original tag; saved reports remain readable through `harpy summarize`. |
| The practical guide lived among website drafts, while an old workflow guide looked current. | Direct onboarding problem; reversible documentation change. | Promote `docs/v1-getting-started.md`, shorten README, add virtual-environment setup, and label historical guides without rewriting their bodies. |
| Website presentation files were mixed into the toolkit repository. | Product boundary already agreed with the user. | Preserve the publication handoff outside Harpy; retain the scientific exporter under `research/figure_export/`. |

Command retirement also removes obsolete command-specific tests. Meaningful
coverage moves to the current CLI: historical readouts, import isolation,
installed-wheel pitch training/reload/evaluation/traces, and the warning before
PPO model deserialization. Direct historical BC/PPO artifact smoke coverage and
current pitch/PPO parameter-update tests remain. Test-count reduction is not
treated as a quality improvement by itself.

## Verification

The pre-change snapshot contains 221 source/documentation files and their hashes,
including all existing uncommitted work. Baseline lint passed. The broad baseline
was deliberately interrupted after **1,578 passes in 658.95 seconds**, with no
reported failures. This is a partial baseline.

Final verification:

- The broad final regression passed **11,565 tests in 535.28 seconds**, with no
  failures or skips. Its two excluded training modules were covered by the
  separate runs below. Reconciling the three JUnit reports against a fresh full
  collection confirms **all 11,580 current tests passed across these runs**, with
  none missing. This is not a claim of one uninterrupted 11,580-test run.
- 84 focused runtime/actor/qualification tests passed after the final provenance
  change. The earlier results/protocol subset passed 32 tests.
- 48 affected CLI/import/readout/training tests passed in 411.57 seconds. The
  installed-wheel pitch smoke passed separately in 112.04 seconds; a focused PPO
  warning-order regression also passed. The installed smoke ran outside Git and
  verified actual training, reload identity, evaluation, and repeatable traces.
- After the metadata validation change, the final reader also revalidated four
  persisted trained artifacts and seven saved smoke results (28 records), with
  exact round-trips and all summary formats. The earlier PPO evaluation existed
  only in memory; its artifact inventory was rechecked and its evaluation remains
  covered by the training tests.
- Ruff lint, formatting for all 118 source/test/example files, `uv lock --check`,
  and `git diff --check` passed.
- Fresh wheel and sdist passed **58 installation checks** outside the checkout,
  with `PYTHONPATH` and `PYTHONHOME` removed. Both base environments have Python
  3.12.3, NumPy 2.5.3, and Gymnasium 1.3.0, with neither Torch nor SB3. Classical
  evaluation, traced noisy execution, all ordinary summary formats, historical
  reading, and missing-dependency errors work. The only console entry point is
  `harpy`; the retired module is absent. All 24 packaged documentation links
  resolve, and package bytes match the reviewed source.
- Both Python examples in the canonical guide ran unchanged against the fresh
  base wheel outside the checkout. The custom adapter produced 12/12 successful
  records with no invalid actions or truncations; traced output and both saved
  results validated without Torch or SB3.
- All **38,200 saved records** in the representation study, robustness archive,
  and confirmation archive still load with unchanged canonical documents and
  archive hashes. The saved reference qualification still passes; this is
  revalidation of existing evidence, not a newly trained model or new cohort.

On the retained 14,400-record representation study, the performance increment
reduced text-summary time from **10.43 to 0.173 seconds** and result loading from
23.50 to 8.55 seconds. Text, Markdown, CSV, and the serialized document stayed
identical. These are single local measurements under shared workload, not
platform guarantees. The benchmark isolates the reporting optimization; the
subsequent metadata-validation change was verified separately.

## Preservation and evidence locations

The 40 protected files containing frozen study source, verification evidence,
protocol membership, and packaged reference content are unchanged. All 11 website
asset/data/provenance files retain their hashes. The relocated exporter reproduced
the compact data exactly; the external renderer reproduced all eight PNG/SVG
files byte-for-byte in the existing plotting environment.

The local publication handoff is `/home/haydenw/Projects/Harpy-publication-draft`.
Its Blog copy, renderer, layout notes, and assets remain available. Docs links
point to the canonical package guide, and reproduction commands reflect the new
exporter location. Its 55 relative links resolve locally; they still need public,
version-pinned destinations before publication. No website was built.

Local review evidence is under `outputs/release-audit-20260909/`:

- `source-before.zip`, `source-before.json`, `status-before.txt`: exact starting state.
- `preservation.json`, `source-after.json`, `website-extraction.json`: change and preservation inventory.
- `runtime-benchmark.json`, `provenance-compatibility.json`: performance and retained-archive checks.
- `legacy-cli-verification.json`, test logs/XML: command and training verification.
- `test-coverage.json`, `final-quality.json`, `review.patch`: reconciled current
  test coverage, final static checks, and the cleanup diff against the initial
  dirty snapshot. The separately inventoried website relocation is omitted from
  that text patch.
- `installed-verification/verification.json`: all distribution checks and package hashes.
- `website-drafts-before-extraction/`: complete original website folder preserved before relocation.

## Remaining limits and next work

Historical codecs and artifact/trainer internals remain substantial. Some are
shared with the current runner or needed to read saved experiments and serialized
PPO models. Removing them wholesale would risk compatibility for little immediate
user value. A future separation of read-only codecs from historical execution
deserves a dedicated compatibility review.

This review adds no new scientific-performance claim. Strong-noise failures of the
learned reference remain in the record. CPU smoke checks are separate from the
previous CUDA training evidence; no new CUDA run or native Windows/macOS
qualification was performed. The optional training checks reused the existing
qualified dependency stack; only the base wheel/sdist installs were fresh.

Before external release, choose the revised release identifier and publish the
exact revision through a separate authorized action. Full ordinary experiment
archives are still local; publishing those with checksums is necessary for readers
to reproduce the complete study from a fresh clone. Automated CI for the fast
contract checks and distribution workflows is worthwhile next; this repository
still has no CI workflow. Website implementation and hosting remain separate work.
