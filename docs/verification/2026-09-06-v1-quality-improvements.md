# Harpy v1 quality improvements — 6 September 2026

The release worktree is now a more usable headless research toolkit. The installed-package training blocker is fixed, ordinary single-model exploration is separated from formal scientific eligibility, saved results have readable summaries, and traces can carry exact model provenance. The renderer is faster without changing benchmark evidence. No historical model, report, actor semantics, success threshold, or evaluation suite was replaced.

Work was performed in `william/headless-research-release`, based on `bdfd147`, preserving the user's pending headless retirement, packaging, license, and notebook changes. No commit, publication, deployment, or new scientific cohort was made. The initial findings remain available in the [pre-implementation review](../reviews/2026-09-05-v1-release-review.md).

## Implemented increments

| Increment | User value | Compatibility and scientific boundary |
| --- | --- | --- |
| Package-content provenance | All three trainers can run from an installed wheel without Git or a checkout lockfile. | A distinct `package_snapshot` source records actual package contents and the matching installed version. It never claims Git cleanliness or scientific eligibility. Existing Git provenance bytes remain unchanged; old readers reject the new source variant. |
| Shared provenance capture | Removes duplicated checkout discovery and prevents a wheel in `.venv` inheriting an enclosing repository's identity. | CPU/E.1 eligibility checks require genuine Git source provenance. |
| Explicit `--exploratory` inspection | One complete pitch model can be measured or diagnosed with an arbitrary seed and package/modified/CUDA training provenance. | Distinct result schemas remain explicitly ineligible; strict artifact validation and matched baselines are retained. Formal cohort gates are unchanged. |
| `summarize REPORT` | Saved evaluation and diagnostic evidence becomes text or Markdown with statuses, failed gates, results, and failure hotspots. | Owning codecs validate evidence before display. No model files, Torch, or SB3 are needed. |
| `run --with-provenance` | Portable traces identify the exact manifest/model, source, training seed, and devices. | Existing human traces and `run --json` bytes remain available unchanged. |
| CLI help and feedback | Commands explain their purpose and constraints; stderr reports workflow start/completion and output locations. | JSON stdout remains machine-readable. Trust warnings precede model-loading workflows. |
| Finite analysis inputs | NaN/Inf in the selected capture produce an actionable error instead of an invalid observation. | Older samples outside the selected capture remain irrelevant. Normal benchmark evidence is unchanged. |
| Constant envelope spans | IDLE/SUSTAIN output is filled directly instead of branching once per sample. | Attack/decay/release arithmetic and exact audio/spectrum output are preserved. |
| Core cleanup | Removes GUI-only preview/inverse helpers and the ineffective `RenderConfig.block_frames` setting. | External callers of these undocumented leftovers must update; the current renderer and documented patch API remain. |
| Research-oriented documentation | README starts with an experiment and interpretation; a separate guide retains formal commands and history. | Current model limitations and failed scientific criteria are prominent. No UI is added. |

The source-quality change does add a small results layer, because people need to interpret and share the evidence Harpy generates. It does not add dependencies, a plugin framework, hosted infrastructure, or a new UI. Core cleanup removed a net 220 lines across eight files. Broader artifact/codec consolidation remains a future maintenance task, rather than a risky rewrite accompanying release fixes.

## Verification

- Initial broad baseline: **11,949 passed, one failed, one deselected**, 438.99 seconds. The failure was a date-dependent fixture. The fixture now supplies a consistent fixed clock; the production completion-time invariant remains intact.
- Core verification: **10,125 passed**, including the exhaustive real-renderer check of **all 9,801 reachable coordinates**. Original/revised envelope lifecycle behavior was bit-identical across five curvatures; representative/extreme candidate audio and spectra were also byte-identical.
- Artifact/provenance increment: **303 focused tests passed**, plus focused package/checkpoint and distribution-version checks. Package checkpoints stay ineligible after strict reload, and forged eligibility is rejected.
- Workflow/trace increment: **128 tests passed**, plus 12 final focused checks. Independent review found a baseline-role labeling hole in the new exploratory codec; it was fixed and regression-tested before the final build.
- CLI/readout increment: **93 focused tests passed**, including strict report validation, failure-path warning order, legacy output preservation, and blocked Torch/SB3 imports.
- Real training smoke: **three passed, zero skipped**, 274.44 seconds. The pitch workflow builds and installs the wheel outside Git, then trains, reloads, diagnoses, evaluates, and produces deterministic traces. BC/PPO cross their real persistence/evaluation paths.
- Final broad regression: **11,376 passed, one deselected**, 253.99 seconds. The exhaustive renderer integration test was verified separately above. The lower case count reflects removal of tests for the retired GUI-only inverse/preview helpers; renderer curve and transition coverage remains. The shell wrapper reported an exit-status interpolation error after pytest's successful summary; the recorded pytest run contains no failures.
- Final checks: Ruff lint passed, all **119 Python files** passed formatting, `uv lock --check` passed, and `git diff --check` passed.
- README synthesis and Gym examples executed: approximately 440.0388 Hz measured for the requested 440 Hz sine; the documented Spectrum Peak episode submitted successfully.
- Saved E.1 evidence was strict-decoded into a Markdown readout. An existing seed-2 model produced a provenance-bearing seed-123 trace with 42 actions, success, and two-cent final error. These are bounded checks of existing artifacts, not a fresh scientific cohort.

Local alternating 11-run median timings:

| Operation | Original | Revised |
| --- | ---: | ---: |
| 262,144 sustained frames | 20.21 ms | 0.184 ms |
| Complete candidate rendering | 47.09 ms | 26.99 ms |

The complete candidate render used approximately **43% less time** in this local probe. This is not a portable throughput guarantee or a timing claim about concurrent final verification jobs.

## Built package and realistic installed acceptance

A fresh source distribution and wheel were built under `outputs/v1-quality-20260906/dist`, preserving the pre-existing `dist` files. The wheel was built from the source distribution and checked byte for byte against all current Python source files before installation outside Git.

| File | SHA-256 |
| --- | --- |
| `harpy_audio-0.1.0-py3-none-any.whl` | `c315bf2e05526d4b009a2ef2ff551b5a34bd785ab223016b40c8bd7c99d3074c` |
| `harpy_audio-0.1.0.tar.gz` | `ae6994a67fd1753bca419b09e4a4761bf7137ced4a544d62160732e00b173c14` |

Final installed-wheel command acceptance: **all 16 commands passed** outside Git. Pitch estimation, BC, and PPO each completed a real smoke training run, reloaded for a provenance-bearing episode, evaluated, and produced a Markdown report. The installed pitch artifact also completed explicit exploratory evaluation and diagnosis with readable summaries. All three artifacts record `package_snapshot` provenance, distribution version `0.1.0`, and scientific ineligibility.

The exact command ledger and durations are in `outputs/v1-quality-20260906/installed-wheel-verification.json`. Stdout/stderr, fresh model artifacts, reports, Markdown readouts, and `verify-installed-wheel.py` are retained beside it. These local outputs are ignored by Git. Installation reused the available optional dependencies; it did not independently resolve a fresh training dependency stack from a public package index. The retained E.1 Markdown readout presents existing scientific evidence, separately from these new smoke artifacts.

## Remaining limits and next work

- The learned pitch policy still misses its recorded reliability criterion. Rare impossible estimates and repeated-state loops warrant a new feasibility-aware policy experiment. The preserved E/E.1 actor and evidence were not silently modified or selectively rerun.
- This remains a clean procedural single-sine research task. No claim is made about recorded, noisy, chordal, or polyphonic audio.
- `summarize` supports learned evaluation/diagnostic reports and the new result envelopes; baseline-only `harpy-sine-gym` JSON remains inspectable with ordinary JSON tools.
- Feedback is at workflow start/completion, not per epoch. Interrupted create-only runs must restart under a fresh path; resumable training was not added.
- The historical v1/v2 evaluation report formats retain their old shape. New exploratory results and optional run envelopes carry provenance; existing reports are not rewritten.
- Native Windows/macOS execution and fresh CUDA training were not qualified. Scientific CPU/CUDA cohorts were not retrained. Validation used the existing Linux/WSL Python 3.12 environment with its installed training stack.
- Further maintenance should consolidate identical codec/publication primitives behind explicit version-specific adapters. A generic plugin system or distributed service remains unjustified by the current problem size.

The application now better serves a researcher who wants to run, inspect, explain, and reproduce a bounded experiment. Dependable learned tuning remains a separate scientific objective, not a release-quality claim supplied by passing engineering tests.
