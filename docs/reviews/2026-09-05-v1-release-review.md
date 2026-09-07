**Harpy v1 release review — 5 September 2026**

This is the initial, pre-implementation assessment. The subsequent authorized improvements and their verification are recorded in [the implementation evidence](../verification/2026-09-06-v1-quality-improvements.md).

Harpy has a credible, carefully engineered research core. The current release candidate is not ready to distribute as documented because installed-package training fails. Its next improvements should make experiments easier to run, inspect, and extend while preserving the scientific record. A UI is unnecessary for this release.

This review covers `william/headless-research-release` at base commit `bdfd147`, including the existing uncommitted headless changes. It evaluates the product as a deterministic audio-control research package, not a general document generator or a production audio tuner. The proposed personal-site visualization remains separate. Source code and existing experiment artifacts were not changed during this review.

| Area | Assessment | Reason |
| --- | --- | --- |
| Core synthesis, planning, and environment | Green | Small, deterministic components with meaningful separation and strong transition contracts. |
| Scientific evidence and honesty | Green | Raw records support recomputed metrics; failed criteria and negative controls remain visible. |
| Installed researcher workflow | Red | The wheel's three training commands require a source Git checkout and fail outside it. |
| Learned tuning as a dependable feature | Experimental | Seed-dependent catastrophic aliases and loops remain; the declared reliability criterion was missed. |
| CLI, documentation, and result interpretation | Yellow | Usable commands and strict JSON exist, but help, progress, readable reports, and artifact identity are incomplete. |
| Maintainability and extensibility | Yellow | Frozen protocols are defensible; duplicated validation and orchestration make new experiments more expensive than necessary. |
| Scalability for the current benchmark | Green with a clear optimization | Finite data and bounded caches fit a local process; unnecessary per-sample Python work slows cold rendering. |

**Release findings, ordered by consequence**

1. **P1 — Installed users cannot run any documented training workflow.**

   `src/harpy/learning/pitch_artifacts.py:2371` unconditionally captures provenance from `Path(__file__)`. Its implementation at lines 2105–2153 requires the installed Python module to live inside a Git repository and reads that repository's `uv.lock`. BC and PPO do the same at `bc.py:555` and `ppo.py:401`, through `artifacts.py:2163–2184`.

   We extracted the existing wheel into a temporary directory outside any Git repository, imported its code with the available optional training dependencies, and ran the three smoke trainers. `train-pitch`, `train-bc`, and `train-ppo` each exited 1 before training with `git ... rev-parse --show-toplevel` exit 128. None created an artifact. All 44 Python files in that wheel match the current release source byte for byte. The wheel and source-distribution configuration omit `uv.lock`.

   This breaks the wheel-install-to-smoke journey in README lines 70–93. It is not merely a restriction on scientific eligibility. The current source-ancestry mechanism also does not independently establish that installed package bytes correspond to an enclosing checkout.

   **Repair:** represent distribution provenance separately from checkout provenance. Allow installed-package exploratory training with truthful identity and eligibility. Embed versioned build/source metadata where appropriate; do not invent a clean Git status or weaken historical criteria. Add an installed-wheel smoke test outside every Git checkout, including the optional dependency path.

2. **P2 — Evaluation and diagnosis conflate inspecting a model with certifying a fixed cohort.**

   `src/harpy/learning/workflows.py:600–620` rejects evaluation of one pitch checkpoint. Lines 1090–1105 similarly require exactly three checkpoint artifacts with seeds 0, 1, and 2 for non-smoke diagnostics. `pitch_artifacts.py:1964–1994` additionally requires clean, eligible CPU artifacts; the CUDA cohort adds exact evaluator-source equality at lines 2078–2099.

   Those are defensible requirements for the published scientific claim. They are an obstacle to normal research: train one checkpoint, inspect it, change an implementation, inspect the next one. A researcher should not need two more training runs or a clean historical source state to obtain an explicitly exploratory measurement. The internal evaluation machinery can evaluate a single actor, but the supported CLI path excludes it.

   **Repair:** expose single-artifact exploratory evaluation and diagnosis, always clearly ineligible for the frozen cohort criterion. Keep the existing strict cohort acceptance path intact. This separation also provides a practical extension point for the next actor.

3. **P2 — Exported traces and older evaluation formats lose experiment identity.**

   `src/harpy/learning/workflows.py:917–955` loads an artifact but returns an episode trace without retaining its identity. `trace.py:292–318` writes no trace schema version, artifact digest, trainer, training seed, or execution device. Its `seed` is the episode seed. Standard schema-v1 reports (`workflows.py:237–251`) and schema-v2 pitch reports (`pitch_reports.py:730–755`) likewise omit immutable artifact identity and source lineage. E.1's schema-v3 report and diagnostic bundles demonstrate a stronger approach.

   Actual generated trace keys and saved report keys confirm the omission. Renaming or sharing these outputs separates the results from the model that produced them; a later reader cannot reliably reconstruct that connection from `pitch-0` alone.

   **Repair:** add a versioned result envelope containing artifact/manifest digests, actor identity, training seed, evaluation device, and provenance, with the existing evidence document as its payload. Preserve historical codecs and golden bytes rather than silently extending their exact-field contracts.

4. **P2 — Public analysis silently produces invalid observations from NaN or infinite samples.**

   `src/harpy/analysis.py:169–179` checks dimensions and length but not sample finiteness before analysis. With a valid 440 Hz sine and one NaN inside the FFT capture, `analyze()` returned `has_signal=True` and no finite values in its 6,820 spectrum bins. One infinite sample similarly poisoned the spectrum and yielded a spurious approximately 20.51 Hz peak.

   The benchmark spectrum encoder already rejects non-finite input, so this finding concerns the explicitly documented reusable analysis API, not ordinary benchmark output.

   **Repair:** reject non-finite values in the selected analysis capture before signal detection and FFT, with focused NaN/Inf coverage. Numerical failure should be an actionable input error.

5. **P2 — The trusted-model warning comes after loading and execution.**

   `src/harpy/learning/cli.py:218–231` calls the model workflow before emitting the warning. Evaluate and diagnose have the same ordering at lines 194–214 and 169–190. README lines 181–182 promise a warning before loading. A bounded event-order probe confirmed `model workflow` precedes `trusted-local warning`; a load failure can bypass the warning entirely.

   **Repair:** emit the warning once after lightweight preflight and before the model-loading workflow. Check ordering rather than just counting warning messages.

**What the features actually achieve**

The strongest product value is a reproducible experimental workbench: renderer, bounded control task, exact planner, baselines, training controls, and evidence. The learned actors serve research questions. The existing results do not establish that they are generally better tuning tools than the classical baseline.

The E.1 acceptance ledger reports the following matched IID results, each over 250 episodes:

| Actor | Submitted within 5 cents | Truncations | Submitted within 1 cent | Mean absolute final error |
| --- | ---: | ---: | ---: | ---: |
| Learned pitch, seed 0 | 94.0% | 15 | 2.0% | 47.556 cents |
| Learned pitch, seed 1 | 96.8% | 8 | 2.0% | 21.388 cents |
| Learned pitch, seed 2 | 100% | 0 | 2.0% | 1.980 cents |
| Spectrum Peak | 100% | 0 | 58.4% | 1.252 cents |

Seed 2 uses fewer mean actions than Spectrum Peak, 27.324 versus 29.492. This is a real tradeoff, not evidence of across-the-board superiority. Five-cent classification also makes excellent direct pitch localization different from reliable one-cent closed-loop tuning. D's behavior cloning and PPO remain scientifically useful negative controls; their low success rates do not make them dependable tuning features.

The recorded E.1 failures have a concrete cause. `pitch_actor.py:81–98,110–124` takes an unrestricted grid argmax, clips the inferred source error, and executes a fresh plan. An impossible pitch estimate becomes a legal but incorrect action. Raw IID diagnostics contain 367 predictions more than five cents wrong across seed 0's 15 failing episodes and 196 across seed 1's eight failing episodes. Every such prediction is outside the physically feasible range implied by the public controls and known source range, allowing for quantization.

In one seed-0 episode, the true pitch reaches 6,602 cents against a 6,600-cent target. The model predicts 1,100 cents and chooses Octave Up. The run ultimately exhausts its budget 2,602 cents wrong with repeated control states. Two fresh CPU inference probes with existing weights reproduced these aliases. Restricting logits to publicly feasible coordinates changed predictions from 1,100 to 6,600 cents in one case and to 6,985 cents for a true 6,983-cent input in the other.

That supports a small, separately versioned feasibility-aware actor experiment. It does not prove full recovery. The current actor's semantics are expressly frozen, so preserve E/E.1 and test a new protocol instead of silently repairing historical behavior or selecting the successful seed. If v1 is advertised as reliable learned tuning, the recorded criterion miss is a release blocker. If it is explicitly an experimental research package, negative results can ship honestly once the engineering workflow works.

Evidence: `docs/verification/2026-08-28-milestone-e1-cuda-device-cohort-acceptance.md:284–366,450–480`; existing E.1 IID diagnostic JSON and two bounded current-renderer inference probes. The historical full cohort was not retrained or reevaluated in this review.

**Research output usability needs a small, entirely headless product pass**

The current command line is more complete as a machine interface than as a researcher interface. Actual help output has no descriptions for commands or options beyond argparse defaults (`cli.py:65–120`). The training path returns silently (`cli.py:139–166`), and long phases have no progress channel. Evaluation prints the complete canonical JSON even when it also writes an output file. The existing E.1 evaluation file is approximately 3.55 MB and its IID diagnostics approximately 10.24 MB.

Provide a read-only results summary that strict-decodes saved evidence and displays eligibility, failed gates, per-seed outcomes, matched baselines, and a short diagnostic failure table. Markdown export would make these results usable in notebooks, READMEs, and reports without changing their canonical JSON source. Add progress to stderr and a final artifact path and eligibility explanation. Keep stdout suitable for piping.

Reorder the README around a first experiment: install, run, inspect a small example result, understand a failed criterion, and reproduce an experiment. Put current outcomes near the beginning. Move the long milestone document list into a history index, and link the completed E.1 acceptance record. Add one short synthesis example and one Gym actor loop for the public API. These improvements address the user's research/documentation journey without adding UI code.

**Which code earns its place**

Keep the deterministic renderer, exact bounded planner, immutable data ownership, transactional environment state changes, strict patch format, artifact inventory/digest checks, and recomputation of metrics from raw records. These prevent specific failure modes. Keep BC/PPO as historical controls if reproducing that research remains a product commitment; weak scientific outcomes are not dead code.

The exact planner evaluates only 125 octave/semitone combinations and projects the cent control into a feasible interval. It does not need a generic search framework. The learning cache has at most 9,801 coordinate keys and approximately 73.3 MiB of spectrum payloads, excluding mapping overhead, and avoids retaining rendered audio. Neither a database nor distributed infrastructure is justified by the current task.

There is nevertheless concrete residue and repetition to remove:

- `synth/curves.py:18–26,58–76,136–185`: envelope-preview data, inverse UI handle conversion, and preview sampling have no remaining production callers; only tests use them. Remove them if they are not intended as a documented public API. Keep the curve evaluation used by the renderer.
- `synth/models.py:82,88`: `RenderConfig.block_frames` is validated but never consumed anywhere under `src`. Remove it or give it a documented effect before treating that configuration as stable.
- Learning comprises 18,756 lines across 28 modules, out of 21,264 source lines across 44 modules. This is not a defect by itself, but `artifacts.py` has 2,219 lines, `pitch_artifacts.py` 2,516, and `workflows.py` 1,991. Source/runtime capture, publication lifecycle, document primitives, and routing recur across versions.
- A source AST inventory found 15 `_integer` helper definitions across the repository, six `_mapping` helpers, and six `_exact_fields` helpers. Their semantics sometimes differ intentionally. Consolidate identical primitives and give differing policies explicit names; do not mechanically merge every similarly named validator.

Extract shared filesystem publication and basic codec operations, then leave each version's scientific semantics in small explicit adapters. Separate report serialization from execution workflows. Add one narrow exploratory actor/evaluator interface before inventing a plugin architecture. New waveforms or recorded-audio evidence should get new environment/evidence identities rather than weakening the frozen sine contract or extending a cache keyed only by pitch.

**A measured performance improvement**

`synth/envelope.py:82–89` visits every sample in Python even during constant IDLE and SUSTAIN spans. Each real benchmark render includes 262,144 sustained frames (`envs/sine_pitch.py:249–250`). One local profile measured 50 ms for a real reset, including 26 ms in the envelope loop and 4 ms in the FFT. An isolated sustain span measured 23.35 ms with the current loop versus 1.07 ms for a bit-identical constant array fill. These are local measurements, not an end-to-end speedup guarantee.

Batch constant spans into array fills while keeping attack/decay/release arithmetic unchanged. Validate block-partition equivalence and exact spectrum bytes. This should improve direct evaluation and cold cache construction without changing the task. Persistent caches, parallel environments, and batching can follow measured demand.

**Recommended release sequence**

1. Fix installed-package provenance and prove the complete smoke journey from the wheel outside Git. Keep eligibility truthful.
2. Fix non-finite public analysis and warning ordering; make exported runs identifiable with a versioned envelope.
3. Add single-artifact exploratory inspection, meaningful help, stderr progress, and a concise saved-report/Markdown readout. Update the README around those workflows and current outcomes.
4. Remove the identified GUI residue and no-op setting. Consolidate shared artifact/codec mechanics in small changes protected by historical fixtures. Optimize constant envelope spans with exact-output checks.
5. Ship with an explicit research-package promise. Pursue feasibility-aware learned tuning as a fresh experiment if reliable learned tuning remains the v1 feature goal.

This calls for focused refactoring and product completion, not a rewrite. Code correctness is strongest where Harpy models its actual domain; complexity is least justified where milestone-specific orchestration has become the only route through ordinary user workflows.

**Verification record**

- Read-only review split across core architecture, research quality, release usability, and artifact/workflow integration.
- Current release wheel verified against all 44 source Python files. Wheel SHA-256: `2faa647174ef8f12eff32ba8e56f3874987fb65abe21b05dd866f5b7d80dcddd`.
- Lint: `All checks passed!`. Formatting: `113 files already formatted`. `uv lock --check` passed. `git diff --check` passed before this report.
- Current wheel-code smoke preflight reproduced all three installed-package training failures before expensive work. Baseline CLI and inspected help routes executed.
- Focused NaN/Inf analysis probes, warning-order probe, trace-key inspection, source call-site inventory, envelope performance probes, and two existing-model inference probes completed.
- Initial broad regression: 11,949 passed, one failed, and one integration test deselected in 438.99 seconds. The failure mixed a real creation date with a fixed August completion date in a test fixture; the subsequent increment fixes the fixture without relaxing the timestamp invariant. Command: `uv run --no-sync pytest -q -m "not integration" --ignore=tests/learning/test_training_smoke.py --basetemp=/tmp/harpy-v1-review-pytest-20260905`.
- Full training smoke, the exhaustive integration test, fresh scientific cohorts, and native Windows/macOS execution were not run. Historical acceptance results are labeled as existing evidence, not fresh release verification.