# Harpy Milestone E.1 CUDA Device Cohort Implementation Plan

**Goal:** Admit one fresh, deterministic CUDA-trained seed cohort under an additive
schema-v3 report protocol while preserving every schema-v1/v2 artifact, report,
criterion, and historical conclusion.

**Contract:**
`docs/superpowers/specs/2026-08-28-harpy-milestone-e1-cuda-device-cohort-design.md`.
The original Milestone E design remains authoritative for the environment, estimator,
data, suites, evaluator, quality thresholds, and artifact semantics.

**Method:** Use RED → observed failure → minimal GREEN → refactor. Keep each
change dependency-light and schema-first. Do not edit existing CPU evidence or reuse
any exploratory CUDA artifact.

## Frozen boundaries

- Do not change the Gym contract, actor, estimator architecture, coordinate splits,
  suites, hyperparameters, baselines, probes, diagnostics, or scientific thresholds.
- Do not change schema-v2 manifest eligibility. CUDA schema-v2 artifacts stay
  individually `ineligible`; only the schema-v3 cohort report owns E.1 eligibility.
- Do not alter existing schema-v1/v2 canonical bytes or default CPU report dispatch.
- Train on CUDA; run internal smoke evaluation, final evaluation, and final
  diagnostics on CPU.
- Require exact fresh seeds `0`, `1`, and `2` from one clean cohort/evaluator source
  commit, dependency lock, compatibility digest, and runtime cohort. Acceptance
  records that exact commit; a later evaluator-code commit requires a fresh reviewed
  cohort.

## Task 1: Pin the additive protocol and regression boundary

**Files:**

- Create `src/harpy/learning/pitch_e1.py`
- Extend pitch report/workflow codec dispatch only where schema `3` is required
- Add focused E.1 codec and workflow tests

**RED contracts:**

- Pin schema version `3` and protocol ID
  `harpy-sine-pitch-e1-homogeneous-device-cohort-v1`.
- Pin the exact report fields, ordered seed provenance, canonical cohort-digest
  preimage, schema-v3-only raw evidence, and top-level re-derived E.1 criterion.
- Mutate every field independently and prove source/runtime/device/manifest/
  compatibility/cohort/evidence inconsistencies fail decode.
- Freeze representative schema-v1 and schema-v2 artifacts and reports and prove
  load/re-encode bytes remain identical.

**GREEN:** Add a separate schema-v3 report model and codec. Reuse schema-v2 raw-field
codecs and semantic validators without serializing a detachable eligible schema-v2
report or changing existing registries and artifact semantics.

## Task 2: Enforce deterministic CUDA training

**Files:**

- Extend `src/harpy/learning/dependencies.py`
- Extend `src/harpy/learning/pitch.py`
- Extend runtime capture in `src/harpy/learning/pitch_artifacts.py`
- Add focused dependency, training, and runtime tests

**RED contracts:**

- Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any CUDA runtime interaction, replacing
  a conflicting caller value deterministically.
- Seed Python, NumPy, Torch CPU, and all CUDA RNGs from the run seed.
- Require deterministic Torch algorithms, deterministic cuDNN, disabled cuDNN
  benchmarking, and disabled matmul/cuDNN TF32.
- Prove training uses float32 without autocast, `GradScaler`, or AMP.
- Preserve the CPU DataLoader generator, deterministic shuffle, ordered validation,
  `num_workers=0`, and finite CPU state-dictionary persistence.
- Capture exact GPU description including compute capability and SM count, CUDA
  runtime, and best-effort driver version without making `nvidia-smi` availability an
  artifact-creation failure.

**GREEN:** Configure CUDA before dependency/device probing and record the existing
strict runtime fields. Do not add fields to the schema-v2 artifact codec.

## Task 3: Add the E.1 cohort preflight

**Files:**

- Extend `src/harpy/learning/pitch_artifacts.py`
- Add focused artifact-set tests

**RED contracts:**

- Accept exactly three complete schema-v2 pitch checkpoint artifacts with ordered
  seeds `(0, 1, 2)`, CUDA training, CPU internal evaluation, clean committed source,
  and individually unchanged schema-v2 ineligibility.
- Require identical source commit, dependency-lock digest, required-input state,
  schema-v2 compatibility digest, and every runtime identity field.
- Require artifact source to equal the clean evaluator source used for the run,
  excluding every pre-protocol CUDA artifact without inventing a self-referential
  implementation hash.
- Validate inventories, hashes, semantic smoke evidence, and model state before any
  final suite/cache/environment construction.
- Reject CPU input on the E.1 path, mixed devices/runtimes, duplicates, wrong seeds,
  stale or dirty source, incompatible locks, incomplete artifacts, corruption, and
  extra inputs before output creation.

**GREEN:** Factor common schema-v2 loading from the frozen CPU preflight, then apply
separate CPU and E.1 admission predicates. Never mutate a loaded manifest.

## Task 4: Integrate evaluation and closed CLI dispatch

**Files:**

- Extend `src/harpy/learning/workflows.py`
- Extend `src/harpy/learning/cli.py`
- Extend workflow and CLI tests

**RED contracts:**

- Keep CLI grammar unchanged: an eligible CPU trio automatically emits the frozen
  schema-v2 report; a qualifying CUDA trio emits schema v3; mixed/incompatible input
  fails closed.
- Require CPU for E.1 final evaluation and diagnostics. Reject CUDA criterion
  evaluation before output or final-suite construction.
- Run the unchanged direct-coordinate, terminal, baseline, probe, OOD aggregate, and
  criterion calculators exactly once.
- Wrap the complete evidence with ordered per-seed manifest hashes, source/runtime/
  compatibility provenance, clean evaluator source, and re-derived cohort digest.
- Preserve canonical stdout/file equality, trusted-local warnings, create-only paths,
  and existing `0/1/2/130` exit behavior.

**GREEN:** Dispatch from strict preflight result type, not a permissive fallback or a
new user flag. Decode reports schema-first across versions `1`, `2`, and `3`.

## Task 5: Engineering verification

Run focused tests first, then:

```bash
uv lock --check
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
git diff --check
```

Also run:

- dependency-free import and `--help` checks proving Torch/SB3/Qt remain lazy;
- a real CUDA smoke train → CPU persisted reload → evaluate → run cycle;
- deterministic-setting assertions on the real CUDA process;
- canonical schema-v3 encode/decode/re-encode and report-hash checks;
- regression fixtures for original schema-v1/v2 reports and CPU dispatch;
- negative mutation tests for every cohort admission field and final evidence field;
  and
- a check that the existing exploratory CUDA seed-0 artifact is rejected by E.1.

No final E-suite run occurs during these tests.

## Task 6: Train the fresh CUDA cohort

After implementation, tests, and review are committed, confirm the worktree is clean
and train exactly once per seed into new create-only directories:

```bash
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 0 \
  --device cuda --output runs/milestone-e1-pitch-cuda-0
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 1 \
  --device cuda --output runs/milestone-e1-pitch-cuda-1
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 2 \
  --device cuda --output runs/milestone-e1-pitch-cuda-2
```

Before final evaluation, strict-load all three artifacts and record:

- exact source commit and dependency-lock SHA-256;
- per-seed manifest SHA-256 and closed inventory hashes;
- compatibility SHA-256;
- Python/library/GPU/compute-capability/SM/CUDA-runtime/driver identity;
- selected epoch, training time, and training/validation metrics; and
- the complete deterministic CUDA settings.

Do not substitute or copy the earlier exploratory CUDA run. If any seed fails or the
runtime changes, discard that attempted cohort as scientific evidence and start a new
three-seed cohort in new directories.

## Task 7: Run final CPU evidence once

With the exact admitted CUDA trio, run final evaluation once on CPU:

```bash
uv run --locked harpy-sine-learn evaluate \
  runs/milestone-e1-pitch-cuda-0 \
  runs/milestone-e1-pitch-cuda-1 \
  runs/milestone-e1-pitch-cuda-2 \
  --device cpu --output runs/milestone-e1-pitch-cuda-report.json
```

Then run the final IID diagnostic bundle once on CPU. Record the schema-v3 report
SHA-256, cohort digest, diagnostic SHA-256, all per-seed direct and rollout metrics,
all 27 rows, seven OOD aggregates, probe margins, truncations, bound blocks, excess
actions, failed gates, and diagnostic root causes.

Recompute the unchanged criterion from persisted records. `criterion_not_met` is a
valid completed result and must not trigger a threshold, model, split, or artifact
rewrite.

## Task 8: Acceptance and independent review

Create
`docs/verification/2026-08-28-milestone-e1-cuda-device-cohort-acceptance.md` without
editing the Milestone E acceptance document. It must record exact commands, runtime,
source/lock/manifest/cohort/report/diagnostic hashes, metrics, warnings, and the honest
criterion outcome.

Obtain independent reviews for:

1. original schema-v1/v2 and CPU evidence immutability;
2. CUDA determinism and homogeneous-cohort provenance;
3. scientific split/evaluation/criterion equivalence; and
4. artifact, codec, CLI, and create-only failure boundaries.

E.1 is complete only when the tracked worktree is clean, no trainer remains running,
all three fresh artifacts and both final evidence files strict-load, the acceptance
document reports the actual result, and no retrospective promotion occurred.
