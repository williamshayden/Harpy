# Harpy Milestone E.1: CUDA Device Cohort Design

Date: 2026-08-28

Status: Approved additive contract. This document does not amend or replace the
Milestone E preregistration, its CPU artifacts, its report, or its recorded outcome.

## Outcome

Milestone E.1 permits one rigorously controlled CUDA-trained cohort to answer the same
scientific question as Milestone E. Training moves to CUDA; internal smoke evaluation
and final direct-coordinate, rollout, baseline, probe, and diagnostic evaluation stay
on CPU. The result is a new schema-v3 cohort report wrapped around three immutable
schema-v2 pitch artifacts.

The intended claim remains narrow:

> Learned spectral perception can reliably close the frozen single-sine tuning task
> through Harpy's frozen symbolic controls and planner.

E.1 changes the admissible training device for a new cohort. It does not change the
model, data, environment, evaluator, thresholds, or meaning of success.

## Historical boundary

The following evidence remains immutable historical fact:

- the approved Milestone E design and implementation plan;
- the original clean CPU artifacts for seeds `0`, `1`, and `2`;
- the original schema-v2 Milestone E report and diagnostics;
- the original `criterion_not_met` outcome and its recorded failure analysis; and
- every schema-v1 Milestone D artifact and codec.

Schema-v2 artifact eligibility is not redefined. A CUDA-trained schema-v2 artifact
continues to contain `eligible_for_aggregate = false` and criterion status
`ineligible`, exactly as its manifest declared when it closed. E.1 never edits,
rewrites, upgrades, or relabels that manifest. Eligibility under E.1 belongs only to
the exact three-artifact cohort and only in its schema-v3 report.

Existing exploratory CUDA artifacts are not promoted. In particular, the CUDA seed-0
run made before this protocol cannot be combined with later runs. All three E.1
artifacts must be newly trained from the same clean post-implementation commit in new
create-only output directories.

## Frozen scientific contract

E.1 inherits without modification:

- `Harpy/SinePitch-v0`, its seven actions, observations, reward, bounds, rendering,
  reset behavior, terminal semantics, and 64-action budget;
- the spectrum-only `1→16→16→1` pitch estimator and stateless replanning actor;
- the `harpy-sine-pitch-grid-v1` split, all E-suite memberships and digests, spectrum
  renderer/cache, labels, optimizer, hyperparameters, epoch selection, and seeds;
- the exact final coordinate records, 27 terminal rows, probes, four baselines,
  two-register OOD aggregate, diagnostics, and criterion recomputation; and
- the strict six-file schema-v2 artifact lifecycle and trusted-local model boundary.

No architecture, data, suite, threshold, fallback, observation, action, or reward
change may be folded into E.1. Such a change requires another reviewed protocol.

## CUDA training protocol

CUDA training is eligible only when the finalized implementation applies all of these
controls before the first CUDA runtime operation:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- Python, NumPy, Torch CPU, and all CUDA RNGs seeded from the declared run seed;
- `torch.use_deterministic_algorithms(True)`;
- `torch.backends.cudnn.benchmark = False`;
- `torch.backends.cudnn.deterministic = True`;
- `torch.backends.cuda.matmul.allow_tf32 = False`;
- `torch.backends.cudnn.allow_tf32 = False`;
- ordinary IEEE float32 model, optimizer, loss, and activation computation; and
- no autocast, `GradScaler`, AMP, or other reduced-precision training path.

The existing CPU-seeded DataLoader generator, deterministic shuffle, ordered
validation, and `num_workers=0` rules remain unchanged. Persisted estimator state is
copied to finite CPU tensors before writing. E.1 makes no bit-identity claim between
CPU and CUDA or across different GPUs, drivers, CUDA runtimes, or Torch releases.

## Exact cohort admission

The E.1 preflight accepts exactly three complete schema-v2 `pitch` checkpoint
artifacts and orders them by exact seeds `(0, 1, 2)`. Every artifact must satisfy all
of the following before a model, cache, environment, or final suite is constructed:

- training device is `cuda` and internal evaluation device is `cpu`;
- the schema-v2 manifest remains individually ineligible solely under the frozen
  device policy; profile, trainer, inventory, hashes, smoke evidence, and model state
  are otherwise valid;
- source is clean, all required inputs are committed, and the source commit and
  dependency-lock SHA-256 are identical across the trio;
- source commit exactly matches the clean cohort/evaluator source used for the run,
  and acceptance records that commit; a future evaluator-code commit requires a fresh
  reviewed cohort rather than retrospective reuse;
- all three share one schema-v2 `compatibility_sha256`;
- all three share one exact runtime cohort: Python, platform, processor, NumPy,
  Gymnasium, Torch, Stable-Baselines3, device type and description, CUDA runtime, and
  CUDA driver; and
- each exact six-file inventory, manifest hash, payload size/hash, semantic smoke
  evaluation, and trusted-local model state strict-validates.

Missing, duplicate, extra, mixed-device, mixed-runtime, dirty, stale-source,
wrong-seed, incompatible, incomplete, corrupted, or otherwise ineligible inputs fail
closed before final evidence access or output creation.

## Closed CLI dispatch

CLI grammar does not change. `evaluate` performs closed automatic dispatch:

- an exact eligible CPU seed trio follows the frozen Milestone E preflight and emits
  the existing schema-v2 report;
- an exact fresh CUDA seed trio satisfying E.1 emits the schema-v3 cohort report; and
- every mixed or incompatible set is rejected.

Final criterion evidence requires CPU evaluation, so the required command remains
`evaluate ... --device cpu` (with CPU also the default). CPU is likewise required for
the final E.1 diagnostics. A CUDA evaluator result is exploratory and cannot become
criterion evidence.

## Schema-v3 cohort report

The report has schema version `3` and protocol ID
`harpy-sine-pitch-e1-homogeneous-device-cohort-v1`. It contains exactly:

- `training_device = cuda` and `evaluation_device = cpu`;
- three ordered artifact-provenance records for seeds `0`, `1`, and `2`, each carrying
  the SHA-256 of the exact `manifest.json`, complete schema-v2 source status, complete
  runtime status, and schema-v2 compatibility SHA-256;
- clean evaluator source status;
- `cohort_digest_sha256`; and
- schema-v3-only raw rows, coordinate evaluations, and OOD aggregates using the
  frozen schema-v2 field codecs but carrying no standalone schema-v2 report identity
  or criterion; and
- a top-level E.1 criterion re-derived as eligible from that raw evidence.

The cohort digest is SHA-256 over canonical JSON plus its trailing newline. Its exact
preimage contains the protocol ID, training and evaluation devices, the three ordered
artifact-provenance records, and evaluator source. It excludes the digest itself and
the raw-evidence payload. Each artifact manifest already closes its six-file inventory;
the final report bytes receive a separate SHA-256 recorded in E.1 acceptance.

Decoding rejects duplicate keys, unknown or missing fields, nonfinite values, bad
digests, reordered or missing seeds, non-CUDA training, non-CPU evaluation, runtime or
source disagreement, compatibility disagreement, and any cohort-digest mismatch. It
also strict-decodes the nested raw evidence and re-derives all aggregate metrics and
the top-level E.1 criterion from persisted coordinate and terminal records. The raw
evidence is deliberately not a detachable valid schema-v2 evaluation report.

## Unchanged quality gates

An admitted E.1 cohort meets the criterion only when every original Milestone E gate
holds:

- every seed estimates at least 99% of the 401 IID coordinates within five cents;
- every seed estimates at least 95% of all 400 OOD coordinates within five cents,
  with neither OOD register below 90%;
- every seed has at least 95% submitted success on the 250-episode IID suite;
- every seed has exactly zero IID bound-blocked actions and zero IID truncations;
- each seed's IID success exceeds both zero-spectrum and shuffled-spectrum success by
  at least 50 percentage points;
- median combined register-OOD submitted success is at least 90%, with no seed below
  80%; and
- every seed's mean successful IID excess actions is at most 4.0.

A valid cohort that misses any gate emits `criterion_not_met` with the exact failed
gates. That remains a scientific result, not an engineering or CLI failure. No
threshold is changed after observing CUDA results, and the old CPU verdict is never
recomputed or replaced.

## Acceptance boundary

E.1 is complete only when the deterministic CUDA controls and closed dispatch pass
their full error matrix, three fresh clean CUDA artifacts strict-load, one CPU final
evaluation and one CPU final diagnostic bundle are produced, all hashes and metrics
are recorded in a new E.1 acceptance document, and independent scientific and
artifact reviews confirm that the original Milestone E evidence was preserved.
