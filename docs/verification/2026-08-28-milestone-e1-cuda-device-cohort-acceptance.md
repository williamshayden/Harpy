# Milestone E.1 CUDA device cohort acceptance evidence

Date exercised: 2026-08-28

## Acceptance status

Engineering status: `engineering_passed = true`. The prospective E.1 protocol admitted
one fresh, clean, homogeneous three-seed CUDA cohort; all three artifacts strict-loaded;
the canonical CPU evaluation and IID diagnostic completed; and the full learning test
suite plus static gates passed.

Scientific status: `eligible = true`, `criterion_met = false`, status
`criterion_not_met`. The failed gates are exactly `iid_submitted_success` and
`iid_truncations`.

This is a valid scientific miss, not an execution failure. CUDA training produced very
strong spectral-coordinate estimates and one flawless seed, but two seeds retained
rare catastrophic aliases that caused repeated-state loops and exhausted the 64-action
budget. The frozen release-quality criterion is therefore not met.

## Historical boundary and nonclaims

Milestone E.1 is additive. It does not change, reinterpret, or promote the original
Milestone E CPU result. The exploratory CUDA artifact trained before the E.1 protocol
was not reused. The fresh CUDA artifacts remain individually schema-v2 `ineligible`, as
required by their frozen manifests; E.1 eligibility exists only for the exact schema-v3
cohort. The model architecture, data, deterministic suites, actions, observations,
reward, evaluator, and acceptance thresholds were not changed after observing results.

No seed was selected or discarded. The canonical result is the first complete final
evaluation of the preregistered ordered seeds `(0, 1, 2)`.

### Frozen CPU and legacy regression ledger

The original Milestone E CPU report remains a 3,551,239-byte canonical schema-v2
document with SHA-256
`0f53da04713adc69cb20f3b455c9550cc3bfa2661f506e085cfe373c6a84afb8`.
It strict-decodes as `PitchEvaluationReport`, re-encodes byte-identically, and retains
its original conclusion: eligible, `criterion_not_met`, with failed gates
`iid_submitted_success` and `iid_truncations`. Its CPU manifest SHA-256 values are:

- Seed 0: `b11036b79f89be85346c48361c465fdb6c5d9097b80664af71987e39619ea155`.
- Seed 1: `a81b15e0bdf1e0f52ff954a05ae2d3234d4b1b6a02e789e21836fcb7bacfd3b2`.
- Seed 2: `ce5b9925725b8cab6ad0f3db0d20f24c4f6b315ccdcc5b3cbb692e22509df552`.

Real frozen CPU preflight revalidated their inventories, semantic smoke evidence, and
model payloads in seed order. All remain CPU-trained, CPU-evaluated, and individually
schema-v2 eligible at source commit
`edd0a6b0572c3d45277938f6d3aad0124314f69c` with the same dependency lock.

The schema-v1 codec modules and schema-v2 report codec have no source diff between that
pre-E.1 commit and the E.1 implementation. `pitch_artifacts.py` changed additively for
E.1 runtime/cohort preflight without changing schema-v2 manifest fields or canonical
bytes. Sixteen focused legacy-byte, artifact, report, trace, CPU-dispatch, and
schema-routing regressions passed in two independent groups of eight. The exact current
schema-v1 aggregate-report and trace fixture hashes are
`b638edadedc8a088b4304f4d79bf13f5893ea87c68724a58e35e699cffa255b4` and
`ba713f0fb5e93013b5c6d1648c4eadf761650a1e7970d944a698be50b5b2cf15`.
There was no earlier standalone Milestone E hash ledger, so these checks establish
source immutability and current canonical bytes without claiming an unavailable
before/after filesystem comparison.

## Exact source, lock, and homogeneous runtime

| Field | Exact value |
| --- | --- |
| Cohort and evaluator source commit | `116548792516778b34ac1cc8a12bb4a889611901` (`feat: admit deterministic cuda pitch cohorts`) |
| Source status | Clean tracked tree; `dirty_tree = false`; `required_inputs_committed = true`; empty tracked-diff SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `uv.lock` | SHA-256 `4bad0c09763801f665d6249760e854991ed537d5375cf5cbeeae910ea8d564d7`; 55,000 bytes |
| Compatibility SHA-256 | `092f3fa97545e07397685cf7686d972a9a00fe6dffd4151b68d53c8b3b8121cc` |
| Python / platform / processor | 3.12.3; `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39`; `x86_64` |
| NumPy / Gymnasium | 2.5.1 / 1.3.0 |
| PyTorch / Stable-Baselines3 | `2.13.0+cu130` / 2.9.0 |
| Training device | CUDA |
| Device | `NVIDIA RTX 500 Ada Generation Laptop GPU; compute capability 8.9; 16 SMs` |
| CUDA runtime / driver | 13.0 / 596.08 |
| Internal and final evaluation device | CPU |

Training set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA probing, seeded Python,
NumPy, Torch, and all CUDA devices, enabled deterministic Torch algorithms, disabled
cuDNN benchmarking, enabled deterministic cuDNN, disabled both TF32 paths, and used
float32 without AMP. Data loading used the recorded CPU shuffle generator and zero
workers.

This record is committed after the evidence run. That later documentation commit does
not change the exact source identity above. Any evaluator or required-input change
requires a fresh cohort.

## Engineering gates and repeatability

The implementation commit passed:

| Gate | Exact result |
| --- | --- |
| `uv lock --check` | Passed |
| Focused E.1/CUDA tests | 259 passed during implementation review; final strict-codec subset 16 passed |
| Full stable learning suite | 1,048 passed in 1,100.10 s pytest time / 1,111.73 s wall |
| Repository-wide suite | 12,396 passed in 1,382.80 s pytest time / 1,424.27 s wall; exit 0; 1,236,484 KiB max RSS |
| `uv run ruff check .` | Passed |
| `uv run ruff format --check .` | Final post-record check: 148 files already formatted |
| `git diff --check` | Passed |
| Recovery-anchor range check | Passed against `9bc0fffb6c46486e9ac81811ef81c3ad451fc357` |
| Dependency-light import | Passed; Torch, Stable-Baselines3, and Qt remained unloaded |
| Console and module `--help` without train group | Passed |
| Schema-v1/v2 regression | Original 3,551,239-byte CPU report strict-decoded and re-encoded byte-identically; original CPU triple preflight passed |

The exact captured final-learning/static/lazy commands were:

```bash
/usr/bin/time -f '\nPYTEST_ELAPSED_SECONDS=%e\nPYTEST_EXIT_STATUS=%x' uv run --locked pytest -q tests/learning --durations=25
/usr/bin/time -v uv run pytest -q
/usr/bin/time -f '\nLOCK_ELAPSED_SECONDS=%e\nLOCK_EXIT_STATUS=%x' uv lock --check
/usr/bin/time -f '\nRUFF_CHECK_ELAPSED_SECONDS=%e\nRUFF_CHECK_EXIT_STATUS=%x' uv run --locked ruff check .
/usr/bin/time -f '\nRUFF_FORMAT_ELAPSED_SECONDS=%e\nRUFF_FORMAT_EXIT_STATUS=%x' uv run --locked ruff format --check .
/usr/bin/time -f '\nDIFF_ELAPSED_SECONDS=%e\nDIFF_EXIT_STATUS=%x' git diff --check
/usr/bin/time -f '\nANCHOR_DIFF_ELAPSED_SECONDS=%e\nANCHOR_DIFF_EXIT_STATUS=%x' git diff --check 9bc0fffb6c46486e9ac81811ef81c3ad451fc357
/usr/bin/time -f '\nIMPORT_ELAPSED_SECONDS=%e\nIMPORT_EXIT_STATUS=%x' uv run --isolated --no-group train --locked python -c "import sys, harpy.learning; assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules; assert not any(n == 'PySide6' or n.startswith('PySide6.') for n in sys.modules)"
/usr/bin/time -f '\nCONSOLE_HELP_ELAPSED_SECONDS=%e\nCONSOLE_HELP_EXIT_STATUS=%x' uv run --isolated --no-group train --locked harpy-sine-learn --help
/usr/bin/time -f '\nMODULE_HELP_ELAPSED_SECONDS=%e\nMODULE_HELP_EXIT_STATUS=%x' uv run --isolated --no-group train --locked python -m harpy.learning.cli --help
if pgrep -af '[/]harpy-sine-learn train-pitch'; then exit 1; fi
```

The repository-wide command emitted the existing two-line PipeWire symbol diagnostic
and one Qt mouse-grab diagnostic; no test warning or failure was introduced by E.1.
The final post-record lock, Ruff, format, staged/unstaged diff, lazy import/help, and
no-trainer checks all exited 0.

Two fresh CUDA smoke trainings at seed 17 took 89.53 s and 87.24 s. Every model
state tensor was bitwise identical. Training summaries matched after excluding wall
time, and CPU base/probe episode and metric content matched after excluding the
inherited `training_wall_time_seconds` field. The outer `model.pt` SHA-256 differed
because PyTorch's zip writer embedded a random temporary root name; after stripping
that packaging-only root, every archive entry, tensor payload, and serialization
identifier was byte-identical. This passes scientific repeatability while recording
the timing and container-byte nuances honestly.

## Fresh CUDA checkpoint artifacts

Operator preflight observed each output path absent before its command; the writer then
enforced create-only semantics. Training was serial, and no artifact path was removed,
resumed, reused, or overwritten. Training and validation rates below are persisted raw
fractions.

```text
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 0 --output runs/milestone-e1-pitch-cuda-0 --device cuda
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 1 --output runs/milestone-e1-pitch-cuda-1 --device cuda
uv run --locked harpy-sine-learn train-pitch --profile checkpoint --seed 2 --output runs/milestone-e1-pitch-cuda-2 --device cuda
```

| Seed | Exit / elapsed / max RSS | Selected epoch / recorded training | Train within 5 / MAE / loss | Validation within 5 / MAE / loss | Persisted smoke success / truncations | Manifest SHA-256 |
| ---: | --- | --- | --- | --- | --- | --- |
| 0 | 0 / 2:11.56 / 1,458,780 KiB | 50 / 15.947194005006168 s | 0.9992857142857143 / 5.372857142857143¢ / 0.014142366110376576 | 0.995 / 30.605¢ / 0.03227419693844695 | 45/50 / 4 | `4eb08e56392fecf8881582f538e7bacd27e838d049991e51ebda556cd7ed0772` |
| 1 | 0 / 2:06.12 / 1,457,812 KiB | 50 / 14.701000834000297 s | 1.0 / 1.2014285714285715¢ / 0.006057306881718724 | 0.995 / 30.605¢ / 0.017871731512652787 | 47/50 / 2 | `e6917a34540bed01b32fa83adef67622cf9f145db0e59556e62826f1d62a8436` |
| 2 | 0 / 2:07.19 / 1,458,284 KiB | 50 / 15.564025227999082 s | 1.0 / 1.2014285714285715¢ / 0.006728444239685522 | 1.0 / 1.2¢ / 0.01733659161447271 | 50/50 / 0 | `39ff7e30c13960b17c057a4991716000e0296181c6889cdd99c7d2ae5011ca85` |

Every artifact reported `complete`, had the exact six-file closed inventory, passed all
payload hashes, strict-loaded on CPU, and shared the source, lock, compatibility, and
runtime identities above. Each schema-v2 manifest truthfully retained
`eligible_for_aggregate = false` and status `ineligible`; no v2 field was rewritten.

The seed-0 and seed-1 persisted smoke rows each also contained one submitted failure,
so the displayed smoke truncation counts are not their entire failure counts. Smoke is
engineering evidence only and does not enter the criterion.

### Closed artifact inventories

The manifest is completion metadata and the other five files are its closed payload
inventory. Sizes and SHA-256 values were rechecked before cohort admission.

| Seed | File | Bytes | SHA-256 |
| ---: | --- | ---: | --- |
| 0 | `manifest.json` | 3,008 | `4eb08e56392fecf8881582f538e7bacd27e838d049991e51ebda556cd7ed0772` |
| 0 | `training-config.json` | 1,459 | `1cf7272af1dedd934959062af8017029535c3dd1b171b9022f0599052273ce4b` |
| 0 | `training-summary.json` | 24,916 | `7eaa04f94775323b0d3c8fea4a21f919cfa7c2366d103c6c1f93566c0e6d22a0` |
| 0 | `model.pt` | 13,445 | `85b227ae3a3936d5c5ebb580e8441e275c29dffd1e998bb92679840806c5ad45` |
| 0 | `evaluation-smoke.json` | 82,876 | `4444c48af89ab2aa538c5db40c78a349f224e0797c77e923a44ab3311229232a` |
| 0 | `evaluation-smoke-probes.json` | 34,006 | `7cb2552078b49b28fc2c039284b9bcc04fc1b56f46afd9a4feadcd305f231c3d` |
| 1 | `manifest.json` | 3,008 | `e6917a34540bed01b32fa83adef67622cf9f145db0e59556e62826f1d62a8436` |
| 1 | `training-config.json` | 1,459 | `24b70cac8f5ed0a6647fbcc802b6db405284370f52d94409f380b3e6d9c729d1` |
| 1 | `training-summary.json` | 24,232 | `bb023e9b3a5b81835a6b30c174be232520f24f64874c656ec36997b04a77afa8` |
| 1 | `model.pt` | 13,445 | `110393cc81fa7dbbd314a9f478380b8a8d30f969a8174276235c1b95de18be50` |
| 1 | `evaluation-smoke.json` | 82,863 | `0a959c67875c7d3a71ed0062cbb67c842c7471102714bb5fb714325e5b5653f9` |
| 1 | `evaluation-smoke-probes.json` | 34,007 | `790642e321fb77959b0ca63d826212f021db432127f9a8d2239eb4e3689dcb2f` |
| 2 | `manifest.json` | 3,008 | `39ff7e30c13960b17c057a4991716000e0296181c6889cdd99c7d2ae5011ca85` |
| 2 | `training-config.json` | 1,459 | `8aee9f38c4aa316cb119441ee59e3beb2eaf56f64e463705d4e073e1041833d8` |
| 2 | `training-summary.json` | 23,883 | `e17db805183fcae0c4042662b11d84ba7c99819434b00ccc14345cc0147a1575` |
| 2 | `model.pt` | 13,445 | `5aa8c8dd0df4eacef64dd427c8b426f1cc314fe5f8275ea9d86408f0b656a145` |
| 2 | `evaluation-smoke.json` | 82,820 | `31492c0b9e9e8c1a06f232f2e78d52c0cb94f1f64b5c8e9875c676669ca980ad` |
| 2 | `evaluation-smoke-probes.json` | 34,007 | `df1ed37722341e69d7cbec9e28593fe43fd9911f5cf86498d526b3633154b5aa` |

## Cohort admission and canonical final evidence

Strict E.1 preflight admitted exact seeds `(0, 1, 2)` only after matching all artifact
manifests, payloads, source, lock, compatibility, complete lifecycle, CPU internal
evaluation, CUDA runtime, driver, device description, and current clean evaluator
source. The cohort identity is:

| Field | Exact value |
| --- | --- |
| Schema | 3 |
| Protocol | `harpy-sine-pitch-e1-homogeneous-device-cohort-v1` |
| Training / evaluation | CUDA / CPU |
| Cohort digest SHA-256 | `812d809071ff5505c8311ef603f18b80e23186b6e40626a376897b9432dfc53a` |

Immediately before final evidence, the exact check
`test ! -e runs/milestone-e1-pitch-cuda-report.json && test ! -e
runs/milestone-e1-pitch-cuda-iid-diagnostics.json` exited 0 while `git status` showed
the clean implementation branch. The final commands then ran once:

```text
uv run --locked harpy-sine-learn evaluate runs/milestone-e1-pitch-cuda-0 runs/milestone-e1-pitch-cuda-1 runs/milestone-e1-pitch-cuda-2 --output runs/milestone-e1-pitch-cuda-report.json --device cpu
uv run --locked harpy-sine-learn diagnose runs/milestone-e1-pitch-cuda-0 runs/milestone-e1-pitch-cuda-1 runs/milestone-e1-pitch-cuda-2 --suite iid --output runs/milestone-e1-pitch-cuda-iid-diagnostics.json --device cpu
```

| Evidence | Exit / elapsed / max RSS | Size | SHA-256 | Strict result |
| --- | --- | ---: | --- | --- |
| `runs/milestone-e1-pitch-cuda-report.json` | 0 / 5:26.17 / 696,788 KiB | 3,553,998 bytes | `6fea51ed2f9d24e8e3cc5cbdcab7903e689355fdef92222c2634b0bc7fe850bc` | Schema-v3 load passed; canonical re-encoding byte-identical |
| `runs/milestone-e1-pitch-cuda-iid-diagnostics.json` | 0 / 2:32.16 / 696,992 KiB | 10,240,601 bytes | `97bc1efe205679ae37985b9b913cb0f82193d15bd92c83d10997b0b10b2d940e` | Strict load passed for all 750 episodes; canonical re-encoding byte-identical |

Each command emitted the expected warning exactly once: `warning: learned model
archives are trusted-local artifacts; do not load files from untrusted sources`. The
artifacts are trusted local outputs and must not be loaded from untrusted sources.

## Scientific criterion summary

| Gate | Frozen threshold | Seed 0 | Seed 1 | Seed 2 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| IID coordinate within 5¢ | >=99% each | 99.7506% | 100% | 100% | Pass |
| Combined OOD coordinate within 5¢ | >=95% each | 99.75% | 99.75% | 100% | Pass |
| Lower OOD coordinate within 5¢ | >=90% each | 100% | 100% | 100% | Pass |
| Upper OOD coordinate within 5¢ | >=90% each | 99.5% | 99.5% | 100% | Pass |
| IID submitted success | >=95% each | **94.0%** | 96.8% | 100% | **Fail** |
| IID bound-blocked actions | 0 each | 0 | 0 | 0 | Pass |
| IID truncations | 0 each | **15** | **8** | 0 | **Fail** |
| Zero-spectrum success | Learned margin >=50 points | 0%; +94.0 | 0%; +96.8 | 0%; +100 | Pass |
| Shuffled-spectrum success | Learned margin >=50 points | 0%; +94.0 | 0%; +96.8 | 0%; +100 | Pass |
| Register-OOD submitted success | >=80% each | 93.75% | 97.25% | 100% | Pass |
| Median register-OOD success | >=90% | \- | 97.25% | \- | Pass |
| IID successful excess actions | <=4.0 each | 2.8255 | 2.8306 | 2.8360 | Pass |

The top-level criterion document is authoritative. It records
`failed_gates = ["iid_submitted_success", "iid_truncations"]` and does not average a
weak seed away.

## Direct-coordinate quality and tail

The persisted partition metrics are:

| Seed | Partition / N | Within 1¢ | Within 5¢ | MAE | p50 / p90 / p95 / p99 | Max |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 0 | IID / 401 | 0.6084788029925187 | 0.9975062344139651 | 14.907730673316708 | 1 / 2 / 2 / 2 | 5,502 |
| 0 | Lower OOD / 200 | 0.6 | 1.0 | 1.2 | 1 / 2 / 2 / 2 | 2 |
| 0 | Upper OOD / 200 | 0.6 | 0.995 | 30.775 | 1 / 2 / 2 / 2 | 5,917 |
| 1 | IID / 401 | 0.6084788029925187 | 1.0 | 1.1920199501246882 | 1 / 2 / 2 / 2 | 2 |
| 1 | Lower OOD / 200 | 0.6 | 1.0 | 1.2 | 1 / 2 / 2 / 2 | 2 |
| 1 | Upper OOD / 200 | 0.6 | 0.995 | 30.775 | 1 / 2 / 2 / 2 | 5,917 |
| 2 | IID / 401 | 0.6084788029925187 | 1.0 | 1.1920199501246882 | 1 / 2 / 2 / 2 | 2 |
| 2 | Lower OOD / 200 | 0.6 | 1.0 | 1.2 | 1 / 2 / 2 / 2 | 2 |
| 2 | Upper OOD / 200 | 0.6 | 1.0 | 1.2 | 1 / 2 / 2 / 2 | 2 |

Exact values below are derived with nearest-rank quantiles from all 801 persisted
direct-coordinate records per seed. The tails explain why ordinary accuracy and
end-to-end reliability can disagree.

| Seed | Mean | p50 | p95 | p99 | p99.5 | p99.9 | Max | Cells >5¢ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 15.447¢ | 1 | 2 | 2 | 2 | 5,917 | 5,917 | 2 |
| 1 | 8.581¢ | 1 | 2 | 2 | 2 | 5,917 | 5,917 | 1 |
| 2 | 1.196¢ | 1 | 2 | 2 | 2 | 2 | 2 | 0 |

At least 99.5% of every seed's coordinate cells are within 2¢. The exact catastrophic
cells are:

- Seed 0 IID: true 6,602¢, predicted 1,100¢, error -5,502¢.
- Seed 0 OOD-upper: true 7,017¢, predicted 1,100¢, error -5,917¢.
- Seed 1 OOD-upper: true 7,017¢, predicted 1,100¢, error -5,917¢.
- Seed 2 has no direct-coordinate error above 2¢.

Thus typical perception is excellent, but the current training objective permits a
tiny seed-dependent aliasing tail.

## End-to-end quality

| Suite | Seed | Success | Within 5¢ | Truncations | Submitted failures | Mean / median error | Mean actions | Successful excess |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| IID, 250 | 0 | 235/250 (94.0%) | 94.0% | 15 | 0 | 47.556¢ / 2¢ | 29.296 | 2.8255 |
| IID, 250 | 1 | 242/250 (96.8%) | 96.8% | 8 | 0 | 21.388¢ / 2¢ | 28.340 | 2.8306 |
| IID, 250 | 2 | 250/250 (100%) | 100% | 0 | 0 | 1.980¢ / 2¢ | 27.324 | 2.8360 |
| OOD, 400 | 0 | 375/400 (93.75%) | 94.5% | 22 | 3 | 40.800¢ / 2¢ | 30.8325 | 2.8907 |
| OOD, 400 | 1 | 389/400 (97.25%) | 97.25% | 9 | 2 | 15.8125¢ / 2¢ | 29.765 | 2.8869 |
| OOD, 400 | 2 | 400/400 (100%) | 100% | 0 | 0 | 1.960¢ / 2¢ | 29.4025 | 2.8900 |

All learned rows had zero invalid actions. Both IID perturbation probes had 0% success
for all three seeds, demonstrating that the actor depends on the spectrum rather than
memorizing an action sequence.

### Complete 27-row terminal ledger

`IID`, `L`, and `U` identify the frozen IID, lower-OOD, and upper-OOD suites. Learned
rows use trainer `pitch`, parameter count 2,497, and 1,400 training examples. `null` is
the persisted JSON null. Rates are raw fractions.

- `IID`: `harpy-sine-pitch-e-iid-v1`, digest
  `5b91c98d269a7e9b7319e8827b9eea74a89ab76cabf4a0478746ec3601ee73f3`.
- `L`: `harpy-sine-pitch-e-ood-lower-v1`, digest
  `c8a04e8270e5aa54c17731cd522e0e02b7fa80fa745e5a4c6802480891e73340`.
- `U`: `harpy-sine-pitch-e-ood-upper-v1`, digest
  `1faa402fc3fdebf6a17c758a0647455c8bc6fab129613b919faf6084d0b4da0d`.

| Row | N | Mean return | Submit success | Submit <=1¢ | Final <=1¢ | Final <=5¢ | Final MAE | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pitch-0 / IID | 250 | 0.989524964590164 | 0.94 | 0.02 | 0.02 | 0.94 | 47.556 | 2.0 |
| pitch-0 / L | 200 | 1.0390309475409838 | 0.93 | 0.04 | 0.04 | 0.94 | 12.53 | 2.0 |
| pitch-0 / U | 200 | 1.0616335975409839 | 0.945 | 0.015 | 0.015 | 0.95 | 69.07 | 2.0 |
| pitch-0 / IID:zero | 250 | -1.2716960655737706 | 0.0 | 0.0 | 0.0 | 0.0 | 2374.612 | 2400.5 |
| pitch-0 / IID:shuffled | 250 | -1.103056740983607 | 0.0 | 0.0 | 0.0 | 0.0 | 1342.928 | 1083.5 |
| pitch-1 / IID | 250 | 1.0498246406557377 | 0.968 | 0.02 | 0.02 | 0.968 | 21.388 | 2.0 |
| pitch-1 / L | 200 | 1.1103004639344263 | 0.965 | 0.04 | 0.04 | 0.965 | 4.85 | 2.0 |
| pitch-1 / U | 200 | 1.1385787040983608 | 0.98 | 0.015 | 0.015 | 0.98 | 26.775 | 2.0 |
| pitch-1 / IID:zero | 250 | -1.2716960655737706 | 0.0 | 0.0 | 0.0 | 0.0 | 2374.612 | 2400.5 |
| pitch-1 / IID:shuffled | 250 | -1.1037784524590168 | 0.0 | 0.0 | 0.0 | 0.0 | 1347.328 | 1083.5 |
| pitch-2 / IID | 250 | 1.1170167600000003 | 1.0 | 0.02 | 0.02 | 1.0 | 1.98 | 2.0 |
| pitch-2 / L | 200 | 1.180779043442623 | 1.0 | 0.04 | 0.04 | 1.0 | 1.945 | 2.0 |
| pitch-2 / U | 200 | 1.1826496278688525 | 1.0 | 0.015 | 0.015 | 1.0 | 1.975 | 2.0 |
| pitch-2 / IID:zero | 250 | -1.2716960655737706 | 0.0 | 0.0 | 0.0 | 0.0 | 2374.612 | 2400.5 |
| pitch-2 / IID:shuffled | 250 | -1.102343029508197 | 0.0 | 0.0 | 0.0 | 0.0 | 1338.528 | 1083.5 |
| random / IID | 250 | -1.0752928813114755 | 0.008 | 0.0 | 0.0 | 0.008 | 1273.932 | 1090.0 |
| random / L | 200 | -1.0446602983606557 | 0.005 | 0.005 | 0.005 | 0.005 | 1439.535 | 1357.5 |
| random / U | 200 | -1.0660325024590163 | 0.0 | 0.0 | 0.0 | 0.0 | 1520.26 | 1277.0 |
| reward_search / IID | 250 | 1.1169048386885247 | 1.0 | 1.0 | 1.0 | 1.0 | 0.008 | 0.0 |
| reward_search / L | 200 | 1.1788237081967214 | 1.0 | 0.99 | 0.99 | 1.0 | 0.035 | 0.0 |
| reward_search / U | 200 | 1.1807640696721313 | 1.0 | 1.0 | 1.0 | 1.0 | 0.02 | 0.0 |
| spectrum_peak / IID | 250 | 1.1171144242622952 | 1.0 | 0.584 | 0.584 | 1.0 | 1.252 | 1.0 |
| spectrum_peak / L | 200 | 1.180886442622951 | 1.0 | 0.595 | 0.595 | 1.0 | 1.17 | 1.0 |
| spectrum_peak / U | 200 | 1.1827592663934428 | 1.0 | 0.56 | 0.56 | 1.0 | 1.19 | 1.0 |
| oracle / IID | 250 | 1.1165854478688524 | 1.0 | 0.02 | 0.02 | 1.0 | 4.784 | 5.0 |
| oracle / L | 200 | 1.180351805737705 | 1.0 | 0.04 | 0.04 | 1.0 | 4.725 | 5.0 |
| oracle / U | 200 | 1.1822117147540985 | 1.0 | 0.015 | 0.015 | 1.0 | 4.825 | 5.0 |

| Row | Mean actions | Invalid rate | Truncation rate | Mean successful excess |
| --- | ---: | ---: | ---: | ---: |
| pitch-0 / IID | 29.296 | 0.0 | 0.06 | 2.825531914893617 |
| pitch-0 / L | 30.96 | 0.0 | 0.06 | 2.838709677419355 |
| pitch-0 / U | 30.705 | 0.0 | 0.05 | 2.941798941798942 |
| pitch-0 / IID:zero | 3.0 | 0.0 | 0.0 | null |
| pitch-0 / IID:shuffled | 51.132 | 0.0 | 0.788 | null |
| pitch-1 / IID | 28.34 | 0.0 | 0.032 | 2.830578512396694 |
| pitch-1 / L | 29.945 | 0.0 | 0.025 | 2.844559585492228 |
| pitch-1 / U | 29.585 | 0.0 | 0.02 | 2.9285714285714284 |
| pitch-1 / IID:zero | 3.0 | 0.0 | 0.0 | null |
| pitch-1 / IID:shuffled | 51.172 | 0.0 | 0.788 | null |
| pitch-2 / IID | 27.324 | 0.0 | 0.0 | 2.836 |
| pitch-2 / L | 29.735 | 0.0 | 0.0 | 2.85 |
| pitch-2 / U | 29.07 | 0.0 | 0.0 | 2.93 |
| pitch-2 / IID:zero | 3.0 | 0.0 | 0.0 | null |
| pitch-2 / IID:shuffled | 51.88 | 0.0 | 0.8 | null |
| random / IID | 6.616 | 0.0 | 0.0 | 1.0 |
| random / L | 6.62 | 0.0 | 0.0 | 1.0 |
| random / U | 7.365 | 0.0 | 0.0 | null |
| reward_search / IID | 38.876 | 0.0008231299516411153 | 0.0 | 14.388 |
| reward_search / L | 41.795 | 0.0051441559995214735 | 0.0 | 14.91 |
| reward_search / U | 39.885 | 0.0052651372696502444 | 0.0 | 13.745 |
| spectrum_peak / IID | 29.492 | 0.0 | 0.0 | 5.004 |
| spectrum_peak / L | 31.7 | 0.0 | 0.0 | 4.815 |
| spectrum_peak / U | 30.975 | 0.0 | 0.0 | 4.835 |
| oracle / IID | 24.488 | 0.0 | 0.0 | 0.0 |
| oracle / L | 26.885 | 0.0 | 0.0 | 0.0 |
| oracle / U | 26.14 | 0.0 | 0.0 | 0.0 |

### Complete seven-row combined-OOD ledger

Each aggregate is the exact ordered union of 200 lower- and 200 upper-OOD episodes.

| Actor | Seed | N | Mean return | Submit success | Submit <=1¢ | Final <=1¢ | Final <=5¢ | MAE | Median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pitch-0 | 0 | 400 | 1.0503322725409838 | 0.9375 | 0.0275 | 0.0275 | 0.945 | 40.8 | 2.0 |
| pitch-1 | 1 | 400 | 1.1244395840163937 | 0.9725 | 0.0275 | 0.0275 | 0.9725 | 15.8125 | 2.0 |
| pitch-2 | 2 | 400 | 1.181714335655738 | 1.0 | 0.0275 | 0.0275 | 1.0 | 1.96 | 2.0 |
| random | null | 400 | -1.055346400409836 | 0.0025 | 0.0025 | 0.0025 | 0.0025 | 1479.8975 | 1324.0 |
| reward_search | null | 400 | 1.1797938889344264 | 1.0 | 0.995 | 0.995 | 1.0 | 0.0275 | 0.0 |
| spectrum_peak | null | 400 | 1.181822854508197 | 1.0 | 0.5775 | 0.5775 | 1.0 | 1.18 | 1.0 |
| oracle | null | 400 | 1.1812817602459018 | 1.0 | 0.0275 | 0.0275 | 1.0 | 4.775 | 5.0 |

| Actor | Seed | Mean actions | Invalid rate | Truncation rate | Mean successful excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| pitch-0 | 0 | 30.8325 | 0.0 | 0.055 | 2.8906666666666667 |
| pitch-1 | 1 | 29.765 | 0.0 | 0.0225 | 2.886889460154242 |
| pitch-2 | 2 | 29.4025 | 0.0 | 0.0 | 2.89 |
| random | null | 6.9925 | 0.0 | 0.0 | 1.0 |
| reward_search | null | 40.84 | 0.005203232125367287 | 0.0 | 14.3275 |
| spectrum_peak | null | 31.3375 | 0.0 | 0.0 | 4.825 |
| oracle | null | 26.5125 | 0.0 | 0.0 | 0.0 |

## Matched baselines

The strongest learned seed is shown because it demonstrates attainable behavior, not
because it replaces the three-seed criterion.

| IID actor | Success | Within 1¢ | Mean error | Mean actions | Excess | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned seed 2 | 100% | 2.0% | 1.980¢ | 27.324 | 2.836 | 0% |
| Spectrum Peak | 100% | 58.4% | 1.252¢ | 29.492 | 5.004 | 0% |
| Oracle | 100% | 2.0% | 4.784¢ | 24.488 | 0 | 0% |
| Reward Search | 100% | 100% | 0.008¢ | 38.876 | 14.388 | 0% |

| Combined-OOD actor | Success | Within 1¢ | Mean error | Mean actions | Excess | Truncation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned seed 2 | 100% | 2.75% | 1.960¢ | 29.4025 | 2.890 | 0% |
| Spectrum Peak | 100% | 57.75% | 1.180¢ | 31.3375 | 4.825 | 0% |
| Oracle | 100% | 2.75% | 4.775¢ | 26.5125 | 0 | 0% |
| Reward Search | 100% | 99.5% | 0.0275¢ | 40.840 | 14.3275 | 0% |

When it behaves correctly, learned seed 2 saves 2.168 IID actions and 1.935 OOD
actions versus Spectrum Peak, and about 11.5 actions versus Reward Search. Spectrum
Peak remains more robust across seeds and more precise than the best learned model.

## Diagnostic root cause

Every canonical IID submission was successful. All 23 IID failures were 64-step budget
exhaustions, and each occurred in an episode with a repeated-state loop.

| Seed | Submitted success / exhausted | Decisions | Canonical / shortest accuracy | Loop / unrecoverable episodes | Repeated visits | Loop transitions | Bound blocks |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 235 / 15 | 7,324 | 85.6499% | 15 / 15 | 626 | 641 | 0 |
| 1 | 242 / 8 | 7,085 | 87.5794% | 8 / 8 | 351 | 359 | 0 |
| 2 | 250 / 0 | 6,831 | 89.6355% | 0 / 0 | 0 | 0 | 0 |

Successful episodes for all three seeds had maximum estimator error 2¢. In contrast,
all 367 seed-0 and all 196 seed-1 estimator errors above 5¢ occurred inside failed
episodes; every one was at least 1,000¢ and the maximum was 8,317¢. The isolated
holes collapse to the minimum output class, 1,100¢. Shared examples include true
coordinates 6,983¢ and 7,017¢, plus their octave-shifted states. Seed 0 also contains
the 6,602¢ hole; seed 2 contains none.

The universal failure sequence is an accurate adjacent coordinate, a step into an
isolated collapse coordinate, a coarse action driven by the false 1,100¢ estimate,
recovery on another adjacent coordinate, and an action back into the same hole. The
stateless controller repeats that cycle until the budget expires. Continuing to adjust
while already inside the environment's +/-5¢ success region contributes to only five
of seed 0's 15 failures and none of seed 1's eight; early submission alone therefore
cannot fix the cohort. A representative seed-1 cycle, 18¢ and then 17¢ below its
7,000¢ target, is:

```text
6982 (accurate) -> Cent Up -> 6983 (estimated 1100)
-> Octave Up -> 8183 (estimated 1100)
-> Cent Down -> 8182 (accurate) -> Octave Down -> 6982
```

Seed 0's loops first appeared at steps 9..35; seed 1's at steps 9..34. Seed 1's eight
failures are a strict subset of seed 0's 15. Seed 2 showed no repeated state, loop
transition, unrecoverable episode, truncation, or failed submission.

The mechanism is therefore localized: rare estimator aliases are amplified by
zero-tolerance stateless replanning and the lack of a loop escape. The evidence does not
support changing WSL, CUDA, the Gym contract, reward, or broad spectrum representation
as the next intervention.

## Independent review and conclusion

The four required independent read-only lanes were exercised separately:

| Independent review, 2026-08-28 | Outcome |
| --- | --- |
| `E1-R1` — original schema-v1/v2 and CPU evidence immutability | Pass. Schema-v1 codecs and the schema-v2 report codec had no E.1 diff; additive artifact-preflight changes preserved v2 fields/bytes; 16 focused byte/routing tests passed; the real CPU trio strict-preflighted; its report round-tripped byte-identically with its original verdict. |
| `E1-R2` — CUDA determinism and homogeneous provenance | Pass. Exact device/runtime/source/lock equality, deterministic settings, strict three-seed admission, and normalized seed-17 repeatability were verified. |
| `E1-R3` — scientific split, evaluator, and criterion equivalence | Pass. Frozen suite digests, CPU evaluator, complete evidence rows, re-derived criterion, and no threshold/model/data substitution were verified; no P0-P2 finding remained. |
| `E1-R4` — artifact, codec, CLI, and create-only boundaries | Pass after fixes. Source equality moved before model/evidence access; schema-v3 owns raw evidence and its re-derived criterion; boolean/integer type confusion was closed; closed dispatch and create-only failures were covered by focused tests. |

Implementation, regression, and science reviews found no remaining P0-P2 issue after
those fixes. The final full learning suite and static checks passed.

Milestone E.1 establishes that deterministic WSL CUDA training is supported and that
the learned estimator can generalize extremely well. It does **not** establish reliable
three-seed closed-loop tuning or satisfy the frozen v1 release-quality criterion.

The next experiment should be a new, explicitly versioned tail-reliability checkpoint.
Preserve this E.1 result unchanged, use these diagnostics to preregister the smallest
intervention that removes estimator collapse cells and/or escapes repeated-state loops,
and generate fresh artifacts. Do not assume early submission is sufficient, relax E.1
thresholds, or rerun/select seeds to change this verdict.
