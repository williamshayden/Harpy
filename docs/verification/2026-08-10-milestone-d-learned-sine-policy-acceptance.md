# Milestone D learned sine policy acceptance evidence

Date exercised: 2026-08-21

## Acceptance status

Engineering status: `engineering_passed` for the clean committed smoke workflow.

BC scientific status: `pending` for the Task 11 checkpoint; the smoke result is
explicitly `ineligible`.

PPO scientific status: `pending` for the Task 11 five-seed checkpoint; the smoke
result is explicitly `ineligible`.

This document separates smoke-profile engineering evidence from checkpoint-profile
scientific evidence. A successful smoke run is never a scientific result, and a valid
checkpoint that reports `criterion_not_met` is not an engineering failure.

## Commit, lock, and runtime

| Field | Exact value |
| --- | --- |
| Source commit | `7506bfed116afc7c449e7e1351ff009dd104a077` (`docs: add learned sine policy workflow`) |
| Source status | Clean tracked tree; empty tracked-diff SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; required inputs committed |
| `uv.lock` SHA-256 | `4bad0c09763801f665d6249760e854991ed537d5375cf5cbeeae910ea8d564d7` (55,000 bytes) |
| Python | 3.12.3 |
| Platform | `Linux-5.15.146.1-microsoft-standard-WSL2-x86_64-with-glibc2.39` |
| Processor / CPU | `x86_64` |
| NumPy | 2.5.1 |
| Gymnasium | 1.3.0 |
| PyTorch | `2.13.0+cu130` |
| Stable-Baselines3 | 2.9.0 |
| Training device | CPU for both artifacts |
| Evaluation device | CPU for both internal artifact evaluations and the combined report |
| Torch/thread settings | 11 intra-op threads; 11 inter-op threads; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` unset |
| CUDA availability/runtime/driver | Unavailable; Torch CUDA runtime 13.0; host driver version 12080 was reported too old; manifest CUDA runtime/driver fields are `null` |

## Full automated gates

All commands must run serially. The training smoke is accepted only when the optional
stack is installed and the result contains no skip.

| Command | Exit | Elapsed | Exact observed result / warnings |
| --- | ---: | ---: | --- |
| `uv sync --locked --group train` | 0 | 0.04 s timing confirmation | `Resolved 48 packages in 0.94ms`; `Checked 48 packages in 7ms` |
| `uv run pytest tests/learning/test_training_smoke.py -q -rs` | 0 | 357.83 s (0:05:57) | `2 passed`; no skip; output log SHA-256 `19fdb10e0b60599d9ae89efe9394e70674b406aebc12ae32f0f8ee292b5cb062` (110 bytes) |
| No-skip check over training-smoke output | 0 | 0.13 s | `rg -q "skipped"` found no match |
| `uv run pytest` | 0 | 1,252.98 s (0:20:52) | `11886 passed`; known PipeWire symbol and Qt mouse-grab diagnostics only |
| `uv run ruff check .` | 0 | 0.28 s | `All checks passed!` |
| `uv run ruff format --check .` | 0 | 0.05 s | `122 files already formatted` |
| `git diff --check` | 0 | 0.00 s | Zero output before this evidence edit |

## BC smoke

| Field | Exact observed value |
| --- | --- |
| Command | `uv run harpy-sine-learn train-bc --profile smoke --seed 0 --output runs/milestone-d-bc-smoke` |
| Exit / elapsed | 0 / 122.53 s |
| Artifact path | `/home/haydenw/Projects/Harpy/runs/milestone-d-bc-smoke` |
| Profile / seed / device | `smoke` / 0 / CPU |
| Configured training episodes | 128 |
| Configured validation episodes | 64 |
| Split digests | training `ac43951b718486402500191dcbee7b41ef7844dfe16fbd8fb727611cba6a0833`; validation `d52d911cde6b941ff329f0531ae168d2867ef06739f7ce93a4cc9439b55414f0` |
| Trajectory-derived training examples | 2,947 |
| Trajectory-derived validation examples | 1,476 |
| Epochs / selected epoch | 2 / 2; epoch-2 validation loss `1.9369362247551867`, validation accuracy `0.42005420054200543` |
| Training wall time | `2.9488152419944527` s |
| Parameter count | 75,687 |
| Persisted reload / evaluation / completion | Persisted `model.pt` reloaded on CPU while manifest was incomplete; 32 base + 32 zero-spectrum + 32 shuffled-spectrum episodes evaluated; exact inventory validated; manifest completed last and strict public reload passed |
| Smoke base row | 32 episodes; submitted ≤5¢ `0.0`; submitted ≤1¢ `0.0`; final ≤5¢ `0.0`; final ≤1¢ `0.0`; mean/median absolute error `718.65625`/`645.0` cents; mean actions `64.0`; successful excess `null`; mean return `-0.999328524590164`; truncation `1.0`; invalid-action rate `0.0` |
| Criterion eligibility | `false`; status `ineligible`; no scientific values computed |
| Warning | One Torch warning reported host CUDA driver version 12080 too old; CPU training/evaluation remained selected and completed |

## PPO smoke

| Field | Exact observed value |
| --- | --- |
| Command | `uv run harpy-sine-learn train-ppo --profile smoke --seed 0 --output runs/milestone-d-ppo-smoke` |
| Exit / elapsed | 0 / 99.55 s |
| Artifact path | `/home/haydenw/Projects/Harpy/runs/milestone-d-ppo-smoke` |
| Profile / seed / device | `smoke` / 0 / CPU |
| Requested environment steps | 2,048 |
| Completed environment steps | 2,048 |
| Training wall time | `91.25147013299284` s |
| Parameter count | 75,816 |
| Persisted reload / evaluation / completion | Persisted `model.zip` reloaded on CPU while manifest was incomplete; 32 base + 32 zero-spectrum + 32 shuffled-spectrum episodes evaluated; exact inventory validated; manifest completed last and strict public reload passed |
| Smoke base row | 32 episodes; submitted ≤5¢ `0.0`; submitted ≤1¢ `0.0`; final ≤5¢ `0.0`; final ≤1¢ `0.0`; mean/median absolute error `726.65625`/`669.5` cents; mean actions `1.0`; successful excess `null`; mean return `-1.0`; truncation `0.0`; invalid-action rate `0.0` |
| Criterion eligibility | `false`; status `ineligible`; no scientific values computed |
| Warning | One Torch warning reported host CUDA driver version 12080 too old; CPU training/evaluation remained selected and completed |
| Human trace | Exit 0 / 3.65 s; full-range seed 123 target C2; action `Submit`; `submitted_failure`; 1,638-cent final error; 1 action; return `-1` |
| Combined evaluation | `uv run harpy-sine-learn evaluate runs/milestone-d-bc-smoke runs/milestone-d-ppo-smoke --output runs/milestone-d-smoke-report.json`; exit 0 / 187.34 s; 10 distinct rows; strict finite canonical JSON; trusted-local warning once |
| Deterministic JSON traces | Two fresh `run ... --seed 123 --json` commands exited 0 in 3.27 s and 3.23 s; trusted-local warning once each; `cmp` exit 0 with zero output; both files are byte-identical |

## Checkpoint artifacts

Task 11 owns these clean CPU checkpoint runs. Every field remains explicitly pending
until those six artifacts have been trained and strictly reloaded.

| Artifact | Seed | Exit / elapsed | Examples or steps | Complete / eligible | Manifest SHA-256 |
| --- | ---: | --- | ---: | --- | --- |
| BC | 0 | `pending` | `pending` | `pending` | `pending` |
| PPO | 0 | `pending` | `pending` | `pending` | `pending` |
| PPO | 1 | `pending` | `pending` | `pending` | `pending` |
| PPO | 2 | `pending` | `pending` | `pending` | `pending` |
| PPO | 3 | `pending` | `pending` | `pending` | `pending` |
| PPO | 4 | `pending` | `pending` | `pending` | `pending` |

## IID rows

The canonical combined IID rows, paired terminal records, and all exact metrics are
`pending` until Task 11. The final table must retain BC, PPO seeds 0–4, and the four
baselines as distinct rows.

| Actor | Submitted ≤5¢ | Submitted ≤1¢ | Final ≤5¢ | Final ≤1¢ | Mean / median abs. error | Mean actions / successful excess | Mean return | Truncation / invalid action |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| BC seed 0 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| PPO seed 0 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| PPO seed 1 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| PPO seed 2 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| PPO seed 3 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| PPO seed 4 | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| Random | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| Reward Search | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| Spectrum Peak | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |
| Oracle | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` | `pending` |

## Register-OOD rows

Register-OOD is informational and is not a criterion gate. Combined checkpoint rows
and paired terminal records are `pending` until Task 11.

### Lower OOD subset

All BC, PPO seed 0–4, Random, Reward Search, Spectrum Peak, and Oracle lower-band rows:
`pending`.

### Upper OOD subset

All BC, PPO seed 0–4, Random, Reward Search, Spectrum Peak, and Oracle upper-band rows:
`pending`.

### Combined OOD subset

All BC, PPO seed 0–4, Random, Reward Search, Spectrum Peak, and Oracle combined rows:
`pending`.

## Four baselines

### Random

Matched smoke row: 32 episodes; submitted ≤5¢ `0.0`; submitted ≤1¢ `0.0`;
final ≤5¢ `0.0`; final ≤1¢ `0.0`; mean/median absolute error
`1228.46875`/`991.0` cents; mean actions `9.3125`; successful excess `null`; mean
return `-1.0823474692622952`; truncation `0.0`; invalid-action rate `0.0`.
Matched IID/register-OOD checkpoint evidence: `pending`.

### Reward Search

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `1.0`;
final ≤5¢ `1.0`; final ≤1¢ `1.0`; mean/median absolute error `0.0`/`0.0` cents;
mean actions `37.375`; successful excess `13.84375`; mean return
`1.118760225409836`; truncation `0.0`; invalid-action rate `0.0`.
Matched IID/register-OOD checkpoint evidence: `pending`.

### Spectrum Peak

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `0.5625`;
final ≤5¢ `1.0`; final ≤1¢ `0.5625`; mean/median absolute error
`1.21875`/`1.0` cents; mean actions `27.625`; successful excess `4.09375`; mean
return `1.118657930327869`; truncation `0.0`; invalid-action rate `0.0`.
Matched IID/register-OOD checkpoint evidence: `pending`. Spectrum Peak is a
classical clean-sine control; PPO is not required to beat it.

### Oracle

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `0.03125`;
final ≤5¢ `1.0`; final ≤1¢ `0.03125`; mean/median absolute error `4.625`/`5.0`
cents; mean actions `23.53125`; successful excess `0.0`; mean return
`1.1181404661885246`; truncation `0.0`; invalid-action rate `0.0`.
Matched IID/register-OOD checkpoint evidence: `pending`. Oracle validates
reachability and planning, not learned perception.

## Zero-spectrum and shuffled-spectrum probes

| Actor / seed | Suite | Base submitted success | Zero-spectrum submitted success | Shuffled-spectrum submitted success | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| BC smoke seed 0 | smoke | `0.0` | `0.0` | `0.0` | Both probe rows exactly matched the base aggregate: 718.65625 mean error, 64.0 actions, -0.999328524590164 return, and 1.0 truncation. This smoke diagnostic showed no aggregate sensitivity. |
| PPO smoke seed 0 | smoke | `0.0` | `0.0` | `0.0` | Both probe rows exactly matched the base aggregate: 726.65625 mean error, 1.0 action, -1.0 return, and 0.0 truncation. This smoke diagnostic showed no aggregate sensitivity. |
| BC checkpoint seed 0 | IID | `pending` | `pending` | `pending` | `pending` |
| PPO checkpoint seed 0 | IID | `pending` | `pending` | `pending` | `pending` |
| PPO checkpoint seed 1 | IID | `pending` | `pending` | `pending` | `pending` |
| PPO checkpoint seed 2 | IID | `pending` | `pending` | `pending` | `pending` |
| PPO checkpoint seed 3 | IID | `pending` | `pending` | `pending` | `pending` |
| PPO checkpoint seed 4 | IID | `pending` | `pending` | `pending` | `pending` |

These probes measure policy sensitivity only. They do not establish that spectral
evidence was causally necessary.

## BC criterion

| Field | Exact value |
| --- | --- |
| Eligibility | Smoke: `false`; checkpoint: `pending` |
| Held-out IID next-action accuracy | Smoke: `null` by design; checkpoint: `pending` |
| IID submitted success | Smoke: `null` by design; checkpoint: `pending` |
| Thresholds | accuracy ≥ 0.90 and submitted success ≥ 0.75 |
| `criterion_met` | Smoke: `null`; checkpoint: `pending` |
| Status | Smoke: `ineligible`; checkpoint: `pending` |

## PPO criterion

| Field | Exact value |
| --- | --- |
| Eligibility | Smoke: `false`; checkpoint aggregate: `pending` |
| Declared seeds | 0, 1, 2, 3, 4 |
| Median IID submitted success | Smoke: `null` by design; checkpoint: `pending` |
| Seeds strictly beating paired Random | Smoke: `null` by design; checkpoint: `pending` |
| Thresholds | median success ≥ 0.50 and at least 4/5 seeds strictly beat Random |
| `criterion_met` | Smoke: `null`; checkpoint: `pending` |
| Status | Smoke: `ineligible`; checkpoint: `pending` |

## Payload hashes

Every manifest-declared smoke payload was size- and SHA-256-verified by the strict
public loader. The manifest itself is listed separately because schema v1 deliberately
does not self-hash.

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `runs/milestone-d-bc-smoke/manifest.json` | 2,109 | `b40fa8a13cbf5ad5cf227804f6abc28717c57ff1e7f63860c8db3f2c28ba986d` |
| `runs/milestone-d-bc-smoke/training-config.json` | 929 | `b906b2a41c0c04920152d8cff0550408319e3af1d4500a560cea420342f10a9a` |
| `runs/milestone-d-bc-smoke/training-summary.json` | 456 | `68b637ae33c915db09a6d0bcd6ee9e837ef4456bc93cb3befa2d8a52c806f7df` |
| `runs/milestone-d-bc-smoke/model.pt` | 308,167 | `f608c32f246070b44d947465ccbfbf4a733ade271bd41b97b1ca8fcd0a4a693c` |
| `runs/milestone-d-bc-smoke/evaluation-smoke.json` | 33,578 | `d5ce9d1eda5f48c1b7a0db8c7aad38429f12789d993c7ad1c368541d4ac5c710` |
| `runs/milestone-d-ppo-smoke/manifest.json` | 2,058 | `e552079229d994fd68e5f3b29813e277b73ed57fdc8e996883dd2a8d527bfd50` |
| `runs/milestone-d-ppo-smoke/training-config.json` | 799 | `db44753daf819b6dbc6d2e862d00f62f0718cebc6df972d403e5203345bc7564` |
| `runs/milestone-d-ppo-smoke/training-summary.json` | 193 | `5f3d178ceaf913d48a0d4fdfecb575e43774818c0f2cd6662fb0eebbb3903ef4` |
| `runs/milestone-d-ppo-smoke/model.zip` | 980,437 | `8e430c7de119cf4abd0d321054030fcbaf375eecc69b591bcd6fae6c84484868` |
| `runs/milestone-d-ppo-smoke/evaluation-smoke.json` | 32,102 | `b585dad4629e60658ae3b649eb9265cf308a5d40f112b039f35825ad4cfe8e64` |
| `runs/milestone-d-smoke-report.json` | 109,569 | `8ed9166dfee91a0734ab4142007dee4a9fd63912cbbe59b37fa20e0052eba3b7` |
| `runs/milestone-d-run-1.json` | 351 | `c70105e32ab9bfa4ff41d2c326a3d982b6a6417e80ca16edb7758438683d9bb0` |
| `runs/milestone-d-run-2.json` | 351 | `c70105e32ab9bfa4ff41d2c326a3d982b6a6417e80ca16edb7758438683d9bb0` |
| `/tmp/harpy-milestone-d-training-smoke.txt` | 110 | `19fdb10e0b60599d9ae89efe9394e70674b406aebc12ae32f0f8ee292b5cb062` |

Checkpoint BC/PPO manifests, every manifest-declared payload, and the canonical
checkpoint report: `pending` until Task 11.

## Concerns and pending items

- Fresh clean-commit smoke gates and all six CLI invocations passed. The smoke
  artifacts/report/traces live under ignored `runs/` and were not staged.
- Both tiny smoke learned policies submitted successfully on 0/32 base episodes. BC
  exhausted all 64 actions on every episode; PPO submitted immediately after one
  action on every episode. This is an honest weak smoke outcome, but the profile is
  execution-only and ineligible for any scientific conclusion.
- CPU runs emitted a Torch warning because the installed `+cu130` build found host
  driver version 12080 too old. CUDA availability was false; both manifests record
  CPU training and CPU evaluation, and all commands completed.
- Checkpoint BC seed 0 and PPO seeds 0–4: `pending` for Task 11.
- IID, lower/upper/combined register-OOD, baseline, and probe results: `pending` for
  Task 11.
- Whole-branch independent review verdicts: `pending` for Task 11.
- Learned model files are trusted-local Torch/SB3 artifacts and are not safe to load
  from untrusted sources.
- Candidate spectrum is continuously present in `Harpy/SinePitch-v0`. Milestone D
  adds no analysis tools and makes no claim about recorded-audio pitch shifting,
  chords, or waveform generalization.
