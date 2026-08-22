# Milestone D learned sine policy acceptance evidence

Date exercised: smoke workflow 2026-08-21; v2 checkpoint and final verification
2026-08-22

## Acceptance status

Engineering status: `engineering_passed = true`. The clean committed smoke workflow,
all six fixed-source v2 checkpoint artifacts, canonical aggregate, traces, three
independent reviews, and final fresh gates passed.

BC scientific status: eligible checkpoint evidence, `criterion_met = false`, status
`criterion_not_met`.

PPO scientific status: eligible five-seed checkpoint evidence,
`criterion_met = false`, status `criterion_not_met`.

This document separates smoke-profile engineering evidence from checkpoint-profile
scientific evidence. A successful smoke run is never a scientific result, and a valid
checkpoint that reports `criterion_not_met` is not an engineering failure.

## Evidence source, lock, and runtime

| Field | Exact value |
| --- | --- |
| Evidence source commit | `709b316175d8835173f246b2bfcb0937fb06355b` (`fix: close learned artifact completion boundary`) |
| Source status | Clean tracked tree; empty tracked-diff SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`; required inputs committed |
| `uv.lock` SHA-256 | `4bad0c09763801f665d6249760e854991ed537d5375cf5cbeeae910ea8d564d7` (55,000 bytes) |
| Python | 3.12.3 |
| Platform | `Linux-5.15.146.1-microsoft-standard-WSL2-x86_64-with-glibc2.39` |
| Processor / CPU | `x86_64`; Intel Core Ultra 7 155H; 22 logical CPUs |
| NumPy | 2.5.1 |
| Gymnasium | 1.3.0 |
| PyTorch | `2.13.0+cu130` |
| Stable-Baselines3 | 2.9.0 |
| Training device | CPU for all six v2 artifacts |
| Evaluation device | CPU for all six internal artifact evaluations and the combined report |
| Torch/thread settings | 11 intra-op threads; 11 inter-op threads; `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` unset |
| CUDA availability/runtime/driver | Unavailable; one RTX 500 Ada Generation Laptop GPU was inventoried; Torch CUDA runtime 13.0, cuDNN 92000, NVIDIA driver 573.44; Torch reported host driver API 12080 too old; CPU remained authoritative |
| Recovery range | Historical `957c140` is absent in this recovered shallow checkout; approved mechanically available base `8de76841739408b167a1f85c889cf75e555eece8` was used for `8de7684...HEAD` checks and the missing parent was not treated as a cleanliness defect |

## Historical smoke workflow (preserved)

This section retains the committed Task 10 smoke evidence. It proves workflow
execution only and is not used for the eligible v2 scientific criteria.

### Historical smoke automated gates

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

### V1 invalidation and v2 authority

The first clean checkpoint used source commit
`547120c93ce2e121da0358dc11bc9540f46b3358`. Its root
`/home/haydenw/Projects/Harpy/runs/milestone-d-v1` and report
`/home/haydenw/Projects/Harpy/runs/milestone-d-v1-report.json` remain untouched audit
evidence, but they are invalidated and excluded from acceptance. The preserved v1
artifact composite is
`91dd050fa6e134edf9581018c2c3dc1f8ccaa62823fe3e46367caf163dd585cc`;
the 3,499,331-byte v1 report SHA-256 is
`aca21866d9577c95c822f2d39bcf37145ecaefbe47d2d5d0687ee98cd66ee7b9`.

The artifact reviewer found two Important lifecycle defects after v1: an interrupt
just after the real final-manifest rename could propagate failure despite a physically
complete artifact, and BC performed a new fallible reload after the completion commit
instead of returning `writer.complete(...)`. Fix commit
`709b316175d8835173f246b2bfcb0937fb06355b` closed both boundaries and its focused
rereview was `APPROVED/CLEAR`. Because trainer-workflow code changed, the approved
invalidation matrix required all six artifacts and the aggregate to be regenerated
under the fresh create-only v2 paths. V1 supports audit history only.

### V2 clean-source preflight

The v2 root, report, and both trace paths were absent before execution. No v2 path was
deleted, reused, resumed, or overwritten. Commands were serial and CPU-only.

| Command | Exit | Exact result |
| --- | ---: | --- |
| `uv sync --locked --group train` | 0 | 0.02 s; 45,816 KiB max RSS; 48 packages resolved/checked |
| `uv run pytest` | 0 | 1,264.97 s wall; `11887 passed in 1255.44s (0:20:55)`; stdout SHA-256 `986656b79d8fea82b8439799af58b8b666b77ca9b5e00b91efe417d202c5a42e`; stderr SHA-256 `740abd95c405d6cf5667ceb1277c049398bcd8cae67bb2dfc39538241c07f4fb` |
| `uv run ruff check .` | 0 | 0.69 s; `All checks passed!` |
| `uv run ruff format --check .` | 0 | 0.05 s; `122 files already formatted` |
| `git diff --check 8de76841739408b167a1f85c889cf75e555eece8...HEAD` | 0 | 0.09 s; zero output |

### V2 serial checkpoint execution

The exact artifact root is
`/home/haydenw/Projects/Harpy/runs/milestone-d-v2`. Every command exited 0. Each
artifact was immediately strict-loaded, trainer-validated, and actor-loaded on CPU
before the next trainer began. Each trainer wrote empty stdout and only the known
639-byte CUDA-driver warning to stderr (SHA-256
`ed07ad96a6f17b37c74cb3520304186e4c0c4b94b275ff9a661b847d77c487e6`).

| Artifact | Exact command output | Exit / elapsed / max RSS | Examples or steps | Complete / eligible | Manifest SHA-256 |
| --- | --- | --- | --- | --- | --- |
| BC seed 0 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/bc-0` | 0 / 1,288.75 s / 928,268 KiB | 4,096 train + 512 validation episodes; 101,511 / 12,710 examples; 50 epochs; selected 49; recorded training wall 972.0971357290109 s | complete; eligible; artifact criterion not met | `b675c6999fabfbfd795645827235381cd2d2717b1f1a7f4a9a455695f9b8ab2f` |
| PPO seed 0 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0` | 0 / 1,408.12 s / 911,832 KiB | 256,000 / 256,000 steps; recorded training wall 1305.725331242982 s | complete; eligible for aggregate | `48702405fb8be65531050a4cdd0c03cc9f91833d08fa2e7cf3236e9a765d5883` |
| PPO seed 1 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-1` | 0 / 1,221.47 s / 1,019,616 KiB | 256,000 / 256,000 steps; recorded training wall 1089.4454831639887 s | complete; eligible for aggregate | `a9dec54f6e1fe54b6a9ea04d58326b26771ef6e52cda3a909a88efa6a27f55c6` |
| PPO seed 2 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-2` | 0 / 1,053.29 s / 956,820 KiB | 256,000 / 256,000 steps; recorded training wall 985.1706004079897 s | complete; eligible for aggregate | `5111d1a391d9ba33fc1bd240c5192c883e12c40cf93a33b4374ac2c532db61bd` |
| PPO seed 3 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-3` | 0 / 1,095.49 s / 1,018,560 KiB | 256,000 / 256,000 steps; recorded training wall 1029.2674870709889 s | complete; eligible for aggregate | `493e395826a7a5d4df8c3bb9fe36948ead5957c5e6a17f2ca41e6d40ffda0de2` |
| PPO seed 4 | `/home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-4` | 0 / 1,140.70 s / 1,032,584 KiB | 256,000 / 256,000 steps; recorded training wall 1048.8930743329984 s | complete; eligible for aggregate | `70a06c7f786cd20e34cc5b4c6721ed90e57e59ab77b4d8bf75a8dc8d2c982fa9` |

Exact commands were:

```text
uv run harpy-sine-learn train-bc --profile checkpoint --seed 0 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/bc-0
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 0 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 1 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-1
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 2 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-2
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 3 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-3
uv run harpy-sine-learn train-ppo --profile checkpoint --seed 4 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-4
```

BC split SHA-256 values were
`f9664dc49d64b9e87e9bf5b0a0e823d7c2e78243b9c380bcbc3e10ecf5bd7a26`
and `3904604026b8738cc27801505340cc6517f0dd452cdcf8965d1926cf8f9e964f`.
Selected epoch 49 recorded validation loss `0.22605255905618546`, validation accuracy
`0.8825334382376082`, and held-out IID next-action accuracy
`0.8834252450980392`.

## IID rows

These are the exact unperturbed combined-IID rows from the canonical v2 report. Each
row retains all 256 typed terminal records; learned rows exactly equal their persisted
`evaluation-iid.json` payload rows.

| Actor | Submitted ≤5¢ | Submitted ≤1¢ | Final ≤5¢ | Final ≤1¢ | Mean / median abs. error | Mean actions / successful excess | Mean return | Truncation / invalid action |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| BC seed 0 | 0.0234375 | 0.00390625 | 0.0234375 | 0.00390625 | 164.9765625 / 132.0 | 17.41015625 / 0.16666666666666666 | -0.8611034445440574 | 0.0078125 / 0.0008974646623289208 |
| PPO seed 0 | 0.01171875 | 0.00390625 | 0.01953125 | 0.00390625 | 234.296875 / 182.5 | 47.46875 / 14.666666666666666 | -0.902727977074795 | 0.5625 / 0.014071757735352205 |
| PPO seed 1 | 0.0 | 0.0 | 0.01171875 | 0.01171875 | 193.40234375 / 178.0 | 59.06640625 / `null` | -0.9146635649334016 | 0.87109375 / 0.002975993651213544 |
| PPO seed 2 | 0.01171875 | 0.00390625 | 0.0234375 | 0.0078125 | 166.8359375 / 109.0 | 45.19921875 / 21.333333333333332 | -0.8859867757428278 | 0.46875 / 0.0022469968023507043 |
| PPO seed 3 | 0.01171875 | 0.00390625 | 0.01953125 | 0.00390625 | 181.359375 / 130.5 | 45.1484375 / 10.0 | -0.9227064561987705 | 0.3359375 / 0.07838726423256619 |
| PPO seed 4 | 0.01171875 | 0.00390625 | 0.01953125 | 0.00390625 | 176.81640625 / 138.0 | 36.0625 / 0.3333333333333333 | -0.8865164728483607 | 0.421875 / 0.0 |
| Random | 0.0 | 0.0 | 0.0 | 0.0 | 1315.43359375 / 1067.0 | 6.49609375 / `null` | -1.0963126453637295 | 0.0 / 0.0 |
| Reward Search | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 / 0.0 | 39.96484375 / 14.46484375 | 1.1186072841956969 | 0.0 / 0.000977421561919656 |
| Spectrum Peak | 1.0 | 0.5546875 | 1.0 | 0.5546875 | 1.234375 / 1.0 | 30.42578125 / 4.92578125 | 1.1188905526383197 | 0.0 / 0.0 |
| Oracle | 1.0 | 0.015625 | 1.0 | 0.015625 | 4.80078125 / 5.0 | 25.5 / 0.0 | 1.1183551536885246 | 0.0 / 0.0 |

## Register-OOD rows

Register-OOD is informational and is not a criterion gate. Combined checkpoint rows
retain exact paired terminal records. Lower and upper contain 128 records each;
combined contains their disjoint ordered union of 256.

### Lower OOD subset

| Actor | Submitted ≤5¢ | Submitted ≤1¢ | Final ≤5¢ | Final ≤1¢ | Mean / median abs. error | Mean actions / successful excess | Mean return | Truncation / invalid action |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| BC seed 0 | 0.0078125 | 0.0 | 0.0078125 | 0.0 | 165.7890625 / 135.0 | 14.9140625 / 0.0 | -0.8330542558913934 | 0.015625 / 0.0 |
| PPO seed 0 | 0.0 | 0.0 | 0.0 | 0.0 | 221.5546875 / 216.0 | 57.4765625 / `null` | -0.8582540522540981 | 0.7421875 / 0.0 |
| PPO seed 1 | 0.0 | 0.0 | 0.0234375 | 0.0078125 | 290.21875 / 295.5 | 58.1015625 / `null` | -0.8760738153176229 | 0.859375 / 0.011294876966518758 |
| PPO seed 2 | 0.0078125 | 0.0 | 0.0078125 | 0.0 | 263.703125 / 151.5 | 43.6953125 / 54.0 | -0.8559523040471311 | 0.296875 / 0.015018773466833541 |
| PPO seed 3 | 0.0078125 | 0.0 | 0.0234375 | 0.0 | 317.0546875 / 178.0 | 58.3125 / 11.0 | -1.0311660399590163 | 0.6484375 / 0.2967577706323687 |
| PPO seed 4 | 0.0078125 | 0.0 | 0.03125 | 0.0078125 | 214.6328125 / 205.5 | 36.6328125 / 45.0 | -0.8412824436475409 | 0.3984375 / 0.0 |
| Random | 0.0 | 0.0 | 0.0 | 0.0 | 1450.0859375 / 1299.5 | 6.9765625 / `null` | -1.0591401959528688 | 0.0 / 0.0 |
| Reward Search | 1.0 | 1.0 | 1.0 | 1.0 | 0.0078125 / 0.0 | 39.703125 / 15.2734375 | 1.1760649513319674 | 0.0 / 0.005509641873278237 |
| Spectrum Peak | 1.0 | 0.6015625 | 1.0 | 0.6015625 | 1.2421875 / 1.0 | 29.2890625 / 4.859375 | 1.1781520478995904 | 0.0 / 0.0 |
| Oracle | 1.0 | 0.015625 | 1.0 | 0.015625 | 4.84375 / 5.0 | 24.4296875 / 0.0 | 1.1776102215676232 | 0.0 / 0.0 |

### Upper OOD subset

| Actor | Submitted ≤5¢ | Submitted ≤1¢ | Final ≤5¢ | Final ≤1¢ | Mean / median abs. error | Mean actions / successful excess | Mean return | Truncation / invalid action |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| BC seed 0 | 0.0859375 | 0.015625 | 0.0859375 | 0.015625 | 159.5703125 / 144.0 | 16.484375 / 0.5454545454545454 | -0.6747578150614754 | 0.0 / 0.0 |
| PPO seed 0 | 0.015625 | 0.0 | 0.0234375 | 0.0078125 | 438.234375 / 291.5 | 41.59375 / 13.5 | -0.9032320888831967 | 0.4375 / 0.10086401202103681 |
| PPO seed 1 | 0.015625 | 0.0 | 0.0390625 | 0.0078125 | 172.6484375 / 158.0 | 58.671875 / 3.5 | -0.8202989574795082 | 0.890625 / 0.0039946737683089215 |
| PPO seed 2 | 0.0078125 | 0.0078125 | 0.0234375 | 0.0234375 | 211.2265625 / 172.5 | 40.796875 / 0.0 | -0.8474512090163935 | 0.5390625 / 0.018958253542703946 |
| PPO seed 3 | 0.0078125 | 0.0078125 | 0.0078125 | 0.0078125 | 260.8203125 / 193.5 | 35.8203125 / 0.0 | -0.8478020350922132 | 0.25 / 0.0 |
| PPO seed 4 | 0.0078125 | 0.0 | 0.015625 | 0.0 | 185.3671875 / 135.5 | 39.1328125 / 1.0 | -0.8354679828381149 | 0.46875 / 0.0 |
| Random | 0.0 | 0.0 | 0.0 | 0.0 | 1504.0546875 / 1261.5 | 6.6796875 / `null` | -1.0669420427766394 | 0.0 / 0.0 |
| Reward Search | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 / 0.0 | 40.2734375 / 13.7734375 | 1.1772591431864756 | 0.0 / 0.00504364694471387 |
| Spectrum Peak | 1.0 | 0.578125 | 1.0 | 0.578125 | 1.2265625 / 1.0 | 31.140625 / 4.640625 | 1.1791786142418035 | 0.0 / 0.0 |
| Oracle | 1.0 | 0.0234375 | 1.0 | 0.0234375 | 4.8125 / 5.0 | 26.5 / 0.0 | 1.178637161885246 | 0.0 / 0.0 |

### Combined OOD subset

| Actor | Submitted ≤5¢ | Submitted ≤1¢ | Final ≤5¢ | Final ≤1¢ | Mean / median abs. error | Mean actions / successful excess | Mean return | Truncation / invalid action |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- |
| BC seed 0 | 0.046875 | 0.0078125 | 0.046875 | 0.0078125 | 162.6796875 / 137.0 | 15.69921875 / 0.5 | -0.7539060354764344 | 0.0078125 / 0.0 |
| PPO seed 0 | 0.0078125 | 0.0 | 0.01171875 | 0.00390625 | 329.89453125 / 234.0 | 49.53515625 / 13.5 | -0.8807430705686474 | 0.58984375 / 0.042346818074284365 |
| PPO seed 1 | 0.0078125 | 0.0 | 0.03125 | 0.0078125 | 231.43359375 / 200.5 | 58.38671875 / 3.5 | -0.8481863863985655 | 0.875 / 0.0076269485515488055 |
| PPO seed 2 | 0.0078125 | 0.00390625 | 0.015625 | 0.01171875 | 237.46484375 / 152.0 | 42.24609375 / 27.0 | -0.8517017565317623 | 0.41796875 / 0.016920943134535366 |
| PPO seed 3 | 0.0078125 | 0.00390625 | 0.015625 | 0.00390625 | 288.9375 / 184.0 | 47.06640625 / 5.5 | -0.9394840375256148 | 0.44921875 / 0.18383268321022492 |
| PPO seed 4 | 0.0078125 | 0.0 | 0.0234375 | 0.00390625 | 200.0 / 160.0 | 37.8828125 / 23.0 | -0.8383752132428278 | 0.43359375 / 0.0 |
| Random | 0.0 | 0.0 | 0.0 | 0.0 | 1477.0703125 / 1277.0 | 6.828125 / `null` | -1.0630411193647542 | 0.0 / 0.0 |
| Reward Search | 1.0 | 1.0 | 1.0 | 1.0 | 0.00390625 / 0.0 | 39.98828125 / 14.5234375 | 1.1766620472592215 | 0.0 / 0.005274982905147992 |
| Spectrum Peak | 1.0 | 0.58984375 | 1.0 | 0.58984375 | 1.234375 / 1.0 | 30.21484375 / 4.75 | 1.178665331070697 | 0.0 / 0.0 |
| Oracle | 1.0 | 0.01953125 | 1.0 | 0.01953125 | 4.828125 / 5.0 | 25.46484375 / 0.0 | 1.1781236917264346 | 0.0 / 0.0 |

## Four baselines

### Random

Matched smoke row: 32 episodes; submitted ≤5¢ `0.0`; submitted ≤1¢ `0.0`;
final ≤5¢ `0.0`; final ≤1¢ `0.0`; mean/median absolute error
`1228.46875`/`991.0` cents; mean actions `9.3125`; successful excess `null`; mean
return `-1.0823474692622952`; truncation `0.0`; invalid-action rate `0.0`.
Matched v2 IID and lower/upper/combined register-OOD checkpoint rows are recorded
exactly above; terminal membership is paired across all actors.

### Reward Search

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `1.0`;
final ≤5¢ `1.0`; final ≤1¢ `1.0`; mean/median absolute error `0.0`/`0.0` cents;
mean actions `37.375`; successful excess `13.84375`; mean return
`1.118760225409836`; truncation `0.0`; invalid-action rate `0.0`.
Matched v2 IID and lower/upper/combined register-OOD checkpoint rows are recorded
exactly above; terminal membership is paired across all actors.

### Spectrum Peak

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `0.5625`;
final ≤5¢ `1.0`; final ≤1¢ `0.5625`; mean/median absolute error
`1.21875`/`1.0` cents; mean actions `27.625`; successful excess `4.09375`; mean
return `1.118657930327869`; truncation `0.0`; invalid-action rate `0.0`.
Matched v2 IID and lower/upper/combined register-OOD checkpoint rows are recorded
exactly above. Spectrum Peak is a
classical clean-sine control; PPO is not required to beat it.

### Oracle

Matched smoke row: 32 episodes; submitted ≤5¢ `1.0`; submitted ≤1¢ `0.03125`;
final ≤5¢ `1.0`; final ≤1¢ `0.03125`; mean/median absolute error `4.625`/`5.0`
cents; mean actions `23.53125`; successful excess `0.0`; mean return
`1.1181404661885246`; truncation `0.0`; invalid-action rate `0.0`.
Matched v2 IID and lower/upper/combined register-OOD checkpoint rows are recorded
exactly above. Oracle validates
reachability and planning, not learned perception.

## Zero-spectrum and shuffled-spectrum probes

| Actor / seed | Suite | Base submitted success | Zero-spectrum submitted success | Shuffled-spectrum submitted success | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| BC smoke seed 0 | smoke | `0.0` | `0.0` | `0.0` | Both probe rows exactly matched the base aggregate: 718.65625 mean error, 64.0 actions, -0.999328524590164 return, and 1.0 truncation. This smoke diagnostic showed no aggregate sensitivity. |
| PPO smoke seed 0 | smoke | `0.0` | `0.0` | `0.0` | Both probe rows exactly matched the base aggregate: 726.65625 mean error, 1.0 action, -1.0 return, and 0.0 truncation. This smoke diagnostic showed no aggregate sensitivity. |
| BC checkpoint seed 0 | IID | 0.0234375 | 0.0 | 0.0 | Both probes reduced submitted success to zero and substantially changed error/action/invalid-action aggregates. |
| PPO checkpoint seed 0 | IID | 0.01171875 | 0.00390625 | 0.00390625 | Both probes retained one submitted success but materially worsened error, truncation, and invalid-action rates. |
| PPO checkpoint seed 1 | IID | 0.0 | 0.0 | 0.0 | Base and both probes had zero submitted success; probe errors and invalid-action rates increased sharply. |
| PPO checkpoint seed 2 | IID | 0.01171875 | 0.00390625 | 0.00390625 | Both probes retained one submitted success while error and invalid-action rates worsened sharply. |
| PPO checkpoint seed 3 | IID | 0.01171875 | 0.0 | 0.0 | Both probes reduced submitted success to zero and sharply increased error and invalid-action rates. |
| PPO checkpoint seed 4 | IID | 0.01171875 | 0.0 | 0.00390625 | Zero removed all submitted success; shuffled retained one success; both probes greatly worsened error and invalid-action rates. |

These probes measure policy sensitivity only. They do not establish that spectral
evidence was causally necessary.

### Exact checkpoint probe rows

All rows contain 256 paired IID terminal records. `null` means successful excess is
undefined because no submitted success occurred.

| Actor | Probe | S5 | S1 | F5 | F1 | Error mean / median | Actions / excess | Return | Truncation | Invalid |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| BC seed 0 | zero spectrum | 0.0 | 0.0 | 0.00390625 | 0.0 | 1359.18359375 / 1367.5 | 20.515625 / `null` | -1.1396455968237704 | 0.19921875 | 0.17574257425742573 |
| BC seed 0 | shuffled spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1062.37890625 / 930.5 | 24.76171875 / `null` | -1.0810419422387294 | 0.23828125 | 0.10522164379239628 |
| PPO seed 0 | zero spectrum | 0.00390625 | 0.0 | 0.00390625 | 0.0 | 865.29296875 / 715.5 | 58.70703125 / 47.0 | -1.2694732018442625 | 0.6796875 | 0.43349524253110655 |
| PPO seed 0 | shuffled spectrum | 0.00390625 | 0.0 | 0.00390625 | 0.0 | 808.90625 / 645.0 | 53.0546875 / 11.0 | -1.1944959618340165 | 0.625 | 0.35576498306582244 |
| PPO seed 1 | zero spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1395.75390625 / 1131.0 | 56.52734375 / `null` | -1.395054050332992 | 0.76171875 | 0.5048027088660079 |
| PPO seed 1 | shuffled spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1331.3125 / 1129.0 | 56.53125 / `null` | -1.3174867213114756 | 0.765625 | 0.3861249309010503 |
| PPO seed 2 | zero spectrum | 0.00390625 | 0.0 | 0.00390625 | 0.0 | 969.15234375 / 837.5 | 50.7578125 / 59.0 | -1.3611079226434428 | 0.40234375 | 0.648684008003694 |
| PPO seed 2 | shuffled spectrum | 0.00390625 | 0.0 | 0.00390625 | 0.0 | 1155.0 / 964.5 | 45.9453125 / 62.0 | -1.3155481499743855 | 0.41796875 | 0.5510967522530182 |
| PPO seed 3 | zero spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1132.7578125 / 908.5 | 59.4296875 / `null` | -1.4235383741034837 | 0.8046875 | 0.6006967266990929 |
| PPO seed 3 | shuffled spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1099.19921875 / 851.0 | 54.578125 / `null` | -1.3351016137295084 | 0.69921875 | 0.5020755797308903 |
| PPO seed 4 | zero spectrum | 0.0 | 0.0 | 0.0 | 0.0 | 1255.37890625 / 928.5 | 62.7890625 / `null` | -1.4106997630635247 | 0.8828125 | 0.5159885529426403 |
| PPO seed 4 | shuffled spectrum | 0.00390625 | 0.00390625 | 0.00390625 | 0.00390625 | 1431.078125 / 1335.5 | 55.20703125 / 16.0 | -1.387906702740779 | 0.71875 | 0.5076063114696101 |

## BC criterion

| Field | Exact value |
| --- | --- |
| Eligibility | Smoke: `false`; checkpoint: `true` |
| Held-out IID next-action accuracy | Smoke: `null` by design; checkpoint: `0.8834252450980392` |
| IID submitted success | Smoke: `null` by design; checkpoint: `0.0234375` (6/256) |
| Thresholds | accuracy ≥ 0.90 and submitted success ≥ 0.75 |
| `criterion_met` | Smoke: `null`; checkpoint: `false` |
| Status | Smoke: `ineligible`; checkpoint: `criterion_not_met` |

## PPO criterion

| Field | Exact value |
| --- | --- |
| Eligibility | Smoke: `false`; checkpoint aggregate: `true` |
| Declared seeds | 0, 1, 2, 3, 4 |
| Per-seed IID submitted successes | Seeds 0–4: 3/256, 0/256, 3/256, 3/256, 3/256 |
| Median IID submitted success | Smoke: `null` by design; checkpoint: `0.01171875` (3/256) |
| Seeds strictly beating paired Random | Smoke: `null` by design; checkpoint: `4` |
| Thresholds | median success ≥ 0.50 and at least 4/5 seeds strictly beat Random |
| `criterion_met` | Smoke: `null`; checkpoint: `false` |
| Status | Smoke: `ineligible`; checkpoint: `criterion_not_met` |

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

### V2 checkpoint inventory

The strict public loader verified every manifest-declared size and digest, exact
closed inventory, JSON payload contract, and trusted-local model. The root contains
exactly 36 files. The shell SHA-256-of-sorted-`sha256sum` composite is
`287d2ffdce02557b573c169c74ebb1586df09873aa69ba3b3c459949618129ac`.

| Artifact | Payload | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `bc-0` | `manifest.json` | 2,398 | `b675c6999fabfbfd795645827235381cd2d2717b1f1a7f4a9a455695f9b8ab2f` |
| `bc-0` | `training-config.json` | 1,069 | `e56a4b0ae8c44a638c62a306d677923315fdad2002342edab1ec47e7ec6d7e58` |
| `bc-0` | `training-summary.json` | 6,540 | `7259f211838c21b1416a07a03edcb729246e403c569b6ffc3cb872c18944a87e` |
| `bc-0` | `model.pt` | 308,167 | `15347571c2c614501e94613b7b81be0666b626f9925b5092dc03bdb264b54f01` |
| `bc-0` | `evaluation-iid.json` | 251,132 | `cc431dc588deed2a4c812015f0cc1a4e9a9633ac099e069cd04ce3babc570eda` |
| `bc-0` | `evaluation-ood.json` | 168,001 | `03bd97401c70509c9c8598ceae8eafa993e5702577e9f061fd6fe8aad8c40a99` |
| `ppo-0` | `manifest.json` | 2,348 | `48702405fb8be65531050a4cdd0c03cc9f91833d08fa2e7cf3236e9a765d5883` |
| `ppo-0` | `training-config.json` | 943 | `448fe0fd028a95e9509ad8731507a70487b37e66d17356e13ecf5fed5e9bdee9` |
| `ppo-0` | `training-summary.json` | 202 | `9af15f238c4d2e3d4bccaa20d6bf8aa66e5586d8e572f2ca6602bd838314d194` |
| `ppo-0` | `model.zip` | 980,445 | `98c3491bb4df91bc60444649c586facc18fe3c2860867d1f8b3e16182832aa57` |
| `ppo-0` | `evaluation-iid.json` | 251,349 | `a5143cd374a3500c6afa06ad2c49c312c15ea7d763e804fd2545c44272f55660` |
| `ppo-0` | `evaluation-ood.json` | 168,239 | `94f50ec3ae49dadfe0743487cf1d4a2c8255ee3f241ec4c2d75472eee0824630` |
| `ppo-1` | `manifest.json` | 2,348 | `a9dec54f6e1fe54b6a9ea04d58326b26771ef6e52cda3a909a88efa6a27f55c6` |
| `ppo-1` | `training-config.json` | 943 | `1817c02f6947f4cf3680897345f7ea521043eb5125ed5e34c022e05dcf98d602` |
| `ppo-1` | `training-summary.json` | 203 | `961ea2f3cda67467f75bbcd1f1bba4d3a50b5de7d6df5c48164874811ec63b99` |
| `ppo-1` | `model.zip` | 980,440 | `a78e5f810e01493370ca956374037aadf01d0bdf1846094a602d956affe438e3` |
| `ppo-1` | `evaluation-iid.json` | 251,154 | `5cd8b547282b50c0aa0bd6960fcd853aeea758099628ca7d797c4d0ab8882f80` |
| `ppo-1` | `evaluation-ood.json` | 167,992 | `d97f5550a6cfb21eb781449681c3f55b5e99b052eda863473374e4319216fe6d` |
| `ppo-2` | `manifest.json` | 2,348 | `5111d1a391d9ba33fc1bd240c5192c883e12c40cf93a33b4374ac2c532db61bd` |
| `ppo-2` | `training-config.json` | 943 | `928174416c55eb7ef8c1ad14302bc7d70167347eb640f0ae54dd05de9a95fe18` |
| `ppo-2` | `training-summary.json` | 202 | `e50948e8d3f8107e46e855d2cec02724c3c0951276d576187856565524bf10ba` |
| `ppo-2` | `model.zip` | 980,440 | `1259f2a135e56e35eeae7fbe3c2c3154a3c33a875e23ee2f4f29850416d4a36c` |
| `ppo-2` | `evaluation-iid.json` | 251,476 | `0efbcb5d68514caa2bb4eb57a7842155b2f5d6da91efa3c56cd96228f85114d6` |
| `ppo-2` | `evaluation-ood.json` | 167,719 | `6785f0e1c0d123988acabeb89d62017eb25ab16107da4055832de040e9472fa7` |
| `ppo-3` | `manifest.json` | 2,348 | `493e395826a7a5d4df8c3bb9fe36948ead5957c5e6a17f2ca41e6d40ffda0de2` |
| `ppo-3` | `training-config.json` | 943 | `98b066a9df8e301c6599bacca15263642b0bdcdd1fd07fc82a0a85ca1eb5e1d0` |
| `ppo-3` | `training-summary.json` | 203 | `69413aa6cde7a711f4a9dc738ffc676ba67b0ff391ae93c0e1f45b444ba0f1ef` |
| `ppo-3` | `model.zip` | 980,445 | `554fe9ba84eb26c769da4611808e8297c3fc9a292aa52e62cbc2e51e7c226b2e` |
| `ppo-3` | `evaluation-iid.json` | 251,462 | `6c322d4b4094a2f82f3682cc8bb1a160394dfe205c7ff2780c0bf9757ff033e3` |
| `ppo-3` | `evaluation-ood.json` | 167,606 | `3f069a111683b7e2e52bedda4b16475c60b4aa6bfe8336d6611c0b77ff8ae1e7` |
| `ppo-4` | `manifest.json` | 2,348 | `70a06c7f786cd20e34cc5b4c6721ed90e57e59ab77b4d8bf75a8dc8d2c982fa9` |
| `ppo-4` | `training-config.json` | 943 | `6eae110eea6966d4dd7e9cbd20ae614feef9868531b8d07addde16a48c08ad67` |
| `ppo-4` | `training-summary.json` | 203 | `e24cbbf0dd5bdd3bfeb58c2336a278dc728593e7ecad39f3872ab1d98786e5a9` |
| `ppo-4` | `model.zip` | 980,445 | `d4d58eb1bf8bde2139c04d4236e1375a857ed0cc4846a0742343daebdc824e3b` |
| `ppo-4` | `evaluation-iid.json` | 251,047 | `ccfd8ef1215357ea5e998c009fe42302d20f0b41ccb6924ab7b94a0571b67ffe` |
| `ppo-4` | `evaluation-ood.json` | 167,063 | `6590db71b56816a9efd64615be1b9391dee94201158f835c7cd36ce6f564e656` |

### V2 canonical report and traces

| Path | Command / result | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `/home/haydenw/Projects/Harpy/runs/milestone-d-v2-report.json` | Exact six-artifact `harpy-sine-learn evaluate ... --output ...`; exit 0; 1,907.20 s; 879,780 KiB; strict duplicate-free decode and canonical re-encode; exactly 52 rows; all 36 learned rows equal persisted payload rows | 3,499,319 | `64b01d62aec7f5c606de430ec2b16c096193280daa42d1b414c330371ec203e5` |
| `/home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-1.json` | Create-only PPO0 seed-123 JSON run; exit 0; 4.82 s; typed strict decode/re-encode | 929 | `e3e2e3b2f819cabaceb0158384e1cc41da84c8e02e9f51bb2d6e9f1b91f7630e` |
| `/home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-2.json` | Independent create-only PPO0 seed-123 JSON run; exit 0; 4.45 s; `cmp` byte-identical to trace 1 | 929 | `e3e2e3b2f819cabaceb0158384e1cc41da84c8e02e9f51bb2d6e9f1b91f7630e` |

The human PPO0 seed-123 command exited 0 in 16.28 s (850,040 KiB). It targeted C2,
selected Semitone Down on steps 1–11, submitted on step 12, and ended
`submitted_failure` at 538 cents with total return `-0.81978213114754095`. Learned
model files are trusted-local Torch/SB3 artifacts and must not be loaded from
untrusted sources.

### Exact aggregate and trace commands

```text
uv run harpy-sine-learn evaluate /home/haydenw/Projects/Harpy/runs/milestone-d-v2/bc-0 /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0 /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-1 /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-2 /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-3 /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-4 --output /home/haydenw/Projects/Harpy/runs/milestone-d-v2-report.json
uv run harpy-sine-learn run /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0 --seed 123
uv run harpy-sine-learn run /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0 --seed 123 --json > /home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-1.json
uv run harpy-sine-learn run /home/haydenw/Projects/Harpy/runs/milestone-d-v2/ppo-0 --seed 123 --json > /home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-2.json
cmp /home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-1.json /home/haydenw/Projects/Harpy/runs/milestone-d-v2-trace-2.json
```

## Independent whole-branch reviews

All final v2 reviews were read-only and ran no broad test, trainer, or edit. The
primary reconciliation accepted no new finding.

| Reviewer | Scope | Final verdict | Exact confirmation |
| --- | --- | --- | --- |
| `/root/d_final_review_env` | environment/cache/leakage/frozen Milestone C | `APPROVED/CLEAR`; no Critical, Important, Minor, or deferred findings | Unchanged environment/cache/suite/evaluation code; six strict artifacts; exact 36-file inventory/composite; frozen suite digests/membership; terminal rows; report/traces; v1 excluded |
| `/root/d_final_review_science` | network/BC/PPO/scientific protocol | `APPROVED/CLEAR`; no Critical, Important, or Minor findings | Six CPU artifacts; exact 36 learned rows / 52-row matrix; BC accuracy `0.8834252450980392` and 6/256 success; PPO `(3,0,3,3,3)/256`, median 3/256, four seeds above Random; criteria and cautious claims correct |
| `/root/d_final_review_artifacts` | artifacts/CLI/errors/atomicity/docs/checkpoint evidence | `APPROVED/CLEAR`; no Critical, Important, or Minor findings | Strict artifacts/models; exact 36 files and composite `287d2ffdce02557b573c169c74ebb1586df09873aa69ba3b3c459949618129ac`; canonical report `64b01d62aec7f5c606de430ec2b16c096193280daa42d1b414c330371ec203e5`; traces `e3e2e3b2f819cabaceb0158384e1cc41da84c8e02e9f51bb2d6e9f1b91f7630e`; fixed lifecycle under `709b316`; v1 preserved and excluded |

The artifact reviewer’s earlier first pass reported the two Important lifecycle
findings recorded in the v1 invalidation section. Fix Round 1 commit `709b316` closed
both; the focused rereview was `APPROVED/CLEAR` before v2 generation, and the final v2
review above remained clear.

## Final fresh verification

Every gate ran serially after reviews and before this documentation edit. No unrelated
pytest or Harpy trainer/evaluator was active at an expensive boundary.

| Command | Exit | Exact observed result |
| --- | ---: | --- |
| `uv sync --locked --group train` | 0 | 0.03 s; 46,292 KiB; `Resolved 48 packages`, `Checked 48 packages`; output SHA-256 `3c261daf7281ae9b766b5681739c76285adf923698e4e2c7b30d126cc15061ab` |
| `uv run pytest tests/learning/test_training_smoke.py -q -rs` | 0 | `2 passed in 356.61s (0:05:56)`; 1,000,600 KiB; 110-byte stdout SHA-256 `d9e04ce9f35f1873432d95daf29daf14626bb5994264f087793fe0bfe62f763f`; empty stderr |
| Case-insensitive `rg -qi "skipped" /tmp/harpy-milestone-d-final-training-smoke.txt` absence assertion | 0 | No match; no skipped test accepted |
| `uv run pytest` | 0 | `11887 passed in 1216.88s (0:20:16)`; 1,228.31 s outer wall; 1,202,272 KiB; stdout SHA-256 `b081b9c7b274e3dadc499a5c2bfac0733331cf729273069051c8abc53fe14d7c`; stderr SHA-256 `740abd95c405d6cf5667ceb1277c049398bcd8cae67bb2dfc39538241c07f4fb` |
| `uv run ruff check .` | 0 | 0.68 s; `All checks passed!` |
| `uv run ruff format --check .` | 0 | 0.06 s; `122 files already formatted` |
| `uv run python -c "import sys, harpy.learning; assert 'torch' not in sys.modules; assert 'stable_baselines3' not in sys.modules"` | 0 | 0.28 s; empty stdout/stderr; lightweight import did not load Torch or SB3 |
| `uv run --isolated --no-group train --locked harpy-sine-learn --help` | 0 | 1.97 s; isolated environment installed 19 non-train packages; complete CLI help; no Torch/SB3 requirement |
| `uv run --isolated --no-group train --locked python -m harpy.learning.cli --help` | 0 | 1.00 s; isolated environment installed 19 non-train packages; byte-identical complete CLI help |
| `git diff --check` | 0 | 0.04 s; zero output before this edit |
| `git diff --check 8de76841739408b167a1f85c889cf75e555eece8...HEAD` | 0 | 0.07 s; zero output |
| Strict v2/v1 evidence audit | 0 | 24.40 s; 874,264 KiB; all six trusted-local models loaded on CPU; 36-file inventory/composite; canonical 52 rows and persisted-row equality; paired membership; typed/canonical trace equality; unchanged v1 composite/report |
| Exact source/lock/status/process audit | 0 | HEAD `709b316175d8835173f246b2bfcb0937fb06355b`; lock `4bad0c09763801f665d6249760e854991ed537d5375cf5cbeeae910ea8d564d7`; tracked status empty; no active test/trainer/evaluator |

The smoke process’s outer `/usr/bin/time` wall reading spanned an observed host/session
pause and date rollover and was `1:22:28`; pytest’s own active duration was 356.61 s.
Both values are retained rather than silently normalizing the anomaly. Final
full-suite stderr contained only the already-baselined PipeWire symbol and Qt
mouse-grab diagnostics.

`docs/project-notebook.md` remained byte-identical throughout with SHA-256
`a85f1bdc9960ed243dfc0030c691212bc046a8ab7ccb43731e5163e9b55a69ac`.

## Concerns and limits

- Engineering passed, but both eligible scientific criteria missed. BC achieved only
  6/256 IID submitted successes and its held-out action accuracy was below 0.90. PPO
  achieved `(3,0,3,3,3)/256`; its median was 3/256, far below 0.50, even though four
  seeds strictly beat the paired Random result of zero.
- PPO base-IID truncation ranged from `0.3359375` to `0.87109375`, and several probe
  lanes had invalid-action rates above 0.5 (maximum `0.648684008003694`) and truncation
  as high as `0.8828125`. BC also showed nonzero base invalid actions and probe
  degradation. No weak seed was hidden, tuned, or retrained for outcome.
- Reward Search succeeded on every episode but retained small nonzero invalid-action
  rates (`0.000977421561919656` IID and `0.005274982905147992` combined OOD).
- The installed `+cu130` Torch build reported the host-driver API 12080 warning. CUDA
  availability was false; all authoritative training and evaluation devices are CPU.
- Model files are trusted-local Torch/SB3 artifacts and are unsafe to load from
  untrusted sources.
- Historical `957c140` remains absent. The approved recovered range base is
  `8de76841739408b167a1f85c889cf75e555eece8`; this recovery fact does not weaken the
  exact clean source/lock evidence.
- Candidate spectrum is continuously present in `Harpy/SinePitch-v0`. Milestone D adds
  no analysis tools and makes no claim about recorded-audio pitch shifting, chords, or
  waveform generalization.
