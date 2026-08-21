# Milestone D learned sine policy acceptance evidence

Date exercised: `pending`

## Acceptance status

Engineering status: `pending`

BC scientific status: `pending`

PPO scientific status: `pending`

This document separates smoke-profile engineering evidence from checkpoint-profile
scientific evidence. A successful smoke run is never a scientific result, and a valid
checkpoint that reports `criterion_not_met` is not an engineering failure.

## Commit, lock, and runtime

| Field | Exact value |
| --- | --- |
| Source commit | `pending` |
| Source status | `pending` |
| `uv.lock` SHA-256 | `pending` |
| Python | `pending` |
| Platform | `pending` |
| Processor / CPU | `pending` |
| NumPy | `pending` |
| Gymnasium | `pending` |
| PyTorch | `pending` |
| Stable-Baselines3 | `pending` |
| Training device | `pending` |
| Evaluation device | `pending` |
| Torch/thread settings | `pending` |
| CUDA availability/runtime/driver | `pending` |

## Full automated gates

All commands must run serially. The training smoke is accepted only when the optional
stack is installed and the result contains no skip.

| Command | Exit | Elapsed | Exact observed result / warnings |
| --- | ---: | ---: | --- |
| `uv sync --locked --group train` | `pending` | `pending` | `pending` |
| `uv run pytest tests/learning/test_training_smoke.py -q -rs` | `pending` | `pending` | `pending` |
| No-skip check over training-smoke output | `pending` | `pending` | `pending` |
| `uv run pytest` | `pending` | `pending` | `pending` |
| `uv run ruff check .` | `pending` | `pending` | `pending` |
| `uv run ruff format --check .` | `pending` | `pending` | `pending` |
| `git diff --check` | `pending` | `pending` | `pending` |

## BC smoke

| Field | Exact observed value |
| --- | --- |
| Command | `pending` |
| Exit / elapsed | `pending` |
| Artifact path | `pending` |
| Profile / seed / device | `pending` |
| Configured training episodes | `pending` |
| Configured validation episodes | `pending` |
| Trajectory-derived training examples | `pending` |
| Trajectory-derived validation examples | `pending` |
| Epochs / selected epoch | `pending` |
| Training wall time | `pending` |
| Parameter count | `pending` |
| Persisted reload / evaluation / completion | `pending` |
| Smoke base row | `pending` |
| Criterion eligibility | `pending` |

## PPO smoke

| Field | Exact observed value |
| --- | --- |
| Command | `pending` |
| Exit / elapsed | `pending` |
| Artifact path | `pending` |
| Profile / seed / device | `pending` |
| Requested environment steps | `pending` |
| Completed environment steps | `pending` |
| Training wall time | `pending` |
| Parameter count | `pending` |
| Persisted reload / evaluation / completion | `pending` |
| Smoke base row | `pending` |
| Criterion eligibility | `pending` |

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

Matched smoke evidence: `pending`. Matched IID/register-OOD checkpoint evidence:
`pending`.

### Reward Search

Matched smoke evidence: `pending`. Matched IID/register-OOD checkpoint evidence:
`pending`.

### Spectrum Peak

Matched smoke evidence: `pending`. Matched IID/register-OOD checkpoint evidence:
`pending`. Spectrum Peak is a classical clean-sine control; PPO is not required to
beat it.

### Oracle

Matched smoke evidence: `pending`. Matched IID/register-OOD checkpoint evidence:
`pending`. Oracle validates reachability and planning, not learned perception.

## Zero-spectrum and shuffled-spectrum probes

| Actor / seed | Suite | Base submitted success | Zero-spectrum submitted success | Shuffled-spectrum submitted success | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| BC smoke seed 0 | smoke | `pending` | `pending` | `pending` | `pending` |
| PPO smoke seed 0 | smoke | `pending` | `pending` | `pending` | `pending` |
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
| Eligibility | `pending` |
| Held-out IID next-action accuracy | `pending` |
| IID submitted success | `pending` |
| Thresholds | accuracy ≥ 0.90 and submitted success ≥ 0.75 |
| `criterion_met` | `pending` |
| Status | `pending` |

## PPO criterion

| Field | Exact value |
| --- | --- |
| Eligibility | `pending` |
| Declared seeds | 0, 1, 2, 3, 4 |
| Median IID submitted success | `pending` |
| Seeds strictly beating paired Random | `pending` |
| Thresholds | median success ≥ 0.50 and at least 4/5 seeds strictly beat Random |
| `criterion_met` | `pending` |
| Status | `pending` |

## Payload hashes

Smoke manifests, every manifest-declared payload, the combined smoke report, and the
two byte-identical trace files: `pending`.

Checkpoint BC/PPO manifests, every manifest-declared payload, and the canonical
checkpoint report: `pending` until Task 11.

## Concerns and pending items

- Fresh clean-commit smoke gates and CLI commands: `pending`.
- Checkpoint BC seed 0 and PPO seeds 0–4: `pending` for Task 11.
- IID, lower/upper/combined register-OOD, baseline, and probe results: `pending` for
  Task 11.
- Whole-branch independent review verdicts: `pending` for Task 11.
- Learned model files are trusted-local Torch/SB3 artifacts and are not safe to load
  from untrusted sources.
- Candidate spectrum is continuously present in `Harpy/SinePitch-v0`. Milestone D
  adds no analysis tools and makes no claim about recorded-audio pitch shifting,
  chords, or waveform generalization.
