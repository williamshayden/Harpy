# Exact objective caching verification

The `harpy-multires-spectral-v1` objective now reuses bounded, process-local
calculations. Its formulas and returned scores are unchanged. The
[verification record](../research/sound_matching/evidence/objective-cache-verification.json)
binds the tested implementation, preserved original source and recorded pilot
fixtures by SHA-256.

## Policy and ownership

Every call validates both full-clip float32 inputs and makes owned float64
snapshots. Ordered SHA-256 hashes of those snapshots identify a target/candidate
pair. Mutating a caller-owned waveform changes the key on the next call; neither
input array is retained.

At most two waveform analyses are cached, with a separate hard ceiling of
20 MiB of retained NumPy array payload. Each analysis holds read-only raw and
log-magnitude spectra, their original-order norms, RMS envelope and level RMS.
The current fixed clip/FFT configuration retains **18,499,488 bytes (17.64 MiB)**
for two analyses. A separate cache holds at most 128 score documents. Every
returned document is a deep copy, including the nested spectral rows.

A process-local reentrant lock protects cache access, analysis, scoring and
clearing. The private `_clear_score_cache()` and `_score_cache_info()` hooks
support isolation and measurement. The 20 MiB array ceiling excludes transient
FFT working memory and the bounded Python score-document objects. It is separate
from the runner's 16 MiB waveform cache. Logical candidate evaluations, actual
DSP render accounting, model calls and proposal/submission behavior remain intact.

## Exactness evidence

Sixteen objective tests passed in 13.58 seconds. The preserved original scorer
has SHA-256 `ca8a448dbc8f583bef686fa77c99cefee079ade92e0b718a02a9fda03f3877f7`.
Comparisons use canonical JSON equality of all returned float64 values, including
their round-trippable representations. They cover:

- All 14 calibration fixtures, including silence, clipping, polarity and phase.
- 32 retained pilot target/candidate pairs across all eight cells and all four
  candidate waveforms. Cached scores equal both the preserved implementation and
  the scores recorded in the original pilot.
- Input-content changes, noncontiguous inputs, nested output mutation, invalid
  inputs after cache priming, absence of retained input arrays, read-only analysis
  ownership, count/byte eviction, and concurrent scoring/clearing.
- Complete pattern-search and random-search episode records: every proposed
  patch, score, ledger event, terminal state and logical/DSP/model counter agrees
  exactly with the preserved scorer, apart from elapsed time.

The objective identifier is retained on this evidence. No renderer, model,
dataset, target membership, loss definition or evaluation budget changed, and no
final-test target was used in these comparisons.

## Bounded profile on 2026-09-22

The profile used the first calibration single-note reconstruction,
`e_e12a1d786058bbc29bc23ea9`, with seed 0. Each implementation ran in a fresh base
Python process, and scoring caches were cold before each actor. The host was the
same Intel Core Ultra 7 155H / Linux WSL environment used for inverse training.

| Actor | Original wall seconds | Cached wall seconds | Reduction |
| --- | ---: | ---: | ---: |
| Pattern search | 2.842 | 1.639 | 42.3% |
| Random search | 2.791 | 1.241 | 55.5% |

Each actor still made 32 charged evaluations: 31 candidate DSP renders and one
waveform-cache hit. Its 64 public scoring calls now required 31 computations and
33 score-cache hits. Magnitude computations fell from 384 to 96 per actor. Peak
process RSS across the two sequential actor profiles was 68.94 MiB with the
preserved scorer and 79.16 MiB with caching. These timings describe this profiling
fixture; the byte and entry ceilings are enforced independently of timing.

Raw profiles and full compared episode records are retained locally under
`outputs/v2-objective-profile-original-20260922/` and
`outputs/v2-objective-profile-cached-20260922/`. Re-run the portable exactness checks
from the source checkout:

```bash
PYTHONPATH=src python -m pytest \
  tests/tasks/test_sound_match_objective.py \
  tests/tasks/test_sound_match_objective_cache.py -q
```
