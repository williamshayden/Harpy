# First v2 implementation slice: verification record

Date: 2026-09-16. Development package: `2.0.0.dev0`. This records the first
executable milestone from the [delivery plan](v2-plan.md), not a v2 release or a
qualified sound-matching model. Use the [development guide](v2-getting-started.md)
to run it.

The tested package Python-source inventory digest is
`d7856e84fed468c3a6abc7641efa48bb983a7150ca3c220a1181f6696cbc3e55`.
It binds the actual `.py` files, independently of later documentation changes.
The environment was Python 3.12.3 / NumPy 2.5.1 on Linux/WSL, with base and
development dependencies from `uv.lock`. Torch and Stable-Baselines3 were absent.

## Implemented behavior

- Deterministic two-second synthesis with one or three notes, four oscillator
  families, shared brightness/ADSR/gain, and strict canonical patches.
- Owned public observations, fresh actors, explicit `Propose`/`Submit`, charged
  repeated/cache auditions, structured invalid decisions, cleanup, and failure
  records. A waveform cache is bounded to 16 MiB by default.
- Fixed, seeded random-search, and bounded pattern-search actors. Search pays to
  restore a previous best candidate before submission.
- Complete create-only result bundles with decision ledgers, portable source and
  actor inventories, target/submitted float WAVs, editable patch JSON, recomputed
  metrics, three readout formats, and explicit audio/score replay verification.
- Single-load strict v1 result dispatch; shared JSON/hash/atomic-writing helpers
  with historical import compatibility; base and optional CPU-training CI jobs.

This is M1 plus part of M0. The full M0 portability/migration work, M2 calibration
and frozen datasets, M3 inverse training, M4 journaling/resume, and M5 release
qualification remain outstanding. Basic chord/mask support is implemented, but
does not substitute for the M2 study.

## Automated checks

The combined new task and CLI selection passed **80 tests**. This covers strict
patch bounds, chord permutations, CPU golden samples, full-clip scores, budget
boundaries, cache invariance, observation ownership, lifecycle and injected
failures, deterministic search, complete bundles, replay, and invalid evidence.

Independent review found and fixed three defects before final verification:
exported WAVs were not bound to the scored samples; Python equality could accept
booleans in place of numeric evidence; and malformed ledger types could fail with
uncaught arithmetic/type errors. Regression cases now reject rehashed swapped
audio, wrong WAV format, patch/metric type substitutions, malformed ledgers, and
overflowing numeric values. Ordinary loading still reads each bundle file once
without synthesizing audio.

The complete base selection used by CI passed **10,758 tests**, with four
Torch-related skips and two explicit deselections, in 110.63 seconds:

```bash
uv run --no-sync pytest -q \
  tests/test_analysis.py tests/test_tuning.py tests/test_imports.py \
  tests/test_cli.py tests/test_serialization.py tests/test_sound_match_cli.py \
  tests/envs tests/synth tests/tasks tests/experiments tests/learning/test_artifacts.py \
  --ignore=tests/experiments/test_training.py \
  -k 'not test_frozen_schema_v1_models_strict_load_and_repeat_seeded_actions and not test_historical_import_retains_model_bytes_and_original_training_identity'
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
```

Both Ruff checks passed (136 Python files). The two deselected cases load frozen
or historical Torch models; the four skips are Torch actor cases. The optional
CI job explicitly covers real CPU training and model compatibility. Its
requirements export was checked, but that job and hosted CI have not been run
here. No new model training or CUDA qualification was performed.

## Installed distributions

The wheel and source archive were built offline, then installed separately into
fresh virtual environments outside the checkout using locked base constraints.
Registry resolution needed network access; only the five base dependencies and
Harpy were installed. Both environments confirmed Torch/SB3 were absent.

Both passed CLI help, v1 classical runs and all readout formats, a waveform-FFT
run/reload, shipped reference-file hash checks, and a sound-matching run with
all readout formats and explicit replay. Each installed package's 70 Python
files matched the source digest above. New v2 documentation is included in the
source distribution. This verifies base-package installation and execution;
it does not substitute for the optional model-training job.

## Actual CLI and audio workflows

For each of `tone-smoke`, `chord-smoke`, and `shaping-smoke`, run:

```bash
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor fixed --actor random-search --actor pattern-search --seed 0 \
  --output runs/tone-comparison
harpy summarize runs/tone-comparison --verify-replay
harpy summarize runs/tone-comparison --format markdown
harpy summarize runs/tone-comparison --format csv
```

All **21 actor-episodes** explicitly submitted, with zero invalid decisions,
errors, or truncations. Every accepted candidate passed DSP replay. All 28 WAV
exports passed independent RIFF/chunk checks: 48 kHz, mono, 32-bit IEEE float,
96,000 finite samples, exactly two seconds, and matching payload hashes. The
maximum observed peak was 0.293; there was no clipping or output normalization.
These are signal/file checks, not a blind listening assessment.

The following means are a workflow readout only. Lower distance is better under
the initial uncalibrated objective. Counts are deliberately small, public, and
consumed; do not infer generalization or a superior method from this table.

| Example group | Targets | Fixed distance | Random-search distance | Pattern-search distance |
| --- | --- | --- | --- | --- |
| Single-note reconstruction | 2 | 1.300092 | 1.081731 | 0.581861 |
| Three-note reconstruction | 2 | 1.925157 | 1.307256 | 0.873823 |
| Single-note pitch-only edit | 1 | 0.947735 | 0.063926 | 0 |
| Three-note envelope-only edit | 1 | 0.463939 | 0.174072 | 0.280617 |
| Single-note brightness-only edit | 1 | 1.867157 | 1.052188 | 0 |

The simple `tone-001` case was matched exactly in three charged evaluations,
from initial distance 0.741089 to zero. The more complex reconstruction examples
retain substantial errors: pattern search's average chord pitch error was 200
cents despite improved spectral distance. A lower distance is not a claim of
correct notes or perceptual equivalence; pitch/envelope/level diagnostics remain
visible in the saved evidence and summaries.

The local ignored archive is `outputs/v2-m1-verification/`, containing three
bundles, readouts, and `workflow-verification.json`. That report's SHA-256 is
`7f6a160781cf595e61351b6db472f0e5e613f6aac2431b0c32fc6cd07fb6e2ef`.
The archive has not been published. A clone can rerun the public smoke examples;
it does not contain this recorded archive.

## Bounded performance observations

The renderer's harmonic recurrence was compared with direct sine sums: raw
amplitude differences were below `4.3e-12`, and both measured full-render golden
hashes were unchanged. On the local pinned CPU environment, 32 render-and-score
iterations took 1.10 seconds for a single sine and 3.83 seconds for a deliberately
expensive low, closely spaced three-note saw patch with maximum release. Peak
process RSS was 48.3 MiB. These bounded probes do not predict training throughput
or establish a cross-machine latency guarantee.

For the retained 17,995,025-byte v1 representation-study result, fresh-process
before/after measurements used the same base interpreter:

| Readout | Whole-file reads / decodes | Peak RSS | Elapsed time |
| --- | --- | --- | --- |
| Released v1 | 2 / 2 | 215.5 MiB | 7.139 s |
| Development implementation | 1 / 1 | 135.5 MiB | 6.741 s |

The 15,311-character summary was byte-identical, SHA-256
`819ec1885e424e6b4b05f746324810d3fdf1354f27a5e69a4a93406e3e5c8ff8`.
Timing is one noisy sample. The one-load path and observed memory reduction are
the demonstrated improvements; historical report codecs still use their existing
internal parsing paths.

## Remaining qualification

Sound-acceptance calibration, the 96-target pilot, frozen learning/evaluation
splits, inverse-model training, interrupted-run resume, independent v1 archive
reproduction, CUDA, native Windows/macOS, and hosted CI remain unqualified.
This milestone makes those next experiments executable; it does not claim their
results in advance.
