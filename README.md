# harPY

**harmonic research in python**

harPY is a headless Python toolkit for controlled audio experiments. Version 2
adds sound matching: an actor hears a short synthesized target, proposes editable
patches, auditions each candidate, and submits a sound within a fixed budget.
It also preserves the released v1 pitch-control benchmark and its saved evidence.

The synthesizer renders two-second mono clips with one or three notes, four
waveforms, shared brightness, an ADSR envelope, and gain. Tasks cover complete
reconstruction and changes restricted to pitch, envelope, or brightness.
Classical search and a trained inverse estimator use the same public observations
and candidate-evaluation rules.

**This checkout contains package 2.0.0.** Local qualification and archive status
are recorded in [v2 acceptance](docs/v2-acceptance.md). A local package or evidence
archive is distinct from a published release. The existing public v1 release uses
tag `v1`; the older `v1.0.0` tag preserves a historical development checkpoint.

## Install and run

Use Python 3.12 on Linux/WSL, the qualified platform, from this checkout or a
built wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .           # Synthesis, classical actors, readers, replay
python -m pip install '.[learn]'  # Optional inverse sound model, using Torch
```

The distribution is `harpy-audio`; the CLI and imports are `harpy`. Base
operation has no Torch or Stable-Baselines3 dependency. The `pitch` extra retains
v1 supervised learning, and `train` adds v1 PPO. Sound matching is CPU-qualified;
native Windows/macOS and new-task CUDA execution are not qualified.

```bash
harpy run --task sound-match-v1 --suite tone-smoke --actor pattern-search \
  --output runs/first-sound
harpy summarize runs/first-sound
harpy summarize runs/first-sound --verify-replay
harpy evaluate --task sound-match-v1 --suite chord-smoke \
  --actor fixed --actor random-search --seed 0 --output runs/chord-comparison
```

Each bundle retains decision traces, objective components, target/submitted WAVs,
and editable submitted patches. The [sound-matching guide](docs/v2-getting-started.md)
shows how to find and audition the files, use shaping masks, compare actors, and
write a custom Python actor.

## Train and compare a learned initial patch

```bash
harpy train inverse --profile quick --seed 0 --output runs/inverse-quick
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor inverse=runs/inverse-quick \
  --actor inverse-refine=runs/inverse-quick \
  --actor pattern-search --output runs/inverse-comparison
```

The inverse model makes one legal patch proposal. `inverse-refine` spends the
remaining budget on ordinary pattern search. A quick run exercises optimization
and reload; the frozen reference profile trains on 2,016 targets and selects
among six epochs using 192 validation targets. All three training seeds are
retained and compared; see the [method](docs/v2-inverse.md).

All actors share a ceiling of 32 candidate evaluations and 64 decisions. Actual
usage differs: fixed uses one evaluation, inverse uses two, and search can use
32. The initial candidate costs one evaluation, and cache hits or restored patches
still cost evaluations. Actors must explicitly submit their current candidate. Submission
means the protocol finished, not that a listener would judge the sound correct.

The objective is a frozen continuous spectral distance, accompanied by pitch,
envelope, and level diagnostics. There is no asserted perceptual success cutoff.
Read reconstruction and shaping cells separately, along with errors, truncations,
invalid actions, model calls, and render costs. Model quality claims apply only
to the recorded artifacts and declared criteria.

The frozen study completed all 6,336 actor-episodes with zero errors, truncations,
or invalid actions. All three inverse seeds passed the declared validation
criterion. Inverse plus refinement has lower equal-eight-cell mean distance than
the declared search baselines on both final tests; individual targets and some
cells still favor other actors. See [all-seed results](docs/v2-results.md).
Actual listening review remains open in the acceptance record.

## Interrupt, resume, and read

```bash
harpy summarize runs/chord-comparison --allow-incomplete
harpy evaluate --resume --output runs/chord-comparison
harpy summarize runs/chord-comparison --format csv
```

Sound evaluations commit one actor/episode at a time. Resume preserves completed
records and runs missing work only, rejecting changed source, runtime,
configuration, membership, or model bytes. Errors and truncations are retained
observations. Interrupted attempts remain visible; unfinished costs are unknown.
Single-writer locks prevent concurrent updates to the same bundle.

Readers work without the actor's model and validate hashes, membership, records,
and recomputed summaries. Portable actor identities are separate from local
lookup paths. Exact audio/action replay is available for both durable tasks.
Outputs and training artifacts are create-only; training does not support resume.

## Keep using the v1 pitch task

Omit `--task` to retain the single-tone pitch-control workflow and JSON outputs:

```bash
harpy evaluate --actor waveform-fft --actor spectrum-peak \
  --suite smoke --output runs/pitch-comparison.json
harpy summarize runs/pitch-comparison.json
harpy run --actor waveform-fft --source-cents 6064 --target-note-index 12 \
  --output runs/pitch-trace.json
# Opt into durable v1 evaluation:
harpy evaluate --bundle --actor waveform-fft --suite smoke --output runs/pitch-bundle
```

V1 permits 64 actions with octave, semitone, and fine-cent controls. Success
requires explicit submission within five cents, inclusive. Its renderer,
observations, seven actions, criteria, reference weights, and old readers retain
their existing contracts. With the `pitch` extra, `--actor reference` still loads
the packaged v1 seed-0 pitch model. `harpy train pitch` and `harpy train ppo`
continue to target v1; PPO remains experimental.

Reference seeds 0, 1, and 2 each passed 650/650 clean benchmark and 1,000/1,000
clean confirmation episodes. All three and matched Spectrum Peak passed level,
phase, and 30 dB noise conditions; each had failures at 10 dB and under combined
corruption. There is no demonstrated learned advantage or one-cent guarantee.
The waveform FFT study separately passed 600/600 pairs in each of eight
conditions. See [v1 qualification](docs/verification/2026-09-08-v1-respec-qualification.md)
and [FFT evidence](docs/v1-waveform-fft.md).

The [public v1 release](https://github.com/williamshayden/Harpy/releases/tag/v1)
includes its wheel, source, and portable representation study. The v2 acceptance
record identifies the separately prepared portable all-seed v1 archive and its
access status. Consumed confirmation and final-test memberships support
reproducing the recorded studies; future model selection needs new test data.

## Documentation and scope

- [V2 getting started](docs/v2-getting-started.md), [inverse method](docs/v2-inverse.md),
  and [calibration](docs/v2-calibration.md).
- [V2 results](docs/v2-results.md), [acceptance](docs/v2-acceptance.md), [migration](docs/v2-migration.md),
  [specification](docs/v2-spec.md), and [delivery plan](docs/v2-plan.md).
- [V1 guide](docs/v1-getting-started.md), [acceptance](docs/v1-acceptance.md),
  and [contributing](docs/v1-contributing.md).

Custom actors use a cooperative Python interface with fresh per-episode state
and declared artifacts. They are not sandboxed. Load PPO archives only from
trusted producers; hashes establish identity, not trust.

External recordings, live audio, VST, effects, a UI, and new-task PPO are outside
v2. It is a controlled known-synth research toolkit, with engineering correctness
and measured model performance reported separately.

MIT licensed. See [LICENSE](LICENSE).
