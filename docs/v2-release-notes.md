# harPY v2 release notes

Version **2.0.0 was published on 2026-10-26 as an experimental research
prerelease**. Its [GitHub release](https://github.com/williamshayden/Harpy/releases/tag/v2.0.0)
provides the packages and complete study evidence. The original artifacts retain
their dated preparation status; the release page records their publication.

This checkout prepares **2.0.1**, a documentation and acceptance update for the
same sound-matching implementation. Listening has been performed; the listener's
matching-quality concerns and the stable-release decision remain under review.
See the [listening review](v2-listening-review.md). The 2.0.0 tag, package bytes,
and frozen scientific evidence are preserved.

## What changes

V2 adds `sound-match-v1`: recreate a two-second synthesized sound with one or
three notes, or change only its declared pitch, envelope, or brightness controls.
Actors propose patches, audition their rendered audio, and explicitly submit a
candidate. Saved results include WAVs, editable patches, decisions, measurements,
and costs.

- Compare fixed patches, random search, pattern search, a supervised inverse
  model, and the same model followed by pattern-search refinement.
- Train an inverse model with the optional Torch-only `learn` extra and evaluate
  its reloaded checkpoint through the ordinary runner.
- Interrupt and resume durable evaluations, read incomplete results, relocate
  bundles, and replay saved actions without loading a model.
- Keep using v1 pitch commands, models, report readers, and default JSON output;
  durable pitch bundles are opt-in. See the [migration guide](v2-migration.md).

The frozen study retains 6,336 actor-episodes with zero errors, truncations, or
invalid actions. All three inverse seeds met the predeclared validation
criterion. Inverse plus refinement achieved lower equal-eight-cell mean spectral
distance than the declared search baselines on both final tests; individual
targets and some cells favored other actors. The mean weights reconstruction at
25% and shaping at 75%. [All-seed results](v2-results.md) preserve those distinctions.

## Install and try it

Use Python 3.12 on Linux/WSL, the qualified CPU environment. From this source
checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Alternatively, download the published
[2.0.0 wheel](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/harpy_audio-2.0.0-py3-none-any.whl)
and install it from your download directory:

```bash
python -m pip install ./harpy_audio-2.0.0-py3-none-any.whl
```

For optional inverse training, use `python -m pip install '.[learn]'` from the
checkout, or install the wheel with its extra:

```bash
python -m pip install './harpy_audio-2.0.0-py3-none-any.whl[learn]'
```

Base synthesis, classical actors, result reading, and replay do not require
Torch or Stable-Baselines3. The distribution is `harpy-audio`; its CLI and imports
are `harpy`.

```bash
harpy run --task sound-match-v1 --suite tone-smoke --actor pattern-search \
  --output runs/first-sound
harpy summarize runs/first-sound --verify-replay
```

Use a new output path for a new run. The [getting-started guide](v2-getting-started.md)
covers chords, shaping, audio inspection, training, and custom Python actors.
The [contribution guide](v2-contributing.md) describes development checks.

## Published artifacts and qualification

The [2.0.0 release assets](https://github.com/williamshayden/Harpy/releases/tag/v2.0.0)
include:

- The [wheel](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/harpy_audio-2.0.0-py3-none-any.whl)
  and [source distribution](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/harpy_audio-2.0.0.tar.gz).
- The [complete study archive](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/v2-complete-study-20260922.tar.gz),
  containing all three selected models, saved evaluations, listening examples,
  frozen source, and the retained v1 evidence.
- [SHA256SUMS](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/SHA256SUMS),
  [publication-index.json](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/publication-index.json),
  and [qualification.json](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/qualification.json).
- Separate installation-proof archives and the
  [41.25-second review reel](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/all-cohorts-review.wav).

Download every file listed in `SHA256SUMS` into one directory and run
`sha256sum -c SHA256SUMS` to verify the complete asset set. Install either
downloaded distribution with:

```bash
python -m pip install ./harpy_audio-2.0.0-py3-none-any.whl
# Or install the downloaded source distribution:
python -m pip install ./harpy_audio-2.0.0.tar.gz
```

The [archive guide](../research/sound_matching/README.md) explains model paths,
reproduction commands, and the distinction between original September evidence
and the October package verification. GitHub assets are the published package
distribution; these instructions do not assume a package-index release.

The 2.0.0 qualification passed 10,938 base tests, with five optional Torch skips
and two model-loading deselections, plus a separate local CPU suite with all
335 tests passing. Wheel and source installs outside the checkout exercised
run, resume, strict replay, and custom actors. The
[hosted base CI run](https://github.com/williamshayden/Harpy/actions/runs/40428101600)
also passed on the release commit. Its optional training job was skipped; local
optional coverage is recorded separately. These checks qualify the published
2.0.0 artifacts. A changed 2.0.1 package needs its own checks and checksum binding.

## Scope and known limitations

This is a controlled synthetic-sound research toolkit. Learned matches remain
approximate. Spectral distance is not perceptual acceptance, and retained failure
examples include substantial pitch errors even when distance improves. Limited
informal audition feedback was received on 2026-09-24. On 2026-11-10, the listener
reported no obvious audio errors in the review examples, but significant
matching misses across them. This is completed listening with an unresolved
matching-quality concern, not a perceptual-acceptance pass. The current
[review record](v2-listening-review.md) retains the feedback and investigation;
the earlier acceptance and archive records preserve their original dated status.

All actors share ceilings of 32 candidate evaluations and 64 decisions. Actual
usage differs: fixed uses one evaluation, inverse uses two, and search may use
32. The initial candidate, repeats, cache hits, and restoring an earlier patch
all consume evaluations. Submission means the protocol finished; it is not a
claim of sound quality.

Native Windows/macOS and new-task CUDA execution remain unqualified.
External recordings, live audio, DAW/VST hosting, effects, a packaged
UI, hosted-model adapters, and new-task PPO are outside this release. Custom
Python actors are cooperative code, not sandboxed plugins.

The final test memberships are consumed evidence. They support reproduction of
the recorded results; future model selection needs new held-out data. See the
[inverse method](v2-inverse.md), [calibration](v2-calibration.md), and
[acceptance record](v2-acceptance.md) for exact criteria and evidence boundaries.
