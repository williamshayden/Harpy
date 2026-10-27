# harPY 2.0.0 release candidate

Status on 2026-10-26: prepared locally for an **experimental known-synth research
release**, not yet published. The publication date and final release links are
pending. Package version `2.0.0` does not imply that a public release exists.

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

Alternatively, install the prepared wheel from the checkout:

```bash
python -m pip install outputs/v2-release-20261026/harpy_audio-2.0.0-py3-none-any.whl
```

For optional inverse training, use `python -m pip install '.[learn]'` from the
checkout, or install the wheel with its extra:

```bash
python -m pip install 'outputs/v2-release-20261026/harpy_audio-2.0.0-py3-none-any.whl[learn]'
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

## Artifacts and publication

The current release preparation uses `outputs/v2-release-20261026/` for:

- `harpy_audio-2.0.0-py3-none-any.whl` and `harpy_audio-2.0.0.tar.gz`.
- `SHA256SUMS`, a flat checksum file for the prepared artifacts.
- `publication-index.json`, the artifact inventory for publication.
- `qualification.json`, the current release verification record.

The retained scientific archive and installed-package proof are separate evidence
assets. The [acceptance record](v2-acceptance.md) identifies their original local
paths; the [archive guide](../research/sound_matching/README.md) explains their
contents, checksums, and reproduction commands. Original study artifacts remain
unchanged. Release preparation does not publish them.

Once the release is published, download the wheel or source distribution and
the accompanying checksum file from that release's assets. Download all files
listed in `SHA256SUMS` into one directory and run `sha256sum -c SHA256SUMS` there,
then install the downloaded wheel with:

```bash
python -m pip install ./harpy_audio-2.0.0-py3-none-any.whl
# Or install the downloaded source distribution:
python -m pip install ./harpy_audio-2.0.0.tar.gz
```

No public v2 download URL or package-index installation is asserted here.

## Scope and known limitations

This is a controlled synthetic-sound research toolkit. Learned matches remain
approximate. Spectral distance is not perceptual acceptance, and retained failure
examples include substantial pitch errors even when distance improves. Limited
informal audition feedback was received on 2026-09-24; formal representative and
failure-case listening acceptance remains open. The earlier acceptance and
archive records preserve the review status at the time they were written.

All actors share ceilings of 32 candidate evaluations and 64 decisions. Actual
usage differs: fixed uses one evaluation, inverse uses two, and search may use
32. The initial candidate, repeats, cache hits, and restoring an earlier patch
all consume evaluations. Submission means the protocol finished; it is not a
claim of sound quality.

Native Windows/macOS, new-task CUDA execution, and hosted CI execution remain
unqualified. External recordings, live audio, DAW/VST hosting, effects, a packaged
UI, hosted-model adapters, and new-task PPO are outside this release. Custom
Python actors are cooperative code, not sandboxed plugins.

The final test memberships are consumed evidence. They support reproduction of
the recorded results; future model selection needs new held-out data. See the
[inverse method](v2-inverse.md), [calibration](v2-calibration.md), and
[acceptance record](v2-acceptance.md) for exact criteria and evidence boundaries.
