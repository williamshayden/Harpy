# Sound matching with harPY v2

The `sound-match-v1` task compares short sounds made by a fixed synthesizer.
An actor hears the target and its current candidate, proposes an editable patch,
hears the rendered result, and explicitly submits its current candidate. The
evaluator retains sounds, patches, decisions, costs, and continuous distances.
The task ID is independent of the package's `2.0.0` version.

Existing v1 pitch commands remain the default when `--task` is omitted.
See the [specification](v2-spec.md), [calibration record](v2-calibration.md),
[inverse method](v2-inverse.md), and [acceptance record](v2-acceptance.md)
for the protocol, measured results, and qualified environment.

## Install

Use Python 3.12 on Linux/WSL from this checkout, or install the built wheel:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
# Optional inverse-model training and inference:
python -m pip install '.[learn]'
```

The base package renders, evaluates classical actors, reads results, and replays
evidence without Torch or Stable-Baselines3. The `learn` extra adds Torch;
`pitch` retains its v1 meaning, and `train` adds v1 PPO. Sound matching uses CPU.
No pretrained-model download, service, or credential is required.
Contributors can use `uv sync --locked --group dev`; see the
[inverse guide](v2-inverse.md) for the explicit CPU Torch installation procedure.
Dependency versions and source identities are recorded with the evidence.

## Run, hear, and inspect a sound

```bash
harpy run --task sound-match-v1 --suite tone-smoke --episode tone-001 \
  --actor pattern-search --output runs/tone-example
harpy summarize runs/tone-example
harpy summarize runs/tone-example --format markdown
harpy summarize runs/tone-example --format csv
harpy summarize runs/tone-example --verify-replay
```

Output paths are create-only. A bundle contains `manifest.json`, `records.jsonl`,
`commits.jsonl`, `attempts.jsonl`, `complete.json`, and `artifacts/`.
Artifacts use their SHA-256 as filenames; repeated content is shared. Each
committed record identifies its target WAV, submitted WAV, and patch by role:

```python
from pathlib import Path
from harpy.tasks.sound_match.durable import iter_records

bundle = Path("runs/tone-example")
for record in iter_records(bundle):
    print(record["key"], record["status"])
    for artifact in record["artifacts"]:
        print(artifact["role"], bundle / artifact["path"])
```

WAV exports are lossless mono float32 at 48 kHz, without gain normalization.
Compare target and submitted WAVs in an audio player. Render a saved patch with:

```python
import json
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import render

patch_file = next(a for a in record["artifacts"] if a["role"] == "submitted_patch")
document = json.loads((bundle / patch_file["path"]).read_text())
audio = render(Patch.from_dict(document["patch"]))  # 96,000 float32 samples
```

Ordinary readout validates committed files, objective components, membership,
and recomputed aggregates. Replay additionally renders recorded candidates and
checks audio hashes and scores, without loading the actor model. Exact replay
requires the recorded renderer/runtime contract.

## Compare actors and train an inverse model

```bash
harpy evaluate --task sound-match-v1 --suite chord-smoke \
  --actor fixed --actor random-search --seed 0 --output runs/chord-comparison
harpy evaluate --task sound-match-v1 --suite shaping-smoke \
  --actor pattern-search --output runs/shaping-comparison
harpy train inverse --profile quick --seed 0 --output runs/inverse-quick
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor inverse=runs/inverse-quick \
  --actor inverse-refine=runs/inverse-quick \
  --actor pattern-search --output runs/inverse-smoke
```

`fixed` submits the initial patch. `random-search` samples legal patches with a
recorded seed. `pattern-search` proposes coordinate changes and retains its best
heard candidate. Each has zero model calls.

`quick` performs real optimization on a small training/validation selection and
checks the workflow. `reference` trains on the frozen 2,016 training and 192
validation examples for six epochs. It selects the earliest checkpoint with the
lowest equally weighted mean across eight validation cells. Seeds 0, 1, and 2
must all be reported. A quick artifact has no reference-quality claim.

The inverse actor makes one model proposal. `inverse-refine` then uses ordinary
pattern search within the remaining budget. Use `LABEL=inverse:PATH` or
`LABEL=inverse-refine:PATH` for comparisons across seeds. Model quality and
workflow correctness are separate; see the acceptance record for measured results.

The prepared [evidence archive](v2-acceptance.md#local-distributions-and-evidence)
retains the selected models under `models/seed0`, `models/seed1`, and
`models/seed2`. After extraction, an already trained model can be used directly:

```bash
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor inverse=EVIDENCE_ROOT/models/seed0 \
  --actor inverse-refine=EVIDENCE_ROOT/models/seed0 \
  --output runs/pretrained-inverse-smoke
```

Seed 0 is shown for a reproducible starting point; the study reports all three
seeds and does not select the best final-test result.

The shared ceilings are 32 candidate evaluations and 64 decisions. Actual
expenditure differs: fixed uses one evaluation, inverse uses two, and search may
use 32. The initial patch consumes evaluation 1. Each legal proposal consumes another evaluation, including
cache hits and repeats. Restoring an earlier best candidate costs an evaluation.
Invalid proposals consume a decision and retain the current patch. Exhausting
decisions without explicit submission is a truncation. Submission is a protocol
action; it does not imply a perceptually successful match.

## Resume and relocate a study

Every sound-matching CLI evaluation uses a durable bundle. After interruption:

```bash
harpy summarize runs/chord-comparison --allow-incomplete
harpy evaluate --resume --output runs/chord-comparison
```

Resume restores omitted options and runs only missing members. Committed errors
and truncations remain completed observations. An interrupted attempt remains
visible and its unfinished cost is unknown. Source, runtime, membership,
configuration, actor metadata, and artifact bytes must match. Concurrent writers
are rejected. Training artifacts are create-only and do not support resume.

Local model paths are separate from scientific identity in `locators.json`.
A copied bundle can be read without the original model files. To resume after
moving a model, repeat the complete actor list with the same labels and new paths:

```bash
harpy evaluate --resume --output runs/inverse-smoke \
  --actor inverse=/new/path/inverse-quick \
  --actor inverse-refine=/new/path/inverse-quick \
  --actor pattern-search
```

The model bytes must match. Publication archives neutralize machine-specific
locators. V1 evaluation still writes ordinary JSON by default; opt into journals:

```bash
harpy evaluate --bundle --suite smoke --actor waveform-fft --output runs/pitch-bundle
harpy evaluate --resume --output runs/pitch-bundle
harpy summarize runs/pitch-bundle --verify-replay
```

## Membership and interpretation

`tone-smoke`, `chord-smoke`, and `shaping-smoke` are consumed engineering examples.
`calibration`, `validation`, `iid-test`, and `compositional-test` are explicit,
hash-bound dataset suites. The frozen study compares all declared seeds on
matched membership. Once final tests are consumed, use them to reproduce that
evidence; further model selection needs a new preregistered test set.
The completed [all-seed results](v2-results.md) retain both final test splits and
the descriptive comparisons alongside their limitations.

Read single-note reconstruction, chord reconstruction, and each shaping mask
separately. Compare initial, last, submitted, and best-audition distance alongside
model calls, evaluations, DSP renders, invalid decisions, errors, and truncations.
The compositional split holds out target combinations; public starting patches
can expose some of those combinations. Details and alias checks are in the
calibration record.

The multiresolution spectral distance is frozen as a continuous research measure.
Envelope, level, and pitch-set diagnostics explain differences. No perceptual
acceptance cutoff is asserted. Distinct patches can sound similar, and lower
objective distance is not a listener assessment.

## Python extension and boundaries

Use `ActorSpec` with a fresh-controller factory and public `Observation`,
`Propose`, and `Submit` types. Only the evaluator receives the target patch.
Actors receive target/current audio, the current patch, bounds, editable fields,
timing, and budgets. Declare configuration and seeds in actor metadata and
immutable model files as artifact paths. Durable factories need inspectable
Python source. Custom actors are a cooperative research interface.

The executable [custom actor example](../examples/sound_match_actor.py) shows a
fresh controller making one legal brightness proposal and then submitting its
heard candidate. Save or copy it as a Python file so the durable writer can
inspect and hash its defining source, then run it with the base package installed:

```bash
python examples/sound_match_actor.py runs/custom-sound
harpy summarize runs/custom-sound --verify-replay
```

This is an API example with no matching-quality claim. Its installed-package
check submitted both smoke targets with two evaluations each, zero invalid
actions, and no Torch installation.

`evaluate_bundle`, `read_bundle`, `iter_records`, and `summarize_bundle` are exposed
from `harpy.tasks.sound_match`. The earlier in-memory `run`/`evaluate` and M1
`save_result`/`load_result` format remain available. See [migration](v2-migration.md).

Synthesis is fixed to two-second clips with one or three distinct notes, a shared
waveform, brightness, ADSR envelope, and gain. Notes are canonicalized before
identity and rendering; locked fields cannot change. External WAV targets, live
audio, VST, UI, effects, new-task PPO, and variable-duration synthesis are outside
this release.
