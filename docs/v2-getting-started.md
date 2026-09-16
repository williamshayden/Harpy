# Sound-matching development workflow

This `2.0.0.dev0` development branch contains the first executable slice of the
[v2 specification](v2-spec.md): deterministic synthesis, metered patch proposals,
classical search, and saved results. The public release is still v1. The new
objective is an uncalibrated research distance; these smoke examples establish
workflow correctness, not sound-quality qualification.

Inverse-model training, the calibration study, and resumable evaluation are
subsequent milestones in the [delivery plan](v2-plan.md). Existing v1 commands
continue to run when `--task` is omitted.
The [milestone verification record](v2-m1-verification.md) gives test scope,
actual smoke outcomes, and unqualified layers.

## Install the development checkout

Use Python 3.12 on Linux/WSL, from this branch's checkout:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The base installation supplies everything in this guide. Torch, Stable-Baselines3,
CUDA, model downloads, and credentials are unnecessary. Contributors can instead
use `uv sync --locked --group dev` for the pinned environment and checks.

## Run, hear, and inspect a candidate

```bash
harpy run --task sound-match-v1 --suite tone-smoke --episode tone-001 \
  --actor pattern-search --output runs/tone-example
harpy summarize runs/tone-example
harpy summarize runs/tone-example --format markdown
harpy summarize runs/tone-example --format csv
harpy summarize runs/tone-example --verify-replay
```

Output paths are create-only. A sound-matching output is a directory containing
validated experiment evidence, patches, and exported audio. Every valid proposal
has an audio hash and measured distance in the decision ledger. Target and
submitted WAVs let you compare the actual sounds; candidate patch JSON remains
editable. Detailed observations and intermediate WAVs are not necessary to read
the compact ledger.

For this single episode, open `audio/target-0000.wav` and
`audio/submitted-0000.wav` inside the result directory. Its editable patch is
`patches/submitted-0000.json`; `records.jsonl` contains the decisions. Exported
WAVs use lossless 32-bit float samples, with no gain normalization.

Ordinary summaries validate the ledger, saved-file hashes, and recomputed
aggregates. `--verify-replay` additionally renders each recorded candidate and
checks its audio hash and score. It costs another round of DSP work and requires
the recorded renderer/runtime contract; it does not load a model.

The default budget is 32 candidate evaluations and 64 decisions. The initial
patch consumes evaluation 1. `Propose(patch)` consumes one more evaluation even
for a repeated candidate; `Submit()` finishes with the currently heard patch.
Search must pay to restore an earlier candidate. Invalid proposals consume a
decision, retain the previous patch, and report a reason. Budget exhaustion
without an explicit submission is a truncation.

The `tone-001` example changes only the starting note. Other examples vary more
controls. A small search is not guaranteed to recover their sound or parameters.

## Compare matched examples

```bash
harpy evaluate --task sound-match-v1 --suite tone-smoke \
  --actor fixed --actor pattern-search --output runs/tone-comparison
harpy evaluate --task sound-match-v1 --suite chord-smoke \
  --actor fixed --actor random-search --seed 0 --output runs/chord-comparison
harpy evaluate --task sound-match-v1 --suite shaping-smoke \
  --actor pattern-search --output runs/shaping-comparison
```

`fixed` immediately submits the starting patch. `random-search` samples legal
patches with a recorded seed. `pattern-search` makes bounded coordinate proposals
and retains the best heard candidate. Their model-call count is zero; custom
actors with unknown counts must not claim zero.

Tone and chord suites each contain two explicit engineering examples. Shaping
contains three examples with pitch, envelope, or brightness editable. Locked
fields cannot change. Notes are an unordered set of one or three distinct pitches;
input order is canonicalized before rendering and comparing locked fields.
These examples are public, consumed smoke membership; do not treat their scores
as a held-out evaluation or combine them into a perceptual success rate.

Inspect initial, last, submitted, and best-audition distances separately, alongside
cost, invalid decisions, errors, and truncations. The multiresolution spectral
distance is supplemented by envelope, level, and pitch diagnostics. Different
patches may produce similar sounds; matching knob labels alone is not success.

## Use the Python API

```python
from dataclasses import replace
from pathlib import Path

from harpy.tasks.sound_match.actors import builtin_actor_spec
from harpy.tasks.sound_match.models import EpisodeSpec
from harpy.tasks.sound_match.patch import default_patch
from harpy.tasks.sound_match.renderer import render
from harpy.tasks.sound_match.results import save_result, summarize
from harpy.tasks.sound_match.runner import run

start = default_patch()
target = replace(start, notes_cents=(6100,))
audio = render(target)  # 96,000 mono float32 samples at 48 kHz
episode = EpisodeSpec("example-001", target_patch=target, initial_patch=start)
result = run(episode, builtin_actor_spec("pattern-search"))
save_result(result, Path("runs/python-example"))
print(summarize(result))
```

The evaluator knows `target_patch`; the actor sees only target/candidate samples,
its current patch, public timing/bounds, the editable mask, and remaining budgets.
Each actor factory creates fresh episode state. Shared immutable weights are
allowed; observation mutations cannot change later observations or other actors.
Python extensions are a cooperative research interface, not a security sandbox.

## Limits of this milestone

Synthesis is fixed to two-second mono clips with shared timbre/envelope and one
or three simultaneous notes. There is no input recording, live audio, VST, UI,
variable timing, effects, or learned sound-matching model in this milestone.
`harpy train pitch` and `harpy train ppo` still refer to the released v1 task.

The new runner saves completed results; durable interrupted-run recovery and
resume remain planned. Retain an incomplete failed output for diagnosis and
choose a new path on retry. There is no sound-matching success threshold or
generalization claim until calibration and frozen evaluation have been completed.
