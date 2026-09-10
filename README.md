# harPY

**harmonic research in python**

harPY is a headless Python toolkit for researchers and developers comparing audio
representations, pitch estimators, and tuning controllers. It supplies a controlled
task, classical and learned baselines, matched experiments, and saved results so
you can change a method and inspect what changed.

The current environment is single-tone pitch matching. A synthesized sine gives
the evaluator an exact source pitch, while the actor must estimate it from audio
and use bounded controls to reach a target note. Controlled level, phase, and noise
variations help separate representation errors from estimation and control behavior.

Bring a custom Python actor or use a built-in baseline. For example, compare an
alternative waveform encoding under the same controller, test a learned estimator
against a classical method, or examine committed planning versus repeated
replanning. Custom actors and explicit episode membership are supported within
the current task; other audio sources and tasks require additional implementation.

**The v1 release is package 1.0.0, tagged `v1`.** It implements the revised v1
research protocol. The older `v1.0.0` tag is a historical development checkpoint,
not the public release; it preserves the earlier workflows and evidence.

## Install

Download the wheel, source archive, and checksums from the
[v1 release](https://github.com/williamshayden/Harpy/releases/tag/v1).
The wheel installs into a Python 3.12 environment with `python -m pip install ./harpy_audio-1.0.0-py3-none-any.whl`.

Use Python 3.12 on Linux/WSL, the qualified environment. Clone the release source,
then create and activate a virtual environment:

```bash
git clone --branch v1 https://github.com/williamshayden/Harpy.git
cd Harpy
python3.12 -m venv .venv
source .venv/bin/activate
```

Then choose the dependencies you need:

```bash
python -m pip install .          # Classical actors and result readers
python -m pip install '.[pitch]'  # Also supervised pitch, using Torch
python -m pip install '.[train]'  # Also PPO, using Torch and Stable-Baselines3
```

A built wheel can replace `.`. The package distribution is `harpy-audio`; the CLI
and Python imports remain `harpy`. Installed execution needs neither Git nor `uv`;
pitch does not require Stable-Baselines3. Native Windows/macOS are not yet
qualified. CPU evaluation and CUDA training evidence are separate; CUDA must be
explicitly requested and available.

## Run a first experiment

Compare two classical actors on the same six clean smoke episodes, then inspect
the saved result:

```bash
harpy evaluate --actor waveform-fft --actor spectrum-peak \
  --suite smoke --output runs/first-comparison.json
harpy summarize runs/first-comparison.json
harpy summarize runs/first-comparison.json --format markdown
harpy summarize runs/first-comparison.json --format csv
harpy run --actor waveform-fft --source-cents 6064 --target-note-index 12 \
  --output runs/first-trace.json
```

`evaluate` compares actors; `run` saves one episode with its decision trace;
`summarize` validates saved results without loading models; `train` creates a
learned artifact. Smoke checks the workflow, not model quality.

The task allows 64 actions using independent octave, semitone, and fine-cent
controls. Success requires **explicit submission within five cents, inclusive**.
Being close without submitting is not success.

Outputs are create-only: choose a new result file or artifact directory for each
run. Interrupted training leaves an incomplete directory for inspection, not a
valid artifact. Shell redirection follows the shell's overwrite rules.

## What you can compare

- Classical actors: `spectrum-peak`, `spectrum-peak-replan`, `waveform-fft`,
  `oracle`, `reward-search`, and `random`. Spectrum Peak remains the default.
- The packaged seed-0 `reference`, or your own pitch/PPO artifact directory.
  Use `LABEL=PATH` to distinguish artifacts with the same training seed.
- Custom Python actor factories with fresh episode state and optionally shared
  immutable model weights. See the [guide](docs/v1-getting-started.md) and
  [waveform adapter example](examples/custom_actor.py).
- Clean, level, phase, and white-noise conditions with matched nuisance identity.
  Noise stays fixed within an episode: these are static-corruption experiments.

With the `pitch` extra installed, use the shipped reference without training:

```bash
harpy evaluate --actor reference --actor spectrum-peak \
  --suite smoke --output runs/reference-smoke.json
```

The guide covers `harpy train pitch` and `harpy train ppo`, checkpoint reload,
and custom Python actors. Pitch trains on clean coordinates
and selects on clean validation. PPO is experimental, with no release-score gate;
its reset sources use the training partition, though control trajectories can
visit other coordinates. Load PPO archives from trusted producers: hashes
establish identity, not trust.

Synthesis, analysis, tuning, and Gymnasium remain supporting APIs. Waveform and
spectrum actors receive the target, controls, and budget, without source pitch,
true error, or nuisance seed. Oracle and Reward Search use different information;
keep those comparisons separate from audio-only actors.

## Read the evidence

Results retain membership, conditions, actor components, source/model identity,
runtime, and terminal records. Readers recompute metrics and evaluation rejects
changed artifacts. Read initial one/five-cent accuracy alongside submitted
success, p99/max error, invalid actions, truncations, and action/inference cost.
Keep observation tracks, seeds, partitions, and conditions separate.

The clean suite has 650 episodes; robustness repeats them under eight conditions.
The 1,000 confirmation combinations were frozen before reference qualification
within the known rendering family. They are now consumed evidence for reproducing
that qualification; future development needs fresh membership for a new final
confirmation claim.

Reference seeds 0, 1, and 2 each passed **650/650 clean benchmark and 1,000/1,000
clean confirmation episodes**, with no invalid actions or truncations. All three
and matched Spectrum Peak passed the level, phase, and 30 dB noise conditions;
each had failures at 10 dB and under combined corruption. There is no demonstrated
learned advantage or one-cent guarantee. These results qualify the recorded
reference artifacts, not arbitrary trained models.

Separately, the waveform FFT prototype passed **600/600 pairs in each of eight
conditions**, matching every clean Spectrum Peak estimate. This supports the
optional baseline, not learned-model gains or universal audio robustness.
Recorded instruments, chords, live microphone tuning, and UI are outside v1.

The source includes the seed-0 checkpoint, protocol membership, and compact
results. The [v1 release](https://github.com/williamshayden/Harpy/releases/tag/v1)
also includes `spectrum-study.json.gz`: all 14,400 records from the representation
study, with a manifest documenting hashes and the removal of local checkout
prefixes from artifact path labels. Unpack it with `gzip -dk spectrum-study.json.gz`
and read it with `harpy summarize spectrum-study.json`.
Full reference-qualification experiments and seed-1/2 artifacts remain local;
reproducing those studies requires their archives in addition to a fresh clone.
The qualification record identifies their hashes and scope.

## Documentation

- [Getting started](docs/v1-getting-started.md): capabilities, installation, traces,
  training, custom actors, and result interpretation.
- [Contributing](docs/v1-contributing.md): development setup, focused checks, and
  the issue and pull-request workflow.
- [Specification](docs/v1-spec.md) and [acceptance checklist](docs/v1-acceptance.md):
  the authoritative contract and verification scope.
- [Reference qualification](docs/verification/2026-09-08-v1-respec-qualification.md)
  and [waveform FFT evidence](docs/v1-waveform-fft.md): results and limitations.
- [Migration](docs/v1-migration.md): historical execution at `v1.0.0`; old reports
  remain readable through `harpy summarize` in text or Markdown.

MIT licensed. See [LICENSE](LICENSE).
