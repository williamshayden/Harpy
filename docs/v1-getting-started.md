# Getting started with harPY

> This guide targets the revised v1 source on `main`, currently Unreleased. The original `v1.0.0` tag uses the historical interfaces.

harPY (harmonic research in python) is a headless toolkit for comparing audio representations, pitch estimators, and tuning controllers. Its current environment uses a single synthesized tone so the source, target, controls, and corruption conditions can be specified exactly. Start with a classical comparison, read its saved results, and inspect an episode before trying optional training or a custom actor.

## Capabilities

- Compare classical actors on waveform and spectrum observations under controlled level, phase, and noise conditions.
- Use the shipped supervised pitch reference, train a pitch estimator, or explore experimental PPO.
- Save explicit episode membership, actor identities, provenance, outcomes, and optional decision traces; read them without loading models.
- Bring a custom Python actor factory while keeping the task and evaluation consistent.

The four operations form a simple workflow: `evaluate` compares actors, `run` records one traced episode, `summarize` reads saved results, and `train` creates an optional learned artifact. The base installation is enough for the first comparison below. V1 covers a controlled single-sine rendering family, not microphone or instrument recordings.

## Install from the revised checkout

Use Python 3.12 on Linux or WSL, the qualified environment. Clone the current source before running the installation commands below:

```bash
git clone https://github.com/williamshayden/Harpy.git
cd Harpy
```

The old `v1.0.0` tag provides historical interfaces. Install this revision from source; a revised PyPI release has not been published. Existing users should run the commands from their updated checkout rather than clone into a directory that already exists.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
harpy --help
```

The base installation supplies classical actors, experiments, and result readers. Add dependencies only for the work you want to do, from the same checkout:

| Installation | Enables |
| --- | --- |
| `python -m pip install .` | Classical comparisons and all ordinary result readouts |
| `python -m pip install '.[pitch]'` | The shipped reference and supervised pitch training, using Torch |
| `python -m pip install '.[train]'` | Pitch plus experimental PPO training, using Torch and Stable-Baselines3 |

Installed execution needs neither Git nor `uv`. The commands below use distinct output paths. Harpy creates new results and artifacts; choose another path when repeating a command.

## Compare two ways of hearing

```bash
harpy evaluate --actor waveform-fft --actor spectrum-peak \
  --suite smoke --output runs/guide-classical-smoke.json
harpy summarize runs/guide-classical-smoke.json
harpy summarize runs/guide-classical-smoke.json --format markdown
harpy summarize runs/guide-classical-smoke.json --format csv
```

This runs six shared clean episodes, two from each source partition, for each actor. `waveform-fft` estimates from the waveform; `spectrum-peak` uses Harpy's spectrum representation. Both commit to a plan after one estimate. Their different observation tracks remain labeled in the report.

The terminal summary is a quick readout. JSON is the saved experiment: membership, conditions, actor identities, provenance, and individual outcomes. `summarize` validates that file and recomputes its metrics. Markdown fits a research note; CSV supports further analysis. These commands print the views without changing the JSON.

Smoke is an installation and workflow check. It is too small to establish robustness or select a model.

## Inspect a complete episode

```bash
harpy run --actor waveform-fft --source-cents 6857 \
  --target-note-index 13 --condition noise-10db \
  --output runs/guide-noise-trace.json
harpy summarize runs/guide-noise-trace.json --format markdown
```

The source coordinate is measured in cents, with MIDI note 69 at 6900. Target indices 0–24 represent MIDI notes 48–72, so index 13 requests MIDI 61. `run` always saves the public decision trace. To inspect the estimate and each subsequent action:

```python
from pathlib import Path
from harpy.envs.models import PitchAction
from harpy.experiments import load_result

result = load_result(Path("runs/guide-noise-trace.json"))
for step in result.records[0].trace:
    print(step["step"], PitchAction(step["action"]).name, step["estimated_candidate_cents"])
```

Later steps show `None` for the estimate because this actor executes its initial plan without another inference. That distinction helps separate perception errors from control behavior.

## Read the outcome before scaling up

Success requires explicit submission within five cents, inclusive, within the 64-action budget. Being close when the budget expires is not success. Inspect initial one/five-cent accuracy alongside final submitted success, p99 and maximum error, invalid actions, truncations, and action/inference counts. A better estimate and a cheaper controller are different improvements.

Keep source partitions, observation tracks, seeds, and corruption conditions separate. The `clean` suite has 650 established episodes; `robustness` repeats them under eight conditions. The 1,000-pair `confirmation` suite reproduces the release qualification on fixed, already evaluated pairs. A new final confirmation claim needs fresh frozen membership; do not use the existing suite for development selection.

Harpy measures a deterministic single-sine rendering family with static level, phase, and noise variations. These comparisons are useful controlled evidence, not validation on microphones, instruments, or chords. The [v1 specification](v1-spec.md) defines the task; the [waveform FFT notes](v1-waveform-fft.md) explain that baseline's evidence and limits.

## Try the learned reference

After installing the `pitch` extra:

```bash
harpy evaluate --actor reference --actor spectrum-peak \
  --suite smoke --output runs/guide-reference-smoke.json
harpy summarize runs/guide-reference-smoke.json
```

`reference` loads the packaged seed-0 checkpoint. It requires no download or training. Its recorded clean qualification is specific to that artifact and protocol; it does not establish superiority over the classical baseline or guarantee strong-noise performance.

To exercise training and checkpoint reload yourself:

```bash
harpy train pitch --profile smoke --seed 0 --output runs/guide-pitch-model
harpy evaluate --actor runs/guide-pitch-model --actor spectrum-peak \
  --suite smoke --output runs/guide-pitch-results.json
```

With the `train` extra, the equivalent PPO workflow is:

```bash
harpy train ppo --profile smoke --seed 0 --output runs/guide-ppo-model
harpy evaluate --actor runs/guide-ppo-model \
  --suite smoke --output runs/guide-ppo-results.json
```

These use CPU by default. Pitch selects its checkpoint using clean validation; PPO saves its final training step. Smoke training checks the mechanics, not learned quality. A completed PPO evaluation can report failed submissions or budget exhaustion: PPO is experimental, with no promised performance threshold.

## Bring an actor factory

A factory creates fresh controller state for every episode. Here is a runnable waveform adapter that reuses the spectrum baseline's decoder and controller. Save it as `compare_adapter.py` in your working directory:

```python
from pathlib import Path
from harpy.envs.models import ObservationMode
from harpy.envs.spectrum import encode_log_spectrum
from harpy.experiments import ActorSpec, evaluate, save_result, spectrum_peak_actor_spec
from harpy.experiments.protocols import smoke_episodes

baseline = spectrum_peak_actor_spec()

class WaveformAdapter:
    def __init__(self):
        self.controller = baseline.factory()
        self.spectrum = None

    def decide(self, observation):
        if self.spectrum is None:
            self.spectrum = encode_log_spectrum(observation["waveform"])
        view = {key: value for key, value in observation.items() if key != "waveform"}
        view["spectrum"] = self.spectrum
        return self.controller.decide(view)

actor = ActorSpec(
    name="guide-waveform-adapter", factory=WaveformAdapter,
    observation_mode=ObservationMode.WAVEFORM,
    estimator="guide-waveform-spectrum-v1",
    decoder=baseline.decoder, controller=baseline.controller,
    artifact_paths=(Path(__file__).resolve(),),
)
result = evaluate(smoke_episodes(), (baseline, actor))
save_result(result, Path("runs/guide-custom-adapter.json"))
```

```bash
python compare_adapter.py
harpy summarize runs/guide-custom-adapter.json
```

This reproduces the spectrum method from waveform input; it introduces no new estimator claim. Change the encoding step to explore a representation while keeping control fixed. The script is included in the recorded artifact hashes. A factory can also share immutable model weights while giving each episode its own controller.

For the experiments behind these choices, see the [reference qualification](verification/2026-09-08-v1-respec-qualification.md) and [waveform FFT evidence](v1-waveform-fft.md).
