# Harpy

Harpy is a headless toolkit for reproducible audio-control research. It gives researchers a deterministic sine renderer, a small Gymnasium tuning task, classical baselines, and optional learned models whose results can be inspected and reproduced.

The task is deliberately narrow: tune one clean, procedurally generated sine through bounded musical controls, then explicitly submit. Harpy makes it possible to separate perception, planning, and learning failures. It does not yet establish reliable tuning of recorded, noisy, or polyphonic audio.

The [1.0.0 release notes](https://github.com/williamshayden/Harpy/blob/v1.0.0/CHANGELOG.md) describe the supported scope and migration from the retired development workbench.

## First experiment

Use Python 3.12. Install a downloaded release wheel and run the baseline comparison:

```bash
python -m pip install ./harpy_audio-1.0.0-py3-none-any.whl
harpy-sine-gym --episodes 10 --seed 0 > baseline.json
python -m json.tool baseline.json
```

After a package-index release, use `python -m pip install harpy-audio` instead. An installed package does not require `uv` or a Git checkout.

The baseline command writes deterministic JSON with separate results for Random, Spectrum Peak, Oracle, and Reward Search. Read `submitted_success_rate`, final error, action count, and truncation together. Reaching the right pitch without submitting is not success. Oracle and Reward Search use different observation tracks; their results must not be pooled with spectrum-based actors.

## Train, inspect, and understand a result

The optional training stack contains PyTorch and Stable-Baselines3:

```bash
python -m pip install './harpy_audio-1.0.0-py3-none-any.whl[train]'
# After a package-index release: python -m pip install 'harpy-audio[train]'

harpy-sine-learn train-pitch \
  --profile smoke --seed 0 --output runs/pitch-smoke --device cpu

harpy-sine-learn evaluate runs/pitch-smoke \
  --output runs/pitch-smoke-report.json > /dev/null
harpy-sine-learn summarize runs/pitch-smoke-report.json

harpy-sine-learn diagnose runs/pitch-smoke \
  --suite smoke --output runs/pitch-smoke-diagnostics.json > /dev/null
harpy-sine-learn summarize runs/pitch-smoke-diagnostics.json --format markdown \
  > runs/pitch-smoke-findings.md

harpy-sine-learn run runs/pitch-smoke --seed 123
harpy-sine-learn run runs/pitch-smoke --seed 123 --with-provenance \
  > runs/pitch-smoke-trace.json
```

These shell examples use POSIX redirection. Progress and completion messages go to stderr. Evaluation and diagnosis write canonical JSON to stdout and, when requested, the same bytes to a file. `summarize` reads saved evidence without loading a model or requiring the training stack. It can produce a compact text readout or Markdown for a research notebook or README.

Artifact directories and evaluation/diagnostic output files are create-only: choose a new path for every run. Keep reports outside their input artifact directories. Shell redirection itself follows your shell's overwrite rules. If training is interrupted, retain the incomplete artifact for inspection and retry at a fresh path.

The smoke profile checks that the engineering workflow works; low model accuracy is expected and is not a scientific result. CPU is the default. CUDA training must be explicitly requested and fails when unavailable. A smoke model's `run` output demonstrates its behavior, not checkpoint-quality tuning.

Model `.pt` and `.zip` files are trusted-local artifacts. The CLI warns before loading them. Only load your own artifacts or models from sources you trust.

## Everyday research versus scientific checkpoints

Inspect one pitch checkpoint without assembling a formal three-seed cohort:

```bash
harpy-sine-learn train-pitch \
  --profile checkpoint --seed 7 --output runs/pitch-experiment --device cpu
harpy-sine-learn evaluate runs/pitch-experiment --exploratory \
  --output runs/pitch-experiment-report.json > /dev/null
harpy-sine-learn diagnose runs/pitch-experiment --exploratory --suite iid \
  --output runs/pitch-experiment-diagnostics.json > /dev/null
harpy-sine-learn summarize runs/pitch-experiment-report.json --format markdown
```

Exploratory evaluation and diagnosis accept one complete pitch artifact, including arbitrary training seeds and artifacts from installed packages or modified source. Their results are explicitly ineligible for the frozen scientific criterion. The model and evaluation are real; the distinction concerns the strength of the claim.

Installed-package artifacts record a `package_snapshot` source identity: a digest of package contents and the distribution version when available. They do not claim a Git commit or a clean checkout. Historical source-checkout artifacts retain their original Git provenance and serialization. Older Harpy readers cannot read the new package-source variant; use this release to inspect it.

Formal evaluation retains the declared CPU cohort with exact seeds 0, 1, and 2, clean committed source and locked dependencies. E.1's separate homogeneous CUDA cohort additionally requires matching evaluator source and CPU evaluation. See the [research workflow and protocol guide](https://github.com/williamshayden/Harpy/blob/v1.0.0/docs/research-workflow.md) for complete commands, eligibility, and historical protocol links.

`run --json` retains the original canonical episode format. `run --with-provenance` writes a versioned JSON envelope with the manifest/model digests, trainer, training seed, source identity, and execution device. Prefer the latter when saving or sharing traces.

## What the current results show

The learned pitch estimator is a small spectrum-only network followed by the exact symbolic planner. It is a learned-perception experiment, not an end-to-end reinforcement-learning result. It has no hidden-pitch access, Spectrum Peak fallback, or evaluator rescue.

The recorded E.1 IID results are:

| Actor | Submitted within 5 cents | Truncated episodes | Mean final error |
| --- | ---: | ---: | ---: |
| Learned pitch, seed 0 | 94.0% | 15 / 250 | 47.556 cents |
| Learned pitch, seed 1 | 96.8% | 8 / 250 | 21.388 cents |
| Learned pitch, seed 2 | 100% | 0 / 250 | 1.980 cents |
| Spectrum Peak | 100% | 0 / 250 | 1.252 cents |

The declared reliability criterion was **not met**. Rare pitch aliases can cause repeated-state loops and large final errors. The successful seed is not selected as a substitute for the complete cohort. Even that seed's submitted accuracy within one cent is 2%, versus Spectrum Peak's 58.4%; its mean action count is lower, 27.324 versus 29.492.

These are preserved historical results, not a newly trained release cohort. The [E.1 acceptance record](https://github.com/williamshayden/Harpy/blob/v1.0.0/docs/verification/2026-08-28-milestone-e1-cuda-device-cohort-acceptance.md) contains full metrics, provenance, probes, and failure analysis. BC and PPO remain reproducible Milestone D controls; their declared scientific criteria also missed. Negative outcomes are useful research evidence, not claims of dependable tuning performance.

## Use the Python API

Render and analyze a sine without any UI or training dependency:

```python
from harpy.analysis import analyze
from harpy.synth import SynthEngine, SynthPatch
from harpy.synth.models import RenderConfig

render = RenderConfig(sample_rate_hz=48_000)
engine = SynthEngine(render, SynthPatch())
engine.note_on(440.0)
samples = engine.render(48_000)  # mono float32; choose each block's length explicitly
observation = analyze(samples, render.sample_rate_hz)
print(observation.peak_frequency_hz)
```

`analyze` uses the most recent complete FFT capture and rejects non-finite samples in that capture. The synth accepts finite positive frequencies strictly below Nyquist. Patch JSON describes sound parameters, not performance state or an application session. `harpy.synth.patch_json` reads strict schema-v1/v2 patches and writes canonical v2; `harpy.tuning.Tuning` handles frequency/note conversions.

Run the spectrum baseline through the public environment:

```python
import gymnasium as gym
import harpy.envs  # registers Harpy's environment IDs
from harpy.envs.baselines import spectrum_peak_plan

with gym.make("Harpy/SinePitch-v0") as env:
    observation, _ = env.reset(seed=0)
    for action in spectrum_peak_plan(observation):
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    print(env.unwrapped.episode_result)  # evaluator truth is available only after done
```

| Environment | Observation track |
| --- | --- |
| `Harpy/SinePitch-v0` | Normalized log-frequency spectrum with shape `(1961,)` |
| `Harpy/SinePitchOracle-v0` | Exact current-pitch coordinate |
| `Harpy/SinePitchRewardOnly-v0` | Controls, target, budget, and scalar feedback |

All tracks expose target, bounded controls, and remaining budget. Action IDs remain `0` Octave Down, `1` Semitone Down, `2` Cent Down, `3` Submit, `4` Cent Up, `5` Semitone Up, `6` Octave Up. Controls are independently bounded to octaves `-2..2`, semitones `-12..12`, and cents `-100..100`; they never carry into each other. Success requires an explicit submission within an inclusive five-cent tolerance and the 64-action budget. One-cent accuracy is reported separately.

Actor observations omit source pitch and exact error. This is a supported API boundary, not a security sandbox against Python code deliberately accessing private state. Every applied action resynthesizes a fresh sine from immutable source truth and controls; it is not recorded-audio pitch shifting.

## Develop and verify

Contributors use Python 3.12 and the committed `uv.lock`:

```bash
uv sync --locked --extra train
uv run --locked --extra train pytest
uv run --locked --extra train ruff check .
uv run --locked --extra train ruff format --check .
uv build
```

Use `uv sync` without the extra for the base package. Real training smoke tests require the extra; a skipped training test is not proof that learning works. The pitch smoke test builds and installs the wheel outside Git, then exercises training, reload, diagnosis, evaluation, and tracing. Verification is currently exercised on Linux/WSL; native Windows and macOS have not been qualified.

Harpy ships no desktop or browser UI. The retired Sine Lab design documents remain historical evidence. A future visualization on the author's personal website is separate from this package. Recorded audio, additional waveforms, chords, polyphony, and generalized experiment storage require new experiments and contracts.

The [project notebook](https://github.com/williamshayden/Harpy/blob/v1.0.0/docs/project-notebook.md) preserves motivation, references, and decisions. The [workflow guide](https://github.com/williamshayden/Harpy/blob/v1.0.0/docs/research-workflow.md) indexes historical protocols. Before v1, unused GUI-only envelope preview helpers and the ineffective `RenderConfig.block_frames` field were removed; external callers of those helpers must update.

Harpy is released under the [MIT License](https://github.com/williamshayden/Harpy/blob/v1.0.0/LICENSE), included in the wheel and source distribution.
