# Harpy project notebook

Last updated: 2026-08-07

Status: exploratory notes, not an approved design or implementation plan.

This is the living record for the project: what we think Harpy is, what prior work exists, what decisions have been made, what remains uncertain, and what would make the work scientifically honest and useful.

## Working thesis

Harpy should be a model-agnostic audio-control benchmark in which an actor receives a reference audio target and a candidate audio asset, then uses a deliberately constrained set of pitch-manipulation actions to tune the candidate toward the reference.

The point is not to discover a new formula for pitch. The point is to measure whether a learned actor can acquire and generalize an effective audio-to-action policy while being restricted to the same hierarchical controls a person or tool-using agent might have.

The environment should make simple, solved cases cheap enough to validate the entire experimental pipeline, then preserve the same interface as the audio becomes less pitch-salient or more polyphonic.

## Critical correction: audio transformation, not synth programming

The actor does **not** control oscillator, filter, envelope, or synthesizer-patch parameters. It manipulates an immutable audio asset through pitch tools.

A synthesizer can still serve two useful roles:

1. Generate reproducible labeled source audio for procedural tests.
2. Let a human create realistic held-out assets in Ableton, Serum, or another DAW while retaining exact musical metadata.

The environment should always derive each candidate render from the original source plus the current cumulative pitch offset. It should never repeatedly pitch-shift the previous output, because cumulative processing artifacts would turn action order into an unintended hidden variable.

## What one episode may look like

Working hypothesis for the first benchmark:

1. Select or generate an immutable base asset.
2. Produce a reference render and an initially detuned candidate render from that same asset.
3. Give the actor the permitted observation, current tool state, and remaining action budget.
4. Let the actor apply one constrained pitch action at a time.
5. Re-render from the immutable base asset after each action.
6. Terminate when the candidate is inside the tuning tolerance; truncate when the action budget is exhausted.
7. Record accuracy, trajectory, latency, environment steps, model metadata, and any available compute telemetry.

The target offset and source metadata are evaluator truth. They must not accidentally leak through filenames, array lengths, loudness, phase, metadata fields, episode IDs, or debug information.

## Candidate action surface

The exact vocabulary is still open. A concrete starting point is a discrete action set with positive and negative versions of hierarchical pitch moves:

- octave: 1,200 cents;
- semitone: 100 cents;
- coarse detune: perhaps 25 cents;
- fine detune: perhaps 5 cents;
- optionally submit/stop.

This preserves the idea of tool use without allowing an actor to set the answer directly. The terms “coarse pitch” and “fine pitch” need to be fixed explicitly before implementation.

The action schema should be defined once and adapted to:

- Gymnasium discrete actions;
- a local policy interface;
- hosted model tool/function calls;
- buttons in a demonstration UI.

## Observation and feedback variants

These variants answer different scientific questions and should not be conflated:

### Reference-conditioned audio

The actor receives the reference and current candidate as audio or a non-oracle audio representation, but never receives frequency or cents error. This tests audio perception plus sequential control and is the leading candidate for the main benchmark.

### Reward-only control

The actor sees its current controls and scalar feedback but not the reference audio. This is black-box search or bandit optimization, not evidence of listening. It is valuable as a leakage and search baseline.

### Oracle descriptors

The actor receives an explicit pitch estimate, chroma, or other engineered descriptor. This is useful as an easy pipeline check and upper-bound-style baseline, but it should not be the main “learned to hear pitch” claim.

### Raw waveform versus spectral representation

Raw waveforms make the perception problem substantially harder. A log-frequency magnitude representation supplies useful inductive bias without handing the actor a pitch number. This deserves its own controlled comparison rather than an accidental implementation choice.

## Proposed difficulty ladder

Every early tier should remain solvable through one **global transposition**. Arbitrarily detuned individual chord voices cannot be repaired by a single global pitch control.

1. **Pipeline sanity:** generated sine, fixed duration, clean signal.
2. **Robust sine:** randomized phase, level, onset, polarity, duration, and mild noise.
3. **Single-note timbre:** sine, triangle, saw, square, filtered and missing-fundamental variants.
4. **Chords:** known voicing and intervals, globally detuned; hold out chord qualities, inversions, and timbres.
5. **Mixed waveforms:** layered harmonic sources and held-out combinations.
6. **Pitched percussion:** decays, transients, noise, and controlled inharmonicity with a declared nominal pitch.
7. **DAW-rendered out-of-domain audio:** Serum/Ableton instruments, unusual presets, effects, and owned recordings.
8. **Time-varying material:** loops or moving pitch targets, only after the static benchmark is understood.

The sine checkpoint proves the environment, seeding, reward, action semantics, logging, and evaluation. It does not by itself establish an interesting RL result.

## Dataset strategy

### Procedural data

A tiny built-in renderer can generate unlimited labeled sine, waveform, chord, and simple percussion episodes. This is ideal for deterministic tests, train/test splits, and exact latent pitch truth. It need not become a general-purpose synthesizer.

### DAW-authored data

DAW bounces are a strong source of realistic, controlled examples. A session can render the same musical event at known global detune offsets, including difficult timbres and effects. These assets should complement procedural data rather than replace it.

A manifest should retain at least:

- stable asset ID and source-file hash;
- ownership/license and redistribution status;
- source type: procedural, DAW, recorded, or uploaded;
- DAW, plugin/instrument, preset, and processing-chain notes where available;
- MIDI notes, chord symbol/voicing, root, and tuning reference;
- known global detune in cents;
- sample rate, bit depth, channels, duration, and loudness treatment;
- recording/render session and base-asset group;
- train, validation, IID-test, or OOD-test split;
- free-form caveats, especially for ambiguous percussion pitch.

Derived detunings of the same original must remain in the same split. Splitting individual rendered files would leak the source timbre across train and test.

For chords, the label is the known global transposition of a known set of notes—not a claim that the chord has one physical fundamental. For pitched percussion and loops, “correct pitch” must be declared operationally; some inharmonic sounds have multiple defensible pitch interpretations.

## Actor, adapter, and trainer boundaries

Model serving is separate from the environment.

Harpy should expose a small actor protocol: initialize an episode, receive an observation, choose a legal action, and optionally finalize. Adapters can then support:

- native trainable RL policies;
- classical DSP plus a deterministic controller;
- supervised audio-to-offset models;
- local audio or multimodal models;
- OpenAI-compatible hosted endpoints, including providers such as OpenRouter;
- tool-using hosted agents where the provider supports the required audio input and function-calling behavior.

A useful working boundary is:

```text
describe() -> ActorSpec
start(EpisodeContext) -> ActorSession
ActorSession.act(ActRequest) -> ActResult
ActorSession.close(EpisodeSummary)
```

Training remains a separate lifecycle that consumes an environment factory and emits a checkpoint plus training report. Evaluation then loads that checkpoint through the same frozen actor interface used by every other lane.

“OpenAI-compatible” describes a transport shape, not equivalent capabilities. Each adapter must declare or probe support for audio input, tool calls, strict structured output, streaming, usage reporting, and accepted media formats. Harpy should keep one canonical lossless WAV artifact with a stable hash, while adapters convert it to waveform arrays, base64 audio, or a declared derived observation as required. Different observation tracks must remain separate in results.

A frozen hosted LLM or multimodal model calling tools is an **evaluation actor**, not RL training. It becomes RL only when its policy weights or an equivalent trainable policy are updated from trajectories and rewards. Harpy should report both modes clearly instead of using “RL” as a blanket label.

The environment must never grant a hosted actor shell or filesystem access. Its only capabilities should be the explicit observation channel and typed pitch actions. API credentials belong in local configuration and must never enter experiment artifacts.

Model tool calls are proposals, not direct environment mutations. The harness should validate the tool name, schema, step ID, episode budget, and arguments, then execute at most one mutating pitch action per environment step. Unknown tools, stale calls, multiple mutations, and tool loops beyond the declared budget should fail closed.

A Codex subscription can plausibly support a distinct, human-owned Codex-agent evaluation lane through ChatGPT sign-in and the Codex SDK. It is not generic OpenAI API credit, and current Codex-oriented text models do not necessarily accept audio directly. Such a lane would need a declared spectral/image observation or a narrow allowlisted audio-analysis tool, and should record subscription billing separately from usage-priced API runs.

References: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [OpenAI audio input](https://developers.openai.com/api/docs/guides/audio), [OpenRouter audio inputs](https://openrouter.ai/docs/guides/overview/multimodal/audio), [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection), [Codex authentication](https://learn.chatgpt.com/docs/auth), and [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk).

## Optional compute telemetry

Compute is a potentially distinctive benchmark dimension, but it should initially be reported alongside task quality rather than silently folded into the reward.

Why keep it separate first:

- a scalar weight between accuracy and energy is an arbitrary research decision;
- actors may learn to terminate early or avoid useful observations in ways that look efficient but are not comparable;
- local and hosted inference expose fundamentally different telemetry;
- raw measurements allow later Pareto-frontier analysis without rerunning every experiment.

Candidate local measurements:

- wall-clock time and time to successful tune;
- environment steps and model inference calls;
- process CPU time and peak/resident memory;
- GPU utilization, VRAM, power draw, and energy integral when the device exposes them;
- idle baseline, sampling interval, warm-up, hardware, driver, power mode, and telemetry coverage.

Candidate hosted measurements:

- request count, latency, retries, input/output tokens or provider units;
- provider-reported model, region/routing metadata, and monetary cost when available;
- no fabricated “GPU energy” estimate when the remote provider does not expose one.

The primary presentation should be an accuracy–latency–energy/cost Pareto view. A later challenge track may define a composite score, but raw components must always remain available.

Telemetry should be capability-based and optional. Missing GPU power counters must never prevent an actor from running.

Working implementation direction:

- `time.perf_counter_ns()` for monotonic wall time;
- [psutil](https://psutil.readthedocs.io/) for root-and-descendant CPU time and peak RSS/USS;
- NVIDIA's official [`nvidia-ml-py`](https://pypi.org/project/nvidia-ml-py/) NVML binding for device GPU utilization, VRAM, power, and cumulative energy when supported;
- `nvidia-smi` only as a diagnostic/fallback, because its text output is not a stable library contract;
- [Zeus](https://ml.energy/zeus/measure/) as a possible later energy-window helper, not a base requirement.

This machine currently exposes whole-device GPU utilization, VRAM, temperature, and power draw under WSL, but not an `nvidia-smi` cumulative-energy field or Linux CPU RAPL counters. If cumulative NVML energy is unavailable, Harpy can integrate sampled power, label it `device_total`, and subtract a separately measured idle baseline. Short operations should be batched into roughly ten-second measurement windows because laptop Ada power readings are averaged and noisy. Cold start and warmed steady state must be reported separately.

The collector should default to best-effort 250 ms sampling, preserve raw samples or summaries plus coverage/errors, and use `null` with an explanation for unsupported counters. Remote provider compute must be labeled `unobserved_remote`, never zero. WSL CPU/RAM measurements cover the VM/process tree rather than all Windows host activity, and CPU energy will require a future native-Windows sidecar if it becomes important.

References: [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html), [NVIDIA CUDA on WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/), and [CodeCarbon methodology](https://docs.codecarbon.io/latest/explanation/methodology/).

## Human lab and demonstration surface

The manual lab is not the training system. Its purposes are:

- validate that the pitch actions sound and behave as specified;
- inspect one episode and its reward/telemetry trace;
- compare reference, initial candidate, and final candidate;
- demonstrate the sandbox and tool surface in the YouTube video;
- replay saved agent trajectories.

Because the project runs under WSL2, browser playback avoids making Windows/WSL audio-device routing a core dependency. The browser should play audio produced by the same headless environment used in training so the demo cannot drift from the benchmark.

## Experiment record

Every run should produce a portable manifest and summary containing:

- Harpy suite and environment version;
- Git commit and dirty-worktree flag;
- actor adapter, model/checkpoint, trainer, and relevant endpoint metadata;
- task distribution, dataset manifest version, and episode IDs;
- observation/action/reward configuration;
- random seeds;
- hardware, operating system, dependency lock, and telemetry capabilities;
- per-episode action trajectory and outcome;
- aggregate metrics with uncertainty;
- pointers to optional audio, plots, logs, and video-ready replays.

This experiment ledger is part of the product, not cleanup work after training.

## Evaluation principles and required baselines

Primary candidate metric: held-out success rate within a declared cents tolerance. Also record final absolute cents error, octave-error rate, action count, excess actions over the shortest legal path, sample efficiency, latency, and generalization gaps.

Required comparisons:

- random legal actions;
- oracle shortest-path controller using latent pitch truth;
- classical F0 estimator plus deterministic action planning;
- reward-only hill climbing/search with audio hidden;
- supervised signed-offset estimator plus planner;
- one standard RL baseline;
- human/manual trajectories where useful for the narrative.

If a classical or search baseline wins the early tasks, that is an honest and useful result. RL becomes more substantively motivated when later tasks introduce partial observation, noisy or delayed feedback, source-dependent processing artifacts, action costs, moving targets, or online adaptation.

## Local machine snapshot

Observed on 2026-08-07:

- Windows physical memory: 15.46 GiB (one 16 GB module);
- WSL-visible memory: about 7.49 GiB;
- WSL-visible swap: 2 GiB;
- CPU: Intel Core Ultra 7 155H, 22 logical CPUs exposed to WSL;
- GPU: NVIDIA RTX 500 Ada Generation Laptop GPU, 4,094 MiB VRAM;
- environment: WSL2 2.1.5, Linux kernel 5.15.146.1.

At the time of the snapshot there was no `%UserProfile%/.wslconfig`. Microsoft documents WSL2's default memory allocation as 50% of Windows physical RAM, which explained the apparent 8 GB limit; it was a VM allocation rather than missing physical memory. On 2026-08-07, `%UserProfile%/.wslconfig` was created with `memory=16GB`. The new ceiling takes effect after WSL is fully shut down and restarted.

Reference: [Microsoft WSL advanced settings](https://learn.microsoft.com/windows/wsl/wsl-config).

## Ecosystem and prior work

The broad claim “RL uses audio similarity to control pitch or synthesis parameters” already exists. Harpy's defensible contribution is the standardized audio-transformation benchmark, curriculum, adapters, telemetry, leakage controls, and reproducible experiment ledger.

### Closest projects and papers

| Work | Relevance | Reuse/constraint |
| --- | --- | --- |
| [RL-synth-control / NIME 2026](https://github.com/vincenzomadaghiele/RL-synth-control) and [paper](https://doi.org/10.5281/zenodo.20784272) | Gymnasium + Stable-Baselines3 agents match live/recorded target audio by changing SignalFlow synth parameters; includes tone, FM, granular, and Benjolin tasks plus multiple audio rewards. | Closest conceptual predecessor, but it controls synth parameters rather than an immutable audio asset. No clear repository license was visible on 2026-08-07, so use as literature unless licensing is clarified. |
| [RL-impro](https://github.com/vincenzomadaghiele/RL-impro) and [AIMC 2024 paper](https://aimc2024.pubpub.org/pub/9zd8yyyv/release/1) | Earlier discrete-action Gymnasium/DQN system with sine, FM, granular, descriptors, lookup tables, Pure Data, and live OSC control. | LGPL-3.0; useful architectural reference, but its Pure Data/lookup-table path is unnecessary for the first Harpy environment. |
| [SynthRL, IJCAI 2025](https://www.ijcai.org/proceedings/2025/1129) and [code](https://github.com/argaaw/SynthRL) | RL fine-tunes Dexed parameter inference with rendered-audio rewards for in- and out-of-domain sound matching. | MIT; same inverse-audio objective, but not an interactive pitch-tool Gym. |
| [DDSynth-RL, ISMIR 2026](https://arxiv.org/abs/2608.03032) and [code](https://github.com/DDSynth-RL/DDSynthRL) | Discrete diffusion over Dexed/MIDI parameters followed by GRPO-style audio-reward fine-tuning. | Apache-2.0 code; powerful current reference, substantially heavier than Harpy's intended first rung. |
| [TorchSynth](https://github.com/torchsynth/torchsynth) | Vectorized differentiable PyTorch synth and procedural labeled-data generator. | Apache-2.0; potentially useful later, but an in-project sine/chord renderer is smaller and easier to audit for v0. |
| [DDSP](https://github.com/magenta/ddsp) | Differentiable harmonic/noise synthesis and multiscale spectral losses. | Apache-2.0; useful as later prior art/baseline, not a necessary base dependency. |
| [SpiegeLib](https://github.com/spiegelib/spiegelib) and [VST-FM benchmark](https://github.com/spiegelib/vst-fm-sound-match) | VST rendering, sound-matching datasets, genetic/supervised baselines, and listening-test scaffolding. | MIT but relatively dormant; useful reference for benchmark structure. |

No field-standard, maintained, installable “AudioGym” suite with Harpy's proposed pitch ladder, actor adapters, compute reporting, and leakage tests was identified in this research pass.

### Candidate base libraries

- [Gymnasium](https://gymnasium.farama.org/) for the environment contract (MIT).
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for first standard RL baselines (MIT).
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for deterministic synthesis and signal processing (BSD-family).
- [SoundFile](https://python-soundfile.readthedocs.io/) for initial WAV/FLAC I/O (BSD-3; libsndfile is LGPL).
- [librosa](https://librosa.org/) for offline resampling, pitch shifting, features, and classical baselines (ISC).
- [psutil](https://psutil.readthedocs.io/) and optional [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) for telemetry (BSD-family).
- PyTorch only where an actor or batched representation actually needs it.

TorchAudio is now officially in maintenance mode and moved media I/O toward TorchCodec, so it should not anchor the base architecture. Heavy or restrictive stacks such as DDSP/TensorFlow, Ray/RLlib, Essentia, aubio, Pedalboard, or Rubber Band should remain optional and justified by a specific later experiment.

## YouTube/process narrative

A strong start-to-finish story could follow this arc:

1. Begin with the intentionally “solved” 440 Hz tuning problem.
2. Explain why solving pitch analytically is not the experiment.
3. Define the actor's limited tools and what information is hidden.
4. Show a human completing the same episode in the manual lab.
5. Build and validate the environment against oracle and classical baselines.
6. Train/evaluate the first small policy locally, including failures.
7. Add harder timbres, chords, percussion, and DAW-authored OOD assets.
8. Compare small local policies with hosted or agentic actors.
9. Plot quality against actions, time, local energy, or hosted cost.
10. End with the actual result, including the possibility that simple methods remain best.

Preserve clean episode replays, audio A/B/C exports, plots, environment diagrams, exact commands/configs, and short decision-log entries as the project develops. These are both reproducibility artifacts and future video assets.

## Open questions

Questions should be resolved one at a time during design:

1. What exactly may the main v0 actor observe: raw audio, a spectral representation, or only scalar feedback?
2. What precise actions correspond to “octave,” “semitone,” “coarse,” and “fine” pitch?
3. What first result counts as success: convergence on seen tones, OOD frequency generalization, or cross-timbre generalization?
4. Should the first reward be sparse latent cents success, dense progress, audio similarity, or separate environment variants?
5. Which renderer is authoritative for uploaded/DAW audio, and what artifact budget is acceptable?
6. Is the first model a small native RL policy, or should the adapter contract be validated first with a deterministic actor?
7. When does the manual lab move from a replay/debug page to a polished public demo?
8. Which assets may be redistributed, and which remain private evaluation material?
9. Should the repository adopt a permissive license, and which one?

## Decision log

### 2026-08-07

- Adopted **Harpy** as the working project name from the workspace.
- Defined the core task as constrained pitch manipulation of audio, not synthesizer parameter control.
- Kept synthesis as a labeled-data source and optional renderer only.
- Treated the manual lab as a debugging, replay, and video-demonstration surface rather than the training loop.
- Required model serving to remain behind an adapter boundary.
- Made compute telemetry optional and capability-based.
- Chose to report resource cost as separate raw objectives before considering a composite reward.
- Initialized a new Git repository for the project notebook and future work.
- Raised the configured WSL2 memory ceiling from its default 50% allocation to 16 GB; activation requires a WSL restart.
