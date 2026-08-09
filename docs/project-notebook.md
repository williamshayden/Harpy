# Harpy project notebook

Last updated: 2026-08-08

Status: exploratory project-level notes. Milestone specifications and their review status live under `docs/superpowers/specs/`.

This is the living record for the project: what we think Harpy is, what prior work exists, what decisions have been made, what remains uncertain, and what would make the work scientifically honest and useful.

## Working thesis

Harpy's initial track should be a model-agnostic audio-control benchmark in which an actor receives source audio plus a **symbolic musical goal**—for example, “tune this note to D4” or “transpose this chord to F minor”—then uses a deliberately constrained set of analysis and pitch-manipulation tools to satisfy that goal.

The actor does not need to hear an ideal target recording. When useful, the evaluator may generate a hidden target render from the same procedural source and use it to measure audio fidelity. That render is evaluator-only; the actor reasons from the source audio, the symbolic goal, its allowed observations, and the effects of its actions.

The point is not to discover a new formula for pitch. The point is to measure whether a learned actor can acquire and generalize an effective audio-to-action policy while being restricted to the same hierarchical controls a person or tool-using agent might have.

The environment should make simple, solved cases cheap enough to validate the entire experimental pipeline, then preserve the same interface as the audio becomes less pitch-salient or more polyphonic. The scientific claim must match the observation track: giving an actor an explicit F0 estimate tests tool use and control, while giving it only audio or a non-oracle spectrum also tests pitch perception.

## Core track first: audio transformation, then inverse synthesis

In the first retuning versions, the actor does **not** control oscillator, filter, envelope, or synthesizer-patch parameters. It manipulates an immutable audio asset through pitch tools.

A synthesizer can still serve two useful roles:

1. Generate reproducible labeled source audio for procedural tests.
2. Generate an evaluator-only ideal render after applying a known symbolic transformation.

A later inverse-synthesis track can let an actor control waveform, ADSR, and eventually richer parameters to match a reference timbre. That can grow into a two-stage challenge: first create a patch that matches a reference, then use the matched patch to reach a requested note, key, or chord. It is a separate benchmark family that can reuse Harpy's rendering, actor, evaluation, and replay infrastructure without being chained into v1.

The environment should always derive each candidate render from the original source plus the current cumulative pitch offset. It should never repeatedly pitch-shift the previous output, because cumulative processing artifacts would turn action order into an unintended hidden variable.

## What one episode may look like

Working hypothesis for the first benchmark:

1. Select or generate an immutable base asset.
2. Declare a symbolic musical goal and, where possible, generate an evaluator-only ideal render from the same source configuration.
3. Give the actor the source/current audio, symbolic goal, permitted analysis tools, current tool state, and remaining action budget.
4. Let the actor apply one constrained pitch action at a time.
5. Re-render from the immutable base asset after each action.
6. Score musical correctness against latent evaluator truth and audio fidelity as separate outcomes.
7. Terminate when the candidate is inside the tuning tolerance; truncate when the action budget is exhausted.
8. Record accuracy, trajectory, latency, environment steps, model metadata, and any available compute telemetry.

The source pitch, generated transform, and hidden ideal render are evaluator truth. The declared musical goal is actor-visible, but evaluator-only values must not accidentally leak through filenames, array lengths, loudness, phase, metadata fields, episode IDs, or debug information.

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

### Symbolic goal with callable analysis tools

The actor receives the source/current audio and a target note, chord, key, or interval. It may call a deliberately limited analyzer such as a spectrum, F0 estimator, or chromagram, then use pitch actions. This is the leading candidate for v0 because it tests whether an actor can combine musical intent, evidence, and constrained control without requiring an audible target.

An explicit F0 or cents estimate makes perception an engineered preprocessing step. That is legitimate for a tool-use/control track, but it cannot support a claim that the policy learned pitch perception.

### Symbolic goal with non-oracle audio representation

The actor receives the same musical goal but only raw audio or a declared representation such as a log-frequency magnitude spectrum. This combines perception and control and is a harder, scientifically distinct track.

### Reference-conditioned audio

The actor receives an audible reference plus the current candidate. This remains a useful sound-matching variant, but it is not required for the core symbolic-retuning question.

### Reward-only control

The actor sees its current controls and scalar feedback but no audio evidence. This is black-box search or bandit optimization, not evidence of listening or musical reasoning. It is valuable as a leakage and search baseline, not the main track.

### Oracle descriptors

The actor receives latent source pitch or exact cents error rather than an estimate derived from audio. This is useful as an easy pipeline check and upper-bound baseline, but it should never be presented as learned listening.

### Raw waveform versus spectral representation

Raw waveforms make the perception problem substantially harder. A log-frequency magnitude representation supplies useful inductive bias without handing the actor a pitch number. This deserves its own controlled comparison rather than an accidental implementation choice.

### Working target semantics

The target format determines what the result demonstrates:

- `cents_offset: +235` is a non-perceptual action/planner sanity test because the answer is already stated.
- `target_note: D4` is the smallest core listening task: the actor must infer the hidden source pitch and reach an octave-specific target.
- `target_pitch_class: D` permits octave-equivalent answers and therefore tests pitch class rather than register.
- `target_chord: F:min` is globally reachable only when the source has the same quality and voicing relationship up to transposition.
- a target key or timed note contour belongs to later, longer-form material.

The main result should include both a one-shot signed-offset prediction view, which isolates perception, and a sequential-action view, which adds planning and refinement. Otherwise, repeated reward queries can conceal the fact that an actor is searching rather than listening.

## Proposed curriculum

The curriculum should vary three axes deliberately instead of treating “harder audio” as one dimension: harmonic richness, polyphony/harmony, and whether the actor transforms audio or programs a synth.

### Retuning track

The user-proposed order is:

1. **V1 — single sine pitch:** one oscillator, one note, hidden source pitch/detune, symbolic target note.
2. **V2 — sine chords in the same mode:** multiple sine voices, initially keeping the harmonic relationship controlled.
3. **V3 — harmonic single notes:** one note from a square, saw, or another band-limited oscillator shape.
4. **V4 — same-mode chords across oscillator shapes:** polyphony plus richer spectra.
5. **V5 — chords across modes with sine voices:** expand harmonic vocabulary while keeping timbre analytically clean.
6. **V6 — chords across modes and oscillator shapes:** combine the harmonic and timbral axes.

Each version can contain its own robustness sub-rungs for phase, level, onset, polarity, duration, noise, voicing, inversion, and held-out frequencies or waveforms. This keeps “V1 single sine” small without losing a path to a meaningful generalization result.

“Across modes” still needs one precise definition. If it means the dataset contains major, minor, and modal chords while each episode transposes a chord without changing its internal intervals, the global pitch action remains sufficient. If it means converting the rendered audio itself from one mode or quality to another, the environment needs per-voice note editing or resynthesis.

### Inverse-synthesis track

After the retuning ladder is established:

1. **Patch match:** control a small synth to match an audible reference timbre at a fixed note.
2. **Patch match, then retune:** create the matching patch, then render it at a requested target note or transposition.
3. **Chord patch match, then reharmonize/retune:** extend the same two-stage flow to controlled chords.

The first inverse-synthesis task should expose a deliberately small patch surface—oscillator shape, level, and envelope before filters, modulation, effects, or Serum-scale parameter spaces. Patch similarity and musical pitch correctness should remain separate scores.

The sine checkpoint proves the environment, seeding, reward, action semantics, logging, and evaluation. It does not by itself establish an interesting RL result.

The global-transposition boundary is important. C minor to F minor is a uniform +5-semitone shift. C major to F minor is not: its E natural must also become E-flat. That second operation is an advanced polyphonic-editing task and should not silently change the meaning of v0.

## Dataset strategy

### Procedural data

A tiny built-in renderer can generate unlimited labeled sine, waveform, chord, and simple percussion episodes. This is ideal for deterministic tests, train/test splits, and exact latent pitch truth. It need not become a general-purpose synthesizer.

### Authored or recorded data

Simple bounces or recordings can supply realistic held-out examples. Harpy does not need to reproduce a DAW session, plugin, preset, or processing chain. A high-level label such as `saw`, `sine`, `layered chord`, or `pitched percussion`, plus known musical truth, is sufficient for the benchmark manifest.

A manifest should retain at least:

- stable asset ID, source-file hash, and procedural seed where applicable;
- ownership/license and redistribution status;
- high-level source family such as sine, saw, chord, or percussion;
- evaluator note set or chord label, tuning reference, target goal, and known global offset in cents;
- sample rate, channels, and duration;
- base-asset group so related renders remain together;
- train, validation, IID-test, or OOD-test split;
- free-form caveats, especially for ambiguous percussion pitch.

Extra production provenance may be kept as optional private notes when it is genuinely useful, but it is not part of the required schema.

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

Telemetry is a deferred, optional benchmark dimension—not a v0 requirement and not part of the first reward. The base experiment record only needs wall-clock latency, environment steps, and model/inference-call count. Those are portable and already help compare actors.

A later local collector may add process CPU/memory plus GPU utilization, VRAM, power, and estimated energy through [psutil](https://psutil.readthedocs.io/) and NVIDIA's official [`nvidia-ml-py`](https://pypi.org/project/nvidia-ml-py/) binding. Hosted actors can report provider usage and cost when available; unobservable remote compute must be labeled unknown rather than zero. Any eventual efficiency score should preserve task quality and resource use as separate raw measurements so the trade-off remains visible.

Telemetry must remain capability-based and best-effort. Missing counters should never prevent an experiment from running.

References: [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html), [NVIDIA CUDA on WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/), and [CodeCarbon methodology](https://docs.codecarbon.io/latest/explanation/methodology/).

## Human lab and demonstration surface

The manual lab is not the training system. Its purposes are:

- validate that the pitch actions sound and behave as specified;
- inspect one episode and its reward/telemetry trace;
- compare the source, initial state, and final output, with an optional evaluator-only target reveal for debugging;
- demonstrate the sandbox and tool surface in the YouTube video;
- replay saved agent trajectories.

The manual lab is a native local desktop application; a browser or local web server is explicitly out of scope. The implemented Milestone A workbench uses PySide6, Qt Multimedia, pyqtgraph, and WSLg. Its live audio device remains a demonstration adapter around the same deterministic block renderer that future offline generation will use, so device behavior cannot become benchmark truth.

## Milestone B instrument checkpoint

Observed and recorded on 2026-08-08, Milestone B makes the native sine workbench a
reproducible source authoring tool. A patch now records strict oscillator,
curve-enabled envelope, and output-gain configuration in canonical schema v2, while
played frequency remains separate performance state. Graph-native A/D/S/R and curve
controls therefore let a researcher create deterministic labeled source renders
without turning the manual lab into the benchmark actor.

This does not change the first retuning track's scientific boundary. The first actor
still manipulates a rendered immutable audio asset with constrained pitch tools; it
does not control the synthesizer or infer its patch. Direct synth control remains a
later inverse-synthesis benchmark family. The sine-only Gym itself remains Milestone C
and requires a new specification for actions, observations, rewards, leakage controls,
model adapters, and benchmark protocol before implementation.

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

The first evaluation should keep three questions separate:

1. **Musical correctness:** held-out success rate within a declared cents tolerance, final absolute cents error, target-note or target-note-set accuracy, and octave-error rate.
2. **Signal fidelity:** timing, envelope, timbre, and processing artifacts relative to the immutable source and, where available, the hidden ideal render.
3. **Control efficiency:** action count, excess actions over the shortest legal path, inference calls, latency, and sample efficiency.

One scalar reward may be required for a particular trainer, but the benchmark report should retain these components. Otherwise, a musically correct but badly damaged render could look equivalent to a clean one, or a high-fidelity render at the wrong pitch could receive undue credit.

Required comparisons:

- random legal actions;
- oracle shortest-path controller using latent pitch truth;
- classical F0 estimator plus deterministic action planning;
- reward-only hill climbing/search with audio hidden;
- supervised signed-offset estimator plus planner;
- one standard RL baseline;
- human/manual trajectories where useful for the narrative.

The oracle and classical controllers are especially important for the “smallest model” question. On the first rungs, a deterministic estimator-plus-planner may need no learned model at all. The meaningful benchmark is then the smallest actor that generalizes under each declared observation track, not whether RL can be made to imitate a formula on a single sine wave.

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

The broad claim “RL uses audio similarity to control pitch or synthesis parameters” already exists. Harpy's defensible contribution is a standardized symbolic-retuning benchmark with explicit observation tracks, a controlled curriculum, model adapters, leakage tests, and a reproducible experiment ledger.

### How established pitch tools bound the problem

- **Global sample transposition** applies one ratio, `2^(cents / 1200)`, to the whole signal. A duration-preserving implementation combines pitch shifting and time stretching, with renderer-dependent transient, formant, and phase artifacts. It works on chords because it preserves every interval; it cannot change chord quality. [Rubber Band technical notes](https://www.breakfastquay.com/rubberband/technical.html) describe a production-oriented implementation, while [librosa documents](https://librosa.org/doc/latest/generated/librosa.effects.pitch_shift.html) a convenient open-source baseline.
- **Auto-Tune-style correction** first estimates a monophonic pitch contour, chooses target notes from a key/scale, MIDI, or edited contour, then applies a smoothed time-varying shift with voicing/formant handling. Antares explicitly scopes its tracker to a single voice or non-chordal instrument; it does not independently tune simultaneous chord voices. References: [original Auto-Tune patent](https://patents.google.com/patent/US5973252A/en), [AutoTune 2026 guide](https://antares-web-frontend.sfo3.cdn.digitaloceanspaces.com/documentation/pdfs/AutoTune_2026_User_Guide.pdf), and [Antares tracking guidance](https://help.antarestech.com/hc/en-us/articles/41115327742740-What-does-the-Tracking-knob-and-other-controls-do-in-Auto-Tune-Pro).
- **Polyphonic note editing** detects overlapping note objects, assigns spectral energy to them, edits individual pitch/time/formant properties, and resynthesizes the mixture. This is closer to source separation plus transcription and resynthesis than to one pitch knob. Celemony notes that its DNA algorithms separate by pitch rather than instrument, so two instruments on the same pitch remain one object. References: [Celemony audio algorithms](https://helpcenter.celemony.com/M5/doc/melodyneStudio5/en/M5tour_AudioAlgorithms?env=standAlone) and [DNA patent](https://patents.google.com/patent/US8022286B2/en).

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
- [PySide6](https://doc.qt.io/qtforpython-6/) and [pyqtgraph](https://pyqtgraph.readthedocs.io/) for the native manual lab and plots.
- [SoundFile](https://python-soundfile.readthedocs.io/) for initial WAV/FLAC I/O (BSD-3; libsndfile is LGPL).
- [librosa](https://librosa.org/) for offline pitch shifting, pYIN/YIN, CQT/chroma features, and classical baselines (ISC).
- [mir_eval](https://mir-eval.readthedocs.io/) for conventional melody, multi-pitch, transcription, chord, and key metrics (MIT), while using stricter 5/10/25-cent tuning thresholds than its conventional 50-cent transcription tolerance.
- [psutil](https://psutil.readthedocs.io/) and optional [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/) for telemetry (BSD-family).
- PyTorch only where an actor or batched representation actually needs it.

For procedural tones, the renderer should synthesize the requested frequency directly instead of pitch-shifting audio; this provides artifact-free ground truth. Saw and square generators must be band-limited because naive discontinuous waveforms alias. For recorded audio, [Rubber Band](https://breakfastquay.com/rubberband/) is a useful optional high-quality global baseline with formant handling, but its engine is GPL-2.0-or-later/commercial dual-licensed and should not silently enter a permissive core dependency. [Basic Pitch](https://github.com/spotify/basic-pitch) can later provide polyphonic transcription diagnostics, but transcription alone does not recover independently editable original note waveforms.

TorchAudio is now officially in maintenance mode and moved media I/O toward TorchCodec, so it should not anchor the base architecture. Heavy stacks such as DDSP/TensorFlow, Ray/RLlib, Essentia, Pedalboard, or neural transcription should remain optional and justified by a specific later experiment.

## YouTube/process narrative

A strong start-to-finish story could follow this arc:

1. Begin with the intentionally “solved” 440 Hz tuning problem.
2. Explain why solving pitch analytically is not the experiment.
3. Give the actor a symbolic target, define its limited tools, and reveal what the evaluator keeps hidden.
4. Show a human completing the same episode in the manual lab.
5. Build and validate the environment against oracle and classical baselines.
6. Train/evaluate the first small policy locally, including failures.
7. Add harder timbres, chords, percussion, and authored OOD assets.
8. Compare small local policies with hosted or agentic actors.
9. Plot quality against model size, actions, and latency; optionally add local energy or hosted cost later.
10. End with the actual result, including the possibility that simple methods remain best.

Preserve clean episode replays, audio A/B/C exports, plots, environment diagrams, exact commands/configs, and short decision-log entries as the project develops. These are both reproducibility artifacts and future video assets.

## Open questions

Questions should be resolved one at a time during design:

1. In V5/V6, does “across modes” mean mode diversity with interval-preserving transposition, or does the actor actually convert one rendered mode/chord quality into another?
2. Is the leading learned track raw audio, a high-resolution log-frequency representation, or a separately labeled F0-assisted tool track?
3. Is the first core symbolic goal an octave-specific note such as D4, with direct cents offset retained only as calibration?
4. What precise actions correspond to “octave,” “semitone,” “coarse,” and “fine” pitch, and which targets are reachable on that action lattice?
5. What first result counts as success: OOD frequency generalization, cross-timbre generalization, or both?
6. Should training use sparse terminal success, dense latent progress, or controlled variants of both while keeping reward out of the evaluation observation?
7. Which renderer is authoritative for uploaded/recorded audio, and what artifact budget is acceptable?
8. What is the smallest supervised and RL actor worth comparing on the same observation encoder?
9. When does the manual lab move from a replay/debug surface to a polished public demo?
10. Which assets may be redistributed, and which remain private evaluation material?
11. Should the repository adopt a permissive license, and which one?

## Decision log

### 2026-08-08

- Completed the graph-native Milestone B implementation for direct A/D/S/R scrubbing,
  transient exact entry, constrained curve handles, envelope-only Reset, strict
  v1-read/v2-write patches, deferred patch application, and the refined frequency dial.
- Kept selected and played frequency outside patch JSON as performance state.
- Confirmed that Milestone B creates reproducible procedural sources but does not give
  the first retuning actor synth controls.
- Kept the sine-only Gym at the Milestone C brainstorming and specification boundary.

### 2026-08-07

- Adopted **Harpy** as the working project name from the workspace.
- Defined the core task as constrained pitch manipulation of audio, not synthesizer parameter control.
- Kept synthesis as a labeled-data source and optional renderer only.
- Selected single-note sine retuning as V1, followed by a curriculum that varies polyphony, oscillator harmonic content, and chord modes deliberately.
- Added a later inverse-synthesis track: match a reference patch first, then use that patch to reach a requested pitch or chord.
- Reframed the leading benchmark from audible-reference matching to source audio plus a symbolic musical goal, with any ideal target render hidden inside evaluation.
- Separated global transposition, monophonic Auto-Tune-style correction, polyphonic note editing, and synth patch matching into distinct task families.
- Simplified authored-audio metadata to high-level source type plus musical/evaluation truth; plugin, preset, and processing-chain reconstruction is not required.
- Treated the manual lab as a debugging, replay, and video-demonstration surface rather than the training loop.
- Required model serving to remain behind an adapter boundary.
- Made compute telemetry optional and capability-based.
- Deferred detailed compute/energy collection beyond v0.
- Chose to report resource cost as separate raw objectives before considering a composite reward.
- Initialized a new Git repository for the project notebook and future work.
- Raised the configured WSL2 memory ceiling from its default 50% allocation to 16 GB; activation requires a WSL restart.
- Approved a native PySide6/Qt manual lab with no browser dependency.
- Selected a small deterministic NumPy sine voice as the v1 reference synth; AMY is the leading optional backend to reconsider for harmonic oscillators and richer patches.
- Set the initial selector to MIDI note 60, displayed as `C3` using Ableton's octave convention, with MIDI note 69 at 440 Hz as a separate tuning reference.
- Fixed the v1 envelope to 1 ms attack, 600 ms decay, -6 dB sustain, and 600 ms release with linear-amplitude segments.
- Chose validated typed configuration values for tuning, note selection, render format, and envelope rather than hidden OS environment variables.
- Chose `uv` for Python environment and package management, with a committed lockfile.
- Kept the first executable slice free of a database, JUCE/C++, MIDI, rendered-sample editing, and RL integration.
