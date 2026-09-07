# Release notes

## 1.0.0

Harpy's first headless release provides a reproducible toolkit for clean, procedural single-sine audio-control research. The supported surface is the documented synthesis/analysis API, three registered Gymnasium environments, the baseline CLI, and the optional learning CLI with strict persisted evidence formats.

Internal learning helpers are implementation details, not a stable extension API.

### Included

- Deterministic synthesis, bounded controls, explicit submission, classical baselines, and separate observation tracks.
- Optional pitch-estimation, behavior-cloning, and PPO training, including working installed-wheel workflows with package-content provenance.
- Explicit single-model exploratory evaluation and diagnostics, portable traces with model identity, and validated text/Markdown result summaries.
- Clearer command help and progress, finite-input validation, and faster constant-envelope rendering with unchanged audio output.
- MIT licensing and installation, research, and reproduction guidance.

### Migration from the development workbench

- The desktop UI, audio playback/capture adapters, and `harpy` GUI command are retired. Use `harpy-sine-gym`, `harpy-sine-learn`, or the documented Python API.
- Install the optional learning dependencies with `harpy-audio[train]`; contributors use `uv sync --locked --extra train`. The former `--group train` command is obsolete.
- Remove `RenderConfig.block_frames` arguments; pass the desired frame count to each render call. GUI-only envelope preview/inverse helpers are removed.
- Existing Git-backed artifacts and historical evidence formats retain their byte contracts. New package-source artifacts and exploratory/provenance envelopes require this release's readers. Environment and evidence schema versions are independent of the package's `1.0.0` version.

### Scope and qualification

Release correctness means that experiments run, persist, evaluate, and report accurately. The learned models remain experimental; their recorded scientific criteria were not met. Dependable learned tuning, recorded/noisy/polyphonic audio, and a future website visualization are outside this release's promise.

Python 3.12 on Linux/WSL is the verified environment. Native Windows/macOS and fresh CUDA training have not been qualified. Scientific reproduction requires the source checkout and committed lockfile; installed-package artifacts are explicitly ineligible for the frozen source-cohort claims. See the README and research workflow guide for the supported workflows and limitations.
