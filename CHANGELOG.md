# Release notes

## Unreleased

- Present the project as **harPY — harmonic research in python**. The distribution,
  CLI, Python imports, artifact formats, and historical release tag retain their
  existing names.

- Re-spec v1 as an extensible headless audio-control research toolkit; `docs/v1-spec.md` is authoritative and the old `v1.0.0` tag remains intact.
- Add one `harpy` CLI for training, matched evaluation, traced episodes, and validated text/Markdown/CSV result views.
- Add custom actor factories with fresh episode state, separate estimator/decoder/controller identities, and ordinary experiment results with explicit provenance.
- Match supervised pitch and Spectrum Peak with public feasibility decoding and committed zero-tolerance plans; label repeated replanning separately.
- Add waveform observations, eight controlled static-corruption conditions, a 128 MiB evidence cache, and a frozen 1,000-pair clean confirmation protocol.
- Add optional `waveform-fft`, a NumPy-only classical actor preserving the separately evaluated quadratic FFT estimator, five-cent decoder, and committed controller. Keep the qualified learned reference and spectrum bytes unchanged; further representation research is outside this v1 scope.
- Add a Torch-only `pitch` extra, retain experimental PPO through `train`, and retire BC from the new active training interface.
- Include the predesignated seed-0 reference checkpoint. Seeds 0, 1, and 2 each passed 650 clean benchmark and 1,000 clean confirmation episodes, with zero invalid actions or truncations. Preserve complete eight-condition results, including failures under strong noise; no learned superiority claim is made.
- Preserve historical readers and prototype evidence. Revised qualification and packaging evidence are tracked in `docs/v1-acceptance.md`; publication and a revised release tag remain separate actions.
- Remove the duplicate historical command module; use the single `harpy` CLI for current work and the original tag for historical execution. Retain saved-report readers and installed-package training checks.
- Avoid repeated action-history reconstruction in result summaries while preserving serialized metrics and readout output. Validate membership for the reserved clean smoke protocol.
- Promote the practical guide to `docs/v1-getting-started.md`, label historical workflow documents, and keep website presentation drafts outside the headless package repository. Retain the scientific figure-data exporter with the research tools.

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
