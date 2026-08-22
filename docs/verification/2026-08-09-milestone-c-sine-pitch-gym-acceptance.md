# Milestone C sine-pitch Gym acceptance evidence

Date exercised: 2026-08-09

Tested implementation range: `dc648d55374568fbd07fbc953e74cbc9ba4ea1dc...e8d3e77`
(`origin/main...e8d3e77`, ending at `feat: add sine gym checkpoint command`)

Environment: WSL2/Linux, Python 3.12.3, Gymnasium 1.3.0, uv 0.12.3

## Acceptance status

The fresh automated, registered-environment, command, import-boundary, and structural
checks below pass for the Task 7 implementation head plus the pre-commit Task 8
documentation edits. They accept Milestone C on those measured headless dimensions.
The controller will perform the post-documentation whole-branch review and clean-tree
completion gate separately.

Task 1–7 reports preserve their own RED/GREEN cycles, mutation checks, focused runs, and
historical full-suite results. Those reports are provenance rather than substitutes for
the fresh commands recorded here. No earlier pass count is presented as Task 8 evidence.

Milestone C has no native GUI, playback, device, screenshot, audio-capture, or subjective
listening acceptance requirement. None was performed or inferred.

## Fresh automated evidence

All commands ran from the Milestone C worktree on 2026-08-09.

| Command/check | Exit | Exact observed result |
| --- | ---: | --- |
| `uv sync` | 0 | `Resolved 19 packages in 0.57ms`; `Checked 19 packages in 0.90ms`. |
| `uv run pytest` | 0 | 11,326 tests collected and passed in 165.85 seconds. After the passing summary, the host emitted its known PipeWire symbol-resolution and unsupported mouse-grab diagnostics. |
| `uv run ruff check .` | 0 | `All checks passed!` |
| `uv run ruff format --check .` | 0 | `82 files already formatted` |
| Qt/GUI-free `import harpy.envs` assertion | 0 | Zero stdout and zero stderr; neither `PySide6` nor any `harpy.gui` module was loaded. |
| Registered `check_env(environment.unwrapped)` for all three IDs | 0 | Zero stdout, zero stderr, and no warning for `Harpy/SinePitch-v0`, `Harpy/SinePitchOracle-v0`, or `Harpy/SinePitchRewardOnly-v0`. |
| `uv run harpy-sine-gym --episodes 2 --seed 0` | 0 | One compact JSON document and one trailing newline on stdout; zero stderr. |
| Two additional `--episodes 2 --seed 0` runs compared in memory | 0 | Byte-identical documents; newline-restored SHA-256 `3bdc34d903d73deb0b8ccdaa5eaefc7bef8bdbaf9cb8d9b4b05c451da72becf1`. |
| `git diff --check` | 0 | Zero output for the pre-commit documentation worktree. |
| `git diff --check origin/main...HEAD` | 0 | Zero output for the committed implementation range through Task 7. |

The exact host diagnostics printed after the passing pytest summary were:

```text
qt.multimedia.symbolsresolver: Couldn't load pipewire-0.3 library
qt.multimedia.symbolsresolver: Couldn't resolve pipewire-0.3 symbols
This plugin does not support grabbing the mouse
```

The full regression suite includes the existing native-workbench tests, which explain
the post-summary host Qt diagnostics. The Milestone C-specific import, checker, and
checkpoint commands themselves were silent on stderr and did not initialize Qt or an
audio device.

## Exact checkpoint document

This is the complete stdout document from the fresh two-episode command:

```json
{"checkpoint_id":"harpy-milestone-c-sine-pitch-v0","config_id":"fixed-default-sine-v0","episodes":2,"results":[{"baseline":"random","environment_id":"Harpy/SinePitch-v0","episodes":2,"final_within_1_cent_rate":0.0,"final_within_5_cents_rate":0.0,"invalid_action_rate":0.0,"mean_absolute_final_error_cents":2327.5,"mean_actions":17.5,"mean_excess_actions":null,"mean_return":-1.324427295081967,"observation_mode":"spectrum","seed":0,"submitted_success_rate":0.0,"submitted_within_1_cent_rate":0.0,"truncation_rate":0.0},{"baseline":"spectrum_peak","environment_id":"Harpy/SinePitch-v0","episodes":2,"final_within_1_cent_rate":0.5,"final_within_5_cents_rate":1.0,"invalid_action_rate":0.0,"mean_absolute_final_error_cents":1.5,"mean_actions":34.5,"mean_excess_actions":6.5,"mean_return":1.0567141803278688,"observation_mode":"spectrum","seed":0,"submitted_success_rate":1.0,"submitted_within_1_cent_rate":0.5,"truncation_rate":0.0},{"baseline":"oracle","environment_id":"Harpy/SinePitchOracle-v0","episodes":2,"final_within_1_cent_rate":0.0,"final_within_5_cents_rate":1.0,"invalid_action_rate":0.0,"mean_absolute_final_error_cents":5.0,"mean_actions":28.0,"mean_excess_actions":0.0,"mean_return":1.0562054098360654,"observation_mode":"oracle","seed":0,"submitted_success_rate":1.0,"submitted_within_1_cent_rate":0.0,"truncation_rate":0.0},{"baseline":"reward_search","environment_id":"Harpy/SinePitchRewardOnly-v0","episodes":2,"final_within_1_cent_rate":1.0,"final_within_5_cents_rate":1.0,"invalid_action_rate":0.0,"mean_absolute_final_error_cents":0.0,"mean_actions":42.0,"mean_excess_actions":14.0,"mean_return":1.0568850819672129,"observation_mode":"reward_only","seed":0,"submitted_success_rate":1.0,"submitted_within_1_cent_rate":1.0,"truncation_rate":0.0}],"schema_version":1,"seed":0}
```

This `episodes=2` artifact proves the command path, schema, lane labeling, and seeded
repeatability for this input. Its tiny sample is not evidence of baseline quality,
learned perception, or generalization.

## Structural evidence

The required forbidden-dependency search over `src/harpy/envs` exited 0 with zero
matches for Qt/GUI, audio-device/workbench, Stable-Baselines/PyTorch, librosa,
OpenRouter, or telemetry terms. The corresponding test search returned exactly one
match:

```text
tests/envs/test_checkpoint.py:219:    if name == "PySide6" or name.startswith("PySide6."):
```

That string belongs to the subprocess import guard which fails if the headless command
loads Qt; it is test evidence, not a production dependency.

Production-module line counts and responsibilities were:

| Module | LOC | Principal responsibility |
| --- | ---: | --- |
| `envs/models.py` | 257 | Frozen actions, modes, controls, constants, and evaluator result validation |
| `envs/planning.py` | 110 | Pure bounded tolerance-optimal musical action planning |
| `envs/spectrum.py` | 43 | Frozen log-frequency grid/config and array-only spectrum encoding |
| `envs/sine_pitch.py` | 368 | Gym spaces, seeded episodes, transactional rendering/steps, reward, and result creation |
| `envs/baselines.py` | 401 | Evidence-isolated planners/policies, rollouts, and immutable aggregate summaries |
| `envs/checkpoint.py` | 69 | Argparse validation and one-document JSON checkpoint command |
| `envs/__init__.py` | 48 | Deliberate public exports and atomic idempotent Gym registration |
| **Total** | **1,296** | |

Inspection found no generic trainer, hosted/local actor framework, storage abstraction,
compatibility alias, logging layer, GUI bridge, or dead alternate environment path.
Each module matches the approved final file map.

## What the evidence establishes

- `Harpy/SinePitch-v0` is the headline fixed-log-spectrum environment;
  `Harpy/SinePitchOracle-v0` and `Harpy/SinePitchRewardOnly-v0` are separately labeled
  controls. Their rows are never pooled.
- The actor uses seven stable Octave/Semitone/Cent/Submit action IDs and three
  independent bounded controls. Success requires explicit Submit at an inclusive
  absolute error of at most 5 cents; accuracy within 1 cent is separate.
- Seeded reset/step behavior, fresh immutable-source rendering, blocked/terminal rules,
  evaluator results, registered Gymnasium contracts, and fixed spectrum evidence are
  exercised by the fresh full suite.
- Random, Spectrum Peak, Oracle, and Reward Search run through the fixed evaluator and
  produce separate JSON-scalar summaries. The command writes no files.

## Leakage boundary and nonclaims

Actor-facing observations and `info` omit source pitch, explicit target/current
frequencies, signed or absolute cents-error fields, analyzer peak, optimal plan, and
`EpisodeResult`-only metrics such as exact final error and invalid-bound count. The
evaluator reads immutable `EpisodeResult` only after termination or truncation. This is
capability separation for supported actors, not a security sandbox against malicious
Python code reaching private attributes or `env.unwrapped`.

The spectrum lane contains a deliberately informative representation of a single clean
sine, and scalar reward permits black-box search. Spectrum results alone therefore do
not prove learned pitch perception, and reward-only results do not prove listening.
Oracle exposes exact current pitch by design.

The procedural backend directly re-synthesizes each effective sine frequency from
immutable source truth and cumulative controls. It is an ideal sine-only transformation,
not recorded-audio pitch shifting. The checkpoint contains no trained policy,
train/validation/test split, held-out generalization result, raw-waveform lane, callable
analyzer, hosted actor, model adapter, telemetry, database, durable experiment ledger,
or Gym episode replay in the GUI. Richer waveforms, chords/polyphony, real assets,
training, adapters, storage, telemetry, and replay remain roadmap work.

## Pre-commit documentation scope

Task 8 is limited to `README.md`, `docs/project-notebook.md`, and this acceptance file.
It changes no production, test, dependency, lockfile, or historical Milestone A/B
acceptance artifact. The post-commit whole-branch review and clean-worktree gate are
owned by the controller and are not inferred here.

After all three documentation artifacts existed, `git diff --check` again exited 0 with
zero output and `git status --short` returned exactly:

```text
 M README.md
 M docs/project-notebook.md
?? docs/verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md
```
