# Contributing to harPY

Use this guide to set up a development checkout, check a change, and submit a
bug report or pull request. Start with a reproducible problem or a specific
improvement to an experiment, the code, or its documentation.

The public v1 release is package 1.0.0, tagged `v1`. New contributions target
`main`; the older `v1.0.0` development checkpoint retains historical interfaces. Read the
[getting-started guide](v1-getting-started.md) for the current workflow and the
[v1 specification](v1-spec.md) for the task and public contracts.

## Set up a development checkout

Use Python 3.12 on Linux or WSL, the qualified environment, with `uv` installed.
Clone `main` for new work, or use your existing development checkout:

```bash
git clone --branch main https://github.com/williamshayden/Harpy.git
cd Harpy
uv sync --locked --group dev --python 3.12
uv run --no-sync harpy --help
```

For the v1 release checkout, use `--branch v1` instead of `--branch main`.

`uv sync` creates `.venv`, installs harPY in editable mode, and installs the
locked base dependencies plus pytest and Ruff. Edits under `src/` are available
without reinstalling. Run subsequent commands from the repository root.

The base environment is enough for documentation, classical actors, ordinary
result readers, and most experiment-runner work. If the change needs supervised
pitch inference or training, add `--extra pitch` to the sync command. For PPO,
use `--extra train`, which includes Torch and Stable-Baselines3. These extras can
be substantial downloads. Keep the chosen extra in later sync commands: an
exact sync removes packages outside the selected dependencies. `--no-sync` in
the commands below uses the environment you already prepared.

## Find the relevant code

| Area | Location and purpose |
| --- | --- |
| Public experiment API | `src/harpy/experiments/`: actor specifications, episode runner, strict results, saved artifacts, and training |
| Current command line | `src/harpy/cli.py`: `evaluate`, `run`, `summarize`, and `train` |
| Task and observations | `src/harpy/envs/`: controls, planning, observation ownership, spectrum encoding, and corruption conditions |
| Audio implementation | `src/harpy/synth/`, `analysis.py`, and `tuning.py` under `src/harpy/` |
| Learning implementation | `src/harpy/learning/`: shared model code and retained historical workflows/readers |
| Examples and tests | `examples/custom_actor.py`; `tests/` broadly follows the implementation areas |
| Contracts and evidence | `docs/v1-*.md` describes the current package; dated verification records and `research/` preserve specific experiments |

The factory-based API in `harpy.experiments` is the current extension point.
Historical learning modules remain useful for reading and reproducing older
work; their presence does not make every internal class a supported public API.

## Check what the change affects

For Python changes, check style and run the tests closest to the behavior:

```bash
uv run --no-sync ruff check src tests examples
uv run --no-sync ruff format --check src tests examples
uv run --no-sync pytest tests/test_imports.py tests/test_cli.py \
  tests/experiments/test_runner.py tests/experiments/test_results.py
git diff --check
```

That test selection covers imports, current CLI behavior, actor lifecycle, and
result validation. It is a useful starting point, not the full test suite. For
example, an FFT estimator change also needs `tests/experiments/test_waveform_fft.py`;
an artifact-loading change needs `tests/experiments/test_artifacts.py`.

For a documentation-only correction, check links and execute the affected
example instead of running unrelated tests. For optional training changes,
`tests/experiments/test_training.py` exercises real CPU smoke updates, checkpoint
reload, and saved-artifact evaluation; install the relevant extra first and
report any skipped cases. GPU training and the full scientific studies are not
prerequisites for every contribution.

Add a regression test when it captures the failure or changed contract. A reader
fix, for example, should reject the inconsistent result that exposed the bug.
A test that merely repeats the implementation adds little evidence.

## Try a custom actor

The existing example adapts waveform observations to the spectrum baseline's
decoder and controller. It runs six shared clean episodes for each actor:

```bash
uv run --no-sync python examples/custom_actor.py \
  --output runs/contributor-adapter-smoke.json
uv run --no-sync harpy summarize runs/contributor-adapter-smoke.json --format markdown
```

Use a fresh output path when repeating it. The example introduces no new
estimator claim; it shows where to substitute an encoding while holding control
fixed.

An `ActorSpec` factory must create fresh controller state for every episode. It
can share immutable model weights. A controller's `decide(observation)` returns
a `Decision` containing an action and, when available, a pitch estimate and
inference count. Declare the observation mode and estimator, decoder, and
controller identities accurately. These labels describe the implementation;
they do not dynamically assemble it. `artifact_paths` can bind custom source
or model files to the result's recorded hashes.

Audio actors receive public waveform or spectrum observations, not the hidden
source coordinate or nuisance seed. Preserve the separation between audio,
reward-feedback, and oracle tracks. A new input family or task needs its own
implementation and validation; a custom actor alone does not extend the
single-sine environment to instruments or recordings.

For model inputs, action parsing, and configuration to record, see
[Evaluate a general-purpose model](v1-getting-started.md#evaluate-a-general-purpose-model).

## Preserve interpretable results

Keep episode membership and corruption conditions matched when comparing
actors. Report initial estimation accuracy separately from submitted success,
and retain partitions, conditions, seeds, invalid actions, truncations, and
action/inference counts. Ordinary result readers validate records and recompute
metrics; do not relax those checks just to accept a malformed fixture.

Saved results and model directories are create-only. Put new runs under ignored
`runs/` or `outputs/` paths, and preserve failed or interrupted artifacts when
they explain a problem. Do not rewrite frozen research sources, membership, or
historical results to make a newer implementation appear to reproduce them.
The existing confirmation cohort has already been evaluated; a new final
confirmation claim needs fresh, frozen membership.

The [acceptance record](v1-acceptance.md) and its linked reviews document broader
qualification at their recorded source identities. Cite that evidence as
historical evidence, separately from checks performed for your change.

## Report a problem or propose a change

Open a [GitHub issue](https://github.com/williamshayden/Harpy/issues) with the
command or smallest example that reproduces the problem, expected and observed
behavior, and your Python and package versions. Include a saved result when it
helps explain the failure.

For a code or documentation contribution, fork the repository, create a branch
for the change, and open a pull request against `main`. Describe the problem,
the changed behavior, the commands you ran, and any untested scope. Link the
relevant contract or example. For a new task or a change to the observation or
action contract, describe the proposal in an issue first so the scope and
evaluation can be discussed before a substantial implementation.
