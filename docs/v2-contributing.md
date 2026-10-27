# Contributing to harPY v2

Use Python 3.12 on Linux/WSL. Start from the v2 source checkout or unpacked
source distribution; public `main` remains v1 until publication. See the
[release notes](v2-release-notes.md) for current preparation and qualification
status, and the [sound-matching guide](v2-getting-started.md) for usage.

## Set up and check a change

From the source root:

```bash
uv sync --locked --group dev --python 3.12
uv run --no-sync harpy --help
uv run --no-sync ruff check src tests examples research/sound_matching
uv run --no-sync ruff format --check src tests examples research/sound_matching
uv run --no-sync pytest -q tests/test_imports.py tests/test_sound_match_cli.py \
  tests/test_cli_bundles.py tests/tasks/test_sound_match_runner.py
```

Run tests for the behavior you change. The repository's
`.github/workflows/ci.yml` defines the wider base contract suite and a separate
optional CPU suite. A skipped optional test is not a training pass. For a
documentation-only change, verify links and affected commands. In a Git
checkout, also run `git diff --check`.

The base environment intentionally omits Torch and Stable-Baselines3. The
`learn` extra adds inverse-model training and inference; `pitch` retains v1
supervised learning, and `train` adds v1 PPO. Follow the explicit CPU installation
procedure in the [inverse guide](v2-inverse.md) when testing these workflows.
Retain the chosen extras on later sync commands: exact sync removes unselected
dependencies. Training correctness, frozen-study reproduction, and perceptual
acceptance are separate checks.

## Find the implementation

| Area | Location |
| --- | --- |
| Sound task, synth, actors, inverse model and strict results | `src/harpy/tasks/sound_match/` |
| Durable evaluation and shared provenance | `src/harpy/_journal.py`, `src/harpy/_provenance.py`, `src/harpy/_serialization.py` |
| Command line | `src/harpy/cli.py` |
| Preserved v1 task and learning | `src/harpy/envs/`, `src/harpy/learning/` |
| Examples | `examples/sound_match_actor.py`, `examples/custom_actor.py` |
| Scientific drivers and frozen evidence | `research/sound_matching/` |

Check the [specification](v2-spec.md) before changing observations, controls,
budget accounting or saved formats. Custom sound actors use the public
`harpy.tasks.sound_match` interface; the example runs from an installed base
package:

```bash
uv run --no-sync python examples/sound_match_actor.py runs/contributor-sound
uv run --no-sync harpy summarize runs/contributor-sound --verify-replay
```

Use a fresh output path for each experiment. Actor factories create fresh
per-episode state and may share immutable weights. Declare model calls and
artifact identities accurately. Custom Python actors are cooperative code and
are not sandboxed.

## Preserve the evidence

Keep actor comparisons matched by episode, seed, cell and declared cost ceiling.
Retain failed, truncated and interrupted attempts. Do not rewrite frozen
memberships, models, archived source or results to make a new run pass. Existing
final tests are consumed; future model selection needs new held-out data.

For a release candidate, build both distributions with `uv build`, install each
outside the checkout with `PYTHONPATH` unset, and exercise run, resume, strict
replay and the custom actor example. Record exact distribution hashes separately
from historical study evidence. Documentation changes can alter package bytes
even when runtime code is identical.

Report bugs with a minimal command, expected and observed behavior, Python and
package versions, and a saved result when useful. Describe each contribution's
behavior change and actual checks; disclose skipped platforms or integrations.
