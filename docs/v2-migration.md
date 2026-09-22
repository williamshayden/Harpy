# Moving to package 2.0.0

The package version and task/evidence schema versions are separate. V2 adds
`sound-match-v1` while preserving the v1 pitch task. Existing commands without
`--task` keep pitch semantics and ordinary JSON outputs. `train pitch`,
`train ppo`, the `pitch`/`train` extras, and the pitch `reference` alias remain.
Use the new Torch-only `learn` extra for inverse sound models.

## Result formats

| Evidence | Writer | Reader |
| --- | --- | --- |
| Historical learning reports | Original historical workflow | `harpy summarize FILE`, text/Markdown |
| Released v1 experiment JSON | Default pitch CLI and existing Python API | `harpy summarize FILE`, text/Markdown/CSV |
| M1 sound bundle (`harpy.sound-match.result/1`) | Existing sound `run`/`evaluate` plus `save_result` API | Existing `load_result` and `harpy summarize DIRECTORY` |
| Durable bundle (`harpy.evaluation-bundle/2`) | Sound CLI by default; pitch CLI with `--bundle`; `evaluate_bundle` API | `read_bundle`, `iter_records`, `summarize_bundle`, and CLI |

Old evidence is not rewritten in place. New durable envelopes use task-specific
record validators, preserve task semantics, and identify their new format.
Python `run`/`evaluate` remain in-memory APIs; choose `evaluate_bundle` when you
need durable incremental output. A complete-only M1 bundle cannot be resumed.

Durable audio/patch exports are content-addressed under `artifacts/`. Read the
record's artifact roles to find files; the M1 `audio/target-NNNN.wav` and
`patches/submitted-NNNN.json` layout applies only to its original format.

## Resume and portability

Choose a new output for new work. Use `run --resume --output DIRECTORY` or
`evaluate --resume --output DIRECTORY` for an existing bundle created by that
operation. Omitted CLI options are restored. All scientific inputs, configuration,
runtime versions, source files, and actor bytes must remain the same. Completed
errors/truncations are not rerun. `--allow-incomplete` explicitly opts into partial
readout; incomplete results cannot masquerade as a completed study.

New actor inventories use portable relative identities. `locators.json` is inert
local lookup metadata and is excluded from scientific identity. A relocated
result can be read without model files. To resume with relocated models, supply
the full actor list with the original labels and new paths; hashes must match.
Original v1 absolute inventory labels are preserved by their readers. Portable
exports separately document relabeling and retain original and exported hashes.

Replay verifies recorded synthesis/actions without importing models. It requires
the recorded runtime/renderer behavior; cross-platform floating-point identity is
not promised. Linux/Python 3.12 is the qualified environment. Journal writer locks
use Linux advisory locking; native Windows/macOS execution is unqualified.

## Compatibility boundary

Shared serialization and provenance helpers now live outside historical learning
execution. Historical type imports remain aliases, including checkpoint-visible
class paths. Historical report readout no longer imports execution workflows.
The v1 environment, model weights, seven actions, 64-step limit, and inclusive
five-cent explicit-submission criterion are unchanged.

Inverse artifacts identify task, renderer, features, decoder, architecture,
training/validation membership, source, selected epoch, tensor identity, and
payload hashes. Unknown identities fail before inference. Loading inverse models
uses tensor-only Torch deserialization; PPO retains its trusted-producer warning.
Neither artifact format treats checksum integrity as a guarantee of model quality.

See [getting started](v2-getting-started.md) for executable examples and
[acceptance](v2-acceptance.md) for qualification and archive access.
