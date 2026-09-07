# Local v1.0.0 release closeout — 7 September 2026

Harpy v1 is scoped to a headless, reproducible single-sine research toolkit. Dependable learned tuning is not a release gate: accurate execution, persistence, evaluation, and reporting are. No new model experiment, UI, platform promise, or scientific criterion was added during closeout.

## Release changes

- Aligned distribution metadata, `harpy.__version__`, lockfile, and installation examples to `1.0.0`.
- Added `CHANGELOG.md` with the supported surface, development-workbench migration, and explicit limitations; included it in the source distribution.
- Made source-checkout commands use `uv run --locked --extra train`, retaining the optional dependencies needed to run the actual training tests and workflows.
- Bound README documentation links to `v1.0.0`, so the published version can refer to its own documentation.
- Preserved the user's headless retirement and all authorized quality improvements, including the new result/readout modules. The closing audit found no internal imports into retired modules and no missing required provenance inputs.

## Closing verification

The [quality-improvement record](2026-09-06-v1-quality-improvements.md) retains the broad baseline: 11,376 passing regression tests, separate exhaustive renderer verification, three real training smokes, and 16 installed-wheel command checks. Closeout changed package version and documentation, with no further runtime-logic changes.

Additional verification on Python 3.12.3/Linux/WSL:

- **312 targeted tests passed in 98.87 seconds**, covering imports, source/artifact contracts, and real pitch training through an installed `1.0.0` wheel outside Git.
- **11 fresh-environment command checks passed**: base installation and dependency consistency, source/version identity, both CLI help paths, two identical baseline runs, saved Markdown reporting without Torch, the expected missing-training diagnostic, and training-extra dependency resolution.
- Base installation independently resolved Gymnasium 1.3.0 and NumPy 2.5.3 into a new isolated environment. Torch, Stable-Baselines3, and Qt were absent.
- Random, Spectrum Peak, Oracle, and Reward Search all ran for ten seed-0 episodes. Each structured baseline submitted successfully in all ten episodes with no truncation; repeated output was byte-identical. These are bounded engineering checks, not a new scientific cohort.
- Missing training dependencies produced exit 1, an installation instruction, and no artifact directory. The one-off acceptance harness initially expected exit 2; its expectation was corrected to the CLI's existing runtime-error contract, with no application change.
- Ruff lint, formatting, lockfile consistency, and whitespace checks passed. Both documented Python examples also executed successfully during the independent closing audit.

The optional training extra was resolved in the fresh base environment with `uv pip install --dry-run`; it was not fully installed there. The installed-wheel training smoke used the existing optional dependency stack. Native Windows/macOS and fresh CUDA training remain unqualified.

## Local release artifacts

The accepted wheel and source distribution were built from the same release inputs:

| Artifact | SHA-256 |
| --- | --- |
| `harpy_audio-1.0.0-py3-none-any.whl` | `bdaccbaab05311c092692c2b045e5fee55d5febeae5c2cae9808b8b4e3aa0b26` |
| `harpy_audio-1.0.0.tar.gz` | `47deb5a7da4249e9c91fc7853baa9c6ac8867a2e7398760bf4b1a16af54693a6` |

The local acceptance script, command ledger, stdout/stderr, candidate packages, and final committed-source build records live under `outputs/v1-closeout-20260907/`, which is ignored by Git. The final build ledger binds the package hashes to the release commit. Historical models and evidence remain separate and unchanged.

This closeout prepares a local release. Pushing the branch/tag, publishing a GitHub release, and uploading to a package index have not been performed.
