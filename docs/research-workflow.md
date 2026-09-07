# Research workflow and protocol history

Harpy separates ordinary measurements from the fixed protocols used to support a scientific claim. Start with the smoke and exploratory workflows in the README. Use the commands below when reproducing the declared historical experiment.

## Formal pitch checkpoint

Use a clean source checkout with the required inputs committed and the declared dependency lock. Package-snapshot artifacts can be inspected and compared through exploration but never qualify for these historical gates. Do not relabel exploratory evidence as a checkpoint.

```bash
uv sync --locked --extra train
uv run --locked --extra train harpy-sine-learn train-pitch \
  --profile checkpoint --seed 0 --output runs/pitch-0 --device cpu
uv run --locked --extra train harpy-sine-learn train-pitch \
  --profile checkpoint --seed 1 --output runs/pitch-1 --device cpu
uv run --locked --extra train harpy-sine-learn train-pitch \
  --profile checkpoint --seed 2 --output runs/pitch-2 --device cpu

uv run --locked --extra train harpy-sine-learn evaluate runs/pitch-0 runs/pitch-1 runs/pitch-2 \
  --output runs/pitch-report.json --device cpu > /dev/null
uv run --locked --extra train harpy-sine-learn summarize runs/pitch-report.json --format markdown \
  > runs/pitch-readout.md
uv run --locked --extra train harpy-sine-learn diagnose runs/pitch-0 runs/pitch-1 runs/pitch-2 \
  --suite iid --output runs/pitch-diagnostics.json --device cpu > /dev/null
```

Formal CPU pitch evaluation requires exact seeds 0, 1, and 2 with matching source, lock, and compatibility identities. It evaluates held-out coordinates and full closed-loop episodes. Lower and upper register generalization, zero/shuffled spectrum probes, action counts, truncations, and baseline comparisons remain distinct.

E.1 admits a separate homogeneous CUDA-trained checkpoint trio, using the same seeds. Training requires explicit `--device cuda`; internal and final evaluations use CPU. E.1 requires the evaluator's clean source to exactly match the artifacts and the training runtime cohort to match across seeds. Individual CUDA manifests remain ineligible under the historical schema-v2 contract; only the admitted E.1 cohort can own its schema-v3 scientific claim. Existing exploratory artifacts are never promoted.

A valid report can say `criterion_not_met`. Preserve failed runs and all declared seeds. Changing actor behavior, data, loss, or thresholds requires a new experiment identity and fresh evidence; do not reinterpret a historical result.

## Historical learned-policy controls

BC tests supervised action learning from oracle labels. PPO starts from a fresh initialization, never BC weights. Both retain their original supported commands:

```bash
uv run --locked --extra train harpy-sine-learn train-bc --profile smoke --seed 0 --output runs/bc-smoke
uv run --locked --extra train harpy-sine-learn train-ppo --profile smoke --seed 0 --output runs/ppo-smoke
uv run --locked --extra train harpy-sine-learn evaluate runs/bc-smoke runs/ppo-smoke \
  --output runs/policy-report.json > /dev/null
uv run --locked --extra train harpy-sine-learn summarize runs/policy-report.json
uv run --locked --extra train harpy-sine-learn diagnose runs/bc-smoke --suite smoke --bound-mask \
  --output runs/bc-mask-diagnostics.json > /dev/null
```

`--bound-mask` is a BC diagnostic intervention only. It does not change the persisted model, original actor, or scientific result. Milestone D's declared BC checkpoint is seed 0; the PPO aggregate uses seeds 0 through 4. CPU is authoritative. CUDA for these controls is exploratory.

## Read and share evidence

Canonical evaluation and diagnostic JSON is the evidence source. `summarize` validates saved evidence before rendering text or Markdown; it does not rerun training, execute model archives, or make a weak result eligible. Keep the original JSON beside a derived Markdown readout. Reports from different observation tracks or different fixed suites do not establish a matched model comparison.

Use `run --with-provenance` for portable demonstrations. It binds the existing episode trace to exact manifest/model digests, actor identity, source, and execution device. Legacy `run --json` remains available when consumers require its original byte contract.

`evaluate --exploratory` and `diagnose --exploratory` produce distinct, versioned, explicitly ineligible envelopes. They permit one complete pitch artifact from a wheel or modified checkout, without the formal cohort gate. They do not silently alter historical schema-v1/v2/v3 scientific reports.

A package snapshot records the actual package content hash and distribution version when installed metadata belongs to those files. It contains no invented Git fields. The source union adds an explicit `source_kind=package_snapshot` variant; older strict readers reject this new variant. Existing Git-backed source objects and their serialized bytes remain unchanged.

Artifact publication is create-only. Interrupted training leaves an incomplete directory, which the public loader rejects as a completed model. Retain it for investigation and use a new path for a retry. Export files must remain outside all input artifacts, whose inventories are closed. Model archives remain trusted-local inputs.

## Designs and evidence

The workbench UI is retired. A/B documents describe historical functionality and justify the synthesis/analysis foundations; they are not installation instructions for current UI features.

| Protocol | Design | Evidence |
| --- | --- | --- |
| A: synth/workbench foundation | [Design](superpowers/specs/2026-08-07-harpy-milestone-a-foundation-workbench-design.md) | [Acceptance](verification/2026-08-07-milestone-a-acceptance.md) |
| B: envelopes | [Design](superpowers/specs/2026-08-08-harpy-milestone-b-envelope-authoring-design.md) | [Acceptance](verification/2026-08-08-milestone-b-acceptance.md) |
| C: sine-pitch Gym benchmark | [Design](superpowers/specs/2026-08-09-harpy-milestone-c-sine-pitch-gym-design.md) | [Acceptance](verification/2026-08-09-milestone-c-sine-pitch-gym-acceptance.md) |
| D: learned policies | [Design](superpowers/specs/2026-08-10-harpy-milestone-d-learned-sine-policy-design.md) | [Acceptance](verification/2026-08-10-milestone-d-learned-sine-policy-acceptance.md) |
| E: learned pitch and symbolic control | [Design](superpowers/specs/2026-08-22-harpy-milestone-e-reliable-learned-tuning-design.md) | See the project notebook and preserved run artifacts |
| E.1: deterministic CUDA cohort | [Design](superpowers/specs/2026-08-28-harpy-milestone-e1-cuda-device-cohort-design.md) | [Acceptance and failure analysis](verification/2026-08-28-milestone-e1-cuda-device-cohort-acceptance.md) |

The [project notebook](project-notebook.md) records the broader hypotheses and references. The [v1 review](reviews/2026-09-05-v1-release-review.md) records the release audit; subsequent implementation evidence is kept separately from the historical scientific cohorts.
