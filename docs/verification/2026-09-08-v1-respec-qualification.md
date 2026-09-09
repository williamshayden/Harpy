# Revised v1 qualification — 2026-09-08

The revised implementation meets the specified clean reference gates and engineering
acceptance checks on Linux/WSL with Python 3.12. It remains **Unreleased**. The
existing `v1.0.0` tag is unchanged; no package publication or revised tag is implied.
The [specification](../v1-spec.md) defines the product contract; historical criteria
have not been retroactively changed.

## What was qualified

The shipped reference is the seed-0 checkpoint designated before evaluation, not
the best-performing seed. Seeds 0, 1, and 2 use the original 2,497-parameter pitch
network, unchanged training objective and exact CUDA-trained weights from the prior
clean study. Import preserves original training configuration, selection history,
source identity and payload hashes. The revised decoder and controller are recorded
as evaluation components, not attributed to the original training code.

The estimator runs once at episode start. Feasibility-constrained argmax uses public
bounds, then a committed plan executes at estimated tolerance zero. The primary
Spectrum Peak baseline uses exactly that decoder/controller combination. Replanning,
Oracle and reward-feedback actors remain separately identified. No local-softmax
intervention, model refitting, seed selection or robustness-specific adaptation was
introduced during qualification.

Training used only clean training coordinates; checkpoint selection used clean
validation. Seed 0 selected epoch 50. Original training source was clean commit
`1f2ba3fe1829537c3f8e274dd45abeeafaa744d2`. Its full-package digest was not recorded
historically and remains explicitly null; the preserved Git/lock/payload identities
are available in the imported metadata.

## Clean acceptance

Each cell is successful explicit submissions / episodes. **All cells had zero
invalid actions and zero budget truncations.** Success is inclusive five-cent
terminal accuracy, not one-cent accuracy.

| Suite / source register | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Benchmark IID | 250/250 | 250/250 | 250/250 |
| Benchmark lower | 200/200 | 200/200 | 200/200 |
| Benchmark upper | 200/200 | 200/200 | 200/200 |
| Benchmark total | **650/650** | **650/650** | **650/650** |
| Confirmation IID | 400/400 | 400/400 | 400/400 |
| Confirmation lower | 300/300 | 300/300 | 300/300 |
| Confirmation upper | 300/300 | 300/300 | 300/300 |
| Confirmation total | **1,000/1,000** | **1,000/1,000** | **1,000/1,000** |

The clean benchmark's one-cent accuracy is 377/650 (58.0%) for each learned seed
and the matched Spectrum Peak baseline: IID 146/250, lower 119/200, upper 112/200.
Maximum clean initial and terminal error is two cents. This is consistent with
five-cent classification and does not establish a one-cent tuning guarantee.

Confirmation membership was frozen before this evaluation: 1,000 unique pairs,
400 IID / 300 lower / 300 upper, excluding 1,324 previously evaluated source/target
pairs identified from historical suites and preserved probes. Transformed versions
stay grouped by source partition. The manifest and exclusion inventory ship in
`harpy.experiments.data`. These are fresh episode combinations within the known
rendering family, **not unseen audio**. Confirmation is now consumed evidence;
future model or protocol development needs a new final confirmation manifest.

Canonical confirmation digest:
`74d5791439eabc71c9d9b91c10780fd7f435a4e7fb3691ffb70fdcf4d0a809e6`.
Canonical exclusion digest:
`c75615d811c08c4aa490437cbbb95382ee14d4770d48daa11400c9095c9a1892`.

## Complete controlled robustness results

Each value is successful submissions out of 650 spectrum-track episodes. Conditions
are reported separately; the register-level CSV also preserves perception tails,
terminal errors, costs, invalid actions and paired changes from clean.

| Condition | Seed 0 | Seed 1 | Seed 2 | Matched Spectrum Peak |
| --- | ---: | ---: | ---: | ---: |
| Clean | 650 | 650 | 650 | 650 |
| Level -12 dB | 650 | 650 | 650 | 650 |
| Level -24 dB | 650 | 650 | 650 | 650 |
| Phase 45 degrees | 650 | 650 | 650 | 650 |
| Phase 90 degrees | 650 | 650 | 650 | 650 |
| Noise 30 dB SNR | 650 | 650 | 650 | 650 |
| Noise 10 dB SNR | 639 | 634 | 638 | 644 |
| Combined -24 dB / 90 degrees / 10 dB SNR | 636 | 637 | 638 | 641 |

All 20,800 robustness actor-episodes and 3,000 confirmation actor-episodes were
recorded and strictly reloaded. There is no robustness superiority gate. These
results show no advantage for the learned reference over the matched classical
baseline on this task. Strong noise can produce large errors, so a high success
rate alone would hide a consequential limitation. Maximum initial errors under
10 dB noise were 2,318 / 1,858 / 1,987 / 2,067 cents for seeds 0/1/2/classical;
under the combined condition they were 2,007 / 1,717 / 1,903 / 2,067 cents.

This is static corruption: one base-episode noise vector is fixed through actions,
shared across actors and relevant conditions. Noise follows phase and gain and is
scaled to measured gained-signal RMS; mixtures are not normalized or clipped.
No live-microphone, recorded-audio or arbitrary-sound claim follows from this grid.

The [one-case diagnostic](v1-respec/spectrum-failure-case.md) reproduces exact
waveform/spectrum hashes for the first failed Spectrum Peak case at 10 dB SNR.
The raw FFT estimated source 6857 cents within 0.028 cents, while the public grid
selected 5935 cents. Both neighboring grid points missed the narrow FFT main lobe;
noise exceeded the sampled signal. This identifies one representation failure,
not its prevalence or the performance of a proposed alternative. Investigating a
peak-preserving spectrum encoding or waveform estimator is a justified next study,
with a new observation identity and fresh evaluation. The frozen clean encoding
was preserved in this revision.

## Engineering evidence

- Renderer checks cover exact clean waveform/spectrum compatibility, phase/gain,
  requested versus realized SNR, revisits, actor/order independence, and warm/cold
  caches. The 128 MiB LRU accounts for retained arrays and evidence overhead; noise
  vectors are generated deterministically and not retained per study episode.
- Runner checks establish fresh actor/controller state, owned observations, no
  hidden-truth fields, explicit actor component identities, and the known
  quantization/replanning counterexample. Custom factories require no registry.
- Result checks replay controls and terminal records, recompute all metrics, reject
  malformed/nonfinite or inconsistent evidence, validate optional traces, and
  preserve explicit unknown provenance. Artifact checks cover hashes, mutation
  during evaluation, create-only outputs and incomplete training directories.
- Real pitch and PPO smoke training changed parameters, persisted and reloaded
  checkpoints, then produced valid new-runner evaluations. Pitch completed 6/6
  smoke episodes successfully. The experimental PPO smoke policy truncated 6/6;
  that is valid engineering evidence and no claim of useful learned control.
- An actual pitch-training subprocess rejects any Stable-Baselines3 import. Base
  wheel and independently installed sdist environments had neither Torch nor SB3,
  yet ran classical experiments, traces and historical/new report readers. Missing
  optional dependencies produced actionable errors without partial outputs.
- The full regression run finished with 11,578 passes, three failures and one setup
  error. Two assertions still described the former enum/CLI choices; two subprocess
  checks could not find the local `uv` binary. The obsolete assertions/retired CLI
  expectation were updated to the approved interfaces, and the local subprocess
  PATH was corrected without a committed platform workaround. All four cases passed
  on focused reruns. The final contract run passed 157 tests, including the new
  public `run` trace/provenance check; the real installed pitch workflow passed
  separately in 100.19 seconds. Thus all 11,582 cases in the full run plus the new
  API case were verified across the full and focused runs, not in one uninterrupted
  all-green run. Ruff lint/format checks also passed.

Full local logs are in `outputs/v1-respec-qualification/`: `regression.log`,
`final-contracts.log`, `installed-training-retry.log`, and the training/installation
verification subdirectories. Tests verify behavior and failure paths; they do not
turn a smoke model into a qualified performance model.

## CPU evaluation and CUDA training

Qualification used Linux/WSL, Python 3.12.3, NumPy 2.5.1, Gymnasium 1.3.0,
Torch 2.13.0+cu130, and **CPU inference with one Torch thread per process**. The
complete condition grid used six independent worker processes, each with its own
128 MiB maximum cache. Results were merged in fixed protocol order after identical
model/source/renderer checks; paired metrics were then recomputed. The initial
serial attempt was interrupted to improve scheduling and produced no complete
published result. Its log is retained. The completed parallel workflow took
1,158.36 seconds including setup, validation and confirmation; this excludes the
interrupted attempt and is not a throughput guarantee.

The three reference checkpoints already had CUDA training evidence from the prior
study. Separately, the new API performed an actual CUDA pitch smoke fit on an
NVIDIA RTX 500 Ada Generation Laptop GPU, CUDA 13.0 / cuDNN 92000: two epochs,
eight batches, 256 training and 64 validation examples. All six parameter tensors
changed. Reload placed parameters on CPU and completed one episode with zero-cent
error, 37 actions and one inference. Its 8.74-second wall time is an engineering
observation, not a scientific or performance claim. See
[the CUDA execution record](v1-respec/cuda-smoke.json). Native Windows/macOS,
other Python versions, arbitrary hardware and cross-version numerical identity
are not qualified.

## Provenance and retained evidence

The frozen pre-evaluation [protocol/model declaration](v1-respec/qualification-protocol.json)
and [qualification gate result](v1-respec/qualification-status.json) accompany
ordinary [robustness CSV](v1-respec/robustness.csv) and
[confirmation CSV](v1-respec/confirmation.csv) exports. These small review artifacts
are retained with the source. The full ordinary JSON records, all three imported
model artifacts, condition shards, execution logs and exact scripts are preserved
locally under `outputs/v1-respec-qualification/`; they are ignored by Git and are
not bundled in the package. Preserve/transfer those full records when publishing
a research archive; the CSV views alone cannot replay full record validation.

| Model | SHA-256 of `model.pt` |
| --- | --- |
| Seed 0, shipped | `3372304c022b0f13f837c79511a92f25fa7b0cd34217be976608fe46be7c6ac2` |
| Seed 1 | `703bdaaae6aed66b2553c6a6f854cbfd66e601f86935ffd373abde6c7561c120` |
| Seed 2 | `2b0ace3c7ab2e09f2c9e211914f9a877b46adc7b6bf4255328cb31fa3e54bd63` |

Qualified evaluator package-content SHA-256:
`e8c36d78062586e48ff7636e62152ee16780af386d2d20711713798c5f72600e`.
Its source base was commit `1f2ba3fe1829537c3f8e274dd45abeeafaa744d2` with the
implementation changes present (dirty tree explicitly recorded). Dependency-lock
SHA-256: `c223e113ebf6f442425fc0fe32a2ae105c98a5d002c9a4daa7ba58b2f49b7a8f`.

| Complete ordinary result | SHA-256 |
| --- | --- |
| `robustness.json` | `8c988346cfa712d692502305fef2dfab953806ccc7f07b3b583153c31575b831` |
| `confirmation.json` | `f3ff171b71438096f3e7c5bd4c2e25d3d7c39d7f9e7274ca00ae183460210013` |

The final API/documentation finish follows that evaluated snapshot: export Python
`train` and `run`, make `run` delegate to the same runner with tracing, and turn
unspecified custom-actor provenance into explicit unknown metadata. It changes
neither weights, rendering, observation bytes, decoding, control nor membership.
The saved qualification is revalidated by the final reader; a source-payload
comparison and fresh reference/example smokes are recorded separately in
`outputs/v1-respec-qualification/final-source-assurance/`. It is not presented as
a second complete 23,800-episode evaluation of the final documentation snapshot.

Wheel/sdist installation checks are retained in
`outputs/v1-respec-qualification/installed-distribution-verification/verification.json`.
Final rebuilt artifacts and fresh base-only installation evidence are under
`final-closeout-dist/` and `installed-closeout-verification/` within the same
output root, with their distribution hashes recorded by the verification script.
The package metadata still says `1.0.0`; these are local Unreleased candidate builds,
not replacements published under the existing tag/version. Assigning release
metadata and a new tag requires the separate release action.

## Reproduce the supported workflow

Install the local revision with the `pitch` extra, then run:

```bash
harpy evaluate --actor reference --actor spectrum-peak --suite clean --output runs/reference-clean.json
harpy evaluate --actor reference --actor spectrum-peak --suite robustness --output runs/reference-robustness.json
harpy summarize runs/reference-robustness.json --format csv
harpy run --actor reference --source-cents 6064 --target-note-index 12 --output runs/reference-trace.json
```

For all-seed reproduction, use the three hash-identified artifacts with repeated
`--actor` arguments. Run confirmation with only those three references. Apply
`harpy.experiments.qualification.qualify_reference` to the loaded complete robustness
and confirmation results. The named qualification checks identity, membership,
controller matching and per-register gates independently of the CLI views.
Training new seeds is supported by `harpy train pitch --profile checkpoint --seed N`;
it does not promise bitwise identical weights or qualification for arbitrary fits.
The [migration map](../v1-migration.md) keeps historical execution at `v1.0.0` clear.
