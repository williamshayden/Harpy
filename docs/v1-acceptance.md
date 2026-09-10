# Revised v1 acceptance checklist

Authoritative contract: [v1-spec.md](v1-spec.md). Implementation and qualification
are complete for the stated Linux/WSL Python 3.12 scope. The public release is
**package 1.0.0, tagged `v1`**. The older `v1.0.0` tag preserves a development
checkpoint. See [the qualification record](verification/2026-09-08-v1-respec-qualification.md)
for the 8 September scientific qualification, and the
[9 September repository review](reviews/2026-09-09-v1-release-review.md) for the
subsequent code, documentation, and distribution checks. Their source identities
and verification scopes remain distinct.

- [x] Product scope, interfaces, seed-0 designation, and fixed acceptance gates documented.
- [x] One factory-based episode runner and explicit estimator/decoder/controller identities.
- [x] Owned waveform/spectrum observations with no hidden truth or nuisance fields.
- [x] Matched committed controllers and quantization/replanning regression coverage.
- [x] Eight deterministic conditions; clean byte compatibility; phase/gain/SNR checks.
- [x] Nuisance identity stable across actor/order/cache; 128 MiB evidence cache.
- [x] Ordinary strict results, recomputed metrics, paired clean degradation, traces.
- [x] Create-only outputs, provenance, artifact mutation and interrupted-output checks.
- [x] Torch-only pitch extra and full pitch/PPO train extra.
- [x] Pitch/PPO smoke parameter changes, reload, and valid evaluation.
- [x] Confirmation 1,000 membership frozen with exclusion evidence before final evaluation.
- [x] Seed 0 clean benchmark 650/650 and confirmation 1,000/1,000; no invalid/truncated episodes.
- [x] Seed 1 clean benchmark 650/650 and confirmation 1,000/1,000; no invalid/truncated episodes.
- [x] Seed 2 clean benchmark 650/650 and confirmation 1,000/1,000; no invalid/truncated episodes.
- [x] All 8 x 650 robustness episodes for each reference seed and matched Spectrum Peak.
- [x] Qualified seed-0 checkpoint included; all seed results retained without selection.
- [x] Wheel and sdist installed; all four documented CLI workflows exercised.
- [x] Base installation classical execution and historical/new report reading without Torch/SB3.
- [x] Linux/WSL Python 3.12 evidence; CPU evaluation and CUDA training identified separately.
- [x] README/examples and migration match actual interfaces and verified limitations.

Evidence highlights:

- Clean benchmark per seed: IID 250/250, lower 200/200, upper 200/200.
- Confirmation per seed: IID 400/400, lower 300/300, upper 300/300.
- All 18 clean seed/suite/register gates passed with zero invalid actions or truncations.
- Robustness: 20,800 records (8 conditions x 650 episodes x 4 actors); confirmation: 3,000 records. Strong noise and combined-condition failures remain reported.
- Original 8 September engineering qualification: 11,578 passes; four initial interface/environment failures resolved on focused reruns. Final contracts: 157 passes. Installed pitch workflow: one additional targeted pass. These are historical counts across multiple runs.
- Wheel and sdist independently installed with no Torch/SB3; classical workflows and new/historical readout verified. Pitch and PPO actual training/reload checks and a separate CUDA pitch smoke completed.
- Compact gate, model, CUDA and full metric exports: `docs/verification/v1-respec/`. Complete ordinary JSON, imported seeds 1/2, logs and execution scripts remain under ignored `outputs/v1-respec-qualification/`.

The 9 September review reconciled all **11,580 current tests** across the broad
regression and separate training runs, with no failures, skips, or missing cases.
Fresh base wheel/sdist environments passed 58 installation checks. The current
reader revalidated all 38,200 saved records across reference qualification and
the separate representation study without changing canonical records or archive
hashes; the scientific cohorts were not rerun. Details and local evidence paths
are in the repository review linked above.

Only seed 0 ships with the package. The public release includes the full 14,400-record
representation study with a portability/hash manifest. Full reference-qualification
results and seed-1/2 artifacts remain local; a fresh clone alone cannot reproduce
those historical studies.
The old `v1.0.0` development tag stays intact. Publishing v1 does not change the
research protocol or relabel historical qualification as a new scientific run.
