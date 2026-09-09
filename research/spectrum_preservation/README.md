# Spectrum preservation: measured improvement and tradeoffs

The study found an avoidable loss of pitch information in Harpy's current
point-sampled log spectrum. Both a cell-maximum representation and a waveform FFT
reference repaired every observed noisy failure in this cohort. **The waveform
FFT reference is the stronger immediate baseline candidate:** it retained the
legacy clean estimates and near-parity encoding cost. Cell-max is a useful
versioned representation experiment, with a small precision and cost tradeoff.

This directory preserves the research implementation and its original evidence.
After the study, its quadratic FFT algorithm was promoted to the optional
[`waveform-fft` built-in](../../docs/v1-waveform-fft.md). The study itself changed
no production code, model weights or public observation bytes. No model training
or adaptation was performed; this does not establish learned-model gains.

## What was compared

The [design](PLAN.md) and [600-pair membership](membership.json) were frozen before
main evaluation: 200 IID, 200 lower and 200 upper pairs, balanced across targets,
excluding 2,324 recorded historical/confirmation pairs and all engineering anchors.
These are fresh combinations within the known rendering family, not unseen audio.

All three actors receive exactly the same waveform capture, public bounds, target,
controls and budget, estimate once, and execute the unchanged committed controller.
Legacy versus cell-max uses the identical feasibility decoder. The raw FFT reference
is also explicitly quantized to five cents; finer control precision cannot explain
its result. Cell-max preserves the existing piecewise-linear dB curve's maximum
within each center +/-2.5 cents, including interpolated cell boundaries.

All eight conditions completed: **14,400 actor-episodes**, saved as ordinary Harpy
experiment results and strictly reloaded. The full invocation took 1,293.40 seconds
with up to four CPU workers; this is an observed study duration, not a throughput
guarantee. No Torch, Stable-Baselines3 or GPU computation was needed.

## Results

Successful explicit submissions out of 600, with inclusive five-cent tolerance:

| Condition | Legacy point encoding | Cell-max | Quantized waveform FFT |
| --- | ---: | ---: | ---: |
| Clean | 600 | 600 | 600 |
| Level -12 dB | 600 | 600 | 600 |
| Level -24 dB | 600 | 600 | 600 |
| Phase 45 degrees | 600 | 600 | 600 |
| Phase 90 degrees | 600 | 600 | 600 |
| Noise 30 dB SNR | 598 | 600 | 600 |
| Noise 10 dB SNR | 588 | 600 | 600 |
| Combined -24 dB / 90 degrees / 10 dB SNR | 587 | 600 | 600 |

Both alternatives rescued the same two / twelve / thirteen failures under 30 dB,
10 dB and combined conditions, respectively, with zero newly failed pairs. The
30 dB rescues were both upper-register; the 10 dB rescues were one IID and eleven
upper; all thirteen combined rescues were upper. Every actor used one inference
per episode. There were no invalid actions or budget truncations anywhere.

Legacy maximum initial error was 1,412 cents at 30 dB noise and 2,013 cents at
10 dB and combined noise. Cell-max stayed within three cents; the FFT reference
stayed within two cents in every condition. Conditions and registers are retained
separately in the [full measurements](results/summary.md) and
[paired CSV](results/paired-actors.csv).

## The clean-precision tradeoff

All actors achieved clean one-cent accuracy on 376/600 pairs (62.67%), with zero
paired one-cent gains or losses. That threshold alone hides a smaller change:
cell-max worsened **31 clean estimates from two to three cents**: nine IID and
22 lower-register, with none in the upper register. Clean mean absolute error
rose from 1.1433 to 1.1950 cents; p99 and maximum rose from two to three cents.
The waveform FFT reference matched every legacy clean estimate and action.

This is consistent with FFT-bin localization error preceding five-cent cell
quantization. Preserving a bin's maximum repairs gross information loss, but does
not itself recover sub-bin location. The quadratic reference supplies that
additional location estimate before applying the same five-cent quantization.

## Encoding cost

A separate one-process microbenchmark ran after the study, on eight fixed captures
and five balanced repetitions per method (40 measured calls each). Each method
was warmed once per capture. Times include analysis and encoding, and exclude
synthesis, controller execution, imports and model inference.

| Method | Median | p95 |
| --- | ---: | ---: |
| Legacy point encoding | 8.38 ms | 9.62 ms |
| Cell-max | 10.67 ms | 11.83 ms |
| Quantized waveform FFT | 8.64 ms | 10.31 ms |

Paired by capture and repetition, cell-max added a median 2.10 ms. The FFT
reference's paired median difference was -0.03 ms, consistent with near parity;
this is not a speedup claim. These are local CPU measurements, not platform-wide
performance guarantees. See [the timing observations](results/encoding-timing.csv).

## What this justifies next

1. The tested waveform FFT actor is now an optional built-in classical baseline,
   retaining its declared five-cent decoder and committed controller. The current
   spectrum baseline remains useful for measuring information lost by the encoder.
2. Keep cell-max under a new experimental encoder identity. To assess learned
   benefit, train matched seeds 0/1/2 on clean data with the existing objective,
   split, budget and clean-validation selection, then use a new frozen final cohort.
3. Promote a revised spectrum representation only after that comparison. Existing
   model preprocessing and qualification must not be silently relabeled.

The evidence supports a representation bottleneck for these procedural single-sine
captures. It does not establish general recorded-audio robustness, arbitrary SNR
performance, or a universal success probability. Repeated source coordinates and
one static nuisance realization per pair remain limitations. This 600-pair study
is now consumed evidence; it must not become the selection set for an unreported
revised algorithm. No novelty claim is made for FFT peak interpolation or pooling.

## Reproduce and inspect

From the source checkout with base dependencies installed:

```bash
PYTHONPATH=src:. python -m research.spectrum_preservation.study --manifest research/spectrum_preservation/membership.json --plan research/spectrum_preservation/PLAN.md --output outputs/spectrum-preservation-reproduction --workers 4
PYTHONPATH=src:. python -m research.spectrum_preservation.readout --result outputs/spectrum-preservation-reproduction/experiment.json --output outputs/spectrum-preservation-reproduction/readout
PYTHONPATH=src:. python -m research.spectrum_preservation.timing --output outputs/spectrum-preservation-reproduction/encoding-timing.csv
```

Run timing after the study finishes. Outputs are create-only. The 48 research tests
passed, including exact native/adapter equivalence, empty cells, preserved peaks,
input validation, deterministic membership/exclusions, pairing, source mutation,
strict result merging and interrupted-output guards. Ruff lint/format checks pass.
Two independent reviews checked the algorithm/fairness and recomputed the final
condition/register and paired outcomes.

The exact ordinary result and condition shards remain locally under
`outputs/spectrum-preservation-study/`; compact exported views are included here.
The full result is about 18 MiB and is not bundled in the package or tracked in Git.
Preserve it when transferring a research archive; CSV views cannot replay the full
record validation. Full result SHA-256:
`69db356439daf33f320d7a596d236584c8053f55b1954927c452dbda8679be8f`.

Membership digest:
`7ec38d3af9e0a30baba4437e00350b1fdfe77166b50f72b92fe0c1ed185c9c48`.
The qualified production package-content SHA-256 at the study snapshot was
`9cca1d94ed865ed07f1a5072a394c83970fccf8575cda7d380067e7152754ab1`.
The ordinary result records separate research-script hashes and actor identities;
[export provenance](results/README.md) records runtime and exact input hashes.
