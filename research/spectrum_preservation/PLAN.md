# Spectrum preservation study, frozen design

Date: 2026-09-08. Status: design fixed before the 600-pair study.
This investigation uses Harpy's existing custom-actor API. It does not replace
the qualified v1 encoder, model, task, result schema, or release evidence.

## Question and intervention

Does preserving FFT peaks when converting to the five-cent logarithmic grid
reduce strong-noise tuning failures without breaking clean five-cent tuning?
The motivation is the already diagnosed source-6857/target-6100 failure; that
case and prior v1 suites are development evidence, not fresh confirmation here.

One primary candidate is fixed: retain the existing calibrated Hann FFT,
262,144 float32 samples at 48 kHz, 1,961 grid centers, -120 dBFS floor and fixed
0..1 mapping. For each center, use the maximum of the existing piecewise-linear
dB curve over center +/-2.5 cents. Check enclosed FFT samples and both
interpolated endpoints. This also defines cells containing no FFT samples.
There is no normalization by observed amplitude and no adaptive bandwidth.

The three actors are:

| Actor | Perception | Decoder | Controller |
| --- | --- | --- | --- |
| `legacy-point` | Exact current log-frequency point interpolation | Public feasibility argmax on the five-cent grid | Existing committed plan, estimated tolerance zero |
| `cell-max` | The fixed cell maximum defined above | Identical public feasibility argmax | Identical committed plan |
| `quadratic-fft` | Strongest FFT peak in the same feasible cells' frequency coverage, concave three-bin dB interpolation bounded to half a bin | Explicit nearest-five-cent quantization, lower-index tie rule | Identical committed plan |

The FFT reference is also quantized to five cents. It is separately labeled and
tests whether useful information remains in the waveform; it is not a new model
or a claim of continuous-frequency precision. Its one-hot controller adapter is
not labeled as a spectrum. All actors receive the same waveform observation track,
share no mutable episode state, and estimate once. The legacy adapter must reproduce
the native Spectrum Peak actor's estimates, actions and evidence exactly.

Band-energy variants, different windows, shorter captures, learned adaptation,
new training seeds and condition-specific choices are outside this study. Band
energy would introduce additional power/PSD and bandwidth-normalization choices.
Max pooling is not frequency-neutral: wider cells admit more noise samples. It
also retains FFT-bin localization error before quantization, so clean one-cent
accuracy could worsen. Those tradeoffs must remain visible.

## Membership and execution

Freeze 600 unique source/target pairs: 200 IID, 200 lower and 200 upper. Within
each partition, select eight eligible sources for each of the 25 targets using
the SHA-256 ranking in `membership.py`, seed 20260908. Exclude the 1,324 recorded
historical pairs and all 1,000 consumed v1 confirmation pairs. No difficulty
filter or outcome-dependent selection is allowed. The engineering anchors
(4800, target index 24), (6064, index 12) and (6857, index 13) do not occur in
the selected membership. Membership is fresh combinations in the known rendering
family, excluding the recorded inventory, not unseen audio or a new domain.

The frozen membership digest is
`7ec38d3af9e0a30baba4437e00350b1fdfe77166b50f72b92fe0c1ed185c9c48`.
The ordered episodes and deterministic nuisance seeds are written to
`membership.json` before the main evaluation starts. Use the existing eight
robustness conditions unchanged, with one static noise realization per pair,
shared across actors and applicable conditions. Total: 14,400 actor-episodes.

Evaluate through the ordinary Harpy runner, with up to four condition workers
and a 128 MiB evidence cache per worker. Workers verify package and research-file
identities; shards are strict-loaded and merged in fixed condition order. Save
ordinary experiment JSON, CSV and Markdown. Retain incomplete output directories
and reject overwrites. No qualification labels or thresholds are retrofitted.

## Measures and interpretation

The primary comparisons are paired `cell-max` versus `legacy-point` submitted
success under 10 dB noise and under the combined condition, reported separately.
Also report clean five-cent success, all eight conditions, and every register.
Keep perception and control metrics separate: one/five-cent accuracy, p99/max
error, invalid actions, truncations, actions and inference counts. Report both
rescued and newly failed pairs; do not conceal clean one-cent regressions behind
an aggregate success score. Per-condition paired changes from clean are provided
by the ordinary result reader. No pooling across conditions into one headline.

These are measured finite-cohort outcomes. Repeated source coordinates and a
single fixed nuisance realization per pair do not establish universal success
probabilities. The new study is consumed evidence after this run. Any revised
algorithm chosen after seeing it needs a separately frozen final evaluation.
No learned-model robustness claim follows without matched clean training and
validation selection for seeds 0, 1 and 2 on a versioned new representation.

After the cohort, measure isolated analysis-plus-encoding costs on the same eight
fixed captures: sources 4800/6000/6857/7200 under clean and 10 dB noise, nuisance
seed 17. Warm up each method once per capture, then use five cyclically balanced
repetitions in one process. Exclude synthesis and control, report median/p95 and
paired overhead, and make no end-to-end throughput or GPU speedup claim. This
measurement does not select or alter the encoder.

## Prior methods

Quadratic peak interpolation is established and can have window-dependent bias;
the single diagnostic's very small error is not a guarantee. See
[Smith's interpolation method](https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html)
and [bias discussion](https://www.dsprelated.com/freebooks/sasp/Bias_Parabolic_Peak_Interpolation.html).
For the distinction between density and spectrum scaling, see the
[SciPy periodogram documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html).
This is an application-specific representation experiment, not a novelty claim.

## Reproduction

From the source checkout with Harpy's base dependencies installed:

```bash
python -m research.spectrum_preservation.study --manifest research/spectrum_preservation/membership.json --plan research/spectrum_preservation/PLAN.md --output outputs/spectrum-preservation-study --workers 4
```

Set `PYTHONPATH=src:.` when running directly from a non-installed checkout.
The membership utility is used once to create the frozen file; existing files
must not be regenerated or overwritten to obtain more favorable episodes.
