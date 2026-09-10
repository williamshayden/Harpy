# harPY v1 specification

This document is authoritative for the revised **v1 research protocol**, delivered
as **package 1.0.0, tagged `v1`**. The protocol identifiers and scientific evidence
retain their existing names.
Historical milestone specifications and reports are evidence, not requirements
that constrain this architecture. The existing `v1.0.0` development tag is preserved.

## Purpose and users

Harpy is a headless audio-control research toolkit for researchers and developers
comparing how models perceive audio and use bounded controls to reach a musical
goal. Its value is an understandable task, fair comparisons, useful extension
points, and inspectable, reproducible evidence. It is not a recorded-audio tuner
or a claim that reinforcement learning is necessary for single-sine tuning.

V1 supports a deterministic single sine, controlled gain/phase/noise, classical
baselines, a qualified supervised pitch reference, experimental PPO training,
custom Python actors, and ordinary saved experiment results. Recorded audio,
additional waveforms, chords, hosted models, UI, and a plugin registry are out of
scope. A personal-site visualization can consume results separately.

## Public interfaces

One `harpy` command provides four operations:

| Operation | Contract |
| --- | --- |
| `train` | Train `pitch` or `ppo` using `smoke` or `checkpoint`; create a new artifact. |
| `evaluate` | Evaluate one or several built-in actors or artifacts on explicit clean or robustness membership. |
| `run` | Run one episode and save/inspect its decisions and terminal outcome. |
| `summarize` | Validate a saved result and render text, Markdown, or CSV without training dependencies. |

Corresponding Python APIs accept explicit episode membership and actor factories.
A factory creates a fresh actor for every episode. A closure may share immutable
model weights. Custom actors do not modify trainer definitions or register plugins.
Synthesis, analysis, tuning, and Gymnasium remain documented supporting APIs.
Caches, persistence plumbing, and historical codecs are implementation details.

The base install supports classical actors and report reading. The `pitch` extra
supplies Torch; `train` supplies Torch and Stable-Baselines3. Pitch execution must
not import or require Stable-Baselines3. A small qualified seed-0 checkpoint ships
with the package; seeds 0, 1, and 2 are evaluated without selecting the best seed.
**Seed 0 is designated here before revised qualification begins.**

## Task invariants

The immutable source coordinate is an integer in 4800..7200 cents. The symbolic
target is one of 25 notes, MIDI 48..72. Controls are independent: octave -2..2,
semitone -12..12, fine cents -100..100. The seven actions decrement/increment each
control or explicitly submit. The action budget is 64. Successful submission
requires absolute true error **at most five cents, inclusive**. Being in tolerance
without submission is not success. Invalid bounded-control actions consume budget.

Every applied control action resynthesizes from immutable source truth plus the
current control offset. It never transforms a previously rendered waveform.

Waveform observations expose the fixed mono float32 capture of 262,144 samples at
48 kHz used to produce the spectrum. Spectrum observations expose its existing
1,961-bin representation. Both tracks expose symbolic target, controls, and budget.
Neither exposes source pitch, exact error, condition identity, or nuisance seed.
Results label tracks separately. Oracle and reward-feedback baselines have distinct
information tracks; they are not ordinary audio-perception baselines.

The exact clean condition preserves existing waveform and spectrum bytes.

## Actors and control

Estimator, decoder, and controller identities are separate result fields. An
estimator change must not implicitly change planning behavior.

The learned reference uses the existing small convolutional pitch estimator and
nearest-five-cent classification objective. Feasibility-constrained argmax limits
decoding to the public source range translated by the observed control offset.
It estimates at episode start and commits to one bounded plan with estimated
tolerance **zero**. A five-cent estimated tolerance could exceed the true five-cent
success bound. Local softmax decoding is excluded: the validation screen did not
justify adding it.

The primary Spectrum Peak comparison uses the same feasibility bounds and committed
controller. Repeated replanning is separately named and evaluated. A known
five-cent-quantization counterexample must remain covered: for target 6000 and
sources 6063..6067, committed plans retain residuals -2..2 whereas repeated
replanning can collapse all residuals to -2. PPO is a direct policy with declared
observation and action semantics. Its score is experimental, not a release gate.

The optional `waveform-fft` classical actor uses the waveform track with no ML
dependencies. It finds the strongest Hann-windowed FFT bin within the public
feasible five-cent cells, refines it with bounded three-bin quadratic interpolation
in dB, and rounds to the nearest feasible five-cent coordinate (lower ties). It
uses the same committed zero-tolerance controller. Its separately evaluated
[baseline contract](v1-waveform-fft.md) preserves the tested research algorithm;
it does not change the spectrum representation or learned reference. The v1
feature scope closes with this addition. Cell-max encoding and any corresponding
model retraining remain future research.

## Controlled robustness

The eight conditions are fixed:

| Condition | Gain relative to clean | Added phase | White-noise SNR |
| --- | ---: | ---: | ---: |
| Clean anchor | 0 dB | 0 degrees | None |
| Lower level | -12 dB | 0 degrees | None |
| Lowest level | -24 dB | 0 degrees | None |
| Phase 45 | 0 dB | 45 degrees | None |
| Phase 90 | 0 dB | 90 degrees | None |
| Moderate noise | 0 dB | 0 degrees | 30 dB |
| Stronger noise | 0 dB | 0 degrees | 10 dB |
| Combined | -24 dB | 90 degrees | 10 dB |

Apply phase during synthesis, gain second, then white noise scaled against the
gained signal's measured RMS. Record requested and realized SNR. Do not normalize
or clip the mixture. Use one deterministic standardized noise realization per base
episode, shared across actors and applicable conditions, fixed throughout that
episode. Revisiting a coordinate must reproduce its observation independently of
evaluation order and cache state. This is **static corruption**, not live microphone
noise. Nuisance identity belongs in evaluator evidence, never actor observations.

The evidence cache includes rendering/encoding configuration, effective pitch,
condition, and nuisance identity in its key. Its retained-array budget is 128 MiB.
It must not retain every long noise waveform encountered during a study.

## Data selection and measurement

Retain coordinate partition `harpy-sine-pitch-grid-v1`: central 5000..7000 partitioned
into 1,400 training, 200 validation, and 401 IID coordinates; lower 4800..4999 and
upper 7001..7200 held out. All transformed versions of a source stay in its source
partition. PPO initial sources use only the training partition. Subsequent control
trajectories can visit coordinates outside that partition; reports disclose this.

Pitch and PPO training use clean data. Select pitch checkpoints using clean
validation only; PPO saves the policy at the final training step without validation
selection or a performance gate. Evaluate all three reference seeds across all
eight conditions with no condition-specific adaptation. Direct perception and
closed-loop outcomes are separate measurements:

- Perception: error within one/five cents, mean, tail (p99), and maximum error.
- Control: submitted success, truncations, invalid actions, action counts, inference
  calls, final-error distribution, and terminal records.
- Robustness: paired degradation from the same clean episodes, retaining actor,
  observation track, register, condition, and seed identity.

Never pool observation tracks or nuisance conditions into a headline score.
Unknown custom-actor provenance is explicitly unknown, not inferred from a name.

## Results and reproducibility

Use one ordinary experiment-result schema. Qualification is a named protocol
applied to ordinary results, not another report format for each intervention.
Record task/condition configuration, observation track, separate actor components,
model/source hashes, seeds, runtime, explicit episode membership, terminal records,
and optional decision traces. Metrics must be recomputed and validated from records
on load. Reject malformed/nonfinite data and changed artifact payloads during
evaluation. Outputs are create-only; interrupted runs cannot masquerade as complete
results or silently replace existing data.

Historical report readers remain available. Historical artifact execution and
training workflows remain reproducible through `v1.0.0`; the revised package need
not maintain every old command or frozen training wrapper.

## Acceptance

For **each** supervised-reference seed 0, 1, and 2:

1. 650/650 submitted successes on the established clean benchmark: 250 IID,
   200 lower-register, and 200 upper-register episodes, each register reported.
2. 1,000/1,000 submitted successes on a newly frozen confirmation manifest:
   400 IID, 300 lower-register, and 300 upper-register episode combinations.
3. Zero invalid actions and zero budget truncations on both suites.

Freeze confirmation membership before final evaluation. Exclude duplicate and
previously evaluated source/target pairs, including prototype probes. Describe this
as fresh combinations within the known rendering family, **not unseen audio**.

Run the complete eight-condition grid on the 650 benchmark episodes for each of
the three learned references and the matched classical baseline. Report all results;
no robustness superiority threshold blocks release. One-cent accuracy is diagnostic.

Engineering acceptance covers clean compatibility; phase/gain/SNR; actor/order/cache
nuisance identity; fresh actor lifecycle; no truth leakage; matched control and the
quantization counterexample; real pitch/PPO parameter updates and reload/evaluation;
strict result round trips, metric recomputation, provenance and artifact mutation;
interrupted outputs; wheel and sdist installs; documented commands; and report
reading without optional training dependencies.

Qualify Linux/WSL on Python 3.12 first. Record CPU evaluation and CUDA training
evidence separately. Do not advertise unverified platforms or guarantees for
arbitrary user-trained models. A failure of any clean reference gate keeps the
revision Unreleased; it must not be fixed by weakening the gate or selecting seeds.

## Delivery

1. Authoritative spec, acceptance checklist, migration map.
2. Consolidated runner, actor lifecycle, observations, ordinary results.
3. Constrained decoding and matched committed control; pitch/PPO workflows.
4. Waveform observations, deterministic corruption, complete protocols.
5. Qualification, seed-0 checkpoint, distribution and examples.

The decoder prototype has been integrated and its evidence preserved. The completed
qualification and subsequent engineering checks are recorded in the
[acceptance checklist](v1-acceptance.md). The public `v1` release delivers this
implementation without revising the scientific protocol or historical evidence.
