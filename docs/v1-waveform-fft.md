# Optional waveform FFT baseline

`waveform-fft` supplies a classical reference for Harpy's waveform observation
track. It uses only the base NumPy/Gymnasium installation. The spectrum baseline,
default CLI actor, public observation bytes, and qualified learned models retain
their existing behavior. This is the final feature addition within revised v1;
further spectrum representations and learned-model changes are future research.

```bash
harpy evaluate --actor waveform-fft --actor spectrum-peak \
  --suite smoke --output runs/fft-comparison.json
harpy run --actor waveform-fft --condition noise-10db \
  --source-cents 6857 --target-note-index 13 --output runs/fft-trace.json
harpy summarize runs/fft-trace.json --format markdown
```

Run a clean episode in memory from Python with the same ordinary runner:

```python
from harpy.experiments import EpisodeSpec, run, waveform_fft_actor_spec

result = run(EpisodeSpec("fft-example", 6857, 13), waveform_fft_actor_spec())
```

## Algorithm and identities

The actor receives only the public 262,144-sample mono float32 capture at 48 kHz,
target, controls, and remaining budget. It computes the existing calibrated
Hann-windowed FFT and searches bins inside the public feasible five-cent cells,
including their outer half-cell edges. It refines the strongest bin using a
concave three-bin quadratic fit in dB, bounded to half a bin, then selects the
nearest feasible five-cent coordinate; ties select the lower coordinate.
Silence selects the lowest feasible coordinate, matching the research prototype's
all-zero decoder output. This is not a confidence or live-audio rejection system.

Each fresh episode actor estimates once and commits to a bounded plan at estimated
tolerance zero. It reports one inference call, including when the plan immediately
submits. The estimator is independent of the target. The controller uses the target
and current controls; the environment enforces the 64-action budget. Five-cent
class spacing and the inclusive five-cent true-error success bound match the
spectrum comparison; the estimated planning tolerance remains zero.

| Component | Identity |
| --- | --- |
| Actor | `waveform-fft` |
| Observation | `waveform` |
| Estimator | `harpy-hann-quadratic-fft-v1` |
| Decoder | `harpy-feasible-fft-peak-nearest-five-cent-v1` |
| Controller | `harpy-committed-zero-tolerance-plan-v1` |

The source-checkout prototype `research/spectrum_preservation/encoders.py` used a
one-hot adapter to express this estimate through the spectrum planner. The built-in
returns the same coordinate directly and shares the ordinary committed controller.
Production execution does not depend on research scripts or optional ML packages.

## Evidence and limits

The frozen spectrum-preservation study compared three waveform actors on 600 pairs
(200 IID, 200 lower, 200 upper) under each of eight conditions: 14,400 actor-episodes.
Its quadratic FFT prototype achieved 600/600 submitted successes in every condition,
with no invalid actions or truncations and a maximum error of two cents. It matched
every clean estimate and action of the legacy spectrum adapter. The legacy adapter
scored 598/600 at 30 dB noise, 588/600 at 10 dB, and 587/600 in the combined condition;
the FFT prototype rescued these failures without introducing new ones.

These are fresh episode combinations within Harpy's known single-sine rendering
family, with one static nuisance realization per pair. They are now consumed
evidence. The result does not establish universal success, recorded-audio
robustness, or a benefit for learned models. The original learned qualification
remains a separate experiment with its original actors and source hashes.

An isolated local CPU microbenchmark measured a median 8.64 ms for FFT analysis
and estimation versus 8.38 ms for the legacy encoding. Paired timing was near
parity; no speedup or general performance guarantee is claimed. Synthesis and
control execution were excluded. No training or GPU computation was used.

The source checkout retains the frozen study, compact metrics and timing views
under `research/spectrum_preservation/`; these research scripts are not installed
package APIs. Complete ordinary results remain in the local research archive
`outputs/spectrum-preservation-study/experiment.json` (about 18 MiB). That archive
is not tracked or bundled; exact validation of the complete saved study requires
obtaining it separately. A fresh clone includes the research code and compact
views, not the full recorded experiment.

- Study result SHA-256: `69db356439daf33f320d7a596d236584c8053f55b1954927c452dbda8679be8f`.
- Membership digest: `7ec38d3af9e0a30baba4437e00350b1fdfe77166b50f72b92fe0c1ed185c9c48`.
- Production package hash at study time: `9cca1d94ed865ed07f1a5072a394c83970fccf8575cda7d380067e7152754ab1`.

Integration verification reconstructs the saved initial captures and compares the
built-in's estimate, first action and inference count with the frozen prototype.
Focused tests cover observation validation, controller lifecycle and decoding;
wheel/sdist checks exercise the public Python API, CLI, traces and report reading
outside the checkout without Torch or Stable-Baselines3. Evidence is retained
under `outputs/waveform-fft-integration/`.
