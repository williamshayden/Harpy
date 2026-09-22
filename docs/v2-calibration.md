# Sound-matching calibration and frozen dataset protocol

This record covers the bounded CPU calibration pilot and the target generator
`harpy-grouped-sound-targets-v1`. It does not claim perceptual acceptance. The
objective remains `harpy-multires-spectral-v1` and the renderer remains
`harpy-compact-synth-v1`; their formulas, precision and timing are unchanged.

## Objective calibration

Fourteen deterministic fixture pairs cover oracle identity, chord permutation,
inactive-decay aliases, 5/100-cent detuning, missing chord tone, brightness,
attack/release, level, silence, clipping and phase. Every fixture retains exact
target/candidate float32 hashes, objective components and lossless WAV pairs.
The missing-tone fixture removes one equally weighted voice without renormalizing
the other two. Clipping, polarity inversion and phase shift are audio fixtures
outside the legal patch space.

| Fixture | Distance |
| --- | ---: |
| Identity | 0 |
| Chord permutation | 0 |
| Inactive decay at 0 dB sustain | 0 |
| Detuning, 5 cents | 0.774162 |
| Detuning, 100 cents | 1.726596 |
| Missing middle chord tone | 0.864569 |
| Brightness, 500 Hz | 1.912252 |
| Attack, 250 ms | 0.719150 |
| Release, 750 ms | 1.268402 |
| Level, -12 dB | 1.347782 |
| Silence | 3.428695 |
| Clip at ±0.025 | 1.331132 |
| Polarity inversion | 0 |
| Circular shift, 17 samples | 0.034974 |

These checks show that the chosen scalar detects the listed substantial errors
and preserves exact aliases. They do not establish a universal perceptual
ordering: the magnitude objective deliberately cannot distinguish polarity
inversion, and small timing changes can receive a nonzero score. Envelope RMSE,
signed RMS level error and evaluator-only sorted pitch-set errors remain separate
diagnostics. **No listening review has been performed, no sound-acceptance cutoff
is frozen, and no perceptual success rate may be reported.** Continuous distance
and submission/error/truncation rates are the release reporting contract.

## Membership and source-family grouping

The evaluator manifests are checked in under
[`src/harpy/tasks/sound_match/data`](../src/harpy/tasks/sound_match/data).
`catalog.json` binds each file's bytes with SHA-256. Entries bind an opaque episode
ID, canonical target and initial patches, editable mask, source-family ID,
cohort/cell, target chord family, structural hash, exact float32 audio hash and the
32-evaluation/64-decision budgets. The actor only sees the ordinary observation;
these evaluator labels and group names are not in its payload.

Each source family contains one reconstruction target and one derived shaping
target. The shaping initial patch is exactly that family's reconstruction target;
its target changes only pitch, envelope or brightness. Source families stay in one
split. The three shaping masks cycle evenly within each voice count. The initial
reconstruction patches are the public one-note/chord defaults.

The generator uses independent SHA-256-derived PCG64 streams from generator seed
20260916, split, voice count, family index and retry count. Independence is enforced
by identities and grouping, rather than asserted from different random seeds.
Single-note pitches range over every integer cent in 4800..7200. Chord roots span
the valid range for their voicing, with three inversions and cent transposition.
Controls vary independently except for the explicit compositional holdout and
the shaping reachability mask. Brightness is sampled uniformly in log frequency;
other scalar controls use uniform legal integer values. Waveforms are sampled
uniformly before rejection of the held-out combination.

Structural alias rule v1 canonicalizes decay to 100 ms when sustain is 0 dB.
Canonical note sorting removes voice labels. Before accepting a family, the
generator checks canonical structural hashes and actual rendered float32 hashes
against every previously accepted target in every split. A collision retries the
whole family. Exact aliases are therefore excluded even between different groups
in the same split. The separate strict cross-split validator checks family IDs,
episode IDs, structural hashes and audio hashes. Shared public starts are allowed;
there is no approximate-similarity transitive closure.

Compositional targets hold out the joint combination `square` waveform and attack
at least 160 ms. Other splits exclude that combination, while retaining each
individual factor elsewhere. All compositional chords additionally use diminished
or augmented pitch-class families; calibration/train/validation/IID use major,
minor and suspended families. The chord-class check includes all transpositions
and inversions. This is a holdout of target distributions. Compositional shaping
starts expose those target families, and baseline proposals may encounter them;
no stronger never-seen-family claim is made.

## Counts, selection and study boundary

| Split | Targets | Each reconstruction cohort | Each shaping cohort | Each shaping mask/voice |
| --- | ---: | ---: | ---: | ---: |
| Calibration pilot | 96 | 24 | 24 | 8 |
| Training | 2016 | 504 | 504 | 168 |
| Validation | 192 | 48 | 48 | 16 |
| IID final test | 192 | 48 | 48 | 16 |
| Compositional final test | 192 | 48 | 48 | 16 |

The 2016-target training profile and 192-target validation profile keep all cells
represented on CPU without storing a waveform corpus. They are bounded research
memberships, not a statistical power claim. Final tests total 384 targets.
Generation of test patches and their identity hashes does not evaluate any actor.
Actor evaluation of these final memberships must wait until M0–M4 pass and the
actors, training configuration, selection rule and source identities are frozen.

Checkpoint selection uses the arithmetic mean of eight validation cell means:
voice count 1 or 3 crossed with all-fields reconstruction, pitch, envelope and
brightness shaping. Each cell has equal weight despite differing row counts.
Every validation episode must have a submitted candidate; missing/unsubmitted
rows make that checkpoint ineligible. Exact ties select the earliest checkpoint.
The reference-quality criterion is fixed before training: each of seeds 0, 1 and
2 must have lower validation macro distance than the fixed actor. This criterion
does not assert superiority over classical search or perceptual acceptance. A
failure retains all evidence and labels the learned actor experimental. Seed 0 is
designated before qualification if one checkpoint is bundled.

Fixed and deterministic pattern search run once for each matched membership;
random search and learned training use seeds 0, 1 and 2. The pilot uses seed 0
only. The existing baseline algorithms and their metadata are retained. Random
streams are derived from actor seed and opaque episode ID, independent of target
generation. All actors use the same starts/masks and 32 total candidate
evaluations, including the initial patch and any best-candidate restoration.
Quality at 1/4/16/32 is the best audition curve, never substituted for submitted
distance. All final metrics retain cohort and terminal-status denominators.

## Reproduction

The command outputs are create-only. Run from an installed checkout or set
`PYTHONPATH=src`. Generation retains metadata and hashes, one waveform at a time.

```bash
python research/sound_matching/calibrate.py fixtures --output runs/calibration-fixtures
python research/sound_matching/calibrate.py manifests --output runs/frozen-memberships
python research/sound_matching/calibrate.py pilot \
  --manifest-directory runs/frozen-memberships --output runs/calibration-pilot
```

The pilot saves a strict portable bundle with all 96 target WAVs, 288 submitted
WAVs and submitted patches, complete ledgers and source/dependency identities.
Calibration fixtures retain another 28 WAVs. WAV retention supports subsequent
human review; file generation is not a claim that anyone listened.

Near-equivalence screening is restricted to the calibration targets. It uses
L2 distance between unit-normalized 256-bin pooled full-clip log-magnitude FFT
features with an investigation threshold of 0.03. This is a coarse diagnostic,
not the ranking objective, a perceptual cutoff, or a rule that merges groups.
Exact cross-split independence is checked across every manifest; approximate
cross-split acoustic independence is not claimed.

## Pilot measurements

The retained 96-target seed-0 pilot completed on 2026-09-16 in 552.13 seconds of
evaluation, with 86.95 MiB process peak RSS and a 16 MiB waveform-cache ceiling.
All 288 actor-episodes explicitly submitted; none errored or truncated. The
complete bundle occupies 153,468,226 bytes (146.36 MiB). Target rendering averaged
27.72 ms, with median 20.79 ms and p90 52.98 ms. These are measurements from one
Linux/WSL Python 3.12.3 / NumPy 2.5.1 CPU run, not throughput guarantees.

Mean submitted distance, lower is better:

| Cohort, 24 targets each | Fixed | Random search | Pattern search |
| --- | ---: | ---: | ---: |
| Single reconstruction | 3.893903 | 1.348460 | 2.870935 |
| Chord reconstruction | 3.079632 | 1.415372 | 2.533958 |
| Single shaping | 1.358549 | 0.331829 | 0.365884 |
| Chord shaping | 0.941574 | 0.463842 | 0.366457 |

The observed search improvement and bounded CPU/memory cost support retaining
the 32-evaluation / 64-decision default. Pattern search is substantially weaker
on reconstruction than random search here; this is retained as a baseline
limitation. Neither algorithm nor the target memberships were changed after this
pilot. Full distributions, 1/4/16/32 audition curves, paired differences, timings
and render/cache counts are retained in
[`pilot.json`](../research/sound_matching/evidence/pilot.json) and
[`pilot-cell-metrics.json`](../research/sound_matching/evidence/pilot-cell-metrics.json).
Of 4,560 calibration fingerprint comparisons, two fell below the investigation
threshold; both were reconstruction/shaping pairs within the same source family.
This does not establish approximate acoustic independence across splits.

On 2026-09-22, strict reloading revalidated all 288 records and bundle artifacts.
All 74 files in the retained pilot source snapshot match their recorded hashes.
The verification also re-rendered all 2,688 frozen targets, checked their exact
float32 hashes, checked group/structural/audio/episode separation across splits,
recomputed all 14 fixture scores and 28 WAV files, and reproduced the calibration
near-equivalence screen. It took 78.60 seconds. The machine-readable record is
[`m2-reverification.json`](../research/sound_matching/evidence/m2-reverification.json).
No final-test actor was evaluated during those checks.

The M2 freeze decision is **continuous distance frozen; perceptual acceptance
unqualified**. The objective, renderer, memberships, budgets, baseline settings,
checkpoint selection rule and three-seed reference criterion above are frozen.
Training and final-study evidence are reported separately. The 24 median/worst
listening pairs in the retained
[`listening index`](../research/sound_matching/evidence/listening-index.json) are
available for human review; a listening review remains unperformed.

The full local retained inputs are `runs/m2-pilot-seed0`,
`runs/m2-calibration-fixtures` and `runs/m2-pilot-source`. Recheck them without
training or actor evaluation:

```bash
python research/sound_matching/verify_m2.py \
  --pilot runs/m2-pilot-seed0 --fixtures runs/m2-calibration-fixtures \
  --snapshot runs/m2-pilot-source --output runs/m2-reverification.json
```

## Matched study driver

[`study.py`](../research/sound_matching/study.py) freezes the evaluator package,
driver, data catalog, all three reference model inventories and study protocol.
Run it from an immutable source snapshot after M0–M4 verification. The eleven
actors are fixed and pattern once, random seeds 0/1/2, and inverse plus
inverse-refine for each training seed. Each split retains all 192 targets.

```bash
mkdir -p STUDY_ROOT
python research/sound_matching/study.py freeze --models MODEL_ROOT --output STUDY_ROOT/plan.json
python research/sound_matching/study.py run --plan STUDY_ROOT/plan.json \
  --models MODEL_ROOT --split validation --output STUDY_ROOT/validation
# Only after the study freeze, use the same command for iid-test and compositional-test.
# An interrupted run uses the original output and adds --resume.
python research/sound_matching/study.py report --plan STUDY_ROOT/plan.json \
  --bundles STUDY_ROOT --output STUDY_ROOT/report.json
```

`MODEL_ROOT` contains `seed0`, `seed1` and `seed2`. The report recomputes all cell
means from strict bundle readers and applies the predeclared all-three-seeds
criterion to validation only. Every seed is reported, including any failure.
