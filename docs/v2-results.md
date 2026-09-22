# Frozen v2 sound-matching results

The 2026-09-22 matched study retains 192 validation, 192 IID-test and 192
compositional-test targets, each evaluated by eleven actors. Models, memberships,
the evaluator and the 32-evaluation/64-decision ceilings were frozen before the
final tests. All three training seeds remain separate; final results select no
new checkpoint or seed.

The [full metric report](../research/sound_matching/evidence/final-study.json)
retains counts, terminal failures, costs, objective components, pitch diagnostics
and distributions for every actor and all eight cells. The
[paired report](../research/sound_matching/evidence/paired-differences.json)
describes exact episode-matched contrasts; see its [method](v2-paired-differences.md).

## Equal-eight-cell mean submitted distance

Lower values mean lower distance under the frozen spectral objective. The two
reconstruction cells contribute 25% of this mean, and the six shaping cells 75%.
Actual expenditure differs despite shared ceilings: fixed uses one evaluation,
inverse uses two, and search may use 32. These descriptive distances do not define
a perceptual success rate or establish statistical significance.

| Actor | Validation | IID test | Compositional test |
| --- | ---: | ---: | ---: |
| fixed | 1.855378 | 1.719892 | 1.759191 |
| pattern-search | 0.976593 | 0.959605 | 0.849539 |
| random-search-0 | 0.713268 | 0.687821 | 0.880416 |
| random-search-1 | 0.744264 | 0.699204 | 0.879922 |
| random-search-2 | 0.739019 | 0.673468 | 0.879673 |
| inverse-0 | 1.151327 | 1.076557 | 1.331970 |
| inverse-refine-0 | 0.522320 | 0.479837 | 0.589246 |
| inverse-1 | 1.204497 | 1.128432 | 1.375202 |
| inverse-refine-1 | 0.524790 | 0.439459 | 0.631621 |
| inverse-2 | 1.185955 | 1.089790 | 1.393210 |
| inverse-refine-2 | 0.528213 | 0.480492 | 0.616029 |

## IID test: reconstruction and shaping

Each cohort has 48 targets. Shaping means equally weight the three masks
within that cohort (16 pitch, 16 envelope and 16 brightness targets). Separate
mask distributions and pitch errors remain in the full report.

| Actor | Single reconstruction | Chord reconstruction | Single shaping | Chord shaping |
| --- | ---: | ---: | ---: | ---: |
| fixed | 3.607707 | 3.968967 | 1.045395 | 1.015425 |
| pattern-search | 2.476750 | 3.287684 | 0.186433 | 0.451036 |
| random-search-0 | 1.378139 | 1.537149 | 0.352505 | 0.509922 |
| random-search-1 | 1.400417 | 1.579251 | 0.331935 | 0.539386 |
| random-search-2 | 1.404633 | 1.452875 | 0.338009 | 0.505403 |
| inverse-0 | 1.944780 | 1.830009 | 0.769639 | 0.842917 |
| inverse-refine-0 | 1.018168 | 1.348074 | 0.087484 | 0.403333 |
| inverse-1 | 1.927029 | 2.063914 | 0.852988 | 0.825850 |
| inverse-refine-1 | 1.004867 | 1.330472 | 0.057242 | 0.336203 |
| inverse-2 | 1.884787 | 1.901522 | 0.795100 | 0.848903 |
| inverse-refine-2 | 1.026464 | 1.423080 | 0.092231 | 0.372566 |

## Compositional test: reconstruction and shaping

Each cohort has 48 targets. Shaping means equally weight the three masks
within that cohort (16 pitch, 16 envelope and 16 brightness targets). Separate
mask distributions and pitch errors remain in the full report.

| Actor | Single reconstruction | Chord reconstruction | Single shaping | Chord shaping |
| --- | ---: | ---: | ---: | ---: |
| fixed | 2.813111 | 3.068798 | 1.296867 | 1.433672 |
| pattern-search | 1.853072 | 2.366046 | 0.292557 | 0.566508 |
| random-search-0 | 1.675886 | 1.785022 | 0.507036 | 0.687103 |
| random-search-1 | 1.687044 | 1.813155 | 0.518082 | 0.661643 |
| random-search-2 | 1.698225 | 1.796192 | 0.475626 | 0.705365 |
| inverse-0 | 1.921053 | 1.901787 | 1.133726 | 1.143913 |
| inverse-refine-0 | 1.377375 | 1.662290 | 0.119310 | 0.438790 |
| inverse-1 | 1.936316 | 1.960215 | 1.239318 | 1.129045 |
| inverse-refine-1 | 1.452645 | 1.684734 | 0.161423 | 0.477106 |
| inverse-2 | 1.968844 | 2.048299 | 1.188689 | 1.187488 |
| inverse-refine-2 | 1.392645 | 1.692890 | 0.124920 | 0.489313 |

## Scope of the result

Inverse plus refinement has lower macro distance than the declared search
baselines on both final tests for every retained seed. This aggregate result
does not imply an improvement on every target or cell:

- On IID chord pitch shaping, refinement seed 0 has mean distance 0.020882 above
  pattern search. Validation refinement seed 1 is also worse in that cell by
  0.042283. Both brightness cells tie pattern search at zero for every seed and
  split.
- On compositional single-note pitch shaping, the fixed mean is 1.996319;
  inverse-only seeds 0/1/2 score 2.049067, 2.231374 and 2.069804. Their improved
  overall means therefore conceal regressions from the starting patch in this
  cell.
- Compositional refinement seeds 0/1/2 have higher submitted distance than
  pattern search on 27, 36 and 28 of 192 individual targets, respectively. The
  corresponding counts against same-seed random search are 36, 41 and 31.

The complete paired report preserves every comparison's denominators and
distributions. Large pitch errors in retained examples remain relevant even when
the spectral-distance mean improves.

The reference criterion is validation-only: all three inverse seeds must have
lower equal-eight-cell mean distance than fixed. It does not require superiority
over search. Learned proposals and refinements remain approximate synthetic
sound-matching research results; submission is an explicit protocol action.

The compositional split holds out target combinations and chord families, while
some combinations remain exposed in public starting patches. This is a target
distribution holdout; see [calibration](v2-calibration.md) for the exact grouping
and exposure inventory. These final tests are now consumed and can reproduce
this evidence; future model selection requires new held-out data.

Numerical waveform inspection and a listening index do not establish audible
quality. See [acceptance](v2-acceptance.md) for the listening-review status, exact
artifacts, installation checks, runtime measurements and platform limitations.
