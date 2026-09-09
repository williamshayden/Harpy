# Spectrum preservation study measurements

Counts use matched episodes. Conditions are never pooled; `all` combines registers within one condition. Full numerical values, terminal p99/max errors, and all paired counts are in [paired-actors.csv](paired-actors.csv).

## clean

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 0 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 0 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |

## level-minus-12db

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 0 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 0 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |

## level-minus-24db

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 0 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 0 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |

## phase-45

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 0 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 0 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |

## phase-90

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 0 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 0 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 0 / 0 |

## noise-30db

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 598 | 2.00 / 1412.00 | 598 | 0 / 0 | 29.78 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 2 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 2 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 198 | 6.45 / 1412.00 | 198 | 0 / 0 | 31.17 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 2 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 2 / 0 |

## noise-10db

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 588 | 449.36 / 2013.00 | 588 | 0 / 0 | 29.73 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 12 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 12 / 0 |
| iid | legacy-point | 200 | 132 / 199 | 2.00 / 402.00 | 199 | 0 / 0 | 29.11 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 1 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 1 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 189 | 1590.01 / 2013.00 | 189 | 0 / 0 | 31.03 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 11 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 11 / 0 |

## combined

| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | legacy-point | 600 | 376 / 587 | 1188.10 / 2013.00 | 587 | 0 / 0 | 29.75 / 1.00 | — |
| all | cell-max | 600 | 376 / 600 | 3.00 / 3.00 | 600 | 0 / 0 | 29.80 / 1.00 | 13 / 0 |
| all | quadratic-fft | 600 | 376 / 600 | 2.00 / 2.00 | 600 | 0 / 0 | 29.78 / 1.00 | 13 / 0 |
| iid | legacy-point | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | — |
| iid | cell-max | 200 | 132 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 29.27 / 1.00 | 0 / 0 |
| iid | quadratic-fft | 200 | 132 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.14 / 1.00 | 0 / 0 |
| lower | legacy-point | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | — |
| lower | cell-max | 200 | 124 / 200 | 3.00 / 3.00 | 200 | 0 / 0 | 28.98 / 1.00 | 0 / 0 |
| lower | quadratic-fft | 200 | 124 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 29.04 / 1.00 | 0 / 0 |
| upper | legacy-point | 200 | 120 / 187 | 1889.14 / 2013.00 | 187 | 0 / 0 | 31.09 / 1.00 | — |
| upper | cell-max | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 13 / 0 |
| upper | quadratic-fft | 200 | 120 / 200 | 2.00 / 2.00 | 200 | 0 / 0 | 31.16 / 1.00 | 13 / 0 |

## Clean one-cent changes versus legacy

Initial perception and successful terminal tuning are separate outcomes. Gains and losses are reported separately.

| Register | Actor | Initial gains | Initial losses | Submitted gains | Submitted losses |
| --- | --- | ---: | ---: | ---: | ---: |
| all | cell-max | 0 | 0 | 0 | 0 |
| all | quadratic-fft | 0 | 0 | 0 | 0 |
| iid | cell-max | 0 | 0 | 0 | 0 |
| iid | quadratic-fft | 0 | 0 | 0 | 0 |
| lower | cell-max | 0 | 0 | 0 | 0 |
| lower | quadratic-fft | 0 | 0 | 0 | 0 |
| upper | cell-max | 0 | 0 | 0 | 0 |
| upper | quadratic-fft | 0 | 0 | 0 | 0 |
