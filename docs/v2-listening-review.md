# V2 listening review and matching-quality assessment

Review date: 2026-11-10. Listening and a bounded matching diagnosis have been
performed. Stable promotion is deferred until matching quality improves. This record supplements the
historical [acceptance record](v2-acceptance.md); it does not change the frozen
objective, models, study memberships, scores, or published 2.0.0 artifacts.

## Material and procedure

The project owner listened to the original 41.25-second validation review reel
through a local browser page showing target/match labels and separate pair
controls. Each pair plays the target first, then the submitted match. The eight
pairs are seed-0 inverse-plus-refinement lower-median and worst examples from
single-note/chord reconstruction and shaping, selected by spectral distance.

Original [review WAV](https://github.com/williamshayden/Harpy/releases/download/v2.0.0/all-cohorts-review.wav)
SHA-256: `170164a8141f0600ad2eb26f942d7367744deadfb59bf3fabfb640372efa919f`.
The [numerical and identity receipt](../research/sound_matching/release_reviews/2026-11-10/listening-checks.json)
records the episode IDs, source hashes, samples, and diagnostics for all eight
pairs. Source samples and original gain were preserved; gaps contain silence.

| Pair | Task and selection | Target seconds | Match seconds |
| --- | --- | --- | --- |
| 1 | Single reconstruction, median | 0–2 | 2.5–4.5 |
| 2 | Single reconstruction, worst | 5.25–7.25 | 7.75–9.75 |
| 3 | Chord reconstruction, median | 10.5–12.5 | 13–15 |
| 4 | Chord reconstruction, worst | 15.75–17.75 | 18.25–20.25 |
| 5 | Single shaping, median | 21–23 | 23.5–25.5 |
| 6 | Single shaping, worst | 26.25–28.25 | 28.75–30.75 |
| 7 | Chord shaping, median | 31.5–33.5 | 34–36 |
| 8 | Chord shaping, worst | 36.75–38.75 | 39.25–41.25 |

This is a limited, unblinded representative review by one listener. It covers all
four cohorts but only five of eight task cells; single-envelope and both
brightness-shaping cells are absent. It does not establish a perceptual success
rate, comparative listening preference, or a population-level quality claim.
No per-pair listener ratings or listening-hardware details were collected.

## Actual listener feedback

The listener reported no obvious audio errors, but significant matching misses
across the reviewed material, and asked whether the cause was the task, scoring
system, or model. This is a qualitative report; it is not a blanket assertion
that every pair failed. No obvious playback/rendering defect was reported in
this session. Matching quality has not been signed off as reliable.

The separate file checks found all sixteen source clips finite and nonzero,
zero first/last samples, and no samples at or beyond magnitude one. Those checks
support file integrity and are not substitutes for listening.

## Verified diagnosis

The inverse model is supervised on patch labels. It uses pitch-bin binary cross
entropy, waveform cross entropy, and normalized scalar squared error. It is not
an RL policy trained against the spectral score. Spectral distance is used for
checkpoint selection and subsequent pattern search; pitch error remains a
separate evaluator diagnostic. See the [inverse method](v2-inverse.md).

Known failures already exist in the frozen [results](v2-results.md): the learned
initializer alone trails search on aggregate distance, and reconstruction is
substantially worse than shaping. The official average weights reconstruction
at 25% and shaping at 75%; a good average therefore does not imply reliable
reconstruction.

The bounded [case-diagnosis receipt](../research/sound_matching/release_reviews/2026-11-10/match-diagnosis.json)
verifies all eight reviewed target patches reproduce their saved WAVs exactly
and score zero against themselves. Submitted patches also reproduce their saved
audio exactly; recomputed distances match the recorded scores, and shaping
controls that should remain locked are unchanged. This finds no renderer,
reachability, persistence, or score-replay defect in these eight examples.

The single-note pitch-shaping failure `e_1e5f7b60033cf734d04b576f` isolates a
model-initialization/local-search failure:

| Stage | Note coordinate, cents | Absolute pitch error, cents | Spectral distance |
| --- | ---: | ---: | ---: |
| Starting patch | 5652 | 812 | 2.321021 |
| Seed-0 inverse proposal | 5871 | 593 | 2.305504 |
| Seed-0 refined submission | 5762 | 702 | 2.119189 |
| Correct target | 6464 | 0 | 0 |

Search improved its chosen distance while worsening pitch relative to the
learned proposal. Every tested neighbor at one, three, six, and twelve cents
either side of the final note scores worse. The recorded search stops after
17 of its 32 allowed evaluations once its one-cent neighborhood is exhausted.
Increasing the evaluation ceiling alone would not change that stop condition.
The already-retained seed-1 refinement result reaches the exact target and zero
distance in 16 evaluations on the same episode. No new actor run was performed
to obtain that comparison.

The score does not rank the wrong note above the perfect target or call the
bad match perfect. It does provide a non-monotonic local search landscape.
The near-seven-semitone error suggests harmonic overlap as one possible cause;
that mechanism has not been isolated by an ablation. These observations support
an interaction between initialization, local search, and the chosen objective,
not a claim that any one component explains all reviewed misses.

## Research follow-up and release boundary

The smallest useful follow-up would hold weights, objective, examples, and the
32-evaluation ceiling fixed while comparing local refinement with coarse/global
or multistart pitch proposals. Separately, hold search and initialization fixed
while testing a pitch-aware score derived from public audio. Ground-truth patch
labels must remain diagnostic-only and cannot leak into an evaluated actor.
Use validation or fresh diagnostic data; reserve new held-out memberships for
later confirmatory claims. Existing final tests are consumed evidence.

This review does not establish dependable sound reconstruction. The project
owner chose to improve matching quality before promoting v2 to stable, and then
prioritized an LLM/HarnessBridge pilot over further tuning of the small inverse
model. The published 2.0.0 remains a prerelease. Current documentation and the
default branch can be maintained independently of that quality gate.

The immediate pilot should evaluate an LLM through the ordinary actor interface
under the same candidate-evaluation rules, while declaring its actual inputs.
The existing HarnessBridge provides text conversations; wave-derived summaries
must be labeled as such, not as native audio understanding. Its results require
separate provenance and new listening evidence and cannot inherit the frozen
study's model-quality claims.
