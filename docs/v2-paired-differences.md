# Matched episode differences

After all three frozen study splits complete, generate the derived report from
the source checkout without importing models or running any actors:

```bash
PYTHONPATH=src python research/sound_matching/paired_differences.py \
  --study STUDY_ROOT --output STUDY_ROOT/paired-differences.json
```

The output is create-only and uses schema `harpy.sound-match.paired-differences/1`.
It binds the study plan, each dataset manifest, each bundle manifest/completion
marker, the postprocessor source, and the reader's source/runtime identities.
Each completed bundle receives one strict record scan. Incomplete splits, changed
runtime/model identities, missing members and mismatched target hashes fail closed.

There are 25 comparisons per split: every non-fixed actor versus fixed;
inverse-refine versus inverse for each seed; and both learned actor variants
versus pattern search and the random-search actor with the same seed. All three
training seeds remain separate. Differences are **actor minus comparator**;
negative values mean lower submitted distance for the named actor.

Every expected episode remains in the nine terminal-status pair counts. Distance
statistics use exact episode-ID pairs with two explicit submissions. Errors and
truncations receive no imputed or last-candidate score. Every terminal failure is
also retained separately, with its reason and available last-distance diagnostic.
An equally weighted eight-cell mean difference is available only when every
expected pair submits.

Cell and cohort statistics report count, arithmetic mean, median and p90 of the
paired distance differences. Evaluation-count distributions on the same submitted
pairs make actual candidate usage visible: the actors share a budget ceiling, and
may consume different numbers of evaluations. No significance or perceptual
acceptance claim follows from these descriptive differences. Reference-quality
qualification remains the separately preregistered validation-only criterion in
the primary study report.

The 13 focused tests passed in 3.07 seconds, covering exact-ID pairing, direction
and quantiles, all-seed comparison coverage, unequal cell sizes, missing members,
failure denominators, real submitted/error/truncated bundles, plan/target identity
rejection, one-scan readout, complete-split gating and model-free imports:

```bash
PYTHONPATH=src python -m pytest tests/tasks/test_sound_match_paired_differences.py -q
```
