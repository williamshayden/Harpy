# Spectrum study derived views

[Summary](summary.md) · [Paired CSV](paired-actors.csv)

These are descriptive finite-cohort measurements, not a qualification decision. Membership is fresh combinations in the known rendering family, excluding the recorded inventory. It becomes consumed evidence after this run. No statistical significance or learned-model claim is inferred.

Success means a submitted terminal within five cents. Rescued/regressed counts compare the same episode under the same condition against legacy-point. Initial perception uses the recorded first estimate. P99 uses the existing ordinary-result linear-interpolation convention. Overall rates use summed numerators and denominators; p99 values are recomputed from pooled records within each condition, never averaged across registers.

## Exact input and execution identity

```json
{
  "actor_identities": [
    {
      "controller": "harpy-committed-zero-tolerance-plan-v1",
      "decoder": "harpy-public-source-feasible-five-cent-argmax-v1",
      "device": "cpu",
      "estimator": "harpy-legacy-log-point-dbfs-v1",
      "name": "legacy-point",
      "observation_mode": "waveform"
    },
    {
      "controller": "harpy-committed-zero-tolerance-plan-v1",
      "decoder": "harpy-public-source-feasible-five-cent-argmax-v1",
      "device": "cpu",
      "estimator": "harpy-research-log-cell-max-dbfs-v1",
      "name": "cell-max",
      "observation_mode": "waveform"
    },
    {
      "controller": "harpy-committed-zero-tolerance-plan-v1",
      "decoder": "harpy-feasible-fft-peak-nearest-five-cent-v1",
      "device": "cpu",
      "estimator": "harpy-research-hann-quadratic-fft-v1",
      "name": "quadratic-fft",
      "observation_mode": "waveform"
    }
  ],
  "protocol_digest_sha256": "7ec38d3af9e0a30baba4437e00350b1fdfe77166b50f72b92fe0c1ed185c9c48",
  "protocol_id": "harpy-spectrum-preservation-membership-v1",
  "provenance": {
    "renderer": "harpy-static-single-sine-corruption-v1",
    "runtime": {
      "condition_evaluation_wall_seconds": {
        "clean": 308.13534407000407,
        "combined": 865.3083719749993,
        "level-minus-12db": 381.5283394379949,
        "level-minus-24db": 385.96555015200283,
        "noise-10db": 868.8638061219826,
        "noise-30db": 855.4476217440097,
        "phase-45": 383.3031621139962,
        "phase-90": 380.730926506978
      },
      "condition_runs": 8,
      "cpu_threads_per_process": 1,
      "elapsed_wall_time_seconds": 1265.9298162409978,
      "evaluation_devices": {
        "cell-max": "cpu",
        "legacy-point": "cpu",
        "quadratic-fft": "cpu"
      },
      "gymnasium": "1.3.0",
      "numpy": "2.5.1",
      "ordering": "fixed condition, episode, actor order; paired conditions reuse nuisance seeds",
      "per_process_cache_budget_bytes": 134217728,
      "platform": "Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
      "python": "3.12.3",
      "stable_baselines3": null,
      "timing_scope": "study setup, eight condition runs in a bounded worker pool, shard writes, and strict reload; excludes final merge validation and exports",
      "torch": null,
      "worker_limit": 4
    },
    "source": {
      "identity": {
        "commit": "1f2ba3fe1829537c3f8e274dd45abeeafaa744d2",
        "dependency_lock_sha256": "c223e113ebf6f442425fc0fe32a2ae105c98a5d002c9a4daa7ba58b2f49b7a8f",
        "dirty_tree": true,
        "required_inputs_committed": true,
        "tracked_diff_sha256": "384631e748790c669c710ec8ff5f5729807bcbd1be423b8df9c4c7c795058997"
      },
      "package_sha256": "9cca1d94ed865ed07f1a5072a394c83970fccf8575cda7d380067e7152754ab1"
    }
  },
  "readout_sha256": "7b447f460b220e97b379d5f50c3c670889c893bfaf0e0545ffd62772bcd3a131",
  "result_path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/outputs/spectrum-preservation-study/experiment.json",
  "result_sha256": "69db356439daf33f320d7a596d236584c8053f55b1954927c452dbda8679be8f",
  "result_size_bytes": 17995025,
  "selection_seed": 20260908,
  "study_files": [
    {
      "path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/research/spectrum_preservation/PLAN.md",
      "sha256": "9d8b89f9187564c1143b314f97fcd441ce2aa80e58347834bae08fe4f72644be",
      "size_bytes": 6862
    },
    {
      "path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/research/spectrum_preservation/encoders.py",
      "sha256": "c295d1ed726ec71010d2a9f66ee441fed92b49ce986eb3fb21cf69e2dfafc23b",
      "size_bytes": 6637
    },
    {
      "path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/research/spectrum_preservation/membership.json",
      "sha256": "90ccc0be6b70dd14577557861b1fd5a492cdb3b2497a5981b69a81b7cd5faa18",
      "size_bytes": 94991
    },
    {
      "path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/research/spectrum_preservation/membership.py",
      "sha256": "fa17b7de1d10baeb9b0e0061588e89589bd562b4d50222586e53e16612cfc4bf",
      "size_bytes": 7599
    },
    {
      "path": "/home/haydenw/Projects/Harpy/.worktrees/pitch-decoding-v1/research/spectrum_preservation/study.py",
      "sha256": "d0a2af4a0285fd2d0876f625e680b30c09d1c18e45e5dc4380b723a78fd3f0b3",
      "size_bytes": 13518
    }
  ]
}
```
