# Migration from the original v1.0.0

The revised v1 is currently **Unreleased**. The existing `v1.0.0` tag remains the
reproducibility boundary for historical training and artifact execution.
The old `python -m harpy.learning.cli` command module is retired alongside its
console entry points. Use `harpy` or `python -m harpy` for the revised workflows.

| Previous surface | Revised surface |
| --- | --- |
| `harpy-sine-learn train-pitch` / `train-ppo` and frozen milestone workflows | `harpy train` with `pitch` or `ppo`, smoke/checkpoint profiles |
| Milestone evaluation wrappers and prototype `compare-pitch` | `harpy evaluate`, explicit actors and clean/robustness membership |
| Checkpoint trace wrappers | `harpy run` with an inspectable decision trace |
| `harpy-sine-learn summarize` | `harpy summarize`, including retained historical report readers |
| Separate report type for an intervention | Ordinary experiment results plus a named qualification protocol |
| Shared stateful actor across a suite | Fresh actor factory per episode; optionally shared model weights |
| Coupled pitch estimator/planning label | Separate estimator, decoder, and controller identities |
| Research-only `quadratic-fft` waveform adapter | Built-in `harpy ... --actor waveform-fft` and Python `waveform_fft_actor_spec()`; same five-cent decoder and committed controller |
| Active behavior-cloning training | Historical reproduction at `v1.0.0`; no BC in the new supported training interface |

The synthesis, analysis, tuning, and Gymnasium capabilities remain supporting
interfaces. Historical milestone documents and saved report formats remain useful
evidence. Internal caches, artifact storage plumbing, and historical codecs are not
new extension contracts. Custom Python actors plug into the ordinary runner through
a factory; no trainer registry change is required.

Historical pitch-model weights can be imported with their original training metadata and
hashes preserved. Re-evaluation uses the revised declared decoder/controller and
records its own source identity; an old model is not silently relabeled as having
been trained by the new code. The prototype decoder evidence stays under
`outputs/decoder-probe` and its verification document.

For exact historical workflows, use a separate checkout of `v1.0.0` and the recorded
dependencies. Do not overwrite an old artifact/result with a revised one. New
evaluation outputs are create-only and use the ordinary revised experiment schema.
