# CPU inverse estimator protocol and evidence

The learned actor is an experimental patch initializer. It conditions a small
convolutional encoder on the target waveform, public note count, editable-field
mask and current patch. The supervised targets belong only to the training
loader. The inference decoder has no target-parameter input.

## Frozen training method

The implementation identities and full numeric configurations are serialized in
every artifact's `training-manifest.json`. The reference profile uses the frozen
2016 training targets and 192 validation targets described in
[the calibration protocol](v2-calibration.md). It runs six epochs, batch size four,
AdamW with learning rate 0.001 and weight decay 0.0001, gradient norm limit 5,
one CPU Torch thread, deterministic algorithms and no loader workers. Seeds 0, 1
and 2 run separately; no seed is discarded. Initial untrained weights are not
eligible checkpoints. No final-test target participates in training or selection.

Features are 128 by 64 fixed log-magnitude spectrograms, computed from the whole
clip using a 4096-sample periodic Hann window, 1024-sample hop, log floor 1e-7,
128 geometrically spaced frequencies from 80 through 12000 Hz, and linear
frequency/time interpolation. The bounded disk cache contains float32 features,
never a waveform corpus. Its identity binds membership, exact target audio hashes,
feature/renderer code and NumPy/Python versions. Minibatches alone become tensors.

The encoder has three convolution/ReLU stages (4, 8, 8 output channels), adaptive
pooling to 16 by 4, and a 64-unit fusion layer with 26 public conditioning values.
Heads predict 2401 distinct cent bins, four waveform logits, and six normalized
scalars. There is no dropout or normalization layer. The architecture configuration
records every kernel, stride and padding value.

Notes use mean binary cross entropy with positive weight `(2401-k)/k`; waveform
uses categorical cross entropy; scalar heads use squared error after sigmoid.
Eligible fields receive equal weight within each example, then examples receive
equal weight. Locked fields and decay at target sustain 0 dB are excluded during
supervision. Brightness and ADSR times are log-normalized; dB controls are linear.

The decoder selects the top public note count of distinct cent bins, breaks ties
toward lower pitch, and sorts. Scalar predictions are clamped, transformed back
to physical units and rounded to a legal integer with ties lower. Locked values
come from the public current patch. Unlocked decay is canonicalized to 100 ms only
when decoded sustain is 0 dB. Inactive-field inference never reads target labels.

Each completed epoch is evaluated through the ordinary bounded runner. Selection
minimizes the equally weighted mean of the eight validation cell means, requiring
every episode to submit. Exact ties select the earlier epoch. Every epoch's loss,
cell means and optimizer-update count remains in `training-summary.json`.
The reference-quality rule remains the preregistered all-three-seeds improvement
over fixed; comparison and qualification belong to the separate matched study.

## Reload and ordinary actor behavior

Artifacts contain tensor-only `model.pt`, a training manifest, a training summary,
and a final inventory manifest binding payload sizes and SHA-256 hashes. Metadata
inspection requires no Torch. Explicit model loading verifies shape, float32 type,
finiteness and a container-independent tensor-state hash. The writer publishes
the final manifest last; interrupted directories cannot load as complete artifacts.

`inverse` proposes once and submits, charging one model call and two total
candidate evaluations when budgets permit. `inverse-refine` makes that same
proposal then uses the ordinary deterministic pattern search with the remaining
public budget. It retains the heard starting candidate and reserves any required
restoration and final submit. A no-edit or insufficient-budget episode submits
the existing candidate with zero model calls.

## Reproduction and verification

Use the official CPU Torch build in a Python 3.12 environment when reproducing
the qualified CPU run. Install the local package after selecting Torch:

```bash
python -m pip install 'torch==2.13.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[learn]'
```

For the complete pinned environment, the optional CPU job in
`.github/workflows/ci.yml` exports `uv.lock`, omits CUDA-only payloads, and selects
`--torch-backend=cpu`. This changes the Torch wheel variant while retaining the
declared base version pins. GPU execution is not qualified by these CPU results.

```bash
PYTHONPATH=src python research/sound_matching/train_inverse.py \
  --profile reference --output runs/inverse-reference
PYTHONPATH=src python -m pytest tests/tasks/test_sound_match_inverse*.py
```

The script records its plan before any seed runs. For concurrent development,
point `PYTHONPATH` at a preserved package snapshot containing the frozen dataset
files. Training rejects package-source changes during an epoch or between seeds.
The source inventory is included in each artifact. Preserve the snapshot, script,
lockfile and snapshot checksums with research artifacts so the exact training code
remains available after later development.

The `quick` profile performs one epoch on the first example of every training and
validation cell (eight each). It is an engineering smoke profile, not evidence for
reference quality. Tests exercise real optimization and weight changes, repeated
seed determinism, tensor/prediction reload equality, normal runner submission and
refinement, locked fields, inactive supervision, budget limits, lazy imports and
artifact tamper rejection. Concrete run evidence is appended after verification.

## Verified run on 2026-09-22

The 18 inverse tests passed in 40.61 seconds against a preserved source copy.
The quick profile performed two real optimizer updates in 6.11 seconds, changed
the tensor-state hash, and reproduced the selected state and epoch statistics
on a second run of seed 0. Reloaded predictions and ordinary inverse/refinement
episodes passed across all eight smoke cells.

The reference study ran on an Intel Core Ultra 7 155H under WSL Ubuntu, with
one CPU Torch/BLAS/OMP thread. Runtime versions were Python 3.12.3, NumPy 2.5.1
and PyTorch 2.13.0+cpu; NumPy/Torch versions match the archived lockfile. Each
192,707-parameter model completed 3,024 optimizer updates over six epochs and
all 192 validation episodes submitted at every epoch.

| Seed | Selected epoch | Validation macro distance | Training seconds | Process peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 | 1.151327000 | 252.61 | 414.64 |
| 1 | 4 | 1.204496888 | 178.44 | 422.58 |
| 2 | 6 | 1.185955253 | 164.41 | 428.24 |

Seed 0 time includes the cold feature-cache build. Seeds 1 and 2 reuse verified
cached features. Peak RSS is cumulative within the single sequential training
process, not a separate per-seed allocation. Total measured training time was
595.47 seconds. Seed 1 correctly selected epoch 4 despite completing all six epochs.

Selected-checkpoint validation cells:

| Cell | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| 1:all | 1.847845 | 1.931257 | 1.909886 |
| 1:brightness | 0.318960 | 0.386866 | 0.367340 |
| 1:envelope | 0.701210 | 0.755626 | 0.732504 |
| 1:pitch | 1.567977 | 1.620749 | 1.531976 |
| 3:all | 1.948227 | 2.083724 | 1.988080 |
| 3:brightness | 0.542386 | 0.580897 | 0.687546 |
| 3:envelope | 0.761881 | 0.755068 | 0.723615 |
| 3:pitch | 1.522131 | 1.521786 | 1.546694 |

All three retained checkpoints were subsequently deserialized on CPU by the
current package, and their exact tensor hashes matched their selected states.
The same read-only check successfully loaded all three archived v1 pitch models.
No model was refitted for that check. Training and reload records alone do not
establish superiority over the fixed or search baselines; the matched evaluation
report applies the preregistered reference-quality criterion. No final-test actor
evaluation or final-test-driven tuning was performed by this training run.

Local evidence is retained at `outputs/v2-m3-reference-20260922/` (three artifacts,
plan, complete progress and summary), `outputs/v2-m3-smoke-verification.json`, and
`outputs/v2-m3-and-v1-model-reload-20260922.json`. The 104-file source/config/test
snapshot was independently checked against its file inventory and archived as
`outputs/v2-m3-reference-source-20260922.tar.gz`.

- Training package source SHA-256: `478eb2413a413a0f12ac8b2cc6ff58ba31d00f1582f3587e47d35a61d7c71d19`.
- Snapshot archive SHA-256: `dd33ba39a1aa400d28eab5b87c67da4bbe42d436b1548386dfe6af4015c09c47`.
- Six-model reload report SHA-256: `dad20d38a05cad24e81170ad546e9354cf0361165bba2f1d5289ac83c57d536c`.
