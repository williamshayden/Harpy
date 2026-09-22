# Sound-matching research evidence

The [calibration protocol](../../docs/v2-calibration.md) fixes memberships,
candidate budgets, baseline seeds, equal-cell checkpoint selection and the
all-three-seeds reference criterion. The scripts here preserve those decisions.
All outputs are local and create-only. Creating an archive does not publish it.

## Evidence and boundaries

- M2 retains the 96-target calibration pilot, all 288 submitted episode records,
  objective fixtures, target/candidate WAVs, frozen manifests and the exact pilot
  source snapshot. `verify_m2.py` rechecks source files, every generated target's
  float32 hash, split separation, fixture scores and audio bytes.
- M3 retains training seeds 0, 1 and 2, every selected checkpoint, training and
  validation memberships, epoch metrics, state hashes and the training source
  snapshot. No seed is selected from final-test performance.
- M5 uses `study.py`: 192 validation, 192 IID final and 192 compositional final
  targets, each evaluated by eleven actors. Fixed and deterministic pattern
  search run once; random search, inverse and inverse-refine retain seeds 0/1/2.
  Final tests start only after the evaluator/model/protocol freeze.
- `archive_v1.py` preserves the earlier three-seed pitch qualification as a
  separate archive. It relabels only machine-specific artifact inventory paths,
  retains original result hashes, validates the three exact model payloads and
  recomputes qualification from all 23,800 records. An accompanying reload report
  records fresh CPU deserialization; archive preparation itself imports no Torch.

These files distinguish validation criterion results, continuous audio distance,
engineering qualification and human listening. There is no perceptual success
cutoff. The learned models remain bounded synthetic-sound research artifacts.
External audio, native Windows/macOS execution and DAW behavior are not qualified
by this study.

## Read a prepared complete archive

`archive_study.py` includes these portable paths:

```text
archive-manifest.json             every contained payload's size and SHA-256
study/plan.json                   evaluator, driver, runtime and model freeze
study/report.json                 strict all-split metrics and validation gate
study/listening-index.json        declared representative/worst listening pairs
study/{validation,iid-test,compositional-test}/
  manifest.json                  exact episode and actor membership
  attempts.jsonl, records.jsonl, commits.jsonl, complete.json
  artifacts/                     content-addressed WAVs and submitted patches
models/{seed0,seed1,seed2}/        every selected inverse model and its provenance
evaluator/                       exact evaluation source, driver and dependency lock
attachments/                     separate training, M2 and v1 evidence archives
```

The outer `.tar.gz.sha256` identifies the compressed download. After extraction,
`archive-manifest.json` identifies uncompressed payloads; these hashes have
different purposes. Verify the compressed file first:

```bash
sha256sum -c COMPLETE_ARCHIVE.tar.gz.sha256
```

Verify extracted files from the archive root without importing ML libraries:

```python
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest = json.loads((root / "archive-manifest.json").read_text())
for item in manifest["files"]:
    path = root / item["path"]
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    assert path.stat().st_size == item["size_bytes"]
    assert digest.hexdigest() == item["sha256"]
print("Every extracted payload matches its archive inventory.")
```

With the candidate harpy wheel installed, recompute metrics without Torch:

```bash
harpy summarize study/validation --format markdown
harpy summarize study/iid-test --format csv
harpy summarize study/compositional-test --format csv
PYTHONPATH=evaluator/source python evaluator/research/sound_matching/study.py report \
  --plan study/plan.json --bundles study --output recomputed-report.json
```

The report reader checks full membership, model inventories, actor algorithms and
seeds, numerical runtime identities, target hashes, every committed record and
the equal-eight-cell all-seed criterion. Report reading does not re-run inference.
The frozen evaluator path is intentional: it keeps report semantics tied to the
archived source. A newer installed reader can be checked separately.

## Re-run the frozen actors

Use the qualified Linux/WSL Python and dependency versions recorded in
`study/plan.json`; install the candidate package's `learn` extra. From the archive
root, run into a new output path:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=evaluator/source python evaluator/research/sound_matching/study.py run \
  --plan study/plan.json --models models --split validation --output repeated-validation
```

The same command with `iid-test` or `compositional-test` reproduces already
consumed final evidence; it is not a fresh held-out evaluation. To resume an
interrupted repetition, retain its output directory and add `--resume`. Never
edit the frozen source, models or plan to bypass a resume mismatch. Training
new weights is a separate experiment and does not promise identical results.

## Listening and failures

`listening.py` selects the lower median and largest submitted distance for each
split/actor/cohort, sorting ties by opaque episode ID. The rule is fixed before
final evaluation. This yields 264 listening pairs when all actors submit on all
splits. The index records the full submitted/episode denominators, target and
submitted WAV checksums, editable patch, objective components and pitch errors.
It also retains every terminal error and truncation, with the last valid patch
when available. No missing submission is replaced by a best audition.

Regenerate the index from complete bundles:

```bash
PYTHONPATH=evaluator/source python evaluator/research/sound_matching/listening.py \
  --study study --output recomputed-listening-index.json
```

Paths inside the index are relative to `study/`. A saved patch can be rendered
through the ordinary Python renderer:

```python
import json
from pathlib import Path
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import render

document = json.loads(Path("SAVED_PATCH.json").read_text())
audio = render(Patch.from_dict(document["patch"]))
assert audio.shape == (96000,)  # Two seconds, mono float32 at 48 kHz.
```

Play retained target and submitted WAVs at the same playback gain. Index generation and numerical
inspection are not a claim that a human listened; listening status remains
explicit in the evidence.

Archive construction streams file contents and compressed verification in
bounded chunks. The small file inventory scales with file count; waveform
corpora and compressed archives are never loaded together into RAM.
